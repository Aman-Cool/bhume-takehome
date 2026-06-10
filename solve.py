#!/usr/bin/env python3
"""
Boundary correction for BhuMe take-home.

Strategy:
  For each plot, rasterize its official boundary edge at the same resolution
  as boundaries.tif, then run a normalised cross-correlation against the
  auto-detected field edges to find the translation that best aligns the
  cadastral outline to the real field.  Confidence comes from how sharp and
  dominant that correlation peak is, adjusted for area-ratio plausibility and
  shift magnitude.  Plots where the peak is too weak get flagged rather than
  moved.

Run:
    uv run solve.py data/34855_vadnerbhairav_chandavad_nashik
    uv run solve.py data/34855_vadnerbhairav_chandavad_nashik data/malatavadi_kolhapur
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import scipy.ndimage as ndi
from pyproj import Transformer
from rasterio.features import rasterize as rio_rasterize
from rasterio.windows import from_bounds
from scipy.signal import fftconvolve
from shapely.affinity import translate
from shapely.ops import transform as shp_transform

from bhume import load, score, write_predictions
from bhume.geo import geom_to_imagery_crs

# ── tunables ──────────────────────────────────────────────────────────────────
PAD_M = 60          # padding (m) around each plot when reading raster patches
AREA_LO = 0.40      # flag if drawn/full-recorded ratio < this (area mismatch)
AREA_HI = 2.50      # flag if drawn/full-recorded ratio > this
MAX_SHIFT_M = 90    # shifts larger than this are penalised
EDGE_DILATE = 2     # pixels of erosion to extract edge from rasterised polygon
BLUR_SIGMA = 1.2    # Gaussian softening applied to both template and signal
MIN_CONF = 0.13     # below this confidence → flag instead of correct
BLEND_THRESH = 0.28 # above this NCC peak → fully trust per-plot shift
# ─────────────────────────────────────────────────────────────────────────────


def _utm_zone(lon: float) -> str:
    return f"EPSG:{32600 + int((lon + 180) // 6) + 1}"


def _to_crs(geom, src_epsg: str, dst_epsg: str):
    tf = Transformer.from_crs(src_epsg, dst_epsg, always_xy=True)
    return shp_transform(lambda xs, ys, z=None: tf.transform(xs, ys), geom)


def _area_m2(geom_4326) -> float:
    utm = _utm_zone(geom_4326.centroid.x)
    return _to_crs(geom_4326, "EPSG:4326", utm).area


def _read_band_patch(src, geom_4326, pad_m: float):
    """Read a single-band raster window around geom_4326 (any CRS)."""
    g = geom_to_imagery_crs(src, geom_4326)
    l, b, r, t = g.bounds
    l -= pad_m; b -= pad_m; r += pad_m; t += pad_m
    dl, db, dr, dt = src.bounds
    l, b, r, t = max(l, dl), max(b, db), min(r, dr), min(t, dt)
    if r <= l or t <= b:
        return None, None
    win = from_bounds(l, b, r, t, transform=src.transform)
    data = src.read(1, window=win)
    return data, src.window_transform(win)


def _rasterise_edge(geom_4326, transform_3857, height: int, width: int) -> np.ndarray:
    """Binary edge mask (not fill) of geom in raster pixel space."""
    geom_3857 = _to_crs(geom_4326, "EPSG:4326", "EPSG:3857")
    filled = rio_rasterize(
        [(geom_3857.__geo_interface__, 1)],
        out_shape=(height, width),
        transform=transform_3857,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    inner = ndi.binary_erosion(filled.astype(bool), iterations=EDGE_DILATE)
    return (filled.astype(bool) & ~inner).astype(np.float32)


def _xcorr_shift(template: np.ndarray, signal: np.ndarray, px_w: float, px_h: float):
    """
    Find the (dx_m, dy_m) translation maximising edge overlap.
    Returns (dx_m, dy_m, peak_fraction, prominence).
      peak_fraction  – fraction of template edge pixels that overlap signal edges at the best shift
      prominence     – peak / 95th-percentile of the correlation map (sharpness)
    """
    binary_sig = (signal > 0).astype(np.float32)
    binary_tpl = (template > 0).astype(np.float32)
    n_edge = binary_tpl.sum()
    if n_edge == 0 or binary_sig.sum() == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Gaussian softening for sub-pixel stability
    smooth_sig = ndi.gaussian_filter(binary_sig, sigma=BLUR_SIGMA)
    smooth_tpl = ndi.gaussian_filter(binary_tpl, sigma=BLUR_SIGMA)

    corr = fftconvolve(smooth_sig, smooth_tpl[::-1, ::-1], mode="same")

    # Re-normalise against a hard edge count so the value is interpretable
    raw_corr = fftconvolve(binary_sig, binary_tpl[::-1, ::-1], mode="same")
    peak_fraction = float(raw_corr.max() / n_edge)

    p95 = float(np.percentile(corr, 95))
    prominence = float(corr.max() / (abs(p95) + 1e-9))

    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    py, px = np.unravel_index(corr.argmax(), corr.shape)
    dx_m = (px - cx) * px_w
    dy_m = -(py - cy) * px_h   # image y is flipped vs geographic y
    return dx_m, dy_m, peak_fraction, prominence


def _confidence(peak_frac: float, prominence: float, shift_m: float, area_ratio: float) -> float:
    """Combine several quality signals into a single 0-1 confidence."""
    # base: fraction of edges aligned
    conf = peak_frac

    # sharpness bonus – a distinct peak is much more trustworthy
    if prominence > 2.5:
        conf = min(1.0, conf * 1.15)
    elif prominence < 1.4:
        conf *= 0.7

    # large-shift penalty
    if shift_m > MAX_SHIFT_M:
        conf *= 0.25
    elif shift_m > 55:
        conf *= 0.65

    # area-ratio plausibility (log penalty, centre = log(1) = 0)
    ar_dev = abs(np.log(area_ratio + 1e-9))
    conf *= max(0.4, 1.0 - ar_dev * 0.4)

    return float(min(1.0, max(0.0, conf)))


def _correct_plot(pn: str, geom, row, src_b):
    """Attempt a per-plot correction. Returns (dx_m, dy_m, conf, note)."""
    # ── area-ratio gate ───────────────────────────────────────────────────────
    drawn_m2 = _area_m2(geom)
    rec_m2 = float(row.get("recorded_area_sqm") or 0)
    pot_m2 = float(row.get("pot_kharaba_ha") or 0) * 10_000
    full_m2 = rec_m2 + pot_m2
    area_ratio = drawn_m2 / full_m2 if full_m2 > 0 else 1.0

    if area_ratio < AREA_LO or area_ratio > AREA_HI:
        return 0.0, 0.0, 0.0, f"area_ratio={area_ratio:.2f}_out_of_range"

    # ── read boundaries patch ─────────────────────────────────────────────────
    signal, tf = _read_band_patch(src_b, geom, pad_m=PAD_M)
    if signal is None:
        return 0.0, 0.0, 0.0, "patch_outside_extent"
    H, W = signal.shape
    if H < 12 or W < 12:
        return 0.0, 0.0, 0.0, "patch_too_small"

    # ── rasterise official edge ───────────────────────────────────────────────
    edge = _rasterise_edge(geom, tf, H, W)
    if edge.sum() < 4:
        return 0.0, 0.0, 0.0, "edge_rasterisation_empty"

    # ── cross-correlation ─────────────────────────────────────────────────────
    px_w = abs(tf.a)   # metres per pixel (EPSG:3857 ≈ metres)
    px_h = abs(tf.e)
    dx_m, dy_m, peak_frac, prom = _xcorr_shift(edge, signal, px_w, px_h)
    shift_m = float(np.sqrt(dx_m**2 + dy_m**2))
    conf = _confidence(peak_frac, prom, shift_m, area_ratio)

    note = (
        f"peak_frac={peak_frac:.3f} prom={prom:.2f} "
        f"shift=({dx_m:.1f},{dy_m:.1f})m ar={area_ratio:.2f}"
    )
    return dx_m, dy_m, conf, note


def correct_village(village_dir: str) -> tuple[gpd.GeoDataFrame, object]:
    village = load(village_dir)
    n_truth = 0 if village.example_truths is None else len(village.example_truths)
    print(f"\nLoaded {village.slug}  ({len(village.plots)} plots, {n_truth} example truths)")

    if not village.boundaries_path:
        raise FileNotFoundError(f"boundaries.tif not found in {village_dir}")

    utm = _utm_zone(village.plots.geometry.iloc[0].centroid.x)
    utm_plots = village.plots.to_crs(utm)

    # ── per-plot cross-correlation ────────────────────────────────────────────
    raw: list[dict] = []
    with rasterio.open(village.boundaries_path) as src_b:
        for i, pn in enumerate(village.plots.index):
            geom = village.plots.loc[pn, "geometry"]
            row = village.plots.loc[pn]
            dx_m, dy_m, conf, note = _correct_plot(pn, geom, row, src_b)
            raw.append(dict(pn=pn, dx=dx_m, dy=dy_m, conf=conf, note=note))
            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{len(village.plots)} plots processed …")

    high_conf = [r for r in raw if r["conf"] >= BLEND_THRESH]
    print(f"  High-confidence per-plot shifts: {len(high_conf)}/{len(raw)}")

    # ── global median shift (from high-conf plots, or example truths fallback) ─
    if len(high_conf) >= 20:
        g_dx = statistics.median(r["dx"] for r in high_conf)
        g_dy = statistics.median(r["dy"] for r in high_conf)
        print(f"  Global shift from high-conf plots: dx={g_dx:.1f}m dy={g_dy:.1f}m")
    elif village.example_truths is not None:
        truth_u = village.example_truths.to_crs(utm)
        dxs, dys = [], []
        for tpn in village.example_truths.index:
            if tpn in utm_plots.index:
                o = utm_plots.loc[tpn, "geometry"].centroid
                t = truth_u.loc[tpn, "geometry"].centroid
                dxs.append(t.x - o.x)
                dys.append(t.y - o.y)
        g_dx = statistics.median(dxs) if dxs else 0.0
        g_dy = statistics.median(dys) if dys else 0.0
        print(f"  Global shift from example truths: dx={g_dx:.1f}m dy={g_dy:.1f}m")
    else:
        g_dx, g_dy = 0.0, 0.0
        print("  No global shift available — using zero")

    # ── consistency check: flag outliers far from the median shift field ──────
    if len(high_conf) >= 20:
        shifts = np.array([[r["dx"], r["dy"]] for r in high_conf])
        med = np.median(shifts, axis=0)
        dists = np.sqrt(((shifts - med) ** 2).sum(axis=1))
        mad = float(np.median(dists))
        outlier_thresh = max(25.0, mad * 3)
        for r in raw:
            d = float(np.sqrt((r["dx"] - g_dx) ** 2 + (r["dy"] - g_dy) ** 2))
            if d > outlier_thresh and r["conf"] >= BLEND_THRESH:
                r["conf"] *= 0.5
                r["note"] += f" outlier_dist={d:.0f}m"

    # ── build final predictions ───────────────────────────────────────────────
    rows = []
    for r in raw:
        pn = r["pn"]
        conf = r["conf"]
        geom_utm = utm_plots.loc[pn, "geometry"]

        if conf < MIN_CONF:
            # Can't place confidently — flag it
            geom_4326 = village.plots.loc[pn, "geometry"]
            rows.append(dict(
                plot_number=pn,
                status="flagged",
                confidence=None,
                method_note=f"low_conf: {r['note']}"[:200],
                geometry=geom_4326,
            ))
            continue

        # Blend per-plot shift toward global for moderate-confidence plots
        blend = min(1.0, conf / BLEND_THRESH)
        final_dx = blend * r["dx"] + (1 - blend) * g_dx
        final_dy = blend * r["dy"] + (1 - blend) * g_dy

        shifted_utm = translate(geom_utm, final_dx, final_dy)
        shifted_4326 = _to_crs(shifted_utm, utm, "EPSG:4326")

        rows.append(dict(
            plot_number=pn,
            status="corrected",
            confidence=round(conf, 4),
            method_note=r["note"][:200],
            geometry=shifted_4326,
        ))

    preds = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    n_corr = (preds["status"] == "corrected").sum()
    n_flag = (preds["status"] == "flagged").sum()
    print(f"  Predictions: {n_corr} corrected, {n_flag} flagged")
    return preds, village


def main(village_dirs: list[str]) -> None:
    for vdir in village_dirs:
        preds, village = correct_village(vdir)
        out = write_predictions(Path(vdir) / "predictions.geojson", preds)
        print(f"  Wrote → {out}")

        if village.example_truths is not None:
            print()
            print(score(preds, village))


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["data/34855_vadnerbhairav_chandavad_nashik"]
    main(dirs)

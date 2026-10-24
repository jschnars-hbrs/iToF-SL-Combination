#!/usr/bin/env python3
"""Fusion approaches on a precomputed calibration: 1 consistency error, 2 Gaussian ML fusion, 3 active brightness trail.

Usage:
    python approaches.py --calibration ../Calibrations/<name>/calibration.json \
        --approaches all --sl test.exr --tof test.pcd [--save --name my_results]
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
import scipy.signal
from scipy.spatial import cKDTree

import overrides
from dot_calibration import DotCalibration, ToFSampler

CONNECT_DOTS = True             # connect dots with lines in the depth comparison plots
ERROR_DISTRIBUTION_CSV = False  # write an error histogram CSV per approach (needs GT_DISTANCE)
GT_DISTANCE = None              # [m] ground truth of the test scene

EPS_THRESHOLD = 0.1             # [1/m] consistency-error gate |1/Z_ToF − 1/Z_tri|

PEAK_MIN_DISTANCE = 3           # samples between AB peaks along a trail
PEAK_PROMINENCE_FRAC = 0.05     # "strong" peak: prominence relative to the trail maximum
WEAK_PEAK_MAX_SIGMA = 3.0       # weak peaks must agree with iToF within this many σ_ToF
AB_SMOOTH_SIGMA = 2.0           # samples, speckle smoothing before peak finding
AB_BG_WIN = 21                  # samples, local baseline window
AB_MIN_CONTRAST = 0.01          # min peak prominence relative to the local baseline
AB_NOISE_K = 3.0                # min peak prominence in σ of the smoothed noise
AB_AMBIGUITY_RATIO = 0.5        # weak peak invalid unless ε_best ≤ ratio · ε_runner-up
AB_CROSS_HALF = 1               # ±px perpendicular averaging along the trail

AB_SELECT_MODE = "mpi"          # "mpi" (one-sided, MPI-aware) | "min_eps"
MPI_MAX_W = 1.2                 # [1/m] max iToF excess in 1/z
MPI_TOL_W = 0.05                # [1/m] tolerance behind the iToF
BLOB_NOISE_K = 2.0              # DoG blob must exceed this many σ of the DoG noise
BLOB_RADIUS = 2.5               # px, trail peak ↔ blob distance
TOF_MAD_WIN = 5                 # px, window of the robust iToF spread
TOF_MAD_MAX = 0.05              # m, above: iToF unusable at that peak
MPI_LOCAL_RADIUS = 40           # px, neighbourhood of the local consistency check
MPI_LOCAL_MIN = 3               # neighbours needed for the check
MPI_LOCAL_TOL = 0.3             # [1/m] allowed deviation from the neighbours' residual

SATURATION_LEVEL = None         # None: dtype max for integer images, no correction for float
QUAD_HALF_WIN_MIN = 2           # samples, min half window of the 1-D quadratic
QUAD_HALF_WIN_MAX = 40          # samples, max plateau half width
AB_MAX_SHIFT = 4                # samples, max refinement shift from the detected peak
AB_SUBPIXEL_MODE = "fast2d"     # "fast2d" | "GPR" | "quad1d"
AB_PATCH_RADIUS = 6             # px half-size of the 2-D refinement patch

SIGMA_U_FALLBACK = 0.1                            # px, if the calibration has no σ_u
SIGMA_TOF_FALLBACK = lambda z: 0.005 + 0.002 * z  # m, if the calibration has no σ_ToF table
SIGMA_FLOOR = 1e-4                                # m, avoids zero variances
SIGMA_U_FLOOR = 0.2                               # px, lower bound on the runtime σ_u
FUSION_MAX_SIGMA = 20.0         # SL and iToF further apart than this many σ are not fused
FUSION_MPI_SIGMA = True         # inflate σ_ToF by how far the iToF lies behind the SL depth
FUSION_MPI_NSIGMA = 3.0         # … only beyond this many combined σ

overrides.apply(globals())      # GUI values for the constants above (no-op without $ITOF_OVERRIDES)


def _nan_for_none(obj):
    """Recursively replace None (JSON null) by NaN."""
    if obj is None:
        return float("nan")
    if isinstance(obj, list):
        return [_nan_for_none(x) for x in obj]
    return obj


def load_calibration(cal_path: str) -> dict:
    """Load a calibration.json with numpy arrays and trails (z recomputed from w)."""
    with open(cal_path) as f:
        data = json.load(f)
    data["K"] = np.array(data["K"])
    data["K_inv"] = np.array(data["K_inv"])
    data["V"] = np.array(_nan_for_none(data["V"]))
    data["B"] = np.array(data["B"], dtype=float)
    if "U" in data:
        data["U"] = np.array(_nan_for_none(data["U"]))
    if data.get("trails") is not None:
        trails = []
        for t in data["trails"]:
            if t is None:
                trails.append(None)
                continue
            w = np.asarray(t["w"], dtype=float)
            with np.errstate(divide="ignore"):
                z = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), np.inf)
            trails.append({"u": np.asarray(t["u"], dtype=float), "v": np.asarray(t["v"], dtype=float),
                           "w": w, "z": z})
        data["trails"] = trails
    if data.get("sigma_tof_table") is not None:
        data["sigma_tof_table"] = np.asarray(data["sigma_tof_table"], dtype=float).reshape(-1, 2)
    return data


def depth_stats(depths):
    """(n_valid, n_total, mean, std) over finite positive depths."""
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 1e-6)
    if not np.any(valid):
        return 0, len(depths), np.nan, np.nan
    return int(np.sum(valid)), len(depths), float(np.mean(depths[valid])), float(np.std(depths[valid]))


def sigma_models(cal):
    """(σ_u [px], σ_SL(z, gain) [m], σ_ToF(z) [m]) from the calibration, with fallbacks."""
    sigma_u = cal.get("sigma_u_ab")
    if sigma_u is None or not np.isfinite(sigma_u) or sigma_u <= 0:
        sigma_u = cal.get("sigma_u")
    if sigma_u is None or not np.isfinite(sigma_u) or sigma_u <= 0:
        print(f"  (calibration has no sigma_u — using fallback {SIGMA_U_FALLBACK} px)")
        sigma_u = SIGMA_U_FALLBACK
    sigma_u = max(float(sigma_u), SIGMA_U_FLOOR)

    mode = cal.get("mode", cal.get("metadata", {}).get("mode", "x"))
    f_axis = abs(float(cal["K"][0, 0] if mode == "x" else cal["K"][1, 1]))
    B_norm = float(np.linalg.norm(cal["B"]))

    def sigma_sl(z, gain=None):
        """σ_SL = z²·σ_u / gain, gain = trail px per unit 1/z (default f·‖B‖)."""
        g = gain if (gain is not None and np.isfinite(gain) and gain > 1e-9) else B_norm * f_axis
        return max(z * z / g * sigma_u, SIGMA_FLOOR)

    table = cal.get("sigma_tof_table")
    if table is not None and len(table) > 0:
        def sigma_tof(z):
            """σ_ToF interpolated from the calibration noise table."""
            return max(float(np.interp(z, table[:, 0], table[:, 1])), SIGMA_FLOOR)
    else:
        print("  (calibration has no sigma_tof_table — using fallback model)")

        def sigma_tof(z):
            """Fallback σ_ToF model."""
            return max(SIGMA_TOF_FALLBACK(z), SIGMA_FLOOR)

    return float(sigma_u), sigma_sl, sigma_tof


def write_error_distribution(depths, csv_path):
    """Histogram CSV of (depth − GT) in 0.5 cm bins."""
    depths = np.asarray(depths, float)
    error_cm = (depths[np.isfinite(depths) & (depths > 1e-6)] - GT_DISTANCE) * 100.0
    counts = Counter(np.round(error_cm / 0.5) * 0.5)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_cm", "count"])
        for err in sorted(counts):
            writer.writerow([f"{err:.1f}", counts[err]])
    print(f"\nError distribution saved to {csv_path}")
    print(f"  Mean error: {error_cm.mean():.2f} cm  Std: {error_cm.std():.2f} cm"
          f"  Min: {error_cm.min():.2f} cm  Max: {error_cm.max():.2f} cm")


def maybe_write_error_distribution(setup, depths, approach_num):
    """Write the error CSV if enabled and an output directory is set."""
    if ERROR_DISTRIBUTION_CSV and GT_DISTANCE is not None and setup.get("_output_dir") is not None:
        write_error_distribution(depths, setup["_output_dir"] / f"approach{approach_num}_error_distribution.csv")


def print_average_error(approach_num, depths):
    """Print the error against GT_DISTANCE plus a machine-readable AVG_ERR line."""
    if GT_DISTANCE is None or depths is None:
        return
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 1e-6)
    if not np.any(valid):
        return
    err = depths[valid] - GT_DISTANCE
    mean_err, abs_mean, std_err = np.mean(err), np.mean(np.abs(err)), np.std(err)
    min_err, max_err = np.min(err), np.max(err)
    n_valid, n_total = int(np.sum(valid)), len(depths)
    print(f"\n  ** Approach {approach_num} — Average Error **")
    print(f"     GT distance:  {GT_DISTANCE:.4f} m")
    print(f"     Valid dots:   {n_valid} / {n_total}")
    print(f"     Mean error:   {mean_err:+.4f} m")
    print(f"     Mean |error|: {abs_mean:.4f} m")
    print(f"     Std error:    {std_err:.4f} m")
    print(f"     Min error:    {min_err:+.4f} m")
    print(f"     Max error:    {max_err:+.4f} m")
    print(f"AVG_ERR|{approach_num}|{GT_DISTANCE:.4f}|{n_valid}|{n_total}|{mean_err:.6f}|{abs_mean:.6f}"
          f"|{std_err:.6f}|{min_err:.6f}|{max_err:.6f}")


def ensure_trails(dotCal, cal, image_shape):
    """The calibration's trails, or trails fitted from its detections for old calibration files."""
    if cal.get("trails") is not None:
        return cal["trails"]
    print("  (calibration has no precomputed trails — fitting from detections)")
    z_min = cal.get("metadata", {}).get("trail_z_min", 0.4)
    if cal.get("subpixel_list") is not None and cal.get("U") is not None:
        return dotCal.fit_trails_from_detections(cal["subpixel_list"], cal["U"], z_min=z_min,
                                                 image_shape=image_shape, V=cal["V"], B=cal["B"], K=cal["K"])
    return dotCal.build_dot_trails_1d(cal["V"], cal["B"], cal["K"], z_min=z_min, image_shape=image_shape)


def common_setup(dotCal, cal, sl_path, tof_path):
    """Load the test SL image and ToF cloud and everything the approaches share; sl_path None: SL image from the PCD's gray values."""
    tof_test = dotCal.load_tof_pcd(tof_path, unit_scale=cal.get("metadata", {}).get("pcd_unit_scale", 0.001),
                                   depth_mode="axial")
    if sl_path is None:
        if "intensity_map" not in tof_test:
            raise ValueError(f"No SL image given and {tof_path} is not an organized cloud with gray values")
        print("SL image: gray values of the ToF cloud (no EXR given)")
        sl_gray = tof_test["intensity_map"].astype(np.float32)
    else:
        sl_gray = dotCal.read_image(sl_path)
    H, W = sl_gray.shape[:2]
    trails = ensure_trails(dotCal, cal, (H, W))
    print(f"Trails: {sum(t is not None for t in trails)} / {len(trails)} dots calibrated "
          f"(mode: {cal.get('mode', 'x')})")

    if SATURATION_LEVEL is not None:
        sat_level = float(SATURATION_LEVEL)
    elif np.issubdtype(sl_gray.dtype, np.integer):
        sat_level = float(np.iinfo(sl_gray.dtype).max)
    else:
        sat_level = None
    return {
        "trails": trails,
        "tof_sampler": ToFSampler(tof_test, cal["K"]),
        "sl_gray": sl_gray,
        "sat_level": sat_level,
        "H": H, "W": W,
        "blob_tree": dot_blob_tree(sl_gray),
        "tof_z": tof_depth_raster(tof_test, cal["K"], H, W),
    }


def dot_blob_tree(sl_gray):
    """KD-tree of the 2-D dot centres (DoG maxima above BLOB_NOISE_K · noise), or None."""
    img = np.asarray(sl_gray, float)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = np.where(np.isfinite(img), img, np.nanmedian(img))
    dog = scipy.ndimage.gaussian_filter(img, 1.5) - scipy.ndimage.gaussian_filter(img, 5.0)
    noise = 1.4826 * float(np.median(np.abs(dog - np.median(dog))))
    peaks = (dog == scipy.ndimage.maximum_filter(dog, 7)) & (dog > BLOB_NOISE_K * noise)
    vy, vx = np.nonzero(peaks)
    return cKDTree(np.c_[vx, vy].astype(float)) if len(vx) else None


def tof_depth_raster(tof_test, K, H, W):
    """Axial iToF depth as an (H, W) image, NaN where there is no return."""
    if "points_map" in tof_test and tof_test["points_map"].shape[:2] == (H, W):
        Z = np.array(tof_test["points_map"][..., 2], float)
    else:
        P = np.asarray(tof_test["points_3d"], float)
        P = P[np.all(np.isfinite(P), axis=1) & (P[:, 2] > 1e-6)]
        uv = np.round(DotCalibration._project_to_pixels(P, K)).astype(int)
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        Z = np.full((H, W), np.nan)
        Z[uv[ok, 1], uv[ok, 0]] = P[ok, 2]
    Z[~(Z > 1e-6)] = np.nan
    return Z


def tof_mad_at(Z, u, v):
    """Robust iToF spread (1.4826·MAD) in a TOF_MAD_WIN window per pixel; inf with < 3 returns."""
    H, W = Z.shape
    h = TOF_MAD_WIN // 2
    ui = np.round(np.asarray(u, float)).astype(int)
    vi = np.round(np.asarray(v, float)).astype(int)
    off = np.arange(-h, h + 1)
    uu = ui[:, None, None] + off[None, None, :]
    vv = vi[:, None, None] + off[None, :, None]
    inside = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
    patch = np.where(inside, Z[vv.clip(0, H - 1), uu.clip(0, W - 1)], np.nan).reshape(len(ui), -1)
    out = np.full(len(ui), np.inf)
    good = np.sum(np.isfinite(patch), axis=1) >= 3
    if np.any(good):
        p = patch[good]
        out[good] = 1.4826 * np.nanmedian(np.abs(p - np.nanmedian(p, axis=1)[:, None]), axis=1)
    return out


def trail_gain(trail):
    """Pixel travel of a trail per unit 1/z."""
    w_span = float(trail["w"][0] - trail["w"][-1])
    return float(np.hypot(trail["u"][0] - trail["u"][-1], trail["v"][0] - trail["v"][-1]) / max(w_span, 1e-9))


def tof_z_at(tof_sampler, u, v):
    """Axial ToF depth at (u, v), NaN if unavailable."""
    P = tof_sampler.point_at(u, v)
    if P is None:
        return np.nan
    z = float(P[2])
    return z if z > 1e-6 else np.nan


def saturation_correct(x, sat_level):
    """Replace saturated trail samples by the mean of their (next) neighbours (Godbaz §4.2)."""
    if sat_level is None:
        return x
    x = np.asarray(x, float).copy()
    sat = x >= sat_level
    n = len(x)
    for i in np.nonzero(sat)[0]:
        if 0 < i < n - 1 and not sat[i - 1] and not sat[i + 1]:
            x[i] = 0.5 * (x[i - 1] + x[i + 1])
        elif 1 < i < n - 2:
            x[i] = 0.5 * (x[i - 2] + x[i + 2])
    return x


def scan_trail(trail, tof_sampler):
    """ToF depth and ε = |1/Z_ToF − w| at every trail sample."""
    Z_tof = np.array([tof_z_at(tof_sampler, u, v) for u, v in zip(trail["u"], trail["v"])])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_tof = np.where(Z_tof > 1e-6, 1.0 / Z_tof, np.nan)
    eps = np.abs(w_tof - trail["w"])
    eps[~np.isfinite(eps)] = np.inf
    return Z_tof, eps


def local_baseline(sig, win=None):
    """Running background of a trail signal: windowed minimum, then box mean."""
    win = AB_BG_WIN if win is None else win
    return scipy.ndimage.uniform_filter1d(scipy.ndimage.minimum_filter1d(sig, win, mode="nearest"),
                                          win, mode="nearest")


def plateau_bounds(sig, k0, rel_level=0.5, baseline=None):
    """(lo, hi) of the region around k0 above rel_level of the local amplitude."""
    n = len(sig)
    bg = float(np.percentile(sig, 20)) if baseline is None else float(np.broadcast_to(baseline, np.shape(sig))[k0])
    thr = bg + rel_level * (float(sig[k0]) - bg)
    lo = k0
    while lo > 0 and sig[lo - 1] >= thr and k0 - lo < QUAD_HALF_WIN_MAX:
        lo -= 1
    hi = k0
    while hi < n - 1 and sig[hi + 1] >= thr and hi - k0 < QUAD_HALF_WIN_MAX:
        hi += 1
    return lo, hi


def quad_subpixel(ab, k0, baseline=None, max_shift=None):
    """Fractional trail coordinate of an AB peak from a quadratic over a symmetric window around its plateau."""
    n = len(ab)
    max_shift = AB_MAX_SHIFT if max_shift is None else max_shift
    lo, hi = plateau_bounds(ab, k0, baseline=baseline)
    margin = max(QUAD_HALF_WIN_MIN, (hi - lo + 1) // 2)
    centre = 0.5 * (lo + hi)
    half_span = min(centre - max(0, lo - margin), min(n - 1, hi + margin) - centre)
    lo_w = int(np.ceil(centre - half_span))
    hi_w = int(np.floor(centre + half_span)) + 1
    xs = np.arange(lo_w, hi_w, dtype=float) - centre
    ys = np.asarray(ab[lo_w:hi_w], float)
    m = np.isfinite(ys)
    if np.sum(m) >= 3:
        a, b, _ = np.polyfit(xs[m], ys[m], 2)
        if a < -1e-15:
            k = centre + float(np.clip(-b / (2.0 * a), -half_span, half_span))
            if abs(k - k0) <= max_shift:
                return k
    if 0 < k0 < n - 1 and np.all(np.isfinite(ab[k0 - 1:k0 + 2])):
        y_m, y_0, y_p = ab[k0 - 1], ab[k0], ab[k0 + 1]
        denom = 2.0 * (y_m - 2.0 * y_0 + y_p)
        if abs(denom) > 1e-15:
            return k0 + float(np.clip((y_m - y_p) / denom, -0.5, 0.5))
    return float(k0)


def fast_center_2d(patch, sigma_blob, oversample=4):
    """Fast stand-in for DotCalibration._gpr_peak: smoothed multi-level mask centroid of the central dot, or None."""
    h, w = patch.shape
    I = np.asarray(patch, float)
    if h < 3 or w < 3 or not np.all(np.isfinite(I)):
        return None
    bg = float(np.percentile(I, 20))
    amp = float(I.max() - bg)
    if amp <= 0:
        return None
    I = np.clip((I - bg) / amp, 0.0, None)
    S = scipy.ndimage.gaussian_filter(I, float(np.clip(0.35 * sigma_blob, 1.0, max(h, w))), mode="nearest")
    Z = scipy.ndimage.zoom(S, oversample, order=1)
    gy, gx = np.meshgrid(np.linspace(0, h - 1, Z.shape[0]), np.linspace(0, w - 1, Z.shape[1]), indexing="ij")
    central = (gx - (w - 1) / 2.0) ** 2 + (gy - (h - 1) / 2.0) ** 2 <= (0.5 * min(h, w)) ** 2
    seed = np.unravel_index(int(np.argmax(np.where(central, Z, -np.inf))), Z.shape)
    bg_m = float(np.percentile(Z, 10))
    amp_m = float(Z[seed]) - bg_m
    if amp_m <= 0:
        return None
    centroids = []
    for level in (0.35, 0.5, 0.65):
        lab, _ = scipy.ndimage.label(Z >= bg_m + level * amp_m)
        comp = lab == lab[seed]
        centroids.append((float(gx[comp].mean()), float(gy[comp].mean())))
    c = np.mean(np.asarray(centroids), axis=0)
    return float(c[0]), float(c[1])


def refine_on_trail(dotCal, sl_gray, trail, k_coarse, mode=None, radius=None, k_anchor=None):
    """(k_sub, d_perp): 2-D dot centre near k_coarse projected onto the trail; (k_coarse, nan) if implausible."""
    mode = AB_SUBPIXEL_MODE if mode is None else mode
    radius = AB_PATCH_RADIUS if radius is None else radius
    if mode == "quad1d":
        return float(k_coarse), np.nan

    u, v = trail["u"], trail["v"]
    n = len(u)
    ks = np.arange(n, dtype=float)
    u0 = float(np.interp(k_coarse, ks, u))
    v0 = float(np.interp(k_coarse, ks, v))
    H, W = sl_gray.shape[:2]
    x0, y0 = int(round(u0)) - radius, int(round(v0)) - radius
    x1, y1 = x0 + 2 * radius + 1, y0 + 2 * radius + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return float(k_coarse), np.nan
    patch = sl_gray[y0:y1, x0:x1]
    if patch.ndim == 3:
        patch = patch.mean(axis=2)

    sigma_blob = radius / np.sqrt(2.0)
    if mode == "GPR":
        c = dotCal._gpr_peak(np.asarray(patch, float), sigma_blob=sigma_blob)
    elif mode == "fast2d":
        c = fast_center_2d(patch, sigma_blob)
    else:
        raise ValueError(f"Unknown AB_SUBPIXEL_MODE: {mode}")
    if c is None:
        return float(k_coarse), np.nan

    i = int(np.clip(np.floor(k_coarse), 0, n - 2))
    t = np.array([u[i + 1] - u[i], v[i + 1] - v[i]], float)
    step = float(np.hypot(*t))
    if step < 1e-9:
        return float(k_coarse), np.nan
    t /= step
    d = np.array([x0 + c[0] - u0, y0 + c[1] - v0])
    along = float(d @ t)
    d_perp = float(abs(d[0] * t[1] - d[1] * t[0]))
    k_sub = k_coarse + along / step
    if abs(along) > radius or d_perp > radius or (k_anchor is not None and abs(k_sub - k_anchor) > AB_MAX_SHIFT):
        return float(k_coarse), np.nan
    return float(np.clip(k_sub, 0, n - 1)), d_perp


def sample_trail_brightness(sl_gray, trail, cross_half=AB_CROSS_HALF):
    """Active brightness along a trail, averaged over ±cross_half px perpendicular to it."""
    u, v = trail["u"], trail["v"]
    du, dv = u[-1] - u[0], v[-1] - v[0]
    norm = float(np.hypot(du, dv))
    if norm < 1e-9 or cross_half == 0:
        return DotCalibration.sample_image_bilinear(sl_gray, u, v)
    pu, pv = -dv / norm, du / norm
    acc = np.zeros(len(u))
    cnt = np.zeros(len(u))
    for o in range(-cross_half, cross_half + 1):
        s = DotCalibration.sample_image_bilinear(sl_gray, u + o * pu, v + o * pv)
        good = np.isfinite(s)
        acc[good] += s[good]
        cnt[good] += 1
    out = np.full(len(u), np.nan)
    out[cnt > 0] = acc[cnt > 0] / cnt[cnt > 0]
    return out


def trail_peaks(sig):
    """(smoothed signal, local baseline, peak indices) of one trail brightness vector."""
    sig_s = scipy.ndimage.gaussian_filter1d(sig, AB_SMOOTH_SIGMA)
    base = local_baseline(sig_s)
    resid = sig - sig_s
    noise = (1.4826 * float(np.median(np.abs(resid - np.median(resid))))
             / np.sqrt(2.0 * np.sqrt(np.pi) * AB_SMOOTH_SIGMA))
    min_prom = max(AB_MIN_CONTRAST * float(np.median(base)), AB_NOISE_K * noise)
    peaks, _ = scipy.signal.find_peaks(sig_s - base, prominence=min_prom, distance=PEAK_MIN_DISTANCE)
    return sig_s, base, peaks


def sl_tof_conflict(z_sl, z_tof, gain, sigma_sl_fn, sigma_tof_fn):
    """True if SL and iToF disagree by more than FUSION_MAX_SIGMA combined σ."""
    if not (np.isfinite(z_sl) and z_sl > 1e-6 and np.isfinite(z_tof) and z_tof > 1e-6):
        return False
    return abs(z_sl - z_tof) > FUSION_MAX_SIGMA * np.hypot(sigma_sl_fn(z_sl, gain), sigma_tof_fn(z_tof))


def weak_peak_rejected(r, sigma_tof_fn, max_sigma=None):
    """True for a weak AB peak that does not agree with iToF within max_sigma · σ_ToF."""
    if not r["weak_peak"]:
        return False
    max_sigma = WEAK_PEAK_MAX_SIGMA if max_sigma is None else max_sigma
    zt = r["z_tof"]
    return not (np.isfinite(zt) and abs(r["z_sl"] - zt) <= max_sigma * sigma_tof_fn(zt))


def ab_gated(r, sigma_tof_fn, weak_max_sigma=None):
    """True if a front-end candidate is not used (in 'min_eps' mode: ε gate + weak-peak check)."""
    if not r["valid"]:
        return True
    if AB_SELECT_MODE == "mpi":
        return False
    return ((EPS_THRESHOLD is not None and r["eps_best"] > EPS_THRESHOLD)
            or weak_peak_rejected(r, sigma_tof_fn, weak_max_sigma))


def ab_trail_candidates(dotCal, cal, setup, keep_curves=False):
    """Shared front-end of Approaches 2/3: AB peaks per trail → peak selection → 2-D refinement → depth (cached per setup)."""
    cache_key = (AB_SUBPIXEL_MODE, AB_PATCH_RADIUS, keep_curves, AB_SMOOTH_SIGMA, AB_CROSS_HALF,
                 PEAK_MIN_DISTANCE, AB_BG_WIN, AB_MIN_CONTRAST, AB_NOISE_K, AB_AMBIGUITY_RATIO,
                 AB_MAX_SHIFT, AB_SELECT_MODE, MPI_MAX_W, MPI_TOL_W, BLOB_RADIUS, TOF_MAD_WIN,
                 TOF_MAD_MAX, MPI_LOCAL_RADIUS, MPI_LOCAL_MIN, MPI_LOCAL_TOL)
    cached = setup.get("_ab_cache", {}).get(cache_key)
    if cached is not None:
        return [dict(r) for r in cached]

    tof_sampler = setup["tof_sampler"]
    sl_gray = setup["sl_gray"]

    def refine(r, trail, sig_s, base, k0):
        """Refine peak k0 and store position, SL / iToF depth and ε in r."""
        k_sub, d_perp = refine_on_trail(dotCal, sl_gray, trail, quad_subpixel(sig_s, k0, baseline=base),
                                        k_anchor=k0)
        ks = np.arange(len(trail["u"]), dtype=float)
        u_sub = float(np.interp(k_sub, ks, trail["u"]))
        v_sub = float(np.interp(k_sub, ks, trail["v"]))
        w_sub = float(np.interp(k_sub, ks, trail["w"]))
        z_sl = 1.0 / w_sub if w_sub > 1e-9 else np.nan
        z_tof = tof_z_at(tof_sampler, u_sub, v_sub)
        eps_ref = eps_signed = np.inf
        if np.isfinite(z_sl) and z_sl > 1e-6 and np.isfinite(z_tof) and z_tof > 1e-6:
            eps_signed = 1.0 / z_tof - 1.0 / z_sl
            eps_ref = abs(eps_signed)
        r.update(eps_best=float(eps_ref), eps_signed=float(eps_signed), k_best=k0, k_sub=float(k_sub),
                 u_sub=u_sub, v_sub=v_sub, z_sl=z_sl, z_tof=z_tof, d_perp=d_perp)

    results = []
    mpi_cands = {}
    for i, trail in enumerate(setup["trails"]):
        if trail is None:
            continue
        r = dict(i=i, valid=False, eps_best=np.inf, k_best=-1, k_sub=np.nan, u_sub=np.nan, v_sub=np.nan,
                 z_sl=np.nan, z_tof=np.nan, n_peaks=0, gain=trail_gain(trail), d_perp=np.nan, ambiguous=False,
                 weak_peak=False, eps_signed=np.nan, reject=None)
        results.append(r)

        ab = saturation_correct(sample_trail_brightness(sl_gray, trail), setup["sat_level"])
        finite = np.isfinite(ab)
        if not np.any(finite):
            continue
        sig = np.where(finite, ab, 0.0)
        if float(sig.max()) <= 0:
            continue
        sig_s, base, peaks = trail_peaks(sig)
        r["n_peaks"] = len(peaks)
        if keep_curves:
            r.update(ab=sig_s, ab_raw=ab, peaks=peaks)
        if len(peaks) == 0:
            continue

        Z_tof_pk = np.array([tof_z_at(tof_sampler, trail["u"][p], trail["v"][p]) for p in peaks])
        with np.errstate(divide="ignore", invalid="ignore"):
            w_tof_pk = np.where(Z_tof_pk > 1e-6, 1.0 / Z_tof_pk, np.nan)
        eps_pk = np.abs(w_tof_pk - trail["w"][peaks])
        eps_pk[~np.isfinite(eps_pk)] = np.inf
        if keep_curves:
            r["eps_peaks"] = eps_pk

        if AB_SELECT_MODE == "mpi":
            r_pk = w_tof_pk - trail["w"][peaks]
            blob = np.zeros(len(peaks), bool)
            if setup.get("blob_tree") is not None:
                blob = setup["blob_tree"].query(np.c_[trail["u"][peaks], trail["v"][peaks]])[0] <= BLOB_RADIUS
            tof_ok = tof_mad_at(setup["tof_z"], trail["u"][peaks], trail["v"][peaks]) < TOF_MAD_MAX
            in_win = np.isfinite(r_pk) & (r_pk >= -MPI_MAX_W) & (r_pk <= MPI_TOL_W)
            ok = blob & tof_ok & in_win
            if not blob.any():
                r["reject"] = "no_blob"
            elif not (blob & tof_ok).any():
                r["reject"] = "tof"
            elif not ok.any():
                r["reject"] = "mpi_window"
            if not ok.any():
                continue
            p_best = int(np.flatnonzero(ok)[np.argmax(r_pk[ok])])
            k0 = int(peaks[p_best])
            weak_peak = ambiguous = False
            r["r_sel"] = float(r_pk[p_best])
            mpi_cands[len(results) - 1] = (r_pk[ok], peaks[ok], trail, sig_s, base)
        else:
            order = np.argsort(eps_pk)
            p_best = int(order[0])
            if not np.isfinite(eps_pk[p_best]):
                continue
            k0 = int(peaks[p_best])
            strong_pk, _ = scipy.signal.find_peaks(sig_s, prominence=PEAK_PROMINENCE_FRAC * float(sig_s.max()),
                                                   distance=PEAK_MIN_DISTANCE)
            weak_peak = not np.any(np.abs(strong_pk - k0) <= AB_MAX_SHIFT)
            rivals = [p for p in order[1:] if abs(int(peaks[p]) - k0) > 2 * AB_MAX_SHIFT]
            ambiguous = weak_peak and bool(rivals) and eps_pk[p_best] > AB_AMBIGUITY_RATIO * eps_pk[rivals[0]]

        r.update(valid=not ambiguous, ambiguous=bool(ambiguous), weak_peak=bool(weak_peak))
        refine(r, trail, sig_s, base, k0)

    if mpi_cands:
        idx = np.array(sorted(mpi_cands))
        uv = np.array([(results[k]["u_sub"], results[k]["v_sub"]) for k in idx])
        r_sel = np.array([results[k]["r_sel"] for k in idx])
        tree = cKDTree(uv)
        for n, k in enumerate(idx):
            nb = [m for m in tree.query_ball_point(uv[n], MPI_LOCAL_RADIUS) if m != n]
            if len(nb) < MPI_LOCAL_MIN:
                continue
            b = float(np.median(r_sel[nb]))
            if abs(r_sel[n] - b) <= MPI_LOCAL_TOL:
                continue
            r_ok, pk_ok, trail, sig_s, base = mpi_cands[k]
            m = int(np.argmin(np.abs(r_ok - b)))
            if abs(r_ok[m] - b) > MPI_LOCAL_TOL:
                results[k].update(valid=False, reject="local")
                continue
            results[k]["r_sel"] = float(r_ok[m])
            refine(results[k], trail, sig_s, base, int(pk_ok[m]))

    setup.setdefault("_ab_cache", {})[cache_key] = results
    return [dict(r) for r in results]


def _std_note(depths, label="Z"):
    """One-line n / mean / σ summary of a depth array."""
    n_valid, n_total, mean, std = depth_stats(depths)
    if n_valid == 0:
        return f"{label}: no valid depths"
    return f"{label}: n={n_valid}/{n_total}  mean={mean:.4f} m  σ={std * 1e3:.2f} mm"


def _eps_curve_fig(ex, suptitle):
    """ToF depth, 1/Z comparison and ε along one trail."""
    x = np.arange(len(ex["eps_curve"]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot(x, ex["Z_tof_curve"], ".", markersize=3)
    axes[0].set_title("ToF depth along trail")
    axes[0].set_xlabel("Trail sample k")
    axes[0].set_ylabel("Z [m]")

    Z = np.asarray(ex["Z_tof_curve"])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_tof = np.where(Z > 1e-6, 1.0 / Z, np.nan)
    axes[1].plot(x, w_tof, ".", markersize=3, label="1/Z_ToF")
    axes[1].plot(x, ex["w_curve"], "-", linewidth=1, label="1/Z_tri (trail)")
    axes[1].set_title("1/Z comparison (disparity space)")
    axes[1].set_xlabel("Trail sample k")
    axes[1].set_ylabel("1/Z [1/m]")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    eps_mm = np.asarray(ex["eps_curve"], float) * 1e-3
    axes[2].semilogy(x, np.where(np.isfinite(eps_mm), eps_mm, np.nan), ".", markersize=3, color="C2")
    axes[2].axvline(ex["k_best"], color="red", linewidth=1.2, linestyle="--", label=f"k*={ex['k_best']}")
    axes[2].set_title("Consistency error ε")
    axes[2].set_xlabel("Trail sample k")
    axes[2].set_ylabel(r"$\varepsilon$ [mm$^{-1}$]")
    axes[2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    return fig


def _depth_comparison_fig(dot_idx, series, title, limit=None):
    """Depth per dot for several (label, values) series, with σ in the legend and the GT line."""
    fig = plt.figure(figsize=(9, 4))
    for label, vals in series:
        vals = np.asarray(vals, float)
        m = np.isfinite(vals)
        std = depth_stats(vals)[3]
        std_txt = f" (σ={std * 1e3:.1f} mm)" if np.isfinite(std) else ""
        plt.plot(np.asarray(dot_idx)[m], vals[m], ".-" if CONNECT_DOTS else ".", markersize=4,
                 label=label + std_txt)
    if GT_DISTANCE is not None:
        plt.axhline(GT_DISTANCE, color="k", linewidth=0.8, linestyle=":", label=f"GT = {GT_DISTANCE:.2f} m")
    plt.title(title)
    if limit is not None:
        plt.ylim(*limit)
    plt.xlabel("Calibrated dot index i")
    plt.ylabel("Depth [m]")
    plt.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", borderaxespad=0, ncol=3)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.24)
    return fig


def _depth_overlay_fig(sl_gray, us, vs, zs, dot_ids, cbar_label, title, std_note=None):
    """Depth-coded markers over the SL image."""
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(sl_gray, cmap="gray")
    us, vs, zs = np.asarray(us, float), np.asarray(vs, float), np.asarray(zs, float)
    valid = np.isfinite(zs) & np.isfinite(us) & np.isfinite(vs)
    sc = ax.scatter(us[valid], vs[valid], s=40, c=zs[valid], cmap="plasma", alpha=0.9)
    if np.any(valid):
        plt.colorbar(sc, ax=ax, label=cbar_label)
    for u, v, z, i in zip(us, vs, zs, dot_ids):
        if np.isfinite(z) and np.isfinite(u):
            ax.text(u + 2, v + 2, f"{z:.2f}", fontsize=5, color="yellow")
            ax.text(u + 2, v + 12, f"i={i}", fontsize=5, color="cyan")
    if std_note:
        ax.text(0.01, 0.99, std_note, transform=ax.transAxes, fontsize=9, color="white", va="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=4))
    ax.set_title(title)
    ax.axis("off")
    return fig


def _eps_hist_fig(eps_all, threshold, title):
    """Histogram of the selected ε values with the threshold."""
    eps_all = np.asarray(eps_all, float)
    fig = plt.figure(figsize=(6, 3))
    plt.hist(eps_all[np.isfinite(eps_all)], bins=30, edgecolor="k", linewidth=0.4)
    if threshold is not None:
        plt.axvline(threshold, color="red", label=f"threshold={threshold:.3e}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title(title)
    plt.xlabel(r"$\varepsilon$ [1/m]")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def run_approach_1(dotCal, cal, setup, make_plots=False, limit=None, dot_sample=None, exclude=None):
    """Approach 1: ToF depth at the min-ε sample of every trail, gated by EPS_THRESHOLD (Godbaz §4.2)."""
    results = []
    for i, trail in enumerate(setup["trails"]):
        if trail is None:
            continue
        Z_tof, eps = scan_trail(trail, setup["tof_sampler"])
        k_best = int(np.argmin(eps))
        eps_best = float(eps[k_best])
        valid = np.isfinite(eps_best) and (EPS_THRESHOLD is None or eps_best <= EPS_THRESHOLD)
        results.append(dict(i=i, k_best=k_best, eps_best=eps_best,
                            Z_out=float(Z_tof[k_best]) if valid and i not in (exclude or ()) else np.nan,
                            u_out=float(trail["u"][k_best]), v_out=float(trail["v"][k_best]),
                            Z_tof_curve=Z_tof, eps_curve=eps, w_curve=trail["w"]))

    z_out = np.array([r["Z_out"] for r in results])
    eps_all = np.array([r["eps_best"] for r in results], float)
    print(f"Approach 1: scanned {len(results)} trails "
          f"({int(np.sum(~np.isfinite(z_out)))} invalidated by ε-threshold {EPS_THRESHOLD})")
    finite_eps = eps_all[np.isfinite(eps_all)]
    if finite_eps.size:
        print(f"  ε range: {finite_eps.min():.3e} – {finite_eps.max():.3e}")
    print(f"  {_std_note(z_out, 'Z_out (ToF @ min-ε)')}")

    figures = {}
    if make_plots:
        finite_idx = [k for k, r in enumerate(results) if np.isfinite(r["eps_best"])]
        examples = []
        if finite_idx:
            examples = [("best ε", finite_idx[int(np.argmin(eps_all[finite_idx]))]),
                        ("worst ε", finite_idx[int(np.argmax(eps_all[finite_idx]))])]
        examples += [(f"picked dot sample {k}", k) for k in (dot_sample or ())]
        for label, k_ex in examples:
            ex = results[k_ex]
            figures[f"approach1_eps_curve_{label.replace(' ', '_').replace('ε', 'eps')}"] = _eps_curve_fig(
                ex, f"Approach 1 – FL_C Fig. 8 top row – dot i={ex['i']}  (ε*={ex['eps_best']:.3e})  ({label})")
        dot_idx = [r["i"] for r in results]
        figures["approach1_depth_comparison"] = _depth_comparison_fig(
            dot_idx, [("Z_out (ToF @ min-ε trail sample)", z_out)],
            "Approach 1 – Consistency error: selected ToF depth per dot", limit=limit)
        figures["approach1_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"], [r["u_out"] for r in results], [r["v_out"] for r in results], z_out, dot_idx,
            "Z_out [m]", "Approach 1 – Consistency error: depth overlay", std_note=_std_note(z_out))
        figures["approach1_eps_histogram"] = _eps_hist_fig(
            eps_all, EPS_THRESHOLD, r"Approach 1 – Distribution of $\varepsilon^* = \min_k |1/Z_{ToF} - 1/Z_{tri}|$")

    maybe_write_error_distribution(setup, z_out, 1)
    return {"results": results, "figures": figures, "depths": z_out}


def _likelihood_fig(ex):
    """SL, iToF and joint Gaussian likelihood of one fused dot."""
    zs, zt, zf = ex["z_sl"], ex["z_tof"], ex["z_fus"]
    ss, st, sf = ex["sigma_sl"], ex["sigma_tof"], ex["sigma_fus"]
    Zp = np.linspace(min(zs - 4 * ss, zt - 4 * st), max(zs + 4 * ss, zt + 4 * st), 800)
    p_sl = np.exp(-0.5 * ((Zp - zs) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))
    p_tof = np.exp(-0.5 * ((Zp - zt) / st) ** 2) / (st * np.sqrt(2 * np.pi))
    p_joint = p_sl * p_tof
    p_joint = p_joint / np.max(p_joint) * max(np.max(p_sl), np.max(p_tof))

    fig = plt.figure(figsize=(9, 5))
    plt.plot(Zp, p_sl, color="C0", linewidth=2, label=f"SL: N({zs:.3f}, σ={ss * 1e3:.1f} mm)")
    plt.plot(Zp, p_tof, color="C1", linewidth=2, label=f"iToF: N({zt:.3f}, σ={st * 1e3:.1f} mm)")
    plt.plot(Zp, p_joint, color="C2", linewidth=3, label=f"Joint (scaled): max at {zf:.3f} m, σ={sf * 1e3:.1f} mm")
    plt.axvline(zs, color="C0", linestyle="--", alpha=0.5)
    plt.axvline(zt, color="C1", linestyle="--", alpha=0.5)
    plt.axvline(zf, color="red", linewidth=1.5)
    plt.title(f"Approach 2 – likelihoods for dot i={ex['i']}")
    plt.xlabel("Depth z [m]")
    plt.ylabel(r"$\rho(d)$ [m$^{-1}$]")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def run_approach_2(dotCal, cal, setup, make_plots=False, limit=None):
    """Approach 2: AB-peak front-end, then precision-weighted fusion of one Gaussian per modality (Agresti eq. 16/17)."""
    _, sigma_sl_fn, sigma_tof_fn = sigma_models(cal)
    cands = ab_trail_candidates(dotCal, cal, setup, keep_curves=make_plots)

    for r in cands:
        r.update(z_fus=np.nan, sigma_sl=np.nan, sigma_tof=np.nan, sigma_fus=np.nan, conflict=False)
        if ab_gated(r, sigma_tof_fn, FUSION_MAX_SIGMA):
            continue
        zs = r["z_sl"] if np.isfinite(r["z_sl"]) and r["z_sl"] > 1e-6 else np.nan
        zt = r["z_tof"] if np.isfinite(r["z_tof"]) and r["z_tof"] > 1e-6 else np.nan
        if np.isfinite(zs):
            r["sigma_sl"] = sigma_sl_fn(zs, r.get("gain"))
        if np.isfinite(zt):
            r["sigma_tof"] = sigma_tof_fn(zt)

        if np.isfinite(zs) and np.isfinite(zt):
            if FUSION_MPI_SIGMA and zt - zs > FUSION_MPI_NSIGMA * np.hypot(r["sigma_sl"], r["sigma_tof"]):
                r["sigma_tof"] = float(np.hypot(r["sigma_tof"], zt - zs))
            ss, st = r["sigma_sl"], r["sigma_tof"]
            tof_fn = (lambda z: st) if FUSION_MPI_SIGMA else sigma_tof_fn
            if sl_tof_conflict(zs, zt, r.get("gain"), sigma_sl_fn, tof_fn):
                r["conflict"] = True
                continue
            prec = 1.0 / ss ** 2 + 1.0 / st ** 2
            r["z_fus"] = (zs / ss ** 2 + zt / st ** 2) / prec
            r["sigma_fus"] = float(np.sqrt(1.0 / prec))
        elif np.isfinite(zs):
            r["z_fus"], r["sigma_fus"] = zs, r["sigma_sl"]
        elif np.isfinite(zt):
            r["z_fus"], r["sigma_fus"] = zt, r["sigma_tof"]

    gated = [ab_gated(r, sigma_tof_fn, FUSION_MAX_SIGMA) for r in cands]
    z_sl = np.array([np.nan if g else r["z_sl"] for r, g in zip(cands, gated)])
    z_tof = np.array([np.nan if g else r["z_tof"] for r, g in zip(cands, gated)])
    z_fus = np.array([r["z_fus"] for r in cands])

    print(f"Approach 2: {len(cands)} trails, {int(np.sum(np.isfinite(z_fus)))} fused depths "
          f"({sum(r['conflict'] for r in cands)} SL/iToF conflicts > {FUSION_MAX_SIGMA:g}σ not fused)")
    print(f"  {_std_note(z_sl, 'SL (triangulated)')}")
    print(f"  {_std_note(z_tof, 'iToF')}")
    print(f"  {_std_note(z_fus, 'Joint likelihood')}")
    for key, lbl in (("sigma_sl", "σ_SL"), ("sigma_tof", "σ_ToF"), ("sigma_fus", "σ_joint")):
        vals = np.array([r[key] for r in cands], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f"  mean predicted {lbl}: {np.mean(vals) * 1e3:.2f} mm")

    figures = {}
    if make_plots:
        dot_idx = [r["i"] for r in cands]
        figures["approach2_depth_comparison"] = _depth_comparison_fig(
            dot_idx, [("iToF", z_tof), ("SL (triangulation)", z_sl), ("ML fused", z_fus)],
            "Approach 2 – Gaussian ML fusion: iToF vs SL vs fused", limit=limit)
        figures["approach2_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"], [r["u_sub"] for r in cands], [r["v_sub"] for r in cands], z_fus, dot_idx,
            "Fused depth [m]", "Approach 2 – Gaussian ML fusion: depth overlay",
            std_note="\n".join([_std_note(z_sl, "SL"), _std_note(z_tof, "iToF"), _std_note(z_fus, "Joint")]))
        both = [r for r in cands if all(np.isfinite(r[k]) for k in ("z_sl", "z_tof", "z_fus", "sigma_sl", "sigma_tof"))]
        if both:
            figures["approach2_likelihood_example"] = _likelihood_fig(both[len(both) // 2])

    maybe_write_error_distribution(setup, z_fus, 2)
    return {"results": cands, "figures": figures, "depths": z_fus}


def _ab_trail_fig(ex, trail, label):
    """AB signal with peaks and ε at the peaks for one dot of Approach 3."""
    ks = np.arange(len(ex["ab"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
    axes[0].plot(ks, ex["ab"], "-", linewidth=0.8, color="C0", label="AB along trail")
    if len(ex["peaks"]) > 0:
        axes[0].plot(ex["peaks"], ex["ab"][ex["peaks"]], "rv", markersize=6, label=f"peaks ({len(ex['peaks'])})")
    axes[0].axvline(ex["k_best"], color="red", linestyle="--", linewidth=1, label=f"k*={ex['k_best']}")
    if np.isfinite(ex["k_sub"]) and abs(ex["k_sub"] - ex["k_best"]) > 0.01:
        axes[0].axvline(ex["k_sub"], color="orange", linestyle=":", linewidth=1.2, label=f"k_sub={ex['k_sub']:.2f}")
    axes[0].set_title("Active brightness (1-D trail vector)")
    axes[0].set_xlabel("Trail sample k")
    axes[0].set_ylabel("Brightness")
    axes[0].legend(fontsize=6, loc="upper right")

    if len(ex["peaks"]) > 0:
        axes[1].semilogy(ex["peaks"], np.where(np.isfinite(ex["eps_peaks"]), ex["eps_peaks"], np.nan),
                         "rv", markersize=6, label="ε at peaks")
    axes[1].axvline(ex["k_best"], color="red", linestyle="--", linewidth=1, label=f"k*={ex['k_best']}")
    axes[1].plot([], [], " ", label=f"z(k*) = {trail['z'][ex['k_best']]:.2f} m")
    axes[1].set_title(f"Consistency error at peaks  ({label})")
    axes[1].set_xlabel("Trail sample k")
    axes[1].set_ylabel(r"$\varepsilon$ [1/m]")
    axes[1].legend(fontsize=6)
    plt.suptitle(f"Approach 3 – Active Brightness trail – dot i={ex['i']}  "
                 f"(ε*={ex['eps_best']:.3e}, Z_out={ex['Z_out']:.3f} m)", y=1.04)
    plt.tight_layout()
    return fig


def run_approach_3(dotCal, cal, setup, make_plots=False, limit=None):
    """Approach 3: AB-peak front-end, returns the triangulated depth of the selected, refined peak (Godbaz §4.2)."""
    _, _, sigma_tof_fn = sigma_models(cal)
    cands = ab_trail_candidates(dotCal, cal, setup, keep_curves=make_plots)
    for r in cands:
        r["Z_out"] = np.nan if ab_gated(r, sigma_tof_fn) else r["z_sl"]

    z_out = np.array([r["Z_out"] for r in cands])
    eps_all = np.array([r["eps_best"] for r in cands], float)
    n_peakless = sum(r["n_peaks"] == 0 for r in cands)
    n_invalid = int(np.sum(~np.isfinite(z_out)))
    if AB_SELECT_MODE == "mpi":
        n_rej = Counter(r.get("reject") for r in cands if r["n_peaks"] > 0)
        print(f"Approach 3: {len(cands)} trails ({n_peakless} without AB peaks, "
              f"{n_rej['no_blob']} without a 2-D dot, {n_rej['tof']} without usable iToF, "
              f"{n_rej['mpi_window']} outside the MPI window, "
              f"{n_rej['local']} inconsistent with their neighbours; {n_invalid} invalid in total)")
    else:
        print(f"Approach 3: {len(cands)} trails ({n_peakless} without AB peaks, "
              f"{sum(r['ambiguous'] for r in cands)} ambiguous, {n_invalid} invalid in total after ε-gate)")
    print(f"  {_std_note(z_out, 'Z_out (triangulated)')}")

    figures = {}
    if make_plots:
        finite_idx = [k for k, r in enumerate(cands) if np.isfinite(r["eps_best"]) and r.get("ab") is not None]
        if finite_idx:
            sub_eps = eps_all[finite_idx]
            for label, k_ex in (("best eps", finite_idx[int(np.argmin(sub_eps))]),
                                ("worst eps", finite_idx[int(np.argmax(sub_eps))])):
                ex = cands[k_ex]
                figures[f"approach3_trail_{label.replace(' ', '_')}"] = _ab_trail_fig(ex, setup["trails"][ex["i"]], label)
        dot_idx = [r["i"] for r in cands]
        figures["approach3_depth_comparison"] = _depth_comparison_fig(
            dot_idx, [("Z_tri (Approach 3)", z_out)],
            "Approach 3 – Active Brightness: triangulated depth per dot", limit=limit)
        figures["approach3_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"], [r["u_sub"] for r in cands], [r["v_sub"] for r in cands], z_out, dot_idx,
            "Z_tri [m]", "Approach 3 – Active Brightness: depth overlay", std_note=_std_note(z_out))
        figures["approach3_eps_histogram"] = _eps_hist_fig(
            eps_all, EPS_THRESHOLD, r"Approach 3 – Distribution of best $\varepsilon$ (AB peaks)")

    maybe_write_error_distribution(setup, z_out, 3)
    return {"results": cands, "figures": figures, "depths": z_out}


RUNNERS = {1: run_approach_1, 2: run_approach_2, 3: run_approach_3}


def main():
    """Command-line entry: run the selected approaches on one SL/ToF pair."""
    global GT_DISTANCE
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", required=True, help="calibration.json")
    p.add_argument("--approaches", nargs="+", required=True, help="1 2 3 or 'all'")
    p.add_argument("--sl", default=None, help="test structured-light image (.exr); default: gray values of the PCD")
    p.add_argument("--tof", required=True, help="test ToF point cloud (.pcd)")
    p.add_argument("--save", action="store_true", help="save plots to Results/<name>/")
    p.add_argument("--name", default=None, help="results folder name (required with --save)")
    args = p.parse_args()
    if args.save and not args.name:
        sys.exit("Error: --name is required when --save is set.")

    GT_DISTANCE = DotCalibration.parse_distance_from_name(args.sl or args.tof)
    if GT_DISTANCE is None:
        print("Warning: no ground-truth distance in the file name.", file=sys.stderr)
    else:
        print(f"Ground-truth distance from filename: {GT_DISTANCE} m")
    if args.save:
        plt.switch_backend("Agg")

    cal = load_calibration(args.calibration)
    dotCal = DotCalibration()
    selected = [1, 2, 3] if "all" in args.approaches else sorted({int(a) for a in args.approaches})
    setup = common_setup(dotCal, cal, args.sl, args.tof)
    if args.save:
        setup["_output_dir"] = Path(__file__).resolve().parent.parent / "Results" / args.name

    all_figures = {}
    for num in selected:
        if num not in RUNNERS:
            print(f"Warning: unknown approach {num}, skipping.", file=sys.stderr)
            continue
        print(f"\n{'=' * 60}\n  Approach {num}\n{'=' * 60}\n")
        result = RUNNERS[num](dotCal, cal, setup, make_plots=args.save)
        all_figures.update(result["figures"])
        print_average_error(num, result["depths"])

    if args.save and all_figures:
        out = setup["_output_dir"]
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in all_figures.items():
            fig.savefig(str(out / f"{name}.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved {len(all_figures)} plot(s) to {out}")


if __name__ == "__main__":
    main()

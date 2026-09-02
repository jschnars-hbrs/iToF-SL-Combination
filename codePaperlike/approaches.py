#!/usr/bin/env python3
"""Run fusion approaches using a pre-computed calibration (paper-faithful).

The runtime iterates over the PRECOMPUTED 1-D dot trails (FL_C §4.2) — no blob
detection is needed at test time:

  1  Consistency Error   – min-ε scan per trail, returns the iToF depth
  2  Gaussian ML fusion  – AB-peak front-end + Agresti eq. 16/17 (single
                           Gaussian per modality, calibration-derived σ)
  3  Active Brightness   – AB-peak front-end, returns the triangulated depth

Usage examples:
    python approaches.py --calibration ../Calibrations/my_cal/calibration.json \
        --approaches 1 --sl test.exr --tof test.pcd

    python approaches.py --calibration ../Calibrations/my_cal/calibration.json \
        --approaches all --sl test.exr --tof test.pcd --save --name my_results
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.signal

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dot_calibration import DotCalibration, ToFSampler


# ═══════════════════════════════════════════════════════════════════════════
# Global configuration
# ═══════════════════════════════════════════════════════════════════════════

CONNECT_DOTS = True  # True: connect dots with lines in depth comparison plots

Error_Distribution = False  # Adds error distribution CSV output (requires GT_DISTANCE to be set)

Average_Error = True  # Adds average error printout to console (requires GT_DISTANCE to be set)

GT_DISTANCE = None  # Set automatically from SL filename (e.g. "SL_Flat_Wall_1.0m_On.exr" → 1.0)

# Consistency-error gate (FL_C §4.2: "We invalidate incorrect detections by
# thresholding ε").  Units 1/m — an ε of 0.1 corresponds to a ToF/triangulation
# depth mismatch of ~10 cm at 1 m (Δz ≈ ε·z²). Wrong-peak selections on
# intersecting trails produce ε well above this; legitimate dots stay below it
# even with ToF noise + ~1 px SL localisation error (Δw ≈ δx/(f·‖B‖)).
EPS_THRESHOLD = 0.1

# Active-brightness peak finding along the 1-D trails (relative to each trail's
# own maximum, so exposure/reflectivity independent). The signal is Gaussian-
# smoothed first: dots appear as WIDE speckly plateaus along the trail, and
# unsmoothed peak finding locks onto individual speckle grains.
PEAK_PROMINENCE_FRAC = 0.05
PEAK_MIN_DISTANCE = 3        # samples (~pixels along the trail)
AB_SMOOTH_SIGMA = 2.0        # samples — speckle suppression before find_peaks
AB_CROSS_HALF = 1            # ±px perpendicular averaging (paper: integrate
                             # "over the high intensity region of each dot",
                             # 3×3 kernel for sparse imaging)

# Saturation level for the trail-based saturation correction (FL_C §4.2).
# None = auto: integer images saturate at the dtype max, float (EXR) images
# are assumed unsaturated unless a level is set here explicitly.
SATURATION_LEVEL = None

# Bounds (in trail samples ≈ pixels) for the adaptive window of the locally
# fitted quadratic used for 1-D subpixel refinement of the selected AB peak
# (FL_C §4.2). The window adapts to the dot's above-half-maximum width along
# the trail — wide plateau dots need a plateau-wide fit, small far-range dots
# a narrow one.
QUAD_HALF_WIN_MIN = 2
QUAD_HALF_WIN_MAX = 40

# Fallbacks when the calibration carries no measured noise data (legacy files).
SIGMA_U_FALLBACK = 0.1       # px
SIGMA_TOF_FALLBACK = lambda z: 0.005 + 0.002 * z  # m
SIGMA_FLOOR = 1e-4           # m — avoids zero variances on noise-free sims


def _parse_gt_distance(sl_path):
    """Extract ground-truth distance in meters from the SL filename.

    Looks for a pattern like '1.0m', '_0.75m_' or '_4.0m.' in the filename, and
    resolves Pos0…Pos9 via the logbook table.
    Returns the distance as float, or None if not found.
    """
    # Same rules as the calibration loader, so bare names like "1.00m.exr"
    # parse as well as "SL_Flat_Wall_1.0m_On.png".
    return DotCalibration.parse_distance_from_name(sl_path)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sanitize_nan(obj):
    """Recursively convert None back to NaN when loading JSON arrays."""
    if obj is None:
        return float("nan")
    if isinstance(obj, list):
        return [_sanitize_nan(x) for x in obj]
    return obj


def load_calibration(cal_path: str) -> dict:
    """Load calibration JSON, converting lists back to numpy arrays.

    New (paperlike) fields, all optional for backward compatibility:
      mode            : "x" or "y" travel mode
      trails          : per-dot {"u", "v", "w"} lists (w = 1/z), None for
                        uncalibrated dots; "z" is recomputed from w
      sigma_u         : subpixel localisation noise [px]
      sigma_tof_table : rows (z, sigma_z) from the flat-wall calibration
    """
    with open(cal_path) as f:
        data = json.load(f)
    data["K"] = np.array(data["K"])
    data["K_inv"] = np.array(data["K_inv"])
    data["V"] = np.array(_sanitize_nan(data["V"]))
    data["B"] = np.array(data["B"], dtype=float)
    if "U" in data:
        data["U"] = np.array(_sanitize_nan(data["U"]))
    if data.get("trails") is not None:
        trails = []
        for t in data["trails"]:
            if t is None:
                trails.append(None)
                continue
            w = np.asarray(t["w"], dtype=float)
            with np.errstate(divide="ignore"):
                z = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), np.inf)
            trails.append({"u": np.asarray(t["u"], dtype=float),
                           "v": np.asarray(t["v"], dtype=float),
                           "w": w, "z": z})
        data["trails"] = trails
    if data.get("sigma_tof_table") is not None:
        data["sigma_tof_table"] = np.asarray(data["sigma_tof_table"], dtype=float).reshape(-1, 2)
    return data


def robust_sigma(vals):
    """MAD-based robust standard deviation estimate."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(vals - np.median(vals)))


def depth_stats(depths):
    """(n_valid, n_total, mean, std) over finite positive depths."""
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 1e-6)
    if not np.any(valid):
        return 0, len(depths), np.nan, np.nan
    return int(np.sum(valid)), len(depths), float(np.mean(depths[valid])), float(np.std(depths[valid]))


def sigma_models(cal):
    """Calibration-derived noise models (Agresti step 5, FL_C flat-wall data).

    Returns (sigma_u [px], sigma_sl(z) [m], sigma_tof(z) [m]).
    σ_SL(z) = z²/(‖B‖·f)·σ_u  — the d⁴/B² variance scaling of Agresti eq. 15 /
    summary eq. 7; σ_ToF(z) interpolates the flat-wall noise table.
    Falls back to documented heuristics for legacy calibrations.
    """
    # Prefer the runtime AB-estimator noise (sigma_u_ab, measured on the
    # calibration images with the same 1-D pipeline the approaches use); the
    # 2-D calibration residual sigma_u understates it.
    sigma_u = cal.get("sigma_u_ab")
    if sigma_u is None or not np.isfinite(sigma_u) or sigma_u <= 0:
        sigma_u = cal.get("sigma_u")
    if sigma_u is None or not np.isfinite(sigma_u) or sigma_u <= 0:
        print(f"  (calibration has no sigma_u — using fallback {SIGMA_U_FALLBACK} px)")
        sigma_u = SIGMA_U_FALLBACK

    mode = cal.get("mode", cal.get("metadata", {}).get("mode", "x"))
    K = cal["K"]
    f_axis = abs(float(K[0, 0])) if mode == "x" else abs(float(K[1, 1]))
    B_norm = float(np.linalg.norm(cal["B"]))

    def sigma_sl(z, gain=None):
        # σ_z = z²·σ_u / gain with gain = px of dot travel per unit 1/z —
        # per-dot from its trail when available, else the global f·‖B‖.
        g = gain if (gain is not None and np.isfinite(gain) and gain > 1e-9) \
            else B_norm * f_axis
        return max(z * z / g * sigma_u, SIGMA_FLOOR)

    table = cal.get("sigma_tof_table")
    if table is not None and len(table) > 0:
        def sigma_tof(z):
            return max(float(np.interp(z, table[:, 0], table[:, 1])), SIGMA_FLOOR)
    else:
        print("  (calibration has no sigma_tof_table — using fallback model)")
        def sigma_tof(z):
            return max(SIGMA_TOF_FALLBACK(z), SIGMA_FLOOR)

    return float(sigma_u), sigma_sl, sigma_tof


def write_error_distribution(depths, csv_path):
    """Write a histogram CSV of (depth − GT) errors, rounded to 0.5 cm bins."""
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 1e-6)
    error_cm_raw = (depths[valid] - GT_DISTANCE) * 100.0
    error_cm_rounded = np.round(error_cm_raw / 0.5) * 0.5
    counts = Counter(error_cm_rounded)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["error_cm", "count"])
        for err in sorted(counts):
            writer.writerow([f"{err:.1f}", counts[err]])
    print(f"\nError distribution saved to {csv_path}")
    print(f"  Mean error: {error_cm_raw.mean():.2f} cm  Std: {error_cm_raw.std():.2f} cm"
          f"  Min: {error_cm_raw.min():.2f} cm  Max: {error_cm_raw.max():.2f} cm")


def maybe_write_error_distribution(setup, depths, approach_num):
    if not Error_Distribution or GT_DISTANCE is None:
        return
    output_dir = setup.get("_output_dir")
    if output_dir is not None:
        write_error_distribution(
            depths, output_dir / f"approach{approach_num}_error_distribution.csv")


def print_average_error(approach_num, depths):
    """Console summary + machine-readable AVG_ERR line (parsed by run_all.py)."""
    if GT_DISTANCE is None or depths is None:
        return
    depths = np.asarray(depths, float)
    valid = np.isfinite(depths) & (depths > 1e-6)
    if not np.any(valid):
        return
    error_m = depths[valid] - GT_DISTANCE
    mean_err, abs_mean = np.mean(error_m), np.mean(np.abs(error_m))
    std_err = np.std(error_m)
    min_err, max_err = np.min(error_m), np.max(error_m)
    n_valid, n_total = int(np.sum(valid)), len(depths)
    print(f"\n  ** Approach {approach_num} — Average Error **")
    print(f"     GT distance:  {GT_DISTANCE:.4f} m")
    print(f"     Valid dots:   {n_valid} / {n_total}")
    print(f"     Mean error:   {mean_err:+.4f} m")
    print(f"     Mean |error|: {abs_mean:.4f} m")
    print(f"     Std error:    {std_err:.4f} m")
    print(f"     Min error:    {min_err:+.4f} m")
    print(f"     Max error:    {max_err:+.4f} m")
    print(f"AVG_ERR|{approach_num}|{GT_DISTANCE:.4f}|{n_valid}|{n_total}"
          f"|{mean_err:.6f}|{abs_mean:.6f}|{std_err:.6f}"
          f"|{min_err:.6f}|{max_err:.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# Common setup (shared by all approaches)
# ═══════════════════════════════════════════════════════════════════════════

def ensure_trails(dotCal, cal, image_shape):
    """Return the per-dot 1-D trails. For calibration files predating the
    precomputed-trail format, fit them from the stored calibration detections
    (absorbs lens distortion), falling back to the analytic V/B/K trails."""
    if cal.get("trails") is not None:
        return cal["trails"]
    print("  (calibration has no precomputed trails — fitting from detections)")
    z_min = cal.get("metadata", {}).get("trail_z_min", 0.4)
    if cal.get("subpixel_list") is not None and cal.get("U") is not None:
        return dotCal.fit_trails_from_detections(
            cal["subpixel_list"], cal["U"], z_min=z_min,
            image_shape=image_shape, V=cal["V"], B=cal["B"], K=cal["K"])
    return dotCal.build_dot_trails_1d(cal["V"], cal["B"], cal["K"],
                                      z_min=z_min, image_shape=image_shape)


def common_setup(dotCal, cal, sl_path, tof_path):
    """Load test data and the precomputed 1-D dot trails.

    The paper's runtime needs no dot detection: every approach iterates over
    the calibrated trails ("resampling along the 'Dot Trail'", FL_C §4.2).
    """
    PCD_SCALE = cal.get("metadata", {}).get("pcd_unit_scale", 0.001)  # mm → m

    tof_test = dotCal.load_tof_pcd(tof_path, unit_scale=PCD_SCALE, depth_mode="axial")
    tof_sampler = ToFSampler(tof_test, cal["K"])

    sl_gray = dotCal.read_image(sl_path)
    H, W = sl_gray.shape[:2]

    trails = ensure_trails(dotCal, cal, (H, W))
    n_trails = sum(1 for t in trails if t is not None)
    print(f"Trails: {n_trails} / {len(trails)} dots calibrated "
          f"(mode: {cal.get('mode', 'x')})")

    if SATURATION_LEVEL is not None:
        sat_level = float(SATURATION_LEVEL)
    elif np.issubdtype(sl_gray.dtype, np.integer):
        sat_level = float(np.iinfo(sl_gray.dtype).max)
    else:
        sat_level = None  # float EXR: assumed unsaturated

    return {
        "trails": trails,
        "tof_sampler": tof_sampler,
        "sl_gray": sl_gray,
        "sat_level": sat_level,
        "H": H, "W": W,
    }


def tof_z_at(tof_sampler, u, v):
    """Axial ToF depth bilinearly sampled at pixel (u, v), NaN if unavailable."""
    P = tof_sampler.point_at(u, v)
    if P is None:
        return np.nan
    z = float(P[2])
    return z if z > 1e-6 else np.nan


def saturation_correct(x, sat_level):
    """FL_C §4.2 trail saturation correction: each saturated sample X[i] is
    replaced by 0.5(X[i−1]+X[i+1]); if a neighbor is saturated too, by
    0.5(X[i−2]+X[i+2])."""
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
    """ε curve of one trail: sample the ToF depth image at every trail pixel
    and compare against the sample's known triangulated depth in the 1/z
    domain — ε[k] = |1/Z_ToF[k] − w[k]| (FL_C §4.2)."""
    Z_tof = np.array([tof_z_at(tof_sampler, u, v)
                      for u, v in zip(trail["u"], trail["v"])])
    with np.errstate(divide="ignore", invalid="ignore"):
        w_tof = np.where(Z_tof > 1e-6, 1.0 / Z_tof, np.nan)
    eps = np.abs(w_tof - trail["w"])
    eps[~np.isfinite(eps)] = np.inf
    return Z_tof, eps


def plateau_bounds(sig, k0, rel_level=0.5):
    """(lo, hi) sample bounds of the contiguous region around k0 that stays
    above rel_level of the local amplitude — the dot's extent along the trail."""
    n = len(sig)
    bg = float(np.percentile(sig, 20))
    thr = bg + rel_level * (float(sig[k0]) - bg)
    lo = k0
    while lo > 0 and sig[lo - 1] >= thr and k0 - lo < QUAD_HALF_WIN_MAX:
        lo -= 1
    hi = k0
    while hi < n - 1 and sig[hi + 1] >= thr and hi - k0 < QUAD_HALF_WIN_MAX:
        hi += 1
    return lo, hi


def quad_subpixel(ab, k0):
    """1-D subpixel refinement of an AB peak with a locally fitted quadratic
    (FL_C §4.2: "the stationary point of a locally fitted quadratic in 1D").

    The fit window is CENTRED ON THE DOT'S PLATEAU, not on the raw peak: the
    above-half-maximum region around k0 is extended by half its width on both
    sides so the quadratic sees the falling edges — for wide flat-top dots the
    interior is uninformative and a peak-centred window would bias the vertex.
    Returns the fractional trail-sample coordinate.
    """
    n = len(ab)
    lo, hi = plateau_bounds(ab, k0)
    margin = max(QUAD_HALF_WIN_MIN, (hi - lo + 1) // 2)
    centre = 0.5 * (lo + hi)
    # SYMMETRIC window around the plateau centre: a window clipped by the
    # trail end (dots near the vanishing point) would otherwise pull the
    # vertex toward the unclipped side — a systematic depth bias.
    half_span = min(centre - max(0, lo - margin),
                    min(n - 1, hi + margin) - centre)
    lo_w = int(np.ceil(centre - half_span))
    hi_w = int(np.floor(centre + half_span)) + 1
    xs = np.arange(lo_w, hi_w, dtype=float) - centre
    ys = np.asarray(ab[lo_w:hi_w], float)
    m = np.isfinite(ys)
    if np.sum(m) >= 3:
        a, b, _ = np.polyfit(xs[m], ys[m], 2)
        if a < -1e-15:
            return centre + float(np.clip(-b / (2.0 * a), -half_span, half_span))
    # 3-point parabola fallback around the raw peak
    if 0 < k0 < n - 1 and np.all(np.isfinite(ab[k0 - 1:k0 + 2])):
        y_m, y_0, y_p = ab[k0 - 1], ab[k0], ab[k0 + 1]
        denom = 2.0 * (y_m - 2.0 * y_0 + y_p)
        if abs(denom) > 1e-15:
            return k0 + float(np.clip((y_m - y_p) / denom, -0.5, 0.5))
    return float(k0)


def sample_trail_brightness(sl_gray, trail, cross_half=AB_CROSS_HALF):
    """Resample active brightness along a trail (FL_C §4.2), averaging over
    ±cross_half px PERPENDICULAR to the trail direction — the paper integrates
    over the dot's high-intensity region (3×3 kernel for sparse imaging), and
    a single pixel line through a speckle-noisy dot is needlessly noisy."""
    u, v = trail["u"], trail["v"]
    du, dv = u[-1] - u[0], v[-1] - v[0]
    norm = float(np.hypot(du, dv))
    if norm < 1e-9 or cross_half == 0:
        return DotCalibration.sample_image_bilinear(sl_gray, u, v)
    pu, pv = -dv / norm, du / norm  # unit perpendicular
    acc = np.zeros(len(u), dtype=float)
    cnt = np.zeros(len(u), dtype=float)
    for o in range(-cross_half, cross_half + 1):
        s = DotCalibration.sample_image_bilinear(sl_gray, u + o * pu, v + o * pv)
        good = np.isfinite(s)
        acc[good] += s[good]
        cnt[good] += 1
    out = np.full(len(u), np.nan)
    nz = cnt > 0
    out[nz] = acc[nz] / cnt[nz]
    return out


def ab_trail_candidates(dotCal, cal, setup, keep_curves=False):
    """Shared front-end of Approaches 2 and 3 (FL_C §4.2, second approach):

      1. resample active brightness along each dot trail (1-D vector,
         saturation-corrected),
      2. find brightness peaks (relative prominence — trails intersect, so
         multiple peaks are expected),
      3. select the peak minimising the consistency error ε against iToF,
      4. refine to a fractional trail coordinate with a locally fitted
         quadratic and interpolate the subpixel image position,
      5. read the depth off the trail parameterisation there (z = 1/w — every
         trail sample carries its representative triangulated depth).

    Returns one dict per calibrated dot (valid or not) with:
      i, valid, eps_best, k_best, k_sub, u_sub, v_sub, z_sl, z_tof, gain,
      n_peaks (+ ab, ab_raw, peaks, eps_peaks if keep_curves)
    """
    tof_sampler = setup["tof_sampler"]
    sl_gray = setup["sl_gray"]

    results = []
    for i, trail in enumerate(setup["trails"]):
        if trail is None:
            continue
        # Per-dot disparity gain [px per unit w]: pixel travel per 1/z — the
        # empirical f·‖B‖ of this dot's trail, used for σ_SL.
        w_span = float(trail["w"][0] - trail["w"][-1])
        gain = float(np.hypot(trail["u"][0] - trail["u"][-1],
                              trail["v"][0] - trail["v"][-1]) / max(w_span, 1e-9))
        r = dict(i=i, valid=False, eps_best=np.inf, k_best=-1, k_sub=np.nan,
                 u_sub=np.nan, v_sub=np.nan, z_sl=np.nan, z_tof=np.nan,
                 n_peaks=0, gain=gain)
        results.append(r)

        ab_raw = sample_trail_brightness(sl_gray, trail)
        ab = saturation_correct(ab_raw, setup["sat_level"])
        finite = np.isfinite(ab)
        if not np.any(finite):
            continue
        sig = np.where(finite, ab, 0.0)
        sig_max = float(sig.max())
        if sig_max <= 0:
            continue
        # Dots appear as wide speckle plateaus along the trail — smooth before
        # peak finding, or the peaks lock onto individual speckle grains.
        sig_s = scipy.ndimage.gaussian_filter1d(sig, AB_SMOOTH_SIGMA)

        peaks, _ = scipy.signal.find_peaks(
            sig_s, prominence=PEAK_PROMINENCE_FRAC * float(sig_s.max()),
            distance=PEAK_MIN_DISTANCE)
        r["n_peaks"] = len(peaks)
        if keep_curves:
            r["ab"] = sig_s
            r["ab_raw"] = ab
            r["peaks"] = peaks
        if len(peaks) == 0:
            continue

        # ε only at the detected peaks (candidate restriction, FL_C Fig. 8 bottom)
        Z_tof_pk = np.array([tof_z_at(tof_sampler, trail["u"][p], trail["v"][p])
                             for p in peaks])
        with np.errstate(divide="ignore", invalid="ignore"):
            w_tof_pk = np.where(Z_tof_pk > 1e-6, 1.0 / Z_tof_pk, np.nan)
        eps_pk = np.abs(w_tof_pk - trail["w"][peaks])
        eps_pk[~np.isfinite(eps_pk)] = np.inf
        if keep_curves:
            r["eps_peaks"] = eps_pk

        p_best = int(np.argmin(eps_pk))
        if not np.isfinite(eps_pk[p_best]):
            continue
        k0 = int(peaks[p_best])

        k_sub = quad_subpixel(sig_s, k0)
        ks = np.arange(len(trail["u"]), dtype=float)
        u_sub = float(np.interp(k_sub, ks, trail["u"]))
        v_sub = float(np.interp(k_sub, ks, trail["v"]))

        # Depth directly from the trail parameterisation (each trail sample
        # has a known representative triangulated depth, FL_C §4.2) — unlike a
        # pinhole triangulation of (u_sub, v_sub), this stays correct where
        # real lens distortion bends the projection.
        w_sub = float(np.interp(k_sub, ks, trail["w"]))
        z_sl = 1.0 / w_sub if w_sub > 1e-9 else np.nan
        z_tof = tof_z_at(tof_sampler, u_sub, v_sub)

        # Gate on ε at the REFINED position: the coarse peak of a wide plateau
        # can sit many samples from the dot centre even when the refined
        # solution is perfectly consistent.
        eps_ref = np.inf
        if np.isfinite(z_sl) and z_sl > 1e-6 and np.isfinite(z_tof) and z_tof > 1e-6:
            eps_ref = abs(1.0 / z_tof - 1.0 / z_sl)

        r.update(valid=True, eps_best=float(eps_ref), k_best=k0,
                 k_sub=float(k_sub), u_sub=u_sub, v_sub=v_sub,
                 z_sl=z_sl, z_tof=z_tof)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Shared plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _std_note(depths, label="Z"):
    n_valid, n_total, mean, std = depth_stats(depths)
    if n_valid == 0:
        return f"{label}: no valid depths"
    return (f"{label}: n={n_valid}/{n_total}  mean={mean:.4f} m  "
            f"σ={std * 1e3:.2f} mm")


def _eps_curve_fig(ex, suptitle):
    """1×3 figure: ToF depth along trail | 1/Z comparison | ε curve."""
    import matplotlib.pyplot as plt
    n = len(ex["eps_curve"])
    x = np.arange(n)
    eps_plot = np.asarray(ex["eps_curve"], float) * 1e-3  # 1/m → 1/mm

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot(x, ex["Z_tof_curve"], ".", markersize=3)
    axes[0].set_title("ToF depth along trail")
    axes[0].set_xlabel("Trail sample k"); axes[0].set_ylabel("Z [m]")

    with np.errstate(divide="ignore", invalid="ignore"):
        w_tof = np.where(np.asarray(ex["Z_tof_curve"]) > 1e-6,
                         1.0 / np.asarray(ex["Z_tof_curve"]), np.nan)
    axes[1].plot(x, w_tof, ".", markersize=3, label="1/Z_ToF")
    axes[1].plot(x, ex["w_curve"], "-", linewidth=1, label="1/Z_tri (trail)")
    axes[1].set_title("1/Z comparison (disparity space)")
    axes[1].set_xlabel("Trail sample k"); axes[1].set_ylabel("1/Z [1/m]")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    finite_eps = np.where(np.isfinite(eps_plot), eps_plot, np.nan)
    axes[2].semilogy(x, finite_eps, ".", markersize=3, color="C2")
    axes[2].axvline(ex["k_best"], color="red", linewidth=1.2, linestyle="--",
                    label=f"k*={ex['k_best']}")
    axes[2].set_title("Consistency error ε")
    axes[2].set_xlabel("Trail sample k")
    axes[2].set_ylabel(r"$\varepsilon$ [mm$^{-1}$]")
    axes[2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    return fig


def _depth_comparison_fig(dot_idx, series, title, limit = None):
    """Per-dot depth series (list of (label, values)), GT line and σ in legend."""
    import matplotlib.pyplot as plt
    _ls = ".-" if CONNECT_DOTS else "."
    fig = plt.figure(figsize=(9, 4))
    for label, vals in series:
        vals = np.asarray(vals, float)
        m = np.isfinite(vals)
        _, _, _, std = depth_stats(vals)
        std_txt = f" (σ={std * 1e3:.1f} mm)" if np.isfinite(std) else ""
        plt.plot(np.asarray(dot_idx)[m], vals[m], _ls, markersize=4,
                 label=label + std_txt)
    if GT_DISTANCE is not None:
        plt.axhline(GT_DISTANCE, color="k", linewidth=0.8, linestyle=":",
                    label=f"GT = {GT_DISTANCE:.2f} m")
    plt.title(title)
    if limit is not None:
        plt.ylim(limit[0],limit[1])
    plt.xlabel("Calibrated dot index i"); plt.ylabel("Depth [m]")
    plt.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", borderaxespad=0, ncol=3)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.24)
    return fig


def _depth_overlay_fig(sl_gray, us, vs, zs, dot_ids, cbar_label, title, std_note=None):
    """Depth-coded markers over the SL image at the resolved trail positions."""
    import matplotlib.pyplot as plt
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
        ax.text(0.01, 0.99, std_note, transform=ax.transAxes, fontsize=9,
                color="white", va="top",
                bbox=dict(facecolor="black", alpha=0.6, pad=4))
    ax.set_title(title)
    ax.axis("off")
    return fig


def _eps_hist_fig(eps_all, threshold, title):
    import matplotlib.pyplot as plt
    eps_all = np.asarray(eps_all, float)
    fig = plt.figure(figsize=(6, 3))
    plt.hist(eps_all[np.isfinite(eps_all)], bins=30, edgecolor="k", linewidth=0.4)
    if threshold is not None:
        plt.axvline(threshold, color="red", label=f"threshold={threshold:.3e}")
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    plt.title(title)
    plt.xlabel(r"$\varepsilon$ [1/m]"); plt.ylabel("Count")
    plt.grid(alpha=0.3); plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Approach 1: Consistency Error (Microsoft-Paper §4.2, first approach)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_1(dotCal, cal, setup, make_plots=False, limit = None, dot_sample = [], exclude = []):
    """Pure trail scan — no dot localisation, no runtime triangulation.

    For every calibrated dot trail, the ToF depth image is resampled along the
    trail, ε[k] = |1/Z_ToF[k] − 1/Z_tri[k]| is evaluated per sample (Z_tri is
    precomputed), and the ToF depth at k* = argmin ε is returned ("there is no
    dependency on active brightness, all that is required is that there is a
    depth measurement").  Dots with ε* above EPS_THRESHOLD are invalidated.
    """
    tof_sampler = setup["tof_sampler"]

    results = []
    for i, trail in enumerate(setup["trails"]):
        if trail is None:
            continue
        Z_tof, eps = scan_trail(trail, tof_sampler)
        k_best = int(np.argmin(eps))
        eps_best = float(eps[k_best])
        valid = np.isfinite(eps_best) and (EPS_THRESHOLD is None or eps_best <= EPS_THRESHOLD)
        results.append(dict(
            i=i, k_best=k_best, eps_best=eps_best,
            Z_out=float(Z_tof[k_best]) if valid else np.nan,
            u_out=float(trail["u"][k_best]), v_out=float(trail["v"][k_best]),
            Z_tof_curve=Z_tof, eps_curve=eps, w_curve=trail["w"],
        ))

    if exclude:
        exclude_set = set(exclude)
        for dot in results:
            if dot["i"] in exclude_set:
                dot["Z_out"] = np.nan

    z_out_arr = np.array([r["Z_out"] for r in results])
    eps_all = np.array([r["eps_best"] for r in results], float)
    n_invalid = int(np.sum(~np.isfinite(z_out_arr)))
    print(f"Approach 1: scanned {len(results)} trails "
          f"({n_invalid} invalidated by ε-threshold {EPS_THRESHOLD})")
    finite_eps = eps_all[np.isfinite(eps_all)]
    if finite_eps.size:
        print(f"  ε range: {finite_eps.min():.3e} – {finite_eps.max():.3e}")
    print(f"  {_std_note(z_out_arr, 'Z_out (ToF @ min-ε)')}")

    figures = {}
    if make_plots:
        finite_idx = [k for k, r in enumerate(results) if np.isfinite(r["eps_best"])]
        examples = []
        if finite_idx:
            examples = [("best ε", finite_idx[int(np.argmin(eps_all[finite_idx]))]),
                        ("worst ε", finite_idx[int(np.argmax(eps_all[finite_idx]))])]
        if len(dot_sample) > 0:
            for dotsample in dot_sample:
                examples.append((f"picked dot sample {dotsample}", dotsample))
    
        for label, k_ex in examples:
            ex = results[k_ex]
            figures[f"approach1_eps_curve_{label.replace(' ', '_').replace('ε', 'eps')}"] = \
                _eps_curve_fig(ex,
                               f"Approach 1 – FL_C Fig. 8 top row – dot i={ex['i']}  "
                               f"(ε*={ex['eps_best']:.3e})  ({label})")

        figures["approach1_depth_comparison"] = _depth_comparison_fig(
            [r["i"] for r in results],
            [("Z_out (ToF @ min-ε trail sample)", z_out_arr)],
            "Approach 1 – Consistency error: selected ToF depth per dot", limit=limit)

        figures["approach1_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"],
            [r["u_out"] for r in results],
            [r["v_out"] for r in results],
            z_out_arr,
            [r["i"] for r in results],
            "Z_out [m]", "Approach 1 – Consistency error: depth overlay",
            std_note=_std_note(z_out_arr))

        figures["approach1_eps_histogram"] = _eps_hist_fig(
            eps_all, EPS_THRESHOLD,
            r"Approach 1 – Distribution of $\varepsilon^* = \min_k |1/Z_{ToF} - 1/Z_{tri}|$")

    maybe_write_error_distribution(setup, z_out_arr, 1)
    return {"results": results, "figures": figures, "depths": z_out_arr}


# ═══════════════════════════════════════════════════════════════════════════
# Approach 2: Gaussian Maximum-Likelihood Fusion (Agresti & Zanuttigh)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_2(dotCal, cal, setup, make_plots=False, limit = None):
    """AB-peak front-end (identical to Approach 3), then Agresti's ML fusion
    with ONE Gaussian per modality (eq. 16/17 without the spatial 7×7 weighted
    sum — each dot has one concrete candidate at known pixels):

      d_SL ~ N(z_sl, σ_SL²) with σ_SL = z²/(‖B‖f)·σ_u   (calibrated σ_u)
      d_ToF ~ N(z_tof, σ_ToF²) with σ_ToF(z) from the flat-wall noise table

    The joint likelihood is the product of the two Gaussians; its maximum (the
    precision-weighted mean) is the fused depth.
    """
    sigma_u, sigma_sl_fn, sigma_tof_fn = sigma_models(cal)
    cands = ab_trail_candidates(dotCal, cal, setup, keep_curves=make_plots)

    for r in cands:
        r["z_fus"] = np.nan
        r["sigma_sl"] = np.nan
        r["sigma_tof"] = np.nan
        r["sigma_fus"] = np.nan
        if not r["valid"] or (EPS_THRESHOLD is not None and r["eps_best"] > EPS_THRESHOLD):
            continue
        zs, zt = r["z_sl"], r["z_tof"]
        if not (np.isfinite(zs) and zs > 1e-6):
            zs = np.nan
        if not (np.isfinite(zt) and zt > 1e-6):
            zt = np.nan
        if np.isfinite(zs):
            r["sigma_sl"] = sigma_sl_fn(zs, r.get("gain"))
        if np.isfinite(zt):
            r["sigma_tof"] = sigma_tof_fn(zt)

        if np.isfinite(zs) and np.isfinite(zt):
            ss, st = r["sigma_sl"], r["sigma_tof"]
            # Maximum of the product of two Gaussians = precision-weighted mean
            # (equals the argmax of the ±3σ candidate grid, cf. Agresti eq. 16).
            prec = 1.0 / ss ** 2 + 1.0 / st ** 2
            r["z_fus"] = (zs / ss ** 2 + zt / st ** 2) / prec
            r["sigma_fus"] = float(np.sqrt(1.0 / prec))
        elif np.isfinite(zs):
            r["z_fus"] = zs
            r["sigma_fus"] = r["sigma_sl"]
        elif np.isfinite(zt):
            r["z_fus"] = zt
            r["sigma_fus"] = r["sigma_tof"]

    z_sl_arr = np.array([r["z_sl"] if r["valid"] else np.nan for r in cands])
    z_tof_arr = np.array([r["z_tof"] if r["valid"] else np.nan for r in cands])
    z_fus_arr = np.array([r["z_fus"] for r in cands])

    print(f"Approach 2: {len(cands)} trails, "
          f"{int(np.sum(np.isfinite(z_fus_arr)))} fused depths")
    print(f"  {_std_note(z_sl_arr, 'SL (triangulated)')}")
    print(f"  {_std_note(z_tof_arr, 'iToF')}")
    print(f"  {_std_note(z_fus_arr, 'Joint likelihood')}")
    for key, lbl in (("sigma_sl", "σ_SL"), ("sigma_tof", "σ_ToF"), ("sigma_fus", "σ_joint")):
        vals = np.array([r[key] for r in cands], float)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            print(f"  mean predicted {lbl}: {np.mean(vals) * 1e3:.2f} mm")

    figures = {}
    if make_plots:
        import matplotlib.pyplot as plt

        dot_idx = [r["i"] for r in cands]
        figures["approach2_depth_comparison"] = _depth_comparison_fig(
            dot_idx,
            [("iToF", z_tof_arr), ("SL (triangulation)", z_sl_arr),
             ("ML fused", z_fus_arr)],
            "Approach 2 – Gaussian ML fusion: iToF vs SL vs fused", limit=limit)

        figures["approach2_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"],
            [r["u_sub"] for r in cands], [r["v_sub"] for r in cands],
            z_fus_arr, dot_idx,
            "Fused depth [m]", "Approach 2 – Gaussian ML fusion: depth overlay",
            std_note="\n".join([_std_note(z_sl_arr, "SL"),
                                _std_note(z_tof_arr, "iToF"),
                                _std_note(z_fus_arr, "Joint")]))

        # Likelihood example (summary Fig. 9): both Gaussians + joint product.
        both = [r for r in cands
                if np.isfinite(r["z_sl"]) and np.isfinite(r["z_tof"])
                and np.isfinite(r["z_fus"]) and np.isfinite(r["sigma_sl"])
                and np.isfinite(r["sigma_tof"])]
        if both:
            ex = both[len(both) // 2]
            zs, zt, zf = ex["z_sl"], ex["z_tof"], ex["z_fus"]
            ss, st, sf = ex["sigma_sl"], ex["sigma_tof"], ex["sigma_fus"]
            lo = min(zs - 4 * ss, zt - 4 * st)
            hi = max(zs + 4 * ss, zt + 4 * st)
            Zp = np.linspace(lo, hi, 800)
            p_sl = np.exp(-0.5 * ((Zp - zs) / ss) ** 2) / (ss * np.sqrt(2 * np.pi))
            p_tof = np.exp(-0.5 * ((Zp - zt) / st) ** 2) / (st * np.sqrt(2 * np.pi))
            p_joint = p_sl * p_tof
            p_joint = p_joint / np.max(p_joint) * max(np.max(p_sl), np.max(p_tof))

            fig_ll = plt.figure(figsize=(9, 5))
            plt.plot(Zp, p_sl, color="C0", linewidth=2,
                     label=f"SL: N({zs:.3f}, σ={ss * 1e3:.1f} mm)")
            plt.plot(Zp, p_tof, color="C1", linewidth=2,
                     label=f"iToF: N({zt:.3f}, σ={st * 1e3:.1f} mm)")
            plt.plot(Zp, p_joint, color="C2", linewidth=3,
                     label=f"Joint (scaled): max at {zf:.3f} m, σ={sf * 1e3:.1f} mm")
            plt.axvline(zs, color="C0", linestyle="--", alpha=0.5)
            plt.axvline(zt, color="C1", linestyle="--", alpha=0.5)
            plt.axvline(zf, color="red", linewidth=1.5)
            plt.title(f"Approach 2 – likelihoods for dot i={ex['i']}")
            plt.xlabel("Depth z [m]"); plt.ylabel(r"$\rho(d)$ [m$^{-1}$]")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
            plt.grid(alpha=0.3); plt.tight_layout()
            figures["approach2_likelihood_example"] = fig_ll

    maybe_write_error_distribution(setup, z_fus_arr, 2)
    return {"results": cands, "figures": figures, "depths": z_fus_arr}


# ═══════════════════════════════════════════════════════════════════════════
# Approach 3: Active Brightness Trail (Microsoft-Paper §4.2, second approach)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_3(dotCal, cal, setup, make_plots=False, limit = None):
    """AB peaks along each 1-D dot trail → min-ε peak (iToF resolves the
    ambiguity) → locally fitted quadratic subpixel → TRIANGULATED depth
    (read off the trail's 1/z parameterisation at the refined coordinate)."""
    cands = ab_trail_candidates(dotCal, cal, setup, keep_curves=make_plots)

    for r in cands:
        gated = (not r["valid"]) or \
            (EPS_THRESHOLD is not None and r["eps_best"] > EPS_THRESHOLD)
        r["Z_out"] = np.nan if gated else r["z_sl"]

    z_out_arr = np.array([r["Z_out"] for r in cands])
    eps_all = np.array([r["eps_best"] for r in cands], float)
    n_peakless = sum(1 for r in cands if r["n_peaks"] == 0)
    n_invalid = int(np.sum(~np.isfinite(z_out_arr)))
    print(f"Approach 3: {len(cands)} trails "
          f"({n_peakless} without AB peaks, {n_invalid} invalid after ε-gate)")
    print(f"  {_std_note(z_out_arr, 'Z_out (triangulated)')}")

    figures = {}
    if make_plots:
        import matplotlib.pyplot as plt

        finite_idx = [k for k, r in enumerate(cands)
                      if np.isfinite(r["eps_best"]) and r.get("ab") is not None]
        examples = []
        if finite_idx:
            sub_eps = eps_all[finite_idx]
            examples = [("best eps", finite_idx[int(np.argmin(sub_eps))]),
                        ("worst eps", finite_idx[int(np.argmax(sub_eps))])]
        for label, k_ex in examples:
            ex = cands[k_ex]
            trail = setup["trails"][ex["i"]]
            ks = np.arange(len(ex["ab"]))

            fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
            axes[0].plot(ks, ex["ab"], "-", linewidth=0.8, color="C0",
                         label="AB along trail")
            if len(ex["peaks"]) > 0:
                axes[0].plot(ex["peaks"], ex["ab"][ex["peaks"]], "rv",
                             markersize=6, label=f"peaks ({len(ex['peaks'])})")
            axes[0].axvline(ex["k_best"], color="red", linestyle="--", linewidth=1,
                            label=f"k*={ex['k_best']}")
            if np.isfinite(ex["k_sub"]) and abs(ex["k_sub"] - ex["k_best"]) > 0.01:
                axes[0].axvline(ex["k_sub"], color="orange", linestyle=":",
                                linewidth=1.2, label=f"k_sub={ex['k_sub']:.2f}")
            axes[0].set_title("Active brightness (1-D trail vector)")
            axes[0].set_xlabel("Trail sample k"); axes[0].set_ylabel("Brightness")
            axes[0].legend(fontsize=6, loc="upper right")

            if len(ex["peaks"]) > 0:
                axes[1].semilogy(ex["peaks"], np.where(np.isfinite(ex["eps_peaks"]),
                                                       ex["eps_peaks"], np.nan),
                                 "rv", markersize=6, label="ε at peaks")
            axes[1].axvline(ex["k_best"], color="red", linestyle="--", linewidth=1,
                            label=f"k*={ex['k_best']}")
            axes[1].plot([], [], " ", label=f"z(k*) = {trail['z'][ex['k_best']]:.2f} m")
            axes[1].set_title(f"Consistency error at peaks  ({label})")
            axes[1].set_xlabel("Trail sample k")
            axes[1].set_ylabel(r"$\varepsilon$ [1/m]")
            axes[1].legend(fontsize=6)

            plt.suptitle(
                f"Approach 3 – Active Brightness trail – dot i={ex['i']}  "
                f"(ε*={ex['eps_best']:.3e}, Z_out={ex['Z_out']:.3f} m)", y=1.04)
            plt.tight_layout()
            figures[f"approach3_trail_{label.replace(' ', '_')}"] = fig

        figures["approach3_depth_comparison"] = _depth_comparison_fig(
            [r["i"] for r in cands],
            [("Z_tri (Approach 3)", z_out_arr)],
            "Approach 3 – Active Brightness: triangulated depth per dot", limit=limit)

        figures["approach3_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"],
            [r["u_sub"] for r in cands], [r["v_sub"] for r in cands],
            z_out_arr, [r["i"] for r in cands],
            "Z_tri [m]", "Approach 3 – Active Brightness: depth overlay",
            std_note=_std_note(z_out_arr))

        figures["approach3_eps_histogram"] = _eps_hist_fig(
            eps_all, EPS_THRESHOLD,
            r"Approach 3 – Distribution of best $\varepsilon$ (AB peaks)")

    maybe_write_error_distribution(setup, z_out_arr, 3)
    return {"results": cands, "figures": figures, "depths": z_out_arr}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

RUNNERS = {
    1: run_approach_1,
    2: run_approach_2,
    3: run_approach_3,
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Run iToF/SL fusion approaches from a saved calibration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--calibration", required=True,
                   help="Path to calibration JSON file")
    p.add_argument("--approaches", nargs="+", required=True,
                   help="Approaches to run: 1 2 3 or 'all'")
    p.add_argument("--sl", required=True,
                   help="Path to test structured-light image (.exr)")
    p.add_argument("--tof", required=True,
                   help="Path to test ToF point cloud (.pcd)")
    p.add_argument("--save", action="store_true",
                   help="Save plots to Results/<name>/")
    p.add_argument("--name", default=None,
                   help="Name for the results folder (required if --save)")
    return p.parse_args()


def main():
    global GT_DISTANCE
    args = parse_args()

    if args.save and not args.name:
        print("Error: --name is required when --save is set.", file=sys.stderr)
        sys.exit(1)

    # Auto-extract ground-truth distance from SL filename
    gt = _parse_gt_distance(args.sl)
    if gt is not None:
        GT_DISTANCE = gt
        print(f"Ground-truth distance from filename: {GT_DISTANCE} m")
    else:
        print("Warning: could not extract ground-truth distance from SL filename. "
              "Average error will not be computed.", file=sys.stderr)

    if args.save:
        import matplotlib
        matplotlib.use("Agg")

    cal = load_calibration(args.calibration)
    dotCal = DotCalibration()

    if "all" in args.approaches:
        selected = [1, 2, 3]
    else:
        selected = sorted(set(int(a) for a in args.approaches))

    setup = common_setup(dotCal, cal, args.sl, args.tof)
    if args.save:
        setup["_output_dir"] = Path(__file__).resolve().parent.parent / "Results" / args.name

    all_figures = {}
    for num in selected:
        if num not in RUNNERS:
            print(f"Warning: unknown approach {num}, skipping.", file=sys.stderr)
            continue
        print(f"\n{'=' * 60}")
        print(f"  Approach {num}")
        print(f"{'=' * 60}\n")
        result = RUNNERS[num](dotCal, cal, setup, make_plots=args.save)
        all_figures.update(result.get("figures", {}))

        if Average_Error:
            print_average_error(num, result.get("depths"))

    if args.save and all_figures:
        output_dir = setup["_output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in all_figures.items():
            fig.savefig(str(output_dir / f"{name}.png"), dpi=150, bbox_inches="tight")
        print(f"\nSaved {len(all_figures)} plot(s) to {output_dir}")


if __name__ == "__main__":
    main()

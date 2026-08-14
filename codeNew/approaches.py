#!/usr/bin/env python3
"""Run fusion approaches using a pre-computed calibration.

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
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.signal
from scipy.spatial import cKDTree

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dot_calibration import DotCalibration, ToFSampler


# ═══════════════════════════════════════════════════════════════════════════
# Global configuration
# ═══════════════════════════════════════════════════════════════════════════

CONNECT_DOTS = True  # True: connect dots with lines in depth comparison plots

Synthetic_Gaussian = False  # Adds synthetic Gaussian at detected blob for GPR

Error_Distribution = False  # Adds error distribution CSV output (requires GT_DISTANCE to be set)

Average_Error = True  # Adds average error printout to console (requires GT_DISTANCE to be set)

GT_DISTANCE = None  # Set automatically from SL filename (e.g. "SL_Flat_Wall_1.0m_On.exr" → 1.0)

# LoG blob-detection parameters for TEST images, per camera family.
# Used as fallback when the calibration metadata carries no "blob_params".
TEST_BLOB_PARAMS = {
    "OnSemi":    dict(max_sigma=20, num_sigma=30, min_sigma=20, threshold=0.01),
    "Schmersal": dict(max_sigma=10, num_sigma=8, min_sigma=10, threshold=0.02),
}


def _parse_gt_distance(sl_path):
    """Extract ground-truth distance in meters from the SL filename.

    Looks for a pattern like '_1.0m_' or '_0.75m_' or '_4.0m.' in the filename.
    Returns the distance as float, or None if not found.
    """
    stem = Path(sl_path).stem  # e.g. "SL_Flat_Wall_1.0m_On"
    m = re.search(r"_(\d+\.?\d*)m(?:_|$)", stem)
    if m:
        return float(m.group(1))
    return None


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
    """Load calibration JSON, converting lists back to numpy arrays."""
    with open(cal_path) as f:
        data = json.load(f)
    data["K"] = np.array(data["K"])
    data["K_inv"] = np.array(data["K_inv"])
    data["V"] = np.array(_sanitize_nan(data["V"]))
    data["B"] = np.array(data["B"])
    data["U"] = np.array(_sanitize_nan(data["U"]))
    return data


def blob_params_for(cal: dict) -> dict:
    """Test-image blob-detection parameters from calibration metadata.

    Prefers metadata["blob_params"]; falls back to TEST_BLOB_PARAMS keyed by
    whether the calibration data path mentions "schm".
    """
    meta = cal.get("metadata", {})
    if "blob_params" in meta:
        return dict(meta["blob_params"])
    is_schmersal = "schm" in str(meta.get("sl_cal_path", "")).lower()
    return dict(TEST_BLOB_PARAMS["Schmersal" if is_schmersal else "OnSemi"])


def robust_sigma(vals):
    """MAD-based robust standard deviation estimate."""
    vals = np.asarray(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(vals - np.median(vals)))


def log_mog_likelihood(Zcand, depths, sigmas, weights):
    """Log mixture-of-Gaussians likelihood over depth candidates (Agresti eq. 17)."""
    m = np.isfinite(depths) & np.isfinite(sigmas) & (sigmas > 1e-9) & (weights > 0)
    if not np.any(m):
        return np.full_like(Zcand, -np.inf)
    d, s, w = depths[m], sigmas[m], weights[m]
    log_pref = np.log(np.maximum(w, 1e-12)) - np.log(s)              # (M,)
    lc = log_pref[None, :] - 0.5 * ((d[None, :] - np.asarray(Zcand)[:, None]) / s[None, :]) ** 2
    mx = np.max(lc, axis=1)
    return mx + np.log(np.sum(np.exp(lc - mx[:, None]), axis=1) + 1e-12)


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

def common_setup(dotCal, cal, sl_path, tof_path):
    """Load test data, detect dots, build trails and the trail KD-tree."""
    subpixel_list = cal["subpixel_list"]
    PCD_SCALE = cal.get("metadata", {}).get("pcd_unit_scale", 0.001)  # mm → m

    trail_xy = dotCal.build_calibration_trails(subpixel_list)
    n_dist, n_dots = trail_xy.shape[:2]

    tof_test = dotCal.load_tof_pcd(tof_path, unit_scale=PCD_SCALE, depth_mode="axial")
    tof_sampler = ToFSampler(tof_test, cal["K"])

    blobs_test, sl_gray = dotCal.detect_blobs(
        sl_path, add_synthetic_gaussian=Synthetic_Gaussian, visualize=False,
        **blob_params_for(cal))
    H, W = sl_gray.shape[:2]

    # Microsoft-Paper: no fixed grid expected at test time either.
    _, subpix_test = dotCal.detect_subpixel_locations(
        blobs_test, sl_gray, mode="geometricCenter"
    )
    test_uv = np.array([[d["x"], d["y"]] for d in subpix_test], dtype=float)
    print(f"Detected dots in test SL: {len(subpix_test)}")

    # One KD-tree over ALL trail samples; nearest sample → owning dot/ray index.
    # Only dots with a valid calibrated unit vector V can be triangulated, so
    # trails of uncalibrated dots are excluded from the matching.
    flat_pts = trail_xy.reshape(-1, 2)
    pt_dot = np.arange(flat_pts.shape[0]) % n_dots
    v_ok = np.all(np.isfinite(np.asarray(cal["V"])), axis=1)
    usable = np.all(np.isfinite(flat_pts), axis=1) & v_ok[pt_dot]
    if not np.any(usable):
        raise ValueError("No calibrated trails with valid unit vectors available.")
    trail_tree = cKDTree(flat_pts[usable])
    trail_dot_of_pt = pt_dot[usable]

    def nearest_trail(uv):
        """Return (dot index i, pixel distance) of the trail closest to uv."""
        d, k = trail_tree.query(uv, k=1)
        return int(trail_dot_of_pt[k]), float(d)

    return {
        "trail_xy": trail_xy,
        "nearest_trail": nearest_trail,
        "tof_sampler": tof_sampler,
        "sl_gray": sl_gray,
        "test_uv": test_uv,
        "subpix_test": subpix_test,
        "H": H, "W": W,
        "n_dist": n_dist,
        "sample_tof_depth": tof_sampler.depth_at,
    }


def consistency_scan(dotCal, cal, setup):
    """Core of Approaches 1 & 4 (Microsoft-Paper §4.2, first approach).

    For each detected test dot: match the nearest calibration trail, then
    evaluate Z_ToF, Z_tri and ε = |1/Z_ToF − 1/Z_tri| at every trail sample.
    Returns one result dict per dot with the full curves and j* = argmin ε.
    """
    V, B, K_inv = cal["V"], cal["B"], cal["K_inv"]
    trail_xy = setup["trail_xy"]
    nearest_trail = setup["nearest_trail"]
    n_dist = setup["n_dist"]
    sample_tof_depth = setup["sample_tof_depth"]

    results = []
    for k, (uu, vv) in enumerate(setup["test_uv"]):
        best_i, _ = nearest_trail([uu, vv])

        Z_tri_arr = np.empty(n_dist)
        Z_tof_arr = np.empty(n_dist)
        eps_arr = np.empty(n_dist)

        for j in range(n_dist):
            uij = float(trail_xy[j, best_i, 0])
            vij = float(trail_xy[j, best_i, 1])
            Z_tri = dotCal.triangulate_depth(uij, vij, V[best_i], B, K_inv)
            Z_tof, _ = sample_tof_depth(uij, vij)
            Z_tri_arr[j] = max(Z_tri, 1e-9)
            Z_tof_arr[j] = max(Z_tof, 1e-9)
            eps_arr[j] = dotCal.consistency_error(Z_tof_arr[j], Z_tri_arr[j])

        j_best = int(np.argmin(eps_arr))
        Z_raw, _ = sample_tof_depth(uu, vv)

        results.append(dict(
            k=k, u=float(uu), v=float(vv),
            Z_raw=Z_raw,
            i_best=best_i, j_best=j_best,
            eps_best=float(eps_arr[j_best]),
            eps_curve=eps_arr.copy(),
            Z_tri_curve=Z_tri_arr.copy(),
            Z_tof_curve=Z_tof_arr.copy(),
        ))
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Shared plotting helpers
# ═══════════════════════════════════════════════════════════════════════════

def _eps_curve_fig(ex, n_dist, suptitle):
    """1×3 figure: ToF depth along trail | 1/Z comparison | ε curve."""
    import matplotlib.pyplot as plt
    x = np.arange(n_dist)
    eps_plot = ex["eps_curve"] * 1e-3  # 1/m → 1/mm

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
    axes[0].plot(x, ex["Z_tof_curve"], "*", markersize=5)
    axes[0].set_title("ToF depth along trail")
    axes[0].set_xlabel("Trail sample j"); axes[0].set_ylabel("Z [m]")

    axes[1].plot(x, 1 / ex["Z_tof_curve"], "*", markersize=5, label="1/Z_ToF")
    axes[1].plot(x, 1 / ex["Z_tri_curve"], "*", markersize=5, label="1/Z_tri")
    axes[1].set_title("1/Z comparison (disparity space)")
    axes[1].set_xlabel("Trail sample j"); axes[1].set_ylabel("1/Z [1/m]")
    axes[1].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    axes[2].semilogy(x, eps_plot, "*", markersize=5, color="C2")
    axes[2].axvline(ex["j_best"], color="red", linewidth=1.2, linestyle="--",
                    label=f"j*={ex['j_best']}")
    axes[2].set_title("Consistency error ε")
    axes[2].set_xlabel("Trail sample j")
    axes[2].set_ylabel(r"$\varepsilon$ [mm$^{-1}$]")
    axes[2].legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    plt.suptitle(suptitle, y=1.02)
    plt.tight_layout()
    return fig


def _depth_comparison_fig(results, out_label, title):
    """Raw ToF vs output depth, sorted by matched calibration ray index."""
    import matplotlib.pyplot as plt
    _ls = ".-" if CONNECT_DOTS else "."
    results_sorted = sorted(results, key=lambda r: r["i_best"])
    x_i = [r["i_best"] for r in results_sorted]
    fig = plt.figure(figsize=(8, 4))
    plt.plot(x_i, [r["Z_raw"] for r in results_sorted],
             _ls, markersize=5, label="Z_ToF (raw @ detected dot)")
    plt.plot(x_i, [r["Z_out"] for r in results_sorted],
             _ls, markersize=5, label=out_label)
    plt.title(title)
    plt.xlabel("Calibrated ray index i*"); plt.ylabel("Depth [m]")
    plt.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", borderaxespad=0, ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout(); plt.subplots_adjust(bottom=0.22)
    return fig


def _depth_overlay_fig(sl_gray, us, vs, zs, i_best, cbar_label, title):
    """Detected dots over the SL image, coloured by output depth."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(sl_gray, cmap="gray")
    valid = np.isfinite(zs)
    sc = ax.scatter(us[valid], vs[valid], s=40, c=zs[valid], cmap="plasma", alpha=0.9)
    if np.any(valid):
        plt.colorbar(sc, ax=ax, label=cbar_label)
    for u, v, z, i in zip(us, vs, zs, i_best):
        if np.isfinite(z):
            ax.text(u + 2, v + 2, f"{z:.2f}", fontsize=5, color="yellow")
            ax.text(u + 2, v + 12, f"i={i}", fontsize=5, color="cyan")
    ax.set_title(title)
    ax.axis("off")
    return fig


def _eps_hist_fig(eps_all, threshold, title):
    import matplotlib.pyplot as plt
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
# Approach 1: Consistency Error (Microsoft-Paper)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_1(dotCal, cal, setup, make_plots=False):
    """Min-ε trail sample selection; output = ToF depth at that sample
    (Microsoft-Paper §4.2: "The range measurement from ToF corresponding to
    the minimal error sample is returned")."""
    EPS_THRESHOLD = None  # set a float to invalidate high-ε dots

    results = consistency_scan(dotCal, cal, setup)
    for r in results:
        r["Z_out"] = float(r["Z_tof_curve"][r["j_best"]])
        if EPS_THRESHOLD is not None and r["eps_best"] > EPS_THRESHOLD:
            r["Z_out"] = r["Z_raw"]

    print(f"Approach 1: processed {len(results)} dots")
    eps_all = np.array([r["eps_best"] for r in results], float)
    print(f"  ε range: {eps_all.min():.3e} – {eps_all.max():.3e}")
    for r in results:
        print(f"  dot {r['k']:3d}  i*={r['i_best']:3d}  j*={r['j_best']:2d}  "
              f"Z_raw={r['Z_raw']:.4f}  Z_out={r['Z_out']:.4f}  ε*={r['eps_best']:.3e}")

    figures = {}
    if make_plots:
        for label, k_ex in [("dot 94", min(94, len(results) - 1)),
                             ("dot 92", min(92, len(results) - 1))]:
            ex = results[k_ex]
            figures[f"approach1_eps_curve_{label.replace(' ', '_')}"] = _eps_curve_fig(
                ex, setup["n_dist"],
                f"Microsoft-Paper Figure 8 – dot k={ex['k']}  (i*={ex['i_best']}, "
                f"ε*={ex['eps_best']:.3e})  ({label})")

        figures["approach1_depth_comparison"] = _depth_comparison_fig(
            results, "Z_out (ToF @ min-ε trail sample)",
            "Approach 1 – Consistency error: raw ToF vs selected depth")

        figures["approach1_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"],
            np.array([r["u"] for r in results]),
            np.array([r["v"] for r in results]),
            np.array([r["Z_out"] for r in results]),
            [r["i_best"] for r in results],
            "Z_out [m]", "Approach 1 – Consistency error: depth overlay")

        figures["approach1_eps_histogram"] = _eps_hist_fig(
            eps_all, EPS_THRESHOLD,
            r"Distribution of best consistency errors $\varepsilon = |1/Z_{ToF} - 1/Z_{tri}|$")

    z_out_arr = np.array([r["Z_out"] for r in results])
    maybe_write_error_distribution(setup, z_out_arr, 1)
    return {"results": results, "figures": figures, "depths": z_out_arr}


# ═══════════════════════════════════════════════════════════════════════════
# Approach 2: Maximum Likelihood Fusion (Agresti & Zanuttigh)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_2(dotCal, cal, setup, make_plots=False):
    V, B, K_inv = cal["V"], cal["B"], cal["K_inv"]
    nearest_trail = setup["nearest_trail"]
    test_uv = setup["test_uv"]
    tof_sampler = setup["tof_sampler"]
    sl_gray = setup["sl_gray"]

    MAX_TOF_PIX_DIST = 6.0
    TOF_FILL_WIN = 2
    TRAIL_SOFT_SCALE = 6.0    # px – inflates σ_SL when dot is far from trail
    wh = 3                    # spatial neighbourhood half-width (Agresti: 7×7)
    Z_CAND_N = 200

    # Per-dot ToF depths
    test_ztof = np.full(len(test_uv), np.nan)
    test_nnpx = np.full(len(test_uv), np.nan)
    for k, (uu, vv) in enumerate(test_uv):
        z, dpx = tof_sampler.depth_at(uu, vv, search_radius=TOF_FILL_WIN, fallback=np.nan)
        test_ztof[k] = z
        test_nnpx[k] = dpx

    # Per-dot SL triangulation depths
    z_sl = np.full(len(test_uv), np.nan)
    i_best_ml = np.full(len(test_uv), -1, int)
    trail_d = np.full(len(test_uv), np.nan)

    for k, (uu, vv) in enumerate(test_uv):
        i, dpx = nearest_trail([uu, vv])
        i_best_ml[k] = i
        trail_d[k] = dpx
        z = dotCal.triangulate_depth(uu, vv, V[i], B, K_inv)
        if np.isfinite(z) and z > 1e-6:
            z_sl[k] = z

    print(f"Approach 2 – ToF valid: {int(np.sum(np.isfinite(test_ztof)))} / {len(test_uv)}"
          f"  range: {np.nanmin(test_ztof):.2f}–{np.nanmax(test_ztof):.2f} m")
    print(f"Approach 2 – SL  valid: {int(np.sum(np.isfinite(z_sl)))} / {len(test_uv)}"
          f"  range: {np.nanmin(z_sl):.2f}–{np.nanmax(z_sl):.2f} m")

    # Uncertainty models (heuristic stand-ins for Agresti eq. 9 / eq. 15;
    # σ_SL keeps the paper's quadratic growth with depth)
    sigma_tof = np.full(len(test_uv), np.nan)
    sigma_sl = np.full(len(test_uv), np.nan)

    for k, (uu, vv) in enumerate(test_uv):
        if np.isfinite(test_ztof[k]) and test_ztof[k] > 1e-6:
            patch = tof_sampler.neighborhood_depths(uu, vv, half_width=3)
            s_loc = robust_sigma(patch)
            if not np.isfinite(s_loc):
                s_loc = 0.02
            sigma_tof[k] = max(0.01, s_loc + 0.02 * (test_nnpx[k] / max(MAX_TOF_PIX_DIST, 1e-6)) ** 2)

        if np.isfinite(z_sl[k]) and z_sl[k] > 1e-6:
            base_s = 0.01 + 0.06 * (z_sl[k] / 4.0) ** 2
            if np.isfinite(trail_d[k]):
                base_s *= 1.0 + (trail_d[k] / max(TRAIL_SOFT_SCALE, 1e-6)) ** 2
            sigma_sl[k] = max(0.01, base_s)

    # ML Fusion (Agresti eq. 16/17, dot-sparse adaptation: k-NN dots replace
    # the dense 7×7 pixel patch; spatial weights from pixel distance)
    dot_tree = cKDTree(test_uv)
    if len(test_uv) >= 5:
        dd, _ = dot_tree.query(test_uv, k=2)
        sigma_s_px = 0.5 * np.median(dd[:, 1])
    else:
        sigma_s_px = 10.0

    knn_patch = (2 * wh + 1) ** 2
    z_fus = np.full(len(test_uv), np.nan)

    for k in range(len(test_uv)):
        zt, zs = test_ztof[k], z_sl[k]
        st, ss = sigma_tof[k], sigma_sl[k]

        if np.isfinite(zs) and not np.isfinite(zt):
            z_fus[k] = float(zs); continue
        if np.isfinite(zt) and not np.isfinite(zs):
            z_fus[k] = float(zt); continue
        if not (np.isfinite(zt) and np.isfinite(zs) and np.isfinite(st) and np.isfinite(ss)):
            continue

        dnn, nn_idx = dot_tree.query(test_uv[k], k=min(knn_patch, len(test_uv)))
        w_sp = np.exp(-0.5 * (dnn / max(sigma_s_px, 1e-6)) ** 2)

        zmin = max(1e-4, min(zt - 3 * st, zs - 3 * ss))
        zmax = max(zt + 3 * st, zs + 3 * ss)
        if not (np.isfinite(zmin) and np.isfinite(zmax)) or zmax <= zmin:
            continue

        Zcand = np.linspace(zmin, zmax, Z_CAND_N)
        ll_tof = log_mog_likelihood(Zcand, test_ztof[nn_idx], sigma_tof[nn_idx], w_sp)
        ll_sl = log_mog_likelihood(Zcand, z_sl[nn_idx], sigma_sl[nn_idx], w_sp)
        z_fus[k] = float(Zcand[int(np.argmax(ll_tof + ll_sl))])

    valid = np.isfinite(z_fus)
    print(f"Approach 2 – Fused valid: {int(np.sum(valid))} / {len(test_uv)}"
          f"  range: {np.nanmin(z_fus):.2f}–{np.nanmax(z_fus):.2f} m")

    for k in range(len(test_uv)):
        print(f"  dot {k:3d}  i*={i_best_ml[k]:3d}  "
              f"Z_tof={test_ztof[k]:.4f}  Z_sl={z_sl[k]:.4f}  Z_fus={z_fus[k]:.4f}")

    maybe_write_error_distribution(setup, z_fus, 2)

    figures = {}
    if make_plots:
        import matplotlib.pyplot as plt

        figures["approach2_depth_overlay"] = _depth_overlay_fig(
            sl_gray, test_uv[:, 0], test_uv[:, 1], z_fus, i_best_ml,
            "Fused depth [m]", "Approach 2 – ML fusion: depth overlay")

        # Depth comparison
        m = (i_best_ml >= 0) & (np.isfinite(test_ztof) | np.isfinite(z_sl) | np.isfinite(z_fus))
        order = np.argsort(i_best_ml[m])
        x_plot = i_best_ml[m][order]

        fig_dc = plt.figure(figsize=(11, 4))
        _c = "-" if CONNECT_DOTS else ""
        mt = np.isfinite(test_ztof[m][order])
        ms = np.isfinite(z_sl[m][order])
        mf = np.isfinite(z_fus[m][order])
        plt.plot(x_plot[mt], test_ztof[m][order][mt], "o" + _c, markersize=3, label="ToF")
        plt.plot(x_plot[ms], z_sl[m][order][ms], "x" + _c, markersize=3, label="SL (triangulation)")
        plt.plot(x_plot[mf], z_fus[m][order][mf], "." + _c, markersize=6, label="ML fused")
        plt.xlabel("Matched ray index i"); plt.ylabel("Depth [m]")
        plt.title("Approach 2 – ML fusion: ToF vs SL vs Fused (sorted by ray index)")
        plt.grid(alpha=0.3); plt.legend(bbox_to_anchor=(0.5, -0.18), loc="upper center", borderaxespad=0, ncol=3)
        plt.tight_layout(); plt.subplots_adjust(bottom=0.22)
        figures["approach2_depth_comparison"] = fig_dc

        # Likelihood example
        valid_k = [k for k in range(len(test_uv))
                   if np.isfinite(test_ztof[k]) and np.isfinite(z_sl[k]) and np.isfinite(z_fus[k])]
        if valid_k:
            k_ex = valid_k[min(42, len(valid_k) - 1)]
            zt, zs = test_ztof[k_ex], z_sl[k_ex]
            st, ss = sigma_tof[k_ex], sigma_sl[k_ex]

            dnn, nn_idx = dot_tree.query(test_uv[k_ex], k=min(knn_patch, len(test_uv)))
            w_sp = np.exp(-0.5 * (dnn / max(sigma_s_px, 1e-6)) ** 2)

            zmin_p = max(1e-4, min(zt - 4 * st, zs - 4 * ss))
            zmax_p = max(zt + 4 * st, zs + 4 * ss)
            Zp = np.linspace(zmin_p, zmax_p, 500)

            ll_t = log_mog_likelihood(Zp, test_ztof[nn_idx], sigma_tof[nn_idx], w_sp)
            ll_s = log_mog_likelihood(Zp, z_sl[nn_idx], sigma_sl[nn_idx], w_sp)
            ll_j = ll_t + ll_s

            prob_t = np.exp(ll_t - np.max(ll_t))
            prob_s = np.exp(ll_s - np.max(ll_s))
            prob_j = np.exp(ll_j - np.max(ll_j))
            z_max = float(Zp[int(np.argmax(prob_j))])

            fig_ll = plt.figure(figsize=(10, 5))
            plt.plot(Zp, prob_t, label="Likelihood ToF", color="C0", linewidth=2, alpha=0.7)
            plt.plot(Zp, prob_s, label="Likelihood SL", color="C1", linewidth=2, alpha=0.7)
            plt.plot(Zp, prob_j, label="Joint likelihood (fused)", color="C2", linewidth=3)
            plt.axvline(zt, color="C0", linestyle="--", alpha=0.5, label=f"Z_ToF = {zt:.3f} m")
            plt.axvline(zs, color="C1", linestyle="--", alpha=0.5, label=f"Z_SL  = {zs:.3f} m")
            plt.axvline(z_max, color="red", linewidth=2, label=f"Z_fused = {z_max:.3f} m")
            plt.title(f"Approach 2 – ML likelihood curves for dot k={k_ex} (normalised)")
            plt.xlabel("Depth candidate Z [m]"); plt.ylabel("Relative likelihood")
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0); plt.grid(alpha=0.3)
            plt.xlim([zmin_p, zmax_p]); plt.tight_layout()
            figures["approach2_likelihood_example"] = fig_ll

    return {"figures": figures, "depths": z_fus}


# ═══════════════════════════════════════════════════════════════════════════
# Approach 3: Active Brightness Trail (Microsoft-Paper, Section 4.2)
# ═══════════════════════════════════════════════════════════════════════════

def run_approach_3(dotCal, cal, setup, make_plots=False):
    V, B, K_inv = cal["V"], cal["B"], cal["K_inv"]
    trail_xy = setup["trail_xy"]
    nearest_trail = setup["nearest_trail"]
    test_uv = setup["test_uv"]
    sl_gray = setup["sl_gray"]
    H, W = setup["H"], setup["W"]
    sample_tof_depth = setup["sample_tof_depth"]

    # ─── Configuration ────────────────────────────────────────────────────
    PEAK_SNR_MIN = None       # e.g. 3.0 — reject if peak SNR < threshold
    EPS_THRESHOLD_AB = None   # e.g. 0.5 — reject if best ε > threshold
    USE_FALLBACK = False      # True: allow min-ε over all columns (not paper-conform)
    HALF_WIN = 10             # half-width for the wider quadratic sub-pixel refinement

    # Per-dot epipolar row = mean v of that dot's calibration trail.
    # FL_C / Microsoft-Paper §4.2: scan along the dot's epipolar line.
    n_trail_dots = trail_xy.shape[1]
    epipolar_rows_dot = np.clip(
        np.round(np.nanmean(trail_xy[:, :, 1], axis=0)).astype(int), 0, H - 1)

    v_ranges = np.nanmax(trail_xy[:, :, 1], axis=0) - np.nanmin(trail_xy[:, :, 1], axis=0)
    print(f"Trail v-range (px):  mean={np.mean(v_ranges):.3f}  "
          f"max={np.max(v_ranges):.3f}  std={np.std(v_ranges):.3f}")
    if np.max(v_ranges) < 1.0:
        print("  ✓ All trails stay within 1 px vertically — horizontal epipolar assumption valid.")
    else:
        print(f"  ⚠ {np.sum(v_ranges >= 1.0)} dot(s) exceed 1 px v-range — "
              f"horizontal assumption may be inaccurate for those dots.")

    rows_to_scan = np.unique(epipolar_rows_dot)
    row_peaks = {}
    for row in rows_to_scan:
        row_signal = sl_gray[row, :]
        row_max = np.max(row_signal)
        if row_max < 1e-12:
            continue
        pks, _ = scipy.signal.find_peaks(row_signal, prominence=0.001 * row_max,
                                         distance=40, height=0.1)
        if len(pks) > 0:
            row_peaks[int(row)] = pks

    print(f"Scanned {len(rows_to_scan)} unique rows — {len(row_peaks)} have brightness peaks, "
          f"total peaks: {sum(len(p) for p in row_peaks.values())}")

    # ε over ALL columns of a row is needed for the (non-paper) fallback and
    # for the diagnostic plots — both off by default, so compute lazily.
    need_full_rows = USE_FALLBACK or make_plots

    def scan_full_row(row):
        eps_all = np.empty(W)
        Z_tri_all = np.empty(W)
        Z_tof_all = np.empty(W)
        for col in range(W):
            Z_tri_all[col] = max(dotCal.triangulate_depth(float(col), float(row),
                                                          V[best_i], B, K_inv), 1e-9)
            Z_tof_all[col] = max(sample_tof_depth(float(col), float(row))[0], 1e-9)
            eps_all[col] = dotCal.consistency_error(Z_tof_all[col], Z_tri_all[col])
        return eps_all, Z_tri_all, Z_tof_all

    def quad_refine(u_best, row_signal):
        """Quadratic sub-pixel refinement of the selected peak column."""
        u_sub = float(u_best)
        if HALF_WIN <= u_best <= W - 1 - HALF_WIN:
            x_win = np.arange(-HALF_WIN, HALF_WIN + 1, dtype=float)
            y_win = row_signal[u_best - HALF_WIN:u_best + HALF_WIN + 1].astype(float)
            a, b, _ = np.polyfit(x_win, y_win, 2)
            if a < -1e-12:
                return u_best + float(np.clip(-b / (2.0 * a), -HALF_WIN, HALF_WIN))
        # 3-point fallback at image borders or non-concave wide fit
        if 0 < u_best < W - 1:
            y_m, y_0, y_p = row_signal[u_best - 1], row_signal[u_best], row_signal[u_best + 1]
            denom = 2.0 * (y_m - 2.0 * y_0 + y_p)
            if abs(denom) > 1e-12:
                return u_best + float(np.clip((y_m - y_p) / denom, -0.5, 0.5))
        return u_sub

    results_ab = []
    for k, (uu, vv) in enumerate(test_uv):
        best_i, _ = nearest_trail([uu, vv])

        row = int(epipolar_rows_dot[best_i])
        row_signal = sl_gray[row, :]
        peaks_on_row = row_peaks.get(row, np.array([], dtype=int))

        eps_peaks = np.full(len(peaks_on_row), np.inf)
        Z_tri_peaks = np.full(len(peaks_on_row), np.nan)
        Z_tof_peaks = np.full(len(peaks_on_row), np.nan)

        for p_idx, u_peak in enumerate(peaks_on_row):
            Z_tri = dotCal.triangulate_depth(float(u_peak), float(row), V[best_i], B, K_inv)
            Z_tof, _ = sample_tof_depth(float(u_peak), float(row))
            Z_tri_peaks[p_idx] = max(Z_tri, 1e-9)
            Z_tof_peaks[p_idx] = max(Z_tof, 1e-9)
            eps_peaks[p_idx] = dotCal.consistency_error(Z_tof_peaks[p_idx], Z_tri_peaks[p_idx])

        used_fallback = False
        if len(peaks_on_row) > 0:
            p_best = int(np.argmin(eps_peaks))
            u_best = int(peaks_on_row[p_best])
            eps_best = float(eps_peaks[p_best])
        else:
            u_best = int(round(uu))
            eps_best = 99999.0
            used_fallback = True

        eps_all_row = Z_tri_all_row = Z_tof_all_row = None
        if need_full_rows:
            eps_all_row, Z_tri_all_row, Z_tof_all_row = scan_full_row(row)
            if USE_FALLBACK:
                col_all_best = int(np.argmin(eps_all_row))
                if eps_all_row[col_all_best] < eps_best:
                    u_best = col_all_best
                    eps_best = float(eps_all_row[col_all_best])
                    used_fallback = True

        u_sub = quad_refine(u_best, row_signal)

        Z_tri_out = float(dotCal.triangulate_depth(u_sub, float(row), V[best_i], B, K_inv))
        Z_raw, _ = sample_tof_depth(uu, vv)

        rejected = False
        peak_snr = np.nan
        if PEAK_SNR_MIN is not None and not used_fallback and len(peaks_on_row) > 0:
            noise_floor = np.median(row_signal)
            peak_snr = (row_signal[u_best] - noise_floor) / max(noise_floor, 1e-12)
            if peak_snr < PEAK_SNR_MIN:
                rejected = True
        if EPS_THRESHOLD_AB is not None and eps_best > EPS_THRESHOLD_AB:
            rejected = True

        if rejected:
            Z_tri_out = float("nan")

        results_ab.append(dict(
            k=k, u=float(uu), v=float(vv),
            u_plot=float(u_sub), v_plot=float(row), row=row,
            Z_raw=Z_raw, Z_out=float(Z_tri_out),
            i_best=best_i, u_best=u_best, u_sub=u_sub,
            eps_best=eps_best, peak_snr=peak_snr,
            used_fallback=used_fallback,
            peaks_on_row=peaks_on_row.copy() if len(peaks_on_row) > 0 else np.array([], dtype=int),
            row_signal=row_signal,
            eps_peaks=eps_peaks,
            eps_all_row=eps_all_row,
            Z_tri_all_row=Z_tri_all_row,
            Z_tof_all_row=Z_tof_all_row,
        ))

    n_fallback = sum(1 for r in results_ab if r["used_fallback"])
    n_rejected = sum(1 for r in results_ab if np.isnan(r["Z_out"]))
    print(f"Approach 3: processed {len(results_ab)} dots  "
          f"({n_fallback} used fallback, {n_rejected} rejected by gates)")

    figures = {}
    if make_plots:
        import matplotlib.pyplot as plt

        eps_ab = np.array([r["eps_best"] for r in results_ab], float)
        cols = np.arange(W)

        for label, k_ex in [("best ε", int(np.argmin(eps_ab))),
                             ("worst ε", int(np.argmax(eps_ab)))]:
            ex = results_ab[k_ex]

            fig, axes = plt.subplots(1, 3, figsize=(16, 3.5))

            axes[0].plot(cols, ex["row_signal"], "-", linewidth=0.5, color="C0",
                         label="SL brightness (row)")
            if len(ex["peaks_on_row"]) > 0:
                axes[0].plot(ex["peaks_on_row"], ex["row_signal"][ex["peaks_on_row"]],
                             "rv", markersize=6, label=f"peaks ({len(ex['peaks_on_row'])})")
            axes[0].axvline(ex["u_best"], color="red", linestyle="--", linewidth=1,
                            label=f"u*={ex['u_best']}")
            if abs(ex["u_sub"] - ex["u_best"]) > 0.01:
                axes[0].axvline(ex["u_sub"], color="orange", linestyle=":", linewidth=1.2,
                                label=f"u_sub={ex['u_sub']:.2f}")
            axes[0].axvline(ex["u"], color="green", linestyle=":", linewidth=1,
                            alpha=0.6, label=f"detected u={ex['u']:.1f}")
            fb_str = " [FALLBACK]" if ex["used_fallback"] else ""
            axes[0].set_title(f"SL brightness — row {ex['row']}{fb_str}")
            axes[0].set_xlabel("Column u [px]"); axes[0].set_ylabel("Brightness")
            axes[0].legend(fontsize=6, loc="upper right")

            axes[1].semilogy(cols, ex["eps_all_row"], "-", linewidth=0.5, color="gray",
                             alpha=0.6, label="ε (all columns)")
            if len(ex["peaks_on_row"]) > 0:
                axes[1].semilogy(ex["peaks_on_row"], ex["eps_peaks"], "rv", markersize=6,
                                 label="ε at peaks")
            axes[1].axvline(ex["u_best"], color="red", linestyle="--", linewidth=1,
                            label=f"u*={ex['u_best']}")
            axes[1].set_title(f"Consistency error ε  ({label})")
            axes[1].set_xlabel("Column u [px]")
            axes[1].set_ylabel(r"$\varepsilon$ [1/m]")
            axes[1].legend(fontsize=6)

            axes[2].plot(cols, 1.0 / ex["Z_tof_all_row"], "-", linewidth=0.5, alpha=0.6,
                         label="1/Z_ToF")
            axes[2].plot(cols, 1.0 / ex["Z_tri_all_row"], "-", linewidth=0.5, alpha=0.6,
                         label="1/Z_tri")
            axes[2].axvline(ex["u_best"], color="red", linestyle="--", linewidth=1,
                            label=f"u*={ex['u_best']}")
            axes[2].set_title("1/Z comparison (disparity space)")
            axes[2].set_xlabel("Column u [px]"); axes[2].set_ylabel("1/Z [1/m]")
            axes[2].legend(fontsize=6)

            fb_tag = " [FALLBACK]" if ex["used_fallback"] else ""
            plt.suptitle(
                f"Approach 3 – Active Brightness (epipolar scan) – dot k={ex['k']}  "
                f"(i*={ex['i_best']}, row={ex['row']}, ε*={ex['eps_best']:.3e}){fb_tag}", y=1.02
            )
            plt.tight_layout()
            figures[f"approach3_trail_{label.replace(' ', '_')}"] = fig

        figures["approach3_depth_comparison"] = _depth_comparison_fig(
            results_ab, "Z_tri (Approach 3)",
            "Approach 3 – Active Brightness (epipolar scan): raw ToF vs triangulation depth")

        # Depth overlay (custom: marks fallback dots)
        fig_do, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(sl_gray, cmap="gray")
        xs = np.array([r["u_plot"] for r in results_ab])
        ys = np.array([r["v_plot"] for r in results_ab])
        zs = np.array([r["Z_out"] for r in results_ab])
        fb = np.array([r["used_fallback"] for r in results_ab])

        m_peak = ~fb & np.isfinite(zs)
        sc = ax.scatter(xs[m_peak], ys[m_peak], s=40, c=zs[m_peak], cmap="plasma",
                        alpha=0.9, edgecolors="none", label="peak-based")
        m_fb = fb & np.isfinite(zs)
        if np.any(m_fb):
            ax.scatter(xs[m_fb], ys[m_fb], s=50, c=zs[m_fb], cmap="plasma",
                       alpha=0.9, edgecolors="orange", linewidths=1.5, label="fallback (min-ε)")
        if np.any(np.isfinite(zs)):
            plt.colorbar(sc, ax=ax, label="Z_tri [m]")
        for r in results_ab:
            if np.isfinite(r["Z_out"]):
                ax.text(r["u_plot"] + 2, r["v_plot"] + 2, f'{r["Z_out"]:.2f}', fontsize=5, color="yellow")
                ax.text(r["u_plot"] + 2, r["v_plot"] + 12, f'i={r["i_best"]}', fontsize=5, color="cyan")
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
        ax.set_title("Approach 3 – Active Brightness (epipolar scan): depth overlay")
        ax.axis("off")
        figures["approach3_depth_overlay"] = fig_do

        figures["approach3_eps_histogram"] = _eps_hist_fig(
            eps_ab, EPS_THRESHOLD_AB,
            r"Approach 3 – Distribution of best $\varepsilon$ (epipolar scan)")

    z_tri_arr = np.array([r["Z_out"] for r in results_ab])
    maybe_write_error_distribution(setup, z_tri_arr, 3)
    return {"results": results_ab, "figures": figures, "depths": z_tri_arr}


# ═══════════════════════════════════════════════════════════════════════════
# Approach 4: Consistency-Error-Guided Triangulation
# ═══════════════════════════════════════════════════════════════════════════

def forward_projection_error(dotCal, cal, make_plots=False):
    """Part A of Approach 4: triangulation error budget from the calibration.

    Forward-projects each calibrated dot ray onto the calibration planes,
    compares with the detected positions (→ localisation error δx), and
    validates the empirical triangulation depth error against the theoretical
    bound δz = δx·z²/(B·h).
    """
    V, B, K_inv = cal["V"], cal["B"], cal["K_inv"]
    K = cal["K"]
    AB = cal["AB"]
    subpixel_list = cal["subpixel_list"]
    cal_dists = cal["cal_dists"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    pixel_pitch = 10e-6
    h_focal = fx * pixel_pitch

    n_cal = len(cal_dists)
    cal_z = np.array(cal_dists)
    n_dots_cal = V.shape[0]

    # Per-distance lookup by dot ID (list position ≠ ID once dots are missing).
    sp_by_id = [{int(d["id"]): (d["x"], d["y"]) for d in spx} for spx in subpixel_list]

    fwd_residuals_all = []
    fwd_res_by_dist = [[] for _ in range(n_cal)]
    dz_by_dist = [[] for _ in range(n_cal)]

    for i in range(n_dots_cal):
        if not np.all(np.isfinite(V[i])) or abs(V[i, 2]) < 1e-12:
            continue

        for j in range(n_cal):
            det = sp_by_id[j].get(i)
            if det is None:
                continue
            u_det, v_det = det
            if not (np.isfinite(u_det) and np.isfinite(v_det)):
                continue

            # Forward project: find where the calibrated ray hits z = cal_z[j]
            t = cal_z[j] / V[i, 2]
            P = B + t * V[i]
            u_pred = fx * P[0] / P[2] + cx
            v_pred = fy * P[1] / P[2] + cy

            du, dv = u_det - u_pred, v_det - v_pred
            fwd_residuals_all.append((du, dv))
            fwd_res_by_dist[j].append(np.hypot(du, dv))

            # Empirical triangulation depth error at the detected position
            Z_tri = dotCal.triangulate_depth(u_det, v_det, V[i], B, K_inv)
            dz_by_dist[j].append(Z_tri - cal_z[j])

    fwd_residuals_all = np.array(fwd_residuals_all)
    dx_pixels = np.sqrt(np.mean(fwd_residuals_all[:, 0] ** 2 + fwd_residuals_all[:, 1] ** 2))
    dx_physical = dx_pixels * pixel_pitch

    print(f"Forward-projection localisation error (RMS):")
    print(f"  dx = {dx_pixels:.4f} px  =  {dx_physical * 1e6:.2f} µm")
    print(f"  Physical focal length h = {h_focal * 1e3:.3f} mm")
    print(f"  Baseline AB = {AB * 100:.3f} cm")
    print(f"  Total residual samples: {len(fwd_residuals_all)}")

    print(f"\nForward-projection residuals by distance:")
    print(f"{'Dist [m]':>10s} {'RMS [px]':>10s} {'N':>5s}")
    for j in range(n_cal):
        if fwd_res_by_dist[j]:
            rms_j = np.sqrt(np.mean(np.array(fwd_res_by_dist[j]) ** 2))
            print(f"{cal_z[j]:10.1f} {rms_j:10.4f} {len(fwd_res_by_dist[j]):5d}")

    dz_means = np.array([np.mean(errs) if errs else np.nan for errs in dz_by_dist])
    dz_stds = np.array([np.std(errs) if errs else np.nan for errs in dz_by_dist])
    dz_rmss = np.array([np.sqrt(np.mean(np.array(errs) ** 2)) if errs else np.nan
                         for errs in dz_by_dist])

    print(f"\nEmpirical triangulation depth error at calibration distances:")
    print(f"{'Dist [m]':>10s} {'Mean [mm]':>10s} {'Std [mm]':>10s} {'RMS [mm]':>10s} {'N':>5s}")
    for j in range(n_cal):
        print(f"{cal_z[j]:10.1f} {dz_means[j] * 1e3:10.2f} {dz_stds[j] * 1e3:10.2f} "
              f"{dz_rmss[j] * 1e3:10.2f} {len(dz_by_dist[j]):5d}")

    ref_distances = np.array([0.5, 1.0, 2.0, 3.5])
    dz_theory_ref = dx_physical * ref_distances ** 2 / (AB * h_focal)
    print(f"\nTheoretical triangulation depth error  dz = dx·z²/(AB·h):")
    for z_ref, dz_ref in zip(ref_distances, dz_theory_ref):
        print(f"  z = {z_ref:.1f} m  →  dz = {dz_ref * 100:.2f} cm  ({dz_ref * 1000:.1f} mm)")

    figures = {}
    if make_plots:
        import matplotlib.pyplot as plt

        euc_res = np.hypot(fwd_residuals_all[:, 0], fwd_residuals_all[:, 1])

        fig_res, axes = plt.subplots(1, 3, figsize=(14, 3.5))
        axes[0].scatter(fwd_residuals_all[:, 0], fwd_residuals_all[:, 1],
                        s=3, alpha=0.3, edgecolors="none")
        axes[0].axhline(0, color="k", linewidth=0.5); axes[0].axvline(0, color="k", linewidth=0.5)
        axes[0].set_aspect("equal")
        axes[0].set_xlabel("Δu [px]"); axes[0].set_ylabel("Δv [px]")
        axes[0].set_title("Forward-projection residuals (u, v)")
        axes[0].grid(alpha=0.3)

        axes[1].hist(euc_res, bins=30, edgecolor="k", linewidth=0.4, color="C0")
        axes[1].axvline(dx_pixels, color="red", linestyle="--",
                        label=f"RMS = {dx_pixels:.4f} px")
        axes[1].set_title("Euclidean residual distribution")
        axes[1].set_xlabel("Residual [px]"); axes[1].set_ylabel("Count")
        axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0); axes[1].grid(alpha=0.3)

        for j in range(n_cal):
            if fwd_res_by_dist[j]:
                axes[2].scatter([cal_z[j]] * len(fwd_res_by_dist[j]),
                                fwd_res_by_dist[j], s=5, alpha=0.3, color="C0")
        axes[2].set_title("Residuals vs calibration distance")
        axes[2].set_xlabel("Distance [m]"); axes[2].set_ylabel("Residual [px]")
        axes[2].grid(alpha=0.3)

        plt.suptitle(f"Forward-projection residuals — dx = {dx_pixels:.4f} px "
                     f"({dx_physical * 1e6:.2f} µm)", y=1.02)
        plt.tight_layout()
        figures["approach4_fwd_residuals"] = fig_res

        # Empirical depth error box plot + theoretical curve
        z_smooth = np.linspace(0.2, 4.5, 200)
        dz_theory_curve = dx_physical * z_smooth ** 2 / (AB * h_focal) * 1e3

        fig_eb, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot([np.array(errs) * 1e3 for errs in dz_by_dist],
                   positions=cal_z, widths=0.06,
                   patch_artist=True, manage_ticks=False,
                   boxprops=dict(facecolor="C0", alpha=0.4),
                   medianprops=dict(color="C3", linewidth=1.5))
        ax.plot(z_smooth, dz_theory_curve, "C3", linewidth=1.5,
                label=r"$+\delta z = \delta x \cdot z^2 / (B \cdot h)$")
        ax.plot(z_smooth, -dz_theory_curve, "C3", linewidth=1.5, linestyle="--",
                label=r"$-\delta z$")
        ax.fill_between(z_smooth, -dz_theory_curve, dz_theory_curve, alpha=0.08, color="C3")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Calibration distance [m]")
        ax.set_ylabel("Depth error  ΔZ = Z_tri − Z_gt  [mm]")
        ax.set_title(f"Empirical triangulation depth error vs theoretical bounds  "
                     f"(dx = {dx_pixels:.4f} px)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0); ax.grid(alpha=0.3)
        plt.tight_layout()
        figures["approach4_error_budget_calibration"] = fig_eb

        # Error budget plot (mean ± std)
        fig_eb2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(z_smooth, dz_theory_curve, "C3", linewidth=1.5,
                label=r"$+\delta z = \delta x \cdot z^2 / (B \cdot h)$")
        ax.plot(z_smooth, -dz_theory_curve, "C3", linewidth=1.5, linestyle="--",
                label=r"$-\delta z$")
        ax.fill_between(z_smooth, -dz_theory_curve, dz_theory_curve, alpha=0.08, color="C3")
        ax.errorbar(cal_z, dz_means * 1e3, yerr=dz_stds * 1e3,
                    fmt="s", markersize=6, color="C0", capsize=4, capthick=1.2,
                    label="Empirical (calibration): mean ± std")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Depth z [m]")
        ax.set_ylabel("Depth error ΔZ [mm]")
        ax.set_title(f"Approach 4 – Triangulation error budget  "
                     f"(dx = {dx_pixels:.4f} px = {dx_physical * 1e6:.1f} µm)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0); ax.grid(alpha=0.3)
        plt.tight_layout()
        figures["approach4_error_budget"] = fig_eb2

    return figures


def run_approach_4(dotCal, cal, setup, make_plots=False):
    """Same pipeline as Approach 1, but the output depth is the TRIANGULATED
    depth at the detected dot position (MPI-free); ε only identifies the
    correct trail sample / gates outliers."""
    V, B, K_inv = cal["V"], cal["B"], cal["K_inv"]

    EPS_THRESHOLD_4 = None  # set a float to invalidate high-ε dots

    # ── Part A: forward-projection residuals / error budget ─────────────
    figures = forward_projection_error(dotCal, cal, make_plots)

    # ── Part B: consistency-error-guided triangulation ──────────────────
    results_4 = consistency_scan(dotCal, cal, setup)
    for r in results_4:
        r["Z_out"] = float(dotCal.triangulate_depth(r["u"], r["v"], V[r["i_best"]], B, K_inv))
        if EPS_THRESHOLD_4 is not None and r["eps_best"] > EPS_THRESHOLD_4:
            r["Z_out"] = float("nan")

    print(f"\nApproach 4: processed {len(results_4)} dots")
    for r in results_4:
        print(f"  dot {r['k']:3d}  i*={r['i_best']:3d}  j*={r['j_best']:2d}  "
              f"Z_raw={r['Z_raw']:.4f}  Z_out={r['Z_out']:.4f}  ε*={r['eps_best']:.3e}")

    if make_plots:
        eps_all_4 = np.array([r["eps_best"] for r in results_4], float)

        for label, k_ex in [("best", int(np.argmin(eps_all_4))),
                             ("worst", int(np.argmax(eps_all_4)))]:
            ex = results_4[k_ex]
            figures[f"approach4_eps_curve_{label}"] = _eps_curve_fig(
                ex, setup["n_dist"],
                f"Approach 4 – Consistency-Error-Guided Triangulation – dot k={ex['k']}  "
                f"(i*={ex['i_best']}, ε*={ex['eps_best']:.3e})  ({label} ε)")

        figures["approach4_depth_comparison"] = _depth_comparison_fig(
            results_4, "Z_tri (Approach 4 @ detected subpixel position)",
            "Approach 4 – raw ToF vs triangulated depth")

        figures["approach4_depth_overlay"] = _depth_overlay_fig(
            setup["sl_gray"],
            np.array([r["u"] for r in results_4]),
            np.array([r["v"] for r in results_4]),
            np.array([r["Z_out"] for r in results_4]),
            [r["i_best"] for r in results_4],
            "Z_tri [m]", "Approach 4 – Consistency-Error-Guided Triangulation: depth overlay")

        figures["approach4_eps_histogram"] = _eps_hist_fig(
            eps_all_4, None,
            r"Approach 4 – Distribution of best $\varepsilon = |1/Z_{ToF} - 1/Z_{tri}|$")

    z_out_arr_4 = np.array([r["Z_out"] for r in results_4])
    maybe_write_error_distribution(setup, z_out_arr_4, 4)
    return {"results": results_4, "figures": figures, "depths": z_out_arr_4}


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

RUNNERS = {
    1: run_approach_1,
    2: run_approach_2,
    3: run_approach_3,
    4: run_approach_4,
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
                   help="Approaches to run: 1 2 3 4 or 'all'")
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
        selected = [1, 2, 3, 4]
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

#!/usr/bin/env python3
"""Headless dot calibration (FL_C §4.1) — paper-faithful pipeline.

Runs the full calibration on a flat-wall series and writes the extended
calibration.json (incl. X/Y mode, full 3-D baseline, precomputed 1-D dot
trails, sigma_u and the sigma_ToF(z) table) used by approaches.py.

Usage:
    python calibrate.py --camera OnSemi --name OnSemi_paperlike [--plots]
    python calibrate.py --camera Schmersal --name Schmersal_paperlike [--plots]

The interactive notebook performs the same steps cell by cell.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from dot_calibration import DotCalibration

ROOT = Path(__file__).resolve().parent.parent

# cal_blob threshold applies to the [0,1]-normalised image (see
# DotCalibration._normalize_for_log), so one value fits PNG and EXR alike.
CAMERAS = {
    "OnSemi": dict(
        sl_cal=ROOT / "Simulation_Pictures/PBRT/SL_ToF_On/SL_OnSemi",
        tof_cal=ROOT / "Simulation_Pictures/PBRT/SL_ToF_On/ToF_OnSemi",
        K=np.array([[1006.2, 0.0, 640.0], [0.0, -1006.2, 480.0], [0.0, 0.0, 1.0]]),
        cal_blob=dict(max_sigma=18, num_sigma=10, min_sigma=12, threshold=0.05),
    ),
    "Schmersal": dict(
        sl_cal=ROOT / "Simulation_Pictures/PBRT/SL_ToF_Schm/SL_Schmersal",
        tof_cal=ROOT / "Simulation_Pictures/PBRT/SL_ToF_Schm/ToF_Schmersal",
        K=np.array([[503.1, 0.0, 320.0], [0.0, -503.1, 240.0], [0.0, 0.0, 1.0]]),
        cal_blob=dict(max_sigma=14, num_sigma=10, min_sigma=8, threshold=0.05),
    ),
    # Real-sensor recordings (EXR SL images, binary PCDs). Farthest distances
    # lose dots (dim/small) — the FOV-aware tracking + min-track invalidation
    # handle the partial detections.
    "SchmersalReal": dict(
        sl_cal=ROOT / "Pictures/Calibration/SchmersalReal/19.06/SL",
        tof_cal=ROOT / "Pictures/Calibration/SchmersalReal/19.06/ToF",
        K=np.array([[503.1, 0.0, 320.0], [0.0, -503.1, 240.0], [0.0, 0.0, 1.0]]),
        cal_blob=dict(max_sigma=10, num_sigma=10, min_sigma=10, threshold=0.05),
    ),    
}


def sanitize(obj):
    """Recursively replace NaN/Inf with None for JSON compatibility."""
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    if isinstance(obj, np.floating):
        return sanitize(float(obj))
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    return obj


def find_sl_image(sl_dir: Path, tof_name: str):
    """Matching SL image for a ToF pcd name (ToF_0.4m.pcd → SL_0.4m.{exr,png}).

    Falls back to matching by parsed distance — the real recordings mix zero
    paddings (ToF_0.680m.pcd vs SL_0.68m.png)."""
    stem = tof_name.replace("ToF_", "SL_").rsplit(".", 1)[0]
    for ext in (".exr", ".png"):
        p = sl_dir / (stem + ext)
        if p.exists():
            return p
    dist = DotCalibration.parse_distance_from_name(tof_name)
    if dist is None:
        return None
    for p in sorted(sl_dir.glob("SL_*")):
        d2 = DotCalibration.parse_distance_from_name(p.name)
        if d2 is not None and abs(d2 - dist) < 1e-6 and p.suffix.lower() in (".exr", ".png"):
            return p
    return None


def estimate_sigma_u_ab(dotCal, subpixel_list, trails, sl_paths, z_min):
    """Along-trail localisation noise [px] of the RUNTIME active-brightness
    estimator, measured on the calibration images themselves: for every
    calibration detection, run the 1-D trail pipeline (brightness resampling →
    smoothing → peak → plateau-centred quadratic) and compare the refined
    trail coordinate against the 2-D subpixel detection projected onto the
    trail. This is the σ_u that belongs in σ_SL for Approach 2 — the 2-D GPR
    calibration residual understates what the 1-D runtime estimator achieves.
    """
    import approaches
    import scipy.ndimage
    import scipy.signal

    residuals = []
    for j, sl_p in enumerate(sl_paths):
        img = dotCal.read_image(str(sl_p))
        if np.issubdtype(img.dtype, np.integer):
            sat_level = float(np.iinfo(img.dtype).max)
        else:
            sat_level = None
        by_id = {int(d["id"]): (float(d["x"]), float(d["y"])) for d in subpixel_list[j]}
        for i, trail in enumerate(trails):
            if trail is None or i not in by_id:
                continue
            n = len(trail["u"])
            if n < 5:
                continue
            # fractional trail coordinate of the 2-D detection
            p0 = np.array([trail["u"][0], trail["v"][0]])
            p1 = np.array([trail["u"][-1], trail["v"][-1]])
            g = p1 - p0
            gg = float(g @ g)
            if gg < 1e-9:
                continue
            k_det = float((np.array(by_id[i]) - p0) @ g / gg) * (n - 1)
            if not (0 <= k_det <= n - 1):
                continue

            ab = approaches.sample_trail_brightness(img, trail)
            ab = approaches.saturation_correct(ab, sat_level)
            sig = np.where(np.isfinite(ab), ab, 0.0)
            if sig.max() <= 0:
                continue
            sig_s = scipy.ndimage.gaussian_filter1d(sig, approaches.AB_SMOOTH_SIGMA)
            peaks, _ = scipy.signal.find_peaks(
                sig_s, prominence=approaches.PEAK_PROMINENCE_FRAC * float(sig_s.max()),
                distance=approaches.PEAK_MIN_DISTANCE)
            if len(peaks) == 0:
                continue
            k0 = int(peaks[int(np.argmin(np.abs(peaks - k_det)))])
            if abs(k0 - k_det) > 20:
                continue
            k_sub = approaches.quad_subpixel(sig_s, k0)
            residuals.append(k_sub - k_det)

    if len(residuals) < 10:
        return float("nan")
    r = np.asarray(residuals)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def run_calibration(camera, name, subpixel_mode="GPR", track_tx_space=False,
                    z_min=0.3, outlier_sigma=3.0, min_points=3,
                    mode_override=None, save_plots=False):
    cfg = CAMERAS[camera]
    K = cfg["K"]
    K_inv = np.linalg.inv(K)
    dotCal = DotCalibration()

    tof_paths = sorted(cfg["tof_cal"].glob("ToF_*m.pcd"),
                       key=lambda p: dotCal.parse_distance_from_name(p.name) or 999.0)
    tof_paths = [str(p) for p in tof_paths]
    cal_dists = [dotCal.parse_distance_from_name(p) for p in tof_paths]
    print(f"Calibration distances ({len(cal_dists)}): {cal_dists}")

    # ── Steps 1+2: LoG detection + subpixel localisation ────────────────
    subpixel_list_unordered = []
    image = None
    for tof_p in tof_paths:
        sl_p = find_sl_image(cfg["sl_cal"], Path(tof_p).name)
        if sl_p is None:
            sys.exit(f"No SL image found for {tof_p}")
        blobs, image = dotCal.detect_blobs(str(sl_p), visualize=False, **cfg["cal_blob"])
        _, subpixels = dotCal.detect_subpixel_locations(blobs, image, mode=subpixel_mode)
        subpixel_list_unordered.append(subpixels)
        print(f"  {Path(sl_p).name}: {len(subpixels)} dots ({subpixel_mode})")
    image_shape = image.shape[:2]

    # ── Step 2b: cross-distance tracking (handles dots entering/leaving) ─
    subpixel_list = dotCal.track_dots_across_distances(
        subpixel_list_unordered, ref_idx=0, threshold_factor=0.5)

    # ── Step 2c: X/Y travel mode + signed baseline guess from parallax ──
    mode_info = dotCal.detect_travel_mode(subpixel_list, cal_dists, K,
                                          mode_override=mode_override)
    mode = mode_info["mode"]
    B_guess_vec = mode_info["B_guess_vec"]
    print(f"Travel mode: {mode}  (du/dw={mode_info['du_dw']:.2f} px·m, "
          f"dv/dw={mode_info['dv_dw']:.2f} px·m)")
    print(f"Signed baseline estimate along {mode}: {mode_info['B_axis_est'] * 100:.3f} cm")

    # Optional paper-style transmitter-space re-tracking with the derived guess.
    if track_tx_space:
        coords_list = dotCal.tx_space_coords(
            subpixel_list_unordered, tof_paths, K=K, baseline_guess=B_guess_vec)
        subpixel_list = dotCal.track_dots_across_distances(
            subpixel_list_unordered, ref_idx=0, threshold_factor=0.5,
            coords_list=coords_list, travel_mode=mode)

    n_dots = max(int(d["id"]) for spx in subpixel_list for d in spx) + 1
    print(f"Tracked roster: {n_dots} dots, per distance: {[len(s) for s in subpixel_list]}")

    # ── Step 3: back-projection ─────────────────────────────────────────
    U, U_tx = dotCal.backproject_calibration_dots(
        subpixel_list, tof_paths, K=K,
        pcd_depth_mode="radial", baseline_guess=B_guess_vec)

    # ── Step 4: baseline (full 3-D, with outlier removal) ───────────────
    intersection, n_lines = dotCal.estimate_baseline(
        U_tx, outlier_sigma=outlier_sigma, min_points=min_points)
    B = B_guess_vec + intersection
    axis = 0 if mode == "x" else 1
    AB = float(B[axis])
    print(f"Baseline from {n_lines} dot lines: B = {B}  (AB along {mode}: {AB * 100:.4f} cm)")

    # ── Step 5: unit vectors ────────────────────────────────────────────
    V = dotCal.estimate_unit_vectors(U, B)
    n_valid_v = int(np.sum(np.all(np.isfinite(V), axis=1)))
    print(f"Unit vectors: {n_valid_v} / {V.shape[0]} valid")

    # ── 1-D dot trails + calibration noise estimates ────────────────────
    # Trails are fitted to the calibration DETECTIONS (affine in w = 1/z),
    # absorbing real lens distortion; the analytic V/B/K trail is the
    # fallback for dots with too few detections.
    trails = dotCal.fit_trails_from_detections(
        subpixel_list, U, z_min=z_min, step_px=1.0, image_shape=image_shape,
        V=V, B=B, K=K)
    n_trails = sum(1 for t in trails if t is not None)
    lens = [len(t["u"]) for t in trails if t is not None]
    print(f"Trails: {n_trails} dots, {int(np.mean(lens))} samples on average "
          f"(z: {z_min} m → ∞, ~1 px steps)")

    sigma_u = dotCal.estimate_sigma_u_trails(subpixel_list, trails)
    sigma_tof_table = dotCal.estimate_sigma_tof(U, expected_z=cal_dists)
    sl_paths = [find_sl_image(cfg["sl_cal"], Path(p).name) for p in tof_paths]
    sigma_u_ab = estimate_sigma_u_ab(dotCal, subpixel_list, trails, sl_paths, z_min)
    print(f"sigma_u (2-D calibration residual) = {sigma_u:.4f} px")
    print(f"sigma_u_ab (1-D runtime AB estimator) = {sigma_u_ab:.4f} px")
    print("sigma_ToF(z) table:")
    for z, s in sigma_tof_table:
        print(f"  z = {z:.3f} m  →  σ = {s * 1e3:.2f} mm")

    # ── Save ────────────────────────────────────────────────────────────
    cal_dir = ROOT / "Calibrations" / name
    cal_dir.mkdir(parents=True, exist_ok=True)
    cal_data = sanitize({
        "metadata": {
            "created": datetime.now().isoformat(),
            "camera": camera,
            "sl_cal_path": str(cfg["sl_cal"]),
            "tof_cal_path": str(cfg["tof_cal"]),
            "pipeline": "codePaperlike",
            "mode": mode,
            "baseline_guess": B_guess_vec.tolist(),
            "subpixel_mode": subpixel_mode,
            "pcd_depth_mode": "radial",
            "pcd_unit_scale": 0.001,
            "track_in_tx_space": track_tx_space,
            "baseline_outlier_sigma": outlier_sigma,
            "baseline_min_points": min_points,
            "trail_z_min": z_min,
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "mode": mode,
        "K": K.tolist(),
        "K_inv": K_inv.tolist(),
        "V": V.tolist(),
        "B": B.tolist(),
        "AB": AB,
        "BASELINE_GUESS": B_guess_vec.tolist(),
        "cal_dists": [float(d) for d in cal_dists],
        "subpixel_list": subpixel_list,
        "U": U.tolist(),
        "trails": [None if t is None else
                   {"u": t["u"].tolist(), "v": t["v"].tolist(), "w": t["w"].tolist()}
                   for t in trails],
        "sigma_u": float(sigma_u),
        "sigma_u_ab": float(sigma_u_ab),
        "sigma_tof_table": sigma_tof_table.tolist(),
    })
    cal_file = cal_dir / "calibration.json"
    with open(cal_file, "w") as f:
        json.dump(cal_data, f)
    print(f"Calibration saved to {cal_file}")

    if save_plots:
        import matplotlib
        matplotlib.use("Agg")
        fig = dotCal.plot_dot_trails(trails, image_shape=image_shape, background=image)
        fig.savefig(str(cal_dir / "dot_trails.png"), dpi=150, bbox_inches="tight")
        print(f"Trail plot saved to {cal_dir / 'dot_trails.png'}")

    return cal_file


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True, choices=sorted(CAMERAS))
    p.add_argument("--name", required=True, help="Calibrations/<name>/calibration.json")
    p.add_argument("--subpixel-mode", default="GPR",
                   choices=["GPR", "geometricCenter", "radial", "center"])
    p.add_argument("--track-tx-space", action="store_true",
                   help="Match dots in transmitter space (FL_C §4.1)")
    p.add_argument("--mode", default=None, choices=["x", "y"],
                   help="Force travel mode instead of auto-detection")
    p.add_argument("--z-min", type=float, default=0.3, help="Near trail limit [m]")
    p.add_argument("--plots", action="store_true", help="Save the trail plot")
    args = p.parse_args()

    run_calibration(args.camera, args.name, subpixel_mode=args.subpixel_mode,
                    track_tx_space=args.track_tx_space, mode_override=args.mode,
                    z_min=args.z_min, save_plots=args.plots)


if __name__ == "__main__":
    main()

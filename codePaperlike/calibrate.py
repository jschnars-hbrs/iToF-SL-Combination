#!/usr/bin/env python3
"""Dot calibration on a flat-wall series (Godbaz §4.1) → Calibrations/<name>/calibration.json.

Usage:
    python calibrate.py --camera SchmersalRealCoherent_v11 --name my_cal [--sor 6 0.4] [--lpc] [--plots]
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import approaches
from dot_calibration import DotCalibration, cache_tag

ROOT = Path(__file__).resolve().parent.parent
CAL_ROOT = ROOT / "Calibrations"

K_SCHMERSAL_REAL = np.array([[500.36, 0.0, 322.84],
                             [0.0, 505.62, 242.18],
                             [0.0, 0.0, 1.0]])

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
    "SchmersalReal": dict(
        sl_cal=ROOT / "Pictures/Calibration/SchmersalReal/25.08/SL",
        tof_cal=ROOT / "Pictures/Calibration/SchmersalReal/25.08/ToF",
        K=K_SCHMERSAL_REAL,
        cal_blob=dict(max_sigma=10, num_sigma=10, min_sigma=10, threshold=0.05),
        pcd_glob="Pos*.pcd",
        exclude=("Pos9",),
        dist_offset=0,
    ),
    "SchmersalRealCoherent": dict(
        sl_cal=ROOT / "Pictures/Calibration/SchmersalReal/30.09/SL",
        tof_cal=ROOT / "Pictures/Calibration/SchmersalReal/30.09/ToF",
        K=K_SCHMERSAL_REAL,
        cal_blob=dict(max_sigma=5, num_sigma=10, min_sigma=3, threshold=0.03),
        pcd_glob="Pos*.pcd",
        exclude=("Pos8", "Pos9"),  # phase unwrapping fails on these frames
        dist_offset=0,
    ),
    "SchmersalReal_v11": dict(
        sl_cal=ROOT / "Pictures/Calibration/SchmersalReal/25.08_v11/SL",
        tof_cal=ROOT / "Pictures/Calibration/SchmersalReal/25.08_v11/ToF",
        K=K_SCHMERSAL_REAL,
        cal_blob=dict(max_sigma=10, num_sigma=10, min_sigma=10, threshold=0.05),
        pcd_glob="ToF_*m.pcd",
        dist_offset=0,
    ),
    "SchmersalRealCoherent_v11": dict(
        sl_cal=ROOT / "Pictures/Calibration/SchmersalReal/30.09_v11/SL",
        tof_cal=ROOT / "Pictures/Calibration/SchmersalReal/30.09_v11/ToF",
        K=K_SCHMERSAL_REAL,
        cal_blob=dict(max_sigma=5, num_sigma=10, min_sigma=3, threshold=0.03),
        pcd_glob="ToF_*m.pcd",
        dist_offset=0,
    ),
}


def distance_for(cfg, dotCal, path) -> float:
    """Calibration distance [m] of one frame, with the camera's logbook offset applied."""
    return dotCal.parse_distance_from_name(str(path), offset=cfg.get("dist_offset", 0.0))


def sanitize(obj):
    """Recursively convert numpy types to JSON types and NaN/Inf to None."""
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
    """SL image matching a ToF file name (ToF_0.4m.pcd → SL_0.4m.{exr,png}), else by parsed distance."""
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
    """Robust along-trail residual [px] of the runtime AB estimator against the calibration detections."""
    residuals = []
    for j, sl_p in enumerate(sl_paths):
        img = dotCal.read_image(str(sl_p))
        sat_level = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else None
        by_id = {int(d["id"]): (float(d["x"]), float(d["y"])) for d in subpixel_list[j]}
        for i, trail in enumerate(trails):
            if trail is None or i not in by_id or len(trail["u"]) < 5:
                continue
            n = len(trail["u"])
            p0 = np.array([trail["u"][0], trail["v"][0]])
            g = np.array([trail["u"][-1], trail["v"][-1]]) - p0
            gg = float(g @ g)
            if gg < 1e-9:
                continue
            k_det = float((np.array(by_id[i]) - p0) @ g / gg) * (n - 1)
            if not 0 <= k_det <= n - 1:
                continue
            ab = approaches.saturation_correct(approaches.sample_trail_brightness(img, trail), sat_level)
            sig = np.where(np.isfinite(ab), ab, 0.0)
            if sig.max() <= 0:
                continue
            sig_s, base, peaks = approaches.trail_peaks(sig)
            if len(peaks) == 0:
                continue
            k0 = int(peaks[int(np.argmin(np.abs(peaks - k_det)))])
            if abs(k0 - k_det) > 20:
                continue
            k_sub, _ = approaches.refine_on_trail(dotCal, img, trail,
                                                  approaches.quad_subpixel(sig_s, k0, baseline=base), k_anchor=k0)
            residuals.append(k_sub - k_det)
    if len(residuals) < 10:
        return float("nan")
    r = np.asarray(residuals)
    return float(1.4826 * np.median(np.abs(r - np.median(r))))


def calibration_frames(cfg, dotCal):
    """(tof_paths, cal_dists, sl_paths) of a camera's calibration series, sorted by distance."""
    excl = set(cfg.get("exclude", ()))
    paths = sorted((p for p in cfg["tof_cal"].glob(cfg.get("pcd_glob", "ToF_*m.pcd")) if p.stem not in excl),
                   key=lambda p: distance_for(cfg, dotCal, p) or 999.0)
    if not paths:
        raise FileNotFoundError(f"No calibration PCDs matched in {cfg['tof_cal']}")
    sl_paths = [find_sl_image(cfg["sl_cal"], p.name) for p in paths]
    missing = [p.name for p, s in zip(paths, sl_paths) if s is None]
    if missing:
        raise FileNotFoundError(f"No SL image for {missing}")
    return [str(p) for p in paths], [distance_for(cfg, dotCal, p) for p in paths], sl_paths


def calibrate(camera, dotCal, subpixel_mode="GPR", track_tx_space=False, outlier_sigma=3.0,
              min_points=3, z_min=0.3, light_path_correction=False, lpc_max_iter=5, lpc_tol=1e-4,
              mode_override=None, show_blobs=False):
    """Full calibration (detection, tracking, baseline, unit vectors, trails, noise); returns the cal dict."""
    cfg = CAMERAS[camera]
    K = cfg["K"]
    tof_sample = cfg.get("tof_sample")
    tof_paths, cal_dists, sl_paths = calibration_frames(cfg, dotCal)
    print(f"Calibration distances: {cal_dists}")

    unordered = []
    for sl_p in sl_paths:
        blobs, image = dotCal.detect_blobs(str(sl_p), visualize=show_blobs, **cfg["cal_blob"])
        unordered.append(dotCal.detect_subpixel_locations(blobs, image, mode=subpixel_mode)[1])
    print(f"Dots per distance: {[len(s) for s in unordered]}")
    subpixel_list = dotCal.track_dots_across_distances(unordered, ref_idx=0, threshold_factor=0.5)

    mode_info = dotCal.detect_travel_mode(subpixel_list, cal_dists, K, mode_override=mode_override)
    mode, B_guess = mode_info["mode"], mode_info["B_guess_vec"]
    print(f"Travel mode: {mode}  (du/dw = {mode_info['du_dw']:.2f}, dv/dw = {mode_info['dv_dw']:.2f} px·m)")
    print(f"B_guess = {np.round(B_guess * 1e3, 2)} mm  (from {mode_info['n_fit']} dots)")
    if track_tx_space:
        coords = dotCal.tx_space_coords(unordered, tof_paths, K=K, baseline_guess=B_guess)
        subpixel_list = dotCal.track_dots_across_distances(unordered, ref_idx=0, threshold_factor=0.5,
                                                           coords_list=coords, travel_mode=mode)
    n_dots = max(int(d["id"]) for spx in subpixel_list for d in spx) + 1
    print(f"Tracked roster: {n_dots} dots, per distance: {[len(s) for s in subpixel_list]}")

    def baseline_pass():
        """Back-project the dots and fit the baseline with the current light origin."""
        U, U_tx = dotCal.backproject_calibration_dots(subpixel_list, tof_paths, K=K, pcd_depth_mode="radial",
                                                      baseline_guess=B_guess, tof_sample=tof_sample)
        intersection, n_lines = dotCal.estimate_baseline(U_tx, outlier_sigma=outlier_sigma, min_points=min_points)
        return U, U_tx, intersection, n_lines, B_guess + intersection

    dotCal.light_origin = None
    U, U_tx, intersection, n_lines, B = baseline_pass()
    if light_path_correction:
        U_raw = U
        for it in range(1, lpc_max_iter + 1):
            dotCal.light_origin = B.copy()
            U, U_tx, intersection, n_lines, B_new = baseline_pass()
            dB = float(np.linalg.norm(B_new - B))
            print(f"  LPC iteration {it}: B = {np.round(B_new * 1e3, 2)} mm   |ΔB| = {dB * 1e3:.3f} mm")
            B = B_new
            if dB < lpc_tol:
                break
        dotCal.light_origin = B.copy()
        dz = (U[..., 2] - U_raw[..., 2])[np.isfinite(U[..., 2]) & np.isfinite(U_raw[..., 2])]
        print(f"  calibration points moved by median {np.median(dz) * 1e3:+.1f} mm in z "
              f"(p5…p95 {np.percentile(dz, 5) * 1e3:+.1f} … {np.percentile(dz, 95) * 1e3:+.1f} mm)")
    AB = float(B[0 if mode == "x" else 1])
    print(f"Baseline from {n_lines} dot lines: B = {np.round(B * 1e3, 2)} mm  (AB along {mode}: {AB * 100:.4f} cm)")

    V = dotCal.estimate_unit_vectors(U, B)
    print(f"Unit vectors: {int(np.all(np.isfinite(V), axis=1).sum())} / {len(V)}")

    trails = dotCal.fit_trails_from_detections(subpixel_list, U, z_min=z_min, step_px=1.0,
                                               image_shape=image.shape[:2], V=V, B=B, K=K)
    lens = [len(t["u"]) for t in trails if t is not None]
    print(f"Trails: {len(lens)} dots, {int(np.mean(lens))} samples on average (z: {z_min} m → ∞)")
    sigma_u = dotCal.estimate_sigma_u_trails(subpixel_list, trails)
    sigma_tof_table = dotCal.estimate_sigma_tof(U, expected_z=cal_dists)
    sigma_u_ab = estimate_sigma_u_ab(dotCal, subpixel_list, trails, sl_paths, z_min)
    print(f"σ_u = {sigma_u:.4f} px (2-D residual), σ_u_ab = {sigma_u_ab:.4f} px (runtime AB)")
    for z, s in sigma_tof_table:
        print(f"  σ_ToF({z:.3f} m) = {s * 1e3:.2f} mm")

    K_inv = np.linalg.inv(K)
    return {
        "metadata": {
            "created": datetime.now().isoformat(),
            "camera": camera,
            "sl_cal_path": str(cfg["sl_cal"]),
            "tof_cal_path": str(cfg["tof_cal"]),
            "pipeline": "codePaperlike",
            "mode": mode,
            "baseline_guess": B_guess.tolist(),
            "subpixel_mode": subpixel_mode,
            "pcd_depth_mode": "radial",
            "tof_sample": tof_sample,
            "pcd_unit_scale": 0.001,
            "track_in_tx_space": track_tx_space,
            "baseline_outlier_sigma": outlier_sigma,
            "baseline_min_points": min_points,
            "trail_z_min": z_min,
            "sor": {"enabled": bool(dotCal.sor), "k": int(dotCal.sor[0]) if dotCal.sor else 0,
                    "nsigma": float(dotCal.sor[1]) if dotCal.sor else 0.0},
            "tof_light_origin": None if dotCal.light_origin is None else np.asarray(dotCal.light_origin).tolist(),
            "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        },
        "mode": mode, "K": K, "K_inv": K_inv, "V": V, "B": B, "AB": AB, "BASELINE_GUESS": B_guess,
        "cal_dists": [float(d) for d in cal_dists], "subpixel_list": subpixel_list, "U": U, "trails": trails,
        "sigma_u": float(sigma_u), "sigma_u_ab": float(sigma_u_ab), "sigma_tof_table": sigma_tof_table,
        "debug": {"image": image, "U_tx": U_tx, "intersection": intersection},
    }


def save_calibration(cal, name, figures=()):
    """Write cal as Calibrations/<name>/calibration.json plus the given (name, fig) plots; returns the file."""
    cal_dir = CAL_ROOT / name
    cal_dir.mkdir(parents=True, exist_ok=True)
    data = {k: v for k, v in cal.items() if k != "debug"}
    data["trails"] = [None if t is None else {k: t[k] for k in ("u", "v", "w")} for t in cal["trails"]]
    cal_file = cal_dir / "calibration.json"
    with open(cal_file, "w") as f:
        json.dump(sanitize(data), f)
    for fig_name, fig in figures:
        fig.savefig(str(cal_dir / f"{fig_name}.png"), dpi=150, bbox_inches="tight")
    print(f"Calibration saved to {cal_file}" + (f" (+{len(figures)} plots)" if figures else ""))
    return cal_file


def find_calibration(camera, tag=""):
    """Newest Calibrations/{camera}_paperlike{tag}*/calibration.json without further _lpc/_corr tags."""
    prefix = f"{camera}_paperlike{tag}"
    cands = sorted((p for p in CAL_ROOT.glob(f"{prefix}*/calibration.json")
                    if not re.match(r"_(lpc|corr)", p.parent.name[len(prefix):])),
                   key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No calibration '{prefix}*' in {CAL_ROOT}")
    return cands[-1]


def load_saved_calibration(dotCal, name=None, camera=None, light_path_correction=False):
    """Load Calibrations/<name> (None: newest matching camera + settings), set its light origin on dotCal, warn on mismatches."""
    if name is None:
        cal_file = find_calibration(camera, cache_tag(dotCal.sor, light_path_correction))
    else:
        cal_file = CAL_ROOT / name / "calibration.json"
    cal = approaches.load_calibration(str(cal_file))
    md = cal.setdefault("metadata", {})
    cal.setdefault("mode", md.get("mode"))
    light = md.get("tof_light_origin")
    dotCal.light_origin = None if light is None else np.asarray(light, dtype=float)

    warnings = []
    if camera is not None and md.get("camera") not in (None, camera):
        warnings.append(f"camera {md['camera']!r} ≠ {camera!r}")
    sor = md.get("sor") or {}
    cur = dotCal.sor
    if sor and (bool(sor.get("enabled")) != bool(cur) or cur and (
            int(sor.get("k", -1)) != cur[0] or abs(float(sor.get("nsigma", -1)) - cur[1]) > 1e-9)):
        warnings.append(f"SOR {sor} ≠ current {cur}")
    if (light is not None) != bool(light_path_correction):
        warnings.append(f"calibration LPC {'on' if light is not None else 'off'} ≠ "
                        f"light_path_correction={light_path_correction} (the calibration's B wins)")
    if md.get("error_model"):
        warnings.append(f"fitted on clouds corrected by the former ToF error model "
                        f"(stages {md['error_model'].get('stages')}) — use a v1.1 calibration")

    print(f"Calibration loaded: {cal_file}")
    print(f"  created {md.get('created', '?')}   mode {cal['mode']}   B = {np.round(cal['B'] * 1e3, 2)} mm")
    print(f"  trails: {sum(t is not None for t in cal['trails'])} / {len(cal['trails'])} dots   "
          f"σ_u = {cal.get('sigma_u')}   σ_u_ab = {cal.get('sigma_u_ab')}")
    print("  light origin (LPC): " + ("off" if light is None else f"{np.round(dotCal.light_origin * 1e3, 2)} mm"))
    for w in warnings:
        print(f"  WARNING: {w}")
    return cal


def _baseline_fig(cal):
    """3-D dot lines in transmitter space and their LS intersection."""
    U_tx, inter = cal["debug"]["U_tx"], cal["debug"]["intersection"]
    pts = U_tx.reshape(-1, 3)
    pts = pts[np.all(np.isfinite(pts), axis=1)]
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=3, alpha=0.3, label="3-D calibration points (U_tx)")
    for i, line in enumerate(U_tx):
        line = line[np.all(np.isfinite(line), axis=1)]
        if len(line) < 2:
            continue
        d = line[-1] - line[0]
        ext = np.vstack([line[0] - 0.3 * d, line, line[-1] + 0.3 * d])
        ax.plot(ext[:, 0], ext[:, 1], ext[:, 2], color=plt.cm.tab10(i % 10), alpha=0.5, linewidth=0.8)
    ax.scatter(*inter, s=200, marker="x", color="red", label=f"LS intersection  AB={cal['AB'] * 100:.3f} cm")
    ax.set_title("Baseline estimation: dot lines + LS intersection")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    return fig


def _unit_vector_figs(cal):
    """3-D plot of every 10th unit vector and the angle-error histogram."""
    U, V, B = cal["U"], cal["V"], cal["B"]
    angle_errs = []
    for i in np.flatnonzero(np.all(np.isfinite(V), axis=1)):
        Q = U[i][np.all(np.isfinite(U[i]), axis=1)] - B
        Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
        angle_errs.extend(np.degrees(np.arccos(np.clip(Qn @ V[i], -1, 1))))
    print(f"Angle error V_i ↔ calibration points: median {np.median(angle_errs):.5f}°, max {np.max(angle_errs):.5f}°")

    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter([0], [0], [0], s=80, marker="o", label="Rx (camera)")
    ax.scatter(*B, s=120, marker="x", color="red", label=f"Tx (AB={cal['AB'] * 100:.4f} cm)")
    for v in V[::10]:
        if np.all(np.isfinite(v)):
            ax.quiver(*B, *v, length=0.4, normalize=True, alpha=0.5)
    ax.set_title("Calibrated transmitter unit vectors V_i  (every 10th dot)")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()

    hist = plt.figure(figsize=(7, 3))
    plt.hist(angle_errs, bins=30, edgecolor="k", linewidth=0.4)
    plt.title("Angle deviation between V_i and calibration points [°]")
    plt.xlabel("Angle error [°]")
    plt.ylabel("Count")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig, hist


def calibration_figures(dotCal, cal):
    """(name, fig) plots of a freshly computed calibration: dots, IDs, baseline, unit vectors, trails."""
    image = cal["debug"]["image"]
    far = cal["subpixel_list"][-1]
    z_far = cal["cal_dists"][-1]

    dots, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image, cmap="gray")
    for d in far:
        ax.add_patch(plt.Circle((d["x"], d["y"]), 2, edgecolor="red", facecolor="none", linewidth=1))
        ax.text(d["x"] + 2, d["y"] + 2, str(d["id"]), fontsize=5, color="yellow")
    ax.set_title(f"{z_far} m  |  {len(far)} dots")
    ax.axis("off")

    trails = dotCal.plot_dot_trails(cal["trails"], image_shape=image.shape[:2], background=image)
    for spx in cal["subpixel_list"]:
        for d in spx:
            trails.axes[0].add_patch(plt.Circle((d["x"], d["y"]), 0.5, edgecolor="red", facecolor="none",
                                                linewidth=0.5, zorder=5))
    trails.axes[0].set_title("Trails + subpixel dots of all distances")

    vectors, hist = _unit_vector_figs(cal)
    return [(f"dot_ids_{z_far}m", dots), ("baseline_3d", _baseline_fig(cal)),
            ("unit_vectors", vectors), ("angle_error_hist", hist), ("dot_trails", trails)]


def run_calibration(camera, name, sor=None, light_path_correction=False, save_plots=False, **kwargs):
    """Calibrate one camera and save it under Calibrations/<name>."""
    dotCal = DotCalibration()
    dotCal.sor = sor
    cal = calibrate(camera, dotCal, light_path_correction=light_path_correction, **kwargs)
    if save_plots:
        plt.switch_backend("Agg")
    return save_calibration(cal, name, calibration_figures(dotCal, cal) if save_plots else ())


def main():
    """Command-line entry."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", required=True, choices=sorted(CAMERAS))
    p.add_argument("--name", required=True, help="Calibrations/<name>/calibration.json")
    p.add_argument("--subpixel-mode", default="GPR", choices=["GPR", "geometricCenter", "radial", "center"])
    p.add_argument("--track-tx-space", action="store_true", help="match dots in transmitter space")
    p.add_argument("--mode", default=None, choices=["x", "y"], help="force the travel mode")
    p.add_argument("--z-min", type=float, default=0.3, help="near trail limit [m]")
    p.add_argument("--sor", type=float, nargs=2, metavar=("K", "NSIGMA"), help="statistical outlier removal")
    p.add_argument("--lpc", action="store_true", help="ToF light-path correction")
    p.add_argument("--plots", action="store_true", help="save the calibration plots")
    args = p.parse_args()
    run_calibration(args.camera, args.name, sor=(int(args.sor[0]), args.sor[1]) if args.sor else None,
                    light_path_correction=args.lpc, save_plots=args.plots, subpixel_mode=args.subpixel_mode,
                    track_tx_space=args.track_tx_space, mode_override=args.mode, z_min=args.z_min)


if __name__ == "__main__":
    main()

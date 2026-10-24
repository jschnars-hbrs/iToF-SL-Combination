#!/usr/bin/env python3
"""Batch-run approaches.py over all test scenes of all cameras and summarise the average errors.

Usage:
    python run_all.py [--dry-run] [--cameras Schmersal] [--scenes Flat_Wall] [--approaches 1 3] [--no-save]
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CAMERAS = {
    "OnSemi": {
        "calibration": ROOT / "Calibrations" / "OnSemi_paperlike" / "calibration.json",
        "test_dir": ROOT / "Simulation_Pictures" / "PBRT" / "SL_ToF_On" / "Test_Scenes",
        "suffix": "_On",
    },
    "Schmersal": {
        "calibration": ROOT / "Calibrations" / "Schmersal_paperlike" / "calibration.json",
        "test_dir": ROOT / "Simulation_Pictures" / "PBRT" / "SL_ToF_Schm" / "Test_Scenes",
        "suffix": "_Schm",
    },
    "SchmersalReal": {
        "calibration": ROOT / "Calibrations" / "SchmersalReal_paperlike" / "calibration.json",
        "test_dir": ROOT / "Pictures" / "Test",
        "suffix": "",
    },
}
FIELDS = ["camera", "scene", "approach", "gt_m", "mean_m", "abs_mean_m", "std_m", "min_m", "max_m",
          "n_valid", "n_total"]


def discover_pairs(test_dir: Path, suffix: str):
    """Yield (scene name, SL path, ToF path) for every SL_*.{exr,png} with a matching ToF_*.pcd."""
    for sl in sorted(list(test_dir.glob("SL_*.exr")) + list(test_dir.glob("SL_*.png"))):
        tof = sl.with_name(sl.name.replace("SL_", "ToF_", 1)).with_suffix(".pcd")
        if not tof.exists():
            continue
        scene = sl.stem[3:]
        if suffix and scene.endswith(suffix):
            scene = scene[:-len(suffix)]
        yield scene, sl, tof


def parse_avg_err_lines(stdout_text, scene_name, cam_name):
    """Rows from the AVG_ERR|approach|gt|n_valid|n_total|mean|abs_mean|std|min|max lines of approaches.py."""
    rows = []
    for line in stdout_text.splitlines():
        parts = line.split("|")
        if parts[0] != "AVG_ERR" or len(parts) != 10:
            continue
        a, gt, nv, nt, mean, absm, std, mn, mx = parts[1:]
        rows.append({"camera": cam_name, "scene": scene_name, "approach": int(a), "gt_m": float(gt),
                     "n_valid": int(nv), "n_total": int(nt), "mean_m": float(mean), "abs_mean_m": float(absm),
                     "std_m": float(std), "min_m": float(mn), "max_m": float(mx)})
    return rows


def print_summary_table(all_errors):
    """Formatted table of all average errors."""
    print(f"\n{'=' * 100}\n  AVERAGE ERROR SUMMARY\n{'=' * 100}")
    print(f"{'Camera':<12} {'Scene':<25} {'Appr':>4} {'GT [m]':>7} {'Mean [m]':>10} {'|Mean| [m]':>10} "
          f"{'Std [m]':>9} {'Min [m]':>9} {'Max [m]':>9} {'Valid':>7}")
    print("-" * 100)
    for r in all_errors:
        print(f"{r['camera']:<12} {r['scene']:<25} {r['approach']:>4} {r['gt_m']:>7.2f} "
              f"{r['mean_m']:>+10.4f} {r['abs_mean_m']:>10.4f} {r['std_m']:>9.4f} "
              f"{r['min_m']:>+9.4f} {r['max_m']:>+9.4f} {r['n_valid']:>3}/{r['n_total']:<3}")
    print("-" * 100)


def save_summary_csv(all_errors, output_path):
    """Summary table as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(all_errors)
    print(f"\nSummary CSV saved to {output_path}")


def main():
    """Command-line entry."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cameras", nargs="+", choices=list(CAMERAS), default=list(CAMERAS))
    p.add_argument("--scenes", nargs="+", default=None, help="substring filters for scene names")
    p.add_argument("--approaches", nargs="+", default=["all"])
    p.add_argument("--dry-run", action="store_true", help="print commands without executing")
    p.add_argument("--no-save", action="store_true", help="do not save plots")
    args = p.parse_args()

    approaches_py = Path(__file__).resolve().parent / "approaches.py"
    total = failed = 0
    all_errors = []
    for cam_name in args.cameras:
        cam = CAMERAS[cam_name]
        if not cam["calibration"].exists():
            print(f"[SKIP] {cam_name}: calibration not found at {cam['calibration']}")
            continue
        for scene, sl, tof in discover_pairs(cam["test_dir"], cam["suffix"]):
            if args.scenes and not any(f in scene for f in args.scenes):
                continue
            name = f"{cam_name}_paperlike/{scene}"
            cmd = [sys.executable, str(approaches_py), "--calibration", str(cam["calibration"]),
                   "--sl", str(sl), "--tof", str(tof), "--approaches", *args.approaches]
            if not args.no_save:
                cmd += ["--save", "--name", name]
            total += 1
            if args.dry_run:
                print(f"[{total}] {name}\n    {' '.join(cmd)}\n")
                continue
            print(f"\n{'#' * 70}\n  [{total}] {name}\n{'#' * 70}\n")
            ret = subprocess.run(cmd, capture_output=True, text=True)
            print(ret.stdout, end="")
            print(ret.stderr, end="", file=sys.stderr)
            if ret.returncode != 0:
                failed += 1
                print(f"[FAIL] {name} (exit code {ret.returncode})")
            else:
                all_errors.extend(parse_avg_err_lines(ret.stdout, scene, cam_name))

    if args.dry_run:
        print(f"Would run {total} scenario(s).")
        return
    print(f"\nDone: {total} scenario(s), {failed} failed.")
    if all_errors:
        print_summary_table(all_errors)
        save_summary_csv(all_errors, ROOT / "Results" / "average_error_summary_paperlike.csv")


if __name__ == "__main__":
    main()

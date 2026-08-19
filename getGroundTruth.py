# Compare the chosen ToF point clouds against the measured ground-truth distances.
#
# For each PosX in SL_iToF_Pairs, load the single chosen PCD and report the average
# and median axial depth (z == distZ, in mm) over the whole valid point cloud. The
# raw mean is dominated by extreme invalid returns (~1e11 mm), so a robust MAD clip
# removes those before averaging; the median is naturally robust. Results, together
# with the ground truth and the error, are printed to the terminal (no files written).
#
# Run with a Python that has numpy, e.g.:
#   & "C:\Users\Julian\AppData\Local\Programs\Python\Python310\python.exe" getGroundTruth.py

import glob
from pathlib import Path

import numpy as np

from removePCDoutliers import read_pcd

ROOT = Path(__file__).resolve().parent
PAIRS_DIR = ROOT / "SL_iToF_Pairs"

# Measured target distances [mm], entered by the user.
#GROUND_TRUTH = {
#    "Pos0": 460, "Pos1": 510, "Pos2": 572, "Pos3": 651, "Pos4": 755,
#    "Pos5": 899, "Pos6": 1111, "Pos7": 1454, "Pos8": 2103, "Pos9": 3800,
#}

offset = 58

GROUND_TRUTH = {
    "Pos0": 460-offset, "Pos1": 510-offset, "Pos2": 572-offset, "Pos3": 651-offset, "Pos4": 755-offset,
    "Pos5": 899-offset, "Pos6": 1111-offset, "Pos7": 1454-offset, "Pos8": 2103-offset, "Pos9": 3800-offset,
}

MAD_K = 3.0   # keep points within MAD_K robust std-devs of the median


def robust_stats(z):
    """Whole-cloud (mean, median) axial depth [mm] after dropping extreme outliers.

    Returns (n_valid, n_kept, mean, median).
    """
    z = np.asarray(z, dtype=float)
    valid = np.isfinite(z) & (z > 1e-6)
    zv = z[valid]
    if zv.size == 0:
        return 0, 0, float("nan"), float("nan")

    m = np.median(zv)
    mad = 1.4826 * np.median(np.abs(zv - m))
    if mad > 0:
        keep = np.abs(zv - m) <= MAD_K * mad
    else:
        lo, hi = np.percentile(zv, [1, 99])
        keep = (zv >= lo) & (zv <= hi)
    zk = zv[keep]
    if zk.size == 0:
        zk = zv
    return int(valid.sum()), int(zk.size), float(zk.mean()), float(np.median(zk))


def main():
    if not PAIRS_DIR.is_dir():
        print(f"Pairs folder not found: {PAIRS_DIR}")
        return

    header = (f"{'Pos':6s}{'n_valid':>9s}{'n_kept':>9s}{'mean':>10s}"
              f"{'median':>10s}{'GT':>8s}{'mean-GT':>10s}{'med-GT':>10s}")
    print(header)
    print("-" * len(header))

    med_errs = []
    for pos in sorted(GROUND_TRUTH):
        gt = GROUND_TRUTH[pos]
        pcds = sorted(glob.glob(str(PAIRS_DIR / pos / "*.pcd")))
        if not pcds:
            print(f"{pos:6s}  (no PCD found)")
            continue

        _, _, arr, *_ = read_pcd(Path(pcds[0]))
        if "z" not in arr.dtype.names:
            print(f"{pos:6s}  (PCD has no 'z' field: {arr.dtype.names})")
            continue

        n_valid, n_kept, mean, median = robust_stats(arr["z"])
        med_errs.append(median - gt)
        print(f"{pos:6s}{n_valid:9d}{n_kept:9d}{mean:10.1f}{median:10.1f}"
              f"{gt:8d}{mean - gt:10.1f}{median - gt:10.1f}")

    if med_errs:
        e = np.array(med_errs)
        print("-" * len(header))
        print(f"median error over {e.size} positions: "
              f"mean {e.mean():+.1f} mm, std {e.std():.1f} mm  "
              f"(min {e.min():+.1f}, max {e.max():+.1f})")


if __name__ == "__main__":
    main()

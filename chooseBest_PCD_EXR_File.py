# Get the subpixel location of the dots and calculate a histogram of the discrepancy. Choose the best PCD and EXR file based on the histogram.
#
# For each position (PosX) there are 100 repeated captures of the same static
# 10x10 dot scene: an SL intensity image (SL_Exr/PosX/Nr{k}.exr) and a matching
# organized ToF point cloud (ToF_PCD_10x10/PosX/Nr{k}.pcd). Same Nr{k} = same
# capture. Downstream code needs ONE representative pair per position.
#
# Method:
#   1. Detect the dots once on the reference frame Nr0.exr (LoG blob detection +
#      GPR subpixel localisation, reusing codePaperlike/dot_calibration.py). The
#      scene is static, so the subpixel dot pixel-locations are reused for every
#      frame.
#   2. For every frame k, sample the axial ToF depth z at each dot location from
#      that frame's PCD.
#   3. Per dot, the 100 repeats form a distribution whose robust centre
#      (median) is the best estimate of the true depth. Score each frame by its
#      MAD-normalized deviation from the per-dot medians (plus a small penalty
#      for dots that dropped out in that frame).
#   4. The frame with the smallest total deviation is the most accurate; copy its
#      EXR + PCD into SL_iToF_Pairs/PosX/.
#
# Run with Python 3.10 (the Jupyter kernel env with OpenEXR + skimage + sklearn):
#   & "C:\Users\Julian\AppData\Local\Programs\Python\Python310\python.exe" chooseBest_PCD_EXR_File.py --plots

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

import numpy as np

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "codePaperlike"))
from dot_calibration import DotCalibration, ToFSampler  # noqa: E402

# ── Defaults (match calibrate.py's SchmersalReal 640x480 config) ──────────────
SRC_EXR = ROOT / "SL_Exr"
SRC_PCD = Path(r"C:\Users\Julian\Documents\Calibrations\iToF_SL_10x10\ToF_PCD_10x10")
OUT = ROOT / "SL_iToF_Pairs"
REF_NAME = "Nr0.exr"

K = np.array([[503.1, 0.0, 320.0],
              [0.0, -503.1, 240.0],
              [0.0, 0.0, 1.0]])
CAL_BLOB = dict(max_sigma=10, num_sigma=8, min_sigma=10, threshold=0.02)
PCD_UNIT_SCALE = 0.001          # mm -> m
MISSING_PENALTY = 1.0           # lambda: added per fraction of dots missing in a frame
_EPS = 1e-6                     # floor for MAD, in metres


def _frame_index(path: Path):
    """Nr{k}.<ext> -> int k, or None if it doesn't match."""
    m = re.fullmatch(r"[Nn]r(\d+)", path.stem)
    return int(m.group(1)) if m else None


def detect_dots(exr_path: Path, dotCal: DotCalibration):
    """LoG blob detection + GPR subpixel localisation on the reference frame.

    Returns an (n_dots, 2) array of subpixel (u, v) pixel coordinates.
    """
    blobs, image = dotCal.detect_blobs(str(exr_path), **CAL_BLOB)
    _, subpixels = dotCal.detect_subpixel_locations(blobs, image, mode="GPR")
    return np.array([[d["x"], d["y"]] for d in subpixels], dtype=float)


def sample_z(pcd_paths, dots, dotCal: DotCalibration):
    """Sample axial z [m] at every dot location in every PCD.

    Returns Z of shape (n_dots, n_frames); NaN where the ToF cloud has no valid
    return at that pixel.
    """
    n_dots = len(dots)
    Z = np.full((n_dots, len(pcd_paths)), np.nan, dtype=float)
    for k, pcd in enumerate(pcd_paths):
        tof = dotCal.load_tof_pcd(str(pcd), unit_scale=PCD_UNIT_SCALE,
                                  depth_mode="axial")
        sampler = ToFSampler(tof, K)
        for i, (u, v) in enumerate(dots):
            P = sampler.point_at(u, v)
            if P is not None:
                Z[i, k] = P[2]
    return Z


def pick_best(Z):
    """Best frame = smallest MAD-normalized deviation from the per-dot median.

    Returns (best_k, scores, n_missing) where scores/n_missing are per frame.
    """
    m = np.nanmedian(Z, axis=1, keepdims=True)                       # (n_dots, 1)
    mad = 1.4826 * np.nanmedian(np.abs(Z - m), axis=1, keepdims=True)
    scale = np.maximum(mad, _EPS)                                     # (n_dots, 1)
    dev = np.abs(Z - m) / scale                                      # (n_dots, n_frames)

    n_dots = Z.shape[0]
    missing = ~np.isfinite(Z)
    n_missing = missing.sum(axis=0)                                  # (n_frames,)
    # nanmean over dots present in each frame, + penalty for dropped dots.
    with np.errstate(invalid="ignore"):
        mean_dev = np.nanmean(dev, axis=0)
    mean_dev = np.where(np.isfinite(mean_dev), mean_dev, np.inf)
    scores = mean_dev + MISSING_PENALTY * (n_missing / max(n_dots, 1))
    return int(np.argmin(scores)), scores, n_missing


def save_plots(pos, Z, scores, best_k, frames, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Frame-score bar chart.
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(frames, scores, color="steelblue")
    ax.axvline(frames[best_k], color="red", lw=1.5,
               label=f"chosen Nr{frames[best_k]}")
    ax.set_xlabel("frame Nr")
    ax.set_ylabel("robust deviation score")
    ax.set_title(f"{pos}: per-frame deviation (lower = better)")
    ax.legend()
    fig.savefig(out_dir / "frame_scores.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Per-dot depth histograms (grid of the first up-to-25 dots).
    n = min(25, Z.shape[0])
    if n:
        cols = 5
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for i in range(n):
            z = Z[i][np.isfinite(Z[i])] * 1e3   # mm
            ax = axes[i]
            if z.size:
                ax.hist(z, bins=20, color="gray")
                zc = float(Z[i, best_k]) * 1e3
                if np.isfinite(zc):
                    ax.axvline(zc, color="red", lw=1.2)
            ax.set_title(f"dot {i}", fontsize=8)
            ax.tick_params(labelsize=6)
        for j in range(n, len(axes)):
            axes[j].axis("off")
        fig.suptitle(f"{pos}: per-dot depth [mm] histograms "
                     f"(red = chosen Nr{frames[best_k]})")
        fig.savefig(out_dir / "dot_histograms.png", dpi=110, bbox_inches="tight")
        plt.close(fig)


def process_position(pos_dir: Path, src_pcd_dir: Path, out_root: Path,
                     dotCal: DotCalibration, make_plots: bool):
    pos = pos_dir.name
    ref = pos_dir / REF_NAME
    if not ref.exists():
        print(f"  {pos}: SKIP (no {REF_NAME})")
        return
    pcd_pos_dir = src_pcd_dir / pos
    if not pcd_pos_dir.is_dir():
        print(f"  {pos}: SKIP (no PCD folder {pcd_pos_dir})")
        return

    # Frames present in BOTH folders, ordered by index.
    exr_by_k = {_frame_index(p): p for p in pos_dir.glob("*.exr")
                if _frame_index(p) is not None}
    pcd_by_k = {_frame_index(p): p for p in pcd_pos_dir.glob("*.pcd")
                if _frame_index(p) is not None}
    frames = sorted(set(exr_by_k) & set(pcd_by_k))
    if not frames:
        print(f"  {pos}: SKIP (no matching EXR/PCD frame numbers)")
        return

    dots = detect_dots(ref, dotCal)
    if len(dots) == 0:
        print(f"  {pos}: SKIP (no dots detected on {REF_NAME})")
        return

    pcd_paths = [pcd_by_k[k] for k in frames]
    Z = sample_z(pcd_paths, dots, dotCal)
    best_i, scores, n_missing = pick_best(Z)
    best_k = frames[best_i]

    out_dir = out_root / pos
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(exr_by_k[best_k], out_dir / exr_by_k[best_k].name)
    shutil.copy2(pcd_by_k[best_k], out_dir / pcd_by_k[best_k].name)

    csv = out_dir / "frame_scores.csv"
    with open(csv, "w") as f:
        f.write("frame,score,n_missing_dots\n")
        for i, k in enumerate(frames):
            f.write(f"{k},{scores[i]:.6f},{int(n_missing[i])}\n")

    if make_plots:
        save_plots(pos, Z, scores, best_i, frames, out_dir)

    print(f"  {pos}: {len(dots)} dots, {len(frames)} frames -> "
          f"chose Nr{best_k}  (score={scores[best_i]:.4f}, "
          f"missing={int(n_missing[best_i])})")


def main():
    ap = argparse.ArgumentParser(
        description="Choose the most accurate (EXR, PCD) pair per position.")
    ap.add_argument("--src-exr", type=Path, default=SRC_EXR)
    ap.add_argument("--src-pcd", type=Path, default=SRC_PCD)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override blob-detection LoG threshold")
    ap.add_argument("--pos", nargs="*", default=None,
                    help="subset of positions, e.g. --pos Pos0 Pos3")
    ap.add_argument("--plots", action="store_true",
                    help="save per-position diagnostic figures")
    args = ap.parse_args()

    if args.threshold is not None:
        CAL_BLOB["threshold"] = args.threshold

    if not args.src_exr.is_dir():
        sys.exit(f"EXR source not found: {args.src_exr}")
    if not args.src_pcd.is_dir():
        sys.exit(f"PCD source not found: {args.src_pcd}")

    pos_dirs = sorted((d for d in args.src_exr.iterdir() if d.is_dir()),
                      key=lambda d: (_frame_index_pos(d.name), d.name))
    if args.pos:
        wanted = set(args.pos)
        pos_dirs = [d for d in pos_dirs if d.name in wanted]
    if not pos_dirs:
        sys.exit(f"no position folders found in {args.src_exr}")

    dotCal = DotCalibration()
    print(f"Choosing best pairs  |  {args.src_exr} + {args.src_pcd} -> {args.out}")
    for pos_dir in pos_dirs:
        process_position(pos_dir, args.src_pcd, args.out, dotCal, args.plots)
    print("Done.")


def _frame_index_pos(name: str):
    """Sort key for 'Pos12' -> 12 (fallback to inf so odd names sort last)."""
    m = re.fullmatch(r"[Pp]os(\d+)", name)
    return int(m.group(1)) if m else float("inf")


if __name__ == "__main__":
    main()

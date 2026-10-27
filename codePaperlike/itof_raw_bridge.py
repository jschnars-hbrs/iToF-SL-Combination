#!/usr/bin/env python3
"""RAW ToF frames (.npy) → PCD, run with the Python of the itofcameracalibration repository.

Uses the iToF calibration (HDF5) and that repository's tof_pipeline; the result is unwrapped by
- dots    : SL dot trails of a codePaperlike calibration.json (binary2pcd_manual.py, one frequency)
- 2F      : dual-frequency pipeline with the matching _set1 frames (binary2pcd_2F.py)
- single  : one frequency, no unwrapping beyond the unambiguity range (binary2pcd.py)
- auto    : 2F if _set1 frames exist, otherwise dots
All frames are averaged into one cloud (<out>/<name>.pcd); --each also writes every frame.

Usage:
    <itof-python> itof_raw_bridge.py --repo <.../itofcameracalibration> --raw Pos0_set0 \
        --kalibration CalibData.hdf5 --dot_calibration calibration.json --out <dir> --name scene
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np


def natural_key(p):
    """Nr2 before Nr10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.name)]


def raw_frames(raw, frames=None):
    """Frame files of a RAW folder or a single .npy (files below 4 MB are incomplete and skipped)."""
    raw = Path(raw)
    files = sorted(raw.glob("*.npy"), key=natural_key) if raw.is_dir() else [raw]
    files = [f for f in files if f.stat().st_size >= 4_000_000]
    return files[:frames] if frames else files


def camera_frequencies(raw, OmegaConf):
    """Modulation frequencies [MHz] from the camera config above a RAW folder/frame: the measurement project's
    .hydra/config.yaml (camera.settings.Frequency) or, for single recordings, the camera YAML saved next to
    them (e.g. Schmersal_AM-T100_RAOI.yaml, settings.Frequency at top level)."""
    for parent in Path(raw).resolve().parents:
        hydra = parent / ".hydra" / "config.yaml"
        candidates = [(hydra, "camera.settings.Frequency")] if hydra.is_file() else []
        candidates += [(cfg, "settings.Frequency") for cfg in sorted(parent.glob("*.yaml"))]
        for cfg, key in candidates:
            f = OmegaConf.select(OmegaConf.load(cfg), key)
            if f is not None:
                return list(f) if OmegaConf.is_list(f) else [f]
    raise FileNotFoundError(f"No camera config (.hydra/config.yaml or *.yaml with settings.Frequency) above {raw}")


def set_index(raw):
    """N of a ..._setN folder (0 if the name has none)."""
    folder = Path(raw) if Path(raw).is_dir() else Path(raw).parent
    m = re.search(r"_set(\d+)$", folder.name)
    return int(m.group(1)) if m else 0


def second_set(raw):
    """The _set1 counterpart of a _set0 folder/frame, or None."""
    raw = Path(raw)
    folder = raw if raw.is_dir() else raw.parent
    if not folder.name.endswith("_set0"):
        return None
    other = folder.with_name(folder.name[:-1] + "1")
    if not other.is_dir():
        return None
    return other if raw.is_dir() else (other / raw.name if (other / raw.name).is_file() else None)


def load(path):
    with open(path, "rb") as f:
        return np.load(f, allow_pickle=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="itofcameracalibration package folder (has tof_processing/, utils/)")
    p.add_argument("--raw", required=True, help="RAW folder (Pos*_set0) or one .npy frame")
    p.add_argument("--kalibration", required=True, help="iToF calibration (HDF5)")
    p.add_argument("--dot_calibration", default=None, help="codePaperlike calibration.json (mode dots)")
    p.add_argument("--mode", default="auto", choices=["auto", "dots", "2F", "single"])
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--name", required=True, help="output file stem")
    p.add_argument("--frames", type=int, default=None, help="only the first N frames")
    p.add_argument("--each", action="store_true", help="also write one PCD per frame")
    p.add_argument("--params", default="{}", help="JSON with binary2pcd_manual options (chi_max, sigma_tof, ...)")
    args = p.parse_args()

    repo = Path(args.repo)
    sys.path[:0] = [str(repo), str(repo / "utils")]
    from omegaconf import OmegaConf
    from camerametaclass import pcd_handler
    from tof_processing.tof_processing import parse_DeltaRawData, tof_pipeline, tof_pipeline2F

    opts = json.loads(args.params)
    H, W = int(opts.get("height", 480)), int(opts.get("width", 640))
    files = raw_frames(args.raw, args.frames)
    if not files:
        sys.exit(f"No RAW frames (≥ 4 MB .npy) in {args.raw}")
    freqs = camera_frequencies(args.raw, OmegaConf)
    mode = args.mode
    if mode == "auto":
        mode = "2F" if second_set(args.raw) is not None and len(freqs) > 1 else "dots"
    print(f"RAW: {len(files)} frame(s) from {args.raw}   frequencies {freqs} MHz   mode {mode}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer = pcd_handler.PCD_Writer()

    def write(path, X, Y, Z, gray, rad):
        writer.dict_to_pcd({"x": X.flatten(), "y": Y.flatten(), "z": Z.flatten(),
                            "grayValue": np.asarray(gray, dtype=np.float64).flatten(),
                            "rad": rad.flatten(), "distZ": Z.flatten()},
                           str(path), sensor_height=H, sensor_width=W, fmt="binary")

    clouds = []  # (name, X, Y, Z, gray, rad) per frame
    if mode == "dots":
        import binary2pcd_manual as m
        if not args.dot_calibration:
            sys.exit("Mode 'dots' needs --dot_calibration (the codePaperlike calibration.json)")
        pipeline = tof_pipeline(fMod=freqs[set_index(args.raw)], height=H, width=W, calibrationFile=args.kalibration)
        R = pipeline.unambiguityRange
        m_gm = pipeline.lens_m_gm if pipeline.lens_m_gm is not None else np.ones((H, W))
        m_gm = np.where(np.isfinite(m_gm) & (m_gm > 0.5) & (m_gm < 2.0), m_gm, np.nan)
        k_max = int(np.floor(float(opts.get("max_dist", 4.0)) * np.nanmax(m_gm) / R))
        print(f"fMod = {pipeline.fMod_MHz} MHz, unambiguity range R = {R:.4f} m, k = 0..{k_max}")
        trails = m.load_dot_calibration(args.dot_calibration)["trails"]

        names, rad_stack, gray_stack = [], [], []
        amp_sum = np.zeros((H, W))
        for f in files:
            Data = parse_DeltaRawData(load(f))
            pipeline.process_2rad(TabsA=Data[0], TabsB=Data[1])
            names.append(f.stem)
            rad_stack.append(np.mod(pipeline.radDist, R).astype(np.float32))
            gray_stack.append(np.asarray(pipeline.intensity, dtype=np.float32))
            amp_sum += pipeline.amplitude
        rad_stack, gray_stack = np.stack(rad_stack), np.stack(gray_stack)
        sl_img = amp_sum / len(names) if opts.get("sl_image", "amplitude") == "amplitude" else gray_stack.mean(axis=0)
        r0_mean = m.circular_mean_dist(rad_stack, R)
        dots = m.locate_dots(trails, sl_img, r0_mean, m_gm, R, k_max, float(opts.get("max_dist", 4.0)),
                             float(opts.get("chi_max", 3.0)), float(opts.get("sigma_sl_px", 1.0)),
                             float(opts.get("sigma_tof", 0.02)), float(opts.get("sigma_perp_px", 0.5)),
                             float(opts.get("dot_radius", 3.0)), float(opts.get("ambiguity_margin", 4.0)))
        m.neighbour_consensus(dots, R, k_max, float(opts.get("chi_max", 3.0)), int(opts.get("neighbours", 3)))
        z_ref_img = np.full((H, W), np.nan)
        m.dot_masks(dots, sl_img, z_ref_img, float(opts.get("dot_radius", 3.0)), H, W)
        mask = np.isfinite(z_ref_img) & np.isfinite(m_gm)
        valid = [d for d in dots if d["valid"]]
        print(f"Dots: {len(valid)}/{len(dots)} valid, k = {sorted(set(d['k'] for d in valid))}, "
              f"{sum(d['ambiguous'] for d in valid)} ambiguous, {sum(d['corrected'] for d in valid)} corrected")
        for name, r_wrapped, gray in zip(names, rad_stack, gray_stack):
            k = np.clip(np.round((z_ref_img * m_gm - r_wrapped) / R), 0, k_max)
            r = np.where(mask, r_wrapped + k * R, np.nan)
            Z, X, Y = pipeline.calc_XYZ(r)
            clouds.append((name, X, Y, Z, gray, r))
    elif mode == "2F":
        pipeline = tof_pipeline2F(fMod1=freqs[0], fMod2=freqs[1], height=H, width=W, calibrationFile=args.kalibration)
        for f in files:
            f2 = second_set(f)
            if f2 is None:
                print(f"  {f.name}: no _set1 frame, skipped")
                continue
            pipeline.process_sequential(parse_DeltaRawData(load(f)), parse_DeltaRawData(load(f2)))
            clouds.append((f.stem, pipeline.X.copy(), pipeline.Y.copy(), pipeline.Z.copy(),
                           np.array(pipeline.intensity, dtype=float), pipeline.radDist.copy()))
    else:
        pipeline = tof_pipeline(fMod=freqs[set_index(args.raw)], height=H, width=W, calibrationFile=args.kalibration)
        print(f"fMod = {pipeline.fMod_MHz} MHz, unambiguity range R = {pipeline.unambiguityRange:.4f} m (no unwrapping)")
        for f in files:
            pipeline.process_fromRaw(load(f), TapOrder="Delta")
            clouds.append((f.stem, pipeline.X.copy(), pipeline.Y.copy(), pipeline.Z.copy(),
                           np.array(pipeline.intensity, dtype=float), pipeline.radDist.copy()))
    if not clouds:
        sys.exit("No frame could be converted")

    if args.each:
        for name, X, Y, Z, gray, rad in clouds:
            write(out / f"{args.name}_{name}.pcd", X, Y, Z, gray, rad)
    with warnings.catch_warnings():  # pixels without a dot are NaN in every frame
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = [np.nanmean(np.stack([c[i] for c in clouds]), axis=0) for i in range(1, 6)]
    path = out / f"{args.name}.pcd"
    write(path, *mean)
    print(f"Mean of {len(clouds)} frame(s) → {path}")
    print(f"BRIDGE_PCD|{path}")


if __name__ == "__main__":
    main()

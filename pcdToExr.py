# Convert PCD to EXR (whole folders at a time)
#
# Reads organized ToF point clouds (640x480, row-major) and writes a single-channel
# float32 EXR intensity image per frame from the PCD's grayValue field, mirroring the
# input folder structure.
#
# Examples:
#     python pcdToExr.py
#     python pcdToExr.py ToF_PCD_10x10 SL_Exr
#     python pcdToExr.py ./in ./out --field grayValue

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import OpenEXR
    import Imath
except ImportError:
    sys.exit("OpenEXR module required (provides OpenEXR + Imath). "
             "Install with: python -m pip install OpenEXR")

from removePCDoutliers import read_pcd

DEFAULT_INPUT = Path("PCD")
DEFAULT_OUTPUT = Path("Test_EXR")
# Channel name must match DotCalibration.SL_CHANNEL (codePaperlike/dot_calibration.py:28)
# so read_image() can pull it by name.
SL_CHANNEL = "S0.940,000nm"


def write_exr(path: Path, img: np.ndarray, channel: str):
    h, w = img.shape
    header = OpenEXR.Header(w, h)
    header["channels"] = {channel: Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    out = OpenEXR.OutputFile(str(path), header)
    out.writePixels({channel: np.ascontiguousarray(img, np.float32).tobytes()})
    out.close()


def process_file(in_path: Path, out_path: Path, field: str, channel: str):
    hdr, dtype, arr, *_ = read_pcd(in_path)
    if field not in dtype.names:
        raise KeyError(f"field '{field}' not in {dtype.names}")
    w, h = int(hdr["WIDTH"]), int(hdr["HEIGHT"])
    if len(arr) != w * h:
        raise ValueError(f"not organized: {len(arr)} points != {w}x{h}")

    img = arr[field].astype(np.float32).reshape(h, w)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_exr(out_path, img, channel)
    return w, h, float(np.nanmin(img)), float(np.nanmax(img))


def main():
    ap = argparse.ArgumentParser(
        description="Convert organized ToF PCDs to single-channel EXR intensity images.")
    ap.add_argument("input_dir", type=Path, nargs="?", default=DEFAULT_INPUT)
    ap.add_argument("output_dir", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    ap.add_argument("--field", default="grayValue",
                    help="PCD field to store as the image (default grayValue)")
    ap.add_argument("--channel", default=SL_CHANNEL,
                    help=f"EXR channel name (default '{SL_CHANNEL}', matches "
                         "DotCalibration.SL_CHANNEL)")
    ap.add_argument("--glob", default="*.pcd", help="filename pattern (default *.pcd)")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"input dir not found: {args.input_dir}")

    files = sorted(p for p in args.input_dir.glob(f"**/{args.glob}") if p.is_file())
    if not files:
        files = sorted(p for p in args.input_dir.glob(f"**/{args.glob.replace('.pcd', '.PCD')}")
                       if p.is_file())
    if not files:
        sys.exit(f"no files matching '{args.glob}' in {args.input_dir}")

    print(f"Converting {len(files)} file(s)  |  field={args.field}  "
          f"channel='{args.channel}'  {args.input_dir} -> {args.output_dir}")
    fails = 0
    for f in files:
        rel = f.relative_to(args.input_dir)
        out = (args.output_dir / rel).with_suffix(".exr")
        try:
            w, h, lo, hi = process_file(f, out, args.field, args.channel)
            print(f"  {str(rel):24s} -> {out}  [{w}x{h}  {lo:.1f}..{hi:.1f}]")
        except Exception as e:
            fails += 1
            print(f"  {str(rel):24s} FAILED: {e}")
    print(f"Done. {len(files)-fails}/{len(files)} ok.")


if __name__ == "__main__":
    main()

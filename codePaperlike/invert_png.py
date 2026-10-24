#!/usr/bin/env python3
"""Invert every PNG in a folder into <folder>_inverted (or --out).

Usage:
    python invert_png.py <png_folder> [--out <folder>]
"""

import argparse
from pathlib import Path

import cv2


def main():
    """Command-line entry."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path, help="folder with PNG images")
    p.add_argument("--out", type=Path, default=None, help="output folder (default: <src>_inverted)")
    args = p.parse_args()
    out = args.out or args.src.with_name(args.src.name + "_inverted")
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(f for f in args.src.iterdir() if f.suffix.lower() == ".png"):
        cv2.imwrite(str(out / f.name), cv2.bitwise_not(cv2.imread(str(f), cv2.IMREAD_UNCHANGED)))


if __name__ == "__main__":
    main()

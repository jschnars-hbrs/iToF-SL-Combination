# Remove extreme outliers from PCD files (whole folders at a time)

"""
Examples:
    python remove_pcd_outliers.py ./raw ./clean
    python remove_pcd_outliers.py ./raw ./clean --factor 3 --recursive
    python remove_pcd_outliers.py ./raw ./clean --sor --sor-std 2.0

    python remove_pcd_outliers.py ToF_PCD ToF_PCD_cleaned --recursive --ref median --factor 2
"""


import argparse
import sys
from pathlib import Path

import numpy as np

_NP_TYPE = {
    ("F", 4): np.float32, ("F", 8): np.float64,
    ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
    ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
}


class PCDError(Exception):
    pass


def parse_header(raw: bytes):
    """Return (header_dict, header_len_bytes, header_lines_list)."""
    # header ends after the DATA line
    idx = raw.find(b"DATA")
    if idx == -1:
        raise PCDError("no DATA line found")
    nl = raw.find(b"\n", idx)
    if nl == -1:
        raise PCDError("truncated DATA line")
    header_len = nl + 1
    text = raw[:header_len].decode("ascii", errors="replace")
    lines = text.splitlines()
    hdr = {}
    order = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        key, _, val = s.partition(" ")
        hdr[key.upper()] = val.strip()
        order.append(key.upper())
    return hdr, header_len, lines


def build_dtype(hdr):
    fields = hdr["FIELDS"].split()
    sizes = [int(x) for x in hdr["SIZE"].split()]
    types = hdr["TYPE"].split()
    counts = [int(x) for x in hdr["COUNT"].split()] if "COUNT" in hdr else [1] * len(fields)
    names, formats = [], []
    for f, s, t, c in zip(fields, sizes, types, counts):
        np_t = _NP_TYPE.get((t, s))
        if np_t is None:
            raise PCDError(f"unsupported field type {t}{s} for field '{f}'")
        if c == 1:
            names.append(f)
            formats.append(np_t)
        else:
            for j in range(c):
                names.append(f"{f}_{j:04d}")
                formats.append(np_t)
    return np.dtype({"names": names, "formats": formats}), fields, sizes, types, counts


def read_pcd(path: Path):
    raw = path.read_bytes()
    hdr, header_len, _ = parse_header(raw)
    dtype, fields, sizes, types, counts = build_dtype(hdr)
    npts = int(hdr["POINTS"]) if "POINTS" in hdr else int(hdr["WIDTH"]) * int(hdr["HEIGHT"])
    data_kind = hdr["DATA"].split()[0].lower()
    body = raw[header_len:]

    if data_kind == "binary":
        arr = np.frombuffer(body, dtype=dtype, count=npts)
    elif data_kind == "ascii":
        # each token maps to one element of the (flattened) dtype
        flat_names = dtype.names
        vals = np.fromstring(body.decode("ascii", errors="replace"), sep=" ")
        if vals.size < npts * len(flat_names):
            raise PCDError("ascii body shorter than expected")
        vals = vals[: npts * len(flat_names)].reshape(npts, len(flat_names))
        arr = np.empty(npts, dtype=dtype)
        for i, nm in enumerate(flat_names):
            arr[nm] = vals[:, i].astype(dtype[nm])
    elif data_kind == "binary_compressed":
        raise PCDError("DATA binary_compressed is not supported; re-export as binary or ascii")
    else:
        raise PCDError(f"unknown DATA type '{data_kind}'")

    return hdr, dtype, arr, fields, sizes, types, counts


def get_xyz(arr, dtype):
    names = dtype.names
    need = ("x", "y", "z")
    if not all(n in names for n in need):
        raise PCDError("cloud has no x/y/z fields")
    xyz = np.stack([arr["x"].astype(np.float64),
                    arr["y"].astype(np.float64),
                    arr["z"].astype(np.float64)], axis=1)
    return xyz


def extreme_mask(xyz, ref, percentile, factor, center, robust_k=1000.0):
    """True = keep. Drops non-finite points and points whose radius exceeds
    factor * reference, where reference is one of:
      - 'percentile': the given upper radius percentile (robust to garbage)
      - 'median'    : median radius (robust; matches the 'typical near value')
      - 'mean'      : mean radius, computed ROBUSTLY by first discarding
                      order-of-magnitude garbage (radius > robust_k * median),
                      so 1e9-1e11 points cannot wreck the mean.
    Returns (keep_mask, n_nonfinite, n_extreme, thr).
    """
    finite = np.isfinite(xyz).all(axis=1)
    keep = finite.copy()
    pts = xyz[finite]
    if pts.size == 0:
        return keep, 0, 0, float("inf")
    c = np.median(pts, axis=0) if center == "median" else np.zeros(3)
    r = np.linalg.norm(pts - c, axis=1)

    med = np.median(r)
    if ref == "percentile":
        base = np.percentile(r, percentile)
    elif ref == "median":
        base = med
    elif ref == "mean":
        r_rob = r[r <= med * robust_k] if med > 0 else r
        base = r_rob.mean() if r_rob.size else med
    else:
        raise PCDError(f"unknown ref '{ref}'")

    thr = factor * base if base > 0 else np.inf
    sub_keep = r <= thr
    idx = np.flatnonzero(finite)
    keep[idx[~sub_keep]] = False
    return keep, int((~finite).sum()), int((~sub_keep).sum()), float(thr)


def sor_mask(xyz, nb_neighbors, std_ratio):
    """Statistical outlier removal on already-cleaned xyz. True = keep."""
    from scipy.spatial import cKDTree
    n = len(xyz)
    if n <= nb_neighbors + 1:
        return np.ones(n, dtype=bool), 0
    tree = cKDTree(xyz)
    d, _ = tree.query(xyz, k=nb_neighbors + 1, workers=-1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    keep = mean_d <= thr
    return keep, int((~keep).sum())


def write_pcd(path: Path, hdr, dtype, arr, data_kind="binary"):
    n = len(arr)
    fields = hdr["FIELDS"]
    size = hdr["SIZE"]
    typ = hdr["TYPE"]
    count = hdr.get("COUNT", " ".join(["1"] * len(fields.split())))
    vp = hdr.get("VIEWPOINT", "0 0 0 1 0 0 0")
    ver = hdr.get("VERSION", "0.7")
    header = (
        "# .PCD v.7 - Point Cloud Data file format\n"
        f"VERSION {ver}\n"
        f"FIELDS {fields}\n"
        f"SIZE {size}\n"
        f"TYPE {typ}\n"
        f"COUNT {count}\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        f"VIEWPOINT {vp}\n"
        f"POINTS {n}\n"
        f"DATA {data_kind}\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        if data_kind == "binary":
            f.write(np.ascontiguousarray(arr).tobytes())
        else:  # ascii
            flat = dtype.names
            for row in arr:
                f.write((" ".join(repr(float(row[nm])) if np.issubdtype(dtype[nm], np.floating)
                                   else str(row[nm]) for nm in flat) + "\n").encode("ascii"))


def process_file(in_path, out_path, args):
    hdr, dtype, arr, *_ = read_pcd(in_path)
    n0 = len(arr)
    xyz = get_xyz(arr, dtype)

    keep, n_nf, n_ext, thr = extreme_mask(
        xyz, args.ref, args.percentile, args.factor, args.center)
    arr = arr[keep]
    n_sor = 0
    if args.sor and len(arr) > 0:
        xyz2 = get_xyz(arr, dtype)
        skeep, n_sor = sor_mask(xyz2, args.sor_neighbors, args.sor_std)
        arr = arr[skeep]

    out_kind = "ascii" if args.ascii_out else "binary"
    write_pcd(out_path, hdr, dtype, arr, data_kind=out_kind)
    n1 = len(arr)
    pct = 100.0 * (n0 - n1) / n0 if n0 else 0.0
    print(f"  {in_path.name:30s} {n0:>8d} -> {n1:>8d}  (-{n0-n1:>5d}, {pct:5.2f}%) "
          f"[thr {thr:.0f}mm, nan/inf {n_nf}, extreme {n_ext}, sor {n_sor}]")
    return n0, n1


def main():
    ap = argparse.ArgumentParser(description="Batch-remove extreme outliers from PCD files.")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--ref", choices=["percentile", "median", "mean"], default="median",
                    help="reference for the radial threshold (default: median = "
                         "the typical near-field value)")
    ap.add_argument("--factor", type=float, default=2.0,
                    help="drop points with radius > factor*reference (default 2.0)")
    ap.add_argument("--percentile", type=float, default=99.9,
                    help="only used when --ref percentile (default 99.9)")
    ap.add_argument("--center", choices=["origin", "median"], default="origin",
                    help="reference point for radial distance (default: sensor origin)")
    ap.add_argument("--sor", action="store_true",
                    help="also apply statistical outlier removal (default off)")
    ap.add_argument("--sor-neighbors", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--recursive", action="store_true",
                    help="recurse into subfolders (mirrors tree in output)")
    ap.add_argument("--ascii-out", action="store_true",
                    help="write ascii PCD instead of binary")
    ap.add_argument("--glob", default="*.pcd", help="filename pattern (default *.pcd)")
    args = ap.parse_args()

    if not args.input_dir.is_dir():
        sys.exit(f"input dir not found: {args.input_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"**/{args.glob}" if args.recursive else args.glob
    files = sorted(p for p in args.input_dir.glob(pattern) if p.is_file())
    # case-insensitive .PCD too
    if not files:
        files = sorted(p for p in args.input_dir.glob(pattern.replace(".pcd", ".PCD")) if p.is_file())
    if not files:
        sys.exit(f"no files matching '{args.glob}' in {args.input_dir}")

    print(f"Processing {len(files)} file(s)  |  factor={args.factor} p={args.percentile} "
          f"center={args.center} sor={args.sor}")
    tot0 = tot1 = 0
    fails = 0
    for f in files:
        rel = f.relative_to(args.input_dir)
        out = args.output_dir / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            a, b = process_file(f, out, args)
            tot0 += a; tot1 += b
        except Exception as e:
            fails += 1
            print(f"  {f.name:30s} FAILED: {e}")
    print(f"Done. {len(files)-fails}/{len(files)} ok. "
          f"Total {tot0} -> {tot1} points (removed {tot0-tot1}).")


if __name__ == "__main__":
    main()
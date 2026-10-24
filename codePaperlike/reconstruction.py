"""3-D reconstruction of the approach depths and export as binary PCD (mm, like the source clouds)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay, QhullError

COLORS = {1: (31, 119, 180), 2: (44, 160, 44), 3: (214, 39, 40)}
DEPTH_KEYS = {1: ("u_out", "v_out", "Z_out"), 2: ("u_sub", "v_sub", "z_fus"), 3: ("u_sub", "v_sub", "Z_out")}


def recon_dots(res, approach):
    """(u, v, Z, dot id) of the valid dots in an approach result."""
    rs = res["results"]
    u, v, z = (np.array([r[k] for r in rs], dtype=float) for k in DEPTH_KEYS[approach])
    ids = np.array([r["i"] for r in rs], dtype=int)
    ok = np.isfinite(u) & np.isfinite(v) & np.isfinite(z) & (z > 1e-6)
    return u[ok], v[ok], z[ok], ids[ok]


def backproject_pixels(u, v, z, K):
    """Pinhole back-projection of pixels with axial depth z."""
    return np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]


def recon_dot_mode(u, v, z, ids, K, size_px):
    """Every dot as a disc of size_px diameter, each pixel on its own ray."""
    r = size_px / 2.0
    n = int(np.ceil(r))
    du, dv = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    disc = du ** 2 + dv ** 2 <= r * r
    du, dv = du[disc].astype(float), dv[disc].astype(float)
    uu = (u[:, None] + du[None, :]).ravel()
    vv = (v[:, None] + dv[None, :]).ravel()
    return backproject_pixels(uu, vv, np.repeat(z, len(du)), K), np.repeat(ids, len(du))


def recon_area_mode(u, v, z, ids, K, H, W, step, max_edge):
    """Surface between the dots: 1/Z interpolated over a Delaunay mesh (planes stay planar), plus the dots."""
    uv = np.c_[u, v]
    try:
        tri = Delaunay(uv)
    except (QhullError, ValueError):
        print("    no surface possible (fewer than 3 dots or all collinear) — dots only")
        return backproject_pixels(u, v, z, K), ids
    corners = uv[tri.simplices]
    edges = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max(axis=1)
    gu, gv = np.meshgrid(np.arange(0, W, step, dtype=float), np.arange(0, H, step, dtype=float))
    q = np.c_[gu.ravel(), gv.ravel()]
    s = tri.find_simplex(q)
    good = s >= 0
    good[good] = edges[s[good]] <= max_edge
    q, s = q[good], s[good]
    T = tri.transform[s]
    b = np.einsum("nij,nj->ni", T[:, :2], q - T[:, 2])
    bary = np.c_[b, 1.0 - b.sum(axis=1)]
    w = (bary * (1.0 / z)[tri.simplices[s]]).sum(axis=1)
    P = backproject_pixels(q[:, 0], q[:, 1], 1.0 / w, K)
    return np.vstack([P, backproject_pixels(u, v, z, K)]), np.r_[np.full(len(P), -1), ids]


def write_pcd_xyzrgb(path, P, rgb, dot_ids, approach_ids, scale=1000.0):
    """Binary PCD with fields x y z rgb dot approach; xyz multiplied by scale."""
    arr = np.zeros(len(P), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"),
                                  ("dot", "<i4"), ("approach", "<i4")])
    arr["x"], arr["y"], arr["z"] = (P * scale).T
    rgb = np.asarray(rgb, dtype=np.uint32)
    arr["rgb"] = ((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]).astype("<u4").view("<f4")
    arr["dot"], arr["approach"] = dot_ids, approach_ids
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb dot approach\n"
              "SIZE 4 4 4 4 4 4\nTYPE F F F F I I\nCOUNT 1 1 1 1 1 1\n"
              f"WIDTH {len(arr)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(arr)}\nDATA binary\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(arr.tobytes())


def export_pcds(results, K, H, W, out_dir, stem, mode="area", dot_size_px=3, area_step_px=1,
                area_max_edge_px=50, combined=False):
    """Reconstruct {approach: result} as 'dot' discs or an 'area' surface and write one PCD each; returns the clouds."""
    if mode not in ("dot", "area"):
        raise ValueError(f"mode must be 'dot' or 'area', not {mode!r}")
    tag = f"dot{dot_size_px}px" if mode == "dot" else f"area{area_step_px}px"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clouds = {}
    for n, res in results.items():
        u, v, z, ids = recon_dots(res, n)
        if mode == "dot":
            P, pid = recon_dot_mode(u, v, z, ids, K, dot_size_px)
        else:
            P, pid = recon_area_mode(u, v, z, ids, K, H, W, area_step_px, area_max_edge_px)
        clouds[n] = (P, pid)
        out = out_dir / f"{stem}_A{n}_{tag}.pcd"
        write_pcd_xyzrgb(out, P, np.tile(COLORS[n], (len(P), 1)), pid, np.full(len(P), n))
        print(f"Approach {n}: {len(u)} dots → {len(P)} points   {out}")

    if combined and len(clouds) > 1:
        ns = list(clouds)
        out = out_dir / f"{stem}_A{''.join(map(str, ns))}_{tag}.pcd"
        write_pcd_xyzrgb(out, np.vstack([clouds[n][0] for n in ns]),
                         np.vstack([np.tile(COLORS[n], (len(clouds[n][0]), 1)) for n in ns]),
                         np.concatenate([clouds[n][1] for n in ns]),
                         np.concatenate([np.full(len(clouds[n][0]), n) for n in ns]))
        print(f"Combined ({', '.join(f'A{n}' for n in ns)}): {out}")
    return clouds


def plot_reconstruction(clouds, title="3-D reconstruction", max_points=20000):
    """Subsampled 3-D scatter of the reconstructed clouds, coloured per approach."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    rng = np.random.default_rng(0)
    for n, (P, _) in clouds.items():
        sel = rng.choice(len(P), size=min(len(P), max_points), replace=False)
        ax.scatter(P[sel, 0], P[sel, 1], P[sel, 2], s=1, color=np.array(COLORS[n]) / 255, label=f"Approach {n}")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend()
    fig.tight_layout()
    return fig

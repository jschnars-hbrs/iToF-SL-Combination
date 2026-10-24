"""Dot calibration (Godbaz et al. 2025, §4.1/4.2) for a combined iToF + dot projector sensor."""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import Imath
import matplotlib.pyplot as plt
import OpenEXR
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from skimage import feature
from skimage.util import img_as_float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def sor_keep_mask(xyz, k=6, nsigma=1.0, z_range=(0.1, 5.0)):
    """CloudCompare-style statistical outlier removal; returns (keep mask, threshold [m], n_valid)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    z_lo, z_hi = z_range
    valid = (np.isfinite(xyz).all(axis=1) & np.any(xyz != 0.0, axis=1)
             & (xyz[:, 2] >= z_lo) & (xyz[:, 2] <= z_hi))
    keep = np.zeros(len(xyz), dtype=bool)
    idx = np.flatnonzero(valid)
    if idx.size <= k + 1:
        keep[idx] = True
        return keep, float("inf"), int(idx.size)
    pts = xyz[idx]
    d, _ = cKDTree(pts).query(pts, k=k + 1, workers=1)
    mean_d = d[:, 1:].mean(axis=1)
    thr = float(mean_d.mean() + nsigma * mean_d.std())
    keep[idx] = mean_d <= thr
    return keep, thr, int(idx.size)


def correct_light_path(P, B):
    """True points for ToF points P (N, 3) measured with the light source at B instead of the lens."""
    P = np.asarray(P, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = np.linalg.norm(P, axis=1)
        d = P / m[:, None]
        r = (4.0 * m * m - B @ B) / (4.0 * m - 2.0 * (d @ B))
    return d * r[:, None]


def cache_tag(sor=None, light_path_correction=False):
    """Name suffix for calibrations / caches, e.g. '_sor6-0.40_lpc'."""
    tag = f"_sor{sor[0]}-{sor[1]:.2f}" if sor else ""
    return tag + ("_lpc" if light_path_correction else "")


def plot_sor_effect(dotCal, tof_path, unit_scale=0.001):
    """Print and plot which pixels SOR removes from one ToF frame."""
    if not dotCal.sor:
        print("SOR disabled — the test frame is the raw cloud.")
        return None
    sor = dotCal.sor
    dotCal.sor = None
    try:
        raw = dotCal.load_tof_pcd(tof_path, unit_scale=unit_scale, depth_mode="axial")
    finally:
        dotCal.sor = sor
    fil = dotCal.load_tof_pcd(tof_path, unit_scale=unit_scale, depth_mode="axial")
    n_raw = int(np.isfinite(raw["distance"]).sum())
    n_fil = int(np.isfinite(fil["distance"]).sum())
    thr = sor_keep_mask(raw["points_3d"], *sor, z_range=dotCal.VALID_Z_RANGE)[1]
    print(f"SOR on {Path(tof_path).name}: {len(raw['distance'])} points, {n_raw} with a ToF return")
    print(f"  kept {n_fil}, rejected {n_raw - n_fil} ({100.0 * (n_raw - n_fil) / max(n_raw, 1):.1f}%)"
          f"   |   max mean-{sor[0]}NN distance = {thr * 1e3:.2f} mm")

    dr, df = raw["depth_map"], fil["depth_map"]
    lo, hi = np.nanpercentile(dr[np.isfinite(dr)], [1, 99])
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.4))
    for ax, img, title in ((axs[0], dr, "raw depth"),
                           (axs[1], df, f"after SOR ({sor[0]} pts, {sor[1]:.2f} σ)")):
        im = ax.imshow(img, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(title, fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="z [m]")
    removed = np.isfinite(dr) & ~np.isfinite(df)
    axs[2].imshow(dr, cmap="gray", vmin=lo, vmax=hi, interpolation="nearest")
    axs[2].imshow(np.ma.masked_where(~removed, removed), cmap="autumn",
                  interpolation="nearest", alpha=0.9)
    axs[2].set_title(f"removed pixels ({removed.sum()})", fontsize=10)
    axs[2].axis("off")
    fig.suptitle("Statistical outlier removal on the test cloud", fontsize=11)
    fig.tight_layout()
    return fig


class DotCalibration:
    """Dot detection, tracking, baseline / unit-vector calibration and 1-D dot trails."""

    SL_CHANNEL = "S0.940,000nm"  # EXR channel of the IR structured-light image
    VALID_Z_RANGE = (0.1, 5.0)   # [m] plausible axial ToF depth; the sensor writes absurd finite values otherwise
    sor = None                   # (k, nsigma) → statistical outlier removal in load_tof_pcd
    light_origin = None          # (3,) [m] ToF light source → light-path correction in load_tof_pcd

    _PCD_TYPE_MAP = {
        ("F", 4): np.float32, ("F", 8): np.float64,
        ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
        ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
    }
    _annulus_warned = set()

    @staticmethod
    def parse_distance_from_name(filename: str, offset: float = 0.0) -> Optional[float]:
        """Distance [m] from a file name ('SL_0.4m.exr'), or from the logbook for Pos0…Pos9 minus offset [mm]."""
        stem = os.path.splitext(os.path.basename(filename))[0]
        logbook_mm = {"Pos0": 402, "Pos1": 439, "Pos2": 486, "Pos3": 545, "Pos4": 621,
                      "Pos5": 720, "Pos6": 857, "Pos7": 1059, "Pos8": 1385, "Pos9": 2000}
        if stem in logbook_mm:
            return (logbook_mm[stem] - offset) / 1000.0
        m = re.search(r"([0-9]+(?:[.,_][0-9]+)?)m", stem, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).replace("_", ".").replace(",", "."))
        except ValueError:
            return None

    def read_image(self, image_path: str) -> np.ndarray:
        """Grayscale image; EXR files are read from the SL_CHANNEL."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if str(image_path).lower().endswith(".exr"):
            exr = OpenEXR.InputFile(str(image_path))
            dw = exr.header()["dataWindow"]
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            raw = exr.channel(self.SL_CHANNEL, Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def _parse_pcd_header(self, pcd_path: str) -> dict:
        """PCD header fields plus data offset."""
        meta = {}
        header_len = 0
        with open(pcd_path, "rb") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                header_len += 1
                s = line.decode("ascii", errors="ignore").strip()
                if not s or s.startswith("#"):
                    continue
                key, *rest = s.split()
                if key == "FIELDS":
                    meta["fields"] = rest
                elif key in ("SIZE", "COUNT"):
                    meta[key.lower()] = [int(v) for v in rest]
                elif key == "TYPE":
                    meta["type"] = rest
                elif key in ("WIDTH", "HEIGHT", "POINTS"):
                    meta[key.lower()] = int(rest[0])
                elif key == "DATA":
                    meta["data"] = rest[0].lower()
                    meta["data_offset"] = f.tell()
                    break
        meta["header_len"] = header_len
        if "data" not in meta:
            raise ValueError(f"PCD header incomplete – no DATA field in {pcd_path}")
        return meta

    def _pcd_binary_dtype(self, meta: dict) -> np.dtype:
        """Structured dtype of a binary PCD."""
        fields = meta["fields"]
        counts = meta.get("count", [1] * len(fields))
        dt = []
        for name, size, typ, count in zip(fields, meta["size"], meta["type"], counts):
            np_type = self._PCD_TYPE_MAP.get((typ.upper(), size))
            if np_type is None:
                raise ValueError(f"Unsupported PCD field type '{typ}{size}' for field '{name}'")
            dt.append((name, np_type) if count == 1 else (name, np_type, count))
        return np.dtype(dt)

    def load_tof_pcd(self, pcd_path: str, unit_scale: float = 0.001,
                     depth_mode: str = "radial",
                     z_range: Optional[Tuple[float, float]] = None) -> dict:
        """Load a ToF PCD (ascii/binary) with optional SOR and light-path correction; returns points, distance and maps."""
        meta = self._parse_pcd_header(pcd_path)
        fields = meta.get("fields")
        if not fields:
            raise ValueError(f"No FIELDS in PCD header: {pcd_path}")

        if meta["data"] == "ascii":
            df = pd.read_csv(pcd_path, sep=r"\s+", header=None, names=fields,
                             skiprows=meta["header_len"], engine="c").dropna(how="all")
            columns = {name: df[name].to_numpy() for name in fields}
        elif meta["data"] == "binary":
            n_points = meta.get("points", meta.get("width", 0) * meta.get("height", 1))
            raw = np.fromfile(pcd_path, dtype=self._pcd_binary_dtype(meta), count=n_points,
                              offset=meta["data_offset"])
            columns = {name: raw[name] for name in fields}
        else:
            raise NotImplementedError(f"Unsupported PCD DATA mode: {meta['data']!r}")
        if not all(c in columns for c in "xyz"):
            raise ValueError(f"PCD missing x/y/z (FIELDS={fields})")

        points = np.column_stack([columns["x"], columns["y"], columns["z"]]).astype(np.float64) * unit_scale
        z_range = self.VALID_Z_RANGE if z_range is None else z_range
        bad = ~np.all(np.isfinite(points), axis=1) | (points[:, 2] < z_range[0]) | (points[:, 2] > z_range[1])
        points[bad] = np.nan
        if self.sor:
            keep = sor_keep_mask(points, *self.sor, z_range=z_range)[0]
            points[~keep] = np.nan
        if self.light_origin is not None:
            points = correct_light_path(points, self.light_origin)

        if "grayValue" in columns:
            intensity = columns["grayValue"].astype(np.float64)
        elif "intensity" in columns:
            intensity = columns["intensity"].astype(np.float64)
        else:
            intensity = np.full(len(points), np.nan)

        if depth_mode == "radial":
            distance = np.linalg.norm(points, axis=1)
        elif depth_mode in ("z", "axial"):
            distance = points[:, 2]
        else:
            raise ValueError("depth_mode must be 'radial' or 'axial'")

        W, H = meta.get("width"), meta.get("height")
        tof_data = {"points_3d": points, "distance": distance, "intensity": intensity,
                    "width": W, "height": H}
        if W is not None and H is not None and W * H == len(points):
            tof_data["points_map"] = points.reshape((H, W, 3))
            tof_data["depth_map"] = distance.reshape((H, W))
            tof_data["intensity_map"] = intensity.reshape((H, W))
        return tof_data

    @staticmethod
    def _sample_points_map(points_map: np.ndarray, u: float, v: float,
                           z_range: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
        """Bilinear 3-D point at (u, v) over the valid corners of an (H, W, 3) map, or None."""
        H, W = points_map.shape[:2]
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        if not (0 <= u0 < W and 0 <= v0 < H):
            return None
        u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
        du, dv = u - u0, v - v0
        corners = points_map[[v0, v0, v1, v1], [u0, u1, u0, u1], :]
        weights = np.array([(1 - du) * (1 - dv), du * (1 - dv), (1 - du) * dv, du * dv])
        z_lo, z_hi = DotCalibration.VALID_Z_RANGE if z_range is None else z_range
        valid = (np.all(np.isfinite(corners), axis=1) & (np.linalg.norm(corners, axis=1) > 1e-6)
                 & (corners[:, 2] >= z_lo) & (corners[:, 2] <= z_hi))
        w_sum = np.sum(weights[valid])
        if not np.any(valid) or w_sum <= 1e-12:
            return None
        return weights[valid] @ corners[valid] / w_sum

    @staticmethod
    def _project_to_pixels(points: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Pinhole projection of camera-space points to (u, v)."""
        z = np.where(np.abs(points[:, 2]) > 1e-9, points[:, 2], 1e-9)
        return np.column_stack([K[0, 0] * points[:, 0] / z + K[0, 2],
                                K[1, 1] * points[:, 1] / z + K[1, 2]])

    @staticmethod
    def _sample_cloud_knn(points: np.ndarray, pixel_tree: cKDTree, u: float, v: float,
                          k: int = 4, max_px: float = 2.0) -> Optional[np.ndarray]:
        """Inverse-distance-weighted 3-D point at (u, v) from an unorganised cloud, or None."""
        dists, idx = pixel_tree.query([u, v], k=k)
        dists = np.atleast_1d(dists)
        idx = np.clip(np.atleast_1d(idx), 0, len(points) - 1)
        valid = np.isfinite(dists) & (dists <= max_px) & np.all(np.isfinite(points[idx]), axis=1)
        if not np.any(valid):
            return None
        w = 1.0 / np.maximum(dists[valid], 1e-6)
        return (w[:, None] * points[idx[valid]]).sum(axis=0) / w.sum()

    @staticmethod
    def _normalize_for_log(image: np.ndarray) -> np.ndarray:
        """Map an image to [0, 1] by its 1 / 99.9 percentiles so the LoG threshold is scale-free."""
        det = img_as_float(image).astype(np.float32)
        lo, hi = np.percentile(det, 1.0), np.percentile(det, 99.9)
        return np.clip((det - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)

    def detect_blobs(self, image_path: str, max_sigma: int = 30, num_sigma: int = 10, min_sigma: int = 5,
                     threshold: float = 0.05, visualize: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """LoG blob detection; returns blobs (y, x, radius) and the image."""
        image = self.read_image(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image for LoG blob detection: {image_path}")
        detect_img = self._normalize_for_log(image)
        blobs = feature.blob_log(detect_img, max_sigma=max_sigma, num_sigma=num_sigma,
                                 min_sigma=min_sigma, threshold=threshold, exclude_border=True)
        blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)
        if visualize:
            fig, ax = plt.subplots()
            ax.imshow(detect_img, cmap="gray")
            for y, x, r in blobs:
                ax.add_patch(plt.Circle((x, y), r, color="red", linewidth=1, fill=False))
            plt.show()
        return blobs, image

    def _gpr_peak(self, patch: np.ndarray, sigma_blob: Optional[float] = None,
                  coarse_oversample: int = 4, max_train: int = 650) -> Tuple[float, float]:
        """GPR subpixel dot centre (x, y) in patch coordinates (Godbaz §4.1), length scale pinned at dot size."""
        h, w = patch.shape
        y_raw = patch.astype(float)
        bg = float(np.percentile(y_raw, 20))
        amp = float(y_raw.max() - bg)
        if amp <= 0 or h < 3 or w < 3:
            return (w - 1) / 2.0, (h - 1) / 2.0
        I = np.clip((y_raw - bg) / amp, 0.0, None)

        if sigma_blob is None or not np.isfinite(sigma_blob) or sigma_blob <= 0:
            sigma_blob = max(h, w) / 4.0
        ls0 = float(np.clip(0.35 * sigma_blob, 1.0, max(h, w)))

        stride = max(1, int(np.ceil(np.sqrt(h * w / max_train))))
        syy, sxx = np.mgrid[0:h:stride, 0:w:stride]
        X = np.column_stack([sxx.ravel(), syy.ravel()]).astype(float)
        t = I[::stride, ::stride].ravel()
        kernel = (ConstantKernel(1.0, "fixed") * RBF(length_scale=ls0, length_scale_bounds="fixed")
                  + WhiteKernel(noise_level=5e-2, noise_level_bounds=(1e-5, 1.0)))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        gp.fit(X, t)

        gx, gy = np.meshgrid(np.linspace(0, w - 1, w * coarse_oversample),
                             np.linspace(0, h - 1, h * coarse_oversample))
        X_fine = np.column_stack([gx.ravel(), gy.ravel()])
        pred = gp.predict(X_fine)
        bg_model = float(np.percentile(pred, 10))
        amp_model = float(pred.max()) - bg_model
        centroids = []
        for level in (0.35, 0.5, 0.65):
            mask = pred >= bg_model + level * amp_model
            if np.any(mask):
                centroids.append((float(X_fine[mask, 0].mean()), float(X_fine[mask, 1].mean())))
        if not centroids:
            k = int(np.argmax(pred))
            return float(X_fine[k, 0]), float(X_fine[k, 1])
        c = np.mean(np.asarray(centroids), axis=0)
        return float(np.clip(c[0], 0, w - 1)), float(np.clip(c[1], 0, h - 1))

    def _radial_symmetry_center(self, patch: np.ndarray, eps: float = 1e-9) -> Tuple[float, float]:
        """Radial-symmetry centre (x, y) in patch coordinates."""
        I = patch.astype(np.float64) - np.median(patch)
        I[I < 0] = 0
        h, w = I.shape
        gy, gx = np.gradient(I)
        mag = np.hypot(gx, gy) + eps
        ux, uy = gx / mag, gy / mag
        yy, xx = np.mgrid[0:h, 0:w]
        mask = mag > np.percentile(mag, 70)
        ux, uy = ux[mask], uy[mask]
        x, y = xx[mask].astype(float), yy[mask].astype(float)
        W = mag[mask].astype(float)
        A = np.column_stack([-uy, ux])
        b = -uy * x + ux * y
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        return float(np.clip(sol[0], 0, w - 1)), float(np.clip(sol[1], 0, h - 1))

    def _geometric_centroid(self, patch: np.ndarray, bg_percentile: int = 20,
                            frac: float = 0.35, peak_percentile: int = 99,
                            connectivity: int = 8) -> Tuple[float, float]:
        """Centroid (x, y) of the thresholded region connected to the brightest pixel."""
        h, w = patch.shape
        values = patch.astype(float)
        y_peak, x_peak = np.unravel_index(int(np.argmax(values)), values.shape)
        bg = float(np.percentile(values, bg_percentile))
        pk = float(np.percentile(values, peak_percentile))
        mask = values >= bg + frac * (pk - bg)
        if not mask[y_peak, x_peak]:
            ys, xs = np.nonzero(mask)
            k = int(np.argmin((xs - x_peak) ** 2 + (ys - y_peak) ** 2))
            y_peak, x_peak = int(ys[k]), int(xs[k])

        neigh = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if connectivity == 8:
            neigh += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        visited = np.zeros_like(mask)
        visited[y_peak, x_peak] = True
        stack, comp = [(y_peak, x_peak)], []
        while stack:
            cy, cx = stack.pop()
            comp.append((cx, cy))
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if mask[ny, nx]:
                        stack.append((ny, nx))
        cx, cy = np.mean(comp, axis=0)
        return float(cx), float(cy)

    def _subpixel_refine(self, patch: np.ndarray, mode: str,
                         sigma_blob: Optional[float] = None) -> Tuple[float, float]:
        """Subpixel centre (x, y) in patch coordinates: 'GPR' | 'center' | 'geometricCenter' | 'radial'."""
        if mode == "GPR":
            return self._gpr_peak(patch, sigma_blob=sigma_blob)
        if mode == "center":
            return patch.shape[1] / 2.0, patch.shape[0] / 2.0
        if mode == "geometricCenter":
            return self._geometric_centroid(patch)
        if mode == "radial":
            return self._radial_symmetry_center(patch)
        raise ValueError(f"Unknown subpixel mode: {mode}")

    def detect_subpixel_locations(self, all_blobs: np.ndarray, image: np.ndarray,
                                  mode: str = "GPR") -> Tuple[List[np.ndarray], List[dict]]:
        """Subpixel {x, y} per blob (no IDs yet) plus the patches used."""
        H, W = image.shape[:2]
        patches, subpixels = [], []
        for blob in np.asarray(all_blobs, dtype=float):
            y_c, x_c = int(blob[0]), int(blob[1])
            r = int(np.ceil(blob[2]))
            y0, y1 = max(0, y_c - r), min(H, y_c + r + 1)
            x0, x1 = max(0, x_c - r), min(W, x_c + r + 1)
            patch = image[y0:y1, x0:x1]
            x_sub, y_sub = self._subpixel_refine(patch, mode, sigma_blob=float(blob[2]) / np.sqrt(2.0))
            patches.append(patch)
            subpixels.append({"x": float(x0 + x_sub), "y": float(y0 + y_sub)})
        return patches, subpixels

    @staticmethod
    def _as_baseline_vector(baseline_guess) -> np.ndarray:
        """Baseline guess as a (3,) vector; a scalar means an x offset."""
        B = np.atleast_1d(np.asarray(baseline_guess, dtype=float)).ravel()
        if B.size == 1:
            return np.array([float(B[0]), 0.0, 0.0])
        if B.size != 3:
            raise ValueError(f"baseline_guess must be a scalar or 3-vector, got shape {B.shape}")
        return B

    def tx_space_coords(self, subpixel_list: List[List[dict]], tof_paths: List[str],
                        K: np.ndarray, baseline_guess, pcd_unit_scale: float = 0.001) -> List[List[dict]]:
        """Parallax-invariant transmitter-space coordinates of every detection (Godbaz §4.1)."""
        K_inv = np.linalg.inv(K)
        B_guess = self._as_baseline_vector(baseline_guess)
        coords_list = []
        for spx, tof_path in zip(subpixel_list, tof_paths):
            sampler = ToFSampler(self.load_tof_pcd(tof_path, unit_scale=pcd_unit_scale), K)
            coords = []
            for d in spx:
                P = sampler.point_at(d["x"], d["y"])
                Q = self.cam_ray(d["x"], d["y"], K_inv) if P is None else P - B_guess
                z = Q[2] if abs(Q[2]) > 1e-9 else 1e-9
                coords.append({"x": Q[0] / z, "y": Q[1] / z})
            coords_list.append(coords)
        return coords_list

    @staticmethod
    def track_dots_across_distances(subpixel_list_unordered: List[List[dict]],
                                    ref_idx: int = 0,
                                    threshold_factor: float = 0.5,
                                    coords_list: Optional[List[List[dict]]] = None,
                                    travel_mode: str = "x") -> List[List[dict]]:
        """Stable dot IDs across distances via greedy nearest-neighbour matching (Godbaz §4.1)."""
        n_dist = len(subpixel_list_unordered)
        if n_dist == 0:
            return []
        if coords_list is None:
            coords_list = subpixel_list_unordered
        out = [[] for _ in range(n_dist)]
        out_coords = [[] for _ in range(n_dist)]

        def register(dist_idx, det_idx, dot_id):
            d = subpixel_list_unordered[dist_idx][det_idx]
            c = coords_list[dist_idx][det_idx]
            out[dist_idx].append({"id": dot_id, "x": float(d["x"]), "y": float(d["y"])})
            out_coords[dist_idx].append((float(c["x"]), float(c["y"])))

        def median_nn(pts):
            if pts.shape[0] < 2:
                return np.inf
            return float(np.median(cKDTree(pts).query(pts, k=2)[0][:, 1]))

        def step(prev_idx, new_idx, next_id):
            prev_ided = out[prev_idx]
            n_new = len(subpixel_list_unordered[new_idx])
            new_to_prev = -np.ones(n_new, dtype=int)
            if prev_ided and n_new:
                prev_xy = np.array(out_coords[prev_idx], float)
                new_xy = np.array([[c["x"], c["y"]] for c in coords_list[new_idx]], float)
                thresh = threshold_factor * median_nn(new_xy)
                k = min(3, len(prev_xy))
                dists, idxs = cKDTree(prev_xy).query(new_xy, k=k)
                dists, idxs = dists.reshape(n_new, k), idxs.reshape(n_new, k)
                pairs = sorted(((dists[n, c], n, int(idxs[n, c])) for n in range(n_new) for c in range(k)),
                               key=lambda t: t[0])
                prev_taken = np.zeros(len(prev_ided), dtype=bool)
                for dist, n, p in pairs:
                    if dist > thresh:
                        break
                    if new_to_prev[n] < 0 and not prev_taken[p]:
                        new_to_prev[n] = p
                        prev_taken[p] = True
            for n in range(n_new):
                if new_to_prev[n] >= 0:
                    register(new_idx, n, prev_ided[new_to_prev[n]]["id"])
                else:
                    register(new_idx, n, next_id)
                    next_id += 1
            return next_id

        seed = subpixel_list_unordered[ref_idx]
        for i, det_idx in enumerate(sorted(range(len(seed)), key=lambda k: (seed[k]["y"], seed[k]["x"]))):
            register(ref_idx, det_idx, i)
        next_id = len(seed)
        for j in range(ref_idx + 1, n_dist):
            next_id = step(j - 1, j, next_id)
        for j in range(ref_idx - 1, -1, -1):
            next_id = step(j + 1, j, next_id)

        travel_axis = 0 if travel_mode == "x" else 1
        cluster_axis = 1 - travel_axis
        sums = np.zeros((next_id, 2))
        counts = np.zeros(next_id, dtype=int)
        for spx in out:
            for d in spx:
                sums[d["id"]] += (d["x"], d["y"])
                counts[d["id"]] += 1
        means = sums / np.maximum(counts, 1)[:, None]
        spacing = float(np.median(cKDTree(means).query(means, k=2)[0][:, 1])) if next_id >= 2 else 1.0

        rows, current, prev_c = [], [], None
        for old_id in np.argsort(means[:, cluster_axis], kind="stable"):
            c = means[old_id, cluster_axis]
            if prev_c is not None and c - prev_c > 0.5 * spacing:
                rows.append(current)
                current = []
            current.append(int(old_id))
            prev_c = c
        if current:
            rows.append(current)
        new_order = [i for row in rows for i in sorted(row, key=lambda i: means[i, travel_axis])]
        old_to_new = np.empty(next_id, dtype=int)
        old_to_new[new_order] = np.arange(next_id)

        for spx in out:
            for d in spx:
                d["id"] = int(old_to_new[d["id"]])
            spx.sort(key=lambda d: d["id"])
        return out

    @staticmethod
    def _robust_linfit(x: np.ndarray, y: np.ndarray, iters: int = 3, n_sigma: float = 2.5) -> Tuple[float, float]:
        """Sigma-clipped least-squares line y = slope·x + intercept."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        slope = intercept = float("nan")
        for _ in range(max(1, iters)):
            if x.size < 2:
                break
            (slope, intercept), *_ = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)
            r = y - (slope * x + intercept)
            sd = float(np.std(r))
            if sd < 1e-12:
                break
            keep = np.abs(r) <= n_sigma * sd
            if keep.all() or keep.sum() < 3:
                break
            x, y = x[keep], y[keep]
        return float(slope), float(intercept)

    @staticmethod
    def detect_travel_mode(subpixel_list: List[List[dict]], cal_dists: List[float], K: np.ndarray,
                           mode_override: Optional[str] = None) -> dict:
        """Travel direction (x/y) and a full 3-D baseline guess from the dots' affine parallax in w = 1/z."""
        n_dist = len(subpixel_list)
        if n_dist < 2:
            raise ValueError("Need at least two calibration distances to detect the travel mode.")
        pos = [{int(d["id"]): (float(d["x"]), float(d["y"])) for d in spx} for spx in subpixel_list]
        w = 1.0 / np.asarray(cal_dists, dtype=float)

        du_dw, dv_dw, u_inf, v_inf = [], [], [], []
        for i in sorted(set().union(*[p.keys() for p in pos])):
            present = [j for j in range(n_dist) if i in pos[j]]
            if len(present) < 2:
                continue
            us = np.array([pos[j][i][0] for j in present])
            vs = np.array([pos[j][i][1] for j in present])
            ws = w[present]
            if np.ptp(ws) < 1e-9:
                continue
            if len(present) >= 3:
                A = np.column_stack([ws, np.ones_like(ws)])
                (au, bu), *_ = np.linalg.lstsq(A, us, rcond=None)
                (av, bv), *_ = np.linalg.lstsq(A, vs, rcond=None)
            else:
                dw = ws[-1] - ws[0]
                au, av = (us[-1] - us[0]) / dw, (vs[-1] - vs[0]) / dw
                bu, bv = us[0] - au * ws[0], vs[0] - av * ws[0]
            du_dw.append(float(au))
            dv_dw.append(float(av))
            u_inf.append(float(bu))
            v_inf.append(float(bv))
        if not du_dw:
            raise ValueError("No dot was tracked across two or more distances.")

        med_du, med_dv = float(np.median(du_dw)), float(np.median(dv_dw))
        mode = mode_override or ("x" if abs(med_du) >= abs(med_dv) else "y")
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

        n_fit = len(du_dw)
        B_vec = None
        if n_fit >= 8 and np.ptp(u_inf) > 0.25 * abs(fx) and np.ptp(v_inf) > 0.25 * abs(fy):
            su, bu = DotCalibration._robust_linfit(np.array(u_inf) - cx, np.array(du_dw))
            sv, bv = DotCalibration._robust_linfit(np.array(v_inf) - cy, np.array(dv_dw))
            if np.all(np.isfinite([su, bu, sv, bv])):
                B_vec = np.array([bu / fx, bv / fy, -0.5 * (su + sv)])
                if np.linalg.norm(B_vec) > 0.5:
                    B_vec = None

        if B_vec is None:
            B_axis = med_du / fx if mode == "x" else med_dv / fy
            B_vec = np.array([B_axis, 0.0, 0.0]) if mode == "x" else np.array([0.0, B_axis, 0.0])
        else:
            B_axis = B_vec[0] if mode == "x" else B_vec[1]
        return {"mode": mode, "B_axis_est": float(B_axis), "B_guess_vec": B_vec,
                "du_dw": med_du, "dv_dw": med_dv, "B_z_est": float(B_vec[2]), "n_fit": int(n_fit)}

    @classmethod
    def _fit_annulus_to_spacing(cls, subpixels: List[dict], cfg: dict,
                                max_frac: float = 0.45, warn_tag: str = "") -> dict:
        """Shrink the annulus so r_out stays below max_frac of the dot spacing."""
        ring = dict(cfg)
        r_in = float(ring.get("r_in", 8.0))
        r_out = float(ring.get("r_out", 14.0))
        if len(subpixels) < 2:
            return ring
        P = np.array([[d["x"], d["y"]] for d in subpixels], float)
        spacing = float(np.median(cKDTree(P).query(P, k=2)[0][:, 1]))
        r_max = max_frac * spacing
        if r_out > r_max:
            scale = r_max / r_out
            ring["r_in"], ring["r_out"] = r_in * scale, r_out * scale
            key = (round(spacing, 1), round(r_in, 1), round(r_out, 1))
            if key not in cls._annulus_warned:
                cls._annulus_warned.add(key)
                print(f"  [annulus] dot spacing {spacing:.1f} px is too tight for r_out={r_out:.1f}; "
                      f"shrinking ring to {ring['r_in']:.1f}–{ring['r_out']:.1f} px"
                      + (f"  ({Path(warn_tag).name})" if warn_tag else ""))
        return ring

    @staticmethod
    def annulus_axial_z(z_map: np.ndarray, u: float, v: float,
                        r_in: float = 8.0, r_out: float = 14.0, min_valid: int = 8) -> float:
        """Median axial ToF depth over a ring around (u, v) that skips the saturated dot itself."""
        H, W = z_map.shape[:2]
        ui, vi = int(round(u)), int(round(v))
        r = int(np.ceil(r_out))
        u0, u1 = max(0, ui - r), min(W, ui + r + 1)
        v0, v1 = max(0, vi - r), min(H, vi + r + 1)
        if u1 <= u0 or v1 <= v0:
            return float("nan")
        sub = z_map[v0:v1, u0:u1]
        yy, xx = np.mgrid[v0 - vi:v1 - vi, u0 - ui:u1 - ui]
        rr = np.hypot(xx, yy)
        m = (rr >= r_in) & (rr <= r_out) & np.isfinite(sub)
        if int(m.sum()) < min_valid:
            return float("nan")
        return float(np.median(sub[m]))

    def backproject_calibration_dots(self, subpixel_list: List[List[dict]], tof_paths: List[str],
                                     K: np.ndarray, pcd_unit_scale: float = 0.001,
                                     pcd_depth_mode: str = "radial", baseline_guess=3.8e-2,
                                     tof_sample: Optional[dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """3-D calibration points U (n_dots, n_dist, 3) from the ToF cloud at each dot, plus U_tx = U − B_guess."""
        all_ids = [int(d["id"]) for spx in subpixel_list for d in spx]
        if not all_ids:
            raise ValueError("subpixel_list contains no IDed dots.")
        U = np.full((max(all_ids) + 1, len(subpixel_list), 3), np.nan)
        cfg = dict(tof_sample or {})
        mode = cfg.pop("mode", "center")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for j, subpixels in enumerate(subpixel_list):
            tof_data = self.load_tof_pcd(tof_paths[j], unit_scale=pcd_unit_scale, depth_mode=pcd_depth_mode)
            if mode == "annulus" and "points_map" in tof_data:
                z_map = tof_data["points_map"][..., 2]
                ring = self._fit_annulus_to_spacing(subpixels, cfg, warn_tag=tof_paths[j])
                for dot in subpixels:
                    z = self.annulus_axial_z(z_map, dot["x"], dot["y"], **ring)
                    if np.isfinite(z):
                        U[dot["id"], j] = z * np.array([(dot["x"] - cx) / fx, (dot["y"] - cy) / fy, 1.0])
                continue
            sampler = ToFSampler(tof_data, K)
            for dot in subpixels:
                P = sampler.point_at(dot["x"], dot["y"])
                if P is not None:
                    U[dot["id"], j] = P

        return U, U - self._as_baseline_vector(baseline_guess)[None, None, :]

    @staticmethod
    def angles_to_unit_vector(theta: float, phi: float) -> np.ndarray:
        """Spherical angles (θ, φ) to a unit vector."""
        ct = np.cos(theta)
        return np.array([ct * np.cos(phi), ct * np.sin(phi), np.sin(theta)])

    @staticmethod
    def unit_vector_to_angles(v: np.ndarray) -> Tuple[float, float]:
        """Vector to spherical angles (θ, φ)."""
        v = v / (np.linalg.norm(v) + 1e-12)
        return float(np.arcsin(np.clip(v[2], -1.0, 1.0))), float(np.arctan2(v[1], v[0]))

    def estimate_baseline(self, U_tx: np.ndarray, outlier_sigma: Optional[float] = 3.0,
                          min_points: int = 3) -> Tuple[np.ndarray, int]:
        """Least-squares intersection of all dot lines in U_tx, with outlier-ray removal; returns (point, n_lines)."""
        def line_residuals(params, pts):
            v = self.angles_to_unit_vector(params[3], params[4])
            return np.concatenate([np.cross(v, p - params[:3]) for p in pts])

        def fit_line(pts):
            v_init = pts[-1] - pts[0]
            if np.linalg.norm(v_init) < 1e-9:
                v_init = np.array([1.0, 0.0, 0.0])
            x0 = np.array([*np.mean(pts, axis=0), *self.unit_vector_to_angles(v_init)])
            lb = [-np.inf, -np.inf, -np.inf, -np.pi / 2, -np.pi]
            ub = [np.inf, np.inf, np.inf, np.pi / 2, np.pi]
            res = least_squares(line_residuals, x0, args=(pts,), bounds=(lb, ub))
            v = self.angles_to_unit_vector(res.x[3], res.x[4])
            return res.x[:3], v / (np.linalg.norm(v) + 1e-12)

        def collect_lines(min_pts):
            lines = []
            for pts in U_tx:
                pts = pts[np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 1e-9)]
                if pts.shape[0] >= min_pts:
                    lines.append(fit_line(pts))
            return lines

        def intersect(line_set):
            A = np.zeros((3, 3))
            b = np.zeros(3)
            for p0, v in line_set:
                P = np.eye(3) - np.outer(v, v)
                A += P
                b += P @ p0
            return np.linalg.lstsq(A, b, rcond=None)[0]

        lines = collect_lines(max(min_points, 2))
        if len(lines) < 3 and min_points > 2:
            lines = collect_lines(2)
        intersection = intersect(lines)

        if outlier_sigma is not None and len(lines) >= 3:
            dists = np.array([np.linalg.norm((np.eye(3) - np.outer(v, v)) @ (p0 - intersection))
                              for p0, v in lines])
            med = np.median(dists)
            mad = 1.4826 * np.median(np.abs(dists - med))
            keep = dists <= med + outlier_sigma * max(mad, 1e-12)
            if np.any(~keep) and np.sum(keep) >= 2:
                lines = [l for l, k in zip(lines, keep) if k]
                intersection = intersect(lines)
        return intersection, len(lines)

    def estimate_unit_vectors(self, U: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Unit direction V[i] per dot, fitted to Q = U − B by minimising the angle error."""
        B = np.asarray(B, dtype=float)

        def residuals(params, Q):
            v = self.angles_to_unit_vector(params[0], params[1])
            return np.concatenate([np.cross(q, v) / (np.linalg.norm(q) + 1e-12) for q in Q])

        V = np.full((U.shape[0], 3), np.nan)
        for i, pts in enumerate(U):
            pts = pts[np.all(np.isfinite(pts), axis=1)]
            if pts.shape[0] < 2:
                continue
            Q = pts - B[None, :]
            m = np.mean(Q, axis=0)
            if np.linalg.norm(m) < 1e-9:
                m = np.array([1.0, 0.0, 0.0])
            res = least_squares(residuals, self.unit_vector_to_angles(m), args=(Q,),
                                bounds=([-np.pi / 2, -np.pi], [np.pi / 2, np.pi]))
            v = self.angles_to_unit_vector(res.x[0], res.x[1])
            V[i] = v / (np.linalg.norm(v) + 1e-12)
        return V

    @staticmethod
    def cam_ray(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
        """Unit camera ray through pixel (u, v)."""
        r = K_inv @ np.array([u, v, 1.0])
        return r / (np.linalg.norm(r) + 1e-12)

    @staticmethod
    def trail_line_params(v_i: np.ndarray, B: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Affine trail pixel(w) = p_inf + w·g (w = 1/z) of one dot ray, or None."""
        if not np.all(np.isfinite(v_i)):
            return None
        v = np.asarray(v_i, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        if abs(v[2]) < 1e-9:
            return None
        B = np.asarray(B, float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_inf = np.array([fx * v[0] / v[2] + cx, fy * v[1] / v[2] + cy])
        g = np.array([fx * (B[0] - B[2] * v[0] / v[2]), fy * (B[1] - B[2] * v[1] / v[2])])
        return p_inf, g

    @staticmethod
    def _sample_trail(p_inf, g, z_min, step_px, image_shape, margin) -> Optional[dict]:
        """Trail samples {u, v, w, z} from z_min to infinity at ~step_px, trimmed to the image."""
        w_max = 1.0 / float(z_min)
        n = max(2, int(np.ceil(float(np.linalg.norm(g)) * w_max / step_px)) + 1)
        w = np.linspace(w_max, 0.0, n)
        u = p_inf[0] + w * g[0]
        v = p_inf[1] + w * g[1]
        if image_shape is not None:
            H, W = image_shape
            inside = (u >= -margin) & (u <= W - 1 + margin) & (v >= -margin) & (v <= H - 1 + margin)
            if not np.any(inside):
                return None
            idx = np.nonzero(inside)[0]
            sl = slice(idx[0], idx[-1] + 1)
            u, v, w = u[sl], v[sl], w[sl]
        with np.errstate(divide="ignore"):
            z = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), np.inf)
        return {"u": u, "v": v, "w": w, "z": z}

    def build_dot_trails_1d(self, V: np.ndarray, B: np.ndarray, K: np.ndarray,
                            z_min: float = 0.3, step_px: float = 1.0,
                            image_shape: Optional[Tuple[int, int]] = None,
                            margin: float = 2.0) -> List[Optional[dict]]:
        """Analytic 1-D dot trails from V, B and K (Godbaz §4.2)."""
        trails = []
        for v_i in V:
            lp = self.trail_line_params(v_i, B, K)
            trails.append(None if lp is None else self._sample_trail(*lp, z_min, step_px, image_shape, margin))
        return trails

    def fit_trails_from_detections(self, subpixel_list: List[List[dict]], U: np.ndarray,
                                   z_min: float = 0.3, step_px: float = 1.0,
                                   image_shape: Optional[Tuple[int, int]] = None,
                                   margin: float = 2.0, min_obs: int = 3,
                                   V: Optional[np.ndarray] = None,
                                   B: Optional[np.ndarray] = None,
                                   K: Optional[np.ndarray] = None) -> List[Optional[dict]]:
        """1-D dot trails fitted to the calibration detections (absorbs lens distortion); analytic fallback."""
        pos_by_id: List[Dict[int, Tuple[float, float]]] = [
            {int(d["id"]): (float(d["x"]), float(d["y"])) for d in spx} for spx in subpixel_list]
        analytic = None
        if V is not None and B is not None and K is not None:
            analytic = self.build_dot_trails_1d(V, B, K, z_min=z_min, step_px=step_px,
                                                image_shape=image_shape, margin=margin)
        trails = []
        for i in range(U.shape[0]):
            obs = [(1.0 / float(U[i, j, 2]), *pos_by_id[j][i]) for j in range(U.shape[1])
                   if np.isfinite(U[i, j, 2]) and U[i, j, 2] > 1e-6 and i in pos_by_id[j]]
            fallback = analytic[i] if analytic is not None else None
            if len(obs) < min_obs:
                trails.append(fallback)
                continue
            ws, us, vs = np.array(obs).T
            A = np.column_stack([np.ones(len(ws)), ws])
            cu = np.linalg.lstsq(A, us, rcond=None)[0]
            cv = np.linalg.lstsq(A, vs, rcond=None)[0]
            g = np.array([cu[1], cv[1]])
            if float(np.linalg.norm(g)) * (1.0 / float(z_min)) < 2.0:
                trails.append(fallback)
                continue
            trails.append(self._sample_trail(np.array([cu[0], cv[0]]), g, z_min, step_px, image_shape, margin))
        return trails

    def estimate_sigma_u_trails(self, subpixel_list: List[List[dict]], trails: List[Optional[dict]]) -> float:
        """Robust perpendicular residual σ_u [px] of all detections to their trail line."""
        lines = []
        for t in trails:
            if t is None or len(t["u"]) < 2:
                lines.append(None)
            else:
                lines.append((np.array([t["u"][-1], t["v"][-1]]),
                              np.array([t["u"][0] - t["u"][-1], t["v"][0] - t["v"][-1]])))
        res = []
        for spx in subpixel_list:
            for d in spx:
                lp = lines[int(d["id"])] if int(d["id"]) < len(lines) else None
                if lp is None or np.linalg.norm(lp[1]) < 1e-9:
                    continue
                p0, g = lp
                ghat = g / np.linalg.norm(g)
                r = np.array([float(d["x"]), float(d["y"])]) - p0
                res.append(float(np.linalg.norm(r - (r @ ghat) * ghat)))
        return float(1.4826 * np.median(res)) if res else float("nan")

    @staticmethod
    def estimate_sigma_tof(U: np.ndarray, min_dots: int = 5, expected_z: Optional[List[float]] = None,
                           rel_tol: float = 0.3) -> np.ndarray:
        """Table (z_median, σ_z) of the robust ToF spread per calibration distance, sorted by z."""
        rows = []
        for j in range(U.shape[1]):
            z = U[:, j, 2]
            z = z[np.isfinite(z) & (z > 1e-6)]
            if z.size < min_dots:
                continue
            med = float(np.median(z))
            if expected_z is not None and j < len(expected_z):
                exp = float(expected_z[j])
                if exp > 0 and abs(med - exp) > rel_tol * exp:
                    continue
            rows.append((med, float(1.4826 * np.median(np.abs(z - med)))))
        rows.sort()
        return np.array(rows, dtype=float).reshape(-1, 2)

    @staticmethod
    def plot_dot_trails(trails: List[Optional[dict]], image_shape: Optional[Tuple[int, int]] = None,
                        background: Optional[np.ndarray] = None,
                        title: str = "Precomputed 1-D dot trails (near → ∞)"):
        """Overlay all dot trails on the image plane; near end cyan, vanishing point red."""
        fig, ax = plt.subplots(figsize=(9, 7))
        if background is not None:
            ax.imshow(background, cmap="gray")
        for t in trails:
            if t is None:
                continue
            ax.plot(t["u"], t["v"], lw=0.8, alpha=0.8)
            ax.plot(t["u"][0], t["v"][0], marker=".", ms=4, color="cyan")
            ax.plot(t["u"][-1], t["v"][-1], marker=".", ms=3, color="red")
        if image_shape is not None:
            H, W = image_shape
            ax.set_xlim(0, W - 1)
            ax.set_ylim(H - 1, 0)
        elif background is None:
            ax.invert_yaxis()
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("u [px]")
        ax.set_ylabel("v [px]")
        return fig

    @staticmethod
    def sample_image_bilinear(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Vectorised bilinear image samples at (u, v); NaN outside the image."""
        H, W = image.shape[:2]
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        out = np.full(u.shape, np.nan)
        ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if not np.any(ok):
            return out
        uu, vv = u[ok], v[ok]
        u0, v0 = np.floor(uu).astype(int), np.floor(vv).astype(int)
        u1, v1 = np.minimum(u0 + 1, W - 1), np.minimum(v0 + 1, H - 1)
        du, dv = uu - u0, vv - v0
        img = image.astype(float)
        out[ok] = (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u1] * du * (1 - dv)
                   + img[v1, u0] * (1 - du) * dv + img[v1, u1] * du * dv)
        return out


class ToFSampler:
    """3-D ToF point lookup at pixel (u, v) for organised grids and unorganised clouds alike."""

    def __init__(self, tof_data: dict, K: np.ndarray):
        """Pick raster lookup for organised clouds, a pixel KD-tree otherwise."""
        self.points = tof_data["points_3d"]
        self.distance = tof_data["distance"]
        depth_map = tof_data.get("depth_map")
        self.organised = depth_map is not None and (tof_data.get("height") or 1) > 1
        if self.organised:
            self.depth_map = depth_map
            self.points_map = tof_data["points_map"]
            self.H, self.W = depth_map.shape
        else:
            keep = np.all(np.isfinite(self.points), axis=1)
            self.points = self.points[keep]
            self.distance = self.distance[keep]
            self.tree = cKDTree(DotCalibration._project_to_pixels(self.points, K))

    def point_at(self, u: float, v: float, k: int = 4, max_px: float = 2.0) -> Optional[np.ndarray]:
        """Interpolated 3-D point at (u, v), or None if none is close enough."""
        if self.organised:
            return DotCalibration._sample_points_map(self.points_map, u, v)
        return DotCalibration._sample_cloud_knn(self.points, self.tree, u, v, k=k, max_px=max_px)

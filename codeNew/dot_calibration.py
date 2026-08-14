import os
import re
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import OpenEXR
import Imath
from skimage import feature
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List, Dict


class DotCalibration:
    """
    Calibration and depth fusion for a combined iToF + dot projector sensor.

    Based on:
      - Godbaz et al. 2025 (Microsoft-Paper / FL_C): dot calibration, consistency
        error, active brightness trail
      - Agresti & Zanuttigh, ECCV 2018: maximum likelihood depth fusion
    """

    SL_CHANNEL = "S0.940,000nm"  # EXR channel to use for structured-light images (not visible in RGB Channel, due to IR-Light)

    # ─────────────────────────────────────────────────────────────────────────
    # File utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_distance_from_name(filename: str) -> Optional[float]:
        """
        Parse target distance in metres from a filename.

        Handles formats like:  SL_0.4m.exr, SL_ToF_1.2m.pcd, 3_6m_frame.png
        """
        m = re.search(r"([0-9]+(?:[.,_][0-9]+)?)m", os.path.basename(filename), re.IGNORECASE)
        if not m:
            return None
        s = m.group(1).replace("_", ".").replace(",", ".")
        try:
            return float(s)
        except ValueError:
            return None

    def load_images(self, folder_path: str, pattern: str = "_image_rendered.png") -> List[str]:
        """Return sorted list of image paths matching *pattern* inside *folder_path*."""
        image_paths = []
        for file in os.listdir(folder_path):
            full_path = os.path.join(folder_path, file)
            if os.path.isfile(full_path) and file.endswith(pattern):
                image_paths.append(full_path)

        image_paths.sort(key=lambda p: self.parse_distance_from_name(p) or 999.0)
        return image_paths

    def read_image(self, image_path: str) -> np.ndarray:
        """Read an image (EXR or PNG) and return it as a grayscale array.

        For EXR files, reads the channel named SL_CHANNEL by name via the OpenEXR
        library (avoids OpenCV's unreliable mapping of non-standard channel names).
        For all other formats, converts to grayscale via OpenCV.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if str(image_path).lower().endswith(".exr"):
            exr = OpenEXR.InputFile(str(image_path))
            dw = exr.header()['dataWindow']
            w = dw.max.x - dw.min.x + 1
            h = dw.max.y - dw.min.y + 1
            raw = exr.channel(self.SL_CHANNEL, Imath.PixelType(Imath.PixelType.FLOAT))
            return np.frombuffer(raw, dtype=np.float32).reshape(h, w)
        img = cv2.imread(image_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ─────────────────────────────────────────────────────────────────────────
    # ToF data loading
    # ─────────────────────────────────────────────────────────────────────────

    _PCD_TYPE_MAP = {
        ("F", 4): np.float32, ("F", 8): np.float64,
        ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
        ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
    }

    def _parse_pcd_header(self, pcd_path: str) -> dict:
        """Parse a PCD header (ASCII or binary), return metadata dict."""
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
                elif key == "SIZE":
                    meta["size"] = [int(v) for v in rest]
                elif key == "TYPE":
                    meta["type"] = rest
                elif key == "COUNT":
                    meta["count"] = [int(v) for v in rest]
                elif key == "WIDTH":
                    meta["width"] = int(rest[0])
                elif key == "HEIGHT":
                    meta["height"] = int(rest[0])
                elif key == "POINTS":
                    meta["points"] = int(rest[0])
                elif key == "DATA":
                    meta["data"] = rest[0].lower()
                    meta["data_offset"] = f.tell()
                    break
        meta["header_len"] = header_len
        if "data" not in meta:
            raise ValueError(f"PCD header incomplete – no DATA field in {pcd_path}")
        return meta

    def _pcd_binary_dtype(self, meta: dict) -> np.dtype:
        """Build a structured numpy dtype from a binary PCD header's FIELDS/SIZE/TYPE/COUNT."""
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
                     depth_mode: str = "radial") -> dict:
        """
        Load ToF data from an organised PCD (ASCII or binary) file.

        Parameters
        ----------
        pcd_path    : path to .pcd file
        unit_scale  : multiply xyz by this factor (0.001 converts mm → m)
        depth_mode  : "radial" = sqrt(x²+y²+z²); "axial" or "z" = z only

        Returns
        -------
        tof_data dict with keys:
          points_3d, distance, intensity, width, height
          + points_map (H,W,3), depth_map (H,W), intensity_map (H,W) for organised clouds
        """
        meta = self._parse_pcd_header(pcd_path)

        fields = meta.get("fields")
        if not fields:
            raise ValueError(f"No FIELDS in PCD header: {pcd_path}")

        if meta["data"] == "ascii":
            df = pd.read_csv(pcd_path, sep=r"\s+", header=None, names=fields,
                             skiprows=meta["header_len"], engine="c").dropna(how="all")
            columns = {name: df[name].to_numpy() for name in fields}
        elif meta["data"] == "binary":
            dtype = self._pcd_binary_dtype(meta)
            n_points = meta.get("points", meta.get("width", 0) * meta.get("height", 1))
            raw = np.fromfile(pcd_path, dtype=dtype, count=n_points, offset=meta["data_offset"])
            columns = {name: raw[name] for name in fields}
        else:
            raise NotImplementedError(
                f"Unsupported PCD DATA mode: {meta['data']!r} (expected 'ascii' or 'binary')")

        for c in ("x", "y", "z"):
            if c not in columns:
                raise ValueError(f"PCD missing column '{c}' (FIELDS={fields})")

        points = np.column_stack([columns["x"], columns["y"], columns["z"]]).astype(np.float64) * unit_scale

        if "grayValue" in columns:
            intensity = columns["grayValue"].astype(np.float64)
        elif "intensity" in columns:
            intensity = columns["intensity"].astype(np.float64)
        else:
            intensity = np.full(len(points), np.nan, dtype=np.float64)

        if depth_mode == "radial":
            distance = np.linalg.norm(points, axis=1)
        elif depth_mode in ("z", "axial"):
            distance = points[:, 2]
        else:
            raise ValueError("depth_mode must be 'radial' or 'axial'")

        tof_data = {
            "points_3d": points,
            "distance": distance,
            "intensity": intensity,
            "width": meta.get("width"),
            "height": meta.get("height"),
        }

        W, H = tof_data["width"], tof_data["height"]
        if W is not None and H is not None and W * H == len(points):
            tof_data["points_map"]   = points.reshape((H, W, 3))
            tof_data["depth_map"]    = distance.reshape((H, W))
            tof_data["intensity_map"] = intensity.reshape((H, W))

        return tof_data

    @staticmethod
    def _sample_points_map(points_map: np.ndarray, u: float, v: float) -> Optional[np.ndarray]:
        """Bilinearly sample a (H, W, 3) point map at subpixel (u, v).

        Corner points that are invalid (non-finite or zero, i.e. no ToF return —
        common between dots in sparsely illuminated clouds) are excluded and the
        bilinear weights renormalised over the valid corners.

        Returns the interpolated 3-D point, or None if out of bounds / no valid
        corner.
        """
        H, W = points_map.shape[:2]
        u0, v0 = int(np.floor(u)), int(np.floor(v))
        if not (0 <= u0 < W and 0 <= v0 < H):
            return None
        u1, v1 = min(u0 + 1, W - 1), min(v0 + 1, H - 1)
        du, dv = u - u0, v - v0

        corners = points_map[[v0, v0, v1, v1], [u0, u1, u0, u1], :]
        weights = np.array([(1 - du) * (1 - dv), du * (1 - dv),
                            (1 - du) * dv, du * dv])
        valid = np.all(np.isfinite(corners), axis=1) \
            & (np.linalg.norm(corners, axis=1) > 1e-6)
        w_sum = np.sum(weights[valid])
        if not np.any(valid) or w_sum <= 1e-12:
            return None
        P = weights[valid] @ corners[valid] / w_sum
        return P

    @staticmethod
    def _project_to_pixels(points: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Project camera-space 3-D points to pixel (u, v) via pinhole intrinsics K."""
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        z = np.where(np.abs(points[:, 2]) > 1e-9, points[:, 2], 1e-9)
        u = fx * points[:, 0] / z + cx
        v = fy * points[:, 1] / z + cy
        return np.column_stack([u, v])

    @staticmethod
    def _sample_cloud_knn(points: np.ndarray, pixel_tree: cKDTree, u: float, v: float,
                           k: int = 4, max_px: float = 2.0) -> Optional[np.ndarray]:
        """Inverse-distance-weighted 3-D point at pixel (u, v) from an unorganised
        cloud, via a KD-tree built over the cloud's projected pixel coordinates.

        Used when the PCD only lists valid returns (no fixed (H, W) raster, e.g.
        the current Schmersal binary export: HEIGHT 1, invalid pixels dropped),
        so `_sample_points_map`'s grid-index bilinear lookup doesn't apply.

        Returns the interpolated 3-D point, or None if no cloud point is within
        `max_px` pixels of (u, v).
        """
        dists, idx = pixel_tree.query([u, v], k=k)
        dists = np.atleast_1d(dists)
        idx = np.atleast_1d(idx)
        valid = np.isfinite(dists) & (dists <= max_px)
        if not np.any(valid):
            return None
        w = 1.0 / np.maximum(dists[valid], 1e-6)
        return (w[:, None] * points[idx[valid]]).sum(axis=0) / w.sum()

    def _tof_point_sampler(self, tof_data: dict, K: np.ndarray):
        """Return f(u, v) -> Optional[3-vector]: looks up the 3-D ToF point at
        pixel (u, v), whether the cloud is a genuine organised (H, W) grid
        (H > 1, e.g. simulated PBRT clouds) or an unorganised list of valid
        returns (H == 1, current real-sensor binary export). See `ToFSampler`.
        """
        sampler = ToFSampler(tof_data, K)
        return sampler.point_at

    # ─────────────────────────────────────────────────────────────────────────
    # Blob detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def add_gaussian_to_detected_blob(image: np.ndarray, blobs: np.ndarray) -> np.ndarray:
        """Add a synthetic 2-D Gaussian at each blob location to create Gaussian intensity profiles."""
        GAUSS_SIGMA = 2   # fixed narrow sigma in pixels
        GAUSS_R     = 4     # half-window size in pixels
        out = image.astype(np.float64).copy()
        H, W = out.shape[:2]
        for blob in blobs:
            y, x = blob[0], blob[1]
            y0, y1 = max(0, int(y) - GAUSS_R), min(H, int(y) + GAUSS_R + 1)
            x0, x1 = max(0, int(x) - GAUSS_R), min(W, int(x) + GAUSS_R + 1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            gauss = np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / GAUSS_SIGMA ** 2)
            vy = int(np.clip(round(y), 0, H - 1))
            vx = int(np.clip(round(x), 0, W - 1))
            peak = float(image[vy, vx])
            out[y0:y1, x0:x1] += peak * gauss
        return out

    def detect_blobs(self, image_path: str, max_sigma: int = 30, num_sigma: int = 10, min_sigma: int = 5,
                     threshold: float = 0.1, visualize: bool = False, add_synthetic_gaussian: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Laplacian-of-Gaussian blob detector (scikit-image).

        Returns
        -------
        blobs : (N, 3) array of (y, x, radius)
        image : grayscale image array (with synthetic Gaussians added if requested)
        """
        image = self.read_image(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image for LoG blob detection: {image_path}")

        blobs = feature.blob_log(image, max_sigma=max_sigma, num_sigma=num_sigma, min_sigma=min_sigma, threshold=threshold)
        blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)  # convert sigma to radius
        if add_synthetic_gaussian:
            image_out = self.add_gaussian_to_detected_blob(image, blobs)
        else:
            image_out = image

        if visualize:
            fig, ax = plt.subplots()
            ax.imshow(image_out, cmap="gray")
            for y, x, r in blobs:
                ax.add_patch(plt.Circle((x, y), r, color="red", linewidth=1, fill=False))
            plt.show()

        return blobs, image_out

    # ─────────────────────────────────────────────────────────────────────────
    # Subpixel localisation
    # ─────────────────────────────────────────────────────────────────────────

    def _gpr_peak(self, patch: np.ndarray, oversample: int = 10) -> Tuple[float, float]:
        """GPR-based subpixel peak within *patch*. Returns (x, y) relative to patch.

        Parameters
        ----------
        oversample : int
            Grid oversampling factor for sub-pixel resolution (e.g. 10 → 0.1 px).
            Higher values are more accurate but slower.
        """
        h, w = patch.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        X = np.column_stack([xx.ravel(), yy.ravel()])
        y = patch.ravel().astype(float)

        ls = max(h, w) / 3.0  # adaptive: scale with patch size
        kernel = 1.0 * RBF(length_scale=ls) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, normalize_y=True)
        gp.fit(X, y)

        # Evaluate on fine sub-pixel grid
        xx_f, yy_f = np.meshgrid(
            np.linspace(0, w - 1, w * oversample),
            np.linspace(0, h - 1, h * oversample)
        )
        X_fine = np.column_stack([xx_f.ravel(), yy_f.ravel()])
        y_fine = gp.predict(X_fine)
        idx = int(np.argmax(y_fine))
        return float(X_fine[idx, 0]), float(X_fine[idx, 1])

    def _radial_symmetry_center(self, patch: np.ndarray, eps: float = 1e-9) -> Tuple[float, float]:
        """Radial-symmetry subpixel centre. Returns (x, y) relative to patch."""
        I = patch.astype(np.float64)
        I = I - np.median(I)
        I[I < 0] = 0

        h, w = I.shape
        gy, gx = np.gradient(I)
        mag = np.hypot(gx, gy) + eps
        ux, uy = gx / mag, gy / mag

        yy, xx = np.mgrid[0:h, 0:w]
        mask = mag > np.percentile(mag, 70)
        ux, uy = ux[mask], uy[mask]
        x, y   = xx[mask].astype(float), yy[mask].astype(float)
        wgt    = (mag[mask] ** 2).astype(float)

        A  = np.column_stack([-uy, ux])
        b  = -uy * x + ux * y
        W  = np.sqrt(wgt)
        sol, *_ = np.linalg.lstsq(A * W[:, None], b * W, rcond=None)
        xc = float(np.clip(sol[0], 0, w - 1))
        yc = float(np.clip(sol[1], 0, h - 1))
        return xc, yc

    def _geometric_centroid(self, patch: np.ndarray, bg_percentile: int = 20,
                            frac: float = 0.35, peak_percentile: int = 99,
                            connectivity: int = 8) -> Tuple[float, float]:
        """Flood-fill centroid around the brightest spot. Returns (x, y) relative to patch."""
        h, w = patch.shape
        xx, yy = np.meshgrid(np.arange(w), np.arange(h))
        y_flat = patch.ravel().astype(float)

        idx_peak = int(np.argmax(y_flat))
        x_peak_i, y_peak_i = int(xx.ravel()[idx_peak]), int(yy.ravel()[idx_peak])

        bg  = float(np.percentile(y_flat, bg_percentile))
        pk  = float(np.percentile(y_flat, peak_percentile))
        thresh = bg + frac * (pk - bg)
        mask2d = patch.astype(float) >= thresh

        if not mask2d[y_peak_i, x_peak_i]:
            ys, xs = np.nonzero(mask2d)
            d2 = (xs - x_peak_i) ** 2 + (ys - y_peak_i) ** 2
            k  = int(np.argmin(d2))
            y_peak_i, x_peak_i = int(ys[k]), int(xs[k])

        neigh = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)] \
                if connectivity == 8 else [(-1,0),(1,0),(0,-1),(0,1)]

        visited = np.zeros_like(mask2d, dtype=bool)
        stack   = [(y_peak_i, x_peak_i)]
        visited[y_peak_i, x_peak_i] = True
        comp_xs, comp_ys = [], []

        while stack:
            cy, cx = stack.pop()
            if not mask2d[cy, cx]:
                continue
            comp_xs.append(cx)
            comp_ys.append(cy)
            for dy, dx in neigh:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                    visited[ny, nx] = True
                    if mask2d[ny, nx]:
                        stack.append((ny, nx))

        return float(np.mean(comp_xs)), float(np.mean(comp_ys))

    def _subpixel_refine(self, patch: np.ndarray, mode: str) -> Tuple[float, float]:
        """Dispatch subpixel refinement on *patch*. Returns (x, y) relative to patch.

        Modes: "GPR" | "center" | "geometricCenter" | "radial"
        """
        if mode == "GPR":
            return self._gpr_peak(patch)
        if mode == "center":
            return float(patch.shape[1]) / 2.0, float(patch.shape[0]) / 2.0
        if mode == "geometricCenter":
            return self._geometric_centroid(patch)
        if mode == "radial":
            return self._radial_symmetry_center(patch)
        raise ValueError(f"Unknown subpixel mode: {mode}")

    def detect_subpixel_locations(self, all_blobs: np.ndarray, image: np.ndarray,
                                  mode: str = "GPR") \
            -> Tuple[List[np.ndarray], List[dict]]:
        """
        Refine blob positions to subpixel accuracy.

        No fixed dot grid is assumed (FL_C / Microsoft-Paper §4.1: "all the dots
        are transformed") — IDs are assigned later by `track_dots_across_distances`.

        Returns
        -------
        patches    : list of image patches, one per detected blob (input order)
        subpixels  : list of dicts {x, y} (no IDs yet), in input order
        """
        H, W = image.shape[:2]
        blobs = np.asarray(all_blobs, dtype=float)
        patches, subpixels = [], []

        for blob in blobs:
            y_c, x_c = int(blob[0]), int(blob[1])
            r = int(np.ceil(blob[2]))
            y0, y1 = max(0, y_c - r), min(H, y_c + r + 1)
            x0, x1 = max(0, x_c - r), min(W, x_c + r + 1)
            patch = image[y0:y1, x0:x1]

            x_sub, y_sub = self._subpixel_refine(patch, mode)
            patches.append(patch)
            subpixels.append({"x": float(x0 + x_sub), "y": float(y0 + y_sub)})

        return patches, subpixels

    # Backward-compatible alias (old name used by code/ and earlier notebooks).
    detect_subpixel_locations_no_grid = detect_subpixel_locations

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 2b – Cross-distance tracker
    # (FL_C / Microsoft-Paper §4.1: "each dot is tracked across multiple
    #  distances using the nearest-neighbor search method"). No fixed grid.
    # ─────────────────────────────────────────────────────────────────────────

    def tx_space_coords(self, subpixel_list: List[List[dict]], tof_paths: List[str],
                        K: np.ndarray, baseline_guess: float,
                        pcd_unit_scale: float = 0.001) -> List[List[dict]]:
        """
        Transform detections into a transmitter-space representation (FL_C §4.1).

        The paper tracks dots in a transmitter-space representation that is
        "approximately parallax invariant": using the ToF depth at each detection,
        the 3-D point is computed, shifted by the baseline guess, and reduced to
        its ray direction as seen from the transmitter. The same dot keeps
        (approximately) the same direction at every distance.

        The direction is expressed in gnomonic coordinates (Q_x/Q_z, Q_y/Q_z) —
        i.e. a normalised image-plane projection from the transmitter — which is
        well-conditioned for forward-facing rays (spherical angles are not:
        φ = atan2(y, x) degenerates near the optical axis).

        Returns a list (per distance) of dicts {x, y} suitable as `coords_list`
        for `track_dots_across_distances`. Detections without a valid ToF depth
        fall back to the camera-ray direction (good approximation since the
        baseline ≪ distance).
        """
        K_inv = np.linalg.inv(K)
        B_guess = np.array([baseline_guess, 0.0, 0.0])
        coords_list = []
        for spx, tof_path in zip(subpixel_list, tof_paths):
            tof = self.load_tof_pcd(tof_path, unit_scale=pcd_unit_scale, depth_mode="radial")
            sample = self._tof_point_sampler(tof, K)
            coords = []
            for d in spx:
                P = sample(d["x"], d["y"])
                if P is None:
                    # Fallback: camera-ray direction (no depth needed).
                    Q = self.cam_ray(d["x"], d["y"], K_inv)
                else:
                    Q = P - B_guess
                z = Q[2] if abs(Q[2]) > 1e-9 else 1e-9
                coords.append({"x": Q[0] / z, "y": Q[1] / z})
            coords_list.append(coords)
        return coords_list

    @staticmethod
    def track_dots_across_distances(subpixel_list_unordered: List[List[dict]],
                                    ref_idx: int = 0,
                                    threshold_factor: float = 0.5,
                                    coords_list: Optional[List[List[dict]]] = None) \
            -> List[List[dict]]:
        """
        Assign stable IDs to dots by tracking them across calibration distances
        with a nearest-neighbor search.

        Algorithm (FL_C / Microsoft-Paper §4.1):
          1. Seed IDs at distance index `ref_idx` (default 0 = closest), ordered
             top-left → bottom-right in image space.
          2. Walk outward (forward and backward in the distance list). For each
             step, greedy-match each detection at the new distance to the closest
             already-IDed dot at the previous distance via cKDTree.
          3. Threshold per step: `threshold_factor × median(nearest-neighbor
             spacing)` of detections at the new distance — adaptive across
             sensor resolutions.
          4. Unmatched detections become NEW IDs (handles dots entering FOV at
             greater distances).
          5. IDed dots that aren't matched at the new distance are absent there
             (NaN downstream).

        Parameters
        ----------
        subpixel_list_unordered : list (one entry per distance) of dicts {x, y}
            As returned by `detect_subpixel_locations`.
        ref_idx          : seed distance index (default 0).
        threshold_factor : multiplier on median NN spacing (default 0.5).
        coords_list      : optional alternative matching coordinates with the same
            structure (e.g. transmitter-space angles from `tx_space_coords`,
            parallax invariant as in FL_C §4.1). Matching and thresholds then
            operate in that space; the OUTPUT always carries the pixel x/y.

        Returns
        -------
        subpixel_list : list (one entry per distance) of dicts {id, x, y}.
            IDs are stable across distances; counts may differ per distance;
            roster size = max(id) + 1.
        """
        n_dist = len(subpixel_list_unordered)
        if n_dist == 0:
            return []
        if coords_list is None:
            coords_list = subpixel_list_unordered

        out: List[List[dict]] = [[] for _ in range(n_dist)]
        # Matching coordinates of the IDed dots, parallel to `out`.
        out_coords: List[List[Tuple[float, float]]] = [[] for _ in range(n_dist)]

        def _register(dist_idx: int, det_idx: int, dot_id: int):
            d = subpixel_list_unordered[dist_idx][det_idx]
            c = coords_list[dist_idx][det_idx]
            out[dist_idx].append({"id": dot_id, "x": float(d["x"]), "y": float(d["y"])})
            out_coords[dist_idx].append((float(c["x"]), float(c["y"])))

        # ── Step 1: seed roster from ref_idx, ordered top-left → bottom-right
        seed = subpixel_list_unordered[ref_idx]
        seed_order = sorted(range(len(seed)), key=lambda k: (seed[k]["y"], seed[k]["x"]))
        for i, det_idx in enumerate(seed_order):
            _register(ref_idx, det_idx, i)

        next_id = len(seed_order)

        def _median_nn(pts_xy: np.ndarray) -> float:
            if pts_xy.shape[0] < 2:
                return np.inf
            tree = cKDTree(pts_xy)
            dists, _ = tree.query(pts_xy, k=2)  # k=1 is self
            return float(np.median(dists[:, 1]))

        def _step(prev_idx: int, new_idx: int, next_id: int) -> int:
            """Match detections at new_idx to IDed dots at prev_idx."""
            prev_ided = out[prev_idx]
            new_dets = subpixel_list_unordered[new_idx]
            if len(prev_ided) == 0 or len(new_dets) == 0:
                # Nothing to match against — register everything as new IDs.
                for n_i in range(len(new_dets)):
                    _register(new_idx, n_i, next_id)
                    next_id += 1
                return next_id

            prev_xy = np.array(out_coords[prev_idx], float)
            new_xy = np.array([[c["x"], c["y"]] for c in coords_list[new_idx]], float)

            thresh = threshold_factor * _median_nn(new_xy)
            if not np.isfinite(thresh):
                thresh = np.inf

            # Greedy assignment: sort all (new, prev) pairs by distance.
            tree = cKDTree(prev_xy)
            # k = min(3, len(prev_xy)) candidates per detection.
            k = min(3, len(prev_xy))
            dists, idxs = tree.query(new_xy, k=k)
            if k == 1:
                dists = dists[:, None]
                idxs = idxs[:, None]

            pairs = []
            for n_i in range(len(new_dets)):
                for c in range(k):
                    pairs.append((dists[n_i, c], n_i, int(idxs[n_i, c])))
            pairs.sort(key=lambda t: t[0])

            new_to_prev = -np.ones(len(new_dets), dtype=int)
            prev_taken = np.zeros(len(prev_ided), dtype=bool)
            for dist, n_i, p_i in pairs:
                if dist > thresh:
                    break
                if new_to_prev[n_i] >= 0 or prev_taken[p_i]:
                    continue
                new_to_prev[n_i] = p_i
                prev_taken[p_i] = True

            for n_i in range(len(new_dets)):
                if new_to_prev[n_i] >= 0:
                    dot_id = prev_ided[new_to_prev[n_i]]["id"]
                else:
                    dot_id = next_id
                    next_id += 1
                _register(new_idx, n_i, dot_id)
            return next_id

        # ── Step 2: walk forward from ref_idx
        for j in range(ref_idx + 1, n_dist):
            next_id = _step(j - 1, j, next_id)

        # ── Step 3: walk backward from ref_idx (only meaningful if ref_idx>0)
        for j in range(ref_idx - 1, -1, -1):
            next_id = _step(j + 1, j, next_id)

        # ── Step 4: renumber so IDs are top-left → bottom-right globally.
        # Pure (y, x) lexsort interleaves rows when dots within a row have
        # small y-jitter. Instead: cluster rows from y-gaps relative to the
        # median nearest-neighbor spacing, then sort each row by x.
        # (Renumbering always uses PIXEL coordinates, regardless of coords_list.)
        roster_size = next_id
        sums = np.zeros((roster_size, 2), dtype=float)
        counts = np.zeros(roster_size, dtype=int)
        for spx in out:
            for d in spx:
                i = int(d["id"])
                sums[i, 0] += d["x"]
                sums[i, 1] += d["y"]
                counts[i] += 1
        means = sums / np.maximum(counts, 1)[:, None]

        if roster_size >= 2:
            grid_spacing = float(np.median(
                cKDTree(means).query(means, k=2)[0][:, 1]
            ))
        else:
            grid_spacing = 1.0
        row_gap_thresh = 0.5 * grid_spacing

        order_by_y = np.argsort(means[:, 1], kind="stable")
        rows: List[List[int]] = []
        current: List[int] = []
        prev_y = None
        for old_id in order_by_y:
            y = means[old_id, 1]
            if prev_y is not None and (y - prev_y) > row_gap_thresh:
                rows.append(current)
                current = []
            current.append(int(old_id))
            prev_y = y
        if current:
            rows.append(current)

        old_to_new = np.empty(roster_size, dtype=int)
        new_id = 0
        for row in rows:
            for old_id in sorted(row, key=lambda i: means[i, 0]):
                old_to_new[old_id] = new_id
                new_id += 1

        for spx in out:
            for d in spx:
                d["id"] = int(old_to_new[int(d["id"])])
            spx.sort(key=lambda d: d["id"])

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 3 – 3-D back-projection
    # ─────────────────────────────────────────────────────────────────────────

    def backproject_calibration_dots(self, subpixel_list: List[List[dict]],
                                      tof_paths: List[str],
                                      K: np.ndarray,
                                      pcd_unit_scale: float = 0.001,
                                      pcd_depth_mode: str = "radial",
                                      baseline_guess: float = 3.8e-2) \
            -> Tuple[np.ndarray, np.ndarray]:
        """
        Back-project each detected dot position into 3-D using the ToF point map
        at that pixel, yielding calibration points U.

        Parameters
        ----------
        subpixel_list   : list (one entry per distance) of subpixel dicts {id, x, y}
        tof_paths       : list of PCD file paths, same order as subpixel_list —
                          either a genuine organised (H, W) grid, or an
                          unorganised list of valid returns (current real-sensor
                          binary export)
        K               : 3×3 camera intrinsics; used to project cloud points to
                          pixel space when the PCD is unorganised
        pcd_unit_scale  : scale for PCD xyz (default 0.001 → mm to m)
        pcd_depth_mode  : "radial" or "axial"
        baseline_guess  : initial x-offset guess for transmitter [m], used to compute U_tx

        Returns
        -------
        U    : (n_dots, n_dist, 3) 3-D calibration points in camera coordinates [m]
        U_tx : (n_dots, n_dist, 3) U shifted into approximate transmitter space
        """
        # Derive roster size from the maximum dot ID across all distances so
        # the array fits even when per-distance counts vary (FL_C tracking).
        all_ids = [int(d["id"]) for spx in subpixel_list for d in spx]
        if not all_ids:
            raise ValueError("subpixel_list contains no IDed dots.")
        n_dots = max(all_ids) + 1
        n_dist = len(subpixel_list)
        U = np.full((n_dots, n_dist, 3), np.nan, dtype=np.float64)

        for j, subpixels in enumerate(subpixel_list):
            tof_data = self.load_tof_pcd(tof_paths[j], unit_scale=pcd_unit_scale,
                                          depth_mode=pcd_depth_mode)
            sample = self._tof_point_sampler(tof_data, K)

            for dot in subpixels:
                P = sample(dot["x"], dot["y"])
                if P is not None:
                    U[dot["id"], j, :] = P

        B_guess = np.array([baseline_guess, 0.0, 0.0])
        U_tx = U - B_guess[None, None, :]
        return U, U_tx

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 4 – Baseline estimation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def angles_to_unit_vector(theta: float, phi: float) -> np.ndarray:
        """Convert spherical angles (θ, φ) to a 3-D unit vector."""
        ct = np.cos(theta)
        return np.array([ct * np.cos(phi), ct * np.sin(phi), np.sin(theta)], dtype=float)

    @staticmethod
    def unit_vector_to_angles(v: np.ndarray) -> Tuple[float, float]:
        """Convert a 3-D vector to spherical angles (θ, φ)."""
        v = v / (np.linalg.norm(v) + 1e-12)
        theta = float(np.arcsin(np.clip(v[2], -1.0, 1.0)))
        phi   = float(np.arctan2(v[1], v[0]))
        return theta, phi

    def estimate_baseline(self, U_tx: np.ndarray,
                          outlier_sigma: Optional[float] = None) \
            -> Tuple[np.ndarray, int]:
        """
        Estimate the transmitter position from U_tx (3-D points in approximate transmitter space).

        Fits a 3-D line through each dot's multi-distance samples, then finds the
        least-squares intersection of all lines.  The x-component of the intersection
        is the correction to the initial baseline guess:
        AB = baseline_guess + intersection[0].

        Parameters
        ----------
        U_tx : (n_dots, n_dist, 3) array of calibration points shifted by baseline_guess
        outlier_sigma : optional float, e.g. 3.0. If set, performs the FL_C §4.1
            outlier removal ("After removing outlier rays, the baseline AB is found
            by an additional least squares optimization"): after a first pass,
            dot lines whose distance to the intersection exceeds
            median + outlier_sigma · MAD are discarded and the intersection refit.

        Returns
        -------
        intersection : (3,) intersection point in U_tx space
        n_lines      : number of dot lines used in the final fit
        """
        def _line_residuals(params, pts):
            p0  = params[:3]
            v   = self.angles_to_unit_vector(params[3], params[4])
            return np.concatenate([np.cross(v, p - p0) for p in pts])

        def _fit_line_3d(pts):
            p0_init = np.mean(pts, axis=0)
            v_init  = pts[-1] - pts[0]
            if np.linalg.norm(v_init) < 1e-9:
                v_init = np.array([1.0, 0.0, 0.0])
            th0, ph0 = self.unit_vector_to_angles(v_init)
            x0 = np.array([*p0_init, th0, ph0], dtype=float)
            lb = np.array([-np.inf, -np.inf, -np.inf, -np.pi / 2, -np.pi])
            ub = np.array([ np.inf,  np.inf,  np.inf,  np.pi / 2,  np.pi])
            res = least_squares(_line_residuals, x0, args=(pts,), bounds=(lb, ub))
            p0 = res.x[:3]
            v  = self.angles_to_unit_vector(res.x[3], res.x[4])
            v /= (np.linalg.norm(v) + 1e-12)
            return p0, v

        # ── Fit one 3-D line per dot ───────────────────────────────────────
        lines = []  # (p0, v) per valid dot
        for i in range(U_tx.shape[0]):
            pts   = U_tx[i, :, :]
            valid = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 1e-9)
            pts   = pts[valid]
            if pts.shape[0] < 2:
                continue
            lines.append(_fit_line_3d(pts))

        I = np.eye(3)

        def _intersect(line_set):
            A_mat = np.zeros((3, 3))
            b_vec = np.zeros(3)
            for p0, v in line_set:
                P = I - np.outer(v, v)
                A_mat += P
                b_vec += P @ p0
            x, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
            return x

        intersection = _intersect(lines)

        # ── Optional FL_C outlier removal + refit ──────────────────────────
        if outlier_sigma is not None and len(lines) >= 3:
            dists = np.array([
                np.linalg.norm((I - np.outer(v, v)) @ (p0 - intersection))
                for p0, v in lines
            ])
            med = np.median(dists)
            mad = 1.4826 * np.median(np.abs(dists - med))
            keep = dists <= med + outlier_sigma * max(mad, 1e-12)
            if np.any(~keep) and np.sum(keep) >= 2:
                lines = [l for l, k in zip(lines, keep) if k]
                intersection = _intersect(lines)

        return intersection, len(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 5 – Unit vector estimation
    # ─────────────────────────────────────────────────────────────────────────

    def estimate_unit_vectors(self, U: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Estimate a unit direction vector V[i] for each projected dot.

        For each dot i, Q_ij = U_ij − B are the calibration points expressed
        relative to the transmitter.  V[i] is fitted by minimising cross-product
        residuals (angle error).

        Parameters
        ----------
        U : (n_dots, n_dist, 3) calibration 3-D points in camera coordinates
        B : (3,) transmitter position in camera coordinates (full offset, not
            just the x-component — `estimate_baseline`'s line-intersection fit
            is 3-D and the real rig need not have a purely horizontal baseline)

        Returns
        -------
        V : (n_dots, 3) unit direction vectors per dot in camera / transmitter space
        """
        B = np.asarray(B, dtype=float)

        def _v_residuals(params, Q):
            v = self.angles_to_unit_vector(params[0], params[1])
            return np.concatenate([np.cross(q, v) / (np.linalg.norm(q) + 1e-12) for q in Q])

        def _fit_unit_vector(Q):
            m = np.mean(Q, axis=0)
            if np.linalg.norm(m) < 1e-9:
                m = np.array([1.0, 0.0, 0.0])
            th0, ph0 = self.unit_vector_to_angles(m)
            res = least_squares(_v_residuals, [th0, ph0], args=(Q,),
                                bounds=([-np.pi / 2, -np.pi], [np.pi / 2, np.pi]))
            v = self.angles_to_unit_vector(res.x[0], res.x[1])
            v /= (np.linalg.norm(v) + 1e-12)
            return v

        n_dots = U.shape[0]
        V = np.full((n_dots, 3), np.nan, dtype=float)

        for i in range(n_dots):
            pts   = U[i, :, :]
            valid = np.all(np.isfinite(pts), axis=1)
            pts   = pts[valid]
            if pts.shape[0] < 2:
                continue
            Q       = pts - B[None, :]
            V[i, :] = _fit_unit_vector(Q)

        return V

    # ─────────────────────────────────────────────────────────────────────────
    # Runtime helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def cam_ray(u: float, v: float, K_inv: np.ndarray) -> np.ndarray:
        """Return the unit camera ray for pixel (u, v) via the inverse intrinsics."""
        r = K_inv @ np.array([u, v, 1.0], dtype=float)
        r /= (np.linalg.norm(r) + 1e-12)
        return r

    @staticmethod
    def triangulate_depth(u: float, v: float, v_i: np.ndarray,
                          B: np.ndarray, K_inv: np.ndarray) -> float:
        """
        Compute triangulated axial depth Z_tri.

        Solves  s·r − t·v_i = B  in LS (r = camera ray at (u,v), v_i = dot unit vector).
        Returns the axial component: Z_tri = (s·r)[2].

        From Microsoft-Paper eq. for the consistency error.
        Returns NaN if the dot direction v_i is invalid (uncalibrated dot).
        """
        if not np.all(np.isfinite(v_i)):
            return float("nan")
        r   = DotCalibration.cam_ray(u, v, K_inv)
        v_i = v_i / (np.linalg.norm(v_i) + 1e-12)
        A   = np.column_stack([r, -v_i])   # 3×2
        st, *_ = np.linalg.lstsq(A, B, rcond=None)
        return abs(float(st[0] * r[2]))

    @staticmethod
    def consistency_error(Z_tof: float, Z_tri: float) -> float:
        """
        Compute the Microsoft-Paper consistency error:  ε = |1/Z_ToF − 1/Z_tri|.

        A small ε indicates agreement between iToF and triangulation (MPI-free).
        Returns inf for invalid (non-positive or NaN) depths.
        """
        if not (Z_tof > 1e-12 and Z_tri > 1e-12):  # also catches NaN
            return np.inf
        return abs(1.0 / Z_tof - 1.0 / Z_tri)

    @staticmethod
    def build_calibration_trails(subpixel_list: List[List[dict]],
                                  n_dots: Optional[int] = None) -> np.ndarray:
        """
        Build a (n_dist, n_dots, 2) array of pixel-space calibration trails.

        Each trail[j, i, :] = (u, v) of dot i at calibration distance index j.
        Missing entries are filled by linear interpolation.

        Parameters
        ----------
        subpixel_list : list (one entry per distance) of subpixel dicts {id, x, y}
        n_dots        : total number of dots. If None, derived from
                        max(id) + 1 across all distances.
        """
        n_dist = len(subpixel_list)
        if n_dots is None:
            all_ids = [int(d["id"]) for spx in subpixel_list for d in spx]
            n_dots = (max(all_ids) + 1) if all_ids else 0
        trail_xy = np.full((n_dist, n_dots, 2), np.nan, dtype=float)

        for j, spx in enumerate(subpixel_list):
            for d in spx:
                i = int(d["id"])
                trail_xy[j, i, 0] = float(d["x"])
                trail_xy[j, i, 1] = float(d["y"])

        jj = np.arange(n_dist, dtype=float)
        for i in range(n_dots):
            xs, ys = trail_xy[:, i, 0], trail_xy[:, i, 1]
            m = np.isfinite(xs) & np.isfinite(ys)
            if not np.any(m):
                continue
            trail_xy[:, i, 0] = np.interp(jj, jj[m], xs[m])
            trail_xy[:, i, 1] = np.interp(jj, jj[m], ys[m])

        return trail_xy


class ToFSampler:
    """Depth / 3-D point lookup at image pixel (u, v) from a loaded `load_tof_pcd`
    result, for organised (H, W) grids and unorganised valid-return clouds alike.

    The current real-sensor binary PCD export has HEIGHT 1 and drops invalid
    pixels, so it can't be raster-indexed like the simulated (organised) PBRT
    clouds; this class picks the matching lookup strategy once at construction
    and hides the difference behind one small interface used by both the
    calibration code and `approaches.py`'s test-time depth fusion.
    """

    def __init__(self, tof_data: dict, K: np.ndarray):
        self.points = tof_data["points_3d"]
        self.distance = tof_data["distance"]
        depth_map = tof_data.get("depth_map")
        height = tof_data.get("height") or 1
        self.organised = depth_map is not None and height > 1
        if self.organised:
            self.depth_map = depth_map
            self.points_map = tof_data["points_map"]
            self.H, self.W = depth_map.shape
        else:
            self.tree = cKDTree(DotCalibration._project_to_pixels(self.points, K))

    def point_at(self, u: float, v: float, k: int = 4, max_px: float = 2.0) -> Optional[np.ndarray]:
        """Interpolated 3-D point at pixel (u, v), or None if none is close enough."""
        if self.organised:
            return DotCalibration._sample_points_map(self.points_map, u, v)
        return DotCalibration._sample_cloud_knn(self.points, self.tree, u, v, k=k, max_px=max_px)

    def depth_at(self, u: float, v: float, search_radius: int = 3,
                 fallback: float = 1e-9) -> Tuple[float, float]:
        """Nearest valid depth to pixel (u, v). Returns (depth, pixel_dist)."""
        if self.organised:
            for r in range(search_radius + 1):
                for dv in range(-r, r + 1):
                    for du in range(-r, r + 1):
                        u2, v2 = int(round(u)) + du, int(round(v)) + dv
                        if 0 <= u2 < self.W and 0 <= v2 < self.H:
                            z = self.depth_map[v2, u2]
                            if np.isfinite(z) and z > 1e-6:
                                return float(z), float(np.hypot(du, dv))
            return fallback, np.inf

        k = (2 * search_radius + 1) ** 2
        dists, idx = self.tree.query([u, v], k=min(k, len(self.distance)))
        dists, idx = np.atleast_1d(dists), np.atleast_1d(idx)
        valid = np.isfinite(dists) & (self.distance[idx] > 1e-6)
        if not np.any(valid):
            return fallback, np.inf
        j = int(np.argmin(dists[valid]))
        return float(self.distance[idx[valid]][j]), float(dists[valid][j])

    def neighborhood_depths(self, u: float, v: float, half_width: int = 3) -> np.ndarray:
        """Local depth samples around pixel (u, v), for robust σ estimates."""
        if self.organised:
            vi, ui = int(round(v)), int(round(u))
            r = half_width
            patch = self.depth_map[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
            return patch.ravel()
        n = (2 * half_width + 1) ** 2
        _, idx = self.tree.query([u, v], k=min(n, len(self.distance)))
        return self.distance[np.atleast_1d(idx)]

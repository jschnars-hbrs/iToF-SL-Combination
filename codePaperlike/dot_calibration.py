import os
import re
import numpy as np
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import OpenEXR
import Imath
from skimage import feature
from skimage.util import img_as_float
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import pandas as pd
from typing import Optional, Tuple, List, Dict
from pathlib import Path


class DotCalibration:
    """
    Calibration and depth fusion for a combined iToF + dot projector sensor.

    Based on:
      - Godbaz et al. 2025 (Microsoft-Paper / FL_C): dot calibration, consistency
        error, active brightness trail
      - Agresti & Zanuttigh, ECCV 2018: maximum likelihood depth fusion
    """

    SL_CHANNEL = "S0.940,000nm"  # EXR channel to use for structured-light images (not visible in RGB Channel, due to IR-Light)

    # Plausible axial-depth window [m] for a ToF return. The real-sensor export
    # marks invalid pixels with finite but absurd values (up to ~2.6e8 m after
    # unit scaling) instead of NaN or 0, so an isfinite() test alone lets them
    # through into the calibration.
    VALID_Z_RANGE = (0.1, 5.0)

    # ─────────────────────────────────────────────────────────────────────────
    # File utilities
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def parse_distance_from_name(filename: str, offset: float = 0.0) -> Optional[float]:
        """
        Parse target distance in metres from a filename.

        Handles formats like:  SL_0.4m.exr, SL_ToF_1.2m.pcd, 3_6m_frame.png

        Pos0…Pos9 carry no distance in the name, so they come from the logbook
        table below. `offset` [mm] is subtracted from every entry — a POSITIVE
        offset means the wall stood NEARER than the tape said. Set it per camera
        via `dist_offset` in calibrate.CAMERAS.
        """

        stem = os.path.splitext(os.path.basename(filename))[0]
       
        GROUND_TRUTH = {
            "Pos0": 402-offset, "Pos1": 439-offset, "Pos2": 486-offset, "Pos3": 545-offset, "Pos4": 621-offset,
            "Pos5": 720-offset, "Pos6": 857-offset, "Pos7": 1059-offset, "Pos8": 1385-offset, "Pos9": 2000-offset,}
        if stem in GROUND_TRUTH:
            return GROUND_TRUTH[stem] / 1000.0

        m = re.search(r"([0-9]+(?:[.,_][0-9]+)?)m", stem, re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1).replace("_", ".").replace(",", "."))
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
                     depth_mode: str = "radial",
                     z_range: Optional[Tuple[float, float]] = None) -> dict:
        """
        Load ToF data from an organised PCD (ASCII or binary) file.

        Parameters
        ----------
        pcd_path    : path to .pcd file
        unit_scale  : multiply xyz by this factor (0.001 converts mm → m)
        depth_mode  : "radial" = sqrt(x²+y²+z²); "axial" or "z" = z only
        z_range     : (z_min, z_max) plausible axial depth window [m]; points
                      outside it are set to NaN. Defaults to `VALID_Z_RANGE`.

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

        # Drop implausible returns (see VALID_Z_RANGE) so that every downstream
        # isfinite() check actually rejects them.
        z_lo, z_hi = self.VALID_Z_RANGE if z_range is None else z_range
        bad = ~np.all(np.isfinite(points), axis=1) \
            | (points[:, 2] < z_lo) | (points[:, 2] > z_hi)
        points[bad] = np.nan

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
    def _sample_points_map(points_map: np.ndarray, u: float, v: float,
                           z_range: Optional[Tuple[float, float]] = None) -> Optional[np.ndarray]:
        """Bilinearly sample a (H, W, 3) point map at subpixel (u, v).

        Corner points that are invalid (non-finite, zero, or outside
        `VALID_Z_RANGE`, i.e. no ToF return, common between dots in sparsely
        illuminated clouds) are excluded and the bilinear weights renormalised
        over the valid corners.

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
        z_lo, z_hi = DotCalibration.VALID_Z_RANGE if z_range is None else z_range
        valid = np.all(np.isfinite(corners), axis=1) \
            & (np.linalg.norm(corners, axis=1) > 1e-6) \
            & (corners[:, 2] >= z_lo) & (corners[:, 2] <= z_hi)
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
        idx = np.clip(idx, 0, len(points) - 1)
        valid = np.isfinite(dists) & (dists <= max_px) \
            & np.all(np.isfinite(points[idx]), axis=1)
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

    @staticmethod
    def _normalize_for_log(image: np.ndarray) -> np.ndarray:
        """Map an image to [0, 1] so blob_log's absolute *threshold* is scale-free.

        blob_log thresholds the scale-normalised LoG response, whose magnitude is
        proportional to the pixel amplitude.  skimage's img_as_float rescales
        integer input (uint8 -> [0, 1]) but is a no-op for float, so raw EXR
        radiance (real sensor: ~150...3900 counts; PBRT renders: ~0...2.7) would
        need a different threshold for every dataset.

        Percentiles rather than min/max: the 1 % floor removes the ambient
        pedestal the real recordings sit on, the 99.9 % ceiling keeps a single hot
        pixel from compressing every dot into the bottom of the range.
        """
        det = img_as_float(image).astype(np.float32)
        lo, hi = np.percentile(det, 1.0), np.percentile(det, 99.9)
        return np.clip((det - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)

    def detect_blobs(self, image_path: str, max_sigma: int = 30, num_sigma: int = 10, min_sigma: int = 5,
                     threshold: float = 0.05, visualize: bool = False, add_synthetic_gaussian: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Laplacian-of-Gaussian blob detector (scikit-image).

        *threshold* applies to the robustly normalised image (see _normalize_for_log),
        so the same value holds for 8-bit PNG, simulated EXR and real-sensor EXR.

        Returns
        -------
        blobs : (N, 3) array of (y, x, radius)
        image : grayscale image array (with synthetic Gaussians added if requested)
        """
        image = self.read_image(image_path)
        if image is None or image.size == 0:
            raise ValueError(f"Invalid image for LoG blob detection: {image_path}")

        detect_img = self._normalize_for_log(image)
        blobs = feature.blob_log(detect_img, max_sigma=max_sigma, num_sigma=num_sigma, min_sigma=min_sigma, threshold=threshold, exclude_border=True)
        blobs[:, 2] = blobs[:, 2] * (2 ** 0.5)  # convert sigma to radius
        if add_synthetic_gaussian:
            image_out = self.add_gaussian_to_detected_blob(image, blobs)
        else:
            image_out = image

        if visualize:
            fig, ax = plt.subplots()
            # raw EXR radiance renders black under imshow's default scaling
            ax.imshow(image_out if add_synthetic_gaussian else detect_img, cmap="gray")
            for y, x, r in blobs:
                ax.add_patch(plt.Circle((x, y), r, color="red", linewidth=1, fill=False))
            plt.show()

        return blobs, image_out

    # ─────────────────────────────────────────────────────────────────────────
    # Subpixel localisation
    # ─────────────────────────────────────────────────────────────────────────

    def _gpr_peak(self, patch: np.ndarray, sigma_blob: Optional[float] = None,
                  coarse_oversample: int = 4, max_train: int = 650) -> Tuple[float, float]:
        """GPR-based subpixel peak within *patch*. Returns (x, y) relative to patch.

        Two-stage localisation per FL_C §4.1: LoG finds the area near the peak,
        then "Gaussian Process Regression is employed to model the intensity
        map for each dot. With the model, a local search is performed to find
        the peak location giving subpixel localization."

        The (simulated) dots are wide, speckle-noisy plateaus, so the GP must
        model the WHOLE dot with a length scale of dot size — anchoring on the
        brightest pixel would lock onto a random speckle. Concretely:

          1. Background-normalise the patch (GP noise level then meaningful on
             any input scale — raw EXR radiance or 8-bit PNG alike).
          2. RBF length scale initialised from the LoG blob scale (the detector
             already measured the dot size), bounded to stay at dot scale; a
             WhiteKernel absorbs the speckle noise.
          3. Fit on a subsampled pixel grid (GP is O(n³)).
          4. Local search on the model: centroid of the above-half-maximum
             region of the posterior mean on an oversampled grid — equals the
             model peak for peaked dots and the plateau centre for flat-top
             dots, at genuine sub-pixel resolution.
        """
        h, w = patch.shape
        y_raw = patch.astype(float)
        bg = float(np.percentile(y_raw, 20))
        amp = float(y_raw.max() - bg)
        if amp <= 0 or h < 3 or w < 3:
            return (w - 1) / 2.0, (h - 1) / 2.0
        I = np.clip((y_raw - bg) / amp, 0.0, None)

        if sigma_blob is None or not np.isfinite(sigma_blob) or sigma_blob <= 0:
            sigma_blob = max(h, w) / 4.0
        # 0.35·σ empirically minimises the cross-distance trail residual on the
        # PBRT data: large enough to suppress the speckle inside a dot, small
        # enough to keep the dot's edges (which carry the localisation
        # information) sharp in the model.
        ls0 = float(np.clip(0.35 * sigma_blob, 1.0, max(h, w)))

        # Subsample the training grid to keep the GP fit tractable.
        stride = max(1, int(np.ceil(np.sqrt(h * w / max_train))))
        syy, sxx = np.mgrid[0:h:stride, 0:w:stride]
        X = np.column_stack([sxx.ravel(), syy.ravel()]).astype(float)
        t = I[::stride, ::stride].ravel()

        # The length scale is PINNED at dot scale: letting the marginal-likelihood
        # optimiser choose it makes the GP fit the speckle inside the dot (small
        # length scale) and the peak lands on a random speckle grain. At dot
        # scale the posterior mean is a smooth dome whose maximum is the dot's
        # photometric centre — flat-top dots included.
        kernel = ConstantKernel(1.0, "fixed") \
            * RBF(length_scale=ls0, length_scale_bounds="fixed") \
            + WhiteKernel(noise_level=5e-2, noise_level_bounds=(1e-5, 1.0))
        gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0,
                                      normalize_y=True)
        gp.fit(X, t)

        # Local search on the model: the subpixel location is the mean of the
        # mask centroids of the model's above-threshold regions at several
        # relative levels (0.35 / 0.5 / 0.65 of the model amplitude on the
        # oversampled grid). For a peaked dot this coincides with the model
        # peak; for flat-top or saturated dots — whose plateau makes a raw
        # argmax ill-conditioned, all localisation information sits in the
        # edges — it is the plateau centre traced at several edge heights.
        # Measured cross-distance trail residuals: 0.05 px (PBRT sim, flat-top
        # speckle dots), 0.10 px (real Schmersal PNGs, saturated dots).
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
                centroids.append((float(X_fine[mask, 0].mean()),
                                  float(X_fine[mask, 1].mean())))
        if not centroids:
            k = int(np.argmax(pred))
            return float(X_fine[k, 0]), float(X_fine[k, 1])
        c = np.mean(np.asarray(centroids), axis=0)
        return (float(np.clip(c[0], 0, w - 1)),
                float(np.clip(c[1], 0, h - 1)))

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

    def _subpixel_refine(self, patch: np.ndarray, mode: str,
                         sigma_blob: Optional[float] = None) -> Tuple[float, float]:
        """Dispatch subpixel refinement on *patch*. Returns (x, y) relative to patch.

        Modes: "GPR" | "center" | "geometricCenter" | "radial"
        sigma_blob: LoG blob scale [px], used as the GP length-scale prior.
        """
        if mode == "GPR":
            return self._gpr_peak(patch, sigma_blob=sigma_blob)
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

            # blob[2] is the LoG radius (= σ·√2); pass σ as the GP length-scale prior
            x_sub, y_sub = self._subpixel_refine(patch, mode,
                                                 sigma_blob=float(blob[2]) / np.sqrt(2.0))
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

    @staticmethod
    def _as_baseline_vector(baseline_guess) -> np.ndarray:
        """Accept a scalar (legacy: x-offset only) or a full 3-vector baseline
        guess and return a (3,) array. The full vector matters in Y mode
        (transmitter above/below the receiver) and whenever the rig's baseline
        is not purely horizontal."""
        B = np.atleast_1d(np.asarray(baseline_guess, dtype=float)).ravel()
        if B.size == 1:
            return np.array([float(B[0]), 0.0, 0.0])
        if B.size != 3:
            raise ValueError(f"baseline_guess must be a scalar or 3-vector, got shape {B.shape}")
        return B.astype(float)

    def tx_space_coords(self, subpixel_list: List[List[dict]], tof_paths: List[str],
                        K: np.ndarray, baseline_guess,
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
        B_guess = self._as_baseline_vector(baseline_guess)
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
                                    coords_list: Optional[List[List[dict]]] = None,
                                    travel_mode: str = "x") \
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
        travel_mode      : "x" (transmitter to the side, dots travel along u —
            renumber row-major) or "y" (transmitter above/below, dots travel
            along v — renumber column-major). Only affects the cosmetic final
            ID ordering, not the matching itself.

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
        # Pure lexsort interleaves rows/columns when dots have small jitter.
        # Instead: cluster along the axis PERPENDICULAR to the travel direction
        # (rows for x-mode, columns for y-mode) from gaps relative to the median
        # nearest-neighbor spacing, then sort within each cluster along the
        # travel axis.
        # (Renumbering always uses PIXEL coordinates, regardless of coords_list.)
        travel_axis = 0 if travel_mode == "x" else 1   # sort within cluster
        cluster_axis = 1 - travel_axis                 # cluster on the other axis
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

        order_by_cluster = np.argsort(means[:, cluster_axis], kind="stable")
        rows: List[List[int]] = []
        current: List[int] = []
        prev_c = None
        for old_id in order_by_cluster:
            c = means[old_id, cluster_axis]
            if prev_c is not None and (c - prev_c) > row_gap_thresh:
                rows.append(current)
                current = []
            current.append(int(old_id))
            prev_c = c
        if current:
            rows.append(current)

        old_to_new = np.empty(roster_size, dtype=int)
        new_id = 0
        for row in rows:
            for old_id in sorted(row, key=lambda i: means[i, travel_axis]):
                old_to_new[old_id] = new_id
                new_id += 1

        for spx in out:
            for d in spx:
                d["id"] = int(old_to_new[int(d["id"])])
            spx.sort(key=lambda d: d["id"])

        return out

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 2c – X/Y travel-mode detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _robust_linfit(x: np.ndarray, y: np.ndarray,
                       iters: int = 3, n_sigma: float = 2.5) -> Tuple[float, float]:
        """Least-squares y = slope·x + intercept with sigma-clipping."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        slope = intercept = float("nan")
        for _ in range(max(1, iters)):
            if x.size < 2:
                break
            A = np.column_stack([x, np.ones_like(x)])
            (slope, intercept), *_ = np.linalg.lstsq(A, y, rcond=None)
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
    def detect_travel_mode(subpixel_list: List[List[dict]],
                           cal_dists: List[float],
                           K: np.ndarray,
                           mode_override: Optional[str] = None) -> dict:
        """
        Detect whether the dots travel along image x or y with distance, i.e.
        whether the transmitter sits to the side (X mode) or above/below the
        receiver (Y mode), and derive a signed baseline guess from the parallax.

        Each dot's pixel position is affine in w = 1/z (see `trail_line_params`):
            u(w) = u_∞ + a_u·w   with   a_u = fx·B_x - B_z·(u_∞ - cx)
            v(w) = v_∞ + a_v·w   with   a_v = fy·B_y - B_z·(v_∞ - cy)
        So fitting (u_∞, a_u, v_∞, a_v) per dot and then regressing a_u against
        (u_∞ - cx), and a_v against (v_∞ - cy), recovers the FULL 3-D baseline
        from pixel data alone: the intercepts give B_x, B_y and the common slope
        gives -B_z. No manually guessed sign, and no assumption that the
        transmitter is coplanar with the receiver: on the Schmersal rig B_z is
        ~33 mm, comparable to B_y, and pinning it to 0 leaves the guess ~9 cm
        off, which is more than the later outlier rejection can absorb.

        Falls back to the pure-axis (B_z = 0) estimate when too few dots are
        tracked across enough distances to make the second regression stable.

        Parameters
        ----------
        subpixel_list : tracked list (per distance) of dicts {id, x, y},
                        ordered near → far like `cal_dists`
        cal_dists     : calibration distances [m], same order
        K             : 3x3 intrinsics (fy may be negative — handled via K)
        mode_override : optional "x"/"y" to force the mode; the baseline
                        component is still estimated along the forced axis

        Returns
        -------
        dict with:
          mode         : "x" or "y"
          B_axis_est   : signed baseline estimate along the travel axis [m]
          B_guess_vec  : (3,) baseline guess vector for back-projection/tracking
          du_dw, dv_dw : median pixel shifts per unit Δ(1/z) (diagnostics)
          B_z_est      : signed baseline component along the optical axis [m]
          n_fit        : number of dots that entered the affine regression
        """
        n_dist = len(subpixel_list)
        if n_dist < 2:
            raise ValueError("Need at least two calibration distances to detect the travel mode.")
        pos = [{int(d["id"]): (float(d["x"]), float(d["y"])) for d in spx}
               for spx in subpixel_list]
        w = 1.0 / np.asarray(cal_dists, dtype=float)

        du_dw, dv_dw = [], []
        u_inf, v_inf = [], []
        all_ids = set().union(*[p.keys() for p in pos])
        for i in sorted(all_ids):
            present = [j for j in range(n_dist) if i in pos[j]]
            if len(present) < 2:
                continue
            us = np.array([pos[j][i][0] for j in present])
            vs = np.array([pos[j][i][1] for j in present])
            ws = w[present]
            if np.ptp(ws) < 1e-9:
                continue
            if len(present) >= 3:
                # Affine fit over ALL observations; the intercept is the
                # vanishing point (w -> 0), needed for the B_z regression.
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

        med_du = float(np.median(du_dw))
        med_dv = float(np.median(dv_dw))
        mode = mode_override or ("x" if abs(med_du) >= abs(med_dv) else "y")

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # a_u = fx·B_x − B_z·(u_∞ − cx):  slope -> −B_z, intercept -> fx·B_x.
        n_fit = len(du_dw)
        B_x = B_y = B_z = None
        # The B_z regression needs the vanishing points to actually SPREAD over
        # the sensor; a single row or column of dots leaves its slope
        # unconstrained and would return a confident but meaningless B_z.
        spread_ok = (n_fit >= 8
                     and np.ptp(u_inf) > 0.25 * abs(fx)
                     and np.ptp(v_inf) > 0.25 * abs(fy))
        if spread_ok:
            su, bu = DotCalibration._robust_linfit(np.array(u_inf) - cx, np.array(du_dw))
            sv, bv = DotCalibration._robust_linfit(np.array(v_inf) - cy, np.array(dv_dw))
            if np.all(np.isfinite([su, bu, sv, bv])):
                B_x, B_y = bu / fx, bv / fy
                # Both regressions see the same B_z; average them.
                B_z = -0.5 * (su + sv)
                if np.linalg.norm([B_x, B_y, B_z]) > 0.5:
                    # Implausible for a hand-held rig — distrust and fall back.
                    B_x = B_y = B_z = None

        if B_z is None:
            # Too few dots for the second regression — fall back to the
            # pure-axis guess (B_z pinned to 0).
            B_axis = (med_du / fx) if mode == "x" else (med_dv / fy)
            B_guess_vec = (np.array([B_axis, 0.0, 0.0]) if mode == "x"
                           else np.array([0.0, B_axis, 0.0]))
            B_z = 0.0
        else:
            B_guess_vec = np.array([B_x, B_y, B_z])
            B_axis = B_guess_vec[0] if mode == "x" else B_guess_vec[1]

        return {"mode": mode, "B_axis_est": float(B_axis),
                "B_guess_vec": B_guess_vec, "du_dw": med_du, "dv_dw": med_dv,
                "B_z_est": float(B_z), "n_fit": int(n_fit)}

    # ─────────────────────────────────────────────────────────────────────────
    # Calibration Step 3 – 3-D back-projection
    # ─────────────────────────────────────────────────────────────────────────

    _annulus_warned = set()

    @classmethod
    def _fit_annulus_to_spacing(cls, subpixels: List[dict], cfg: dict,
                                max_frac: float = 0.45,
                                warn_tag: str = "") -> dict:
        """
        Shrink the annulus so it cannot reach the neighbouring dots.

        The corrupted halo scales with the dot's PSF while the usable outer
        radius scales with the dot SPACING, and the two are independent: the
        sparse 19.06 set has ~42 px spacing with a ~7 px halo, the Coherent set
        ~15 px spacing with a ~3 px halo. A ring tuned for one silently samples
        its neighbours' corrupted pixels on the other, so clamp r_out to
        `max_frac` of the median nearest-neighbour spacing and scale r_in with
        it. Returns the kwargs for `annulus_axial_z`.
        """
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
                print(f"  [annulus] dot spacing {spacing:.1f} px is too tight for "
                      f"r_out={r_out:.1f}; shrinking ring to "
                      f"{ring['r_in']:.1f}–{ring['r_out']:.1f} px"
                      + (f"  ({Path(warn_tag).name})" if warn_tag else ""))
        return ring

    @staticmethod
    def annulus_axial_z(z_map: np.ndarray, u: float, v: float,
                        r_in: float = 8.0, r_out: float = 14.0,
                        min_valid: int = 8) -> float:
        """
        Median axial ToF depth over an annulus around pixel (u, v).

        On this rig the dot projector is on during the ToF capture, so the dot
        pixels themselves are at/near saturation and their phase — hence their
        depth — is wrong, while the surface between the dots reads correctly.
        Sampling a ring that clears the dot's corrupted halo (which reaches
        ~±9 px) but stays inside the dot spacing (~46 px) recovers the surface
        depth at the dot without assuming anything about the target's shape.

        The window is clipped at the image border rather than skipping border
        dots. Returns NaN if fewer than `min_valid` finite pixels remain.
        """
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

    def backproject_calibration_dots(self, subpixel_list: List[List[dict]],
                                      tof_paths: List[str],
                                      K: np.ndarray,
                                      pcd_unit_scale: float = 0.001,
                                      pcd_depth_mode: str = "radial",
                                      baseline_guess=3.8e-2,
                                      tof_sample: Optional[dict] = None) \
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
        baseline_guess  : initial transmitter offset guess [m] — scalar (legacy
                          x-offset) or full 3-vector; used to compute U_tx
        tof_sample      : how to read the ToF depth at a dot.
                          None / {"mode": "center"} (default) bilinearly samples
                          the point map AT the dot — correct for the simulated
                          PBRT clouds and for captures whose ToF frame has the
                          projector off.
                          {"mode": "annulus", "r_in": 8.0, "r_out": 14.0} takes
                          the median axial depth of a ring around the dot and
                          places U on the dot's own camera ray (see
                          `annulus_axial_z`) — needed when the projector is on
                          during the ToF capture and saturates the dot pixels.
                          Requires an organised cloud and a correct K; falls
                          back to "center" for unorganised clouds.

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

        cfg = dict(tof_sample or {})
        mode = cfg.pop("mode", "center")
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        for j, subpixels in enumerate(subpixel_list):
            tof_data = self.load_tof_pcd(tof_paths[j], unit_scale=pcd_unit_scale,
                                          depth_mode=pcd_depth_mode)

            if mode == "annulus" and "points_map" in tof_data:
                z_map = tof_data["points_map"][..., 2]
                ring = self._fit_annulus_to_spacing(subpixels, cfg, warn_tag=tof_paths[j])
                for dot in subpixels:
                    z = self.annulus_axial_z(z_map, dot["x"], dot["y"], **ring)
                    if not np.isfinite(z):
                        continue
                    # Place the point on the DOT's ray, not at the annulus
                    # centroid — a ring median is unbiased in depth but says
                    # nothing about where the dot sits laterally.
                    U[dot["id"], j, :] = z * np.array(
                        [(dot["x"] - cx) / fx, (dot["y"] - cy) / fy, 1.0])
                continue

            sample = self._tof_point_sampler(tof_data, K)
            for dot in subpixels:
                P = sample(dot["x"], dot["y"])
                if P is not None:
                    U[dot["id"], j, :] = P

        B_guess = self._as_baseline_vector(baseline_guess)
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
                          outlier_sigma: Optional[float] = 3.0,
                          min_points: int = 3) \
            -> Tuple[np.ndarray, int]:
        """
        Estimate the transmitter position from U_tx (3-D points in approximate transmitter space).

        Fits a 3-D line through each dot's multi-distance samples, then finds the
        least-squares intersection of all lines.  The FULL 3-D intersection is the
        correction to the initial baseline guess:
        B = B_guess + intersection  (keep all three components — the rig's
        baseline need not be purely axis-aligned).

        Parameters
        ----------
        U_tx : (n_dots, n_dist, 3) array of calibration points shifted by baseline_guess
        outlier_sigma : float (default 3.0, None disables). Performs the FL_C §4.1
            outlier removal ("After removing outlier rays, the baseline AB is found
            by an additional least squares optimization"): after a first pass,
            dot lines whose distance to the intersection exceeds
            median + outlier_sigma · MAD are discarded and the intersection refit.
        min_points : minimum number of distances a dot must be observed at to
            contribute a line (FL_C §4.1 low-confidence invalidation; a dot seen
            at only 2 distances gives a poorly constrained line). Relaxed to 2
            automatically if fewer than 3 dots satisfy it (partial FOV tracks).

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
        def _collect_lines(min_pts: int):
            lines = []  # (p0, v) per valid dot
            for i in range(U_tx.shape[0]):
                pts   = U_tx[i, :, :]
                valid = np.all(np.isfinite(pts), axis=1) & (np.linalg.norm(pts, axis=1) > 1e-9)
                pts   = pts[valid]
                if pts.shape[0] < min_pts:
                    continue
                lines.append(_fit_line_3d(pts))
            return lines

        lines = _collect_lines(max(min_points, 2))
        if len(lines) < 3 and min_points > 2:
            lines = _collect_lines(2)

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

    # ─────────────────────────────────────────────────────────────────────────
    # Dot trails as 1-D vectors (FL_C §4.2) + calibration noise estimates
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def trail_line_params(v_i: np.ndarray, B: np.ndarray, K: np.ndarray) \
            -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Affine epipolar-trail parameters of one dot: pixel(w) = p_inf + w·g,
        with w = 1/z (axial).

        Derivation: a point on the dot ray is X(t) = B + t·V̂ with axial depth
        z = X_z, so X_x/z = V̂x/V̂z + (B_x − B_z·V̂x/V̂z)·w — LINEAR in w (same
        for y). p_inf is the vanishing point (z → ∞); ‖g‖·Δw is the pixel step.
        This is the analytic form of the paper's "resampled along the epipolar
        lines" dot trail and holds for any 3-D baseline (X or Y mode alike).

        Returns (p_inf(2,), g(2,)), or None for an invalid V or a ray parallel
        to the image plane.
        """
        if not np.all(np.isfinite(v_i)):
            return None
        v = np.asarray(v_i, float)
        v = v / (np.linalg.norm(v) + 1e-12)
        if abs(v[2]) < 1e-9:
            return None
        B = np.asarray(B, float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        p_inf = np.array([fx * v[0] / v[2] + cx, fy * v[1] / v[2] + cy])
        g = np.array([fx * (B[0] - B[2] * v[0] / v[2]),
                      fy * (B[1] - B[2] * v[1] / v[2])])
        return p_inf, g

    def build_dot_trails_1d(self, V: np.ndarray, B: np.ndarray, K: np.ndarray,
                            z_min: float = 0.3, step_px: float = 1.0,
                            image_shape: Optional[Tuple[int, int]] = None,
                            margin: float = 2.0) -> List[Optional[dict]]:
        """
        Precompute the 1-D dot trails (FL_C §4.2): for every calibrated dot,
        sample its epipolar path from z_min out to INFINITY at ~step_px pixel
        spacing. Each sample has a known representative triangulated depth, so
        no triangulation is needed at runtime — the consistency error becomes a
        pure 1-D scan.

        Sampling is uniform in w = 1/z, which IS uniform in pixels because the
        trail is affine in w (see `trail_line_params`) — this mirrors the
        paper's equispaced-reciprocal design.

        Parameters
        ----------
        V           : (n_dots, 3) dot unit vectors
        B           : (3,) transmitter position (full 3-D baseline)
        K           : 3×3 intrinsics
        z_min       : nearest trail depth [m] (paper traces from ~30 cm)
        step_px     : approximate pixel spacing of trail samples
        image_shape : optional (H, W); trims trail ends outside the image
        margin      : pixels of slack kept beyond the image border when trimming

        Returns
        -------
        trails : list of length n_dots; entry i is None for uncalibrated dots,
                 else {"u", "v", "w", "z"} float arrays ordered near → far
                 (w descending to 0, z ascending to inf).
        """
        B = np.asarray(B, float)
        trails: List[Optional[dict]] = []
        w_max = 1.0 / float(z_min)
        for i in range(V.shape[0]):
            lp = self.trail_line_params(V[i], B, K)
            if lp is None:
                trails.append(None)
                continue
            p_inf, g = lp
            span_px = float(np.linalg.norm(g)) * w_max
            n = max(2, int(np.ceil(span_px / step_px)) + 1)
            w = np.linspace(w_max, 0.0, n)
            u = p_inf[0] + w * g[0]
            v = p_inf[1] + w * g[1]
            if image_shape is not None:
                H, W = image_shape
                inside = ((u >= -margin) & (u <= W - 1 + margin)
                          & (v >= -margin) & (v <= H - 1 + margin))
                if not np.any(inside):
                    trails.append(None)
                    continue
                idx = np.nonzero(inside)[0]
                sl = slice(idx[0], idx[-1] + 1)   # contiguous, keeps 1-D order
                u, v, w = u[sl], v[sl], w[sl]
            with np.errstate(divide="ignore"):
                z = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), np.inf)
            trails.append({"u": u, "v": v, "w": w, "z": z})
        return trails

    def fit_trails_from_detections(self, subpixel_list: List[List[dict]],
                                   U: np.ndarray,
                                   z_min: float = 0.3, step_px: float = 1.0,
                                   image_shape: Optional[Tuple[int, int]] = None,
                                   margin: float = 2.0, min_obs: int = 3,
                                   V: Optional[np.ndarray] = None,
                                   B: Optional[np.ndarray] = None,
                                   K: Optional[np.ndarray] = None) \
            -> List[Optional[dict]]:
        """
        1-D dot trails traced FROM THE CALIBRATION DETECTIONS (FL_C §4.1: "if
        we trace the dot epipolar lines … using the dot calibration we can
        generate the 'dot trails'").

        Per dot, the detected pixel positions are affine in w = 1/z (z = the
        measured axial ToF depth of that detection): u(w) = u_∞ + a_u·w,
        v(w) = v_∞ + a_v·w. Fitting this to the detections instead of
        reprojecting the 3-D ray through a pinhole ABSORBS REAL LENS
        DISTORTION locally — on a real camera the pinhole reprojection can be
        off by >10 px toward the image corners. Extrapolating to w = 0 gives
        the vanishing point.

        Dots with fewer than min_obs detections fall back to the analytic
        V/B/K trail (if provided). Output format identical to
        `build_dot_trails_1d` ({"u","v","w","z"} per dot, near → ∞).
        """
        n_dots = U.shape[0]
        pos_by_id: List[Dict[int, Tuple[float, float]]] = [
            {int(d["id"]): (float(d["x"]), float(d["y"])) for d in spx}
            for spx in subpixel_list]
        w_max = 1.0 / float(z_min)
        trails: List[Optional[dict]] = []
        analytic = None
        if V is not None and B is not None and K is not None:
            analytic = self.build_dot_trails_1d(V, B, K, z_min=z_min,
                                                step_px=step_px,
                                                image_shape=image_shape,
                                                margin=margin)
        for i in range(n_dots):
            ws, us, vs = [], [], []
            for j in range(U.shape[1]):
                z = U[i, j, 2]
                if not (np.isfinite(z) and z > 1e-6) or i not in pos_by_id[j]:
                    continue
                ws.append(1.0 / float(z))
                us.append(pos_by_id[j][i][0])
                vs.append(pos_by_id[j][i][1])
            if len(ws) < min_obs:
                trails.append(analytic[i] if analytic is not None else None)
                continue
            A = np.column_stack([np.ones(len(ws)), np.asarray(ws)])
            cu, *_ = np.linalg.lstsq(A, np.asarray(us), rcond=None)
            cv, *_ = np.linalg.lstsq(A, np.asarray(vs), rcond=None)
            g = np.array([cu[1], cv[1]])       # px per unit w
            p_inf = np.array([cu[0], cv[0]])   # vanishing point
            span_px = float(np.linalg.norm(g)) * w_max
            if span_px < 2.0:
                trails.append(analytic[i] if analytic is not None else None)
                continue
            n = max(2, int(np.ceil(span_px / step_px)) + 1)
            w = np.linspace(w_max, 0.0, n)
            u = p_inf[0] + w * g[0]
            v = p_inf[1] + w * g[1]
            if image_shape is not None:
                H, W = image_shape
                inside = ((u >= -margin) & (u <= W - 1 + margin)
                          & (v >= -margin) & (v <= H - 1 + margin))
                if not np.any(inside):
                    trails.append(None)
                    continue
                idx = np.nonzero(inside)[0]
                sl = slice(idx[0], idx[-1] + 1)
                u, v, w = u[sl], v[sl], w[sl]
            with np.errstate(divide="ignore"):
                z = np.where(w > 1e-12, 1.0 / np.maximum(w, 1e-12), np.inf)
            trails.append({"u": u, "v": v, "w": w, "z": z})
        return trails

    def estimate_sigma_u(self, subpixel_list: List[List[dict]], V: np.ndarray,
                         B: np.ndarray, K: np.ndarray) -> float:
        """
        Subpixel localisation noise σ_u [px]: robust std of the perpendicular
        residuals of all calibration detections to their dot's analytic trail
        line. Used for σ_SL = z²/(‖B‖·f)·σ_u (summary eq. 7 / Agresti eq. 15
        d⁴/B² scaling).

        Uses 1.4826·median(|residual|) — the half-normal MAD estimator — so a
        few mistracked dots don't inflate the estimate.
        """
        lines = [self.trail_line_params(V[i], B, K) for i in range(V.shape[0])]
        return self._sigma_u_from_lines(subpixel_list, lines)

    def estimate_sigma_u_trails(self, subpixel_list: List[List[dict]],
                                trails: List[Optional[dict]]) -> float:
        """Like `estimate_sigma_u`, but residuals are measured against the
        (possibly detection-fitted, distortion-absorbing) trail lines."""
        lines = []
        for t in trails:
            if t is None or len(t["u"]) < 2:
                lines.append(None)
                continue
            p0 = np.array([t["u"][-1], t["v"][-1]])   # vanishing end
            g = np.array([t["u"][0] - t["u"][-1], t["v"][0] - t["v"][-1]])
            lines.append((p0, g))
        return self._sigma_u_from_lines(subpixel_list, lines)

    @staticmethod
    def _sigma_u_from_lines(subpixel_list, lines) -> float:
        res = []
        for spx in subpixel_list:
            for d in spx:
                lp = lines[int(d["id"])] if int(d["id"]) < len(lines) else None
                if lp is None:
                    continue
                p_inf, g = lp
                gn = np.linalg.norm(g)
                if gn < 1e-9:
                    continue
                ghat = g / gn
                rvec = np.array([float(d["x"]), float(d["y"])]) - p_inf
                perp = rvec - (rvec @ ghat) * ghat
                res.append(float(np.linalg.norm(perp)))
        if not res:
            return float("nan")
        return float(1.4826 * np.median(res))

    @staticmethod
    def estimate_sigma_tof(U: np.ndarray, min_dots: int = 5,
                           expected_z: Optional[List[float]] = None,
                           rel_tol: float = 0.3) -> np.ndarray:
        """
        ToF depth noise vs. depth from the flat-wall calibration data: per
        calibration distance, robust std (1.4826·MAD) of the dots' axial ToF
        depths around their median. On a flat wall orthogonal to the optical
        axis the spread is measurement noise, so this is the empirical
        counterpart of Agresti's σ_ToF error-propagation model.

        expected_z (optional, one entry per distance): rows whose median
        deviates more than rel_tol·expected from the known distance are
        dropped — far distances where only a handful of (mis-)sampled dots
        remain otherwise poison the interpolation table.

        Returns
        -------
        (m, 2) array of rows (z_median [m], sigma_z [m]), sorted by z.
        Interpolate at runtime; may be empty if no distance has enough dots.
        """
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
            sig = float(1.4826 * np.median(np.abs(z - med)))
            rows.append((med, sig))
        rows.sort()
        return np.array(rows, dtype=float).reshape(-1, 2)

    @staticmethod
    def plot_dot_trails(trails: List[Optional[dict]],
                        image_shape: Optional[Tuple[int, int]] = None,
                        background: Optional[np.ndarray] = None,
                        title: str = "Precomputed 1-D dot trails (near → ∞)"):
        """Overlay all dot trails on the image plane (cf. FL_C Fig. 7, middle).
        Near end drawn as a dot, vanishing point as a red marker."""
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
        """Vectorised bilinear sampling of a grayscale image at subpixel (u, v).
        Out-of-bounds samples return NaN. Used to resample active brightness
        along the dot trails (FL_C §4.2)."""
        H, W = image.shape[:2]
        u = np.asarray(u, float)
        v = np.asarray(v, float)
        out = np.full(u.shape, np.nan, dtype=float)
        ok = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)
        if not np.any(ok):
            return out
        uu, vv = u[ok], v[ok]
        u0 = np.floor(uu).astype(int)
        v0 = np.floor(vv).astype(int)
        u1 = np.minimum(u0 + 1, W - 1)
        v1 = np.minimum(v0 + 1, H - 1)
        du, dv = uu - u0, vv - v0
        img = image.astype(float)
        out[ok] = (img[v0, u0] * (1 - du) * (1 - dv)
                   + img[v0, u1] * du * (1 - dv)
                   + img[v1, u0] * (1 - du) * dv
                   + img[v1, u1] * du * dv)
        return out


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
            # NaN-marked invalid returns would poison both the tree and every
            # query, so index the finite subset only.
            keep = np.all(np.isfinite(self.points), axis=1)
            self.points = self.points[keep]
            self.distance = self.distance[keep]
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

"""3-D point cloud viewer (PyVista): all generated / loaded PCDs, z filter, SOR, mesh, measuring, export."""

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QSpinBox, QSplitter, QVBoxLayout, QWidget)
from scipy.spatial import Delaunay

from dot_calibration import DotCalibration, sor_keep_mask
from reconstruction import COLORS as APPROACH_COLORS

PALETTE = ["#ff7f0e", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]  # A1–A3 colours kept free
SOLID, RGB = "Einfarbig", "RGB (aus PCD)"


def read_pcd(path):
    """{field: array}, width, height of an ASCII or binary PCD (header parsing from DotCalibration)."""
    dc = DotCalibration()
    meta = dc._parse_pcd_header(str(path))
    fields = meta["fields"]
    if meta["data"] == "ascii":
        df = pd.read_csv(path, sep=r"\s+", header=None, names=fields, skiprows=meta["header_len"], engine="c")
        cols = {f: df[f].to_numpy() for f in fields}
    elif meta["data"] == "binary":
        n = meta.get("points", meta.get("width", 0) * meta.get("height", 1))
        raw = np.fromfile(path, dtype=dc._pcd_binary_dtype(meta), count=n, offset=meta["data_offset"])
        cols = {f: raw[f] for f in fields}
    else:
        raise NotImplementedError(f"PCD DATA {meta['data']!r} is not supported")
    return cols, meta.get("width"), meta.get("height")


def write_pcd(path, cols):
    """Binary PCD of {field: 1-D array} (float32 / int32 fields)."""
    names = list(cols)
    dtype = [(n, "<i4" if np.issubdtype(np.asarray(cols[n]).dtype, np.integer) else "<f4") for n in names]
    n = len(cols[names[0]])
    arr = np.zeros(n, dtype=dtype)
    for name in names:
        arr[name] = cols[name]
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\n"
              f"FIELDS {' '.join(names)}\nSIZE {' '.join('4' for _ in names)}\n"
              f"TYPE {' '.join('I' if t == '<i4' else 'F' for _, t in dtype)}\nCOUNT {' '.join('1' for _ in names)}\n"
              f"WIDTH {n}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {n}\nDATA binary\n")
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(arr.tobytes())


def filter_triangles(P, tris, valid, max_edge):
    """Triangles whose corners are all valid and whose longest edge is ≤ max_edge."""
    tris = tris[valid[tris].all(axis=1)]
    v = P[tris]
    edge = np.linalg.norm(v - np.roll(v, 1, axis=1), axis=2).max(axis=1)
    return tris[edge <= max_edge]


def grid_triangles(H, W):
    """Two triangles per pixel quad of an organized H×W cloud."""
    idx = np.arange(H * W).reshape(H, W)
    a, b = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    c, d = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    return np.concatenate([np.c_[a, b, c], np.c_[b, d, c]])


def image_plane_triangles(P, valid):
    """Delaunay triangles of an unorganized cloud in the image plane (x/z, y/z)."""
    ids = np.flatnonzero(valid)
    if len(ids) < 3:
        return np.zeros((0, 3), int)
    q = P[ids, :2] / P[ids, 2:3]
    try:
        return ids[Delaunay(q).simplices]
    except Exception:
        return np.zeros((0, 3), int)


def find_cloudcompare():
    """Command that starts CloudCompare, or None."""
    for name in ("CloudCompare", "cloudcompare", "cloudcompare.CloudCompare"):
        exe = shutil.which(name)
        if exe:
            return [exe]
    for p in (Path("C:/Program Files/CloudCompare/CloudCompare.exe"),
              Path("/Applications/CloudCompare.app/Contents/MacOS/CloudCompare")):
        if p.exists():
            return [str(p)]
    if shutil.which("flatpak"):
        ok = subprocess.run(["flatpak", "info", "org.cloudcompare.CloudCompare"], capture_output=True).returncode == 0
        if ok:
            return ["flatpak", "run", "org.cloudcompare.CloudCompare"]
    return None


@dataclass
class Cloud:
    label: str
    path: Path
    P: np.ndarray                  # (N, 3) in metres, NaN for missing points
    fields: dict                   # extra per-point fields
    H: Optional[int] = None        # set for organized clouds
    W: Optional[int] = None
    scale: float = 1.0             # file unit → metres
    color: str = "#1f77b4"
    sor_cache: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path, label=None, color=PALETTE[0]):
        cols, W, H = read_pcd(path)
        P = np.column_stack([cols["x"], cols["y"], cols["z"]]).astype(np.float64)
        P[~np.isfinite(P).all(axis=1)] = np.nan
        z = np.abs(P[:, 2])
        z = z[z > 0]
        scale = 1e-3 if z.size and np.median(z) > 20 else 1.0   # mm clouds → m
        P *= scale
        P[np.abs(P[:, 2]) < 1e-9] = np.nan                     # sensors write 0 for "no return"
        fields = {k: np.asarray(v) for k, v in cols.items() if k not in ("x", "y", "z")}
        organized = W and H and W * H == len(P) and H > 1
        return cls(label or Path(path).stem, Path(path), P, fields, H if organized else None,
                   W if organized else None, scale, color)

    def rgb(self):
        """(N, 3) uint8 from a packed float 'rgb' field."""
        packed = self.fields["rgb"].astype(np.float32).view(np.uint32)
        return np.c_[(packed >> 16) & 255, (packed >> 8) & 255, packed & 255].astype(np.uint8)

    def sor_keep(self, k, nsigma):
        key = (k, nsigma)
        if key not in self.sor_cache:
            self.sor_cache[key] = sor_keep_mask(self.P, k, nsigma, z_range=(-np.inf, np.inf))[0]
        return self.sor_cache[key]


class Viewer(QWidget):
    """Cloud list + filters on the left, VTK render window on the right."""

    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.clouds = []
        self.actors = {}
        self.picked = []
        self._update_timer = QTimer(self, singleShot=True, interval=120)
        self._update_timer.timeout.connect(self.render_all)

        side = QWidget()
        sl = QVBoxLayout(side)
        sl.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.ExtendedSelection)
        self.list.itemChanged.connect(lambda *_: self.schedule())
        sl.addWidget(QLabel("Punktwolken (Häkchen = sichtbar)"))
        sl.addWidget(self.list, 1)
        row = QHBoxLayout()
        for text, slot in (("Laden …", self.open_files), ("Entfernen", self.remove_selected),
                           ("Alle aus", lambda: self.set_all_checked(False))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        sl.addLayout(row)

        box = QGroupBox("Darstellung & Filter")
        form = QFormLayout(box)
        self.color_by = QComboBox()
        self.color_by.addItems(["z", "grayValue", RGB, SOLID])
        self.color_by.setEditable(True)
        self.color_by.setToolTip("Einfärbung: z, ein PCD-Feld (grayValue, rad, dot, approach, …), RGB oder einfarbig")
        self.color_by.currentTextChanged.connect(lambda *_: self.schedule())
        form.addRow("Farbe nach", self.color_by)
        self.z_min, self.z_max = QDoubleSpinBox(), QDoubleSpinBox()
        for s, v in ((self.z_min, 0.0), (self.z_max, 10.0)):
            s.setRange(-100.0, 100.0)
            s.setDecimals(3)
            s.setSingleStep(0.01)
            s.setSuffix(" m")
            s.setValue(v)
            s.valueChanged.connect(lambda *_: self.schedule())
        zrow = QHBoxLayout()
        zrow.addWidget(self.z_min)
        zrow.addWidget(QLabel("–"))
        zrow.addWidget(self.z_max)
        auto = QPushButton("Auto")
        auto.setToolTip("Bereich auf das 1–99 %-Perzentil der sichtbaren Wolken setzen")
        auto.clicked.connect(self.auto_range)
        zrow.addWidget(auto)
        form.addRow("z von / bis", zrow)
        self.point_size = QSpinBox()
        self.point_size.setRange(1, 20)
        self.point_size.setValue(3)
        self.point_size.valueChanged.connect(lambda *_: self.schedule())
        form.addRow("Punktgröße", self.point_size)

        self.sor_on = QCheckBox("SOR")
        self.sor_k = QSpinBox()
        self.sor_k.setRange(1, 100)
        self.sor_k.setValue(6)
        self.sor_n = QDoubleSpinBox()
        self.sor_n.setRange(0.0, 10.0)
        self.sor_n.setSingleStep(0.1)
        self.sor_n.setValue(1.0)
        for w in (self.sor_on, self.sor_k, self.sor_n):
            (w.toggled if w is self.sor_on else w.valueChanged).connect(lambda *_: self.schedule())
        srow = QHBoxLayout()
        srow.addWidget(self.sor_on)
        srow.addWidget(QLabel("k"))
        srow.addWidget(self.sor_k)
        srow.addWidget(QLabel("nσ"))
        srow.addWidget(self.sor_n)
        form.addRow("Ausreißer", srow)

        self.mesh_on = QCheckBox("Mesh")
        self.mesh_on.setToolTip("Organisierte Wolken: Pixelgitter; sonst Delaunay in der Bildebene. "
                                "Dreiecke mit längerer Kante bleiben leer.")
        self.max_edge = QDoubleSpinBox()
        self.max_edge.setRange(0.1, 1000.0)
        self.max_edge.setValue(20.0)
        self.max_edge.setSuffix(" mm")
        for w in (self.mesh_on, self.max_edge):
            (w.toggled if w is self.mesh_on else w.valueChanged).connect(lambda *_: self.schedule())
        mrow = QHBoxLayout()
        mrow.addWidget(self.mesh_on)
        mrow.addWidget(QLabel("max. Kante"))
        mrow.addWidget(self.max_edge)
        form.addRow("Oberfläche", mrow)
        sl.addWidget(box)

        tools = QGroupBox("Werkzeuge")
        tl = QVBoxLayout(tools)
        self.measure = QPushButton("Abstand messen")
        self.measure.setCheckable(True)
        self.measure.setToolTip("Zwei Punkte anklicken: Abstand und Δz erscheinen unten")
        self.measure.toggled.connect(self.toggle_measure)
        tl.addWidget(self.measure)
        r2 = QHBoxLayout()
        for text, slot in (("Ansicht zurücksetzen", self.reset_camera), ("Gefiltert exportieren …", self.export),
                           ("In CloudCompare", self.open_cloudcompare)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            r2.addWidget(b)
        tl.addLayout(r2)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        tl.addWidget(self.info)
        sl.addWidget(tools)

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#1e1f22")
        self.plotter.add_axes()

        split = QSplitter()
        split.addWidget(side)
        split.addWidget(self.plotter.interactor)
        split.setStretchFactor(1, 1)
        split.setSizes([340, 900])
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)
        self._fit_camera = True       # fit the view to the data on the next render

    # ---- cloud list -------------------------------------------------------------------------------------------
    def add_cloud(self, path, label=None, visible=True):
        """Load a PCD and add it to the list (a path already shown is reloaded)."""
        path = Path(path)
        for i, c in enumerate(self.clouds):
            if c.path == path:
                self.clouds.pop(i)
                self.list.takeItem(i)
                break
        try:
            cloud = Cloud.load(path, label, PALETTE[len(self.clouds) % len(PALETTE)])
        except Exception as exc:
            self.status.emit(f"PCD nicht lesbar: {path.name}: {exc}")
            return None
        approach = np.unique(cloud.fields.get("approach", []))
        if len(approach) == 1 and int(approach[0]) in APPROACH_COLORS:  # exported A1/A2/A3 clouds
            cloud.color = "#%02x%02x%02x" % APPROACH_COLORS[int(approach[0])]
        self.clouds.append(cloud)
        item = QListWidgetItem(cloud.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if visible else Qt.Unchecked)
        item.setForeground(QColor(cloud.color))
        n = int(np.isfinite(cloud.P[:, 2]).sum())
        item.setToolTip(f"{path}\n{n:,} Punkte"
                        + (f", organisiert {cloud.W}×{cloud.H}" if cloud.H else "")
                        + f"\nFelder: {', '.join(cloud.fields) or '–'}")
        self.list.blockSignals(True)
        self.list.addItem(item)
        self.list.blockSignals(False)
        for f in cloud.fields:
            if self.color_by.findText(f) < 0:
                self.color_by.addItem(f)
        self.schedule()
        return cloud

    def open_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "PCDs laden", "", "Punktwolken (*.pcd)")
        for f in files:
            self.add_cloud(f)
        if files:
            self.auto_range()

    def remove_selected(self):
        for item in sorted(self.list.selectedItems(), key=self.list.row, reverse=True):
            i = self.list.row(item)
            self.list.takeItem(i)
            self.clouds.pop(i)
        self.schedule()

    def set_all_checked(self, on):
        self.list.blockSignals(True)
        for i in range(self.list.count()):
            self.list.item(i).setCheckState(Qt.Checked if on else Qt.Unchecked)
        self.list.blockSignals(False)
        self.schedule()

    def visible(self):
        return [c for i, c in enumerate(self.clouds) if self.list.item(i).checkState() == Qt.Checked]

    # ---- rendering --------------------------------------------------------------------------------------------
    def schedule(self):
        self._update_timer.start()

    def mask(self, c):
        """Valid points of a cloud after z filter and SOR."""
        z = c.P[:, 2]
        with np.errstate(invalid="ignore"):
            m = np.isfinite(z) & (z >= self.z_min.value()) & (z <= self.z_max.value())
        if self.sor_on.isChecked():
            m &= c.sor_keep(self.sor_k.value(), self.sor_n.value())
        return m

    def geometry(self, c):
        """(PolyData, point ids) of the filtered cloud, as mesh or points."""
        m = self.mask(c)
        if self.mesh_on.isChecked():
            tris = grid_triangles(c.H, c.W) if c.H else image_plane_triangles(c.P, m)
            tris = filter_triangles(c.P, tris, m, self.max_edge.value() * 1e-3) if len(tris) else tris
            if len(tris):
                ids, inv = np.unique(tris, return_inverse=True)
                faces = np.c_[np.full(len(tris), 3), inv.reshape(-1, 3)].ravel()
                return pv.PolyData(c.P[ids], faces), ids
        ids = np.flatnonzero(m)
        return pv.PolyData(c.P[ids]), ids

    def render_all(self):
        """Redraw every visible cloud with the current filters."""
        for actor in self.actors.values():
            self.plotter.remove_actor(actor, render=False)
        self.actors = {}
        mode = self.color_by.currentText().strip()
        total = 0
        for c in self.visible():
            poly, ids = self.geometry(c)
            if poly.n_points == 0:
                continue
            total += len(ids)
            kw = dict(point_size=self.point_size.value(), render_points_as_spheres=False, show_scalar_bar=False,
                      name=str(c.path))
            if mode == RGB and "rgb" in c.fields:
                poly["rgb"] = c.rgb()[ids]
                kw.update(scalars="rgb", rgb=True)
            elif mode == "z":
                poly["z [m]"] = c.P[ids, 2]
                kw.update(scalars="z [m]", cmap="viridis", clim=(self.z_min.value(), self.z_max.value()))
            elif mode in c.fields:
                vals = c.fields[mode][ids].astype(float)
                poly[mode] = vals
                fin = vals[np.isfinite(vals)]
                clim = tuple(np.percentile(fin, [1, 99])) if fin.size else (0, 1)
                kw.update(scalars=mode, cmap="gray" if mode in ("grayValue", "intensity") else "turbo", clim=clim)
            else:
                kw.update(color=c.color)
            self.actors[c.path] = self.plotter.add_mesh(poly, **kw)
        if self.actors and mode not in (SOLID, RGB):
            first = next(iter(self.actors.values()))
            self.plotter.add_scalar_bar(title=mode, mapper=first.mapper, vertical=True, n_labels=5)
        elif self.plotter.scalar_bars:
            self.plotter.remove_scalar_bar()
        if self._fit_camera and self.actors:
            self._fit_camera = False
            self.reset_camera()
        self.plotter.render()
        self.status.emit(f"{len(self.actors)} Wolke(n), {total:,} Punkte sichtbar")

    def auto_range(self):
        zs = [c.P[:, 2][np.isfinite(c.P[:, 2])] for c in (self.visible() or self.clouds)]
        zs = np.concatenate(zs) if zs else np.array([])
        if zs.size:
            lo, hi = np.percentile(zs, [1, 99])
            pad = 0.05 * (hi - lo) + 1e-3
            self.z_min.setValue(lo - pad)
            self.z_max.setValue(hi + pad)
        self._fit_camera = True
        self.schedule()

    def reset_camera(self):
        """Sensor view: from the camera origin towards the centre of the visible data, image up = −y."""
        b = np.asarray(self.plotter.bounds, float) if self.actors else np.array([0, 0, 0, 0, 0.5, 1.5])
        centre = ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2)
        self.plotter.camera_position = [(0, 0, 0), centre, (0, -1, 0)]
        self.plotter.camera.view_angle = 55.0  # a little tighter than the ToF field of view
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    # ---- tools ------------------------------------------------------------------------------------------------
    def toggle_measure(self, on):
        if on:
            self.picked = []
            self.plotter.enable_point_picking(callback=self._picked, show_message=False, left_clicking=True,
                                              show_point=True, picker="point", point_size=12, color="orange")
            self.info.setText("Ersten Punkt anklicken …")
        else:
            self.plotter.disable_picking()
            self.plotter.remove_actor("measure_line")
            self.info.setText("")

    def _picked(self, point, *_):
        self.picked = (self.picked + [np.asarray(point, float)])[-2:]
        if len(self.picked) == 1:
            p = self.picked[0]
            self.info.setText(f"P1 = ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m — zweiten Punkt anklicken …")
            return
        a, b = self.picked
        self.plotter.add_mesh(pv.Line(a, b), color="orange", line_width=3, name="measure_line")
        self.info.setText(f"Abstand {np.linalg.norm(b - a) * 1e3:.2f} mm   Δz {(b[2] - a[2]) * 1e3:+.2f} mm\n"
                          f"P1 z = {a[2]:.4f} m   P2 z = {b[2]:.4f} m")
        self.picked = []

    def export(self):
        """Filtered selected (or all visible) clouds as PCD (original units and fields) or mesh as PLY/STL/OBJ."""
        clouds = [self.clouds[self.list.row(i)] for i in self.list.selectedItems()] or self.visible()
        if not clouds:
            return
        default = str(clouds[0].path.with_name(clouds[0].path.stem + "_filtered.pcd"))
        path, _ = QFileDialog.getSaveFileName(self, "Gefiltert exportieren", default,
                                              "PCD (*.pcd);;PLY (*.ply);;STL (*.stl);;OBJ (*.obj)")
        if not path:
            return
        path = Path(path)
        for k, c in enumerate(clouds):
            out = path if len(clouds) == 1 else path.with_name(f"{path.stem}_{k}_{c.path.stem}{path.suffix}")
            poly, ids = self.geometry(c)
            if out.suffix.lower() == ".pcd":
                cols = {"x": c.P[ids, 0] / c.scale, "y": c.P[ids, 1] / c.scale, "z": c.P[ids, 2] / c.scale}
                cols.update({f: v[ids] for f, v in c.fields.items() if np.asarray(v).ndim == 1})
                write_pcd(out, cols)
            else:
                poly.save(str(out))
            self.status.emit(f"Exportiert: {out}")

    def open_cloudcompare(self):
        clouds = [self.clouds[self.list.row(i)] for i in self.list.selectedItems()] or self.visible()
        cmd = find_cloudcompare()
        if cmd is None:
            QMessageBox.information(self, "CloudCompare", "CloudCompare wurde nicht gefunden "
                                    "(PATH, Flatpak org.cloudcompare.CloudCompare oder Standard-Installationsordner).")
            return
        subprocess.Popen(cmd + [str(c.path) for c in clouds],
                         start_new_session=sys.platform != "win32")

    def closeEvent(self, event):
        self.plotter.close()
        super().closeEvent(event)

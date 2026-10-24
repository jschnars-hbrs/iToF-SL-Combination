"""Main window: parameter panel (discovered from the code), calibration, test runs, 3-D viewer, log."""

import ast
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures come pickled from the worker and are embedded explicitly
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT  # noqa: E402
from PySide6.QtCore import (QFileSystemWatcher, QObject, QProcess, QProcessEnvironment, QSettings,  # noqa: E402
                            QStandardPaths, Qt, QTimer, QUrl, Signal)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QKeySequence, QPixmap  # noqa: E402
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,  # noqa: E402
                               QDialogButtonBox, QDockWidget, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                               QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QMenu,
                               QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSplitter,
                               QTableWidget, QTableWidgetItem, QTabWidget, QToolBar, QVBoxLayout, QWidget)

from dot_calibration import DotCalibration, cache_tag  # noqa: E402
from gui import params as P  # noqa: E402
from gui.inputs import CODE_DIR, ROOT, Scene, collect, from_session_path, to_session_path  # noqa: E402
from gui.viewer import Viewer  # noqa: E402

CAL_ROOT = ROOT / "Calibrations"
RUNS_ROOT = ROOT / "Results" / "gui"
BRIDGE = CODE_DIR / "itof_raw_bridge.py"
BRIDGE_PATH_ARGS = ("repo", "raw", "kalibration", "dot_calibration", "out", "name", "params")
MANUAL_PATH_ARGS = ("input", "kalibration", "dot_calibration", "dot_calibration_root", "output", "start_dist",
                    "step_dist", "start", "stop", "step", "frames", "gt_offset_mm", "max_gt")
APPROACH_COLORS = {1: "#1f77b4", 2: "#2ca02c", 3: "#d62728"}


# ---- helpers ----------------------------------------------------------------------------------------------------
def camera_entries():
    """{camera: {sl_cal, tof_cal}} from calibrate.CAMERAS via AST (ROOT / "..." paths evaluated)."""
    tree = ast.parse((CODE_DIR / "calibrate.py").read_text(encoding="utf-8"))
    node = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "CAMERAS")

    def path(e):
        if isinstance(e, ast.Name) and e.id == "ROOT":
            return ROOT
        if isinstance(e, ast.Name) and e.id == "CAL_ROOT":
            return CAL_ROOT
        if isinstance(e, ast.Constant) and isinstance(e.value, str):
            return e.value
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Div):
            return Path(path(e.left)) / path(e.right)
        raise ValueError

    out = {}
    for k, v in zip(node.keys, node.values):
        entry = {}
        for kw in getattr(v, "keywords", []):
            if kw.arg in ("sl_cal", "tof_cal"):
                try:
                    entry[kw.arg] = Path(path(kw.value))
                except (ValueError, TypeError):
                    pass
        out[k.value] = entry
    return out


def cal_metadata(cal_file):
    """The metadata block at the start of a calibration.json without reading the whole file."""
    with open(cal_file, "rb") as f:
        head = f.read(65536).decode("utf-8", "ignore")
    i = head.find('"metadata"')
    if i < 0:
        return {}
    try:
        return json.JSONDecoder().raw_decode(head, head.find("{", i))[0]
    except ValueError:
        return {}


def find_itof_defaults():
    """Best guess of (repo, python) of the itofcameracalibration project."""
    bases = [Path.home() / "Documents" / "Calibrations", Path.home() / "Documents"]
    repo = None
    for b in bases:
        for cand in (b / "itofcameracalibration_new" / "itofcameracalibration", b / "itofcameracalibration"):
            if (cand / "tof_processing").is_dir():
                repo = cand
                break
        if repo:
            break
    python = None
    if repo:
        for d in [repo, *list(repo.parents)[:3]]:
            for venv in [d / ".venv", *d.glob("*/.venv")]:
                for exe in (venv / "bin" / "python", venv / "Scripts" / "python.exe"):
                    if exe.exists():
                        python = exe
                        break
                if python:
                    break
            if python:
                break
    return repo, python


def bold(item):
    f = item.font()
    f.setBold(True)
    item.setFont(f)
    return item


# ---- job runner -------------------------------------------------------------------------------------------------
class Runner(QObject):
    """Runs gui.worker in a QProcess, streams its output, turns GUI_EVENT lines into signals."""

    line = Signal(str)
    event = Signal(dict)
    finished = Signal(int, Path)

    def __init__(self):
        super().__init__()
        self.proc = None
        self.run_dir = None
        self._buf = ""
        self._log = None

    def busy(self):
        return self.proc is not None and self.proc.state() != QProcess.NotRunning

    def start(self, job, overrides, label):
        """Write job.json + overrides.json into a fresh run folder and start the worker."""
        self.run_dir = RUNS_ROOT / f"{datetime.now():%Y-%m-%d_%H%M%S}_{label}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "job.json").write_text(json.dumps(job, indent=1, default=str), encoding="utf-8")
        ov = self.run_dir / "overrides.json"
        ov.write_text(json.dumps(overrides, indent=1), encoding="utf-8")
        self._log = open(self.run_dir / "log.txt", "w", encoding="utf-8")
        env = QProcessEnvironment.systemEnvironment()
        for k, v in (("ITOF_OVERRIDES", str(ov)), ("PYTHONUNBUFFERED", "1"), ("PYTHONIOENCODING", "utf-8"),
                     ("MPLBACKEND", "Agg")):
            env.insert(k, v)
        self.proc = QProcess(self)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(CODE_DIR))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read)
        self.proc.finished.connect(self._finished)
        self._buf = ""
        self.line.emit(f"▶ {label}   →  {self.run_dir}")
        self.proc.start(sys.executable, ["-m", "gui.worker", str(self.run_dir / "job.json")])

    def stop(self):
        if self.busy():
            self.proc.kill()
            self.line.emit("■ abgebrochen")

    def _read(self):
        self._buf += bytes(self.proc.readAllStandardOutput()).decode("utf-8", "replace")
        *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            ln = ln.rstrip("\r")
            if self._log:
                self._log.write(ln + "\n")
            if ln.startswith("GUI_EVENT|"):
                try:
                    self.event.emit(json.loads(ln.split("|", 1)[1]))
                except ValueError:
                    pass
            else:
                self.line.emit(ln)

    def _finished(self, code, _status):
        self._read()
        if self._buf:
            self.line.emit(self._buf)
        if self._log:
            self._log.close()
            self._log = None
        self.finished.emit(code, self.run_dir)


# ---- figures ----------------------------------------------------------------------------------------------------
class FigureBrowser(QSplitter):
    """List of figures (per scene) + interactive matplotlib canvas (falls back to the PNG)."""

    def __init__(self):
        super().__init__()
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._show)
        self.view = QWidget()
        self.view_lay = QVBoxLayout(self.view)
        self.view_lay.setContentsMargins(0, 0, 0, 0)
        self.addWidget(self.list)
        self.addWidget(self.view)
        self.setStretchFactor(1, 1)
        self.setSizes([260, 800])
        self._fig = None

    def clear(self):
        self.list.clear()
        self._clear_view()

    def add_dir(self, scene, folder):
        """All PNGs of a figure folder; the pickle next to a PNG makes it interactive."""
        for png in sorted(Path(folder).glob("*.png")):
            item = QListWidgetItem(f"{scene} · {png.stem}")
            item.setData(Qt.UserRole, str(png))
            item.setToolTip(str(png))
            self.list.addItem(item)
        if self.list.currentRow() < 0 and self.list.count():
            self.list.setCurrentRow(0)

    def _clear_view(self):
        while self.view_lay.count():
            w = self.view_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._fig = None

    def _show(self, item, _prev=None):
        self._clear_view()
        if item is None:
            return
        png = Path(item.data(Qt.UserRole))
        pkl = png.with_suffix(".fig.pkl")
        if pkl.exists():
            try:
                with open(pkl, "rb") as f:
                    fig = pickle.load(f)
                plt.close(fig)
                canvas = FigureCanvasQTAgg(fig)
                self.view_lay.addWidget(NavigationToolbar2QT(canvas, self.view))
                self.view_lay.addWidget(canvas, 1)
                canvas.draw_idle()
                self._fig = fig
                return
            except Exception as exc:
                print(f"figure pickle {pkl.name}: {exc}")
        label = QLabel()
        label.setAlignment(Qt.AlignCenter)
        label.setPixmap(QPixmap(str(png)))
        scroll = QScrollArea()
        scroll.setWidget(label)
        scroll.setWidgetResizable(True)
        self.view_lay.addWidget(scroll)


# ---- tabs -------------------------------------------------------------------------------------------------------
class CalibrationTab(QWidget):
    """Compute a calibration for a camera of calibrate.CAMERAS, or pick an existing one."""

    run = Signal()
    activate = Signal(Path)

    def __init__(self):
        super().__init__()
        self.cameras = {}
        top = QGroupBox("Neue Kalibrierung (Godbaz §4.1)")
        form = QFormLayout(top)
        self.camera = QComboBox()
        self.camera.currentTextChanged.connect(self._camera_changed)
        form.addRow("Kamera (calibrate.CAMERAS)", self.camera)
        self.sl_dir, self.tof_dir = QLineEdit(), QLineEdit()
        for label, edit in (("SL-Ordner", self.sl_dir), ("ToF-Ordner", self.tof_dir)):
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("…")
            b.setFixedWidth(32)
            b.clicked.connect(lambda _=False, e=edit: self._browse(e))
            row.addWidget(b)
            form.addRow(label, row)
        self.name = QLineEdit()
        self.name.setPlaceholderText("Calibrations/<Name>/calibration.json")
        form.addRow("Speichern als", self.name)
        self.plots = QCheckBox("Kalibrier-Plots erzeugen und speichern")
        self.plots.setChecked(True)
        form.addRow("", self.plots)
        self.go = QPushButton("Kalibrierung berechnen")
        self.go.setToolTip("Optionen: Parameter-Seitenleiste → »Kalibrierung« und »ToF-Vorverarbeitung«")
        self.go.clicked.connect(self.run)
        form.addRow("", self.go)

        existing = QGroupBox("Vorhandene Kalibrierungen (Doppelklick = für Tests verwenden)")
        el = QVBoxLayout(existing)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Name", "Erstellt", "Kamera", "Mode", "SOR", "Pfad"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.setColumnHidden(5, True)
        self.table.cellDoubleClicked.connect(lambda r, _c: self.activate.emit(Path(self.table.item(r, 5).text())))
        self.table.currentCellChanged.connect(lambda r, *_: self._show_plots(r))
        el.addWidget(self.table)
        b = QPushButton("Liste aktualisieren")
        b.clicked.connect(self.refresh)
        el.addWidget(b)

        self.figures = FigureBrowser()
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(top)
        ll.addWidget(existing, 1)
        split = QSplitter()
        split.addWidget(left)
        split.addWidget(self.figures)
        split.setSizes([700, 700])
        lay = QVBoxLayout(self)
        lay.addWidget(split)
        self.reload_cameras()
        self.refresh()

    def reload_cameras(self):
        current = self.camera.currentText()
        try:
            self.cameras = camera_entries()
        except Exception as exc:
            print(f"CAMERAS not readable: {exc}")
            return
        self.camera.blockSignals(True)
        self.camera.clear()
        self.camera.addItems(list(self.cameras))
        self.camera.blockSignals(False)
        if current in self.cameras:
            self.camera.setCurrentText(current)
        else:
            self._camera_changed(self.camera.currentText())

    def _camera_changed(self, cam):
        e = self.cameras.get(cam, {})
        self.sl_dir.setText(str(e.get("sl_cal", "")))
        self.tof_dir.setText(str(e.get("tof_cal", "")))
        self.name.clear()

    def _browse(self, edit):
        d = QFileDialog.getExistingDirectory(self, "Ordner wählen", edit.text() or str(ROOT))
        if d:
            edit.setText(str(Path(d)))

    def default_name(self, sor, lpc):
        """As in the notebook: <camera>_paperlike<cache_tag>_<dd.mm>."""
        return f"{self.camera.currentText()}_paperlike{cache_tag(sor, lpc)}_{datetime.now():%d.%m}"

    def refresh(self):
        files = sorted(CAL_ROOT.glob("*/calibration.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.table.setRowCount(len(files))
        for r, f in enumerate(files):
            md = cal_metadata(f)
            sor = md.get("sor") or {}
            vals = [f.parent.name, str(md.get("created", ""))[:16].replace("T", " "), md.get("camera", ""),
                    md.get("mode", ""), f"{sor.get('k')}/{sor.get('nsigma')}" if sor.get("enabled") else "–", str(f)]
            for c, v in enumerate(vals):
                self.table.setItem(r, c, QTableWidgetItem(str(v)))
        return files

    def _show_plots(self, row):
        if row < 0 or self.table.item(row, 5) is None:
            return
        folder = Path(self.table.item(row, 5).text()).parent
        self.figures.clear()
        self.figures.add_dir(folder.name, folder)


class SceneTable(QTableWidget):
    """Test scenes: enabled, name, type, ToF/RAW, SL, GT. Accepts dropped files and folders."""

    COLS = ["", "Szene", "Typ", "ToF / RAW", "SL-Bild", "GT [m]"]
    dropped = Signal(list)
    show_in_viewer = Signal(Path)

    def __init__(self):
        super().__init__(0, len(self.COLS))
        self.scenes = []
        self.setHorizontalHeaderLabels(self.COLS)
        self.verticalHeader().hide()
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        h = self.horizontalHeader()
        h.setSectionResizeMode(QHeaderView.ResizeToContents)
        h.setSectionResizeMode(3, QHeaderView.Stretch)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.itemChanged.connect(self._edited)
        self.cellDoubleClicked.connect(self._double)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._menu)

    # drag & drop
    def dragEnterEvent(self, e):
        e.acceptProposedAction() if e.mimeData().hasUrls() else super().dragEnterEvent(e)

    dragMoveEvent = dragEnterEvent

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            self.dropped.emit([Path(u.toLocalFile()) for u in e.mimeData().urls()])
            e.acceptProposedAction()

    def set_scenes(self, scenes):
        self.scenes = list(scenes)
        self.blockSignals(True)
        self.setRowCount(len(self.scenes))
        for r, s in enumerate(self.scenes):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            chk.setCheckState(Qt.Checked if s.enabled else Qt.Unchecked)
            self.setItem(r, 0, chk)
            self.setItem(r, 1, QTableWidgetItem(s.name))
            src = s.raw or s.tof
            for c, text, editable in ((2, "RAW" if s.raw else "PCD", False),
                                      (3, to_session_path(src) or "", False),
                                      (4, to_session_path(s.sl) if s.sl else "aus PCD-Grauwerten", False),
                                      (5, "" if s.gt is None else f"{s.gt:g}", True)):
                it = QTableWidgetItem(text)
                if not editable:
                    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                if c == 4 and not s.sl:
                    it.setForeground(QColor("#888888"))
                if c == 3:
                    it.setToolTip(str(src))
                elif c == 4:
                    it.setToolTip(f"{s.sl or 'Grauwerte der PCD'}\nDoppelklick: SL-Bild wählen")
                self.setItem(r, c, it)
        self.blockSignals(False)

    def _edited(self, item):
        s = self.scenes[item.row()]
        if item.column() == 0:
            s.enabled = item.checkState() == Qt.Checked
        elif item.column() == 1:
            s.name = item.text().strip() or s.name
        elif item.column() == 5:
            try:
                s.gt = float(item.text().replace(",", ".")) if item.text().strip() else None
            except ValueError:
                self.set_scenes(self.scenes)

    def _double(self, row, col):
        if col == 4:
            self.choose_sl(row)

    def choose_sl(self, row):
        f, _ = QFileDialog.getOpenFileName(self, "SL-Bild wählen", str(self.scenes[row].tof or ROOT),
                                           "Bilder (*.exr *.png *.tif *.tiff)")
        if f:
            self.scenes[row].sl = Path(f)
            self.set_scenes(self.scenes)

    def _menu(self, pos):
        rows = sorted({i.row() for i in self.selectedItems()})
        if not rows:
            return
        m = QMenu(self)
        m.addAction("SL-Bild wählen …", lambda: self.choose_sl(rows[0]))
        m.addAction("SL aus PCD-Grauwerten", lambda: self._set_sl_none(rows))
        if self.scenes[rows[0]].tof:
            m.addAction("ToF im Viewer zeigen", lambda: self.show_in_viewer.emit(self.scenes[rows[0]].tof))
        m.addSeparator()
        m.addAction("Entfernen", self.remove_selected)
        m.exec(self.viewport().mapToGlobal(pos))

    def _set_sl_none(self, rows):
        for r in rows:
            self.scenes[r].sl = None
        self.set_scenes(self.scenes)

    def remove_selected(self):
        rows = {i.row() for i in self.selectedItems()}
        self.set_scenes([s for r, s in enumerate(self.scenes) if r not in rows])


class TestTab(QWidget):
    """Scenes + run options, result table, figures."""

    run = Signal()

    def __init__(self):
        super().__init__()
        self.table = SceneTable()
        self.table.dropped.connect(self.add_paths)
        btns = QHBoxLayout()
        for text, slot in (("Dateien …", self._add_files), ("Ordner …", self._add_folder),
                           ("Entfernen", self.table.remove_selected), ("Leeren", lambda: self.table.set_scenes([]))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btns.addWidget(b)
        btns.addStretch(1)
        hint = QLabel("PCD, RAW (.npy / Pos*_set0-Ordner), EXR/PNG hineinziehen; ohne SL-Bild werden die "
                      "Grauwerte der PCD genutzt.")
        hint.setStyleSheet("color: #888;")

        opts = QHBoxLayout()
        self.approach = {}
        for n, text in ((1, "A1 Konsistenzfehler"), (2, "A2 ML-Fusion"), (3, "A3 AB-Trail")):
            cb = QCheckBox(text)
            cb.setChecked(True)
            cb.setStyleSheet(f"QCheckBox {{ color: {APPROACH_COLORS[n]}; font-weight: bold; }}")
            self.approach[n] = cb
            opts.addWidget(cb)
        opts.addSpacing(16)
        self.plots = QCheckBox("Plots")
        self.plots.setChecked(True)
        self.export = QCheckBox("PCDs exportieren")
        self.export.setChecked(True)
        self.export.setToolTip("reconstruction.export_pcds; Optionen in der Seitenleiste → »PCD-Export«")
        opts.addWidget(self.plots)
        opts.addWidget(self.export)
        opts.addStretch(1)
        self.go = QPushButton("▶  Ausführen")
        self.go.setMinimumWidth(140)
        self.go.clicked.connect(self.run)
        opts.addWidget(self.go)

        self.results = QTableWidget(0, 10)
        self.results.setHorizontalHeaderLabels(["Szene", "A", "gültig", "z̄ [m]", "σ_z [mm]", "GT [m]",
                                                "Fehler [mm]", "|Fehler| [mm]", "σ_Fehler [mm]", "min / max [mm]"])
        self.results.verticalHeader().hide()
        self.results.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.results.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.results.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.figures = FigureBrowser()

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.addLayout(btns)
        tl.addWidget(hint)
        tl.addWidget(self.table, 1)
        tl.addLayout(opts)
        bottom = QSplitter(Qt.Vertical)
        bottom.addWidget(self.results)
        bottom.addWidget(self.figures)
        bottom.setSizes([160, 500])
        split = QSplitter(Qt.Vertical)
        split.addWidget(top)
        split.addWidget(bottom)
        split.setSizes([260, 660])
        lay = QVBoxLayout(self)
        lay.addWidget(split)

    def add_paths(self, paths):
        scenes, unused = collect(paths, parse_gt=DotCalibration.parse_distance_from_name)
        known = {(s.tof, s.raw) for s in self.table.scenes}
        self.table.set_scenes(self.table.scenes + [s for s in scenes if (s.tof, s.raw) not in known])
        if unused:
            QMessageBox.information(self, "Ohne ToF", "Diese Bilder haben keine passende PCD/RAW und wurden "
                                    "ignoriert:\n" + "\n".join(p.name for p in unused[:20]))

    def _add_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Testdateien", str(ROOT),
                                                "Messdaten (*.pcd *.npy *.exr *.png *.tif *.tiff)")
        if files:
            self.add_paths([Path(f) for f in files])

    def _add_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Testordner", str(ROOT))
        if d:
            self.add_paths([Path(d)])

    def selected_approaches(self):
        return [n for n, cb in self.approach.items() if cb.isChecked()]

    def add_result(self, row):
        r = self.results.rowCount()
        self.results.insertRow(r)

        def mm(key):
            v = row.get(key)
            return "" if v is None or v != v else f"{v * 1e3:+.1f}" if key in ("mean_m",) else f"{v * 1e3:.1f}"

        vals = [row["scene"], f"A{row['approach']}", f"{row['n_valid']}/{row['n_total']}",
                "" if row.get("mean_z_m") != row.get("mean_z_m") else f"{row['mean_z_m']:.4f}",
                "" if row.get("std_z_m") != row.get("std_z_m") else f"{row['std_z_m'] * 1e3:.1f}",
                "" if row.get("gt_m") is None else f"{row['gt_m']:.3f}", mm("mean_m"), mm("abs_mean_m"), mm("std_m"),
                "" if row.get("min_m") is None else f"{row['min_m'] * 1e3:+.1f} / {row['max_m'] * 1e3:+.1f}"]
        for c, v in enumerate(vals):
            it = QTableWidgetItem(v)
            if c == 1:
                it.setForeground(QColor(APPROACH_COLORS.get(row["approach"], "#000")))
                bold(it)
            if c >= 2:
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.results.setItem(r, c, it)


# ---- settings ---------------------------------------------------------------------------------------------------
class SettingsDialog(QDialog):
    """Machine-local paths (stored per user, not in the repository)."""

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Einstellungen")
        self.settings = settings
        form = QFormLayout(self)
        form.addRow(QLabel("<b>RAW → PCD</b> (Projekt itofcameracalibration, eigene Python-Umgebung)"))
        self.edits = {}
        for key, label, kind, flt in (("itof/repo", "Paket-Ordner (tof_processing/, utils/)", "dir", ""),
                                      ("itof/python", "Python dieser Umgebung", "file", "Python (python*)"),
                                      ("itof/kalibration", "iToF-Kalibrierung (HDF5)", "file", "HDF5 (*.hdf5 *.h5)")):
            edit = QLineEdit(str(settings.value(key, "") or ""))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            b = QPushButton("…")
            b.setFixedWidth(32)
            b.clicked.connect(lambda _=False, e=edit, k=kind, f=flt: self._browse(e, k, f))
            row.addWidget(b)
            form.addRow(label, row)
            self.edits[key] = edit
        auto = QPushButton("Automatisch suchen")
        auto.clicked.connect(self._auto)
        form.addRow("", auto)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self.resize(720, 0)

    def _browse(self, edit, kind, flt):
        start = edit.text() or str(Path.home())
        if kind == "dir":
            p = QFileDialog.getExistingDirectory(self, "Ordner", start)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Datei", start, flt)
        if p:
            edit.setText(str(Path(p)))

    def _auto(self):
        repo, python = find_itof_defaults()
        if repo:
            self.edits["itof/repo"].setText(str(repo))
        if python:
            self.edits["itof/python"].setText(str(python))

    def accept(self):
        for k, e in self.edits.items():
            self.settings.setValue(k, e.text().strip())
        super().accept()


# ---- main window ------------------------------------------------------------------------------------------------
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("iToF + SL Fusion")
        self.settings = QSettings("iToF-SL-Combination", "FusionGUI")
        if not self.settings.value("itof/repo"):
            repo, python = find_itof_defaults()
            self.settings.setValue("itof/repo", str(repo or ""))
            self.settings.setValue("itof/python", str(python or ""))
        self.runner = Runner()
        self.runner.line.connect(self.log_line)
        self.runner.event.connect(self.on_event)
        self.runner.finished.connect(self.on_finished)

        # parameter panel: everything here is read from the source files
        self.panel = P.ParamPanel([
            P.Section("approaches", "Approaches (approaches.py)",
                      lambda: P.module_constants(CODE_DIR / "approaches.py", exclude=("GT_DISTANCE",))),
            P.Section("dotcal", "ToF-Vorverarbeitung (DotCalibration)",
                      lambda: P.class_attributes(CODE_DIR / "dot_calibration.py", "DotCalibration",
                                                 exclude=("light_origin",))),
            P.Section("calibrate", "Kalibrierung (calibrate.calibrate)",
                      lambda: P.function_keywords(CODE_DIR / "calibrate.py", "calibrate", exclude=("show_blobs",))),
            P.Section("tilt", "SL-Tilt-Korrektur (correct_cal)",
                      lambda: P.function_keywords(CODE_DIR / "sl_tilt_correction.py", "correct_cal")),
            P.Section("export", "PCD-Export (export_pcds)",
                      lambda: P.function_keywords(CODE_DIR / "reconstruction.py", "export_pcds")),
            P.Section("raw", "RAW → PCD (itof_raw_bridge + binary2pcd_manual)", self._raw_params),
        ])
        dock = QDockWidget("Parameter", self)
        dock.setObjectName("params")
        dock.setWidget(self.panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self.cal_tab = CalibrationTab()
        self.cal_tab.run.connect(self.run_calibration)
        self.cal_tab.activate.connect(self.set_active_calibration)
        self.test_tab = TestTab()
        self.test_tab.run.connect(self.run_test)
        self.viewer = Viewer()
        self.viewer.status.connect(lambda s: self.statusBar().showMessage(s, 6000))
        self.test_tab.table.show_in_viewer.connect(lambda p: (self.viewer.add_cloud(p), self.tabs.setCurrentWidget(self.viewer)))
        self.tabs = QTabWidget()
        self.tabs.addTab(self.cal_tab, "Kalibrierung")
        self.tabs.addTab(self.test_tab, "Test")
        self.tabs.addTab(self.viewer, "3D-Viewer")
        self.tabs.currentChanged.connect(self._tab_changed)
        self.setCentralWidget(self.tabs)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(50000)
        self.log.setStyleSheet("font-family: monospace; font-size: 9pt;")
        log_dock = QDockWidget("Log", self)
        log_dock.setObjectName("log")
        log_dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        tb = QToolBar("Haupt")
        tb.setObjectName("main")
        self.addToolBar(tb)
        tb.addWidget(QLabel(" Kalibrierung für Tests: "))
        self.active_cal = QComboBox()
        self.active_cal.setMinimumWidth(380)
        self.active_cal.setToolTip("calibration.json, die Test-Läufe und RAW-Unwrapping (Modus dots) verwenden")
        tb.addWidget(self.active_cal)
        tb.addSeparator()
        self.stop_btn = QPushButton("■ Stopp")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.runner.stop)
        tb.addWidget(self.stop_btn)
        tb.addSeparator()
        for text, slot, key in (("Session laden …", self.load_session, QKeySequence.Open),
                                ("Session speichern …", self.save_session, QKeySequence.Save),
                                ("Parameter zurücksetzen", self.panel.reset, None),
                                ("Ergebnisordner", self.open_results, None),
                                ("Einstellungen", self.edit_settings, None)):
            a = QAction(text, self)
            a.triggered.connect(slot)
            if key:
                a.setShortcut(key)
            tb.addAction(a)
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(220)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        # rebuild the parameter forms when the code changes
        self.watcher = QFileSystemWatcher(self)
        self._rebuild_timer = QTimer(self, singleShot=True, interval=400)
        self._rebuild_timer.timeout.connect(self._code_changed)
        self.watcher.fileChanged.connect(lambda *_: self._rebuild_timer.start())
        self._watch()

        self.refresh_calibrations()
        self.resize(1600, 1000)
        self.restore_session(self._autosave_path(), quiet=True)
        geo = self.settings.value("window/geometry")
        if geo is not None:
            self.restoreGeometry(geo)
            self.restoreState(self.settings.value("window/state"))

    # ---- parameter sources ------------------------------------------------------------------------------------
    def _manual_converter(self):
        repo = self.settings.value("itof/repo", "")
        f = Path(repo) / "utils" / "binary2pcd_manual.py" if repo else None
        return f if f and f.is_file() else None

    def _raw_params(self):
        params = P.argparse_options(BRIDGE, exclude=BRIDGE_PATH_ARGS)
        manual = self._manual_converter()
        if manual:
            extra = P.argparse_options(manual, exclude=MANUAL_PATH_ARGS)
            for p in extra:
                p.block = 1
            params += extra
        return params

    def _sources(self):
        files = [CODE_DIR / n for n in ("approaches.py", "dot_calibration.py", "calibrate.py",
                                        "sl_tilt_correction.py", "reconstruction.py", "itof_raw_bridge.py")]
        manual = self._manual_converter()
        return [str(f) for f in files + ([manual] if manual else []) if f.exists()]

    def _watch(self):
        if self.watcher.files():
            self.watcher.removePaths(self.watcher.files())
        self.watcher.addPaths(self._sources())

    def _code_changed(self):
        self._watch()  # editors that save by renaming drop the watch
        self.panel.rebuild()
        self.cal_tab.reload_cameras()
        self.statusBar().showMessage("Code geändert: Parameter neu eingelesen", 5000)

    # ---- calibrations -----------------------------------------------------------------------------------------
    def refresh_calibrations(self, select=None):
        current = select or self.active_cal.currentData()
        files = self.cal_tab.refresh()
        self.active_cal.blockSignals(True)
        self.active_cal.clear()
        for f in files:
            self.active_cal.addItem(f.parent.name, str(f))
        self.active_cal.blockSignals(False)
        if current:
            i = self.active_cal.findData(str(current))
            if i >= 0:
                self.active_cal.setCurrentIndex(i)

    def set_active_calibration(self, path):
        i = self.active_cal.findData(str(path))
        if i >= 0:
            self.active_cal.setCurrentIndex(i)
            self.statusBar().showMessage(f"Kalibrierung für Tests: {Path(path).parent.name}", 5000)
            self.tabs.setCurrentWidget(self.test_tab)

    # ---- running ----------------------------------------------------------------------------------------------
    def _start(self, job, label):
        if self.runner.busy():
            QMessageBox.warning(self, "Läuft noch", "Es läuft bereits ein Job.")
            return False
        job["class_overrides"] = self.panel.changed_values("dotcal")
        self.runner.start(job, {"approaches": self.panel.changed_values("approaches")}, label)
        self.cal_tab.go.setEnabled(False)
        self.test_tab.go.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        return True

    def run_calibration(self):
        t = self.cal_tab
        kwargs = self.panel.values("calibrate")
        name = t.name.text().strip() or t.default_name(self.panel.values("dotcal").get("sor"),
                                                      kwargs.get("light_path_correction", False))
        if (CAL_ROOT / name).exists() and QMessageBox.question(
                self, "Überschreiben?", f"Calibrations/{name} existiert bereits. Überschreiben?") != QMessageBox.Yes:
            return
        job = {"kind": "calibrate", "calibrate": {"camera": t.camera.currentText(), "sl_cal": t.sl_dir.text(),
                                                  "tof_cal": t.tof_dir.text(), "kwargs": kwargs, "name": name,
                                                  "plots": t.plots.isChecked()}}
        if self._start(job, f"cal_{name}"):
            t.figures.clear()

    def run_test(self):
        t = self.test_tab
        scenes = [s for s in t.table.scenes if s.enabled]
        cal = self.active_cal.currentData()
        if not scenes:
            QMessageBox.information(self, "Keine Szenen", "Bitte Testdateien hinzufügen (oder hineinziehen).")
            return
        if not cal or not t.selected_approaches():
            QMessageBox.information(self, "Fehlt", "Kalibrierung und mindestens einen Approach wählen.")
            return
        raw_vals = self.panel.values("raw")
        bridge_keys = {p.name for p in P.argparse_options(BRIDGE, exclude=BRIDGE_PATH_ARGS)}
        if any(s.raw for s in scenes):
            missing = [k for k in ("itof/repo", "itof/python", "itof/kalibration") if not self.settings.value(k)]
            if missing:
                QMessageBox.information(self, "RAW-Einstellungen", "Für RAW-Dateien bitte in den Einstellungen "
                                        "Repository, Python und HDF5-Kalibrierung angeben.")
                self.edit_settings()
                return
        job = {"kind": "test", "test": {
            "calibration": cal,
            "light_path_correction": self.panel.values("calibrate").get("light_path_correction", False),
            "tilt": self.panel.values("tilt"),
            "approaches": t.selected_approaches(), "plots": t.plots.isChecked(), "export": t.export.isChecked(),
            "export_kwargs": self.panel.values("export"),
            "scenes": [{"name": s.name, "tof": str(s.tof) if s.tof else None, "raw": str(s.raw) if s.raw else None,
                        "sl": str(s.sl) if s.sl else None, "gt": s.gt} for s in scenes],
            "raw": {"python": self.settings.value("itof/python", ""), "repo": self.settings.value("itof/repo", ""),
                    "kalibration": self.settings.value("itof/kalibration", ""),
                    **{k: v for k, v in raw_vals.items() if k in bridge_keys},
                    "params": {k: v for k, v in raw_vals.items() if k not in bridge_keys}}}}
        label = scenes[0].name if len(scenes) == 1 else f"{len(scenes)}_scenes"
        if self._start(job, "test_" + label.replace("/", "_")):
            t.results.setRowCount(0)
            t.figures.clear()
            self._viewer_scene = None

    def on_event(self, e):
        kind = e.get("type")
        if kind == "scene_start":
            self.progress.setRange(0, e["n"])
            self.progress.setValue(e["i"])
            self.progress.setFormat(f"{e['i'] + 1}/{e['n']}  {e['scene']}")
        elif kind == "result":
            self.test_tab.add_result(e["row"])
        elif kind == "figures":
            browser = self.cal_tab.figures if e["scene"] == "Kalibrierung" else self.test_tab.figures
            browser.add_dir(e["scene"], e["dir"])
        elif kind == "pcd":
            if getattr(self, "_viewer_scene", None) != e["scene"]:
                self.viewer.set_all_checked(False)  # a new scene: show only its clouds
                self._viewer_scene = e["scene"]
            self.viewer.add_cloud(e["path"], e.get("label"))
        elif kind == "calibration":
            self.refresh_calibrations(select=e["path"])
        elif kind == "scene_failed":
            self.log_line(f"✖ Szene {e['scene']} fehlgeschlagen (siehe oben)")

    def on_finished(self, code, run_dir):
        self.cal_tab.go.setEnabled(True)
        self.test_tab.go.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress.setVisible(False)
        msg = "fertig" if code == 0 else f"beendet mit Fehler (Code {code})"
        self.log_line(f"■ {msg}   ({run_dir})")
        self.statusBar().showMessage(f"Job {msg}", 8000)
        if self.viewer.clouds:
            self.viewer.auto_range()

    def log_line(self, text):
        self.log.appendPlainText(text)

    def _tab_changed(self, _i):
        if self.tabs.currentWidget() is self.cal_tab:
            self.panel.show_section("calibrate")
        elif self.tabs.currentWidget() is self.test_tab:
            self.panel.show_section("approaches")

    # ---- drag & drop anywhere ---------------------------------------------------------------------------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls()]
        if self.tabs.currentWidget() is self.viewer:
            pcds = [p for p in paths if p.suffix.lower() == ".pcd"]
            pcds += [f for p in paths if p.is_dir() for f in sorted(p.rglob("*.pcd"))]
            for f in pcds:
                self.viewer.add_cloud(f)
            self.viewer.auto_range()
        else:
            self.tabs.setCurrentWidget(self.test_tab)
            self.test_tab.add_paths(paths)

    # ---- sessions ---------------------------------------------------------------------------------------------
    def session(self):
        """Everything needed to reproduce the current state; paths relative to the repository."""
        t = self.test_tab
        return {"params": self.panel.changed_values(),
                "calibration": to_session_path(self.active_cal.currentData()),
                "camera": self.cal_tab.camera.currentText(),
                "scenes": [s.to_json() for s in t.table.scenes],
                "approaches": t.selected_approaches(), "plots": t.plots.isChecked(), "export": t.export.isChecked()}

    def restore_session(self, path, quiet=False):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            if not quiet:
                QMessageBox.warning(self, "Session", f"Nicht lesbar: {exc}")
            return
        self.panel.reset()
        self.panel.set_values(data.get("params", {}))
        if data.get("calibration"):
            i = self.active_cal.findData(str(from_session_path(data["calibration"])))
            if i >= 0:
                self.active_cal.setCurrentIndex(i)
        if data.get("camera"):
            self.cal_tab.camera.setCurrentText(data["camera"])
        t = self.test_tab
        t.table.set_scenes([Scene.from_json(s) for s in data.get("scenes", [])])
        for n, cb in t.approach.items():
            cb.setChecked(n in data.get("approaches", [1, 2, 3]))
        t.plots.setChecked(data.get("plots", True))
        t.export.setChecked(data.get("export", True))

    def save_session(self):
        f, _ = QFileDialog.getSaveFileName(self, "Session speichern", str(ROOT / "Results" / "session.json"),
                                           "Session (*.json)")
        if f:
            Path(f).write_text(json.dumps(self.session(), indent=1), encoding="utf-8")

    def load_session(self):
        f, _ = QFileDialog.getOpenFileName(self, "Session laden", str(ROOT / "Results"), "Session (*.json)")
        if f:
            self.restore_session(f)

    def _autosave_path(self):
        d = Path(QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation))
        d.mkdir(parents=True, exist_ok=True)
        return d / "last_session.json"

    def open_results(self):
        folder = self.runner.run_dir or RUNS_ROOT
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def edit_settings(self):
        if SettingsDialog(self.settings, self).exec():
            self._watch()
            self.panel.rebuild()

    def closeEvent(self, e):
        if self.runner.busy() and QMessageBox.question(self, "Beenden?", "Ein Job läuft noch. Abbrechen und "
                                                       "beenden?") != QMessageBox.Yes:
            e.ignore()
            return
        self.runner.stop()
        self._autosave_path().write_text(json.dumps(self.session(), indent=1), encoding="utf-8")
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/state", self.saveState())
        self.viewer.close()
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("FusionGUI")
    app.setOrganizationName("iToF-SL-Combination")
    w = MainWindow()
    w.setAcceptDrops(True)
    w.show()
    sys.exit(app.exec())

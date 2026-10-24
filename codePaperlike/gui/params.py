"""Parameter discovery straight from the source files (AST, nothing imported) and the Qt forms built from it.

Sources:
- module constants      NAME = value  # comment       (approaches.py)
- class attributes      name = value  # comment       (DotCalibration)
- function keywords     def f(..., name=value)         (calibrate.calibrate, export_pcds, ...)
- argparse options      p.add_argument("--name", ...)  (binary2pcd_manual.py)
A new constant / keyword / option shows up in the GUI without touching it.
"""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QFont
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QScrollArea, QSpinBox, QToolBox, QToolButton, QVBoxLayout, QWidget)

KINDS = (bool, int, float, str)


@dataclass
class Param:
    name: str
    default: Any
    kind: str                      # bool | int | float | str | choice | literal | any
    doc: str = ""
    choices: list = field(default_factory=list)
    block: int = 0                 # blank-line block in the source, drawn as a separator


def _kind(value, choices=()):
    """Editor kind for a default value, None if it cannot be edited."""
    if choices:
        return "choice"
    if value is None:
        return "any"
    for t in KINDS:
        if type(value) is t:
            return t.__name__
    if isinstance(value, (tuple, list)):
        return "literal"
    return None


def _literal(node):
    """literal_eval of an AST node, with a sentinel when it is not a literal (lambdas, calls, ...)."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _literal


def _comment_choices(comment):
    """'"a" (text) | "b"' in a comment → ['a', 'b']."""
    return re.findall(r'"([^"]+)"', comment) if "|" in comment else []


def _assignments(body, lines, keep):
    """Params of simple NAME = literal  # comment assignments in a statement list."""
    params, block, last_line = [], 0, None
    for node in body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        if not keep(name):
            continue
        if last_line is not None and node.lineno - last_line > 1:
            block += 1
        last_line = node.end_lineno
        value = _literal(node.value)
        src = lines[node.lineno - 1]
        comment = src.split("#", 1)[1].strip() if "#" in src[node.value.end_col_offset:] else ""
        choices = _comment_choices(comment)
        kind = _kind(value, choices)
        if value is _literal or kind is None:
            continue
        params.append(Param(name, value, kind, comment, choices, block))
    return params


def _parse(path):
    """(AST, source lines) of a file."""
    text = Path(path).read_text(encoding="utf-8")
    return ast.parse(text), text.splitlines()


def module_constants(path, exclude=()):
    """UPPER_CASE module constants with their trailing comments."""
    tree, lines = _parse(path)
    return _assignments(tree.body, lines, lambda n: n.isupper() and n not in exclude)


def class_attributes(path, class_name, exclude=()):
    """Public class attributes of one class."""
    tree, lines = _parse(path)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return _assignments(cls.body, lines, lambda n: not n.startswith("_") and n not in exclude)


def argparse_options(path, exclude=()):
    """Options of every add_argument("--x", default=..., type=..., help=..., choices=[...]) call in a file."""
    tree, _ = _parse(path)
    params = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)
                and str(node.args[0].value).startswith("--")):
            continue
        kw = {k.arg: k.value for k in node.keywords}
        name = node.args[0].value.lstrip("-").replace("-", "_")
        if name in exclude:
            continue
        action = _literal(kw["action"]) if "action" in kw else None
        default = _literal(kw["default"]) if "default" in kw else (False if action == "store_true" else None)
        typ = kw["type"].id if isinstance(kw.get("type"), ast.Name) else None
        choices = _literal(kw["choices"]) if "choices" in kw else []
        if default is _literal or choices is _literal or "nargs" in kw:
            continue
        help_ = _literal(kw["help"]) if "help" in kw else ""
        kind = _kind(default, choices)
        if default is None and typ in ("int", "float") and not choices:
            kind = "any"
        if kind:
            params.append(Param(name, default, kind, help_ if isinstance(help_, str) else "", list(choices)))
    return params


def function_keywords(path, func_name, exclude=()):
    """Keyword parameters with literal defaults of a top-level function; choices and help from argparse in the same file."""
    tree, _ = _parse(path)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == func_name)
    args = fn.args.args + fn.args.kwonlyargs
    defaults = [None] * (len(fn.args.args) - len(fn.args.defaults)) + fn.args.defaults + fn.args.kw_defaults
    cli = {p.name: p for p in argparse_options(path)}
    checked = {}                   # if name not in ("a", "b"): raise ...  → choices
    for node in ast.walk(fn):
        if (isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) and len(node.ops) == 1
                and isinstance(node.ops[0], (ast.In, ast.NotIn))):
            options = _literal(node.comparators[0])
            if isinstance(options, (tuple, list, set)) and all(isinstance(o, str) for o in options):
                checked[node.left.id] = list(options)
    params = []
    for a, d in zip(args, defaults):
        if d is None or a.arg in exclude:
            continue
        value = _literal(d)
        opt = cli.get(a.arg)
        choices = opt.choices if opt else checked.get(a.arg, [])
        kind = _kind(value, choices)
        if value is _literal or kind is None:
            continue
        params.append(Param(a.arg, value, kind, opt.doc if opt else "", choices))
    return params


def parse_text(text, default):
    """Value typed into a free-form field: a Python literal, 'None', or the plain string."""
    text = text.strip()
    if text in ("", "None"):
        return None
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text if not isinstance(default, (tuple, list)) else default


class ParamRow(QWidget):
    """Editor for one parameter plus a reset button; emits changed()."""

    changed = Signal()

    def __init__(self, param: Param):
        super().__init__()
        self.param = param
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        p = param
        if p.kind == "bool":
            self.editor = QCheckBox()
            self.editor.toggled.connect(self._changed)
        elif p.kind == "int":
            self.editor = QSpinBox()
            self.editor.setRange(-10 ** 9, 10 ** 9)
            self.editor.valueChanged.connect(self._changed)
        elif p.kind == "choice":
            self.editor = QComboBox()
            self.editor.addItems([str(c) for c in p.choices])
            self.editor.currentIndexChanged.connect(self._changed)
        else:
            self.editor = QLineEdit()
            if p.kind == "float":
                v = QDoubleValidator()
                v.setNotation(QDoubleValidator.ScientificNotation)
                self.editor.setValidator(v)
            elif p.kind == "any":
                self.editor.setPlaceholderText("None")
            self.editor.textChanged.connect(self._changed)
        self.editor.setToolTip(p.doc)
        self.reset = QToolButton()
        self.reset.setText("↺")
        self.reset.setToolTip(f"Default aus dem Code: {p.default!r}")
        self.reset.setAutoRaise(True)
        self.reset.clicked.connect(lambda: self.set_value(self.param.default))
        lay.addWidget(self.editor, 1)
        lay.addWidget(self.reset)
        self.set_value(p.default)

    def value(self):
        """Current value in the parameter's type."""
        p, e = self.param, self.editor
        if p.kind == "bool":
            return e.isChecked()
        if p.kind == "int":
            return e.value()
        if p.kind == "choice":
            return p.choices[e.currentIndex()]
        if p.kind == "float":
            try:
                return float(e.text().replace(",", "."))
            except ValueError:
                return p.default
        if p.kind == "str":
            return e.text()
        return parse_text(e.text(), p.default)

    def set_value(self, value):
        """Show a value (falls back to the default if it does not fit)."""
        p, e = self.param, self.editor
        if p.kind == "bool":
            e.setChecked(bool(value))
        elif p.kind == "int":
            e.setValue(int(value) if isinstance(value, (int, float)) else p.default)
        elif p.kind == "choice":
            idx = [str(c) for c in p.choices].index(str(value)) if str(value) in map(str, p.choices) else 0
            e.setCurrentIndex(idx)
        elif p.kind == "float":
            e.setText(repr(float(value)) if isinstance(value, (int, float)) else repr(p.default))
        elif p.kind == "str":
            e.setText(str(value))
        else:
            e.setText("" if value is None else repr(tuple(value) if isinstance(value, list) else value))
        self._changed()

    def is_default(self):
        """True while the value equals the code default."""
        v, d = self.value(), self.param.default
        return (list(v) == list(d)) if isinstance(d, (tuple, list)) and isinstance(v, (tuple, list)) else v == d

    def _changed(self, *_):
        self.reset.setVisible(not self.is_default())
        self.changed.emit()


class ParamForm(QWidget):
    """Form of one parameter section; changed values are shown bold."""

    changed = Signal()

    def __init__(self, params):
        super().__init__()
        self.rows, self.labels = {}, {}
        form = QFormLayout(self)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setLabelAlignment(Qt.AlignRight)
        mono = QFont("monospace")
        mono.setStyleHint(QFont.Monospace)
        last_block = None
        for p in params:
            if last_block is not None and p.block != last_block:
                line = QFrame()
                line.setFrameShape(QFrame.HLine)
                line.setFrameShadow(QFrame.Sunken)
                form.addRow(line)
            last_block = p.block
            row = ParamRow(p)
            label = QLabel(p.name)
            label.setFont(mono)
            label.setToolTip(p.doc)
            row.changed.connect(lambda r=row, l=label: self._mark(r, l))
            form.addRow(label, row)
            self.rows[p.name], self.labels[p.name] = row, label
            self._mark(row, label)
        if not params:
            form.addRow(QLabel("(keine Parameter gefunden)"))

    def _mark(self, row, label):
        f = label.font()
        f.setBold(not row.is_default())
        label.setFont(f)
        self.changed.emit()

    def changed_values(self):
        """{name: value} of everything that differs from the code."""
        return {n: r.value() for n, r in self.rows.items() if not r.is_default()}

    def values(self):
        """{name: value} of all parameters."""
        return {n: r.value() for n, r in self.rows.items()}

    def set_values(self, values):
        """Apply stored values; unknown names are ignored (the parameter was removed from the code)."""
        for n, v in values.items():
            if n in self.rows:
                self.rows[n].set_value(v)

    def reset(self):
        """All parameters back to the code defaults."""
        for r in self.rows.values():
            r.set_value(r.param.default)

    def filter(self, text):
        """Hide rows whose name and doc do not contain text; returns the number still visible."""
        text = text.lower()
        visible = 0
        form = self.layout()
        for n, r in self.rows.items():
            show = not text or text in n.lower() or text in r.param.doc.lower()
            form.setRowVisible(r, show)
            visible += show
        return visible


@dataclass
class Section:
    """A parameter section of the side panel: key, title and a function returning its Params."""
    key: str
    title: str
    discover: Any
    watch: tuple = ()              # source files; the panel rebuilds when one changes


class ParamPanel(QWidget):
    """Search field + collapsible sections; rebuilds from the source files, keeping the user's changes."""

    changed = Signal()

    def __init__(self, sections):
        super().__init__()
        self.sections = sections
        self.forms = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Parameter suchen …")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        lay.addWidget(self.search)
        self.toolbox = QToolBox()
        lay.addWidget(self.toolbox, 1)
        self.rebuild()

    def rebuild(self, keep=None):
        """(Re)discover all sections; keep = {section: {name: value}} to restore (default: current changes)."""
        keep = self.changed_values() if keep is None else keep
        current = self.toolbox.currentIndex()
        while self.toolbox.count():
            w = self.toolbox.widget(0)
            self.toolbox.removeItem(0)
            w.deleteLater()
        self.forms = {}
        for s in self.sections:
            try:
                params = s.discover()
                error = None
            except Exception as exc:  # a half-edited source file must not kill the GUI
                params, error = [], exc
            form = ParamForm(params)
            if error is not None:
                form.layout().addRow(QLabel(f"Fehler beim Lesen: {error}"))
            form.set_values(keep.get(s.key, {}))
            form.changed.connect(self.changed)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(form)
            scroll.setFrameShape(QFrame.NoFrame)
            self.toolbox.addItem(scroll, s.title)
            self.forms[s.key] = form
        if 0 <= current < self.toolbox.count():
            self.toolbox.setCurrentIndex(current)
        self._filter(self.search.text())

    def _filter(self, text):
        for i, s in enumerate(self.sections):
            n = self.forms[s.key].filter(text)
            self.toolbox.setItemText(i, f"{s.title}  ({n})" if text else s.title)

    def show_section(self, key):
        """Open one section."""
        keys = [s.key for s in self.sections]
        if key in keys:
            self.toolbox.setCurrentIndex(keys.index(key))

    def values(self, key):
        """All values of one section."""
        return self.forms[key].values() if key in self.forms else {}

    def changed_values(self, key: Optional[str] = None):
        """Changed values of one section, or {section: {...}} of all."""
        if key is not None:
            return self.forms[key].changed_values() if key in self.forms else {}
        return {k: f.changed_values() for k, f in self.forms.items() if f.changed_values()}

    def set_values(self, values):
        """Restore {section: {name: value}}."""
        for k, v in values.items():
            if k in self.forms:
                self.forms[k].set_values(v)

    def reset(self):
        """Everything back to the code defaults."""
        for f in self.forms.values():
            f.reset()

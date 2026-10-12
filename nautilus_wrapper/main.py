"""Nautilus Wrapper GUI - A PyQt6 interface for the Nautilus astrochemistry code."""

import math
import operator as _op
import os
import re
import shutil
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QListWidget, QListWidgetItem, QPushButton, QLabel, QComboBox,
    QLineEdit, QFileDialog, QGroupBox, QMessageBox, QPlainTextEdit, QTextEdit,
    QTableView, QHeaderView, QProgressBar, QStackedWidget, QInputDialog,
    QTabWidget, QDialog, QDialogButtonBox, QFormLayout,
    QListView, QTreeView, QAbstractItemView, QSpinBox, QGridLayout,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction, QIcon

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, LogLocator, MaxNLocator

try:
    from .parsers import (
        PARSERS, SERIALIZERS, ELEMENT_COLS, parse_species,
        parse_observations, evaluate_observations,
        parse_control_grid, obs_name_to_species, CONTROL_GRID_SIZE,
        parse_plot_settings, serialize_plot_settings, DISAGREEMENT_TOKEN,
    )
except ImportError:
    from parsers import (
        PARSERS, SERIALIZERS, ELEMENT_COLS, parse_species,
        parse_observations, evaluate_observations,
        parse_control_grid, obs_name_to_species, CONTROL_GRID_SIZE,
        parse_plot_settings, serialize_plot_settings, DISAGREEMENT_TOKEN,
    )


REQUIRED_INPUT_FILES = [
    "parameters.in",
    "element.in",
    "abundances.in",
    "gas_species.in",
    "grain_species.in",
    "gas_reactions.in",
    "grain_reactions.in",
    "activation_energies_refs.in",
    "surface_parameters.in",
    "1D_static.dat",
    "structure_evolution.dat",
]

# Not part of the required set: absent by default, not included in the
# "empty input files" export check, and auto-created (empty) with a warning
# the first time a run is launched without one.
OPTIONAL_INPUT_FILES = ["observations.txt"]
ALL_INPUT_FILES = REQUIRED_INPUT_FILES + OPTIONAL_INPUT_FILES

STRUCTURED_FILES = set(PARSERS.keys())
HEAVY_FILES = {"gas_reactions.in", "grain_reactions.in"}
REACTION_FILES = {"gas_reactions.in", "grain_reactions.in"}
# Files whose structured view shows a per-row activate/deactivate checkbox.
TOGGLE_ACTIVE_FILES = REACTION_FILES | {"observations.txt"}
# Among those, files where an unchecked row is written back out as a '#'
# comment (kept, but disabled) rather than dropped — and whose checkbox
# state on load comes from the file's own '#' markers, not a blanket default.
PRESERVE_INACTIVE_FILES = {"observations.txt"}
CHUNK_SIZE = 4000


_PKG_DIR = Path(__file__).resolve().parent
# Per-OS application icon. macOS wants the full-bleed rounded-square PNG,
# Windows/Linux the padded square; the SVG is the last-resort fallback.
_ICON_FILES = {
    "darwin": "icon_mac.png",
    "win32": "icon_win.png",
}


def app_icon() -> QIcon:
    """The window/taskbar icon for the current platform (falls back to the
    generic Linux PNG, then the SVG)."""
    for name in (_ICON_FILES.get(sys.platform, "icon_linux.png"),
                 "icon_linux.png", "icon.svg"):
        path = _PKG_DIR / name
        if path.exists():
            return QIcon(str(path))
    return QIcon()


ASSETS_DIR = Path(__file__).resolve().parent.parent / "inputs"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
# "Plot controls" reads every *.txt in this folder as one 4x4 grid (one per
# element, e.g. C.txt / N.txt / O.txt / S.txt / ions.txt), each drawn as its
# own sub-tab. Kept out of the input sets so it's shared across runs.
CONTROL_PLOTS_DIR = Path(__file__).resolve().parent.parent / "control_plots"

# Persisted abundance-plot configuration (species, layout, style, output
# distribution). Written every time "Plot" runs and read back at startup so
# the selection survives restarts. Hand-editable; see parsers.py for the format.
#
# The file is *linked to a run*: it lives inside the folder of the "main"
# output — the one at the top of the "Outputs to plot" list. The repo-root
# path below is only a fallback/template used before any run is available.
PLOT_SETTINGS_NAME = "plot_settings.txt"
PLOT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / PLOT_SETTINGS_NAME

# Default x-axis range (time, years) for every plot; editable in Edit ▸
# Settings ▸ Plotting.
DEFAULT_PLOT_XLIM = (1e3, 1e7)

# Key atoms the distance of disagreement is broken down over.
KEY_ATOMS = ("C", "N", "O", "S")
# Nautilus is run through the PyPI CLI (``uv tool install
# nautilus-astrochem@latest``): ``run`` writes output.nc in the run folder,
# then ``write-ascii abundances`` turns it into the ab/<species>.ab files
# the plots read.
NAUTILUS_CLI = "nautilus-astrochem"
RUN_LABEL = "nautilus run"
WRITE_AB_LABEL = "write-ascii abundances"
# Every Nautilus ab/ output value is multiplied by this on read: the raw
# columns are per H2, so ×2 gives the abundance relative to H (total H
# nuclei) — the quantity every plot and the disagreement metric use.
NAUTILUS_OUTPUT_SCALE = 2.0
# log10 floor so a predicted 0 doesn't blow up log10().
_LOG_FLOOR = 1e-300
# Abundances below this are treated as "absent" when plotting log10(abundance)
# so a curve dipping to ~1e-30 can't dominate the shared y-range.
_DISPLAY_FLOOR = 1e-20

# In the abundance plot each species gets a colour and each output a
# linestyle. Session runs (nautilus run this session) are always solid;
# imported outputs cycle through these.
_SPECIES_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
    "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7",
    "#dbdb8d", "#9edae5",
]
_IMPORT_LINESTYLES = ["--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
# Pseudo-entry, always first in the Layout-tab list: assign it to a subplot
# to draw the distance-of-disagreement curve there instead of species.
_DISAGREEMENT_KEY = "◆ Distance of disagreement"


# Fixed 1-decimal format so every linear-axis tick label has the same width
# (the distance-of-disagreement y-axis; every other axis is log, whose
# 10^n labels are already uniform).
_DEX_FMT = FormatStrFormatter("%.1f")


def _tidy_ticks(ax, xticks: int = 8, yticks: int = 8, integer_y: bool = False):
    """No minor ticks anywhere; whole-decade major ticks on log axes, capped
    so labels on small panels don't overlap."""
    ax.minorticks_off()
    if ax.get_xscale() == "log":
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=xticks))
    if ax.get_yscale() == "log":
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=yticks))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(
            nbins=yticks, integer=integer_y,
            prune=(None if integer_y else "both")))


def _fit_y_to_x(ax, x0: float, x1: float):
    """Scale the (linear) y-axis to the data visible within the x-range
    ``[x0, x1]`` (curve points in range + any horizontal-bar levels)."""
    ys: list[float] = []
    for ln in ax.lines:
        xd, yd = ln.get_xdata(), ln.get_ydata()
        if len(xd) == 2 and xd[0] == xd[1]:                    # vertical line
            continue
        if len(xd) == 2 and len(yd) == 2 and yd[0] == yd[1]:   # axhline
            ys.append(yd[0])
            continue
        for x, y in zip(xd, yd):
            if x0 <= x <= x1 and y == y and y not in (math.inf, -math.inf):
                ys.append(y)
    if not ys:
        return
    lo, hi = min(ys), max(ys)
    span = hi - lo
    if span < 2:                          # keep a few integer ticks
        mid = (lo + hi) / 2
        lo, hi = mid - 1.5, mid + 1.5
    else:
        pad = 0.06 * span
        lo, hi = lo - pad, hi + pad
    ax.set_ylim(lo, hi)


def fmt_num(x: float) -> str:
    """Compact number for a settings text field: ``1000.0`` -> ``1000``,
    ``1e7`` -> ``1e7``."""
    s = f"{x:g}"
    return s.replace("e+0", "e").replace("e-0", "e-").replace("e+", "e")

def get_input_sets() -> dict[str, Path]:
    """Scan ASSETS_DIR for subdirectories containing input files."""
    sets = {}
    if not ASSETS_DIR.exists():
        return sets
    for d in sorted(ASSETS_DIR.iterdir()):
        if d.is_dir() and (d / "parameters.in").exists():
            sets[d.name] = d
    return sets


def find_ambiguities(set_dir: Path) -> dict[str, list[Path]]:
    """Find groups of files in set_dir that are candidates for one expected input.

    For an expected file like ``abundances.in`` a candidate is ``abundances.in``
    itself or a variant sharing the stem and extension and separated by ``_`` or
    ``-`` (e.g. ``abundances_solar.in``). Returns only groups with >1 candidate.
    """
    all_files = [f for f in set_dir.iterdir() if f.is_file()]
    groups: dict[str, list[Path]] = {}

    for expected in ALL_INPUT_FILES:
        stem, _, ext = expected.rpartition(".")
        matches = [
            f for f in all_files
            if f.name == expected
            or (
                f.suffix == f".{ext}"
                and (f.stem == stem or f.stem.startswith(f"{stem}_") or f.stem.startswith(f"{stem}-"))
            )
        ]
        if len(matches) > 1:
            groups[expected] = sorted(matches)

    return groups


# ---------------------------------------------------------------------------
# Syntax highlighter
# ---------------------------------------------------------------------------
class NautilusHighlighter(QSyntaxHighlighter):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._comment_fmt = QTextCharFormat()
        self._comment_fmt.setForeground(QColor("#6A9955"))
        self._key_fmt = QTextCharFormat()
        self._key_fmt.setForeground(QColor("#569CD6"))
        self._key_fmt.setFontWeight(QFont.Weight.Bold)
        self._number_fmt = QTextCharFormat()
        self._number_fmt.setForeground(QColor("#B5CEA8"))

    def highlightBlock(self, text):
        import re
        comment_idx = text.find("!")
        if comment_idx >= 0:
            self.setFormat(comment_idx, len(text) - comment_idx, self._comment_fmt)
        for m in re.finditer(r"(\w+)\s*=", text):
            self.setFormat(m.start(1), m.end(1) - m.start(1), self._key_fmt)
        for m in re.finditer(r"[+-]?\d+[\d.]*[eE]?[+-]?\d*", text):
            if comment_idx < 0 or m.start() < comment_idx:
                self.setFormat(m.start(), m.end() - m.start(), self._number_fmt)


def mark_filter_validity(line_edit: QLineEdit, error: str | None):
    """Give ``line_edit`` a red outline + explanatory tooltip when ``error`` is
    set (the filter text is invalid); restore the normal look otherwise."""
    if error:
        line_edit.setStyleSheet(
            "QLineEdit { border: 1px solid #d9534f;"
            " background-color: rgba(217, 83, 79, 0.12); }"
        )
        line_edit.setToolTip(f"Invalid filter — {error}")
    else:
        line_edit.setStyleSheet("")
        line_edit.setToolTip("")


# ---------------------------------------------------------------------------
# Model for QTableView
# ---------------------------------------------------------------------------
class TableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self._headers: list[str] = []
        self._data: list[list[str]] = []
        self._active_col: int | None = None
        self._active: list[bool] = []
        self._sort_col: int | None = None
        self._sort_asc: bool = True
        self._filter_expr: str = ""
        # Grey out unchecked rows (meaningful for reaction lists where unchecked
        # == excluded; noise for a pick-list where unchecked is the default).
        self._grey_inactive: bool = True
        self._visible: list[bool] = []  # parallel to _data, True = row matches filter
        # Cached visual-row -> data-row map. None means "identity" (no filter
        # active / every row visible), which lets data() and rowCount() stay
        # O(1) instead of rebuilding a list on every cell access.
        self._row_map: list[int] | None = None
        # Data-column indices whose cells are all numbers: shown right-aligned in
        # a monospace font so digits line up.
        self._numeric_cols: set[int] = set()
        self._mono_font = QFont("Monaco")
        self._mono_font.setStyleHint(QFont.StyleHint.Monospace)

    @staticmethod
    def _looks_numeric(text: str) -> bool:
        text = text.strip()
        if not text:
            return True  # blank cells don't disqualify a column
        try:
            float(text.replace("D", "e").replace("d", "e"))
            return True
        except ValueError:
            return False

    def _detect_numeric_cols(self, sample: int = 400):
        self._numeric_cols = set()
        if not self._data:
            return
        ncols = max(len(r) for r in self._data)
        for c in range(ncols):
            seen = False
            all_num = True
            for row in self._data[:sample]:
                if c >= len(row) or not row[c].strip():
                    continue
                seen = True
                if not self._looks_numeric(row[c]):
                    all_num = False
                    break
            if seen and all_num:
                self._numeric_cols.add(c)

    def _refresh_row_map(self):
        if not self._visible or all(self._visible):
            self._row_map = None
        else:
            self._row_map = [i for i, v in enumerate(self._visible) if v]

    def data_headers(self) -> list[str]:
        if self._active_col is not None:
            return self._headers[1:]
        return self._headers

    def set_headers(self, headers: list[str], active_col: bool = False):
        self.beginResetModel()
        if active_col:
            self._headers = [""] + headers
            self._active_col = 0
        else:
            self._headers = headers
            self._active_col = None
        self._data = []
        self._active = []
        self._visible = []
        self._row_map = None
        self._numeric_cols = set()
        self.endResetModel()

    def append_rows(self, rows: list[dict], headers: list[str]):
        if not rows:
            return
        start = len(self._data)
        self.beginInsertRows(QModelIndex(), start, start + len(rows) - 1)
        for row in rows:
            self._data.append([str(row.get(h, "")) for h in headers])
            self._active.append(True)
            self._visible.append(True)
        self.endInsertRows()
        self._detect_numeric_cols()
        if self._filter_expr:
            self._apply_filter()
        else:
            self._row_map = None

    def _next_unique_id(self, id_col: int) -> int:
        """Smallest positive integer not already used in data column
        ``id_col`` — but prefer ``max + 1`` while it still fits the 6-digit
        reaction ID field."""
        used: set[int] = set()
        for r in self._data:
            if id_col < len(r):
                try:
                    used.add(int(float(r[id_col].strip())))
                except (ValueError, TypeError):
                    pass
        if used:
            nxt = max(used) + 1
            if nxt <= 999999:
                return nxt
        n = 1
        while n in used:
            n += 1
        return n

    def add_blank_row(self) -> int:
        """Append an empty, active row and return its data-row index. If the
        model has an ``ID`` column the new row is given an ID that is unique
        among the existing rows, so it survives export cleanly."""
        hdrs = self.data_headers()
        values = ["" for _ in hdrs]
        id_col = next((i for i, h in enumerate(hdrs)
                       if h.strip().lower() == "id"), None)
        if id_col is not None:
            values[id_col] = str(self._next_unique_id(id_col))
        start = len(self._data)
        view_row = self.rowCount()
        self.beginInsertRows(QModelIndex(), view_row, view_row)
        self._data.append(values)
        self._active.append(True)
        self._visible.append(True)
        if self._row_map is not None:
            self._row_map.append(start)   # keep it visible under an active filter
        self.endInsertRows()
        self._detect_numeric_cols()
        return start

    def delete_rows(self, data_rows) -> int:
        """Remove the given data-row indices. Returns how many were deleted."""
        drop = {i for i in data_rows if 0 <= i < len(self._data)}
        if not drop:
            return 0
        self.beginResetModel()
        keep = [i for i in range(len(self._data)) if i not in drop]
        self._data = [self._data[i] for i in keep]
        self._active = [self._active[i] for i in keep]
        self._visible = [self._visible[i] for i in keep]
        self._refresh_row_map()
        self._detect_numeric_cols()
        self.endResetModel()
        return len(drop)

    def set_data(self, rows: list[dict], headers: list[str], active_col: bool = False,
                 active_default: bool = True):
        self.beginResetModel()
        if active_col:
            self._headers = [""] + headers
            self._active_col = 0
        else:
            self._headers = headers
            self._active_col = None
        self._data = [[str(row.get(h, "")) for h in headers] for row in rows]
        self._active = [active_default] * len(rows)
        self._visible = [True] * len(rows)
        self._row_map = None
        self._detect_numeric_cols()
        self.endResetModel()

    def set_active_values(self, header: str, values):
        """Check exactly the rows whose ``header`` cell is in ``values``."""
        data_headers = self.data_headers()
        if header not in data_headers or not self._data:
            return
        col = data_headers.index(header)
        wanted = set(values)
        for i, row in enumerate(self._data):
            self._active[i] = row[col] in wanted
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
            [Qt.ItemDataRole.CheckStateRole,
             Qt.ItemDataRole.BackgroundRole,
             Qt.ItemDataRole.ForegroundRole],
        )

    def clear(self):
        self.beginResetModel()
        self._data = []
        self._headers = []
        self._active = []
        self._visible = []
        self._row_map = None
        self.endResetModel()

    def _visible_rows(self):
        """Iterable of data-row indices that are currently visible.

        Returns a ``range`` (no allocation) when nothing is filtered out.
        """
        if self._row_map is None:
            return range(len(self._data))
        return self._row_map

    def rowCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._data) if self._row_map is None else len(self._row_map)

    def columnCount(self, parent=QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._headers)

    def _map_row(self, row: int) -> int:
        if self._row_map is None:
            return row
        return self._row_map[row] if 0 <= row < len(self._row_map) else row

    _DATA_ROLES = frozenset({
        Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.EditRole,
        Qt.ItemDataRole.CheckStateRole, Qt.ItemDataRole.BackgroundRole,
        Qt.ItemDataRole.ForegroundRole,
        Qt.ItemDataRole.FontRole, Qt.ItemDataRole.TextAlignmentRole,
    })

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role not in self._DATA_ROLES:
            return None
        row = self._map_row(index.row())
        col = index.column()

        if self._active_col is not None and col == self._active_col:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if self._active[row] else Qt.CheckState.Unchecked
            return None

        data_col = col - (1 if self._active_col is not None else 0)

        if data_col in self._numeric_cols:
            if role == Qt.ItemDataRole.FontRole:
                return self._mono_font
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        elif role in (Qt.ItemDataRole.FontRole, Qt.ItemDataRole.TextAlignmentRole):
            return None

        if self._grey_inactive and not self._active[row]:
            if role == Qt.ItemDataRole.BackgroundRole:
                return QColor(180, 180, 180)
            if role == Qt.ItemDataRole.ForegroundRole:
                return QColor(80, 80, 80)

        if role == Qt.ItemDataRole.DisplayRole or role == Qt.ItemDataRole.EditRole:
            return self._data[row][data_col] if data_col < len(self._data[row]) else ""
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section] if section < len(self._headers) else None
        return None

    def flags(self, index: QModelIndex):
        base = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        if self._active_col is not None and index.column() == self._active_col:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        return base | Qt.ItemFlag.ItemIsEditable

    def setData(self, index: QModelIndex, value, role=Qt.ItemDataRole.EditRole):
        row = self._map_row(index.row())
        col = index.column()

        if self._active_col is not None and col == self._active_col and role == Qt.ItemDataRole.CheckStateRole:
            self._active[row] = not self._active[row]
            visual_row = index.row()
            left = self.index(visual_row, 0)
            right = self.index(visual_row, self.columnCount() - 1)
            self.dataChanged.emit(left, right, [
                Qt.ItemDataRole.CheckStateRole,
                Qt.ItemDataRole.BackgroundRole,
                Qt.ItemDataRole.ForegroundRole,
            ])
            return True

        if role == Qt.ItemDataRole.EditRole:
            data_col = col - (1 if self._active_col is not None else 0)
            self._data[row][data_col] = str(value)
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def to_dicts(self, only_active: bool = True, include_active: bool = False) -> list[dict]:
        data_headers = self.data_headers()
        result = []
        for i, row in enumerate(self._data):
            if only_active and self._active_col is not None and not self._active[i]:
                continue
            d = dict(zip(data_headers, row))
            if include_active:
                d["_active"] = self._active[i] if self._active_col is not None else True
            result.append(d)
        return result

    def set_active_flags(self, flags: list[bool]):
        """Set each row's checkbox state directly, in row order (used to seed
        activation from the file's own markup instead of a blanket default)."""
        if not self._data:
            return
        n = min(len(flags), len(self._active))
        self._active[:n] = flags[:n]
        self.dataChanged.emit(
            self.index(0, 0),
            self.index(self.rowCount() - 1, self.columnCount() - 1),
            [Qt.ItemDataRole.CheckStateRole,
             Qt.ItemDataRole.BackgroundRole,
             Qt.ItemDataRole.ForegroundRole],
        )

    def toggle_all(self, state: bool):
        if self._active_col is None or not self._data:
            return
        visible = self._visible_rows()
        for i in visible:
            self._active[i] = state
        top = self.index(0, 0)
        bot = self.index(self.rowCount() - 1, self.columnCount() - 1)
        self.dataChanged.emit(top, bot, [
            Qt.ItemDataRole.CheckStateRole,
            Qt.ItemDataRole.BackgroundRole,
            Qt.ItemDataRole.ForegroundRole,
        ])

    @staticmethod
    def _sort_key(value: str):
        """Sort numerically when the whole cell parses as a number, else lexically.

        The tuple's first element keeps numbers and strings in separate ordered
        groups so the two are never compared directly.
        """
        try:
            return (0, float(value), "")
        except (TypeError, ValueError):
            return (1, 0.0, value)

    def sort_by_column(self, col: int, ascending: bool = True):
        data_col = col - (1 if self._active_col is not None else 0)
        if data_col < 0:
            return
        self.beginResetModel()
        paired = list(zip(self._data, self._active, self._visible))
        paired.sort(key=lambda x: self._sort_key(x[0][data_col]), reverse=not ascending)
        self._data = [p[0] for p in paired]
        self._active = [p[1] for p in paired]
        self._visible = [p[2] for p in paired]
        self._refresh_row_map()
        self.endResetModel()

    def set_filter(self, expr: str):
        self._filter_expr = expr.strip()
        self._apply_filter()

    def _apply_filter(self):
        self.layoutAboutToBeChanged.emit()
        if not self._filter_expr or not self._data:
            self._visible = [True] * len(self._data)
        else:
            hdrs = self.data_headers()
            col_map = {h: i for i, h in enumerate(hdrs)}
            checked_idx = None
            if self._active_col is not None:
                checked_idx = len(hdrs)
                for alias in self._CHECKED_ALIASES:
                    col_map.setdefault(alias, checked_idx)
            self._visible = []
            for i, row in enumerate(self._data):
                if checked_idx is not None:
                    row = row + ["1" if self._active[i] else "0"]
                try:
                    self._visible.append(self._eval_filter(self._filter_expr, col_map, row))
                except Exception:
                    self._visible.append(True)
        self._refresh_row_map()
        self.layoutChanged.emit()

    # Boolean operators, lowest precedence first. Each accepts word and symbol
    # aliases:  OR = or | ||    XOR = xor ^ ⊕    AND = and & && ∧
    _OR_RE = re.compile(r"\s*(?:\|\|?|\bor\b)\s*", re.IGNORECASE)
    _XOR_RE = re.compile(r"\s*(?:\^|⊕|\bxor\b)\s*", re.IGNORECASE)
    _AND_RE = re.compile(r"\s*(?:&&?|∧|\band\b)\s*", re.IGNORECASE)
    # Equality is a single "=" ("==" still accepted). "!=", "<=", ">=" must be
    # matched before the bare "<", ">", "=" alternatives.
    _CMP_RE = re.compile(r"\s*(!=|<=|>=|==?|<|>)\s*")
    _OPS = {
        "=": _op.eq, "==": _op.eq, "!=": _op.ne,
        "<": _op.lt, "<=": _op.le, ">": _op.gt, ">=": _op.ge,
    }
    # Virtual column names that resolve to a row's checkbox state ("1"/"0")
    # when the model has an active column.
    _CHECKED_ALIASES = ("Checked", "checked", "Active", "active")

    def filter_error(self, expr: str) -> str | None:
        """Return a message describing why ``expr`` is not a usable filter, or
        ``None`` if it is fine (an empty expression is fine)."""
        expr = expr.strip()
        if not expr:
            return None
        columns = self.data_headers()
        if not columns:
            return None
        valid_cols = set(columns)
        if self._active_col is not None:
            valid_cols.update(self._CHECKED_ALIASES)
        leaves: list[str] = []
        for or_part in self._OR_RE.split(expr):
            for xor_part in self._XOR_RE.split(or_part):
                leaves.extend(self._AND_RE.split(xor_part))
        for leaf in leaves:
            leaf = leaf.strip()
            if not leaf:
                return "incomplete expression (dangling and/or/xor)"
            parts = self._CMP_RE.split(leaf, 1)
            if len(parts) != 3:
                # No comparison operator: a bare word is a free-text search
                # across every column, which is always valid.
                continue
            col, _cmp, val = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not col:
                return f"missing column before '{_cmp}'"
            if col not in valid_cols:
                return f"unknown column '{col}'"
            if not val:
                return f"missing value after '{_cmp}'"
            if self._CMP_RE.search(val):
                return (f"'{leaf.strip()}' has more than one comparison "
                        "(missing/misspelled and/or/xor?)")
        return None

    def _eval_filter(self, expr: str, col_map: dict, row: list[str]) -> bool:
        return self._eval_or(expr.strip(), col_map, row)

    def _eval_or(self, expr: str, col_map: dict, row: list[str]) -> bool:
        parts = self._OR_RE.split(expr)
        if len(parts) > 1:
            return any(self._eval_xor(p, col_map, row) for p in parts)
        return self._eval_xor(expr, col_map, row)

    def _eval_xor(self, expr: str, col_map: dict, row: list[str]) -> bool:
        parts = self._XOR_RE.split(expr)
        if len(parts) > 1:
            # xor == "exactly one of the operands is true"
            return sum(
                1 for p in parts if self._eval_and(p, col_map, row)
            ) == 1
        return self._eval_and(expr, col_map, row)

    def _eval_and(self, expr: str, col_map: dict, row: list[str]) -> bool:
        parts = self._AND_RE.split(expr)
        if len(parts) > 1:
            return all(self._eval_condition(p, col_map, row) for p in parts)
        return self._eval_condition(expr, col_map, row)

    def _eval_condition(self, expr: str, col_map: dict, row: list[str]) -> bool:
        ops = self._OPS
        parts = self._CMP_RE.split(expr, 1)
        if len(parts) != 3:
            # No operator: substring search over every cell in the row
            # (case-insensitive). "X" -> rows where any field contains X.
            term = expr.strip().strip('"').strip("'").lower()
            if not term:
                return True
            return any(term in cell.lower() for cell in row)
        col_name, cmp, val = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if col_name not in col_map:
            return True
        cell = row[col_map[col_name]]
        val = val.strip('"').strip("'")
        try:
            val_num = float(val)
        except ValueError:
            val_num = None
        if val_num is not None:
            # Numeric predicate: a non-numeric cell simply doesn't match
            # (except for !=, where it does).
            try:
                return ops[cmp](float(cell), val_num)
            except ValueError:
                return cmp == "!="
        return ops[cmp](cell.strip('"').strip("'"), val)


# ---------------------------------------------------------------------------
# Custom header with context menu for sorting
# ---------------------------------------------------------------------------
class SortableHeaderView(QHeaderView):
    sort_requested = pyqtSignal(int, bool)

    def __init__(self, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.setSectionsClickable(True)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    def _show_menu(self, pos):
        col = self.logicalIndexAt(pos)
        if col < 0:
            return
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        asc = menu.addAction("Sort Ascending")
        desc = menu.addAction("Sort Descending")
        action = menu.exec(self.mapToGlobal(pos))
        if action == asc:
            self.sort_requested.emit(col, True)
        elif action == desc:
            self.sort_requested.emit(col, False)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
class RunWorker(QThread):
    log_line = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)
    finished_signal = pyqtSignal(int)

    def __init__(self, cmd: list[str], cwd: str):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import subprocess, re
        try:
            proc = subprocess.Popen(
                self._cmd, cwd=self._cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                if self._cancelled:
                    proc.kill()
                    break
                # tqdm bar from `write-ascii` (\r-separated, so one "line"
                # per update): drive the progress bar, don't flood the log.
                t = re.search(r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)", line)
                if t:
                    self.progress_update.emit(
                        int(t.group(1)), f"{t.group(2)}/{t.group(3)} species")
                    continue
                self.log_line.emit(line.rstrip("\n"))
                m = re.search(r"\[\s*([\d.]+)\s*%\].*remaining:\s*([\d.]+)\s*(\w+)", line)
                if m:
                    pct = int(float(m.group(1)))
                    remaining = m.group(2)
                    unit = m.group(3)
                    self.progress_update.emit(pct, f"~{remaining} {unit} remaining")
            proc.wait()
            self.finished_signal.emit(proc.returncode)
        except FileNotFoundError:
            self.log_line.emit(f"ERROR: Could not find '{self._cmd[0]}'")
            self.finished_signal.emit(-1)
        except Exception as e:
            self.log_line.emit(f"ERROR: {e}")
            self.finished_signal.emit(-1)


# ---------------------------------------------------------------------------
# Background worker for loading files
# ---------------------------------------------------------------------------
class LoadWorker(QThread):
    progress = pyqtSignal(int, int)
    chunk_ready = pyqtSignal(list, list, int)
    total_rows = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, parser, text: str, heavy: bool = False):
        super().__init__()
        self._parser = parser
        self._text = text
        self._heavy = heavy
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        rows, headers = self._parser(self._text)
        total = len(rows)
        self.total_rows.emit(total)

        if not self._heavy or total <= CHUNK_SIZE:
            self.chunk_ready.emit(rows, headers, 0)
            self.finished_signal.emit()
            return

        offset = 0
        while offset < total:
            if self._cancelled:
                return
            chunk = rows[offset:offset + CHUNK_SIZE]
            self.chunk_ready.emit(chunk, headers, offset)
            offset += len(chunk)
            self.progress.emit(offset, total)

        self.finished_signal.emit()


# ---------------------------------------------------------------------------
# Structured view
# ---------------------------------------------------------------------------
class StructuredView(QWidget):
    data_changed = pyqtSignal()

    def __init__(self, progress_bar: QProgressBar, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.progress_bar = progress_bar

        # Filter bar — shown for every structured file. The activate/deactivate
        # controls below it only apply to files with a checkbox column.
        self._filter_bar = QWidget()
        filter_row = QHBoxLayout(self._filter_bar)
        filter_row.setContentsMargins(0, 0, 0, 0)
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText(
            "Filter, e.g.  Checked=1 & A<1e-10   ·   bare word = search every "
            "column   ·   and/or/xor  & | ^"
        )
        self._filter_input.returnPressed.connect(self._apply_filter)
        filter_row.addWidget(self._filter_input, 1)
        self._filter_info = QLabel()
        self._filter_info.setStyleSheet("color: #666;")
        filter_row.addWidget(self._filter_info)
        self._btn_clear_filter = QPushButton("Clear")
        self._btn_clear_filter.setFixedWidth(50)
        self._btn_clear_filter.clicked.connect(self._clear_filter)
        filter_row.addWidget(self._btn_clear_filter)
        layout.addWidget(self._filter_bar)
        self._filter_bar.setVisible(False)

        # Actions bar — shown for every structured file. Add/Delete line always;
        # the activate-all controls and the count only where rows have a checkbox.
        self._toggle_bar = QWidget()
        toggle_lay = QVBoxLayout(self._toggle_bar)
        toggle_lay.setContentsMargins(0, 0, 0, 0)
        toggle_lay.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._btn_add_line = QPushButton("Add line")
        self._btn_add_line.setFixedWidth(76)
        self._btn_add_line.setToolTip(
            "Append a new row to the list (kept on export; gets a unique ID "
            "where the file has an ID column)"
        )
        self._btn_add_line.clicked.connect(self._add_line)
        self._btn_delete_line = QPushButton("Delete line")
        self._btn_delete_line.setFixedWidth(86)
        self._btn_delete_line.setToolTip(
            "Delete the selected row(s) from the list (asks for confirmation)"
        )
        self._btn_delete_line.clicked.connect(self._delete_line)
        self._btn_deactivate_all = QPushButton("Deactivate All")
        self._btn_deactivate_all.setFixedWidth(120)
        self._btn_deactivate_all.clicked.connect(lambda: self.model.toggle_all(False))
        self._btn_activate_all = QPushButton("Activate All")
        self._btn_activate_all.setFixedWidth(108)
        self._btn_activate_all.clicked.connect(lambda: self.model.toggle_all(True))
        self._toggle_info = QLabel()
        self._toggle_info.setStyleSheet("color: #666;")
        btn_row.addWidget(self._btn_add_line)
        btn_row.addWidget(self._btn_delete_line)
        btn_row.addWidget(self._btn_deactivate_all)
        btn_row.addWidget(self._btn_activate_all)
        btn_row.addWidget(self._toggle_info, 1)
        toggle_lay.addLayout(btn_row)
        layout.addWidget(self._toggle_bar)
        self._toggle_bar.setVisible(False)

        self.model = TableModel()
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setAlternatingRowColors(True)
        self._header = SortableHeaderView()
        self.table.setHorizontalHeader(self._header)
        self._header.setStretchLastSection(True)
        # Bound how many rows Qt scans when auto-sizing columns; without this a
        # reaction file (>10k rows) makes every column resize walk the whole
        # table.
        self._header.setResizeContentsPrecision(50)
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
        self.table.setWordWrap(False)
        self._header.sort_requested.connect(self._on_sort)
        layout.addWidget(self.table)

        self._parser = None
        self._serializer = None
        self._filename: str | None = None
        self._worker: LoadWorker | None = None
        self._active_col = False
        self.model.dataChanged.connect(self._update_info)
        self.model.dataChanged.connect(lambda i, j, r: self.table.viewport().update())
        self.model.dataChanged.connect(lambda i, j, r: self.data_changed.emit())

    def load(self, filename: str, text: str):
        self._cancel_pending()
        self._parser = PARSERS.get(filename)
        self._serializer = SERIALIZERS.get(filename)
        if not self._parser:
            self._filter_bar.setVisible(False)
            self._toggle_bar.setVisible(False)
            return

        self._filename = filename
        self._active_col = filename in TOGGLE_ACTIVE_FILES
        # Filter + Add/Delete line apply to any structured file; the
        # activate-all controls and the count only make sense with a checkbox.
        self._filter_bar.setVisible(True)
        self._toggle_bar.setVisible(True)
        self._btn_deactivate_all.setVisible(self._active_col)
        self._btn_activate_all.setVisible(self._active_col)
        self._toggle_info.setVisible(self._active_col)
        # Don't carry a filter typed for one file over to the next.
        self._filter_input.blockSignals(True)
        self._filter_input.clear()
        self._filter_input.blockSignals(False)
        mark_filter_validity(self._filter_input, None)
        self.model.set_filter("")

        heavy = filename in HEAVY_FILES

        if not heavy:
            rows, headers = self._parser(text)
            self.model.set_data(rows, headers, active_col=self._active_col)
            if filename in PRESERVE_INACTIVE_FILES:
                # Seed checkboxes from the file's own '#' markers instead of
                # defaulting everything active.
                self.model.set_active_flags([bool(r.get("_active", True)) for r in rows])
            self._fit_columns()
            self._resize_checkbox_col()
            self._update_info()
            return

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.model.clear()

        self._worker = LoadWorker(self._parser, text, heavy=True)
        self._worker.total_rows.connect(self._on_total_rows)
        self._worker.chunk_ready.connect(self._on_chunk)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_load_done)
        self._worker.start()

    def _cancel_pending(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
            self._worker = None

    def _on_total_rows(self, total):
        self.model.clear()

    def _on_chunk(self, chunk_rows, headers, offset):
        if offset == 0:
            self.model.set_headers(headers, active_col=self._active_col)
        self.model.append_rows(chunk_rows, headers)
        if offset == 0:
            self._hide_internal_columns()
            self.table.resizeColumnsToContents()
            self._resize_checkbox_col()

    def _on_progress(self, current, total):
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(current)

    def _on_load_done(self):
        self.progress_bar.setVisible(False)
        self._fit_columns()
        self._update_info()
        self._worker = None

    def _on_sort(self, col: int, ascending: bool):
        self.model.sort_by_column(col, ascending)

    def _update_info(self):
        total = len(self.model._data)
        visible = len(self.model._visible_rows())
        noun = "reactions" if self._filename in REACTION_FILES else "rows"
        # Row-count feedback for every structured file (shown only while a
        # filter is actually hiding something).
        if self._filter_input.text().strip() and visible != total:
            self._filter_info.setText(f"{visible}/{total} {noun}")
        else:
            self._filter_info.setText("")
        if not self._active_col or not self.model._active:
            return
        total = len(self.model._active)
        active = sum(1 for i in self.model._visible_rows() if self.model._active[i])
        if visible == total:
            self._toggle_info.setText(f"{active}/{total} {noun} active")
        else:
            self._toggle_info.setText(f"{active}/{visible}/{total} {noun} active (filtered)")

    def _apply_filter(self):
        expr = self._filter_input.text().strip()
        error = self.model.filter_error(expr)
        mark_filter_validity(self._filter_input, error)
        if error:
            return  # leave the last good result in place; the box stays flagged
        self.model.set_filter(expr)
        self._update_info()

    def _clear_filter(self):
        self._filter_input.clear()
        mark_filter_validity(self._filter_input, None)
        self.model.set_filter("")
        self._update_info()

    def _add_line(self):
        """Append a blank row to the current list, flush it so it is exported,
        and start editing its first cell."""
        if not self._parser or (self._worker and self._worker.isRunning()):
            return
        self.model.add_blank_row()
        # Flush the new row into the parent's file cache + mark unexported.
        self.data_changed.emit()
        self._update_info()
        # Reveal and focus the new row (it is the last visible one).
        view_row = self.model.rowCount() - 1
        first_col = 1 if self._active_col else 0
        idx = self.model.index(view_row, first_col)
        if idx.isValid():
            self.table.setCurrentIndex(idx)
            self.table.scrollTo(idx)
            self.table.edit(idx)

    def _delete_line(self):
        """Delete the selected row(s) after a warning + confirmation."""
        if not self._parser or (self._worker and self._worker.isRunning()):
            return
        sel = self.table.selectionModel()
        data_rows = sorted(
            {self.model._map_row(ix.row()) for ix in sel.selectedIndexes()}
        ) if sel else []
        if not data_rows:
            QMessageBox.information(
                self, "Delete line",
                "Select the row (or rows) you want to delete first.",
            )
            return

        hdrs = self.model.data_headers()
        id_col = next((i for i, h in enumerate(hdrs)
                       if h.strip().lower() == "id"), None)
        # Columns to show as "line content": every visible (non-internal) one.
        show_cols = [i for i, h in enumerate(hdrs) if not h.startswith("_")]

        def _describe(r: int) -> str:
            row = self.model._data[r]
            label = (row[0].strip() if row and row[0].strip() else f"row {r + 1}")
            if id_col is not None and id_col < len(row) and row[id_col].strip():
                label += f" (ID {row[id_col].strip()})"
            content = "  ".join(
                f"{hdrs[i]}={row[i].strip()}"
                for i in show_cols
                if i < len(row) and row[i].strip()
            )
            if len(content) > 160:
                content = content[:159] + "…"
            return f"• {label}\n    {content}" if content else f"• {label}"

        shown = data_rows[:5]
        preview = "\n".join(_describe(r) for r in shown)
        if len(data_rows) > len(shown):
            preview += f"\n• … and {len(data_rows) - len(shown)} more"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Delete line")
        box.setText(
            f"Delete {len(data_rows)} selected "
            f"{'row' if len(data_rows) == 1 else 'rows'} from {self._filename}?"
        )
        box.setInformativeText(
            f"{preview}\n\nThis cannot be undone. The rows are removed from the "
            "list now and dropped from the file on the next export."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        box.button(QMessageBox.StandardButton.Yes).setText("Delete")
        if box.exec() != QMessageBox.StandardButton.Yes:
            return

        self.model.delete_rows(data_rows)
        self.data_changed.emit()
        self._update_info()

    def _hide_internal_columns(self):
        """Hide columns whose header starts with '_' (verbatim round-trip data)."""
        for j in range(self.model.columnCount()):
            name = self.model._headers[j] if j < len(self.model._headers) else ""
            self.table.setColumnHidden(j, name.startswith("_"))

    def _fit_columns(self):
        # One-shot sizing (bounded by setResizeContentsPrecision), then leave the
        # columns Interactive. Persistent ResizeToContents on a >10k-row table
        # makes Qt re-measure every column on every scroll/repaint.
        self._hide_internal_columns()
        h = self.table.horizontalHeader()
        for j in range(self.model.columnCount()):
            h.setSectionResizeMode(j, QHeaderView.ResizeMode.Interactive)
        self.table.resizeColumnsToContents()
        self._resize_checkbox_col()

    def _resize_checkbox_col(self):
        if self._active_col:
            h = self.table.horizontalHeader()
            h.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
            h.resizeSection(0, 32)

    def to_text(self) -> str:
        if not self._serializer:
            return ""
        if self._filename in PRESERVE_INACTIVE_FILES:
            return self._serializer(self.model.to_dicts(only_active=False, include_active=True))
        return self._serializer(self.model.to_dicts())


# ---------------------------------------------------------------------------
# Plot view (embedded matplotlib)
# ---------------------------------------------------------------------------
def read_ab_file(path: Path, column: int = 1) -> tuple[list[float], list[float]]:
    """Read a nautilus ``ab/<species>.ab`` file.

    Column 0 is time [years]; ``column`` (default 1, the first spatial point)
    holds the abundance, scaled by :data:`NAUTILUS_OUTPUT_SCALE` on read so
    every caller works in abundance relative to H. Comment/blank lines are
    skipped.
    """
    times: list[float] = []
    values: list[float] = []
    with path.open(errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("!"):
                continue
            parts = s.split()
            if len(parts) <= column:
                continue
            try:
                t = float(parts[0])
                v = float(parts[column]) * NAUTILUS_OUTPUT_SCALE
            except ValueError:
                continue
            times.append(t)
            values.append(v)
    return times, values


class PlotView(QWidget):
    """A matplotlib canvas (+ navigation toolbar) for abundance plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4))
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._axes: list = []
        self._use_single()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self.show_placeholder()

    def _use_single(self):
        self._figure.clear()
        self._ax = self._figure.add_subplot(111)
        self._figure.subplots_adjust(left=0.13, right=0.95, top=0.9, bottom=0.13)
        self._axes = [self._ax]

    def _use_grid(self, rows: int, cols: int):
        self._figure.clear()
        axs = self._figure.subplots(rows, cols, squeeze=False)
        # even spacing: identical wspace and hspace, equal-size cells
        self._figure.subplots_adjust(left=0.09, right=0.975, top=0.90,
                                     bottom=0.10, wspace=0.22, hspace=0.22)
        self._axes = list(axs.flat)
        self._ax = self._axes[0]

    def show_placeholder(self, message='Run "Plot CO, H2O abundances" to see a plot here.'):
        self._use_single()
        self._ax.text(0.5, 0.5, message, ha="center", va="center",
                      transform=self._ax.transAxes, color="gray", fontsize=10)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw_idle()

    def plot_series(self, series: dict[str, tuple[list[float], list[float]]],
                    title: str = "", xlabel: str = "Time / yr",
                    ylabel: str = "Abundance", xlog: bool = True, ylog: bool = True,
                    xlim: tuple[float, float] | None = None,
                    observed: dict[str, float] | None = None,
                    vline: float | None = None,
                    black_labels: set[str] | None = None):
        """``series`` maps a label -> (x = times in years, y). ``xlog`` /
        ``ylog`` set the axis scales (x log, no minor ticks; y log unless
        it's the distance of disagreement). ``xlim`` is a year range.
        ``observed`` -> dashed horizontal bar per label. ``vline`` (a year)
        draws a solid black vertical line. ``black_labels`` draws those
        series as a thick black line on top with the rest thin and faint."""
        self._use_single()
        for label, (xs, ys) in series.items():
            if black_labels and label in black_labels:
                kw = dict(color="black", lw=2.6, zorder=10)
            elif black_labels:
                kw = dict(lw=0.9, alpha=0.45, zorder=2)
            else:
                kw = {}
            line, = self._ax.plot(xs, ys, label=label, **kw)
            ov = observed.get(label) if observed else None
            if ov and ov > 0:
                self._ax.axhline(ov, color=line.get_color(), ls="--",
                                 lw=0.9, alpha=0.7)
        self._ax.set_xscale("log" if xlog else "linear")
        self._ax.set_yscale("log" if ylog else "linear")
        if xlim:
            self._ax.set_xlim(*xlim)
            if not ylog:
                _fit_y_to_x(self._ax, *xlim)
        _tidy_ticks(self._ax)
        if not ylog:
            self._ax.yaxis.set_major_formatter(_DEX_FMT)
        if vline is not None:
            self._ax.axvline(vline, color="black", lw=1)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(ylabel)
        self._ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if series:
            n = len(series)
            self._ax.legend(
                loc="best",
                fontsize=("x-small" if n > 12 else "small" if n > 6 else "medium"),
                ncol=(3 if n > 24 else 2 if n > 10 else 1),
            )
        self._canvas.draw_idle()

    def set_xlim(self, xlim: tuple[float, float] | None):
        """Re-apply the x-range (a year range) and re-fit y to it."""
        if not xlim:
            return
        touched = False
        for ax in self._axes:
            if ax.lines:
                ax.set_xlim(*xlim)
                if ax.get_yscale() != "log":
                    _fit_y_to_x(ax, *xlim)
                touched = True
        if touched:
            self._canvas.draw_idle()

    @staticmethod
    def _draw_abundance_axes(ax, curves, observed, vline, xlim,
                             legend_fs="x-small"):
        """Draw one axes: log10(abundance) on a linear y vs log time.

        Colours are reset here — species are coloured from the tab palette
        in first-appearance order within *this* axes, so the first curve of
        every (sub)plot is tab blue.
        """
        order = list(dict.fromkeys(c[0] for c in curves))
        colour = {sp: _SPECIES_PALETTE[i % len(_SPECIES_PALETTE)]
                  for i, sp in enumerate(order)}
        for sp, _out, ls, xs, ys in curves:
            y = [math.log10(v) if v > _DISPLAY_FLOOR else math.nan for v in ys]
            ax.plot(xs, y, color=colour[sp], linestyle=ls, lw=1.2)
        if observed:
            for sp in order:
                ov = observed.get(sp)
                if ov and ov > 0:
                    ax.axhline(math.log10(ov), color=colour[sp], ls="--",
                               lw=0.8, alpha=0.6)
        ax.set_xscale("log")
        ax.set_yscale("linear")
        if xlim:
            ax.set_xlim(*xlim)
            _fit_y_to_x(ax, *xlim)          # scale y to the data in this x range
        _tidy_ticks(ax, xticks=5, yticks=6, integer_y=True)
        if vline is not None:
            ax.axvline(vline, color="black", lw=1)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        if order:
            ax.legend(
                [Line2D([], [], color=colour[s], lw=1.6) for s in order],
                order, loc="best", fontsize=legend_fs,
                ncol=(2 if len(order) > 10 else 1),
            )
        return order

    def plot_abundances(self, curves, species_order, output_styles,
                        observed=None, vline=None,
                        xlim: tuple[float, float] | None = None, title: str = ""):
        """Overlay abundance curves for several outputs on a single axes.

        ``curves``        : list of ``(species, output_name, linestyle, xs, ys)``
        ``output_styles`` : list of ``(output_name, linestyle)`` in display order
        ``observed``      : ``{species: observed abundance}`` -> dashed bar
        """
        self._use_single()
        self._draw_abundance_axes(self._ax, curves, observed, vline, xlim)
        self._ax.set_xlabel("Time / yr")
        self._ax.set_ylabel("log₁₀ abundance / n(H)")
        if len(output_styles) > 1:
            self._figure.legend(
                [Line2D([], [], color="black", linestyle=ls) for _n, ls in output_styles],
                [n for n, _ls in output_styles], loc="upper right",
                fontsize="x-small", title="output",
            )
        self._canvas.draw_idle()

    @staticmethod
    def _draw_disagreement_axes(ax, times, total, xlim):
        """Only the global (weighted) distance of disagreement, in black."""
        ax.plot(times, total, color="black", lw=2.0, zorder=10)
        mi = min(range(len(total)), key=total.__getitem__)
        ax.axvline(times[mi], color="black", lw=1)
        ax.text(0.5, 0.94, f"age={times[mi]:.2e}, D={total[mi]:.2f}",
                transform=ax.transAxes, ha="center", va="top", fontsize=7)
        ax.set_xscale("log")
        ax.set_yscale("linear")
        if xlim:
            ax.set_xlim(*xlim)
            _fit_y_to_x(ax, *xlim)
        _tidy_ticks(ax, xticks=5, yticks=6)
        ax.yaxis.set_major_formatter(_DEX_FMT)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)

    def plot_abundances_grid(self, cell_curves, cell_dis, species_order,
                             output_styles, cols: int, rows: int,
                             observed=None, vline=None,
                             xlim: tuple[float, float] | None = None,
                             title: str = "", disagreement=None):
        """Lay the abundance curves out in a ``rows x cols`` grid of subplots.

        ``cell_curves`` : list (row-major, length rows*cols) of per-cell
        ``curves`` lists. ``cell_dis[i]`` -> render that cell as the global
        distance-of-disagreement curve from ``disagreement`` (times, total)
        instead of species.
        """
        self._use_grid(rows, cols)
        for i, ax in enumerate(self._axes):
            if i < len(cell_dis) and cell_dis[i] and disagreement:
                self._draw_disagreement_axes(ax, *disagreement, xlim)
                ax.tick_params(labelsize=7)
                continue
            cell = cell_curves[i] if i < len(cell_curves) else []
            if not cell:
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, "—", ha="center", va="center",
                        transform=ax.transAxes, color="lightgray", fontsize=14)
                continue
            self._draw_abundance_axes(ax, cell, observed, vline, xlim,
                                      legend_fs="xx-small")
            ax.tick_params(labelsize=7)
        self._figure.supxlabel("Time / yr", fontsize=9)
        self._figure.supylabel("log₁₀ abundance / n(H)", fontsize=9)
        if len(output_styles) > 1:
            self._figure.legend(
                [Line2D([], [], color="black", linestyle=ls) for _n, ls in output_styles],
                [n for n, _ls in output_styles], loc="upper right",
                fontsize="x-small", title="output",
            )
        self._canvas.draw_idle()


class ControlPlotsView(QWidget):
    """A NxN grid of small abundance panels (the "Control plots" tab).

    Cell (0, 0) is the distance-of-disagreement curve; every other cell
    overlays the abundance-vs-time curves for the species assigned to it in
    ``control_plots.txt``.
    """

    def __init__(self, n: int = CONTROL_GRID_SIZE, parent=None):
        super().__init__(parent)
        self._n = n
        # Fixed margins (no constrained/tight layout solve) — the grid is
        # always the same shape, so pay that cost once here, not every draw.
        self._figure = Figure(figsize=(11, 9))
        # even spacing: identical wspace / hspace so every panel gap is equal.
        self._figure.subplots_adjust(left=0.07, right=0.985, top=0.92,
                                     bottom=0.08, wspace=0.22, hspace=0.22)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._axes = self._figure.subplots(n, n, squeeze=False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)
        self.show_placeholder()

    def show_placeholder(self, message='Click "Plot controls" to draw the grid.'):
        for ax in self._axes.flat:
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
        self._axes[0][0].text(
            0.5, 0.5, message, ha="center", va="center",
            transform=self._axes[0][0].transAxes, color="gray", fontsize=9,
        )
        self._canvas.draw_idle()

    def plot_grid(self, cells: list[list[dict]], disagreement, run_name: str = "",
                  xlim: tuple[float, float] | None = None):
        """``cells`` is an ``n x n`` list of ``{label: (xs, ys, observed)}``
        dicts (cell [0][0] is ignored; ``observed`` is the observed abundance
        or ``None``); ``disagreement`` is ``(xs, ys)`` or ``None``.

        Species panels plot log10(abundance) (linear y) vs log time (no
        minor ticks, no per-panel axis labels), sharing the x-range but
        each auto-scaling
        y to the data inside it. Cell (0, 0) is the linear-y distance of
        disagreement. Every panel gets a black vertical line at its minimum.
        """
        n = self._n
        tmin = None
        if disagreement and disagreement[0]:
            dx, dy = disagreement
            tmin = dx[min(range(len(dy)), key=dy.__getitem__)]

        xr = xlim
        if xr is None and disagreement and disagreement[0]:
            xr = (min(disagreement[0]), max(disagreement[0]))

        for r in range(n):
            for c in range(n):
                ax = self._axes[r][c]
                ax.clear()
                ax.tick_params(labelsize=6)
                ax.grid(True, which="both", linestyle=":", alpha=0.4)
                ax.set_xscale("log")
                if xr:
                    ax.set_xlim(*xr)

                if r == 0 and c == 0:
                    if disagreement and disagreement[0]:
                        dx, dy = disagreement
                        ax.plot(dx, dy, lw=1.2, color="black")
                        ax.set_yscale("linear")
                        mi = min(range(len(dy)), key=dy.__getitem__)
                        ax.text(0.5, 0.94, f"age={dx[mi]:.2e}, D={dy[mi]:.2f}",
                                transform=ax.transAxes, ha="center", va="top",
                                fontsize=6)
                    else:
                        ax.text(0.5, 0.5, "no observations", ha="center", va="center",
                                transform=ax.transAxes, color="gray", fontsize=8)
                    if tmin is not None:
                        ax.axvline(tmin, color="black", lw=1)
                    if xr:
                        _fit_y_to_x(ax, *xr)
                    _tidy_ticks(ax, xticks=4, yticks=4)
                    if not (disagreement and disagreement[0]):
                        ax.set_xticks([]); ax.set_yticks([])
                    else:
                        ax.yaxis.set_major_formatter(_DEX_FMT)
                    continue

                # species panels: log10(abundance) on a linear axis; ticks
                # read -6, -8, -10 (= 1e-6, 1e-8, 1e-10)
                ax.set_yscale("linear")
                series = cells[r][c] if r < len(cells) and c < len(cells[r]) else {}
                if not series:
                    ax.set_xticks([]); ax.set_yticks([])
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes, color="lightgray", fontsize=12)
                    if tmin is not None:
                        ax.axvline(tmin, color="black", lw=1)
                    continue

                for label, (xs, ys, observed) in series.items():
                    y = [math.log10(v) if v > _DISPLAY_FLOOR else math.nan
                         for v in ys]
                    line, = ax.plot(xs, y, lw=1, label=label)
                    if observed and observed > 0:
                        ax.axhline(math.log10(observed), color=line.get_color(),
                                   ls="--", lw=0.9, alpha=0.7)
                if tmin is not None:
                    ax.axvline(tmin, color="black", lw=1)
                ax.legend(fontsize=5, loc="best", ncol=1, handlelength=1.1,
                          borderpad=0.2, labelspacing=0.2, borderaxespad=0.2)
                if xr:
                    _fit_y_to_x(ax, *xr)
                _tidy_ticks(ax, xticks=4, yticks=4, integer_y=True)

        # x is shared -> only the bottom row shows its labels
        for r in range(n):
            for c in range(n):
                self._axes[r][c].tick_params(labelbottom=(r == n - 1))

        self._figure.supxlabel("Time / yr", fontsize=8)
        self._figure.supylabel("log₁₀ abundance / n(H)", fontsize=8)
        self._canvas.draw_idle()

    def set_xlim(self, xlim: tuple[float, float] | None):
        """Re-apply the shared x-range (a year range) and re-fit y to it."""
        xr = xlim
        if not xr:
            return
        drawn = False
        for row in self._axes:
            for ax in row:
                if ax.lines or ax.texts:
                    ax.set_xlim(*xr)
                    _fit_y_to_x(ax, *xr)
                    drawn = True
        if drawn:
            self._canvas.draw_idle()


# ---------------------------------------------------------------------------
# Export destination dialog (the ⚙ next to "Export to Folder")
# ---------------------------------------------------------------------------
class ExportSettingsDialog(QDialog):
    """Pick the base directory that run folders are created in. Records only —
    the run folder name is chosen later, when exporting."""

    def __init__(self, base_dir: Path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Location")
        self.setMinimumWidth(460)
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Run folders are created inside:"))
        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit(str(base_dir))
        browse = QPushButton("Browse…")
        browse.setFixedWidth(84)
        browse.clicked.connect(self._browse)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse)
        layout.addLayout(dir_row)

        hint = QLabel("You choose the run folder name when you click “Export to Folder”.")
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _browse(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Directory that run folders are created in", self._dir_edit.text()
        )
        if chosen:
            self._dir_edit.setText(chosen)

    def base_dir(self) -> Path | None:
        text = self._dir_edit.text().strip()
        return Path(text).expanduser() if text else None


# ---------------------------------------------------------------------------
# Settings window (its own top-level, non-modal window)
# ---------------------------------------------------------------------------
class SettingsWindow(QDialog):
    """Tabbed application settings — General (default outputs folder) and
    Plotting (x-axis limits) — as a separate, non-modal window. Emits
    :attr:`applied` with the validated values on OK / Apply.
    """

    applied = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(False)
        # A real independent window (own title bar, not glued on top of the
        # main window).
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)

        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        # --- General ---
        general = QWidget()
        gform = QFormLayout(general)
        dir_row = QHBoxLayout()
        self._outputs_edit = QLineEdit()
        browse = QPushButton("Browse…")
        browse.setFixedWidth(84)
        browse.clicked.connect(self._browse_outputs)
        dir_row.addWidget(self._outputs_edit, 1)
        dir_row.addWidget(browse)
        gform.addRow("Default outputs folder:", dir_row)
        self._tabs.addTab(general, "General")

        # --- Plot settings ---
        plotting = QWidget()
        pform = QFormLayout(plotting)
        self._xmin_edit = QLineEdit()
        self._xmax_edit = QLineEdit()
        pform.addRow("X min (years):", self._xmin_edit)
        pform.addRow("X max (years):", self._xmax_edit)
        hint = QLabel("X limits are applied to every plot's time axis.")
        hint.setStyleSheet("color: #666;")
        pform.addRow("", hint)

        self._obs_combo = QComboBox()
        self._obs_combo.addItem("Both", "both")
        self._obs_combo.addItem("In control plots", "control")
        self._obs_combo.addItem("In plots", "plots")
        pform.addRow("Display observed abundances:", self._obs_combo)

        self._tabs.addTab(plotting, "Plot settings")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).clicked.connect(self._ok)
        buttons.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self._apply)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

    # -- populate / read -------------------------------------------------
    def load(self, outputs_dir: str, xlim: tuple[float, float],
             observed_display: str = "both"):
        """Fill the fields from the current settings (called each open)."""
        self._outputs_edit.setText(outputs_dir)
        self._xmin_edit.setText(fmt_num(xlim[0]))
        self._xmax_edit.setText(fmt_num(xlim[1]))
        i = self._obs_combo.findData(observed_display)
        self._obs_combo.setCurrentIndex(i if i >= 0 else 0)

    def show_tab(self, index: int):
        self._tabs.setCurrentIndex(index)

    def _browse_outputs(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Default outputs folder", self._outputs_edit.text()
        )
        if chosen:
            self._outputs_edit.setText(chosen)

    def _validated(self) -> dict | None:
        try:
            xmin = float(self._xmin_edit.text())
            xmax = float(self._xmax_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Settings", "X limits must be numbers.")
            self._tabs.setCurrentIndex(1)
            return None
        if not (xmin > 0 and xmax > xmin):
            QMessageBox.warning(self, "Settings", "Need 0 < X min < X max.")
            self._tabs.setCurrentIndex(1)
            return None
        if not self._outputs_edit.text().strip():
            QMessageBox.warning(self, "Settings",
                                "Default outputs folder can't be empty.")
            self._tabs.setCurrentIndex(0)
            return None
        return {
            "outputs_dir": Path(self._outputs_edit.text().strip()).expanduser(),
            "xlim": (xmin, xmax),
            "observed_display": self._obs_combo.currentData(),
        }

    def _apply(self):
        vals = self._validated()
        if vals is not None:
            self.applied.emit(vals)

    def _ok(self):
        vals = self._validated()
        if vals is not None:
            self.applied.emit(vals)
            self.close()


# ---------------------------------------------------------------------------
# Plot-layout tab (arrange the checked species into a grid of subplots)
# ---------------------------------------------------------------------------
class _LayoutTab(QWidget):
    """cols x rows schematic; click a cell, then (un)check species to set
    what that subplot draws. ``get_layout()`` returns the assignment."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all: list[str] = []
        self._cells: list[set[str]] = [set()]
        self._active = 0

        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self._cols = QSpinBox(); self._cols.setRange(1, 6); self._cols.setValue(1)
        self._rows = QSpinBox(); self._rows.setRange(1, 6); self._rows.setValue(1)
        self._cols.valueChanged.connect(self._resize_grid)
        self._rows.valueChanged.connect(self._resize_grid)
        row.addWidget(QLabel("columns"))
        row.addWidget(self._cols)
        row.addWidget(QLabel("×  rows"))
        row.addWidget(self._rows)
        row.addStretch()
        v.addLayout(row)

        self._schematic = QWidget()
        self._schematic_lay = QGridLayout(self._schematic)
        self._schematic_lay.setSpacing(3)
        v.addWidget(self._schematic)

        v.addWidget(QLabel("Species in the selected subplot:"))
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        v.addWidget(self._list, 1)

        self._build_grid()

    # -- external -------------------------------------------------------
    def load(self, checked_species: set[str], layout: dict | None):
        self._all = sorted(checked_species)
        keep = set(checked_species) | {_DISAGREEMENT_KEY}
        if layout and layout.get("cells"):
            self._cols.blockSignals(True); self._rows.blockSignals(True)
            self._cols.setValue(max(1, min(6, layout.get("cols", 1))))
            self._rows.setValue(max(1, min(6, layout.get("rows", 1))))
            self._cols.blockSignals(False); self._rows.blockSignals(False)
            self._cells = [set(c) & keep for c in layout["cells"]]
        n = self._cols.value() * self._rows.value()
        self._cells = (self._cells + [set() for _ in range(n)])[:n] or [set()]
        self._active = 0
        self._build_grid()

    def get_layout(self) -> dict:
        return {"cols": self._cols.value(), "rows": self._rows.value(),
                "cells": [sorted(c) for c in self._cells]}

    # -- internals ----------------------------------------------------------
    def _resize_grid(self):
        n = self._cols.value() * self._rows.value()
        self._cells = (self._cells + [set() for _ in range(n)])[:n]
        self._active = min(self._active, n - 1)
        self._build_grid()

    def _build_grid(self):
        while self._schematic_lay.count():
            w = self._schematic_lay.takeAt(0).widget()
            if w:
                w.deleteLater()
        cols, rows = self._cols.value(), self._rows.value()
        self._cell_btns = []
        for i in range(cols * rows):
            b = QPushButton()
            b.setMinimumSize(70, 44)
            b.clicked.connect(lambda _c, k=i: self._select_cell(k))
            self._schematic_lay.addWidget(b, i // cols, i % cols)
            self._cell_btns.append(b)
        self._refresh()

    def _select_cell(self, i: int):
        self._active = i
        self._refresh()

    def _refresh(self):
        for i, b in enumerate(self._cell_btns):
            b.setText(f"[{i + 1}]\n{len(self._cells[i])} sp")
            b.setStyleSheet(
                "font-weight: bold; border: 2px solid #2a7ade;"
                if i == self._active else ""
            )
        self._list.blockSignals(True)
        self._list.clear()
        for sp in (_DISAGREEMENT_KEY, *self._all):
            it = QListWidgetItem(sp)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked
                             if sp in self._cells[self._active]
                             else Qt.CheckState.Unchecked)
            self._list.addItem(it)
        self._list.blockSignals(False)

    def _on_item_changed(self, item: QListWidgetItem):
        sp = item.text()
        if item.checkState() == Qt.CheckState.Checked:
            self._cells[self._active].add(sp)
        else:
            self._cells[self._active].discard(sp)
        self._cell_btns[self._active].setText(
            f"[{self._active + 1}]\n{len(self._cells[self._active])} sp"
        )


# ---------------------------------------------------------------------------
# Species selector dialog (for choosing what to plot)
# ---------------------------------------------------------------------------
class SpeciesSelectorDialog(QDialog):
    """A sortable / filterable checklist of every gas and grain species,
    plus a Layout tab to arrange the checked species into a subplot grid."""

    def __init__(self, rows: list[dict], preselected: set[str],
                 layout_state: dict | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Species to plot")
        self.resize(560, 640)

        headers = ["Species", "Type"] + list(ELEMENT_COLS)

        # --- model (built first: the Layout tab and count need it) ---
        self._model = TableModel()
        self._model._grey_inactive = False
        self._model.set_data(rows, headers, active_col=True, active_default=False)
        if preselected:
            self._model.set_active_values("Species", preselected)

        dlg_layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        dlg_layout.addWidget(self._tabs, 1)
        species_tab = QWidget()
        self._tabs.addTab(species_tab, "Species")
        self._layout_tab = _LayoutTab()
        self._layout_tab.load(self.selected_species(), layout_state)
        self._tabs.addTab(self._layout_tab, "Layout")
        self._tabs.currentChanged.connect(self._on_tab_changed)
        layout = QVBoxLayout(species_tab)

        # --- filter row ---
        filter_row = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter, e.g.  Checked=1 & O>0   ·   Type=grain   ·   bare word = "
            "search every column   ·   and/or/xor  & | ^"
        )
        self._filter.returnPressed.connect(self._apply_filter)
        filter_row.addWidget(self._filter, 1)
        btn_clear = QPushButton("Clear")
        btn_clear.setFixedWidth(52)
        btn_clear.clicked.connect(self._clear_filter)
        filter_row.addWidget(btn_clear)
        layout.addLayout(filter_row)

        # --- check-all / count row ---
        tools = QHBoxLayout()
        btn_check = QPushButton("Check All")
        btn_check.clicked.connect(lambda: self._model.toggle_all(True))
        btn_uncheck = QPushButton("Uncheck All")
        btn_uncheck.clicked.connect(lambda: self._model.toggle_all(False))
        self._count = QLabel()
        self._count.setStyleSheet("color: #666;")
        tools.addWidget(btn_check)
        tools.addWidget(btn_uncheck)
        tools.addWidget(self._count, 1)
        layout.addLayout(tools)

        # --- table ---
        self._table = QTableView()
        self._table.setModel(self._model)
        self._table.setAlternatingRowColors(True)
        self._table.setWordWrap(False)
        self._table.verticalHeader().setVisible(False)
        header = SortableHeaderView()
        self._table.setHorizontalHeader(header)
        header.setStretchLastSection(True)
        header.setResizeContentsPrecision(50)
        header.sort_requested.connect(self._model.sort_by_column)
        self._table.resizeColumnsToContents()
        self._table.setColumnWidth(0, 32)
        layout.addWidget(self._table)

        self._model.dataChanged.connect(self._update_count)

        # --- OK / Cancel ---
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # Don't let Enter (used to apply the filter) trigger OK.
        for btn in buttons.buttons():
            btn.setAutoDefault(False)
            btn.setDefault(False)
        dlg_layout.addWidget(buttons)

        self._update_count()

    def _on_tab_changed(self, index: int):
        # Refresh the Layout tab from the current species selection each time
        # it is shown.
        if self._tabs.widget(index) is self._layout_tab:
            self._layout_tab.load(self.selected_species(),
                                  self._layout_tab.get_layout())

    def layout_state(self) -> dict:
        return self._layout_tab.get_layout()

    def keyPressEvent(self, event):
        # Enter/Return applies the filter instead of accepting the dialog.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._apply_filter()
            return
        super().keyPressEvent(event)

    def _apply_filter(self):
        error = self._model.filter_error(self._filter.text())
        mark_filter_validity(self._filter, error)
        if error:
            return
        self._model.set_filter(self._filter.text())
        self._update_count()

    def _clear_filter(self):
        self._filter.clear()
        mark_filter_validity(self._filter, None)
        self._model.set_filter("")
        self._update_count()

    def _update_count(self, *args):
        total = len(self._model._data)
        checked = sum(1 for a in self._model._active if a)
        visible = len(self._model._visible_rows())
        if visible == total:
            self._count.setText(f"{checked} checked / {total} species")
        else:
            self._count.setText(f"{checked} checked / {visible} shown / {total} species")

    def selected_species(self) -> set[str]:
        return {d["Species"] for d in self._model.to_dicts(only_active=True)}


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------
class NautilusWrapper(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Nautilus Wrapper")
        self.setWindowIcon(app_icon())
        self.setMinimumSize(1200, 700)
        self._file_contents: dict[str, str] = {}
        self._mode = "comprehensive"
        # Which file the editor / structured view currently hold, and a guard
        # that suppresses the change->flush handlers while we populate a view
        # programmatically (so a load doesn't look like a user edit).
        self._current_file: str | None = None
        self._syncing = False
        self._run_dir: Path | None = None
        # Outputs available for the abundance plot: each entry is
        # {path, name, session: bool, checked: bool, linestyle}. Session runs
        # (nautilus run this session) sit on top and stay solid.
        self._outputs: list[dict] = []
        self._export_parent: Path = OUTPUTS_DIR  # default outputs folder
        self._plot_xlim: tuple[float, float] = DEFAULT_PLOT_XLIM
        # Where observed-abundance bars are drawn: "both" | "control" | "plots".
        self._observed_display: str = "both"
        self._plot_view_redraw = None  # last Plot-tab action, for live re-render
        self._files_exported = False
        self._exported_dirs: set[Path] = set()  # folders this session has written to
        self._observations_warned: set[Path] = set()  # run dirs already warned about
        self._plot_selection: set[str] = set()
        # Optional subplot layout for the abundance plot (from the ⚙ Layout
        # tab): {"cols", "rows", "cells": [species-per-cell]} or None.
        self._plot_layout: dict | None = None
        # Control-plots speedups: cache parsed ab/*.ab per run dir, keep the
        # per-file grid widgets alive, and render each grid tab only when it's
        # first shown.
        self._ab_cache: dict[tuple[str, str], tuple | None] = {}
        self._control_views: dict[str, ControlPlotsView] = {}
        self._control_files: dict[str, Path] = {}
        self._control_rendered: set[str] = set()
        self._disagreement_cache = None
        self._input_sets = get_input_sets()
        self._current_set_dir: Path | None = None
        self._init_ui()
        self._load_first_set()
        self.file_list.setCurrentRow(0)
        self._load_plot_settings()

    def _load_first_set(self):
        if self._input_sets:
            first_name = next(iter(self._input_sets))
            self._load_set(first_name)

    def _load_set(self, name: str):
        set_dir = self._input_sets.get(name)
        if set_dir is None:
            return
        self._current_set_dir = set_dir

        ambiguities = find_ambiguities(set_dir)
        self._file_overrides: dict[str, Path] = {}
        if ambiguities:
            self._resolve_ambiguities(ambiguities, set_dir)

        for fname in ALL_INPUT_FILES:
            if fname in self._file_overrides:
                path = self._file_overrides[fname]
            else:
                path = set_dir / fname
            if path.exists():
                self._file_contents[fname] = path.read_text(errors="replace")
            else:
                self._file_contents[fname] = ""
        self._files_exported = False
        if self._run_dir is not None:
            self._folder_label.setText(str(self._run_dir) + " (modified)")
        # The freshly-read contents supersede whatever the views hold; show
        # them without flushing the old set's data back over the new.
        self._refresh_current_view()

    def _resolve_ambiguities(self, ambiguities: dict[str, list[Path]], set_dir: Path):
        for expected, candidates in ambiguities.items():
            names = [f.name for f in candidates]
            default_idx = names.index(expected) if expected in names else 0
            chosen, ok = QInputDialog.getItem(
                self, "Ambiguous Files",
                f"Multiple files found for '{expected}':",
                names, default_idx, False,
            )
            if ok and chosen:
                self._file_overrides[expected] = set_dir / chosen

    def _on_set_changed(self, name: str):
        if name and name in self._input_sets:
            self._load_set(name)
            self.file_list.setCurrentRow(0)

    def _init_menu(self):
        self._settings_window: SettingsWindow | None = None
        settings_act = QAction("Settings…", self)
        # PreferencesRole => macOS moves it into the application menu (⌘,),
        # the OS-standard spot; elsewhere it just sits in the Edit menu.
        settings_act.setMenuRole(QAction.MenuRole.PreferencesRole)
        settings_act.triggered.connect(lambda *_a: self._open_settings())

        if sys.platform == "darwin":
            # Any menu works as a carrier; Qt harvests the action into the
            # app menu and hides the (now empty) carrier menu.
            self.menuBar().addMenu("Settings").addAction(settings_act)
        else:
            settings_act.setShortcut("Ctrl+,")
            self.menuBar().addMenu("&Edit").addAction(settings_act)

    def _open_settings(self, tab: int = 0):
        if self._settings_window is None:
            self._settings_window = SettingsWindow(self)
            self._settings_window.applied.connect(self._apply_settings)
        win = self._settings_window
        win.load(str(self._export_parent), self._plot_xlim, self._observed_display)
        win.show_tab(tab)
        win.show()
        win.raise_()
        win.activateWindow()

    def _apply_settings(self, vals: dict):
        self._export_parent = vals["outputs_dir"]
        self._plot_xlim = vals["xlim"]
        obs_changed = vals["observed_display"] != self._observed_display
        self._observed_display = vals["observed_display"]
        self._write_plot_settings()
        if self._run_dir is None:
            self._folder_label.setText(f"New runs → {self._export_parent}")
        # Push the new x-range onto everything already on screen.
        self._plot_view.set_xlim(self._plot_xlim)
        for view in self._control_views.values():
            view.set_xlim(self._plot_xlim)
        if obs_changed:
            # Observed bars are baked in at plot time — re-draw every grid
            # that's already been rendered, plus the current Plot-tab view.
            stale = set(self._control_rendered)
            self._control_rendered.clear()
            for i in range(self._control_tabs.count()):
                if self._control_tabs.tabText(i) in stale:
                    self._render_control_tab(i)
            if callable(self._plot_view_redraw) and self._plot_view._ax.lines:
                self._plot_view_redraw()
        self.statusBar().showMessage(
            f"Settings saved · outputs → {self._export_parent} · "
            f"x-limits {fmt_num(self._plot_xlim[0])}–{fmt_num(self._plot_xlim[1])} · "
            f"observed bars: {vals['observed_display']}"
        )

    def _init_ui(self):
        self._init_menu()
        # Every panel is a dock widget: they can be dragged, floated, re-docked
        # anywhere, and stacked on top of each other to form tab groups.
        self.setDockNestingEnabled(True)
        self.setDockOptions(
            QMainWindow.DockOption.AnimatedDocks
            | QMainWindow.DockOption.AllowNestedDocks
            | QMainWindow.DockOption.AllowTabbedDocks
            | QMainWindow.DockOption.GroupedDragging
        )
        self.setTabPosition(Qt.DockWidgetArea.AllDockWidgetAreas,
                            QTabWidget.TabPosition.North)
        # No central widget - the dock areas fill the whole window.
        _center_filler = QWidget()
        _center_filler.setFixedSize(0, 0)
        self.setCentralWidget(_center_filler)

        _ALL_AREAS = Qt.DockWidgetArea.AllDockWidgetAreas
        _NO_CLOSE = (
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # --- Left dock: file list ---
        dock_files = QDockWidget("Input Files", self)
        dock_files.setAllowedAreas(_ALL_AREAS)
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)

        self._set_combo = QComboBox()
        for name in self._input_sets:
            self._set_combo.addItem(name)
        self._set_combo.currentTextChanged.connect(self._on_set_changed)
        left_lay.addWidget(self._set_combo)

        self.file_list = QListWidget()
        for name in ALL_INPUT_FILES:
            self.file_list.addItem(name)
        self.file_list.currentTextChanged.connect(self._on_file_selected)
        left_lay.addWidget(self.file_list)
        dock_files.setWidget(left)

        # --- Editor dock: text editor / structured view ---
        self._dock_editor = QDockWidget("Editor", self)
        self._dock_editor.setAllowedAreas(_ALL_AREAS)
        self._dock_editor.setFeatures(_NO_CLOSE)
        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)

        hdr = QHBoxLayout()
        self.file_label = QLabel("Select a file")
        self.file_label.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px;")
        hdr.addWidget(self.file_label, 1)
        self.btn_switch = QPushButton("Switch to Text Mode")
        self.btn_switch.setFixedWidth(180)
        self.btn_switch.clicked.connect(self._toggle_mode)
        hdr.addWidget(self.btn_switch)
        center_lay.addLayout(hdr)

        # --- Progress bar in status bar ---
        self._progress_bar = QProgressBar()
        self._progress_bar.setVisible(False)
        self._progress_bar.setMaximumWidth(300)
        self._progress_bar.setMaximumHeight(18)
        self.statusBar().addPermanentWidget(self._progress_bar)

        self._stack = QStackedWidget()

        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Monaco", 11))
        self.editor.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        NautilusHighlighter(self.editor.document())
        self.editor.textChanged.connect(self._on_editor_changed)
        self._stack.addWidget(self.editor)

        self.struct_view = StructuredView(self._progress_bar)
        self.struct_view.data_changed.connect(self._on_struct_changed)
        self._stack.addWidget(self.struct_view)

        self._stack.setCurrentIndex(1)
        center_lay.addWidget(self._stack)
        self._dock_editor.setWidget(center)

        # --- Plot dock: embedded matplotlib canvas ---
        self._dock_plot = QDockWidget("Plot", self)
        self._dock_plot.setAllowedAreas(_ALL_AREAS)
        self._dock_plot.setFeatures(_NO_CLOSE)
        self._plot_view = PlotView()
        self._dock_plot.setWidget(self._plot_view)

        # --- Control plots dock: one 4x4-grid sub-tab per element file ---
        self._dock_control = QDockWidget("Control plots", self)
        self._dock_control.setAllowedAreas(_ALL_AREAS)
        self._dock_control.setFeatures(_NO_CLOSE)
        self._control_tabs = QTabWidget()
        self._control_tabs.setDocumentMode(True)
        self._control_tabs.currentChanged.connect(self._render_control_tab)
        self._dock_control.setWidget(self._control_tabs)

        # --- Right dock: actions ---
        dock_actions = QDockWidget("Actions", self)
        dock_actions.setAllowedAreas(_ALL_AREAS)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        folder_group = QGroupBox("Working Folder")
        folder_lay = QVBoxLayout(folder_group)
        self._folder_label = QLabel("Not set — created on export")
        self._folder_label.setWordWrap(True)
        self._folder_label.setMaximumHeight(48)
        self._folder_label.setStyleSheet("color: #666;")
        folder_lay.addWidget(self._folder_label)

        export_row = QHBoxLayout()
        self.btn_export_settings = QPushButton("⚙")
        self.btn_export_settings.setFixedWidth(36)
        self.btn_export_settings.setToolTip(
            "Set the parent directory and name for the export folder"
        )
        self.btn_export_settings.clicked.connect(self._configure_export_dir)
        self.btn_export = QPushButton("Export to Folder")
        self.btn_export.setToolTip(
            "Write the current inputs to the working folder "
            "(asks for a name the first time)"
        )
        self.btn_export.clicked.connect(self._export_files)
        export_row.addWidget(self.btn_export_settings)
        export_row.addWidget(self.btn_export, 1)
        folder_lay.addLayout(export_row)
        right_lay.addWidget(folder_group)

        import_group = QGroupBox("Outputs to plot")
        import_lay = QVBoxLayout(import_group)
        imp_btn_row = QHBoxLayout()
        self.btn_import_output = QPushButton("Import…")
        self.btn_import_output.setToolTip(
            "Pick one or more previous run folders to overlay in the abundance plot"
        )
        self.btn_import_output.clicked.connect(self._import_output)
        self.btn_output_up = QPushButton("▲")
        self.btn_output_up.setFixedWidth(32)
        self.btn_output_up.setToolTip(
            "Move the selected output up. The top row is the main run — "
            "plot_settings.txt is linked to it."
        )
        self.btn_output_up.clicked.connect(lambda: self._move_selected_output(-1))
        self.btn_output_down = QPushButton("▼")
        self.btn_output_down.setFixedWidth(32)
        self.btn_output_down.setToolTip("Move the selected output down")
        self.btn_output_down.clicked.connect(lambda: self._move_selected_output(1))
        self.btn_remove_output = QPushButton("Remove")
        self.btn_remove_output.setFixedWidth(80)
        self.btn_remove_output.setToolTip("Remove the selected output from the list")
        self.btn_remove_output.clicked.connect(self._remove_selected_output)
        imp_btn_row.addWidget(self.btn_import_output, 1)
        imp_btn_row.addWidget(self.btn_output_up)
        imp_btn_row.addWidget(self.btn_output_down)
        imp_btn_row.addWidget(self.btn_remove_output)
        import_lay.addLayout(imp_btn_row)
        self.output_list = QListWidget()
        self.output_list.setMaximumHeight(120)
        self.output_list.setToolTip(
            "Checked outputs are overlaid by “Plot”. Each species keeps its "
            "colour; each output gets its own linestyle (session runs solid).\n"
            "The top row is the main run: plot_settings.txt is saved there."
        )
        self.output_list.itemChanged.connect(self._on_output_item_changed)
        import_lay.addWidget(self.output_list)
        right_lay.addWidget(import_group)

        run_group = QGroupBox("Run")
        run_lay = QVBoxLayout(run_group)
        self.btn_run_code = QPushButton("Run nautilus")
        self.btn_run_code.setToolTip(
            f"{NAUTILUS_CLI} run — integrates the model, writes output.nc")
        self.btn_run_code.clicked.connect(self._run_nautilus_code)
        run_lay.addWidget(self.btn_run_code)
        self.btn_run_outputs = QPushButton("Write ab/ outputs")
        self.btn_run_outputs.setToolTip(
            f"{NAUTILUS_CLI} write-ascii abundances — converts output.nc "
            "to ab/<species>.ab")
        self.btn_run_outputs.clicked.connect(self._run_nautilus_outputs)
        run_lay.addWidget(self.btn_run_outputs)
        plot_row = QHBoxLayout()
        self.btn_plot_settings = QPushButton("⚙")
        self.btn_plot_settings.setFixedWidth(36)
        self.btn_plot_settings.setToolTip("Choose which species to plot…")
        self.btn_plot_settings.clicked.connect(self._open_plot_settings)
        self.btn_run_plot = QPushButton("Plot")
        self.btn_run_plot.setToolTip("Plot the checked species from the ab/ output")
        self.btn_run_plot.clicked.connect(self._run_plot_abundances)
        plot_row.addWidget(self.btn_plot_settings)
        plot_row.addWidget(self.btn_run_plot, 1)
        run_lay.addLayout(plot_row)
        self.btn_plot_disagreement = QPushButton("Plot Distance of Disagreement")
        self.btn_plot_disagreement.setToolTip(
            "For every activated observations.txt species with a matching "
            "ab/<species>.ab file, sum |observed − predicted| abundance "
            "over time and plot it"
        )
        self.btn_plot_disagreement.clicked.connect(self._run_plot_disagreement)
        run_lay.addWidget(self.btn_plot_disagreement)
        self.btn_plot_controls = QPushButton("Plot controls")
        self.btn_plot_controls.setToolTip(
            f"Draw a {CONTROL_GRID_SIZE}x{CONTROL_GRID_SIZE} grid for each "
            f"{CONTROL_PLOTS_DIR.name}/*.txt file (top-left = distance of "
            "disagreement) in the “Control plots” tab"
        )
        self.btn_plot_controls.clicked.connect(self._run_plot_controls)
        run_lay.addWidget(self.btn_plot_controls)
        self._run_progress = QProgressBar()
        self._run_progress.setVisible(False)
        self._run_progress.setMaximumHeight(18)
        run_lay.addWidget(self._run_progress)
        self._run_progress_label = QLabel()
        self._run_progress_label.setStyleSheet("color: #666; font-size: 11px;")
        self._run_progress_label.setVisible(False)
        run_lay.addWidget(self._run_progress_label)
        right_lay.addWidget(run_group)
        right_lay.addStretch()

        dock_actions.setWidget(right)

        # --- Logs dock ---
        dock_logs = QDockWidget("Logs", self)
        dock_logs.setAllowedAreas(_ALL_AREAS)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Monaco", 10))
        dock_logs.setWidget(self._log_view)

        # --- Assemble the default layout: 3 columns ------------------------
        #   1. Input Files
        #   2. Editor / Plot  (tab group)
        #   3. Actions (top) + Logs (below)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_files)
        self.splitDockWidget(dock_files, self._dock_editor, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._dock_editor, dock_actions, Qt.Orientation.Horizontal)
        self.splitDockWidget(dock_actions, dock_logs, Qt.Orientation.Vertical)
        # Stack the plots on the editor -> tabs in the same column. The
        # "Control plots" tab stays hidden until "Plot controls" is clicked.
        self.tabifyDockWidget(self._dock_editor, self._dock_plot)
        self.tabifyDockWidget(self._dock_plot, self._dock_control)
        self._dock_editor.raise_()
        self._dock_control.hide()

        self.resizeDocks(
            [dock_files, self._dock_editor, dock_actions],
            [220, 760, 320],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks([dock_actions, dock_logs], [260, 440], Qt.Orientation.Vertical)

        self.statusBar().showMessage("Ready")

    # -- text <-> structured linkage -------------------------------------
    # The two views never diverge: every edit in either is flushed straight
    # into ``self._file_contents`` (the single source of truth), and the
    # other view is rebuilt from it whenever it next becomes visible — on a
    # mode toggle or a file switch — so no change is ever lost or stale.

    def _flush_views(self, fname: str | None):
        """Write whatever the currently-shown view holds for ``fname`` back
        into ``self._file_contents``."""
        if not fname:
            return
        if self._mode == "text":
            self._file_contents[fname] = self.editor.toPlainText()
        elif fname in STRUCTURED_FILES:
            self._file_contents[fname] = self.struct_view.to_text()

    def _refresh_current_view(self):
        """(Re)populate the active view from ``self._file_contents`` without
        flushing first — for use after a bulk reload (input-set switch)."""
        item = self.file_list.currentItem()
        name = item.text() if item else None
        self._current_file = name
        if name is None:
            return
        self.file_label.setText(name)
        self._syncing = True
        try:
            if self._mode == "text":
                self.editor.setPlainText(self._file_contents.get(name, ""))
            else:
                self._load_struct_for_current()
        finally:
            self._syncing = False

    def _on_editor_changed(self):
        if self._syncing:
            return
        self._mark_unexported()
        if self._mode == "text" and self._current_file:
            self._file_contents[self._current_file] = self.editor.toPlainText()
            if self._current_file == "observations.txt":
                self._disagreement_cache = None

    def _on_struct_changed(self):
        if self._syncing:
            return
        self._mark_unexported()
        if self._mode == "comprehensive" and self._current_file in STRUCTURED_FILES:
            self._file_contents[self._current_file] = self.struct_view.to_text()
            if self._current_file == "observations.txt":
                self._disagreement_cache = None

    def _toggle_mode(self):
        self._flush_views(self._current_file)
        self._syncing = True
        try:
            if self._mode == "comprehensive":
                self._mode = "text"
                self._stack.setCurrentIndex(0)
                self.btn_switch.setText("Switch to Structured Mode")
                self.editor.setPlainText(self._file_contents.get(self._current_file, ""))
            else:
                self._mode = "comprehensive"
                self._stack.setCurrentIndex(1)
                self.btn_switch.setText("Switch to Text Mode")
                self._load_struct_for_current()
        finally:
            self._syncing = False

    def _on_file_selected(self, name: str):
        if name is None:
            return
        if not self._syncing:
            self._flush_views(self._current_file)
        self._current_file = name
        self.file_label.setText(name)
        self._syncing = True
        try:
            if self._mode == "text":
                self.editor.setPlainText(self._file_contents.get(name, ""))
            else:
                self._load_struct_for_current()
        finally:
            self._syncing = False

    def _load_struct_for_current(self):
        name = self.file_list.currentItem()
        if name is None:
            return
        name = name.text()
        if name in STRUCTURED_FILES:
            self.struct_view.load(name, self._file_contents.get(name, ""))
            self.struct_view.setVisible(True)
        else:
            self.struct_view.setVisible(False)

    def _next_run_dir(self, parent: Path | None = None) -> Path:
        """A ``<parent>/yyyymmdd_runN`` path, auto-incrementing N (default parent: outputs/)."""
        parent = parent or OUTPUTS_DIR
        parent.mkdir(parents=True, exist_ok=True)
        today = __import__("datetime").date.today().strftime("%Y%m%d")
        n = 1
        while (parent / f"{today}_run{n}").exists():
            n += 1
        return parent / f"{today}_run{n}"

    def _set_run_dir(self, path: Path, note: str = ""):
        if path != self._run_dir:
            self._disagreement_cache = None
            self._control_rendered.clear()
        self._run_dir = path
        self._folder_label.setText(str(path) + (f"  {note}" if note else ""))
        self._folder_label.setStyleSheet("")

    def _prompt_run_folder(self) -> Path | None:
        """Ask for the run folder name at export time; returns the full path."""
        parent = self._export_parent
        if self._run_dir is not None and self._run_dir.parent == parent:
            default = self._run_dir.name          # re-export to the same run
        else:
            default = self._next_run_dir(parent).name
        name, ok = QInputDialog.getText(
            self, "Export to Folder",
            f"Run folder name (created in {parent}):",
            text=default,
        )
        if not ok or not name.strip():
            return None
        return parent / name.strip()

    def _configure_export_dir(self):
        """The ⚙: set the base directory that run folders are created in.

        Records only — the run folder name is chosen when exporting.
        """
        dialog = ExportSettingsDialog(self._export_parent, self)
        if not dialog.exec():
            return
        base = dialog.base_dir()
        if base is None:
            return
        self._export_parent = base
        if self._run_dir is None:
            self._folder_label.setText(f"New runs → {base}")
        self.statusBar().showMessage(f"Export location: {base}")

    # -- outputs list --------------------------------------------------------
    def _pick_output_dirs(self) -> list[Path]:
        """A directory picker that allows selecting several folders at once."""
        dlg = QFileDialog(self, "Select one or more run folders",
                          str(self._export_parent))
        dlg.setFileMode(QFileDialog.FileMode.Directory)
        dlg.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
        for view in dlg.findChildren((QListView, QTreeView)):
            view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        if not dlg.exec():
            return []
        return [Path(p) for p in dlg.selectedFiles()]

    def _assign_output_linestyles(self):
        i = 0
        for e in self._outputs:
            if e.get("session"):
                e["linestyle"] = "-"
            else:
                e["linestyle"] = _IMPORT_LINESTYLES[i % len(_IMPORT_LINESTYLES)]
                i += 1

    def _main_output(self) -> dict | None:
        """The output at the top of the list — the "main" run that
        plot_settings.txt is linked to (or ``None`` if the list is empty)."""
        return self._outputs[0] if self._outputs else None

    def _plot_settings_path(self) -> Path:
        """Where plot_settings.txt lives: inside the main run's folder if there
        is one, else the current working run, else the repo-root fallback."""
        main = self._main_output()
        if main is not None:
            return Path(main["path"]) / PLOT_SETTINGS_NAME
        if self._run_dir is not None:
            return self._run_dir / PLOT_SETTINGS_NAME
        return PLOT_SETTINGS_PATH

    def _refresh_output_list(self):
        self._assign_output_linestyles()
        self.output_list.blockSignals(True)
        self.output_list.clear()
        for idx, e in enumerate(self._outputs):
            tags = []
            if idx == 0:
                tags.append("main")
            if e.get("session"):
                tags.append("session")
            label = e["name"] + (f"   [{', '.join(tags)}]" if tags else "")
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if e["checked"]
                             else Qt.CheckState.Unchecked)
            it.setData(Qt.ItemDataRole.UserRole, str(e["path"]))
            tip = str(e["path"])
            if idx == 0:
                f = it.font()
                f.setBold(True)
                it.setFont(f)
                tip += f"\nMain run — plot_settings.txt is linked here:\n{self._plot_settings_path()}"
            it.setToolTip(tip)
            self.output_list.addItem(it)
        self.output_list.blockSignals(False)

    def _on_output_item_changed(self, item: QListWidgetItem):
        p = item.data(Qt.ItemDataRole.UserRole)
        for e in self._outputs:
            if str(e["path"]) == p:
                e["checked"] = item.checkState() == Qt.CheckState.Checked
        if self._plot_view_redraw is not None and getattr(
                self._plot_view_redraw, "__func__", None
        ) is NautilusWrapper._run_plot_abundances:
            self._run_plot_abundances()

    def _move_selected_output(self, delta: int):
        """Shift the selected output up (delta<0) or down (delta>0) in the
        list. Moving a row to the top makes it the main run."""
        sel = self.output_list.selectedItems()
        if not sel:
            return
        path = sel[0].data(Qt.ItemDataRole.UserRole)
        idx = next((i for i, e in enumerate(self._outputs)
                    if str(e["path"]) == path), None)
        if idx is None:
            return
        new = idx + delta
        if not (0 <= new < len(self._outputs)):
            return
        prev_main = self._outputs[0]["path"]
        self._outputs[idx], self._outputs[new] = (
            self._outputs[new], self._outputs[idx])
        self._refresh_output_list()
        self.output_list.setCurrentRow(new)
        if self._outputs[0]["path"] != prev_main:
            self._on_main_output_changed()

    def _on_main_output_changed(self):
        """The top-of-list (main) run changed. Adopt that run's linked
        plot_settings.txt if it has one, so the file stays tied to the run;
        re-render the abundance plot if one is on screen."""
        path = self._plot_settings_path()
        if path.exists():
            self._load_plot_settings()
        self.statusBar().showMessage(
            f"Main run: {self._outputs[0]['name']} · plot_settings.txt → {path}"
        )
        if getattr(self._plot_view_redraw, "__func__", None) \
                is NautilusWrapper._run_plot_abundances:
            self._run_plot_abundances()

    def _remove_selected_output(self):
        paths = {it.data(Qt.ItemDataRole.UserRole)
                 for it in self.output_list.selectedItems()}
        if not paths:
            return
        prev_main = self._outputs[0]["path"] if self._outputs else None
        self._outputs = [e for e in self._outputs if str(e["path"]) not in paths]
        self._refresh_output_list()
        new_main = self._outputs[0]["path"] if self._outputs else None
        if new_main is not None and new_main != prev_main:
            self._on_main_output_changed()

    def _register_session_output(self, path: Path):
        """Put a run generated this session on top of the list, checked, solid."""
        prev_main = self._outputs[0]["path"] if self._outputs else None
        self._outputs = [e for e in self._outputs if e["path"] != path]
        self._outputs.insert(0, {"path": path, "name": path.name,
                                 "session": True, "checked": True})
        self._refresh_output_list()
        if path != prev_main:
            self._on_main_output_changed()

    def _import_output(self):
        """Add one or more previous run folders to the outputs list."""
        had_outputs = bool(self._outputs)
        added, skipped = [], []
        for folder in self._pick_output_dirs():
            ab_dir = folder / "ab"
            if not ab_dir.is_dir() or not any(ab_dir.glob("*.ab")):
                skipped.append(folder.name)
                continue
            if any(e["path"] == folder for e in self._outputs):
                continue
            self._outputs.append({"path": folder, "name": folder.name,
                                  "session": False, "checked": True})
            added.append(folder.name)
        if skipped:
            QMessageBox.warning(
                self, "No output",
                "No 'ab/' folder with .ab files in:\n  " + "\n  ".join(skipped),
            )
        if not added:
            return
        self._refresh_output_list()
        # Point the disagreement / control-plot / run target at the import too.
        self._set_run_dir(self._outputs[-1]["path"], note="(imported)")
        self._files_exported = False
        self.statusBar().showMessage(f"Imported: {', '.join(added)}")
        if not had_outputs:
            # First output(s) in the list: the top one is now the main run.
            self._on_main_output_changed()
        elif self._plot_selection:
            self._run_plot_abundances()

    def _sync_current_file(self):
        self._flush_views(self._current_file)

    def _mark_unexported(self):
        self._files_exported = False
        if self._run_dir is not None:
            self._folder_label.setText(str(self._run_dir) + " (modified)")

    def _export_files(self):
        self._sync_current_file()

        target = self._prompt_run_folder()
        if target is None:
            return

        empty = [f for f in REQUIRED_INPUT_FILES if not self._file_contents.get(f, "").strip()]
        if empty:
            resp = QMessageBox.question(
                self, "Empty Input Files",
                "These required input files are empty and will be written as empty "
                "files (Nautilus may refuse to run or fail):\n\n  "
                + "\n  ".join(empty)
                + "\n\nExport anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        self._set_run_dir(target)
        dest = self._run_dir
        if dest.exists() and dest.resolve() not in self._exported_dirs:
            clashing = sorted(f for f in self._file_contents if (dest / f).exists())
            has_output = ((dest / "ab").is_dir() or (dest / "output.nc").exists()
                          or any(dest.glob("*.out")))
            if clashing or has_output:
                details = []
                if clashing:
                    details.append(
                        f"{len(clashing)} input file(s) will be overwritten:\n  "
                        + ", ".join(clashing)
                    )
                if has_output:
                    details.append(
                        "The folder also holds results from a previous run "
                        "(output.nc, ab/, *.out) — those are left in place but will no longer "
                        "match the new inputs."
                    )
                resp = QMessageBox.warning(
                    self, "Folder Already Exists",
                    f"{dest}\n\n" + "\n\n".join(details) + "\n\nExport anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return

        dest.mkdir(parents=True, exist_ok=True)
        for fname, content in self._file_contents.items():
            (dest / fname).write_text(content)
        self._exported_dirs.add(dest.resolve())

        self._files_exported = True
        self._folder_label.setText(str(self._run_dir))
        self.statusBar().showMessage(f"Exported {len(self._file_contents)} files to {dest}")
        QMessageBox.information(
            self, "Export Complete",
            f"Successfully exported {len(self._file_contents)} files to:\n{dest}"
        )

    def _check_ready_to_run(self) -> bool:
        if self._run_dir is None:
            QMessageBox.warning(self, "No Folder", "Export the inputs to a folder first.")
            return False
        if not self._files_exported:
            QMessageBox.warning(
                self, "Not Exported",
                "Files have not been exported since last edit.\n"
                "Export to folder first."
            )
            return False
        self._ensure_observations_file()
        return True

    def _ensure_observations_file(self):
        """If the run folder has no observations.txt (or an empty one),
        write an empty placeholder and warn once — the "Plot Distance of
        Disagreement" button needs observed abundances in it to do anything."""
        if self._run_dir is None:
            return
        path = self._run_dir / "observations.txt"
        has_content = path.exists() and bool(path.read_text(errors="replace").strip())
        if has_content:
            return
        path.write_text("")
        resolved = path.resolve()
        if resolved in self._observations_warned:
            return
        self._observations_warned.add(resolved)
        QMessageBox.warning(
            self, "No Observations",
            "No observations.txt was found for this input set (or it was "
            f"empty).\n\nAn empty placeholder was created at:\n{path}\n\n"
            'Add observed abundances to it (one "Species <- value  # ref" '
            "line per species; prefix a line with '#' to deactivate it) to "
            'enable the "Plot Distance of Disagreement" button.',
        )

    def _log(self, text: str):
        self._log_view.append(text)
        self._log_view.verticalScrollBar().setValue(
            self._log_view.verticalScrollBar().maximum()
        )

    def _run_command(self, cmd: list[str], label: str, show_progress: bool = False):
        if not self._check_ready_to_run():
            return

        self._log(f"\n{'='*60}\n> {label}\n  dir: {self._run_dir}\n  cmd: {' '.join(cmd)}\n{'='*60}")
        self.statusBar().showMessage(f"Running {label}...")
        if show_progress:
            self._run_progress.setRange(0, 100)
            self._run_progress.setValue(0)
            self._run_progress.setVisible(True)
            self._run_progress_label.setVisible(True)
            self._run_progress_label.setText("Starting...")
        else:
            self._run_progress.setVisible(True)
            self._run_progress.setRange(0, 0)
            self._run_progress_label.setVisible(True)
            self._run_progress_label.setText("Generating outputs...")
        self._run_worker = RunWorker(cmd, str(self._run_dir))
        self._run_worker.log_line.connect(self._log)
        if show_progress:
            self._run_worker.progress_update.connect(self._on_run_progress)
        self._run_worker.finished_signal.connect(
            lambda code: self._on_run_finished(label, code)
        )
        self._run_worker.start()

    def _on_run_progress(self, pct: int, text: str):
        self._run_progress.setValue(pct)
        self._run_progress_label.setText(text)

    def _on_run_finished(self, label: str, code: int):
        self._run_progress.setVisible(False)
        self._run_progress_label.setVisible(False)
        if code == 0:
            # Fresh ab/ output — drop the caches so the next plot re-reads it.
            self._ab_cache.clear()
            self._disagreement_cache = None
            self._control_rendered.clear()
            if label == WRITE_AB_LABEL and self._run_dir is not None \
                    and (self._run_dir / "ab").is_dir():
                self._register_session_output(self._run_dir)
            self.statusBar().showMessage(f"{label} completed successfully")
            self._log(f"[{label}] completed (exit 0)")
        else:
            self.statusBar().showMessage(f"{label} failed (exit code {code})")
            self._log(f"[{label}] FAILED (exit {code})")
            if code == -1:
                detail = "Could not start the process (see the log for details)."
            else:
                detail = f"Process exited with code {code}. See the log for details."
            QMessageBox.warning(self, label, f"{label} failed.\n\n{detail}")

    def _run_nautilus_code(self):
        nautilus_bin = self._find_nautilus_bin()
        if nautilus_bin:
            self._run_command([str(nautilus_bin), "run"], RUN_LABEL,
                              show_progress=True)

    def _run_nautilus_outputs(self):
        if self._run_dir is not None and not (self._run_dir / "output.nc").exists():
            QMessageBox.warning(
                self, "No output.nc",
                f"No output.nc in:\n{self._run_dir}\n\nRun nautilus first.")
            return
        nautilus_bin = self._find_nautilus_bin()
        if nautilus_bin:
            self._run_command([str(nautilus_bin), "write-ascii", "abundances"],
                              WRITE_AB_LABEL, show_progress=True)

    def _collect_species_rows(self) -> list[dict]:
        """One row per gas/grain species: name, type, elemental composition."""
        rows: list[dict] = []
        for fname, kind in (("gas_species.in", "gas"), ("grain_species.in", "grain")):
            text = self._file_contents.get(fname, "")
            if not text.strip():
                continue
            parsed, _ = parse_species(text)
            for r in parsed:
                row = {"Species": r["Species"], "Type": kind}
                for col in ELEMENT_COLS:
                    row[col] = r.get(col, "0")
                rows.append(row)
        return rows

    def _load_plot_settings(self):
        """Restore species / layout / style from the main run's
        plot_settings.txt if present.

        A missing or malformed file is not an error — the app just keeps its
        current (or built-in default) settings.
        """
        path = self._plot_settings_path()
        try:
            text = path.read_text(errors="replace")
        except OSError:
            return
        try:
            cfg = parse_plot_settings(text)
        except Exception as exc:  # a bad file must never block startup
            self._log(f"[Plot] ignoring {path}: {exc}")
            return

        if "species" in cfg:
            self._plot_selection = set(cfg["species"])
        lay = cfg.get("layout")
        if lay:
            cells = [
                [_DISAGREEMENT_KEY if t == DISAGREEMENT_TOKEN else t
                 for t in cell]
                for cell in lay["cells"]
            ]
            self._plot_layout = {"cols": lay["cols"], "rows": lay["rows"],
                                 "cells": cells}
        style = cfg.get("style", {})
        if "xlim" in style:
            self._plot_xlim = style["xlim"]
        if "observed" in style:
            self._observed_display = style["observed"]
        # `distribution` records the last plot's per-output line styles; the
        # output list is rebuilt from disk each run, so there is nothing to
        # restore from it here.

    def _write_plot_settings(self):
        """Persist the current abundance-plot configuration to the main run's
        plot_settings.txt (plotted species, subplot layout, style and the
        per-output line-style distribution). The main run is the output at the
        top of the "Outputs to plot" list."""
        lay = self._plot_layout or {}
        cells = [
            [DISAGREEMENT_TOKEN if t == _DISAGREEMENT_KEY else t for t in cell]
            for cell in lay.get("cells", [])
        ]
        settings = {
            "species": sorted(self._plot_selection),
            "layout": {"cols": lay.get("cols", 1), "rows": lay.get("rows", 1),
                       "cells": cells},
            "style": {"xlim": self._plot_xlim,
                      "observed": self._observed_display},
            "distribution": [(e["name"], e.get("linestyle", "-"))
                             for e in self._outputs if e["checked"]],
        }
        path = self._plot_settings_path()
        try:
            path.write_text(serialize_plot_settings(settings))
        except OSError as exc:
            self._log(f"[Plot] could not write {path}: {exc}")

    def _open_plot_settings(self):
        rows = self._collect_species_rows()
        if not rows:
            QMessageBox.warning(
                self, "No species",
                "gas_species.in / grain_species.in are empty.\n"
                "Load an input set first.",
            )
            return
        dialog = SpeciesSelectorDialog(rows, self._plot_selection,
                                       self._plot_layout, self)
        if dialog.exec():
            self._plot_selection = dialog.selected_species()
            self._plot_layout = dialog.layout_state()
            self._write_plot_settings()
            lay = self._plot_layout
            n = lay["cols"] * lay["rows"]
            self.statusBar().showMessage(
                f"{len(self._plot_selection)} species selected"
                + (f" · layout {lay['cols']}×{lay['rows']}" if n > 1 else "")
            )
            if getattr(self._plot_view_redraw, "__func__", None) \
                    is NautilusWrapper._run_plot_abundances:
                self._run_plot_abundances()

    def _run_plot_abundances(self):
        """Overlay the selected species for every checked output in the Plot
        tab — one colour per species, one linestyle per output."""
        outs = [e for e in self._outputs if e["checked"]]
        if not outs:
            QMessageBox.warning(
                self, "No output",
                "No outputs are checked in “Outputs to plot”.\n\n"
                "Run nautilus and write the ab/ outputs, or use Import… to add a run folder.",
            )
            return
        if not self._plot_selection:
            QMessageBox.warning(
                self, "No species selected",
                "No species are checked for plotting.\n\n"
                "Click the ⚙ button next to Plot to choose species.",
            )
            return

        self._assign_output_linestyles()
        species = sorted(self._plot_selection)
        curves = []          # (species, output_name, linestyle, xs, ys)
        missing: list[str] = []
        for e in outs:
            ab_dir = e["path"] / "ab"
            for sp in species:
                cv = self._cached_ab(ab_dir, sp)
                if cv is None:
                    missing.append(f"{e['name']}/{sp}")
                    continue
                curves.append((sp, e["name"], e["linestyle"], cv[0], cv[1]))

        if not curves:
            QMessageBox.warning(
                self, "No abundance data",
                f"None of the {len(species)} selected species have a .ab file "
                f"in the {len(outs)} checked output(s).\n\n"
                "Run nautilus and write the ab/ outputs first.",
            )
            return

        observed = (self._observed_map()
                    if self._observed_display in ("both", "plots") else None)
        out_styles = [(e["name"], e["linestyle"]) for e in outs]

        lay = self._plot_layout
        use_grid = bool(lay and lay["cols"] * lay["rows"] > 1
                        and any(lay["cells"]))
        if use_grid:
            cell_curves, cell_dis = [], []
            for cell in lay["cells"]:
                cs = set(cell)
                cell_dis.append(_DISAGREEMENT_KEY in cs)
                cs.discard(_DISAGREEMENT_KEY)
                cell_curves.append([cu for cu in curves if cu[0] in cs])
            dis_data = None
            if any(cell_dis):
                t, tot, _dinfo = self._disagreement_curve()
                if t is not None:
                    dis_data = (t, tot)
            self._plot_view.plot_abundances_grid(
                cell_curves, cell_dis, species, out_styles,
                lay["cols"], lay["rows"],
                observed=observed, vline=self._disagreement_tmin(),
                xlim=self._plot_xlim, disagreement=dis_data,
            )
        else:
            self._plot_view.plot_abundances(
                curves, species, out_styles,
                observed=observed, vline=self._disagreement_tmin(),
                xlim=self._plot_xlim,
            )
        self._plot_view_redraw = self._run_plot_abundances
        self._write_plot_settings()
        self._dock_plot.show()
        self._dock_plot.raise_()
        if missing:
            self._log(f"[Plot] no .ab file for: {', '.join(missing)}")
        self.statusBar().showMessage(
            f"Plotted {len(species)} species across {len(outs)} output(s)"
            + (f" in a {lay['cols']}×{lay['rows']} grid" if use_grid else "")
        )

    def _observed_map(self) -> dict[str, float]:
        """``{species-label: observed abundance}`` from the active
        observations.txt (empty when there is none)."""
        cache = self._disagreement_cache
        if cache is not None and cache[1].get("observed"):
            return cache[1]["observed"]
        obs_text = self._file_contents.get("observations.txt", "")
        if not obs_text.strip() and self._run_dir is not None:
            p = self._run_dir / "observations.txt"
            if p.exists():
                obs_text = p.read_text(errors="replace")
        if not obs_text.strip():
            return {}
        rows, _ = parse_observations(obs_text)
        by, _ = evaluate_observations(rows)
        return {lbl: val for lbl, (val, _c, _a) in by.items()}

    def _disagreement_tmin(self) -> float | None:
        """Time of minimum distance of disagreement for the current run, or
        ``None`` when it can't be computed. Reuses / fills the cache."""
        if self._disagreement_cache is None:
            t, total, info = self._disagreement_curve()
            self._disagreement_cache = ((t, total) if t is not None else None, info)
        dis = self._disagreement_cache[0]
        if not dis or not dis[0]:
            return None
        times, total = dis
        return times[min(range(len(total)), key=total.__getitem__)]

    def _cached_ab(self, ab_dir: Path, species: str):
        """``read_ab_file`` for ``ab_dir/<species>.ab``, memoised per
        (folder, species). The cache is dropped when a nautilus run
        completes or the run dir changes.

        Returns ``(times, values)`` or ``None`` (missing / empty).
        """
        key = (str(ab_dir), species)
        hit = self._ab_cache.get(key, 0)  # 0 = not looked up yet
        if hit != 0:
            return hit
        path = ab_dir / f"{species}.ab"
        try:
            res = read_ab_file(path)
        except OSError:
            res = None
        if not res or not res[0]:
            res = None
        self._ab_cache[key] = res
        return res

    def _read_component_sum(self, ab_dir: Path, components):
        """Sum the ``ab/<comp>.ab`` abundance curves for ``components`` on a
        common time grid. Returns ``(times, values)`` or ``None`` if none of
        the component .ab files exist."""
        series = [cv for comp in components
                  if (cv := self._cached_ab(ab_dir, comp)) is not None]
        if not series:
            return None
        grid = max((t for t, _v in series), key=len)
        summed = [0.0] * len(grid)
        for _t, v in series:
            for i in range(min(len(summed), len(v))):
                summed[i] += v[i]
        return grid, summed

    def _disagreement_curve(self):
        """Compute the distance-of-disagreement curve for the current run.

        Species are grouped by key atom (C, N, O, S) from the ``### <name>``
        section they sit under in observations.txt. For each atom an
        intermediate curve is the mean over that section's observed species
        of ``|log10(predicted(t)) − log10(observed)|`` (predicted is already
        the ×2 abundance relative to H; normalised by the species count).
        The returned curve is the species-count-weighted average of those
        intermediates.

        Returns ``(times, total, info)``; ``times`` is ``None`` when it can't
        be computed, with ``info["reason"]`` explaining why. ``info`` also
        carries ``per_atom`` (``{atom: (curve, n_species)}``), ``n_species``,
        ``n_observed``, ``observed``, ``missing`` / ``partial`` / ``errors``.
        """
        info = {"reason": None, "n_species": 0, "n_observed": 0,
                "missing": [], "partial": [], "errors": {}, "observed": {},
                "per_atom": {}}
        if self._run_dir is None:
            info["reason"] = "No working folder yet — export the inputs, or use “Import output”."
            return None, None, info

        obs_text = self._file_contents.get("observations.txt", "")
        if not obs_text.strip():
            obs_path = self._run_dir / "observations.txt"
            if obs_path.exists():
                obs_text = obs_path.read_text(errors="replace")
        if not obs_text.strip():
            info["reason"] = ("observations.txt has no observed abundances to "
                              "compare against.")
            return None, None, info

        rows, _ = parse_observations(obs_text)
        obs_by_species, errors = evaluate_observations(rows)
        info["errors"] = errors
        info["n_observed"] = len(obs_by_species)
        info["observed"] = {lbl: val for lbl, (val, _c, _a) in obs_by_species.items()}
        if not obs_by_species:
            info["reason"] = ("No activated (uncommented) line in "
                              "observations.txt could be evaluated.")
            return None, None, info

        ab_dir = self._run_dir / "ab"
        # atom -> list of per-species |Δlog10| curves
        atom_terms: dict[str, list[list[float]]] = {a: [] for a in KEY_ATOMS}
        used: set[str] = set()
        times: list[float] | None = None

        for label, (observed, components, atom) in obs_by_species.items():
            res = self._read_component_sum(ab_dir, components)
            if res is None:
                info["missing"].append(
                    f"{label} ({'+'.join(components)})"
                    if len(components) > 1 else label
                )
                continue
            grid, predicted = res
            if len(components) > 1:
                present = [c for c in components
                           if self._cached_ab(ab_dir, c) is not None]
                if len(present) < len(components):
                    info["partial"].append(f"{label} (only {'+'.join(present)})")
            if times is None or len(grid) > len(times):
                times = grid

            if atom not in KEY_ATOMS or observed <= 0:
                continue
            log_obs = math.log10(observed)
            dlog = [abs(math.log10(max(p, _LOG_FLOOR)) - log_obs) for p in predicted]
            atom_terms[atom].append(dlog)
            used.add(label)

        per_atom = {a: terms for a, terms in atom_terms.items() if terms}
        if not per_atom or times is None:
            info["reason"] = ("No observed species sit under a "
                              "### Carbon/Nitrogen/Oxygen/Sulfur section with "
                              f"matching .ab output in:\n{ab_dir}")
            return None, None, info

        n = len(times)
        atom_curve: dict[str, list[float]] = {}
        atom_n: dict[str, int] = {}
        for a, terms in per_atom.items():
            acc = [0.0] * n
            for dl in terms:
                for i in range(min(n, len(dl))):
                    acc[i] += dl[i]
            atom_n[a] = len(terms)
            atom_curve[a] = [x / len(terms) for x in acc]   # normalise by count

        total_w = sum(atom_n.values())
        total = [0.0] * n
        for a, curve in atom_curve.items():
            w = atom_n[a]
            for i in range(n):
                total[i] += w * curve[i]
        total = [x / total_w for x in total]                 # weighted average

        info["per_atom"] = {a: (atom_curve[a], atom_n[a]) for a in atom_curve}
        info["n_species"] = len(used)
        return times, total, info

    def _log_disagreement_notes(self, info: dict):
        if info.get("missing"):
            self._log(f"[Disagreement] no .ab output for: {', '.join(sorted(info['missing']))}")
        if info.get("partial"):
            self._log(f"[Disagreement] summed only part of: {', '.join(sorted(info['partial']))}")
        if info.get("errors"):
            self._log(
                "[Disagreement] could not evaluate: "
                + ", ".join(f"{k} ({v})" for k, v in info["errors"].items())
            )

    def _run_plot_disagreement(self):
        """Plot the distance of disagreement — the C/N/O/S species-count-
        weighted average of the per-atom mean ``|log10(predicted(t)) −
        log10(observed)|`` (black) — plus each per-atom intermediate curve,
        with a black vertical line at its minimum."""
        times, total, info = self._disagreement_curve()
        if times is None:
            QMessageBox.warning(self, "Distance of Disagreement", info["reason"])
            return

        tmin = times[min(range(len(total)), key=total.__getitem__)]
        series = {"D(t)": (times, total)}
        for a, (curve, na) in sorted(info.get("per_atom", {}).items()):
            series[f"{a}  ({na})"] = (times, curve)
        self._plot_view.plot_series(
            series,
            title=f"Distance of Disagreement ({info['n_species']} species) - {self._run_dir.name}",
            ylabel="distance of disagreement / dex",
            xlog=True, ylog=False, xlim=self._plot_xlim,
            vline=tmin, black_labels={"D(t)"},
        )
        self._plot_view_redraw = self._run_plot_disagreement
        self._dock_plot.show()
        self._dock_plot.raise_()
        self._log_disagreement_notes(info)
        self.statusBar().showMessage(
            f"Plotted distance of disagreement over {info['n_species']} species"
            + (f" ({len(info['missing'])} without output)" if info["missing"] else "")
        )

    def _run_plot_controls(self):
        """Draw one NxN grid per ``control_plots/*.txt`` file, each in its own
        sub-tab of the "Control plots" dock.

        Cell (0, 0) of every grid is the distance-of-disagreement curve;
        every other cell overlays the abundance-vs-time curves of the
        species listed for it. In the file a line is a grid *column* (the
        file's rows/columns are the transpose of the displayed grid). Only
        the visible sub-tab is rendered now; the rest render when first shown.
        """
        if self._run_dir is None:
            QMessageBox.warning(
                self, "No Folder",
                "No working folder yet — export the inputs, or use “Import output”.",
            )
            return

        files = (
            sorted(f for f in CONTROL_PLOTS_DIR.glob("*.txt")
                   if f.read_text(errors="replace").strip())
            if CONTROL_PLOTS_DIR.is_dir() else []
        )
        if not files:
            QMessageBox.warning(
                self, "No Control-Plots Files",
                f"No non-empty *.txt grid files in:\n{CONTROL_PLOTS_DIR}\n\n"
                f"Add one per element (e.g. C.txt): {CONTROL_GRID_SIZE} lines, "
                f"{CONTROL_GRID_SIZE} '|'-separated cells per line, each cell a "
                "species list. The first cell of the first line is always the "
                "distance of disagreement.",
            )
            return

        self._control_files = {f.stem: f for f in files}
        self._control_rendered.clear()  # redraw on an explicit click
        # _disagreement_cache is kept across clicks; it's invalidated on a run
        # completing, the run dir changing, or observations.txt being edited.

        stems = list(self._control_files)
        current = [self._control_tabs.tabText(i)
                   for i in range(self._control_tabs.count())]
        if current != stems:
            # Tabs get a cheap placeholder; the (expensive) matplotlib grid is
            # built by _render_control_tab when the tab is first shown.
            self._control_tabs.blockSignals(True)
            while self._control_tabs.count():
                self._control_tabs.removeTab(0)
            for stem in stems:
                placeholder = QLabel("  rendering…")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._control_tabs.addTab(
                    self._control_views.get(stem) or placeholder, stem
                )
            self._control_tabs.blockSignals(False)

        self._dock_control.show()
        self._dock_control.raise_()
        self._render_control_tab(self._control_tabs.currentIndex())
        self.statusBar().showMessage(
            f"Control plots: {', '.join(stems)} "
            f"(showing {stems[self._control_tabs.currentIndex()] if stems else ''})"
        )

    def _render_control_tab(self, idx: int):
        """Build + draw one control-plot grid, lazily and once per replot."""
        if idx < 0 or not self._control_files:
            return
        stem = self._control_tabs.tabText(idx)
        if stem in self._control_rendered or stem not in self._control_files:
            return
        view = self._control_views.get(stem)
        if view is None:
            view = ControlPlotsView()
            self._control_views[stem] = view
            self._control_tabs.blockSignals(True)
            self._control_tabs.removeTab(idx)
            self._control_tabs.insertTab(idx, view, stem)
            self._control_tabs.setCurrentIndex(idx)
            self._control_tabs.blockSignals(False)

        if self._disagreement_cache is None:
            times, total, info = self._disagreement_curve()
            self._disagreement_cache = (
                (times, total) if times is not None else None, info
            )
            if self._disagreement_cache[0] is None:
                self._log(f"[Control plots] distance of disagreement "
                          f"unavailable: {info['reason']}")
            else:
                self._log_disagreement_notes(info)
        disagreement, info = self._disagreement_cache
        observed = (info.get("observed", {})
                    if self._observed_display in ("both", "control") else {})

        ab_dir = self._run_dir / "ab"
        n = CONTROL_GRID_SIZE
        raw = parse_control_grid(self._control_files[stem].read_text(errors="replace"))
        # File lines/columns are the transpose of the displayed grid.
        grid = [[raw[c][r] for c in range(n)] for r in range(n)]
        cells: list[list[dict]] = [[{} for _ in range(n)] for _ in range(n)]
        missing: list[str] = []
        for r in range(n):
            for c in range(n):
                if r == 0 and c == 0:
                    continue
                for tok in grid[r][c]:
                    label, components = obs_name_to_species(tok)
                    res = self._read_component_sum(ab_dir, components)
                    if res is None:
                        missing.append(f"[{r},{c}] {tok}")
                    else:
                        xs, ys = res
                        cells[r][c][label] = (xs, ys, observed.get(label))

        view.plot_grid(cells, disagreement,
                       run_name=f"{stem} — {self._run_dir.name}",
                       xlim=self._plot_xlim)
        self._control_rendered.add(stem)
        if missing:
            self._log(f"[Control plots] {stem}: no .ab output for: "
                      + ", ".join(missing))

    def _find_nautilus_bin(self) -> Path | None:
        """Locate the ``nautilus-astrochem`` CLI (installed with
        ``uv tool install nautilus-astrochem@latest``). A GUI launched from
        Finder/the Dock doesn't inherit the shell PATH, so uv's default tool
        bin dirs are checked explicitly after PATH."""
        found = shutil.which(NAUTILUS_CLI)
        if found:
            return Path(found)
        bin_dirs = [os.environ.get("UV_TOOL_BIN_DIR"),
                    os.environ.get("XDG_BIN_HOME"),
                    str(Path.home() / ".local" / "bin")]
        for d in bin_dirs:
            if d and (Path(d) / NAUTILUS_CLI).exists():
                return Path(d) / NAUTILUS_CLI
        QMessageBox.critical(
            self, "Error",
            f"Could not find '{NAUTILUS_CLI}'.\n\n"
            "Install it with:\n"
            "  uv tool install nautilus-astrochem@latest\n\n"
            f"then check that '{NAUTILUS_CLI} -h' works in a terminal."
        )
        return None


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setWindowIcon(app_icon())
    window = NautilusWrapper()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

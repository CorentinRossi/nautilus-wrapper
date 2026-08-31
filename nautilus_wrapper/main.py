"""Nautilus Wrapper GUI - A PyQt6 interface for the Nautilus astrochemistry code."""

import operator as _op
import re
import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QListWidget, QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QGroupBox, QMessageBox, QPlainTextEdit, QTextEdit,
    QTableView, QHeaderView, QProgressBar, QStackedWidget, QInputDialog,
    QTabWidget, QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

try:
    from .parsers import PARSERS, SERIALIZERS, ELEMENT_COLS, parse_species
except ImportError:
    from parsers import PARSERS, SERIALIZERS, ELEMENT_COLS, parse_species


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

STRUCTURED_FILES = set(PARSERS.keys())
HEAVY_FILES = {"gas_reactions.in", "grain_reactions.in"}
REACTION_FILES = {"gas_reactions.in", "grain_reactions.in"}
CHUNK_SIZE = 4000


ASSETS_DIR = Path(__file__).resolve().parent.parent / "inputs"
OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"

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

    for expected in REQUIRED_INPUT_FILES:
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

    def to_dicts(self, only_active: bool = True) -> list[dict]:
        data_headers = self.data_headers()
        result = []
        for i, row in enumerate(self._data):
            if only_active and self._active_col is not None and not self._active[i]:
                continue
            result.append(dict(zip(data_headers, row)))
        return result

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
            self._visible = []
            for row in self._data:
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
    _CMP_RE = re.compile(r"\s*(==|!=|<=|>=|<|>)\s*")
    _OPS = {
        "==": _op.eq, "!=": _op.ne,
        "<": _op.lt, "<=": _op.le, ">": _op.gt, ">=": _op.ge,
    }

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
                return f"'{leaf}' is not a comparison (need ==, !=, <, <=, >, >=)"
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
            return True
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

        self._toggle_bar = QWidget()
        toggle_lay = QVBoxLayout(self._toggle_bar)
        toggle_lay.setContentsMargins(0, 0, 0, 0)
        toggle_lay.setSpacing(4)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        self._filter_input = QLineEdit()
        self._filter_input.setPlaceholderText(
            "Filter, e.g.  Reactant1 == H & A < 1e-10   (and/or/xor · & | ^)"
        )
        self._filter_input.returnPressed.connect(self._apply_filter)
        filter_row.addWidget(self._filter_input, 1)
        self._btn_clear_filter = QPushButton("Clear")
        self._btn_clear_filter.setFixedWidth(50)
        self._btn_clear_filter.clicked.connect(self._clear_filter)
        filter_row.addWidget(self._btn_clear_filter)
        toggle_lay.addLayout(filter_row)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        self._btn_deactivate_all = QPushButton("Deactivate All")
        self._btn_deactivate_all.setFixedWidth(140)
        self._btn_deactivate_all.clicked.connect(lambda: self.model.toggle_all(False))
        self._btn_activate_all = QPushButton("Activate All")
        self._btn_activate_all.setFixedWidth(120)
        self._btn_activate_all.clicked.connect(lambda: self.model.toggle_all(True))
        self._toggle_info = QLabel()
        self._toggle_info.setStyleSheet("color: #666;")
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
            return

        self._active_col = filename in REACTION_FILES
        self._toggle_bar.setVisible(self._active_col)

        heavy = filename in HEAVY_FILES

        if not heavy:
            rows, headers = self._parser(text)
            self.model.set_data(rows, headers, active_col=self._active_col)
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
        if not self._active_col or not self.model._active:
            return
        total = len(self.model._active)
        visible = len(self.model._visible_rows())
        active = sum(1 for i in self.model._visible_rows() if self.model._active[i])
        if visible == total:
            self._toggle_info.setText(f"{active}/{total} reactions active")
        else:
            self._toggle_info.setText(f"{active}/{visible}/{total} reactions active (filtered)")

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
        return self._serializer(self.model.to_dicts())


# ---------------------------------------------------------------------------
# Plot view (embedded matplotlib)
# ---------------------------------------------------------------------------
def read_ab_file(path: Path, column: int = 1) -> tuple[list[float], list[float]]:
    """Read a nautilus ``ab/<species>.ab`` file.

    Column 0 is time [years]; ``column`` (default 1, the first spatial point)
    is the abundance relative to H. Comment/blank lines are skipped.
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
                v = float(parts[column])
            except ValueError:
                continue
            times.append(t)
            values.append(v)
    return times, values


class PlotView(QWidget):
    """A matplotlib canvas (+ navigation toolbar) for abundance plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._figure = Figure(figsize=(6, 4), layout="constrained")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._ax = self._figure.add_subplot(111)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self.show_placeholder()

    def show_placeholder(self, message='Run "Plot CO, H2O abundances" to see a plot here.'):
        self._ax.clear()
        self._ax.text(0.5, 0.5, message, ha="center", va="center",
                      transform=self._ax.transAxes, color="gray", fontsize=10)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self._canvas.draw_idle()

    def plot_series(self, series: dict[str, tuple[list[float], list[float]]],
                    title: str = "", xlabel: str = "Time [years]",
                    ylabel: str = "Abundance"):
        """``series`` maps a label -> (x values, y values); drawn on log-log axes."""
        self._ax.clear()
        for label, (xs, ys) in series.items():
            self._ax.loglog(xs, ys, label=label)
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
        if title:
            self._ax.set_title(title)
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
# Species selector dialog (for choosing what to plot)
# ---------------------------------------------------------------------------
class SpeciesSelectorDialog(QDialog):
    """A sortable / filterable checklist of every gas and grain species."""

    def __init__(self, rows: list[dict], preselected: set[str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Species to plot")
        self.resize(560, 640)

        headers = ["Species", "Type"] + list(ELEMENT_COLS)
        layout = QVBoxLayout(self)

        # --- filter row ---
        filter_row = QHBoxLayout()
        self._filter = QLineEdit()
        self._filter.setPlaceholderText(
            "Filter, e.g.  Type == grain | O > 0   (and/or/xor · & | ^)"
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
        self._model = TableModel()
        self._model._grey_inactive = False
        self._model.set_data(rows, headers, active_col=True, active_default=False)
        if preselected:
            self._model.set_active_values("Species", preselected)

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
        layout.addWidget(buttons)

        self._update_count()

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
        self.setMinimumSize(1200, 700)
        self._file_contents: dict[str, str] = {}
        self._mode = "comprehensive"
        self._run_dir: Path | None = None
        self._export_parent: Path = OUTPUTS_DIR  # where run folders are created
        self._files_exported = False
        self._exported_dirs: set[Path] = set()  # folders this session has written to
        self._plot_selection: set[str] = set()
        self._input_sets = get_input_sets()
        self._current_set_dir: Path | None = None
        self._init_ui()
        self._load_first_set()
        self.file_list.setCurrentRow(0)

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

        for fname in REQUIRED_INPUT_FILES:
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

    def _init_ui(self):
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
        for name in REQUIRED_INPUT_FILES:
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
        self.editor.textChanged.connect(self._mark_unexported)
        self._stack.addWidget(self.editor)

        self.struct_view = StructuredView(self._progress_bar)
        self.struct_view.data_changed.connect(self._mark_unexported)
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

        import_group = QGroupBox("Import Output")
        import_lay = QVBoxLayout(import_group)
        self.btn_import_output = QPushButton("Import output…")
        self.btn_import_output.setToolTip(
            "Pick a previous run folder and plot its abundance results"
        )
        self.btn_import_output.clicked.connect(self._import_output)
        import_lay.addWidget(self.btn_import_output)
        right_lay.addWidget(import_group)

        run_group = QGroupBox("Run")
        run_lay = QVBoxLayout(run_group)
        self.btn_run_code = QPushButton("Run nautilus_code")
        self.btn_run_code.clicked.connect(self._run_nautilus_code)
        run_lay.addWidget(self.btn_run_code)
        self.btn_run_outputs = QPushButton("Run nautilus_outputs")
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
        # Stack the plot on the editor -> two tabs in the same column.
        self.tabifyDockWidget(self._dock_editor, self._dock_plot)
        self._dock_editor.raise_()

        self.resizeDocks(
            [dock_files, self._dock_editor, dock_actions],
            [220, 760, 320],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks([dock_actions, dock_logs], [260, 440], Qt.Orientation.Vertical)

        self.statusBar().showMessage("Ready")

    def _toggle_mode(self):
        if self._mode == "comprehensive":
            self._sync_struct_to_text()
            self._mode = "text"
            self._stack.setCurrentIndex(0)
            self.btn_switch.setText("Switch to Structured Mode")
        else:
            self._mode = "comprehensive"
            self._stack.setCurrentIndex(1)
            self.btn_switch.setText("Switch to Text Mode")
            self._load_struct_for_current()

    def _on_file_selected(self, name: str):
        if name is None:
            return
        self.file_label.setText(name)
        if self._mode == "text":
            self.editor.setPlainText(self._file_contents.get(name, ""))
        else:
            self._load_struct_for_current()

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

    def _sync_struct_to_text(self):
        name = self.file_list.currentItem()
        if name is None:
            return
        name = name.text()
        if name in STRUCTURED_FILES and self.struct_view.isVisible():
            self._file_contents[name] = self.struct_view.to_text()
        self.editor.setPlainText(self._file_contents.get(name, ""))

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

    def _import_output(self):
        """Pick a previous run folder and plot whatever abundance output it has."""
        start = str(self._export_parent)
        chosen = QFileDialog.getExistingDirectory(
            self, "Select a previous run folder", start
        )
        if not chosen:
            return
        folder = Path(chosen)
        ab_dir = folder / "ab"
        if not ab_dir.is_dir() or not any(ab_dir.glob("*.ab")):
            QMessageBox.warning(
                self, "No output",
                f"No abundance output (an 'ab/' folder with .ab files) was found in:\n{folder}",
            )
            return
        self._set_run_dir(folder, note="(imported)")
        self._files_exported = False
        if not self._plot_selection:
            self._open_plot_settings()
        if self._plot_selection:
            self._run_plot_abundances()

    def _sync_current_file(self):
        name = self.file_list.currentItem()
        if name is None:
            return
        name = name.text()
        if self._mode == "text":
            self._file_contents[name] = self.editor.toPlainText()
        elif name in STRUCTURED_FILES:
            self._file_contents[name] = self.struct_view.to_text()

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
            has_output = (dest / "ab").is_dir() or any(dest.glob("*.out"))
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
                        "(ab/, *.out) — those are left in place but will no longer "
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
        return True

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
        nautilus_bin = self._find_nautilus_bin("nautilus_code")
        if nautilus_bin:
            self._run_command([str(nautilus_bin)], "nautilus_code", show_progress=True)

    def _run_nautilus_outputs(self):
        nautilus_bin = self._find_nautilus_bin("nautilus_outputs")
        if nautilus_bin:
            self._run_command([str(nautilus_bin)], "nautilus_outputs", show_progress=False)

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

    def _open_plot_settings(self):
        rows = self._collect_species_rows()
        if not rows:
            QMessageBox.warning(
                self, "No species",
                "gas_species.in / grain_species.in are empty.\n"
                "Load an input set first.",
            )
            return
        dialog = SpeciesSelectorDialog(rows, self._plot_selection, self)
        if dialog.exec():
            self._plot_selection = dialog.selected_species()
            self.statusBar().showMessage(
                f"{len(self._plot_selection)} species selected for plotting"
            )

    def _run_plot_abundances(self):
        """Draw the currently-selected species from the ab/ output in the Plot tab."""
        if self._run_dir is None:
            QMessageBox.warning(
                self, "No Folder",
                "No working folder yet — export the inputs, or use “Import output”.",
            )
            return
        if not self._plot_selection:
            QMessageBox.warning(
                self, "No species selected",
                "No species are checked for plotting.\n\n"
                "Click the ⚙ button next to Plot to choose species.",
            )
            return

        ab_dir = self._run_dir / "ab"
        series: dict[str, tuple[list[float], list[float]]] = {}
        missing: list[str] = []
        for name in sorted(self._plot_selection):
            path = ab_dir / f"{name}.ab"
            if not path.exists():
                missing.append(name)
                continue
            times, values = read_ab_file(path)
            if times:
                series[name] = (times, values)

        if not series:
            QMessageBox.warning(
                self, "No abundance data",
                f"None of the {len(self._plot_selection)} selected species have a "
                f".ab file in:\n{ab_dir}\n\n"
                "Run nautilus_code and then nautilus_outputs first.",
            )
            return

        self._plot_view.plot_series(series, title=f"Abundances - {self._run_dir.name}")
        self._dock_plot.show()
        self._dock_plot.raise_()
        if missing:
            self._log(f"[Plot] no .ab file for: {', '.join(missing)}")
        self.statusBar().showMessage(
            f"Plotted {len(series)} species from {ab_dir}"
        )

    def _find_nautilus_bin(self, name: str) -> Path | None:
        pnautilus = Path(__file__).resolve().parent.parent.parent / "pnautilus"
        candidates = [
            pnautilus / "install" / "bin" / name,
            pnautilus / "build" / "src" / name,
        ]
        for p in candidates:
            if p.exists():
                return p
        QMessageBox.critical(
            self, "Error",
            f"Could not find '{name}'.\nMake sure pnautilus is built and installed."
        )
        return None


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = NautilusWrapper()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

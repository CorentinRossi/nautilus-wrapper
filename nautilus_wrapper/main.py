"""Nautilus Wrapper GUI - A PyQt6 interface for the Nautilus astrochemistry code."""

import sys
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QDockWidget, QListWidget, QPushButton, QLabel, QComboBox, QLineEdit,
    QFileDialog, QGroupBox, QMessageBox, QPlainTextEdit, QTextEdit,
    QTableView, QHeaderView, QProgressBar, QStackedWidget, QInputDialog,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QAbstractTableModel, QModelIndex, QSortFilterProxyModel
from PyQt6.QtGui import QFont, QColor, QTextCharFormat, QSyntaxHighlighter, QAction

try:
    from .parsers import PARSERS, SERIALIZERS
except ImportError:
    from parsers import PARSERS, SERIALIZERS


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
CHUNK_SIZE = 500

ASSETS_DIR = Path(__file__).resolve().parent.parent / "inputs"

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
    """Find groups of files in set_dir that share a base name with underscore variants.

    For each expected input file like 'abundances.in', find all files matching
    'abundances*.in' or similar patterns. Returns groups with >1 match.
    """
    all_files = [f for f in set_dir.iterdir() if f.is_file()]
    groups: dict[str, list[Path]] = {}

    for expected in REQUIRED_INPUT_FILES:
        base = expected.split(".")[0]
        matches = [f for f in all_files if f.name.startswith(base) and f.name.endswith(".in") or f.name.endswith(".dat")]
        # Filter to files that share the base name prefix
        matches = [f for f in all_files if f.name.startswith(base)]
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
        self._visible: list[bool] = []  # parallel to _data, True = row matches filter

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
        self._apply_filter()

    def set_data(self, rows: list[dict], headers: list[str], active_col: bool = False):
        self.beginResetModel()
        if active_col:
            self._headers = [""] + headers
            self._active_col = 0
        else:
            self._headers = headers
            self._active_col = None
        self._data = [[str(row.get(h, "")) for h in headers] for row in rows]
        self._active = [True] * len(rows)
        self._visible = [True] * len(rows)
        self.endResetModel()

    def clear(self):
        self.beginResetModel()
        self._data = []
        self._headers = []
        self._active = []
        self._visible = []
        self.endResetModel()

    def _visible_rows(self) -> list[int]:
        if not self._visible or all(self._visible):
            return list(range(len(self._data)))
        return [i for i, v in enumerate(self._visible) if v]

    def rowCount(self, parent=QModelIndex()):
        return len(self._visible_rows())

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def _map_row(self, row: int) -> int:
        visible = self._visible_rows()
        return visible[row] if row < len(visible) else row

    def data(self, index: QModelIndex, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._map_row(index.row())
        col = index.column()

        if self._active_col is not None and col == self._active_col:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if self._active[row] else Qt.CheckState.Unchecked
            return None

        data_col = col - (1 if self._active_col is not None else 0)

        if not self._active[row]:
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

    def sort_by_column(self, col: int, ascending: bool = True):
        data_col = col - (1 if self._active_col is not None else 0)
        if data_col < 0:
            return
        self.beginResetModel()
        paired = list(zip(self._data, self._active, self._visible))
        paired.sort(key=lambda x: x[0][data_col], reverse=not ascending)
        self._data = [p[0] for p in paired]
        self._active = [p[1] for p in paired]
        self._visible = [p[2] for p in paired]
        self.endResetModel()

    def set_filter(self, expr: str):
        self._filter_expr = expr.strip()
        self._apply_filter()

    def _apply_filter(self):
        if not self._filter_expr or not self._data:
            self._visible = [True] * len(self._data)
            self.layoutChanged.emit()
            return

        hdrs = self.data_headers()
        col_map = {h: i for i, h in enumerate(hdrs)}

        self._visible = []
        for row in self._data:
            try:
                self._visible.append(self._eval_filter(self._filter_expr, col_map, row))
            except Exception:
                self._visible.append(True)
        self.layoutChanged.emit()

    def _eval_filter(self, expr: str, col_map: dict, row: list[str]) -> bool:
        import re, operator as op
        ops = {
            "==": op.eq, "!=": op.ne,
            "<": op.lt, "<=": op.le, ">": op.gt, ">=": op.ge,
        }
        parts = re.split(r"\s*(==|!=|<=|>=|<|>)\s*", expr, maxsplit=1)
        if len(parts) != 3:
            return True
        col_name, cmp, val = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if col_name not in col_map:
            return True
        cell = row[col_map[col_name]]
        try:
            cell_num = float(cell)
            val_num = float(val.strip('"').strip("'"))
            return ops[cmp](cell_num, val_num)
        except ValueError:
            cell_str = cell.strip('"').strip("'")
            val_str = val.strip('"').strip("'")
            return ops[cmp](cell_str, val_str)


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
        self._filter_input.setPlaceholderText("Filter: e.g. Reactant1 == \"H\" or Rate < 1e-10")
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
        self.table.verticalHeader().setVisible(False)
        self.table.setSortingEnabled(False)
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
            self.table.resizeColumnsToContents()
            self._resize_checkbox_col()
        if self._active_col:
            self._update_info()

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
        self.model.set_filter(expr)
        self._update_info()

    def _clear_filter(self):
        self._filter_input.clear()
        self.model.set_filter("")
        self._update_info()

    def _fit_columns(self):
        self.table.resizeColumnsToContents()
        h = self.table.horizontalHeader()
        for j in range(self.model.columnCount()):
            h.setSectionResizeMode(j, QHeaderView.ResizeMode.ResizeToContents)

    def _resize_checkbox_col(self):
        if self._active_col is not None:
            h = self.table.horizontalHeader()
            h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            h.resizeSection(0, h.sectionSizeHint(0))

    def to_text(self) -> str:
        if not self._serializer:
            return ""
        return self._serializer(self.model.to_dicts())


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
        self._files_exported = False
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
            chosen, ok = QInputDialog.getItem(
                self, "Ambiguous Files",
                f"Multiple files found for '{expected}':",
                names, 0, False,
            )
            if ok and chosen:
                chosen_path = set_dir / chosen
                for fname in REQUIRED_INPUT_FILES:
                    if fname == expected or chosen.startswith(fname.split(".")[0]):
                        self._file_overrides[fname] = chosen_path

    def _on_set_changed(self, name: str):
        if name and name in self._input_sets:
            self._load_set(name)
            self.file_list.setCurrentRow(0)

    def _init_ui(self):
        # --- Left dock: file list ---
        dock_files = QDockWidget("Input Files", self)
        dock_files.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
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
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock_files)

        # --- Center: editor / structured view ---
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
        self.setCentralWidget(center)

        # --- Right dock: actions ---
        dock_actions = QDockWidget("Actions", self)
        dock_actions.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        folder_group = QGroupBox("Working Folder")
        folder_lay = QVBoxLayout(folder_group)
        self._folder_label = QLabel("No folder selected")
        self._folder_label.setWordWrap(True)
        self._folder_label.setMaximumHeight(40)
        self._folder_label.setStyleSheet("color: #666;")
        folder_lay.addWidget(self._folder_label)
        self.btn_select_folder = QPushButton("Select Folder...")
        self.btn_select_folder.clicked.connect(self._select_folder)
        folder_lay.addWidget(self.btn_select_folder)
        right_lay.addWidget(folder_group)

        export_group = QGroupBox("Export")
        export_lay = QVBoxLayout(export_group)
        self.btn_export = QPushButton("Export to Folder")
        self.btn_export.clicked.connect(self._export_files)
        export_lay.addWidget(self.btn_export)
        right_lay.addWidget(export_group)

        run_group = QGroupBox("Run")
        run_lay = QVBoxLayout(run_group)
        self.btn_run_code = QPushButton("Run nautilus_code")
        self.btn_run_code.clicked.connect(self._run_nautilus_code)
        run_lay.addWidget(self.btn_run_code)
        self.btn_run_outputs = QPushButton("Run nautilus_outputs")
        self.btn_run_outputs.clicked.connect(self._run_nautilus_outputs)
        run_lay.addWidget(self.btn_run_outputs)
        self.btn_run_plot = QPushButton("Plot CO, H2O abundances")
        self.btn_run_plot.clicked.connect(self._run_plot_abundances)
        run_lay.addWidget(self.btn_run_plot)
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
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock_actions)

        # --- Bottom dock: logs ---
        dock_logs = QDockWidget("Logs", self)
        dock_logs.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        self._log_view = QTextEdit()
        self._log_view.setReadOnly(True)
        self._log_view.setFont(QFont("Monaco", 10))
        dock_logs.setWidget(self._log_view)
        self.splitDockWidget(dock_actions, dock_logs, Qt.Orientation.Vertical)

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

    def _select_folder(self):
        default_dir = self._next_run_dir()
        from PyQt6.QtWidgets import QInputDialog
        text, ok = QInputDialog.getText(
            self, "Working Folder",
            "Folder name (inside outputs/):",
            text=default_dir.name,
        )
        if not ok or not text.strip():
            return
        self._run_dir = default_dir.parent / text.strip()
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._folder_label.setText(str(self._run_dir))
        self._folder_label.setStyleSheet("")
        self.statusBar().showMessage(f"Working folder: {self._run_dir}")

    def _next_run_dir(self) -> Path:
        """Generate outputs/yyyymmdd_runN path, auto-incrementing N."""
        outputs = Path(__file__).resolve().parent.parent / "outputs"
        outputs.mkdir(exist_ok=True)
        today = __import__("datetime").date.today().strftime("%Y%m%d")
        n = 1
        while True:
            candidate = outputs / f"{today}_run{n}"
            if not candidate.exists():
                return candidate
            n += 1

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

        if self._run_dir is None:
            default_dir = self._next_run_dir()
            default_dir.mkdir(parents=True, exist_ok=True)
            self._run_dir = default_dir
            self._folder_label.setText(str(self._run_dir))
            self._folder_label.setStyleSheet("")

        dest = self._run_dir
        for fname, content in self._file_contents.items():
            (dest / fname).write_text(content)

        self._files_exported = True
        self._folder_label.setText(str(self._run_dir))
        self.statusBar().showMessage(f"Exported {len(self._file_contents)} files to {dest}")
        QMessageBox.information(
            self, "Export Complete",
            f"Successfully exported {len(self._file_contents)} files to:\n{dest}"
        )

    def _check_ready_to_run(self) -> bool:
        if self._run_dir is None:
            QMessageBox.warning(self, "No Folder", "Select a working folder first.")
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
            QMessageBox.warning(self, label, f"{label} timed out after 5 minutes.")

    def _run_nautilus_code(self):
        nautilus_bin = self._find_nautilus_bin("nautilus_code")
        if nautilus_bin:
            self._run_command([str(nautilus_bin)], "nautilus_code", show_progress=True)

    def _run_nautilus_outputs(self):
        nautilus_bin = self._find_nautilus_bin("nautilus_outputs")
        if nautilus_bin:
            self._run_command([str(nautilus_bin)], "nautilus_outputs", show_progress=False)

    def _run_plot_abundances(self):
        plot_script = Path(__file__).resolve().parent.parent.parent / "pnautilus" / "nautilus_scripts" / "plot_abundances.py"
        if not plot_script.exists():
            QMessageBox.critical(self, "Error", f"Could not find plot_abundances.py at:\n{plot_script}")
            return
        self._run_command(
            [sys.executable, str(plot_script), "species=CO,H2O"],
            "Plot abundances (CO, H2O)",
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

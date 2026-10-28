"""Base Data menu for the pyRadPlan main window.

Lists and manages the base data files (machines, RBE tables, stopping power tables)
shipped with pyRadPlan, added by the user or in shared folders (see
:mod:`pyRadPlan.core._base_data`). Phantoms are base data too; they are listed in the
File menu instead.
Added files are found by name like the shipped ones, e.g. in the plan editor's machine
and biological model selections.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMenu,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.core import (
    BaseDataFile,
    bundled_base_data_dir,
    find_base_data,
    get_base_data_kind,
    get_data_dir,
    list_base_data,
    remove_base_data_file,
    shared_base_data_dirs,
    user_base_data_dir,
)

_SOURCE_LABELS = {"bundled": "pyRadPlan", "user": "User", "shared": "Shared"}


def _describe_machine(entry: BaseDataFile) -> tuple[list[str], str]:
    stem = os.path.splitext(entry.name)[0]
    mode, _, name = stem.partition("_")
    return [mode, name, _SOURCE_LABELS[entry.source], entry.name], ""


def _describe_rbe_table(entry: BaseDataFile) -> tuple[list[str], str]:
    # Deferred: the biology package is only needed once the tables are listed.
    from pyRadPlan.biology.models import TabulatedAlphaBetaModel  # noqa: PLC0415

    try:
        table = TabulatedAlphaBetaModel.quantity_table_kernel_set(entry.name)
    except Exception as exc:  # noqa: BLE001 - an unreadable table is listed as such
        return [entry.name, "unreadable", "", _SOURCE_LABELS[entry.source], ""], str(exc)
    tissues = ", ".join(
        f"α_x={c.tissue.alpha_x:g}, β_x={c.tissue.beta_x:g}" for c in table.tissue_classes
    )
    cells = [
        entry.name,
        table.provenance_label,
        tissues,
        _SOURCE_LABELS[entry.source],
        table.reference,
    ]
    return cells, "\n".join(table.problems)


def _describe_stopping_power_table(entry: BaseDataFile) -> tuple[list[str], str]:
    return [entry.name, _SOURCE_LABELS[entry.source]], ""


@dataclass(frozen=True)
class _KindView:
    """How the dialog shows and adds one kind of base data."""

    kind: str
    title: str
    columns: tuple[str, ...]
    describe: Callable[[BaseDataFile], tuple[list[str], str]]


_VIEWS = (
    _KindView(
        "machines", "Machines", ("Radiation mode", "Name", "Source", "File"), _describe_machine
    ),
    _KindView(
        "rbe_tables",
        "RBE Tables",
        ("File", "Model", "Tissue classes", "Source", "Reference"),
        _describe_rbe_table,
    ),
    _KindView(
        "stopping_power_tables",
        "Stopping Power Tables",
        ("File", "Source"),
        _describe_stopping_power_table,
    ),
)


def open_folder_in_browser(folder: Path) -> None:
    """Show ``folder`` in the system's file browser, creating it if needed."""
    folder.mkdir(parents=True, exist_ok=True)
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def add_base_data_interactively(
    parent: Optional[QWidget], kind: str, filename: Optional[str] = None
) -> Optional[Path]:
    """Add a base data file of ``kind``, asking for it unless ``filename`` is given.

    The file is checked before it is added. Name clashes are resolved interactively: a
    file the user added before may be replaced, otherwise another name is asked for.

    Returns
    -------
    Path or None
        The added file, ``None`` if cancelled or invalid.
    """
    label = get_base_data_kind(kind).label
    title = f"Add {label[0].upper()}{label[1:]}"
    if filename is None:
        filename, _ = QFileDialog.getOpenFileName(
            parent, title, "", f"{label[0].upper()}{label[1:]}s (*.mat);;All files (*)"
        )
        if not filename:
            return None

    QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        add, name, file_name_of = _adder(kind, filename)
    except Exception as exc:  # noqa: BLE001 - any read/validation failure is reported
        QGuiApplication.restoreOverrideCursor()
        QMessageBox.critical(parent, title, f"Could not read a {label} from\n{filename}\n\n{exc}")
        return None
    QGuiApplication.restoreOverrideCursor()

    overwrite = False
    while True:
        error = None
        QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            return add(name, overwrite)
        except FileExistsError:
            pass
        except (OSError, ValueError) as exc:
            error = exc
        finally:
            QGuiApplication.restoreOverrideCursor()
        if error is not None:
            QMessageBox.critical(parent, title, str(error))
            return None

        existing = find_base_data(kind, file_name_of(name))
        if existing is not None and existing.source == "user":
            answer = QMessageBox.question(
                parent,
                title,
                f"A {label} named '{name}' was already added.\nReplace it?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.No,
            )
            if answer == QMessageBox.Cancel:
                return None
            if answer == QMessageBox.Yes:
                overwrite = True
                continue
            prompt = f"Name to add the {label} under:"
        else:
            prompt = (
                f"A {label} named '{name}' ships with pyRadPlan.\nName to add the {label} under:"
            )
        name, ok = QInputDialog.getText(parent, title, prompt, text=name)
        name = name.strip()
        if not ok or not name:
            return None


def _adder(
    kind: str, filename: str
) -> tuple[Callable[[str, bool], Path], str, Callable[[str], str]]:
    """Return how to add ``filename``, its proposed name and the file name of a name.

    The name is what the user is asked for on a clash: a machine's name, a table's file
    name. A machine file is read here, for its name; tables are checked when added.
    """
    if kind == "machines":
        from pyRadPlan.machines import add_machine_file, load_machine  # noqa: PLC0415

        machine = load_machine(filename)
        mode = machine.radiation_mode

        def add_machine(name: str, overwrite: bool) -> Path:
            return add_machine_file(
                filename, machine_name=name, radiation_mode=mode, overwrite=overwrite
            )

        return add_machine, machine.name, lambda name: f"{mode}_{name}.mat"

    from pyRadPlan.biology.models import (  # noqa: PLC0415
        add_rbe_table,
        add_stopping_power_table,
    )

    if kind == "rbe_tables":
        add_table = add_rbe_table
    elif kind == "stopping_power_tables":
        add_table = add_stopping_power_table
    else:
        raise ValueError(f"Unknown base data kind {kind!r}")

    def file_name_of(name: str) -> str:
        return name if name.lower().endswith(".mat") else f"{name}.mat"

    def add(name: str, overwrite: bool) -> Path:
        return add_table(filename, name=file_name_of(name), overwrite=overwrite)

    return add, Path(filename).name, file_name_of


class _KindTab(QWidget):
    """One tab of the :class:`BaseDataDialog`: the files of one kind and their actions."""

    changed = Signal()

    def __init__(self, view: _KindView, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.view = view
        self._entries: list[BaseDataFile] = []

        layout = QVBoxLayout(self)
        user_dir = user_base_data_dir(view.kind)
        lines = [
            f"Shipped with pyRadPlan: {bundled_base_data_dir(view.kind)}",
            f"Added by you: {user_dir}",
            *(f"Shared: {folder}" for folder in shared_base_data_dirs(view.kind)),
        ]
        folders = QLabel("<br>".join(lines))
        folders.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        folders.setWordWrap(True)
        layout.addWidget(folders)

        self.table = QTableWidget(0, len(view.columns))
        self.table.setHorizontalHeaderLabels(list(view.columns))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._update_buttons)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.btn_add = QPushButton("Add…")
        self.btn_add.clicked.connect(lambda checked=False: self.add())
        buttons.addWidget(self.btn_add)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setToolTip("Only files you added can be removed")
        self.btn_remove.clicked.connect(self.remove_selected)
        buttons.addWidget(self.btn_remove)
        buttons.addStretch()
        self.btn_open = QPushButton("Open Folder")
        self.btn_open.setToolTip(str(user_dir))
        self.btn_open.clicked.connect(
            lambda: open_folder_in_browser(user_base_data_dir(view.kind))
        )
        buttons.addWidget(self.btn_open)
        layout.addLayout(buttons)

        self.populate()

    def populate(self) -> None:
        """List the files of the kind."""
        self._entries = list_base_data(self.view.kind)
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            cells, problems = self.view.describe(entry)
            tooltip = "\n".join(filter(None, [str(entry.path), problems]))
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                if problems:
                    item.setForeground(QColor("darkorange"))
                self.table.setItem(row, column, item)
        self.table.resizeColumnsToContents()
        self._update_buttons()

    def selected_entry(self) -> Optional[BaseDataFile]:
        rows = self.table.selectionModel().selectedRows()
        return self._entries[rows[0].row()] if rows else None

    def _update_buttons(self) -> None:
        entry = self.selected_entry()
        self.btn_remove.setEnabled(entry is not None and entry.source == "user")

    def add(self, filename: Optional[str] = None) -> Optional[Path]:
        target = add_base_data_interactively(self, self.view.kind, filename)
        if target is not None:
            self.changed.emit()
        return target

    def remove_selected(self) -> bool:
        entry = self.selected_entry()
        if entry is None or entry.source != "user":
            return False
        answer = QMessageBox.question(
            self,
            "Remove Base Data",
            f"Remove the {entry.kind.label} '{entry.name}'?\n"
            "Its file is deleted from your base data folder.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return False
        try:
            remove_base_data_file(entry.kind, entry.name)
        except OSError as exc:
            QMessageBox.critical(self, "Remove Base Data", str(exc))
            return False
        self.changed.emit()
        return True


class BaseDataDialog(QDialog):
    """Overview of the available base data, one tab per kind.

    Lists the files shipped with pyRadPlan and those added by the user, and offers
    adding files, removing added ones and opening the user's folder of each kind.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    #: Emitted after a file was added or removed.
    base_data_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Base Data")
        layout = QVBoxLayout(self)

        self.tabs = QTabWidget()
        self.kind_tabs: dict[str, _KindTab] = {}
        for view in _VIEWS:
            tab = _KindTab(view)
            tab.changed.connect(self._on_changed)
            self.kind_tabs[view.kind] = tab
            self.tabs.addTab(tab, view.title)
        layout.addWidget(self.tabs, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.resize(760, 420)

    def _on_changed(self) -> None:
        for tab in self.kind_tabs.values():
            tab.populate()
        self.base_data_changed.emit()


class BaseDataMenu(QMenu):
    """The main window's *Base Data* menu.

    Opens the :class:`BaseDataDialog` overview, adds machine files and RBE tables directly
    and opens the user's data folder.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    #: Emitted after base data was added or removed.
    base_data_changed = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__("&Base Data", parent)
        self.setToolTipsVisible(True)

        self._act_show = self.addAction("Show Available Base Data…")
        self._act_show.setToolTip(
            "Machines, RBE tables and stopping power tables shipped with pyRadPlan "
            "and added by you"
        )
        self._act_show.triggered.connect(lambda checked=False: self.show_base_data())

        self.addSeparator()
        self._act_add_machine = self.addAction("Add Machine File…")
        self._act_add_machine.setToolTip(
            "Make a machine file (matRad *.mat) available by name, also in later sessions"
        )
        self._act_add_machine.triggered.connect(lambda checked=False: self.add("machines"))
        self._act_add_rbe = self.addAction("Add RBE Table…")
        self._act_add_rbe.setToolTip(
            "Make an α/β RBE table available to the 'α/β tables + fluence spectra' model"
        )
        self._act_add_rbe.triggered.connect(lambda checked=False: self.add("rbe_tables"))

        self.addSeparator()
        self._act_open_folder = self.addAction("Open Data Folder")
        self._act_open_folder.triggered.connect(
            lambda checked=False: open_folder_in_browser(get_data_dir())
        )
        self.aboutToShow.connect(lambda: self._act_open_folder.setToolTip(str(get_data_dir())))

    def add(self, kind: str, filename: Optional[str] = None) -> Optional[Path]:
        """Add a base data file of ``kind``, see :func:`add_base_data_interactively`."""
        target = add_base_data_interactively(self.parentWidget(), kind, filename)
        if target is not None:
            self.base_data_changed.emit()
            label = get_base_data_kind(kind).label
            QMessageBox.information(
                self.parentWidget(), "Base Data", f"Added the {label} '{target.name}'."
            )
        return target

    def show_base_data(self) -> BaseDataDialog:
        """Open the base data overview (modal)."""
        dialog = BaseDataDialog(self.parentWidget())
        dialog.base_data_changed.connect(self.base_data_changed)
        dialog.exec()
        return dialog

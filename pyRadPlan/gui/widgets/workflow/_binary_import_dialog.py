"""Dialog to import a foreign folder: a CT image plus per-file binary masks.

Presents an editable review table (file, name, type) over the masks found in the
folder, mirroring matRad's binary import widget. The heavy loading is done by the
caller via :func:`pyRadPlan.io.load_binary_patient`; this dialog only gathers the
CT path and the per-mask selections.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

#: VOI types offered per structure; IGNORED drops the mask on import.
_VOI_TYPES = ("TARGET", "OAR", "EXTERNAL", "IGNORED")

#: Qt item-data role used to stash the full mask path on the File cell.
_PATH_ROLE = Qt.UserRole


def _scan_folder(directory: str) -> tuple[Optional[str], list[str]]:
    """Return (ct_candidate, mask_files) for a folder.

    The first top-level image is proposed as the CT; the remaining top-level
    images plus every image in immediate subfolders are proposed as masks.
    """
    from pyRadPlan.io import list_image_files  # noqa: PLC0415

    top = list_image_files(directory)
    ct_candidate = top[0] if top else None
    masks = list(top[1:])
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isdir(full):
            masks.extend(list_image_files(full))
    return ct_candidate, masks


class BinaryImportDialog(QDialog):
    """Choose a CT image and a set of binary masks (with names/types) to import."""

    def __init__(self, directory: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Binary Patient")
        self.resize(640, 480)
        self._directory = directory

        ct_candidate, mask_files = _scan_folder(directory)

        root = QVBoxLayout(self)

        # --- CT file row ---------------------------------------------------
        root.addWidget(QLabel("Patient CT file [HU]:"))
        ct_row = QHBoxLayout()
        self._ct_edit = QLineEdit(ct_candidate or "")
        ct_browse = QPushButton("Browse…")
        ct_browse.clicked.connect(self._browse_ct)
        ct_row.addWidget(self._ct_edit)
        ct_row.addWidget(ct_browse)
        root.addLayout(ct_row)

        # --- Structure table ----------------------------------------------
        root.addWidget(QLabel("Structures (binary masks):"))
        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["File", "Name", "Type"])
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        root.addWidget(self._table)

        for path in mask_files:
            self._add_row(path)

        # --- Structure buttons --------------------------------------------
        btns = QHBoxLayout()
        add_files = QPushButton("Add File(s)…")
        add_files.clicked.connect(self._add_files)
        add_folder = QPushButton("Add Folder…")
        add_folder.clicked.connect(self._add_folder)
        remove = QPushButton("Remove Selected")
        remove.clicked.connect(self._remove_selected)
        btns.addWidget(add_files)
        btns.addWidget(add_folder)
        btns.addWidget(remove)
        btns.addStretch()
        root.addLayout(btns)

        # --- OK / Cancel ---------------------------------------------------
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    # ------------------------------------------------------------------
    # Row / selection management
    # ------------------------------------------------------------------

    def _add_row(self, path: str) -> None:
        from pyRadPlan.io.sitk_based._binary_import import _file_stem  # noqa: PLC0415
        from pyRadPlan.io._helpers import determine_structure_type  # noqa: PLC0415

        # Skip a file already listed.
        if any(
            self._table.item(r, 0).data(_PATH_ROLE) == path for r in range(self._table.rowCount())
        ):
            return

        stem = _file_stem(path)
        row = self._table.rowCount()
        self._table.insertRow(row)

        file_item = QTableWidgetItem(os.path.basename(path))
        file_item.setToolTip(path)
        file_item.setData(_PATH_ROLE, path)
        file_item.setFlags(file_item.flags() & ~Qt.ItemIsEditable)
        self._table.setItem(row, 0, file_item)

        self._table.setItem(row, 1, QTableWidgetItem(stem))

        combo = QComboBox()
        combo.addItems(_VOI_TYPES)
        combo.setCurrentText(determine_structure_type(stem))
        self._table.setCellWidget(row, 2, combo)

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Add mask files", self._directory, "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd)"
        )
        for path in files:
            self._add_row(path)

    def _add_folder(self) -> None:
        from pyRadPlan.io import list_image_files  # noqa: PLC0415

        folder = QFileDialog.getExistingDirectory(self, "Add mask folder", self._directory)
        if folder:
            for path in list_image_files(folder):
                self._add_row(path)

    def _remove_selected(self) -> None:
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()}, reverse=True)
        for row in rows:
            self._table.removeRow(row)

    def _browse_ct(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose CT file", self._directory, "Images (*.nii *.nii.gz *.nrrd *.mha *.mhd)"
        )
        if path:
            self._ct_edit.setText(path)

    def _accept(self) -> None:
        if not self._ct_edit.text().strip():
            # Keep the dialog open; a CT is required.
            self._ct_edit.setFocus()
            return
        self.accept()

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def ct_file(self) -> str:
        """Return the chosen CT file path."""
        return self._ct_edit.text().strip()

    def selections(self) -> list[dict]:
        """Return one ``{path, name, voi_type}`` per table row (IGNORED included)."""
        out = []
        for row in range(self._table.rowCount()):
            path = self._table.item(row, 0).data(_PATH_ROLE)
            name = self._table.item(row, 1).text().strip()
            voi_type = self._table.cellWidget(row, 2).currentText()
            out.append({"path": path, "name": name or None, "voi_type": voi_type})
        return out

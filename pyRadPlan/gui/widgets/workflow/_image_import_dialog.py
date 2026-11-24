"""Dialog asking what a single bare image file represents (CT, structures, dose).

A plain image file (``.nii``/``.nrrd``/``.mha``/...) carries no semantics, so the
user decides how to import it:

- **CT — new patient**: clear the whole workspace and load the image as the CT.
- **CT — replace only**: swap the CT, keeping other data. If the grid differs
  from the current CT, dependent data (structures, dose influence, results) no
  longer fits and will be cleared — the dialog warns about this beforehand.
- **Structure(s)**: add the mask (or every label of a label map) to the current
  StructureSet; name clashes are resolved by numeric suffix.
- **Dose**: add the image to the result collection under a user-supplied name.

The dialog only shows what :func:`pyRadPlan.app._io.inspect_image` finds (labels,
geometry, a guessed kind); the import is the session's ``import_image`` operation.
"""

from __future__ import annotations

import os
from typing import Optional

import SimpleITK as sitk
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.app._io import inspect_image


class ImageImportDialog(QDialog):
    """Choose how to import a single bare image file into the workspace."""

    def __init__(
        self,
        path: str,
        *,
        has_ct: bool,
        ct_image: Optional[sitk.Image] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Image")

        info = inspect_image(path, ct_image)
        self._grid_matches = info.grid_matches
        self._default_name = info.default_name

        root = QVBoxLayout(self)
        root.addWidget(QLabel(f"Import <b>{os.path.basename(path)}</b> as:"))

        self._rb_ct_new = QRadioButton("CT — new patient (clears all loaded data)")
        self._rb_ct_replace = QRadioButton("CT — replace current CT only")
        self._rb_structures = QRadioButton(
            f"Structure(s) — add to structure set ({info.num_labels} label(s) found)"
        )
        self._rb_dose = QRadioButton("Dose — add to result collection")

        root.addWidget(self._rb_ct_new)
        root.addWidget(self._rb_ct_replace)

        self._replace_warning = QLabel(
            "⚠ The image grid differs from the current CT; structures, dose "
            "influence and results will be cleared."
        )
        self._replace_warning.setWordWrap(True)
        self._replace_warning.setStyleSheet("color: #c0392b; margin-left: 20px;")
        self._replace_warning.setVisible(False)
        root.addWidget(self._replace_warning)

        root.addWidget(self._rb_structures)
        root.addWidget(self._rb_dose)

        self._dose_name = QLineEdit(info.default_name)
        self._dose_name.setPlaceholderText("Result name")
        self._dose_name.setVisible(False)
        dose_row = QVBoxLayout()
        dose_row.setContentsMargins(20, 0, 0, 0)
        dose_row.addWidget(self._dose_name)
        root.addLayout(dose_row)

        # Structures and dose need a CT to reference; replace needs a CT to replace.
        self._rb_ct_replace.setEnabled(has_ct)
        self._rb_structures.setEnabled(has_ct)
        self._rb_dose.setEnabled(has_ct)
        self._preselect(info.inferred, has_ct)

        self._rb_ct_replace.toggled.connect(self._update_detail_rows)
        self._rb_dose.toggled.connect(self._update_detail_rows)
        self._update_detail_rows()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _preselect(self, inferred: str, has_ct: bool) -> None:
        """Preselect the mode inferred from the image's pixel type and values.

        Boolean/unsigned images preselect structures, non-negative floats dose,
        anything with negative values a CT (replace when a CT with a matching
        grid is loaded, new patient otherwise). Falls back to "new patient" when
        the inferred option is unavailable. Only a default — the user decides.
        """
        if inferred == "structures" and self._rb_structures.isEnabled():
            self._rb_structures.setChecked(True)
        elif inferred == "dose" and self._rb_dose.isEnabled():
            self._rb_dose.setChecked(True)
        elif inferred == "ct" and has_ct and self._grid_matches:
            self._rb_ct_replace.setChecked(True)
        else:
            self._rb_ct_new.setChecked(True)

    def _update_detail_rows(self) -> None:
        self._replace_warning.setVisible(
            self._rb_ct_replace.isChecked() and not self._grid_matches
        )
        self._dose_name.setVisible(self._rb_dose.isChecked())
        self.adjustSize()

    def selection(self) -> dict:
        """Return the chosen import mode and its parameters.

        Returns
        -------
        dict
            ``mode`` (``"ct_new"``, ``"ct_replace"``, ``"structures"`` or
            ``"dose"``); for ``"ct_replace"`` additionally ``grid_matches``; for
            ``"dose"`` additionally ``name``.
        """
        if self._rb_ct_replace.isChecked():
            return {"mode": "ct_replace", "grid_matches": self._grid_matches}
        if self._rb_structures.isChecked():
            return {"mode": "structures"}
        if self._rb_dose.isChecked():
            name = self._dose_name.text().strip() or self._default_name
            return {"mode": "dose", "name": name}
        return {"mode": "ct_new"}

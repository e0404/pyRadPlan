"""Dialog to import from a DICOM folder: pick a CT series, structure set and dose.

Enumeration (a header read per file) is done by the :class:`DicomImporter`; the
caller normally does it in a worker thread and hands the result in as *catalog*,
so a large folder does not freeze the GUI while it is scanned.  The actual pixel
loading also runs in the caller's worker thread, using the selectors returned by
:meth:`DicomImportDialog.selection`.
"""

from __future__ import annotations

import os
from typing import Optional

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QWidget,
)


class DicomImportDialog(QDialog):
    """Choose which CT series, structure set and dose to import from a DICOM folder.

    Parameters
    ----------
    importer : DicomImporter
        The importer bound to the folder. Only used to enumerate the contents
        when no *catalog* is given.
    parent : QWidget, optional
        The parent widget.
    catalog : dict, optional
        A previously enumerated folder content with the keys ``series``,
        ``structs`` and ``doses`` (see :meth:`DicomImporter.list_ct_series` and
        friends). Pass the result of a background scan here to keep the dialog
        construction instant.
    """

    def __init__(
        self,
        importer,
        parent: Optional[QWidget] = None,
        catalog: Optional[dict] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import DICOM")
        self.resize(520, 200)

        if catalog is None:
            catalog = scan_folder(importer)
        self._series = catalog["series"]
        self._structs = catalog["structs"]
        self._doses = catalog["doses"]

        form = QFormLayout(self)

        # CT series (required).
        self._ct_combo = QComboBox()
        for s in self._series:
            desc = s["description"] or s["series_uid"][:12]
            self._ct_combo.addItem(f"{desc}  ({s['num_slices']} slices)", s["series_uid"])
        form.addRow("CT series:", self._ct_combo)

        # Structure set (optional).
        self._struct_combo = QComboBox()
        self._struct_combo.addItem("None", None)
        for st in self._structs:
            n = len(st["structure_names"])
            self._struct_combo.addItem(
                f"{st['label']}  [{st['modality']}, {n} structures]", st["path"]
            )
        if self._structs:
            self._struct_combo.setCurrentIndex(1)  # default to the first real set
        form.addRow("Structure set:", self._struct_combo)

        # Dose (optional). "Auto" defers to the importer's plan-physical selection.
        self._dose_combo = QComboBox()
        self._dose_combo.addItem("None", _NO_DOSE)
        if self._doses:
            self._dose_combo.addItem("Auto (plan physical dose)", _AUTO_DOSE)
            for d in self._doses:
                label = d["description"] or os.path.basename(d["path"])
                self._dose_combo.addItem(f"{label}  [{d['summation']}]", d["path"])
            self._dose_combo.setCurrentIndex(1)  # default to Auto
        form.addRow("Dose:", self._dose_combo)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def selection(self) -> dict:
        """Return the chosen selectors.

        Returns
        -------
        dict
            ``series_uid`` (``None`` lets the importer choose), ``struct_file``
            (``None`` skips structures), ``load_dose`` (whether to load a dose at
            all) and ``dose_file`` (an explicit path, or ``None`` to let
            :meth:`DicomImporter.load_dose` auto-select the plan physical dose).
        """
        dose_data = self._dose_combo.currentData()
        load_dose = dose_data != _NO_DOSE
        dose_file = None if dose_data in (_NO_DOSE, _AUTO_DOSE) else dose_data

        return {
            "series_uid": self._ct_combo.currentData(),
            "struct_file": self._struct_combo.currentData(),
            "load_dose": load_dose,
            "dose_file": dose_file,
        }


def scan_folder(importer) -> dict:
    """Enumerate a DICOM folder into the *catalog* consumed by the dialog.

    Reads one header per file, so it belongs in a worker thread for anything
    larger than a toy folder; the importer reports its progress while it runs.
    """
    return {
        "series": importer.list_ct_series(),
        "structs": importer.list_structure_sets(),
        "doses": importer.list_doses(),
    }


#: Sentinels distinguishing "no dose" from "auto-select a dose" in the combo data.
_NO_DOSE = "__none__"
_AUTO_DOSE = "__auto__"

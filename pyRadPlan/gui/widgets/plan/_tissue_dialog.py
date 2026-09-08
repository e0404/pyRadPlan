"""Dialog editing the reference photon LQ parameters (alpha_x, beta_x) of all VOIs."""

from __future__ import annotations

from typing import Optional

from pydantic import ValidationError
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QLabel,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.cst import StructureSet


class TissueParametersDialog(QDialog):
    """
    Table of the VOIs' ``alpha_x`` / ``beta_x`` with editable values.

    Accepting the dialog writes the values to the VOIs of the given structure set;
    :attr:`changed` tells whether any value differs from before.
    """

    def __init__(self, cst: StructureSet, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tissue parameters")
        self._cst = cst
        self.changed = False

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Reference photon LQ parameters per structure. The biological model maps "
                "them to the tissue classes of its base data."
            )
        )

        self._table = QTableWidget(len(cst.vois), 4, self)
        self._table.setHorizontalHeaderLabels(["Structure", "Type", "α_x [Gy⁻¹]", "β_x [Gy⁻²]"])
        self._table.verticalHeader().setVisible(False)
        self._alpha_spins: list[QDoubleSpinBox] = []
        self._beta_spins: list[QDoubleSpinBox] = []
        for row, voi in enumerate(cst.vois):
            for col, text in enumerate((voi.name, voi.voi_type)):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, col, item)
            alpha = self._make_spin(voi.alpha_x)
            beta = self._make_spin(voi.beta_x)
            self._alpha_spins.append(alpha)
            self._beta_spins.append(beta)
            self._table.setCellWidget(row, 2, alpha)
            self._table.setCellWidget(row, 3, beta)
        self._table.resizeColumnsToContents()
        layout.addWidget(self._table)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _make_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(4)
        spin.setRange(0.0, 100.0)
        spin.setSingleStep(0.01)
        spin.setValue(float(value))
        return spin

    def set_values(self, row: int, alpha_x: float, beta_x: float) -> None:
        """Set the edited values of one row (mainly for scripting / tests)."""
        self._alpha_spins[row].setValue(alpha_x)
        self._beta_spins[row].setValue(beta_x)

    def accept(self) -> None:
        """Validate and write the values to the VOIs before closing."""
        previous = [(voi.alpha_x, voi.beta_x) for voi in self._cst.vois]
        try:
            for voi, alpha, beta in zip(self._cst.vois, self._alpha_spins, self._beta_spins):
                if voi.alpha_x != alpha.value() or voi.beta_x != beta.value():
                    voi.alpha_x = alpha.value()
                    voi.beta_x = beta.value()
                    self.changed = True
        except (ValidationError, ValueError, TypeError) as exc:
            for voi, (alpha_x, beta_x) in zip(self._cst.vois, previous):
                voi.alpha_x, voi.beta_x = alpha_x, beta_x
            self.changed = False
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        super().accept()

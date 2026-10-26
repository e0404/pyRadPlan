"""Dialog confirming a D_x / V_x point picked in the DVH plot."""

from __future__ import annotations

from typing import Literal, Optional

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

PointKind = Literal["dx", "vx"]


class QIPointDialog(QDialog):
    """Confirm a D_x / V_x reference point before it goes into the QI table.

    Without a fixed *kind* (a double-click in the plot) both metrics are offered,
    prefilled with the clicked volume and dose.  With a *kind* and a *replace*
    metric (a dragged parameter line) only that metric is shown, and the point can
    replace the metric it was dragged from or be added as a new one.

    Parameters
    ----------
    dose : float
        Dose prefilled as the V_x reference dose.
    volume : float
        Volume in % prefilled as the D_x reference volume.
    dose_unit : str, optional
        Unit symbol shown next to the dose.
    kind : {"dx", "vx"} or None, optional
        Metric to offer exclusively.
    replace : str or None, optional
        Metric id the point was dragged from; enables "Replace".
    parent : QWidget or None, optional
        Parent widget.
    """

    def __init__(
        self,
        dose: float,
        volume: float,
        dose_unit: str = "",
        kind: Optional[PointKind] = None,
        replace: Optional[str] = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Change {replace}" if replace else "Add quality indicator")
        self._replace_chosen = False

        layout = QVBoxLayout(self)
        grid = QGridLayout()
        layout.addLayout(grid)

        self.dx_radio = QRadioButton("D_x: dose covering")
        self.volume_spin = QDoubleSpinBox()
        self.volume_spin.setRange(0.0, 100.0)
        self.volume_spin.setDecimals(1)
        self.volume_spin.setValue(volume)
        dx_row = (self.dx_radio, self.volume_spin, QLabel("% of the volume"))

        self.vx_radio = QRadioButton("V_x: volume receiving at least")
        self.dose_spin = QDoubleSpinBox()
        self.dose_spin.setRange(0.0, 1e6)
        self.dose_spin.setDecimals(2)
        self.dose_spin.setValue(dose)
        vx_row = (self.vx_radio, self.dose_spin, QLabel(dose_unit))

        for row, widgets in enumerate((dx_row, vx_row)):
            for col, widget in enumerate(widgets):
                grid.addWidget(widget, row, col)
        self.vx_radio.setChecked(kind == "vx")
        self.dx_radio.setChecked(kind != "vx")
        if kind is not None:
            for widget in vx_row if kind == "dx" else dx_row:
                widget.hide()

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        if replace:
            self.replace_button = buttons.addButton(
                f"Replace {replace}", QDialogButtonBox.ButtonRole.AcceptRole
            )
            add_text = "Add as new"
        else:
            self.replace_button = None
            add_text = "Add"
        self.add_button: QPushButton = buttons.addButton(
            add_text, QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.add_button.setDefault(True)
        # The box emits clicked before accepted, so the choice is known on accept.
        buttons.clicked.connect(self._on_button_clicked)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_button_clicked(self, button) -> None:
        self._replace_chosen = button is self.replace_button

    def chosen(self) -> tuple[PointKind, float, bool]:
        """Return the metric kind, its parameter and whether to replace the old metric."""
        if self.dx_radio.isChecked():
            return "dx", self.volume_spin.value(), self._replace_chosen
        return "vx", self.dose_spin.value(), self._replace_chosen

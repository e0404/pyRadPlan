"""Visualization settings widget."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QDialog,
    QLineEdit,
    QDialogButtonBox,
)


class VisualizationWidget(QWidget):
    """Widget for visualization settings (overlays, isolines, etc.)."""

    # Signals
    overlay_toggled = Signal(str, bool)  # overlay name, checked
    isolines_toggled = Signal(bool)
    isocenter_toggled = Signal(bool)
    quantity_changed = Signal(str)
    isolines_set = Signal(list)  # list of float levels
    recenter_requested = Signal()
    show_analysis_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        vis_group = QGroupBox("Visualization")
        vis_layout = QVBoxLayout()
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # Checkboxes
        self.use_ct_checkbox = QCheckBox("CT")
        self.use_ct_checkbox.setChecked(True)
        self.use_ct_checkbox.stateChanged.connect(
            lambda s: self.overlay_toggled.emit("CT", bool(s))
        )
        vis_layout.addWidget(self.use_ct_checkbox)

        self.use_cst_checkbox = QCheckBox("CST")
        self.use_cst_checkbox.setChecked(True)
        self.use_cst_checkbox.stateChanged.connect(
            lambda s: self.overlay_toggled.emit("CST", bool(s))
        )
        vis_layout.addWidget(self.use_cst_checkbox)

        self.use_quantity_checkbox = QCheckBox("Quantity")
        self.use_quantity_checkbox.setChecked(True)
        self.use_quantity_checkbox.stateChanged.connect(
            lambda s: self.overlay_toggled.emit("quantity", bool(s))
        )
        vis_layout.addWidget(self.use_quantity_checkbox)

        self.isolines_checkbox = QCheckBox("Isolines")
        self.isolines_checkbox.stateChanged.connect(lambda s: self.isolines_toggled.emit(bool(s)))
        vis_layout.addWidget(self.isolines_checkbox)

        self.isocenter_checkbox = QCheckBox("Isocenter")
        self.isocenter_checkbox.stateChanged.connect(
            lambda s: self.isocenter_toggled.emit(bool(s))
        )
        vis_layout.addWidget(self.isocenter_checkbox)

        # Quantity Selector
        vis_layout.addWidget(QLabel("Overlay Quantity:"))
        self.quantity_selector = QComboBox()
        self.quantity_selector.currentTextChanged.connect(self.quantity_changed)
        vis_layout.addWidget(self.quantity_selector)

        # Buttons
        self.set_isolines_btn = QPushButton("Set Isolines")
        self.set_isolines_btn.setToolTip("Set custom isoline levels (e.g. 30 50 95)")
        self.set_isolines_btn.clicked.connect(self._on_set_isolines)
        vis_layout.addWidget(self.set_isolines_btn)

        self.recenter_btn = QPushButton("Recenter")
        self.recenter_btn.setToolTip("Recenter view to isocenter")
        self.recenter_btn.clicked.connect(self.recenter_requested)
        vis_layout.addWidget(self.recenter_btn)

        self.dvh_btn = QPushButton("Show DVH / QI")
        self.dvh_btn.setToolTip("Show DVH and QI analysis window")
        self.dvh_btn.setEnabled(True)
        self.dvh_btn.clicked.connect(self.show_analysis_requested)
        vis_layout.addWidget(self.dvh_btn)

        layout.addStretch(1)

    def update_quantity_selector(self, quantities: list[str], active: str | None = None) -> None:
        """Update the quantity selector items."""
        self.quantity_selector.blockSignals(True)
        self.quantity_selector.clear()
        self.quantity_selector.addItems(quantities)
        if active and active in quantities:
            self.quantity_selector.setCurrentText(active)
        elif quantities:
            self.quantity_selector.setCurrentIndex(0)
        self.quantity_selector.blockSignals(False)

    def _on_set_isolines(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Isolines")
        layout = QVBoxLayout(dialog)

        layout.addWidget(QLabel("Enter isoline levels (Gy) separated by space:"))

        line_edit = QLineEdit()
        layout.addWidget(line_edit)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        reset_btn = btn_box.addButton("Reset", QDialogButtonBox.ResetRole)

        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)

        # Handle Reset
        def on_reset():
            self.isolines_set.emit([])  # Emit empty list to reset to defaults
            dialog.reject()

        reset_btn.clicked.connect(on_reset)

        layout.addWidget(btn_box)

        if dialog.exec() == QDialog.Accepted:
            text = line_edit.text()
            if text:
                try:
                    levels = [float(x) for x in text.split()]
                    self.isolines_set.emit(levels)
                except ValueError:
                    pass

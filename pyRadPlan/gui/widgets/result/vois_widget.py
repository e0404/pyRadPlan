"""VOIs selection widget."""

from __future__ import annotations

import html
import math
from typing import Callable, Iterable, Optional

import SimpleITK as sitk
from pydantic import ValidationError

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QStyleOptionButton,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.biology.models import TissueComparison, apply_provenance_fixes
from pyRadPlan.cst import VOI, create_voi
from pyRadPlan.biology.tissues import TissueProperties

from .._tissue_params import (
    APPLY_ANYWAY,
    DERIVE_REFERENCE_TOOLTIP,
    MATCH_BASE_DATA_TEXT,
    MODEL_GROUP_TITLE,
    REFERENCE_GROUP_TITLE,
    REFERENCE_RADIATION_FIELD,
    REFERENCE_RADIATION_LABEL,
    MATCH_BASE_DATA_TOOLTIP,
    RELABEL_TOOLTIP,
    TISSUE_MODELS,
    adopt_preset,
    apply_base_data_class,
    apply_derived_reference,
    ask_tissue_issues,
    available_tissue_classes,
    bio_model_quietly,
    build_tissue_properties,
    cell_line_presets,
    combo_value,
    compare_tissues_quietly,
    comparison_row_note,
    current_pick,
    derives_reference_params,
    fill_tissue_picker,
    machine_summary,
    make_derive_button,
    make_param_combo,
    make_param_spin,
    make_reference_radiation_combo,
    model_label_name,
    model_param_fields,
    param_field,
    param_label,
    param_source,
    picker_title,
    ranked_tissue_picks,
    reference_group_note,
    reference_param_fields,
    reference_radiation_fields,
    reference_radiation_value,
    reference_source_of,
    reference_sources,
    relabel_issues,
    relabel_text,
    set_base_tooltip,
    set_editor_mismatch,
    show_comparison,
    spin_value,
    target_provenance,
    tissue_choice_fields,
    tissue_class,
    tissue_edit_fields,
    tissue_errors,
    tissue_issue_text,
    tissue_issues_quietly,
    tissue_model_label,
    write_tissue,
)
from ._labels import TruncatedCheckBox


# Order in which groups are rendered, and their display titles.
_TYPE_GROUPS: tuple[tuple[str, str], ...] = (
    ("TARGET", "Targets"),
    ("OAR", "OARs"),
    ("EXTERNAL", "External"),
    ("HELPER", "Helpers"),
)
_OTHER_GROUP = ("OTHER", "Other")

# Types instantiable via create_voi, offered in the metadata editor.
_VOI_TYPES = ("TARGET", "OAR", "EXTERNAL", "HELPER")

_TISSUE_COMBO_TOOLTIP = "RBE model whose tissue parameters are edited below"

#: Delay after the last edit before the tissue is compared with the base data again.
COMPARISON_DELAY_MS = 150

_GROUPBOX_STYLE = (
    "QGroupBox { font-size: 10pt; font-weight: 600; margin-top: 6px; "
    "border: 1px solid palette(mid); border-radius: 4px; padding: 4px 6px 4px 6px; }"
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)


class _VOICheckBox(TruncatedCheckBox):
    """Checkbox whose label area emits *label_clicked* instead of toggling.

    Toggling remains available on the indicator box and via keyboard.
    """

    label_clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            opt = QStyleOptionButton()
            self.initStyleOption(opt)
            indicator = self.style().subElementRect(
                QStyle.SubElement.SE_CheckBoxIndicator, opt, self
            )
            if event.position().toPoint().x() > indicator.right():
                self.label_clicked.emit()
                event.accept()
                return
        super().mousePressEvent(event)


def _editable_values(tissue: TissueProperties) -> dict[str, object]:
    """Values of a tissue the editor offers, its reference radiation included."""
    tissue_type = type(tissue)
    names = tissue_edit_fields(tissue_type) + reference_radiation_fields(tissue_type)
    return {name: getattr(tissue, name) for name in names}


def _picker_row(*widgets: QWidget, stretch: int = 0) -> QWidget:
    """Wrap widgets into one form row, the one at `stretch` taking the spare room."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    for position, widget in enumerate(widgets):
        layout.addWidget(widget, 1 if position == stretch else 0)
    return row


class VOIMetadataDialog(QDialog):
    """Small popup to edit the basic metadata of a single VOI.

    On accept the VOI is recreated from the edited values through ``create_voi``, so
    everything is validated together, and the result is exposed via
    :attr:`updated_voi` (the input VOI is left untouched). Both the VOI type and the
    tissue model are bound to a model class, hence the recreation.

    The tissue parameters shown are those of the selected RBE model
    (:class:`TissueProperties` for reference parameters only,
    :class:`LEMTissueProperties` or :class:`MKMTissueProperties` for MKM and mMKM), in two
    groups: the model parameters, and the reference parameters α_x / β_x of the radiation
    the RBE refers to. Reference parameters that were not given explicitly (derived, from
    base data or the default) show in grey italics. MKM and mMKM derive them from the model
    parameters while these are edited, for the reference radiation selected in the
    reference group (:attr:`reference_radiation_combo`); typing a value overrides them, and
    "Derive from model" (:attr:`derive_button`) discards the override.

    The tissue picker offers the tissue classes of the base data first - those the
    selected model reads, else the machine's own plus those of a model bringing its own
    tables - together with the cell line presets of that base data's provenance; below a
    separator follow all other presets, grouped by provenance. A preset of another
    provenance is adopted as the base data's.

    With a machine and model, the tissue is compared with the base data of the dose engine
    `engine` while editing, together with the other structures of `cst`: parameters
    differing from the closest tissue class are tinted, and "Match base data" applies that
    class. A model (or LEM version) of another provenance than the matching class is a soft
    label: it is tinted more softly, the model is evaluated with the base data's provenance,
    and "Relabel as ..." optionally adopts it. Accepting an edited tissue checks it; a
    failing tissue is highlighted and the user asked whether to apply it anyway, while a
    tissue that only warns never asks.
    """

    def __init__(
        self,
        voi: VOI,
        machine: object = None,
        bio_model: object = None,
        parent: QWidget | None = None,
        cst: object = None,
        engine: object = None,
    ) -> None:
        super().__init__(parent)
        self._voi = voi
        self._machine = machine
        summary = machine_summary(machine)
        resolved = bio_model_quietly(bio_model, summary.radiation_mode if summary else "")
        self._bio_model = resolved if resolved is not None else bio_model
        self._cst = cst
        self._engine = engine
        self.updated_voi: VOI = voi
        self.setWindowTitle(f"Edit VOI: {voi.name}")

        form = self._form = QFormLayout(self)
        form.addRow("Name:", QLabel(voi.name))

        self.type_combo = QComboBox()
        self.type_combo.addItems(list(_VOI_TYPES))
        if voi.voi_type not in _VOI_TYPES:
            self.type_combo.addItem(voi.voi_type)
        self.type_combo.setCurrentText(voi.voi_type)
        form.addRow("Type:", self.type_combo)

        self.tissue_combo = QComboBox()
        self.tissue_combo.addItems([label for label, _cls in TISSUE_MODELS])
        set_base_tooltip(self.tissue_combo, _TISSUE_COMBO_TOOLTIP)
        self.tissue_combo.setCurrentText(tissue_model_label(voi.tissue_properties))
        form.addRow("Tissue model:", self.tissue_combo)

        self._target_provenance = target_provenance(machine, self._bio_model, engine)
        self._presets = cell_line_presets()
        self._machine_tissues = available_tissue_classes(machine, self._bio_model, engine)
        self._picks = ranked_tissue_picks(
            self._machine_tissues, self._presets, self._target_provenance
        )
        self.preset_combo: QComboBox | None = None
        if self._picks:
            form.addRow(
                "Tissue:", self._build_picker(picker_title(machine, self._bio_model, engine))
            )

        # Tissue the cell line and model name of the edited properties are taken from,
        # replaced by a preset or tissue class when one is applied.
        self._tissue_identity: TissueProperties = voi.tissue_properties

        # Values entered so far, kept across model switches so that changing the model
        # back and forth does not lose them, with where the reference values came from.
        # NaN means "left unset".
        self._tissue_values: dict[str, object] = _editable_values(voi.tissue_properties)
        self._tissue_sources: dict[str, tuple[str, str]] = reference_sources(voi.tissue_properties)
        self._tissue_spins: dict[str, QDoubleSpinBox] = {}
        # Parameters offering a fixed set of choices, e.g. the LEM version
        self._tissue_choices: dict[str, QComboBox] = {}
        # Only built for a model deriving its reference parameters (MKM, mMKM)
        self.derive_button: QPushButton | None = None
        #: Selector of the reference radiation, for a model deriving its reference (MKM).
        self.reference_radiation_combo: QComboBox | None = None
        # Writing a whole tissue must not derive from its half-written model parameters
        self._writing_tissue = False
        self._comparison: TissueComparison | None = None
        self._comparison_timer = QTimer(self)
        self._comparison_timer.setSingleShot(True)
        self._comparison_timer.setInterval(COMPARISON_DELAY_MS)
        self._comparison_timer.timeout.connect(self.refresh_comparison)

        #: The model parameters of the selected tissue model.
        self.model_group = QGroupBox(MODEL_GROUP_TITLE)
        self._model_form = QFormLayout(self.model_group)
        #: The reference parameters, with how they come about for the selected model.
        self.reference_group = QGroupBox(REFERENCE_GROUP_TITLE)
        self._reference_form = QFormLayout(self.reference_group)
        self.reference_note = QLabel()
        tissue_params = QWidget()
        tissue_layout = QVBoxLayout(tissue_params)
        tissue_layout.setContentsMargins(0, 0, 0, 0)
        tissue_layout.addWidget(self.model_group)
        tissue_layout.addWidget(self.reference_group)
        form.addRow(tissue_params)
        self._rebuild_tissue_form()
        self.tissue_combo.currentTextChanged.connect(self._on_tissue_model_changed)

        #: The matching or closest tissue class of the base data.
        self.comparison_label = QLabel()
        self.comparison_label.setWordWrap(True)
        self.match_button = QPushButton(MATCH_BASE_DATA_TEXT)
        self.match_button.setToolTip(MATCH_BASE_DATA_TOOLTIP)
        self.match_button.clicked.connect(self.match_base_data)
        #: Optionally relabels a tissue matching the base data under another provenance.
        self.relabel_button = QPushButton("Relabel")
        self.relabel_button.setToolTip(RELABEL_TOOLTIP)
        self.relabel_button.clicked.connect(self.relabel)
        self._comparison_row = _picker_row(
            self.comparison_label, self.match_button, self.relabel_button
        )
        form.addRow("Base data:", self._comparison_row)

        #: Where the values last applied from a preset or tissue class came from.
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray;")
        form.addRow(self.status_label)

        self.priority_spin = QSpinBox()
        self.priority_spin.setRange(0, 9999)
        self.priority_spin.setValue(voi.overlap_priority)
        self.priority_spin.setToolTip("Lower priority numbers overlap higher ones")
        form.addRow("Overlap priority:", self.priority_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.refresh_comparison()

    @property
    def alpha_spin(self) -> QDoubleSpinBox:
        """Spin box for ``alpha_x``, present for every tissue model."""
        return self._tissue_spins["alpha_x"]

    @property
    def beta_spin(self) -> QDoubleSpinBox:
        """Spin box for ``beta_x``, present for every tissue model."""
        return self._tissue_spins["beta_x"]

    def _build_picker(self, title: str) -> QWidget:
        """Build the row picking a tissue class of the base data or a cell line preset."""
        self.preset_combo = QComboBox()
        fill_tissue_picker(self.preset_combo, self._picks, title)
        self.preset_combo.setToolTip(
            "Tissue classes of the base data and the cell line presets of its model first "
            "(✓: matches the base data exactly), then the presets of other models. A preset "
            "of another model is converted to the base data's, keeping the parameters both "
            "share."
        )

        apply_button = QPushButton("Apply")
        apply_button.setToolTip(
            "Fill the tissue parameters of this structure. A preset describes a whole "
            "tissue and sets all parameters of its model; a tissue class of the base data "
            "sets its model and the parameters identifying it, keeping the others (e.g. "
            "alpha_0 / beta_0 of MKM)."
        )
        apply_button.clicked.connect(self._on_apply_pick)

        return _picker_row(self.preset_combo, apply_button)

    def _on_apply_pick(self) -> None:
        pick = current_pick(self.preset_combo, self._picks) if self.preset_combo else None
        if pick is None:
            return
        if pick.kind == "class":
            self.apply_machine_tissue(pick.index)
        else:
            self.apply_cell_line_preset(pick.index)

    def apply_cell_line_preset(self, index: int) -> str:
        """
        Fill the form from one of the cell line presets (:attr:`_presets`).

        A preset describes a complete tissue, so all parameters of its model are taken
        over, together with the cell line. A preset of another provenance than the base
        data the selected model reads is adopted as that provenance first. Returns the
        note on where the values came from, which the dialog shows.
        """
        tissue, note = adopt_preset(self._presets[index][1], self._target_provenance)
        self._write_tissue(tissue)
        self.status_label.setText(note)
        return note

    def apply_machine_tissue(self, index: int) -> None:
        """
        Apply one of the base data's tissue classes (:attr:`_machine_tissues`) to the form.

        The tissue adopts the class's model and the parameters identifying it, since those
        are what the RBE model matches against its base data; the parameters describing
        the tissue itself (e.g. `alpha_0` / `beta_0`, which MKM takes per voxel) are kept,
        see :func:`~pyRadPlan.biology.tissues.apply_tissue_class`.
        """
        tissue_class = self._machine_tissues[index]
        tissue, class_name = apply_base_data_class(self._current_tissue(), tissue_class)
        self._write_tissue(tissue, class_name)
        self.status_label.setText(
            f"Applied the base data class {tissue_class.cell_line or tissue_class.display_name}"
        )

    def match_base_data(self) -> bool:
        """Apply the closest tissue class of the base data; False if there is none."""
        self.refresh_comparison()
        comparison = self._comparison
        if comparison is None or comparison.tissue_class is None:
            return False
        tissue, class_name = apply_base_data_class(self._current_tissue(), comparison.tissue_class)
        self._write_tissue(tissue, class_name)
        self.status_label.setText("Matched the closest base data class")
        self.refresh_comparison()
        return True

    def _current_tissue(self) -> TissueProperties:
        """Tissue of the form, the last applied one if the entered values do not validate."""
        try:
            return self.tissue_properties()
        except (ValidationError, ValueError, TypeError):
            return self._tissue_identity

    def _write_tissue(self, tissue: TissueProperties, class_name: str | None = None) -> None:
        """Show a tissue in the form, which then carries its cell line and model name."""
        # Switching the model rebuilds the parameter rows, so set the values afterwards
        self._writing_tissue = True
        try:
            self._tissue_identity = tissue
            self.tissue_combo.setCurrentText(tissue_model_label(tissue))
            write_tissue(
                self._tissue_spins,
                self._tissue_choices,
                tissue,
                class_name,
                radiation=self.reference_radiation_combo,
            )
        finally:
            self._writing_tissue = False
        self._update_reference_note()
        self._schedule_comparison()

    def _selected_tissue_class(self) -> type[TissueProperties]:
        """Tissue properties class for the currently selected tissue model."""
        return tissue_class(self.tissue_combo.currentText())

    def _rebuild_tissue_form(self) -> None:
        """Show one editor per parameter of the selected tissue model, in two groups."""
        for group_form in (self._model_form, self._reference_form):
            while group_form.rowCount():
                group_form.removeRow(0)
        self._tissue_spins.clear()
        self._tissue_choices.clear()

        tissue_cls = self._selected_tissue_class()
        for name in tissue_choice_fields(tissue_cls):
            value = self._tissue_values.get(name, tissue_cls.model_fields[name].default)
            combo = make_param_combo(name, value)
            combo.currentTextChanged.connect(self._schedule_comparison)
            self._model_form.addRow(f"{param_label(name)}:", combo)
            self._tissue_choices[name] = combo
        for group_form, names in (
            (self._model_form, model_param_fields(tissue_cls)),
            (self._reference_form, reference_param_fields(tissue_cls)),
        ):
            for name in names:
                value = self._tissue_values.get(name, tissue_cls.model_fields[name].default)
                source, note = self._tissue_sources.get(name, ("explicit", ""))
                spin = make_param_spin(name, value, source=source, note=note)
                spin.valueChanged.connect(self._schedule_comparison)
                if group_form is self._model_form:
                    spin.valueChanged.connect(self._on_model_param_changed)
                group_form.addRow(f"{param_label(name)}:", spin)
                self._tissue_spins[name] = spin
        self.model_group.setTitle(f"{MODEL_GROUP_TITLE} · {self._model_name()}")
        self.model_group.setVisible(self._model_form.rowCount() > 0)

        self.reference_radiation_combo = None
        if reference_radiation_fields(tissue_cls):
            current = self._tissue_values.get(REFERENCE_RADIATION_FIELD)
            self.reference_radiation_combo = make_reference_radiation_combo(current)
            self.reference_radiation_combo.currentIndexChanged.connect(
                self._on_reference_radiation_changed
            )
            self._reference_form.insertRow(
                0, f"{REFERENCE_RADIATION_LABEL}:", self.reference_radiation_combo
            )

        self.reference_note = QLabel()
        self.reference_note.setWordWrap(True)
        self.reference_note.setStyleSheet("color: gray;")
        self._reference_form.addRow(self.reference_note)
        self.derive_button = None
        if derives_reference_params(tissue_cls):
            self.derive_button = make_derive_button()
            self.derive_button.clicked.connect(self.derive_reference_params)
            self._reference_form.addRow("", self.derive_button)
        self._update_reference_note()

    def _reference_radiation(self) -> str | None:
        """Return the reference radiation selected for the tissue, None for the default one."""
        if self.reference_radiation_combo is not None:
            return reference_radiation_value(self.reference_radiation_combo)
        return getattr(self._tissue_identity, REFERENCE_RADIATION_FIELD, None)

    def set_reference_radiation(self, name: str) -> None:
        """Select the reference radiation by registered name (mainly for scripting / tests)."""
        combo = self.reference_radiation_combo
        if combo is None:
            raise ValueError(f"The {self._model_name()} tissue has no reference radiation.")
        index = combo.findData(name)
        if index < 0:
            raise ValueError(f"Unknown reference radiation {name!r}.")
        combo.setCurrentIndex(index)

    def _on_reference_radiation_changed(self, *_args) -> None:
        """Derive the reference parameters for the newly selected reference radiation."""
        if self._writing_tissue:
            return
        self._on_model_param_changed()
        self._update_reference_note()
        self._schedule_comparison()

    def _model_name(self) -> str:
        """Label of the selected tissue model, naming MKM or mMKM for MKM tissues."""
        return self.tissue_combo.currentText()

    def _update_reference_note(self) -> None:
        """Explain how the reference parameters of the selected model come about."""
        tissue_cls = self._selected_tissue_class()
        note = reference_group_note(tissue_cls, self._model_name(), self._reference_radiation())
        self.reference_note.setText(note)
        for name in reference_param_fields(tissue_cls):
            spin = self._tissue_spins.get(name)
            if spin is not None:
                set_base_tooltip(spin, f"{param_field(name).description}\n{note}")

    def _on_model_param_changed(self, *_args) -> None:
        """Derive the reference parameters live from the edited model parameters."""
        if self._writing_tissue:
            return
        apply_derived_reference(
            self._tissue_spins,
            self._selected_tissue_class(),
            self._model_name(),
            self._reference_radiation(),
        )

    def derive_reference_params(self) -> dict[str, float]:
        """
        Derive the reference parameters from the model parameters entered in the form.

        Replaces an explicit override, since the button is an explicit request to do so;
        the values are marked as derived. Only being unable to derive anything is reported
        in a message box.
        """
        derived = apply_derived_reference(
            self._tissue_spins,
            self._selected_tissue_class(),
            self._model_name(),
            self._reference_radiation(),
            force=True,
        )
        if not derived:
            QMessageBox.information(
                self,
                "Nothing to derive from",
                "Set the model parameters first.\n\n" + DERIVE_REFERENCE_TOOLTIP,
            )
        return derived

    def _tissue_form_values(self) -> dict[str, object]:
        """Parameters currently entered in the form, by field name."""
        values: dict[str, object] = {
            name: spin_value(spin) for name, spin in self._tissue_spins.items()
        }
        values.update({name: combo_value(combo) for name, combo in self._tissue_choices.items()})
        if self.reference_radiation_combo is not None:
            radiation = reference_radiation_value(self.reference_radiation_combo)
            if radiation:
                values[REFERENCE_RADIATION_FIELD] = radiation
        return values

    def _on_tissue_model_changed(self, _label: str) -> None:
        self._tissue_values.update(self._tissue_form_values())
        self._tissue_sources.update(
            {name: param_source(spin) for name, spin in self._tissue_spins.items()}
        )
        self._rebuild_tissue_form()
        self._on_model_param_changed()
        self._schedule_comparison()

    def tissue_properties(self) -> TissueProperties:
        """Tissue properties described by the current state of the form."""
        label = self.tissue_combo.currentText()
        return build_tissue_properties(
            tissue_class(label),
            self._tissue_form_values(),
            self._tissue_identity,
            reference_sources=reference_source_of(self._tissue_spins),
            model=model_label_name(label, self._tissue_identity),
        )

    def _structures(self, tissue: TissueProperties) -> list[tuple[str, TissueProperties]]:
        """Every structure with its tissue, this VOI's replaced by `tissue`."""
        others = getattr(self._cst, "vois", self._cst) or []
        structures = [
            (voi.name, voi.tissue_properties) for voi in others if voi.name != self._voi.name
        ]
        structures.append((self._voi.name, tissue))
        return structures

    def _schedule_comparison(self, *_args) -> None:
        """Compare the form with the base data once the edits come to a rest."""
        self._comparison_timer.start()

    def refresh_comparison(self) -> None:
        """
        Compare the tissue with the base data of the selected model and highlight it.

        Parameters differing from the closest tissue class are tinted, with the declared
        value as tooltip, as is the tissue model if it is of another provenance (softly if
        the parameters match, as that is a warning only); parameters the class leaves
        undeclared are never. The base data row names the matching or closest class, with
        "Relabel as ..." while it applies; it is hidden without base data to compare against.
        """
        self._comparison_timer.stop()
        comparisons = compare_tissues_quietly(
            self._machine,
            self._bio_model,
            self._structures(self._current_tissue()),
            self._engine,
        )
        self._comparison = next(
            (c for c in reversed(comparisons) if c.structure == self._voi.name), None
        )
        comparison = self._comparison
        compared = comparison is not None and comparison.kernel_set is not None
        show_comparison(
            (self._tissue_choices, self._tissue_spins),
            self.tissue_combo,
            comparison if compared else None,
        )
        self.comparison_label.setText(comparison_row_note(comparison) if compared else "")
        self.match_button.setEnabled(compared and comparison.tissue_class is not None)
        relabel = relabel_issues([comparison]) if compared else []
        text = relabel_text(issue.fix_provenance for issue in relabel)
        self.relabel_button.setText(text or "Relabel")
        self.relabel_button.setVisible(text is not None)
        self._form.setRowVisible(self._comparison_row, compared)

    def comparison(self) -> TissueComparison | None:
        """Last comparison of the tissue with the base data, None if there is none."""
        return self._comparison

    def check_tissue(self, tissue: TissueProperties) -> list:
        """
        Check a tissue for this VOI against the biological model and machine.

        The other structures of the structure set take part with their own tissues, since
        they can decide which base data the model reads. Returns the issues of this VOI,
        warnings included, and highlights the tissue model box with them (softly if there
        are only warnings); empty without a model or machine summary to check against.
        """
        issues = [
            issue
            for issue in tissue_issues_quietly(
                self._machine, self._bio_model, self._structures(tissue), self._engine
            )
            if issue.structure == self._voi.name
        ]
        set_editor_mismatch(
            self.tissue_combo,
            "\n".join(tissue_issue_text(issue) for issue in issues),
            warning=not tissue_errors(issues),
        )
        return issues

    def relabel(self) -> bool:
        """
        Relabel a tissue matching the base data under another provenance with the base data's.

        Keeps every parameter, see :func:`~pyRadPlan.biology.models.apply_provenance_fixes`.
        Optional, since the tissue is evaluated with the base data's model either way.
        Returns whether the tissue was changed.
        """
        self.refresh_comparison()
        tissue = self._current_tissue()
        issues = relabel_issues([self._comparison])
        fixed = apply_provenance_fixes({self._voi.name: tissue}, issues)[self._voi.name]
        if fixed is tissue:
            return False
        self._write_tissue(fixed)
        self.status_label.setText(f"Relabelled as the base data's model ({fixed.display_name})")
        self.refresh_comparison()
        return True

    def accept(self) -> None:
        """Validate the edited values and expose the updated VOI before closing.

        An edited VOI is recreated rather than mutated: it mirrors ``alpha_x`` /
        ``beta_x`` of its tissue properties and rejects any assignment that leaves the
        two inconsistent, so tissue edits cannot be applied field by field. Without any
        edit, :attr:`updated_voi` stays the input VOI itself. An edited tissue the
        biological model cannot be evaluated for (errors) is only taken over if the user
        chooses to apply it anyway; otherwise the dialog stays open for editing. Warnings,
        e.g. a model label differing from the base data's, never ask.
        """
        new_type = self.type_combo.currentText()
        try:
            tissue = self.tissue_properties()
            if tissue != self._voi.tissue_properties:
                errors = tissue_errors(self.check_tissue(tissue))
                if errors and ask_tissue_issues(self, errors) != APPLY_ANYWAY:
                    return
            if (
                new_type == self._voi.voi_type
                and self.priority_spin.value() == self._voi.overlap_priority
                and tissue == self._voi.tissue_properties
            ):
                self.updated_voi = self._voi
                super().accept()
                return
            data = dict(self._voi)
            data.update(
                voi_type=new_type,
                tissue_properties=tissue,
                alpha_x=tissue.alpha_x,
                beta_x=tissue.beta_x,
                overlap_priority=self.priority_spin.value(),
            )
            # An auto-assigned color is left unset (StructureSet.set_colors() keeps it that
            # way); an explicit pick stays.
            auto_color = "color" not in self._voi.model_fields_set
            if auto_color and new_type != self._voi.voi_type:
                # Let the new type's default_factory supply its own color
                data.pop("color", None)
            self.updated_voi = create_voi(data)
            if auto_color:
                # Recreating marks a copied color as set; keep it reading as auto-assigned
                self.updated_voi.model_fields_set.discard("color")
        except (ValidationError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        super().accept()


class VOIsWidget(QWidget):
    """Widget for VOI selection."""

    # Signals
    voi_toggled = Signal(str, bool)  # voi name, checked
    selection_changed = Signal(list)  # list[str] of selected VOIs
    color_changed = Signal(str, tuple)  # voi name, new RGB tuple
    metadata_changed = Signal(str)  # voi name whose metadata was edited
    voi_replaced = Signal(str, object)  # voi name, new VOI instance (after a type change)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._vois: list[VOI] = []
        self._voi_by_name: dict[str, VOI] = {}
        self._group_mode: str = "type"
        self._voi_checkboxes: dict[str, TruncatedCheckBox] = {}
        self._voi_colors: dict[str, tuple[int, int, int]] = {}
        self._voi_color_swatches: dict[str, QPushButton] = {}
        self._machine_provider: Optional[Callable[[], object]] = None
        self._bio_model_provider: Optional[Callable[[], object]] = None
        self._engine_provider: Optional[Callable[[], object]] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._vois_scroll = QScrollArea()
        self._vois_scroll.setWidgetResizable(True)
        self._vois_container = QWidget()
        self._vois_layout = QVBoxLayout(self._vois_container)
        self._vois_layout.setContentsMargins(2, 2, 2, 2)
        self._vois_layout.setSpacing(4)
        self._vois_scroll.setWidget(self._vois_container)
        layout.addWidget(self._vois_scroll)

        # All / None quick-select buttons
        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        all_btn = QPushButton("All")
        all_btn.setFixedWidth(48)
        none_btn = QPushButton("None")
        none_btn.setFixedWidth(48)
        all_btn.clicked.connect(self._select_all_vois)
        none_btn.clicked.connect(self._deselect_all_vois)
        btn_row.addWidget(all_btn)
        btn_row.addWidget(none_btn)
        btn_row.addStretch(1)

        self._group_combo = QComboBox()
        self._group_combo.addItems(["By Type", "By Overlap"])
        self._group_combo.setToolTip("Group the VOI list by type or by overlap priority")
        self._group_combo.currentIndexChanged.connect(self._on_group_mode_changed)
        btn_row.addWidget(self._group_combo)

        layout.addLayout(btn_row)

    def set_machine_provider(self, provider: Optional[Callable[[], object]]) -> None:
        """
        Set a callable returning the machine whose tissue classes the editor offers.

        Called when the editor opens rather than stored, so that it always reflects the
        current plan without this widget having to watch it.
        """
        self._machine_provider = provider

    def set_bio_model_provider(self, provider: Optional[Callable[[], object]]) -> None:
        """
        Set a callable returning the biological model whose own tissue classes are offered.

        Resolved when the editor opens, like the machine provider, and for the same reason.
        """
        self._bio_model_provider = provider

    def set_engine_provider(self, provider: Optional[Callable[[], object]]) -> None:
        """
        Set a callable returning the dose engine the biological model is computed with.

        Resolved when the editor opens, like the machine provider, and for the same reason.
        """
        self._engine_provider = provider

    def set_vois(self, vois: list[VOI], checked: Optional[Iterable[str]] = None) -> None:
        """Populate the VOIs section with a variable number of checkboxes."""
        self._vois = list(vois)
        self._voi_by_name = {voi.name: voi for voi in self._vois}

        sorted_vois = sorted(self._vois, key=lambda v: v.name.lower())
        empty_names = self._empty_voi_names(sorted_vois)
        checked_set = self._determine_checked_vois(sorted_vois, checked) - empty_names

        self._render(checked_set)

    def _render(self, checked_set: set[str]) -> None:
        """Rebuild the group boxes for the stored VOIs with the given checked state."""
        for cb in self._voi_checkboxes.values():
            cb.deleteLater()
        self._voi_checkboxes.clear()
        self._voi_color_swatches.clear()

        while self._vois_layout.count():
            item = self._vois_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        sorted_vois = sorted(self._vois, key=lambda v: v.name.lower())

        if self._group_mode == "overlap":
            self._render_by_overlap(sorted_vois, checked_set)
        else:
            self._render_by_type(sorted_vois, checked_set)

        self._vois_layout.addStretch(1)

        self.selection_changed.emit(self.selected_vois())

    def _render_by_type(self, sorted_vois: list[VOI], checked_set: set[str]) -> None:
        """Render one group box per VOI type."""
        groups: dict[str, list[VOI]] = {}
        for voi in sorted_vois:
            key = getattr(voi, "voi_type", "").upper() or _OTHER_GROUP[0]
            groups.setdefault(key, []).append(voi)

        for key, title in (*_TYPE_GROUPS, _OTHER_GROUP):
            members = groups.pop(key, [])
            if not members:
                continue
            self._add_group_box(title, members, checked_set)

        # Any unknown types that slipped through
        for key, members in groups.items():
            if not members:
                continue
            self._add_group_box(key.title(), members, checked_set)

    def _render_by_overlap(self, sorted_vois: list[VOI], checked_set: set[str]) -> None:
        """Render one group box per overlap priority level, lowest (dominant) first."""
        groups: dict[int, list[VOI]] = {}
        for voi in sorted_vois:
            groups.setdefault(int(voi.overlap_priority), []).append(voi)

        for priority in sorted(groups):
            self._add_group_box(f"Overlap priority {priority}", groups[priority], checked_set)

    def _on_group_mode_changed(self, index: int) -> None:
        self._group_mode = "overlap" if index == 1 else "type"
        self._render(set(self.selected_vois()))

    def selected_vois(self) -> list[str]:
        """Return the list of currently checked VOIs by name."""
        return [name for name, cb in self._voi_checkboxes.items() if cb.isChecked()]

    def get_voi_colors(self) -> dict[str, tuple[int, int, int]]:
        """Return the color mapping for VOIs."""
        return self._voi_colors

    @staticmethod
    def _empty_voi_names(vois: list[VOI]) -> set[str]:
        """Return names of VOIs whose mask contains no voxels."""
        empty: set[str] = set()
        for voi in vois:
            try:
                if not sitk.GetArrayViewFromImage(voi.mask).any():
                    empty.add(voi.name)
            except Exception:
                pass
        return empty

    def _determine_checked_vois(
        self, vois: list[VOI], checked: Optional[Iterable[str]]
    ) -> set[str]:
        """Determine which VOIs should be checked by default."""
        if checked is not None:
            return set(checked)

        with_objectives = {
            voi.name
            for voi in vois
            if any(obj is not None for obj in getattr(voi, "objectives", None) or [])
        }
        if with_objectives:
            return with_objectives

        body_name, target_name, oar_name = self._select_by_voi_type(vois)
        body_name, target_name, oar_name = self._fill_by_name_heuristic(
            vois, body_name, target_name, oar_name
        )
        return {n for n in [body_name, target_name, oar_name] if n}

    def _select_by_voi_type(
        self, vois: list[VOI]
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (body, target, oar) names selected by voi_type attribute."""
        body_name: Optional[str] = None
        target_name: Optional[str] = None
        oar_name: Optional[str] = None
        for voi in vois:
            vtype = getattr(voi, "voi_type", "").upper()
            if body_name is None and vtype == "EXTERNAL":
                body_name = voi.name
            if target_name is None and vtype == "TARGET":
                target_name = voi.name
            if oar_name is None and vtype == "OAR":
                oar_name = voi.name
            if body_name and target_name and oar_name:
                break
        return body_name, target_name, oar_name

    def _fill_by_name_heuristic(
        self,
        vois: list[VOI],
        body_name: Optional[str],
        target_name: Optional[str],
        oar_name: Optional[str],
    ) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """Fill in any ``None`` entry via name substring matching."""
        if body_name is None:
            for voi in vois:
                if "body" in voi.name.lower():
                    body_name = voi.name
                    break
        if target_name is None:
            for voi in vois:
                if "target" in voi.name.lower():
                    target_name = voi.name
                    break
        if oar_name is None:
            for voi in vois:
                if "oar" in voi.name.lower():
                    oar_name = voi.name
                    break
        return body_name, target_name, oar_name

    def _add_group_box(self, title: str, members: list[VOI], checked_set: set[str]) -> None:
        """Create a small QGroupBox for *title* containing a 2-column grid of VOIs."""
        group = QGroupBox(title)
        group.setStyleSheet(_GROUPBOX_STYLE)
        grid = QGridLayout(group)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        for i, voi in enumerate(members):
            row_widget = self._build_voi_row(voi, checked_set)
            grid_row, grid_col = divmod(i, 2)
            grid.addWidget(row_widget, grid_row, grid_col)

        self._vois_layout.addWidget(group)

    def _build_voi_row(self, voi: VOI, checked_set: set[str]) -> QWidget:
        """Build a single [swatch | checkbox] row widget for *voi*."""
        cb = _VOICheckBox(voi.name)
        cb.setChecked(voi.name in checked_set)
        cb.setToolTip(self._voi_tooltip(voi))
        cb.stateChanged.connect(lambda s, n=voi.name: self._on_voi_toggled(n, bool(s)))
        cb.label_clicked.connect(lambda n=voi.name: self._edit_voi(n))

        rgb = tuple(int(c) for c in voi.color)
        self._voi_colors[voi.name] = rgb

        swatch = QPushButton()
        swatch.setFixedSize(12, 12)
        swatch.setFlat(True)
        swatch.setStyleSheet(
            f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #444;"
        )
        swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        swatch.clicked.connect(lambda _, n=voi.name: self._pick_color(n))
        self._voi_color_swatches[voi.name] = swatch

        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)
        row_layout.addWidget(swatch, 0)
        row_layout.addWidget(cb, 1)

        self._voi_checkboxes[voi.name] = cb
        return row

    @staticmethod
    def _voi_tooltip(voi: VOI) -> str:
        """Build a rich-text tooltip summarizing the VOI metadata."""
        tissue = voi.tissue_properties
        if tissue.beta_x:
            ratio = f"{tissue.alpha_x / tissue.beta_x:.4g}&nbsp;Gy"
        else:
            ratio = "&#8734;"

        model_params = ", ".join(
            f"{param_label(name)} = {getattr(tissue, name):g}"
            for name in model_param_fields(type(tissue))
            if not math.isnan(getattr(tissue, name))
        )
        lines = [
            f"<b>{html.escape(voi.name)}</b> ({html.escape(voi.voi_type)})",
            f"Tissue model: {html.escape(tissue.model)}",
            f"&alpha;<sub>x</sub> = {tissue.alpha_x:g}&nbsp;Gy<sup>&#8722;1</sup>, "
            f"&beta;<sub>x</sub> = {tissue.beta_x:g}&nbsp;Gy<sup>&#8722;2</sup>",
            f"&alpha;/&beta; = {ratio}",
        ]
        if model_params:
            lines.append(html.escape(model_params))
        lines.append(f"Overlap priority: {voi.overlap_priority}")
        lines.append("<i>Click the name to edit</i>")
        return "<br/>".join(lines)

    def _edit_voi(self, name: str) -> None:
        """Open the metadata editor popup for the given VOI."""
        voi = self._voi_by_name.get(name)
        if voi is None:
            return
        machine = self._machine_provider() if self._machine_provider is not None else None
        bio_model = self._bio_model_provider() if self._bio_model_provider is not None else None
        engine = self._engine_provider() if self._engine_provider is not None else None
        dialog = VOIMetadataDialog(voi, machine, bio_model, self, cst=self._vois, engine=engine)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_voi = dialog.updated_voi
        if new_voi is not voi:
            self._vois = [new_voi if v is voi else v for v in self._vois]
            self._voi_by_name[name] = new_voi
        self._render(set(self.selected_vois()))
        if new_voi is not voi:
            self.voi_replaced.emit(name, new_voi)
        self.metadata_changed.emit(name)

    def _pick_color(self, name: str) -> None:
        """Open a color dialog for the given VOI and apply the result."""
        current = self._voi_colors.get(name, (128, 128, 128))
        color = QColorDialog.getColor(QColor(*current), self, f"Pick color for {name}")
        if color.isValid():
            rgb = (color.red(), color.green(), color.blue())
            self._voi_colors[name] = rgb
            voi = self._voi_by_name.get(name)
            if voi is not None:
                voi.color = rgb
            swatch = self._voi_color_swatches[name]
            swatch.setStyleSheet(
                f"background-color: rgb({rgb[0]}, {rgb[1]}, {rgb[2]}); border: 1px solid #444;"
            )
            self.color_changed.emit(name, rgb)

    def _on_voi_toggled(self, name: str, checked: bool) -> None:
        self.voi_toggled.emit(name, checked)
        self.selection_changed.emit(self.selected_vois())

    def _select_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
        self.selection_changed.emit(self.selected_vois())

    def _deselect_all_vois(self) -> None:
        for cb in self._voi_checkboxes.values():
            cb.blockSignals(True)
            cb.setChecked(False)
            cb.blockSignals(False)
        self.selection_changed.emit(self.selected_vois())

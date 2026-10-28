"""Dialog editing the tissue properties of all VOIs of a structure set."""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import ValidationError
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pyRadPlan.biology.models import TissueComparison, apply_provenance_fixes
from pyRadPlan.cst import StructureSet, create_voi
from pyRadPlan.biology.tissues import REFERENCE_FIELDS, TissueProperties

from .._tissue_params import (
    APPLY_ANYWAY,
    DERIVE_REFERENCE_TOOLTIP,
    MATCH_BASE_DATA_TEXT,
    MATCH_BASE_DATA_TOOLTIP,
    MODEL_GROUP_TITLE,
    REFERENCE_GROUP_TITLE,
    REFERENCE_RADIATION_FIELD,
    REFERENCE_RADIATION_LABEL,
    RELABEL_TOOLTIP,
    TISSUE_MODELS,
    adopt_preset,
    all_tissue_choice_fields,
    all_tissue_param_fields,
    apply_base_data_class,
    apply_derived_reference,
    ask_tissue_issues,
    available_tissue_classes,
    bio_model_quietly,
    build_tissue_properties,
    cell_container,
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
    param_field,
    param_label,
    param_unit,
    picker_title,
    provenance_fixes,
    ranked_tissue_picks,
    reference_group_note,
    reference_radiation_fields,
    reference_radiation_value,
    reference_source_of,
    reference_sources,
    relabel_issues,
    relabel_text,
    set_base_tooltip,
    show_comparison,
    spin_value,
    target_provenance,
    tissue_class,
    tissue_edit_fields,
    tissue_errors,
    tissue_issue_text,
    tissue_issues_quietly,
    tissue_model_label,
    write_tissue,
)

#: Background of the structure cell of a row whose tissue fails the model's check.
ISSUE_BRUSH = QBrush(QColor(230, 126, 34, 90))

#: Softer background of the structure cell of a row whose tissue only warns, i.e. matches
#: the base data by its parameters but is labelled with another provenance.
WARNING_BRUSH = QBrush(QColor(241, 196, 15, 70))

#: Delay after the last edit before the table is compared with the base data again.
COMPARISON_DELAY_MS = 150

_MODEL_COMBO_TOOLTIP = "RBE model whose tissue parameters the row holds"

#: Marks the column header of a reference parameter.
REFERENCE_HEADER_MARK = "ref."

# Tooltip of the derive button while some structure uses a model deriving its reference.
DERIVE_BUTTON_TOOLTIP = (
    "Derive the reference parameters of the selected MKM / mMKM structures, or of all of "
    "them if none are selected, from their model parameters, discarding overrides.\n\n"
    + DERIVE_REFERENCE_TOOLTIP
)

# Tooltip of the derive button while it is disabled, i.e. no structure uses MKM.
DERIVE_BUTTON_DISABLED_TOOLTIP = (
    "Only MKM and mMKM derive their reference parameters from their model parameters, and "
    "no structure uses them. Set the model of a structure to MKM or mMKM to enable this."
    "\n\n" + DERIVE_REFERENCE_TOOLTIP
)

_TABLE_NOTE = (
    "Tissue parameters per structure: the model parameters, then the reference parameters "
    f"α_x / β_x ({REFERENCE_HEADER_MARK}) of the radiation the RBE refers to. MKM and mMKM "
    "derive these from their model parameters for the selected reference radiation (grey "
    "italics) unless overridden. The biological model maps the parameters to the tissue "
    "classes of its base data."
)

# Columns of the reference group, marked with REFERENCE_HEADER_MARK.
_REFERENCE_COLUMNS = (REFERENCE_RADIATION_FIELD, *REFERENCE_FIELDS)


class TissueParametersDialog(QDialog):
    """
    Table of the VOIs' tissue properties with editable values.

    Every structure carries the tissue parameters of its RBE model (reference
    parameters only, LEM with its version, MKM or mMKM); only the parameter columns of the
    models currently selected are shown, the model parameters first and the reference
    parameters α_x / β_x (marked "ref.") last. Reference parameters that were not given
    explicitly (derived, from base data or the default) show in grey italics. MKM and mMKM
    rows derive them live from their model parameters, for the reference radiation selected
    in the row's "Reference radiation (ref.)" column; typing a value overrides them, and
    "Derive from model" (:attr:`derive_button`) discards the override.

    Rows can be filled from the tissue picker. It offers the tissue classes of the base
    data first - those the selected model reads, else the machine's own plus those of a
    model bringing its own tables - together with the cell line presets of that base
    data's provenance; below a separator follow all other presets, grouped by provenance.
    A preset of another provenance is adopted as the base data's (see
    :func:`~pyRadPlan.biology.tissues.adopt_provenance`); a tissue class is applied with
    :func:`~pyRadPlan.biology.tissues.apply_tissue_class`.

    While editing, the table is compared with the base data of the selected model: values
    differing from the closest tissue class are tinted. A model (or LEM version) of another
    provenance than the matching class is a soft label: it is tinted more softly, the model
    is evaluated with the base data's provenance, and "Relabel as ..." optionally adopts it.
    "Match base data" applies the closest class. Accepting the dialog checks the edited
    tissues against the model and machine; rows failing the check are highlighted and the
    user asked whether to apply them anyway, while rows that only warn never ask.
    Accepting writes the values to the VOIs of the given structure set; :attr:`changed`
    tells whether any value differs from before.

    The dose engine (class, name or instance) decides, with the machine, which base data the
    model reads: an engine not using the machine's kernels restricts nothing.
    """

    def __init__(
        self,
        cst: StructureSet,
        machine: Any = None,
        bio_model: Any = None,
        parent: Optional[QWidget] = None,
        engine: Any = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Tissue parameters")
        self._cst = cst
        self._machine = machine
        summary = machine_summary(machine)
        resolved = bio_model_quietly(bio_model, summary.radiation_mode if summary else "")
        self._bio_model = resolved if resolved is not None else bio_model
        self._engine = engine
        self.changed = False

        layout = QVBoxLayout(self)
        note = QLabel(_TABLE_NOTE)
        note.setWordWrap(True)
        layout.addWidget(note)

        self._target_provenance = target_provenance(machine, self._bio_model, engine)
        self._presets = cell_line_presets()
        self._machine_tissues = available_tissue_classes(machine, self._bio_model, engine)
        self._picks = ranked_tissue_picks(
            self._machine_tissues, self._presets, self._target_provenance
        )
        self.preset_combo: Optional[QComboBox] = None
        if self._picks:
            layout.addLayout(self._build_picker(picker_title(machine, self._bio_model, engine)))

        # Tissue each row takes its cell line and model name from, replaced by a preset
        # or tissue class when one is applied to that row.
        self._tissue_identities = [voi.tissue_properties for voi in cst.vois]
        # Writing a whole tissue must not derive from its half-written model parameters
        self._writing_row: Optional[int] = None
        self._comparisons: dict[int, TissueComparison] = {}
        self._comparison_timer = QTimer(self)
        self._comparison_timer.setSingleShot(True)
        self._comparison_timer.setInterval(COMPARISON_DELAY_MS)
        self._comparison_timer.timeout.connect(self.refresh_comparison)

        # Parameters offering a fixed set of choices (the LEM version) are shown next
        # to the model they belong to, the numeric ones after them.
        # The reference radiation (MKM / mMKM) heads the reference columns.
        self._choice_fields = all_tissue_choice_fields()
        self._param_fields = all_tissue_param_fields()
        first_reference = next(
            (i for i, name in enumerate(self._param_fields) if name in REFERENCE_FIELDS),
            len(self._param_fields),
        )
        self._fields = (
            self._choice_fields
            + self._param_fields[:first_reference]
            + [REFERENCE_RADIATION_FIELD]
            + self._param_fields[first_reference:]
        )
        headers = ["Structure", "Type", "Model"] + [self._header(name) for name in self._fields]
        self._table = QTableWidget(len(cst.vois), len(headers), self)
        self._table.setHorizontalHeaderLabels(headers)
        for name in self._fields:
            group = REFERENCE_GROUP_TITLE if name in _REFERENCE_COLUMNS else MODEL_GROUP_TITLE
            header = self._table.horizontalHeaderItem(self._param_column(name))
            header.setToolTip(f"{group}: {param_field(name).description}")
        self._table.verticalHeader().setVisible(False)
        # Clicking a structure name selects its row, which the pickers apply to
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

        self._model_combos: list[QComboBox] = []
        self._spins: list[dict[str, QDoubleSpinBox]] = []
        self._choices: list[dict[str, QComboBox]] = []
        #: Reference radiation selector of each row, shown for MKM / mMKM rows only.
        self._radiation_combos: list[QComboBox] = []
        for row, voi in enumerate(cst.vois):
            self._build_row(row, voi)

        layout.addWidget(self._table)
        layout.addLayout(self._build_bulk_editor())
        self._update_columns()

        #: Where the values last applied from a preset or tissue class came from.
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: gray;")
        layout.addWidget(self.status_label)

        self.match_button = QPushButton(MATCH_BASE_DATA_TEXT)
        self.match_button.setToolTip(
            MATCH_BASE_DATA_TOOLTIP + "\n\nApplies to the selected structures, or to all "
            "structures if none are selected."
        )
        self.match_button.clicked.connect(lambda checked=False: self.match_base_data())
        #: Optionally relabels rows matching the base data under another provenance.
        self.relabel_button = QPushButton()
        self.relabel_button.setToolTip(
            RELABEL_TOOLTIP + "\n\nApplies to the selected structures, or to all "
            "structures if none are selected."
        )
        self.relabel_button.clicked.connect(lambda checked=False: self.relabel())
        #: Derives the reference parameters of MKM / mMKM rows, discarding overrides.
        self.derive_button = make_derive_button()
        self.derive_button.clicked.connect(self._on_derive_reference_params)
        self._update_derive_button()
        action_row = QHBoxLayout()
        action_row.addWidget(self.match_button)
        action_row.addWidget(self.relabel_button)
        action_row.addStretch(1)
        action_row.addWidget(self.derive_button)
        layout.addLayout(action_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.refresh_comparison()

    def _build_row(self, row: int, voi: Any) -> None:
        """Fill one table row with the editors of one structure's tissue properties."""
        for col, text in enumerate((voi.name, voi.voi_type)):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, col, item)

        tissue = voi.tissue_properties
        combo = QComboBox()
        combo.addItems([label for label, _cls in TISSUE_MODELS])
        combo.setCurrentText(tissue_model_label(tissue))
        set_base_tooltip(combo, _MODEL_COMBO_TOOLTIP)
        combo.currentTextChanged.connect(lambda _text, r=row: self._on_model_changed(r))
        self._table.setCellWidget(row, 2, combo)
        self._model_combos.append(combo)

        choices: dict[str, QComboBox] = {}
        for name in self._choice_fields:
            value = getattr(tissue, name, param_field(name).default)
            choice = make_param_combo(name, value)
            choice.currentTextChanged.connect(self._schedule_comparison)
            self._table.setCellWidget(row, self._param_column(name), cell_container(choice))
            choices[name] = choice
        self._choices.append(choices)

        radiation = make_reference_radiation_combo(
            getattr(tissue, REFERENCE_RADIATION_FIELD, None)
        )
        radiation.currentIndexChanged.connect(
            lambda _index, r=row: self._on_reference_radiation_changed(r)
        )
        self._table.setCellWidget(
            row, self._param_column(REFERENCE_RADIATION_FIELD), cell_container(radiation)
        )
        self._radiation_combos.append(radiation)

        sources = reference_sources(tissue)
        spins: dict[str, QDoubleSpinBox] = {}
        for name in self._param_fields:
            source, note = sources.get(name, ("explicit", ""))
            value = getattr(tissue, name, float("nan"))
            spin = make_param_spin(name, value, with_unit=False, source=source, note=note)
            spin.valueChanged.connect(self._schedule_comparison)
            if name not in REFERENCE_FIELDS:
                spin.valueChanged.connect(lambda _value, r=row: self._on_model_param_changed(r))
            self._table.setCellWidget(row, self._param_column(name), cell_container(spin))
            spins[name] = spin
        self._spins.append(spins)
        self._update_row(row)

    @staticmethod
    def _header(name: str) -> str:
        """Column header of a tissue parameter, marking the reference parameters."""
        label = param_label(name)
        if name == REFERENCE_RADIATION_FIELD:
            label = REFERENCE_RADIATION_LABEL
        if name in _REFERENCE_COLUMNS:
            label = f"{label} ({REFERENCE_HEADER_MARK})"
        return f"{label} [{param_unit(name)}]" if param_unit(name) else label

    def _build_picker(self, title: str) -> QHBoxLayout:
        """Build the row picking a tissue class of the base data or a cell line preset."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Tissue:"))

        self.preset_combo = QComboBox()
        fill_tissue_picker(self.preset_combo, self._picks, title)
        self.preset_combo.setToolTip(
            "Tissue classes of the base data and the cell line presets of its model first "
            "(✓: matches the base data exactly), then the presets of other models. A preset "
            "of another model is converted to the base data's, keeping the parameters both "
            "share."
        )
        row.addWidget(self.preset_combo, 1)

        apply_button = QPushButton("Apply")
        apply_button.setToolTip(
            "Fill the selected structures, or all structures if none are selected. A "
            "preset describes a whole tissue and sets all parameters of its model; a tissue "
            "class of the base data sets its model and the parameters identifying it, "
            "keeping the others (e.g. alpha_0 / beta_0 of MKM)."
        )
        apply_button.clicked.connect(self._on_apply_pick)
        row.addWidget(apply_button)
        return row

    def _build_bulk_editor(self) -> QHBoxLayout:
        """Build the row setting one parameter to the same value in several structures."""
        row = QHBoxLayout()
        row.addWidget(QLabel("Set parameter:"))
        #: Numeric parameter the bulk editor sets, of those shown in the table.
        self.bulk_param_combo = QComboBox()
        self.bulk_param_combo.setToolTip("Tissue parameter to set in several structures at once")
        self.bulk_param_combo.currentIndexChanged.connect(self._on_bulk_param_changed)
        row.addWidget(self.bulk_param_combo)
        #: Value the bulk editor sets.
        self.bulk_value_spin = QDoubleSpinBox()
        self.bulk_value_spin.setDecimals(4)
        self.bulk_value_spin.setRange(0.0, 100.0)
        self.bulk_value_spin.setSingleStep(0.01)
        row.addWidget(self.bulk_value_spin)
        self.bulk_set_button = QPushButton("Set")
        self.bulk_set_button.setToolTip(
            "Set the parameter in the selected structures, or in all structures if none are "
            "selected. Structures whose model does not have the parameter are left alone."
        )
        self.bulk_set_button.clicked.connect(lambda checked=False: self.set_param_for_rows())
        row.addWidget(self.bulk_set_button)
        row.addStretch(1)
        return row

    def _update_bulk_params(self, used: set[str]) -> None:
        """Offer the numeric parameters some row makes use of in the bulk editor."""
        current = self.bulk_param_combo.currentData()
        self.bulk_param_combo.blockSignals(True)
        self.bulk_param_combo.clear()
        for name in self._param_fields:
            if name in used:
                self.bulk_param_combo.addItem(self._header(name), name)
        index = self.bulk_param_combo.findData(current)
        self.bulk_param_combo.setCurrentIndex(max(index, 0))
        self.bulk_param_combo.blockSignals(False)
        if self.bulk_param_combo.currentData() != current:
            self._on_bulk_param_changed()
        enabled = self.bulk_param_combo.count() > 0
        self.bulk_value_spin.setEnabled(enabled)
        self.bulk_set_button.setEnabled(enabled)

    def _on_bulk_param_changed(self, *_args) -> None:
        """Start the bulk value from the parameter's value in the first row having it."""
        name = self.bulk_param_combo.currentData()
        if name is None:
            return
        set_base_tooltip(self.bulk_value_spin, param_field(name).description or "")
        for row in range(len(self._spins)):
            if name in tissue_edit_fields(self._row_tissue_class(row)):
                value = spin_value(self._spins[row][name])
                if not math.isnan(value):
                    self.bulk_value_spin.setValue(value)
                return

    def set_param_for_rows(
        self, name: Optional[str] = None, value: Optional[float] = None, rows=None
    ) -> list[int]:
        """
        Set one numeric parameter to the same value in several rows.

        Defaults to the parameter and value of the bulk editor and to the selected rows, or
        all rows if none are selected. Rows whose model does not have the parameter are
        skipped. Returns the rows that were set.
        """
        name = self.bulk_param_combo.currentData() if name is None else name
        value = self.bulk_value_spin.value() if value is None else value
        if name is None:
            return []
        targets = self._target_rows() if rows is None else rows
        changed = [
            row for row in targets if name in tissue_edit_fields(self._row_tissue_class(row))
        ]
        for row in changed:
            self.set_param(row, name, value)
        text = f"Set {param_label(name)} = {value:g} in {len(changed)} structure(s)"
        skipped = len(targets) - len(changed)
        if skipped:
            text += f"; {skipped} structure(s) whose model lacks it left unchanged"
        self.status_label.setText(text)
        return changed

    def _on_apply_pick(self) -> None:
        pick = current_pick(self.preset_combo, self._picks) if self.preset_combo else None
        if pick is None:
            return
        for row in self._target_rows():
            if pick.kind == "class":
                self.apply_machine_tissue(row, pick.index)
            else:
                self.apply_cell_line_preset(row, pick.index)

    def apply_cell_line_preset(self, row: int, index: int) -> str:
        """
        Fill one row from one of the cell line presets (:attr:`_presets`).

        A preset describes a complete tissue, so all parameters of its model are taken
        over, together with the cell line. A preset of another provenance than the base
        data the selected model reads is adopted as that provenance first. Returns the
        note on where the values came from, which the dialog shows.
        """
        tissue, note = adopt_preset(self._presets[index][1], self._target_provenance)
        self._write_row(row, tissue)
        self.status_label.setText(note)
        return note

    def apply_machine_tissue(self, row: int, index: int) -> None:
        """
        Apply one of the base data's tissue classes (:attr:`_machine_tissues`) to one row.

        The row's tissue adopts the class's model and the parameters identifying it,
        since those are what the RBE model matches against its base data; the parameters
        describing the tissue itself (e.g. `alpha_0` / `beta_0`, which MKM takes per voxel)
        are kept, see :func:`~pyRadPlan.biology.tissues.apply_tissue_class`.
        """
        tissue_class = self._machine_tissues[index]
        tissue, class_name = apply_base_data_class(self._current_tissue(row), tissue_class)
        self._write_row(row, tissue, class_name)
        self.status_label.setText(
            f"Applied the base data class {tissue_class.cell_line or tissue_class.display_name}"
        )

    def match_base_data(self, rows: Optional[list[int]] = None) -> list[int]:
        """
        Apply the closest base data class to rows: the selected ones, or all if none are.

        Rows without a class to compare against are left as they are. Returns the rows
        a class was applied to.
        """
        self.refresh_comparison()
        matched = []
        for row in self._target_rows() if rows is None else rows:
            comparison = self._comparisons.get(row)
            if comparison is None or comparison.tissue_class is None:
                continue
            tissue, class_name = apply_base_data_class(
                self._current_tissue(row), comparison.tissue_class
            )
            self._write_row(row, tissue, class_name)
            matched.append(row)
        if matched:
            self.status_label.setText(
                f"Matched {len(matched)} structure(s) to their closest base data class"
            )
        self.refresh_comparison()
        return matched

    def _current_tissue(self, row: int) -> TissueProperties:
        """Tissue of a row, its last applied one if the entered values do not validate."""
        try:
            return self.tissue_properties(row)
        except (ValidationError, ValueError, TypeError):
            return self._tissue_identities[row]

    def _write_row(self, row: int, tissue: TissueProperties, class_name: Optional[str] = None):
        """Show a tissue in one row, which then carries its cell line and model name."""
        self._writing_row = row
        try:
            self._tissue_identities[row] = tissue
            self.set_tissue_model(row, tissue_model_label(tissue))
            write_tissue(
                self._spins[row],
                self._choices[row],
                tissue,
                class_name,
                radiation=self._radiation_combos[row],
            )
        finally:
            self._writing_row = None
        self._update_reference_tooltips(row)
        self._schedule_comparison()

    def _target_rows(self) -> list[int]:
        """Rows a picker applies to: the selected ones, or all if none are selected."""
        rows = {item.row() for item in self._table.selectedItems()}
        return sorted(rows or range(len(self._spins)))

    def _row_reference_radiation(self, row: int) -> Optional[str]:
        """Return the reference radiation selected for one row, None for the default one."""
        return reference_radiation_value(self._radiation_combos[row])

    def set_reference_radiation(self, row: int, name: str) -> None:
        """Select the reference radiation of one row by name (mainly for scripting / tests)."""
        index = self._radiation_combos[row].findData(name)
        if index < 0:
            raise ValueError(f"Unknown reference radiation {name!r}.")
        self._radiation_combos[row].setCurrentIndex(index)

    def _on_reference_radiation_changed(self, row: int) -> None:
        """Derive one row's reference parameters for its newly selected reference radiation."""
        if self._writing_row == row:
            return
        self._derive_row(row, force=False)
        self._update_reference_tooltips(row)
        self._schedule_comparison()

    def _derive_row(self, row: int, force: bool) -> dict[str, float]:
        return apply_derived_reference(
            self._spins[row],
            self._row_tissue_class(row),
            self._model_combos[row].currentText(),
            self._row_reference_radiation(row),
            force=force,
        )

    def _on_model_param_changed(self, row: int) -> None:
        """Derive the reference parameters of one row live from its model parameters."""
        if self._writing_row != row:
            self._derive_row(row, force=False)

    def derive_reference_params(self, row: int) -> dict[str, float]:
        """
        Derive the reference parameters of one row from its model parameters.

        Replaces an explicit override, since the button is an explicit request to do so;
        the values are marked as derived. Does nothing for a row whose model takes its
        reference parameters as given, i.e. any model other than MKM and mMKM.
        """
        return self._derive_row(row, force=True)

    def _on_derive_reference_params(self) -> None:
        derived = {row: self.derive_reference_params(row) for row in self._target_rows()}
        if not any(derived.values()):
            QMessageBox.information(
                self,
                "Nothing to derive",
                "No selected structure is an MKM or mMKM tissue with its model parameters "
                "set.\n\n" + DERIVE_REFERENCE_TOOLTIP,
            )

    def _update_reference_tooltips(self, row: int) -> None:
        """Explain in the reference cells of one row how they come about for its model."""
        note = reference_group_note(
            self._row_tissue_class(row),
            self._model_combos[row].currentText(),
            self._row_reference_radiation(row),
        )
        for name in REFERENCE_FIELDS:
            spin = self._spins[row].get(name)
            if spin is not None:
                set_base_tooltip(spin, f"{param_field(name).description}\n{note}")

    def _param_column(self, name: str) -> int:
        """Table column showing a tissue parameter."""
        return 3 + self._fields.index(name)

    def _row_tissue_class(self, row: int):
        """Tissue properties class selected for one row."""
        return tissue_class(self._model_combos[row].currentText())

    def _update_row(self, row: int) -> None:
        """
        Show only the parameters belonging to the row's tissue model.

        The editors of the other parameters are hidden rather than emptied, so their
        value is still there when the model is switched back; their cell stays empty
        because each editor sits in a container of its own, see :func:`cell_container`.
        """
        tissue_cls = self._row_tissue_class(row)
        applicable = tissue_edit_fields(tissue_cls)
        for editors in (self._choices[row], self._spins[row]):
            for name, editor in editors.items():
                editor.setVisible(name in applicable)
        self._radiation_combos[row].setVisible(bool(reference_radiation_fields(tissue_cls)))
        self._update_reference_tooltips(row)

    def _update_columns(self) -> None:
        """Hide the parameter columns no row makes use of."""
        used: set[str] = set()
        for row in range(len(self._model_combos)):
            tissue_cls = self._row_tissue_class(row)
            used.update(tissue_edit_fields(tissue_cls) + reference_radiation_fields(tissue_cls))
        for name in self._fields:
            self._table.setColumnHidden(self._param_column(name), name not in used)
        self._table.resizeColumnsToContents()
        self._update_bulk_params(used)

    def _update_derive_button(self) -> None:
        """
        Enable the derive button only while a structure uses a model deriving its reference.

        Only MKM and mMKM derive their reference parameters from their model parameters, so
        with no such structure in the table the button has nothing it could apply to; its
        tooltip then says so instead of describing what it would do.
        """
        derivable = any(
            derives_reference_params(self._row_tissue_class(row))
            for row in range(len(self._model_combos))
        )
        self.derive_button.setEnabled(derivable)
        self.derive_button.setToolTip(
            DERIVE_BUTTON_TOOLTIP if derivable else DERIVE_BUTTON_DISABLED_TOOLTIP
        )

    def _on_model_changed(self, row: int) -> None:
        self._update_row(row)
        self._update_columns()
        self._update_derive_button()
        self._on_model_param_changed(row)
        self._schedule_comparison()

    def _schedule_comparison(self, *_args) -> None:
        """Compare the table with the base data once the edits come to a rest."""
        self._comparison_timer.start()

    def refresh_comparison(self) -> None:
        """
        Compare every row with the base data of the selected model and highlight it.

        Parameters differing from the row's closest tissue class are tinted, with the
        declared value as tooltip, as is the model of a row of another provenance (softly
        if the parameters match, as that is a warning only); parameters the class leaves
        undeclared are never. The structure cell names the matching or closest class and is
        tinted while the row matches none (softly while it only warns). Without base data
        to compare against, all highlighting is cleared, and "Relabel as ..." is shown only
        while a row offers it.
        """
        self._comparison_timer.stop()
        names = [voi.name for voi in self._cst.vois]
        tissues = [self._current_tissue(row) for row in range(len(names))]
        comparisons = compare_tissues_quietly(
            self._machine, self._bio_model, list(zip(names, tissues)), self._engine
        )
        self._comparisons = dict(enumerate(comparisons)) if len(comparisons) == len(names) else {}
        for row in range(len(names)):
            self._show_comparison(row, self._comparisons.get(row))
        self.match_button.setEnabled(
            any(c.tissue_class is not None for c in self._comparisons.values())
        )
        text = relabel_text(provenance_fixes(relabel_issues(self._comparisons.values())))
        self.relabel_button.setText(text or "Relabel")
        self.relabel_button.setVisible(text is not None)

    def _show_comparison(self, row: int, comparison: Optional[TissueComparison]) -> None:
        """Highlight one row after its comparison with the base data (None clears it)."""
        show_comparison(
            (self._choices[row], self._spins[row]), self._model_combos[row], comparison
        )
        item = self._table.item(row, 0)
        compared = comparison is not None and comparison.kernel_set is not None
        brush = QBrush()
        if compared and not comparison.matched:
            brush = ISSUE_BRUSH
        elif compared and comparison.provenance_mismatch:
            brush = WARNING_BRUSH
        item.setBackground(brush)
        item.setToolTip(comparison_row_note(comparison) if compared else "")

    def comparison(self, row: int) -> Optional[TissueComparison]:
        """Last comparison of one row with the base data, None if there is none."""
        return self._comparisons.get(row)

    def set_values(self, row: int, alpha_x: float, beta_x: float) -> None:
        """Set the reference parameters of one row (mainly for scripting / tests)."""
        self._spins[row]["alpha_x"].setValue(alpha_x)
        self._spins[row]["beta_x"].setValue(beta_x)

    def set_tissue_model(self, row: int, label: str) -> None:
        """Select the tissue model of one row (mainly for scripting / tests)."""
        self._model_combos[row].setCurrentText(label)

    def set_param(self, row: int, name: str, value: float) -> None:
        """Set one numeric tissue parameter of one row (mainly for scripting / tests)."""
        self._spins[row][name].setValue(value)

    def set_choice(self, row: int, name: str, value: object) -> None:
        """Select one choice parameter of one row (mainly for scripting / tests)."""
        self._choices[row][name].setCurrentText(str(value))

    def _row_values(self, row: int) -> dict[str, object]:
        """Parameters currently entered in one row, by field name."""
        values: dict[str, object] = {
            name: spin_value(spin) for name, spin in self._spins[row].items()
        }
        values.update({name: combo_value(combo) for name, combo in self._choices[row].items()})
        if reference_radiation_fields(self._row_tissue_class(row)):
            radiation = self._row_reference_radiation(row)
            if radiation:
                values[REFERENCE_RADIATION_FIELD] = radiation
        return values

    def tissue_properties(self, row: int):
        """Tissue properties described by the current state of one row."""
        label = self._model_combos[row].currentText()
        return build_tissue_properties(
            tissue_class(label),
            self._row_values(row),
            self._tissue_identities[row],
            reference_sources=reference_source_of(self._spins[row]),
            model=model_label_name(label, self._tissue_identities[row]),
        )

    def check_tissues(self, tissues: list) -> list:
        """
        Check tissues (one per row) against the biological model and machine.

        Highlights the structure cell of each row with an issue, with the issues and the
        closest tissue class as its tooltip (softly if it only has warnings), and returns
        the issues of the structures, warnings included. Empty without a model or machine
        summary to check against.
        """
        names = [voi.name for voi in self._cst.vois]
        issues = [
            issue
            for issue in tissue_issues_quietly(
                self._machine, self._bio_model, list(zip(names, tissues)), self._engine
            )
            if issue.structure is not None
        ]
        for row, name in enumerate(names):
            item = self._table.item(row, 0)
            own = [issue for issue in issues if issue.structure == name]
            brush = QBrush()
            if tissue_errors(own):
                brush = ISSUE_BRUSH
            elif own:
                brush = WARNING_BRUSH
            item.setBackground(brush)
            item.setToolTip("\n".join(tissue_issue_text(issue) for issue in own))
        return issues

    def relabel(self, rows: Optional[list[int]] = None) -> list[int]:
        """
        Relabel rows matching the base data under another provenance with the base data's.

        Applies to `rows`, else the selected rows, or all rows if none are selected; only
        rows whose comparison offers it (:attr:`TissueComparison.fix_provenance`) change.
        Every parameter is kept, see :func:`~pyRadPlan.biology.models.apply_provenance_fixes`.
        Relabelling is optional, since such rows are evaluated with the base data's model
        either way. Returns the relabelled rows.
        """
        self.refresh_comparison()
        targets = self._target_rows() if rows is None else rows
        names = [voi.name for voi in self._cst.vois]
        issues = relabel_issues(self._comparisons.get(row) for row in targets)
        tissues = [self._current_tissue(row) for row in range(len(names))]
        fixed = apply_provenance_fixes(dict(zip(names, tissues)), issues)
        changed = [row for row, name in enumerate(names) if fixed[name] is not tissues[row]]
        for row in changed:
            self._write_row(row, fixed[names[row]])
        if changed:
            self.status_label.setText(
                f"Relabelled {len(changed)} structure(s) with the base data's model"
            )
        self.refresh_comparison()
        return changed

    def accept(self) -> None:
        """
        Validate the edited values and write them to the structure set before closing.

        Edited tissues the biological model cannot be evaluated for (errors) are only
        written if the user chooses to apply them anyway; otherwise the dialog stays open
        for editing. Warnings, e.g. a model label differing from the base data's, never ask.
        """
        try:
            tissues = [self.tissue_properties(row) for row in range(len(self._cst.vois))]
        except (ValidationError, ValueError, TypeError) as exc:
            self.changed = False
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        edited = any(t != voi.tissue_properties for voi, t in zip(self._cst.vois, tissues))
        errors = tissue_errors(self.check_tissues(tissues))
        if edited and errors and ask_tissue_issues(self, errors) != APPLY_ANYWAY:
            return
        try:
            vois = []
            changed = False
            for voi, tissue in zip(self._cst.vois, tissues):
                if tissue == voi.tissue_properties:
                    vois.append(voi)
                    continue
                # VOIs mirror alpha_x/beta_x of their tissue properties and reject
                # assignments leaving the two inconsistent, hence the recreation.
                data = dict(voi)
                data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
                new_voi = create_voi(data)
                if "color" not in voi.model_fields_set:
                    # Recreating marks the copied color as set; keep it reading as auto-assigned
                    new_voi.model_fields_set.discard("color")
                vois.append(new_voi)
                changed = True
            if changed:
                self._cst.vois = vois
            self.changed = changed
        except (ValidationError, ValueError, TypeError) as exc:
            self.changed = False
            QMessageBox.warning(self, "Invalid value", str(exc))
            return
        super().accept()

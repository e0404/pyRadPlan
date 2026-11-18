"""Shared helpers for editing tissue properties in the GUI.

The offered parameters are derived from the tissue properties classes themselves, so a
parameter added to one of them shows up in the editors without further changes here.
"""

from __future__ import annotations

import logging
import math
from typing import (
    Any,
    Iterable,
    Mapping,
    Optional,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QWidget,
)
from pyRadPlan.biology.models import TissueComparison, TissueIssue
from pyRadPlan.biology.tissues import (
    DEFAULT_ALPHA_X,
    DEFAULT_BETA_X,
    FIXED_REFERENCE_SOURCES,
    get_reference_radiation,
    provenance_label,
    REFERENCE_FIELDS,
    registered_reference_radiations,
    TissueProperties,
)

from pyRadPlan.biology._editing import (  # noqa: F401 - re-exported for the GUI
    SELECTION_PARAMETERS,
    TISSUE_MODELS,
    TISSUE_FIELD_LABELS,
    UNSET_TEXT,
    MODEL_GROUP_TITLE,
    REFERENCE_GROUP_TITLE,
    DERIVE_REFERENCE_TEXT,
    DERIVE_REFERENCE_TOOLTIP,
    tissue_param_fields,
    _numeric_fields,
    model_param_fields,
    reference_param_fields,
    derives_reference_params,
    reference_radiation_text,
    REFERENCE_RADIATION_FIELD,
    REFERENCE_RADIATION_LABEL,
    reference_radiation_fields,
    reference_radiation_label,
    reference_radiation_tooltip,
    REFERENCE_RADIATION_TOOLTIP,
    reference_group_note,
    tissue_choice_fields,
    tissue_edit_fields,
    _all_fields_of,
    all_tissue_param_fields,
    all_tissue_choice_fields,
    param_field,
    param_choices,
    is_optional_param,
    param_label,
    param_unit,
    tissue_model_label,
    model_label_name,
    tissue_class,
    machine_tissue_classes,
    bio_model_quietly,
    model_tissue_classes,
    available_tissue_classes,
    _MACHINE_SUMMARY_CACHE,
    _machine_file_stamp,
    load_machine_quietly,
    machine_summary,
    engine_capability_quietly,
    dose_engine_quietly,
    plan_dose_engine,
    bio_options_quietly,
    selection_parameters,
    matching_bio_option,
    selected_bio_option,
    tissue_issues_quietly,
    tissue_errors,
    tissue_warnings,
    tissue_issue_text,
    tissue_issues_text,
    compare_tissues_quietly,
    _format_value,
    mismatch_note,
    comparison_row_note,
    PROVENANCE_WARNING_NOTE,
    provenance_mismatch_note,
    provenance_field_notes,
    relabel_issues,
    relabel_text,
    RELABEL_TOOLTIP,
    APPLY_ANYWAY_TEXT,
    KEEP_EDITING_TEXT,
    APPLY_ANYWAY,
    KEEP_EDITING,
    MATCH_BASE_DATA_TEXT,
    MATCH_BASE_DATA_TOOLTIP,
    provenance_fixes,
    param_summary,
    tissue_summary,
    cell_line_presets,
    tissue_preset,
    published_note,
    preset_label,
    preset_tooltip,
    class_label,
    EXACT_MARK,
    TissuePick,
    _provenance_rank,
    ranked_tissue_picks,
    picker_title,
    target_provenance,
    adopt_preset,
    apply_base_data_class,
    DEFAULT_NOTE,
    derived_note,
    base_data_note,
    reference_sources,
    build_tissue_properties,
    _KEPT_SETTINGS,
    _same_value,
)

logger = logging.getLogger(__name__)

# Spin box properties holding the exact value a parameter was set to and the rounded
# value the box shows for it, see set_param_value.
_EXACT_VALUE = "tissue_exact_value"


_SHOWN_VALUE = "tissue_shown_value"


def make_reference_radiation_combo(current: Optional[str] = None) -> QComboBox:
    """
    Build the selector of the registered reference radiations, selecting `current`.

    Each entry shows the radiation's label and its source in short, with its value and
    full citation as tooltip; its item data is the registered name (see
    :func:`reference_radiation_value`). A `current` name no longer registered is added
    as it is, so an editor never changes a tissue by merely showing it.
    """
    combo = QComboBox()
    for radiation in registered_reference_radiations():
        combo.addItem(reference_radiation_label(radiation), radiation.name)
        combo.setItemData(
            combo.count() - 1,
            reference_radiation_tooltip(radiation),
            Qt.ItemDataRole.ToolTipRole,
        )
    combo.setProperty(_PARAM_NAME, REFERENCE_RADIATION_FIELD)
    combo.setProperty(_BASE_TOOLTIP, REFERENCE_RADIATION_TOOLTIP)
    set_reference_radiation(combo, current)
    combo.currentIndexChanged.connect(lambda _index, box=combo: _show_radiation_tooltip(box))
    return combo


def set_reference_radiation(combo: QComboBox, name: Optional[str]) -> None:
    """Select a reference radiation by name in a selector, the default one for None."""
    name = name or get_reference_radiation().name
    index = combo.findData(name)
    if index < 0:
        combo.addItem(str(name), name)
        index = combo.count() - 1
    combo.setCurrentIndex(index)
    _show_radiation_tooltip(combo)


def reference_radiation_value(combo: QComboBox) -> Optional[str]:
    """Name of the reference radiation selected in a selector, None if there is none."""
    name = combo.currentData()
    return str(name) if name else None


def _show_radiation_tooltip(combo: QComboBox) -> None:
    item = combo.itemData(combo.currentIndex(), Qt.ItemDataRole.ToolTipRole)
    set_base_tooltip(combo, "\n".join(t for t in (REFERENCE_RADIATION_TOOLTIP, item) if t))


def show_comparison(
    editors: Iterable[Mapping[str, QWidget]],
    model_editor: QWidget,
    comparison: Optional[TissueComparison],
) -> None:
    """
    Tint the editors of one tissue after its comparison with the base data (None clears).

    Parameters differing from the (closest) class get :data:`MISMATCH_COLOR`; a provenance
    label differing although the parameters match (a warning) tints the model editor and
    the differing labels (the LEM version) with :data:`PROVENANCE_WARNING_COLOR`, while a
    parameter class of another provenance (an error) tints the model editor as a mismatch.
    """
    notes: dict[str, str] = {}
    if comparison is not None:
        notes = {p.name: mismatch_note(p) for p in comparison.parameters if not p.matches}
    labels = provenance_field_notes(comparison)
    for group in editors:
        for name, editor in group.items():
            if name in notes:
                set_editor_mismatch(editor, notes[name])
            else:
                set_editor_mismatch(editor, labels.get(name), warning=True)
    differs = comparison is not None and comparison.provenance_mismatch
    set_editor_mismatch(
        model_editor,
        provenance_mismatch_note(comparison) if differs else None,
        warning=differs and comparison.matched,
    )


def ask_tissue_issues(parent: Optional[QWidget], issues: list[TissueIssue]) -> str:
    """
    Ask whether to accept tissue parameters the biological model cannot be evaluated for.

    Only for errors; warnings never prompt (pass the errors alone, see
    :func:`tissue_errors`). Returns :data:`APPLY_ANYWAY` for "Apply anyway" and
    :data:`KEEP_EDITING`, the default, for "Keep editing".
    """
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Icon.Warning)
    box.setWindowTitle("Tissue parameters")
    box.setText("The selected biological model cannot be evaluated for these tissues:")
    box.setInformativeText(
        tissue_issues_text(issues) + "\n\nUntil they match the base data, dose calculation "
        "offers to calculate the physical dose only."
    )
    apply_button = box.addButton(APPLY_ANYWAY_TEXT, QMessageBox.ButtonRole.AcceptRole)
    keep_button = box.addButton(KEEP_EDITING_TEXT, QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(keep_button)
    box.setEscapeButton(keep_button)
    box.exec()
    return APPLY_ANYWAY if box.clickedButton() is apply_button else KEEP_EDITING


def _add_picker_header(combo: QComboBox, text: str) -> None:
    combo.addItem(text)
    item = combo.model().item(combo.count() - 1)
    item.setEnabled(False)
    font = QFont(combo.font())
    font.setItalic(True)
    item.setFont(font)


def _add_pick(combo: QComboBox, pick: TissuePick, position: int) -> None:
    combo.addItem(pick.label, position)
    if pick.tooltip:
        combo.setItemData(combo.count() - 1, pick.tooltip, Qt.ItemDataRole.ToolTipRole)


def fill_tissue_picker(combo: QComboBox, picks: list[TissuePick], title: str = "") -> None:
    """
    Fill a combo box with the picks of :func:`ranked_tissue_picks`.

    The preferred picks come first, under the header `title` if given, those matching the
    base data exactly in bold. A separator divides them from the others, which are listed
    under a header per provenance. Headers are disabled; the item data of a pick is its
    position in `picks`, see :func:`current_pick`, and its tooltip that of the pick.
    """
    combo.clear()
    preferred = [position for position, pick in enumerate(picks) if pick.preferred]
    if preferred and title:
        _add_picker_header(combo, title)
    for position in preferred:
        _add_pick(combo, picks[position], position)
        if picks[position].exact:
            font = QFont(combo.font())
            font.setBold(True)
            combo.setItemData(combo.count() - 1, font, Qt.ItemDataRole.FontRole)

    group: Optional[str] = None
    for position, pick in enumerate(picks):
        if pick.preferred:
            continue
        if group is None and preferred:
            combo.insertSeparator(combo.count())
        if pick.provenance != group:
            group = pick.provenance
            prefix = "Other presets" if preferred else "Presets"
            _add_picker_header(combo, f"{prefix} · {provenance_label(group)}")
        _add_pick(combo, pick, position)

    first = next((i for i in range(combo.count()) if isinstance(combo.itemData(i), int)), -1)
    combo.setCurrentIndex(first)


def current_pick(combo: QComboBox, picks: list[TissuePick]) -> Optional[TissuePick]:
    """Pick selected in a tissue picker, None for a header or an empty picker."""
    position = combo.currentData()
    return picks[position] if isinstance(position, int) else None


#: Background of an editor whose value differs from the base data's.
MISMATCH_COLOR = "rgba(230, 126, 34, 90)"


#: Softer background of an editor whose provenance label differs from the base data's,
#: which is a warning only (the model is evaluated with the base data's provenance).
PROVENANCE_WARNING_COLOR = "rgba(241, 196, 15, 70)"


# Editor properties: the parameter shown, the tooltip describing it, where a reference
# value came from (with a note saying so) and how it differs from the base data.
_PARAM_NAME = "tissue_param_name"


_BASE_TOOLTIP = "tissue_base_tooltip"


_SOURCE = "tissue_reference_source"


_SOURCE_NOTE = "tissue_reference_source_note"


_MISMATCH = "tissue_mismatch_note"


_MISMATCH_SOFT = "tissue_mismatch_is_warning"


def write_tissue(
    spins: Mapping[str, QDoubleSpinBox],
    combos: Mapping[str, QComboBox],
    tissue: TissueProperties,
    class_name: Optional[str] = None,
    radiation: Optional[QComboBox] = None,
) -> None:
    """
    Show every parameter of a tissue's model in the editors, with its reference sources.

    Editors of other parameters are left as they are; an unset (NaN) value is written as
    the "unset" placeholder of an optional parameter. See :func:`reference_sources` for
    `class_name`. The reference radiation selector `radiation`, if given, selects the
    tissue's reference radiation (if it has one).
    """
    fields = tissue_edit_fields(type(tissue))
    if radiation is not None and reference_radiation_fields(type(tissue)):
        set_reference_radiation(radiation, getattr(tissue, REFERENCE_RADIATION_FIELD))
    sources = reference_sources(tissue, class_name)
    for name in fields:
        spin = spins.get(name)
        if spin is not None:
            set_param_value(spin, getattr(tissue, name), *sources.get(name, ("explicit", "")))
    apply_tissue_choices(combos, tissue, fields)


def apply_tissue_values(
    spins: Mapping[str, QDoubleSpinBox], tissue: TissueProperties, fields: Iterable[str]
) -> None:
    """
    Copy the given parameters of a tissue into the spin boxes showing them.

    Parameters the spin boxes do not cover are skipped, an unset (NaN) value is written
    as the "unset" placeholder of an optional parameter.
    """
    sources = reference_sources(tissue)
    for name in fields:
        spin = spins.get(name)
        if spin is None:
            continue
        set_param_value(
            spin, getattr(tissue, name, float("nan")), *sources.get(name, ("explicit", ""))
        )


def apply_tissue_choices(
    combos: Mapping[str, QComboBox], tissue: TissueProperties, fields: Iterable[str]
) -> None:
    """
    Select the given choice parameters of a tissue in the combo boxes showing them.

    Parameters the combo boxes do not cover, or the tissue does not carry, are skipped.
    """
    for name in fields:
        combo = combos.get(name)
        if combo is not None and hasattr(tissue, name):
            combo.setCurrentText(str(getattr(tissue, name)))


def make_param_spin(
    name: str,
    value: Any,
    with_unit: bool = True,
    source: str = "explicit",
    note: str = "",
) -> QDoubleSpinBox:
    """
    Build the spin box for one tissue parameter, showing `value` (NaN meaning unset).

    Parameters
    ----------
    name : str
        Name of the tissue parameter.
    value : float
        Value to show; NaN selects the "unset" placeholder of an optional parameter.
    with_unit : bool
        Whether to append the parameter's unit to the box, off for table cells that
        carry the unit in their column header.
    source, note : str
        Where a reference parameter's value came from, see :func:`set_param_value`.
    """
    field = param_field(name)
    spin = QDoubleSpinBox()
    spin.setDecimals(4)
    spin.setRange(0.0, 100.0)
    spin.setSingleStep(0.01)
    if with_unit and param_unit(name):
        spin.setSuffix(f" {param_unit(name)}")
    spin.setProperty(_PARAM_NAME, name)
    spin.setProperty(_BASE_TOOLTIP, field.description or "")
    if is_optional_param(field):
        # Parameters defaulting to NaN may be left unset, shown at the minimum
        spin.setSpecialValueText(UNSET_TEXT)
    set_param_value(spin, value, source, note)
    spin.valueChanged.connect(lambda _value, box=spin: refresh_editor_decoration(box))
    return spin


def set_param_value(
    spin: QDoubleSpinBox, value: Any, source: str = "explicit", note: str = ""
) -> None:
    """
    Show `value` (NaN or None meaning unset) in a parameter spin box.

    The box rounds to the decimals it shows, so the exact value is kept alongside and
    returned by :func:`spin_value` for as long as the box still shows it. Values the user
    did not touch thus survive an editor unchanged, rather than being rounded.

    `source` (one of :data:`~pyRadPlan.biology.tissues.REFERENCE_SOURCES`) and `note`
    say where the value came from; they hold as long as the box shows it, editing the value
    makes it explicit, see :func:`param_source`.
    """
    exact = float("nan") if value is None else float(value)
    spin.setProperty(_SOURCE, source)
    spin.setProperty(_SOURCE_NOTE, note)
    spin.setValue(0.0 if math.isnan(exact) else exact)
    spin.setProperty(_EXACT_VALUE, exact)
    spin.setProperty(_SHOWN_VALUE, spin.value())
    refresh_editor_decoration(spin)


def _shows_set_value(spin: QDoubleSpinBox) -> bool:
    return spin.property(_SHOWN_VALUE) == spin.value() and spin.property(_EXACT_VALUE) is not None


def spin_value(spin: QDoubleSpinBox) -> float:
    """
    Value of a parameter spin box, NaN when it shows its "unset" placeholder.

    A box still showing the value it was set to by :func:`set_param_value` returns that
    value exactly, unrounded.
    """
    if _shows_set_value(spin):
        return float(spin.property(_EXACT_VALUE))
    if spin.specialValueText() and spin.value() == spin.minimum():
        return float("nan")
    return spin.value()


def param_source(spin: QDoubleSpinBox) -> tuple[str, str]:
    """
    Where the value of a parameter spin box came from, as (source, note).

    The source given to :func:`set_param_value` while the box still shows that value,
    ``("explicit", "")`` once it was edited.
    """
    if _shows_set_value(spin) and spin.property(_SOURCE):
        return str(spin.property(_SOURCE)), str(spin.property(_SOURCE_NOTE) or "")
    return "explicit", ""


def reference_source_of(spins: Mapping[str, QDoubleSpinBox]) -> dict[str, str]:
    """
    Return the source of each reference parameter shown in `spins`, by parameter name.

    As a tissue records them (:attr:`~TissueProperties.reference_sources`); empty without
    any reference parameter.
    """
    return {name: param_source(spins[name])[0] for name in REFERENCE_FIELDS if name in spins}


def set_base_tooltip(editor: QWidget, text: str) -> None:
    """Set the tooltip an editor shows besides the notes on its state."""
    editor.setProperty(_BASE_TOOLTIP, text)
    refresh_editor_decoration(editor)


def set_editor_mismatch(editor: QWidget, note: Optional[str], warning: bool = False) -> None:
    """
    Tint a parameter editor whose value differs from the base data, `note` saying how.

    With `warning`, the difference does not keep the model from being evaluated (a
    differing provenance label) and the editor gets the softer
    :data:`PROVENANCE_WARNING_COLOR` instead of :data:`MISMATCH_COLOR`.
    """
    editor.setProperty(_MISMATCH, note or "")
    editor.setProperty(_MISMATCH_SOFT, bool(note) and warning)
    refresh_editor_decoration(editor)


def editor_mismatch(editor: QWidget) -> str:
    """How a parameter editor differs from the base data, empty if it does not."""
    return str(editor.property(_MISMATCH) or "")


def editor_mismatch_is_warning(editor: QWidget) -> bool:
    """Whether an editor's difference from the base data is a warning only."""
    return bool(editor_mismatch(editor)) and bool(editor.property(_MISMATCH_SOFT))


def refresh_editor_decoration(editor: QWidget) -> None:
    """
    Style a parameter editor after its state, with a tooltip explaining it.

    A value differing from the base data is tinted (more softly for a warning); a reference
    value that is not given (estimated, from base data or the default) shows in grey
    italics. A published value shows as given, its publication in the tooltip.
    """
    styles = []
    lines = [str(editor.property(_BASE_TOOLTIP) or "")]
    mismatch = editor_mismatch(editor)
    if mismatch:
        color = PROVENANCE_WARNING_COLOR if editor_mismatch_is_warning(editor) else MISMATCH_COLOR
        styles.append(f"background-color: {color};")
        lines.append(mismatch)
    if isinstance(editor, QDoubleSpinBox) and editor.property(_PARAM_NAME) in REFERENCE_FIELDS:
        source, note = param_source(editor)
        if source not in FIXED_REFERENCE_SOURCES:
            styles.append("color: gray; font-style: italic;")
        if source != "explicit":
            lines.append(f"Value {note or source}; editing it makes it explicit.")
    editor.setStyleSheet(f"{type(editor).__name__} {{ {' '.join(styles)} }}" if styles else "")
    editor.setToolTip("\n".join(line for line in lines if line))


def make_derive_button() -> QPushButton:
    """Build the button deriving the reference parameters from the model ones (MKM)."""
    button = QPushButton(DERIVE_REFERENCE_TEXT)
    button.setToolTip(DERIVE_REFERENCE_TOOLTIP)
    return button


def apply_derived_reference(
    spins: Mapping[str, QDoubleSpinBox],
    tissue_cls: type[TissueProperties],
    model: Optional[str] = None,
    reference_radiation: Optional[str] = None,
    force: bool = False,
) -> dict[str, float]:
    """
    Derive the reference parameters from the model parameters entered in `spins`.

    Called live while the model parameters are edited, it updates only reference values
    that are neither explicit nor published, the way the tissue itself does; with `force`
    (the "Derive from model" button) it replaces those as well. A derived value is written as
    ``"estimated"`` (see :func:`param_source`); one the entered parameters no longer
    derive falls back to the default rather than staying stale. Does nothing for a model
    taking its reference parameters as given. An explicit value the entered parameters do
    not derive is kept even with `force`.

    Parameters
    ----------
    spins : Mapping[str, QDoubleSpinBox]
        Spin boxes of the tissue parameters, by name.
    tissue_cls : type[TissueProperties]
        Tissue class whose conversion to use.
    model : str, optional
        Provenance label of the tissue (MKM or mMKM), choosing the plain or the
        saturation-corrected specific energy.
    reference_radiation : str, optional
        Reference radiation of the tissue; the default one if None.
    force : bool
        Whether to replace explicit values too.

    Returns
    -------
    dict[str, float]
        The derived values written.
    """
    if not derives_reference_params(tissue_cls):
        return {}
    names = [
        name
        for name in REFERENCE_FIELDS
        if name in spins and (force or param_source(spins[name])[0] not in FIXED_REFERENCE_SOURCES)
    ]
    if not names:
        return {}
    values: dict[str, Any] = {name: spin_value(spin) for name, spin in spins.items()}
    if model:
        values["model"] = model
    if reference_radiation:
        values["reference_radiation"] = reference_radiation
    derived = tissue_cls.derive_reference(values)
    written: dict[str, float] = {}
    defaults = {"alpha_x": DEFAULT_ALPHA_X, "beta_x": DEFAULT_BETA_X}
    for name in names:
        spin = spins[name]
        if name in derived:
            note = derived_note(name, model, reference_radiation)
            if param_source(spin) != ("estimated", note) or spin_value(spin) != derived[name]:
                set_param_value(spin, derived[name], "estimated", note)
            written[name] = derived[name]
        elif param_source(spin)[0] == "estimated":
            set_param_value(spin, defaults[name], "default", DEFAULT_NOTE)
    return written


def make_param_combo(name: str, value: Any) -> QComboBox:
    """
    Build the combo box for one choice parameter, showing `value`.

    The choices are the values the parameter's ``Literal`` type allows, offered by their
    string form; a `value` that is not among them leaves the first choice selected.
    """
    field = param_field(name)
    combo = QComboBox()
    combo.addItems([str(choice) for choice in param_choices(name)])
    combo.setProperty(_PARAM_NAME, name)
    combo.setProperty(_BASE_TOOLTIP, field.description or "")
    refresh_editor_decoration(combo)
    combo.setCurrentText(str(value))
    return combo


def combo_value(combo: QComboBox) -> str:
    """Choice selected in a parameter combo box."""
    return combo.currentText()


def cell_container(editor: QWidget) -> QWidget:
    """
    Wrap an editor for use as a table cell widget, so that hiding it empties the cell.

    A table lays its cell widgets out again whenever a column is shown or resized and
    shows every one of them, which would undo hiding an editor whose parameter does not
    apply to its row. Only the container is shown that way, leaving an editor hidden
    inside it hidden - and its cell empty.
    """
    container = QWidget()
    layout = QHBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(editor)
    return container

"""Shared helpers for editing tissue properties in the GUI.

The offered parameters are derived from the tissue properties classes themselves, so a
parameter added to one of them shows up in the editors without further changes here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Optional, get_args, get_origin

from pydantic.fields import FieldInfo
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

from pyRadPlan.biology.models import (
    BioMachineSummary,
    BioModelOption,
    BiologicalModel,
    EngineBioCapability,
    ParameterComparison,
    TabulatedRBEModel,
    TissueComparison,
    TissueIssue,
    bio_model_options,
    create_bio_model,
    engine_bio_capability,
)
from pyRadPlan.core import base_data_generation
from pyRadPlan.dose.engines import get_engine_class
from pyRadPlan.machines import find_machine_file, load_from_name
from pyRadPlan.biology.tissues import (
    DEFAULT_ALPHA_X,
    DEFAULT_BETA_X,
    FIXED_REFERENCE_SOURCES,
    REFERENCE_FIELDS,
    CellLinePreset,
    LEMTissueProperties,
    MKMTissueProperties,
    ReferenceRadiation,
    TissueProperties,
    adopt_provenance,
    apply_tissue_class,
    declared_reference_parameters,
    get_cell_line_preset,
    get_provenance,
    get_reference_radiation,
    provenance_label,
    registered_cell_line_presets,
    registered_provenances,
    registered_reference_radiations,
    tissue_provenance,
)

logger = logging.getLogger(__name__)

#: Biological model parameters choosing the base data a model reads; the model dropdown
#: sets them from the selected option.
SELECTION_PARAMETERS: tuple[str, ...] = (
    "data_model",
    "kernel_set",
    "quantity_variants",
    "quantity_table_name",
)

# Tissue models offered in the editors, in display order. MKM and mMKM share their
# parameters but are distinct provenances, told apart by the tissue's model name.
TISSUE_MODELS: tuple[tuple[str, type[TissueProperties]], ...] = (
    ("Generic", TissueProperties),
    ("LEM", LEMTissueProperties),
    ("MKM", MKMTissueProperties),
    ("mMKM", MKMTissueProperties),
)

# Display label and unit per tissue parameter. Parameters missing here fall back to
# their field name without a unit, so new tissue parameters still show up.
TISSUE_FIELD_LABELS: dict[str, tuple[str, str]] = {
    "version": ("Version", ""),
    "alpha_x": ("α_x", "Gy⁻¹"),
    "beta_x": ("β_x", "Gy⁻²"),
    "d_t": ("D_t", "Gy"),
    "r_nucleus": ("r_nucleus", "µm"),
    "r_domain": ("r_domain", "µm"),
    "alpha_0": ("α_0", "Gy⁻¹"),
    "beta_0": ("β_0", "Gy⁻²"),
}

# Shown by the spin box of an optional parameter that is left unset.
UNSET_TEXT = "unset"

# Spin box properties holding the exact value a parameter was set to and the rounded
# value the box shows for it, see set_param_value.
_EXACT_VALUE = "tissue_exact_value"
_SHOWN_VALUE = "tissue_shown_value"

#: Titles of the two groups of tissue parameters in the editors.
MODEL_GROUP_TITLE = "Model parameters"
REFERENCE_GROUP_TITLE = "Reference radiation (for RBE)"

#: Label and tooltip of the button deriving the reference parameters from the model ones.
DERIVE_REFERENCE_TEXT = "Derive from model"
DERIVE_REFERENCE_TOOLTIP = (
    "The reference parameters of MKM / mMKM are not model inputs: they follow from the model "
    "parameters by α_x = α_0 + β_0 · z_1D,ref and β_x = β_0, with z_1D,ref the dose-mean "
    "specific energy the reference radiation deposits in a domain of radius r_domain "
    "(saturation-corrected z*_1D,ref for mMKM, which needs r_nucleus as well). They are "
    "derived live while not overridden; typing a value overrides them, and this button "
    "discards the override and derives them again."
)


def tissue_param_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """
    Numeric parameters of a tissue properties class: model parameters, then reference ones.

    Both groups keep their declaration order, see :func:`model_param_fields` and
    :func:`reference_param_fields`.
    """
    return model_param_fields(tissue_cls) + reference_param_fields(tissue_cls)


def _numeric_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    return [name for name, field in tissue_cls.model_fields.items() if field.annotation is float]


def model_param_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """
    Numeric parameters of a tissue's model, shown under :data:`MODEL_GROUP_TITLE`.

    The reference parameters are left out even where the model takes them as input (LEM):
    they are edited in the reference group.
    """
    return [name for name in _numeric_fields(tissue_cls) if name not in REFERENCE_FIELDS]


def reference_param_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """Return the reference parameters of a tissue class, see :data:`REFERENCE_GROUP_TITLE`."""
    return [name for name in _numeric_fields(tissue_cls) if name in REFERENCE_FIELDS]


def derives_reference_params(tissue_cls: type[TissueProperties]) -> bool:
    """Whether a tissue model derives its reference parameters from its model ones (MKM)."""
    return bool(getattr(tissue_cls, "derives_reference", False))


def reference_radiation_text(name: Optional[str] = None) -> str:
    """Describe a reference radiation, e.g. ``"200 kV X-rays (y_D = 4.2 keV/µm)"``."""
    try:
        return get_reference_radiation(name).describe()
    except KeyError:
        return str(name)


#: Field of a tissue deriving its reference parameters naming the reference radiation.
REFERENCE_RADIATION_FIELD = "reference_radiation"

#: Label of the reference radiation selector in the editors.
REFERENCE_RADIATION_LABEL = "Reference radiation"


def reference_radiation_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """
    Return the settings of a tissue class choosing its reference radiation.

    They are edited by a selector (:func:`make_reference_radiation_combo`).

    Only a model deriving its reference parameters (MKM) refers to a registered reference
    radiation; the others take `alpha_x` / `beta_x` as given.
    """
    if derives_reference_params(tissue_cls) and REFERENCE_RADIATION_FIELD in (
        tissue_cls.model_fields
    ):
        return [REFERENCE_RADIATION_FIELD]
    return []


def reference_radiation_label(radiation: ReferenceRadiation) -> str:
    """Selector entry of a reference radiation, e.g. ``"200 kV X-rays (Kase 2006)"``."""
    return f"{radiation.label} ({radiation.source.label})"


def reference_radiation_tooltip(radiation: ReferenceRadiation) -> str:
    """Tooltip of a reference radiation: its value, full source and notes."""
    lines = [f"{radiation.label}: {radiation.details()}", f"Source: {radiation.source.citation}"]
    if not radiation.source.verified:
        lines.append("The origin of this value has not been confirmed.")
    if radiation.source.note:
        lines.append(radiation.source.note)
    return "\n".join(lines)


#: Tooltip of the reference radiation selector.
REFERENCE_RADIATION_TOOLTIP = (
    "Radiation the RBE is taken relative to. MKM / mMKM derive α_x / β_x for it from the "
    "model parameters, via the dose-mean specific energy z_1D,ref it deposits in a domain; "
    "changing it derives them again unless they were overridden."
)


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


def reference_group_note(
    tissue_cls: type[TissueProperties],
    model: Optional[str] = None,
    reference_radiation: Optional[str] = None,
) -> str:
    """
    Explanation shown with the reference parameters of a tissue model.

    For MKM / mMKM it names the conversion and the reference radiation, for LEM that the
    reference parameters are model inputs as well.
    """
    if derives_reference_params(tissue_cls):
        saturated = MKMTissueProperties.saturation_corrected(model or "MKM")
        z_ref = "z*_1D,ref (saturation-corrected)" if saturated else "z_1D,ref"
        return (
            f"Derived from the model: α_x = α_0 + β_0 · {z_ref}, β_x = β_0, for "
            f"{reference_radiation_text(reference_radiation)}."
        )
    if set(REFERENCE_FIELDS) & set(tissue_cls.model_parameter_fields):
        return "Photon LQ parameters the RBE refers to; the model takes them as inputs too."
    return "LQ parameters of the reference radiation the RBE is taken relative to."


def tissue_choice_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """
    Parameters of a tissue properties class offering a fixed set of choices.

    These are the parameters typed as a ``Literal``, such as the LEM version, which are
    edited by a combo box rather than by a spin box.
    """
    return [
        name
        for name, field in tissue_cls.model_fields.items()
        if get_origin(field.annotation) is Literal
    ]


def tissue_edit_fields(tissue_cls: type[TissueProperties]) -> list[str]:
    """Every parameter of a tissue properties class offered in the editors."""
    return tissue_choice_fields(tissue_cls) + tissue_param_fields(tissue_cls)


def _all_fields_of(kind) -> list[str]:
    """Union of the parameters `kind` reports for every offered model, in display order."""
    fields: list[str] = []
    for _label, tissue_cls in TISSUE_MODELS:
        fields.extend(name for name in kind(tissue_cls) if name not in fields)
    return fields


def all_tissue_param_fields() -> list[str]:
    """
    Union of the numeric parameters of every offered tissue model, in display order.

    The model parameters of all models come first, the reference parameters last.
    """
    return _all_fields_of(model_param_fields) + _all_fields_of(reference_param_fields)


def all_tissue_choice_fields() -> list[str]:
    """Union of the choice parameters of every offered tissue model, in display order."""
    return _all_fields_of(tissue_choice_fields)


def param_field(name: str) -> FieldInfo:
    """Field definition of a tissue parameter, from the first model declaring it."""
    for _label, tissue_cls in TISSUE_MODELS:
        if name in tissue_cls.model_fields:
            return tissue_cls.model_fields[name]
    raise KeyError(f"No offered tissue model declares parameter {name!r}")


def param_choices(name: str) -> tuple[Any, ...]:
    """Values a choice parameter may take, in declaration order."""
    return get_args(param_field(name).annotation)


def is_optional_param(field: FieldInfo) -> bool:
    """Whether a parameter may be left unset, i.e. its own default is NaN."""
    return isinstance(field.default, float) and math.isnan(field.default)


def param_label(name: str) -> str:
    """Display label of a tissue parameter."""
    return TISSUE_FIELD_LABELS.get(name, (name, ""))[0]


def param_unit(name: str) -> str:
    """Return the unit of a tissue parameter, empty if it has none."""
    return TISSUE_FIELD_LABELS.get(name, (name, ""))[1]


def tissue_model_label(tissue: Any) -> str:
    """
    Label of the tissue model of a tissue, or of a tissue properties class.

    The label of the most specific class the tissue is an instance of; of several labels
    sharing that class (MKM, mMKM), the one naming the tissue's model, else the first.
    """
    tissue_cls = tissue if isinstance(tissue, type) else type(tissue)
    candidates = [(label, cls) for label, cls in TISSUE_MODELS if issubclass(tissue_cls, cls)]
    if not candidates:
        return TISSUE_MODELS[0][0]
    most_specific = max((cls for _label, cls in candidates), key=lambda cls: len(cls.__mro__))
    labels = [label for label, cls in candidates if cls is most_specific]
    model = None if isinstance(tissue, type) else getattr(tissue, "model", None)
    return model if model in labels else labels[0]


def model_label_name(label: str, current: Optional[TissueProperties] = None) -> Optional[str]:
    """
    Model name a tissue gets when its model is chosen by `label`, None to keep its own.

    Set for labels naming one provenance of a class shared by several (MKM, mMKM); the
    LEM version has a field of its own and the generic model keeps any name. A `current`
    tissue already shown under `label` keeps its model name, whatever it is.
    """
    if current is not None and tissue_model_label(current) == label:
        return None
    shared = [cls for _other, cls in TISSUE_MODELS if cls is tissue_class(label)]
    return label if len(shared) > 1 else None


def tissue_class(label: str) -> type[TissueProperties]:
    """Tissue properties class offered under a model label."""
    return dict(TISSUE_MODELS)[label]


def machine_tissue_classes(machine: Any) -> list[TissueProperties]:
    """
    Return the tissue classes a machine's base data provides.

    Empty if there is no machine or it carries none. Machines hold them as an array,
    whose truth value must not be tested.
    """
    tissues = getattr(machine, "tissue_properties", None)
    return [] if tissues is None else list(tissues)


def bio_model_quietly(spec: Any, radiation_mode: str = "") -> Optional[BiologicalModel]:
    """
    Build a biological model from a plan's specification, returning None if that fails.

    Used to offer what the selected model contributes to an editor, which is a
    convenience: a model that cannot be built must never keep an editor from opening.
    """
    if spec is None or spec == "none":
        return None
    if isinstance(spec, BiologicalModel):
        return spec
    try:
        return create_bio_model(spec, radiation_mode or None)
    except Exception:  # noqa: BLE001 - convenience only, must never block an editor
        logger.debug("Could not create the biological model %r", spec, exc_info=True)
        return None


def model_tissue_classes(
    machine: Any, bio_model: Any, engine: Any = None
) -> list[TissueProperties]:
    """
    Return the tissue classes a biological model brings along itself.

    A tabulated RBE model carries its own alpha/beta tables, and with them its own tissue
    classes, rather than reading the per-tissue-class kernels of the machine. Every table
    of the model's table folder is scanned, so the editors offer each tissue the model can
    be run for; the classes of the table the model is currently configured for come first.
    Running it for a class of another table needs that table selected as the model's
    ``quantity_table_name`` as well - the tissue alone does not switch the table.

    The tables are dose-averaged over the fragment fluence spectra of the machine, so they
    are only usable with a machine providing them; a machine the model is unavailable for
    therefore contributes none, as does a dose engine not using them. Every other model
    reads the machine's base data and adds nothing here.
    """
    model = bio_model_quietly(bio_model, getattr(machine, "radiation_mode", ""))
    if not isinstance(model, TabulatedRBEModel):
        return []

    capability = engine_capability_quietly(engine)
    if capability is not None and capability.model_reason(model):
        return []
    provided = getattr(machine, "provided_quantities", None)
    quantities = list(provided()) if callable(provided) else []
    available, reason = model.is_available(getattr(machine, "radiation_mode", ""), quantities)
    if not available:
        logger.debug("Not offering the tissue classes of model '%s': %s", model.model, reason)
        return []

    try:
        classes = list(model.table_tissue_classes)
        for tissue in model.available_table_tissue_classes():
            if tissue not in classes:
                classes.append(tissue)
        return classes
    except Exception:  # noqa: BLE001 - convenience only, must never block an editor
        logger.warning(
            "Could not read the tissue classes of model '%s'", model.model, exc_info=True
        )
        return []


def available_tissue_classes(
    machine: Any, bio_model: Any = None, engine: Any = None
) -> list[TissueProperties]:
    """
    Return the tissue classes the base data of a calculation offers.

    If the model reads tissue-dependent base data of the machine with the dose engine (see
    :func:`selected_bio_option`), only the tissue classes of that base data are offered.
    Otherwise the classes of the machine come first, followed by those a biological model
    provides itself (see :func:`model_tissue_classes`); a class offered by both is listed
    once.
    """
    option = selected_bio_option(machine, bio_model, engine)
    if option is not None:
        return list(option.tissue_classes)
    classes = machine_tissue_classes(machine)
    for tissue in model_tissue_classes(machine, bio_model, engine):
        if tissue not in classes:
            classes.append(tissue)
    return classes


#: Machine summary per (radiation mode, machine name), with the file stamp it was read at.
_MACHINE_SUMMARY_CACHE: dict[tuple[str, str], tuple[Any, Optional[BioMachineSummary]]] = {}


def _machine_file_stamp(radiation_mode: str, machine_name: str) -> Any:
    """
    Identity of a machine's base data file: its path, size and modification time.

    Includes :func:`~pyRadPlan.core.base_data_generation`, so a machine file replaced
    through pyRadPlan is read again even if its size and modification time are unchanged.

    None if the file is not found or cannot be stat'ed, which never compares equal to
    itself, so an unreadable machine is retried rather than cached as missing.
    """
    path = find_machine_file(radiation_mode, machine_name)
    try:
        stat = path.stat()
    except (AttributeError, OSError):
        return None
    return (str(path), stat.st_size, stat.st_mtime_ns, base_data_generation())


def load_machine_quietly(radiation_mode: str, machine_name: str) -> Optional[BioMachineSummary]:
    """
    Summarize machine base data for the GUI, None if it cannot be loaded.

    Used to offer the machine's biological models and tissue classes and to check the
    tissues against them, which is a convenience: a machine that cannot be loaded must
    never keep an editor from opening.

    Base data files run to hundreds of megabytes and reading one takes seconds, which an
    editor would otherwise pay every time it opens. Only the kernel-free
    :class:`~pyRadPlan.biology.models.BioMachineSummary` is kept, so the machine itself is not
    held on to, and it is reused as long as the file is unchanged: a file replaced on disk
    is read again, an untouched one only once.
    """
    machine_name = machine_name or "Generic"
    key = (radiation_mode, machine_name)
    stamp = _machine_file_stamp(*key)
    cached = _MACHINE_SUMMARY_CACHE.get(key)
    if cached is not None and stamp is not None and cached[0] == stamp:
        return cached[1]

    try:
        summary = BioMachineSummary.from_machine(load_from_name(*key))
    except Exception:  # noqa: BLE001 - convenience only, must never block an editor
        logger.debug(
            "Could not load base data of machine '%s' for %s",
            machine_name,
            radiation_mode,
            exc_info=True,
        )
        summary = None
    _MACHINE_SUMMARY_CACHE[key] = (stamp, summary)
    return summary


def machine_summary(machine: Any) -> Optional[BioMachineSummary]:
    """Kernel-free summary of a machine (or summary), None if there is none."""
    if machine is None:
        return None
    try:
        return BioMachineSummary.from_machine(machine)
    except Exception:  # noqa: BLE001 - convenience only, must never block an editor
        logger.debug("Could not summarize machine %r", machine, exc_info=True)
        return None


def engine_capability_quietly(engine: Any) -> Optional[EngineBioCapability]:
    """Return what a dose engine supplies to biological models, None if unknown or not given."""
    try:
        return engine_bio_capability(engine)
    except Exception:  # noqa: BLE001 - convenience only, judge by the machine alone
        logger.debug("Could not determine the capability of dose engine %r", engine)
        return None


def dose_engine_quietly(radiation_mode: str, engine: Optional[str] = None) -> Optional[type]:
    """
    Dose engine class a calculation uses, as the dose engine factory chooses it.

    The named engine if it is available for the radiation mode, otherwise the default
    engine of the mode; None if there is none.
    """
    try:
        return get_engine_class(radiation_mode, engine or None)
    except Exception:  # noqa: BLE001 - convenience only, judge by the machine alone
        logger.debug("Could not determine the dose engine for %s", radiation_mode, exc_info=True)
        return None


def plan_dose_engine(pln: Any) -> Optional[type]:
    """Dose engine class a plan is calculated with (its own or the default), None if none."""
    if pln is None:
        return None
    prop_dose_calc = getattr(pln, "prop_dose_calc", None)
    if prop_dose_calc is not None and callable(getattr(prop_dose_calc, "bio_capability", None)):
        return type(prop_dose_calc)
    name = prop_dose_calc.get("engine") if isinstance(prop_dose_calc, Mapping) else None
    return dose_engine_quietly(getattr(pln, "radiation_mode", ""), name)


def bio_options_quietly(
    radiation_mode: str, machine: Any = None, engine: Any = None
) -> list[BioModelOption]:
    """
    Biological models offered for a radiation mode, machine and dose engine.

    The plain formalism list if there is no machine or its options cannot be determined;
    without an engine (or one whose capability cannot be determined), the machine alone
    decides.
    """
    capability = engine_capability_quietly(engine)
    summary = machine_summary(machine)
    if summary is not None:
        try:
            return bio_model_options(radiation_mode, summary, capability)
        except Exception:  # noqa: BLE001 - fall back to the plain formalism list
            logger.warning(
                "Could not determine the biological models of machine '%s'",
                summary.name,
                exc_info=True,
            )
    return bio_model_options(radiation_mode, engine=capability)


def selection_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """Return the set model parameters choosing its base data (:data:`SELECTION_PARAMETERS`)."""
    return {
        key: value
        for key, value in parameters.items()
        if key in SELECTION_PARAMETERS and value is not None and value not in ({}, "")
    }


def matching_bio_option(
    options: Iterable[BioModelOption], bio_model: Any, machine: Any = None
) -> Optional[BioModelOption]:
    """
    Option describing a biological model, None if none does.

    An option matches if it is of the model's formalism and reads the same base data:
    either it carries exactly the model's base data parameters (:data:`SELECTION_PARAMETERS`)
    or, with a machine, the model resolves to the option's kernel set (or table). Without a
    machine, the formalism alone decides.
    """
    summary = machine_summary(machine)
    model = bio_model_quietly(bio_model, summary.radiation_mode if summary else "")
    name = model.model if model is not None else "none"
    candidates = [option for option in options if option.model == name]
    if model is None or not candidates:
        return candidates[0] if candidates else None
    selection = selection_parameters(model.parameters)
    for option in candidates:
        if dict(option.parameters) == selection:
            return option
    if summary is None:
        return candidates[0]
    try:
        kernel_set = model.select_kernel_set(summary)
    except Exception:  # noqa: BLE001 - a model selecting no base data matches no option
        return None
    kernel_set_id = kernel_set.id if kernel_set is not None else None
    return next((option for option in candidates if option.kernel_set == kernel_set_id), None)


def selected_bio_option(
    machine: Any, bio_model: Any, engine: Any = None
) -> Optional[BioModelOption]:
    """
    Option whose tissue-dependent base data the selected biological model reads.

    None if there is no machine or model, the model matches no option the machine and dose
    engine can run, or the option reads no tissue-dependent base data; the editors then
    offer every tissue class and preset.
    """
    summary = machine_summary(machine)
    if summary is None or bio_model_quietly(bio_model, summary.radiation_mode) is None:
        return None
    options = bio_options_quietly(summary.radiation_mode, summary, engine)
    option = matching_bio_option(options, bio_model, summary)
    if option is None or not option.available or option.provenance is None:
        return None
    return option


def tissue_issues_quietly(
    machine: Any, bio_model: Any, cst: Any, engine: Any = None
) -> list[TissueIssue]:
    """
    Issues of the structures' tissues for a biological model, machine and dose engine.

    Empty without a biological model (or ``none``), machine or structures, or if the
    check itself fails: the dose engines run the same check and report such failures.
    """
    summary = machine_summary(machine)
    if summary is None or cst is None:
        return []
    model = bio_model_quietly(bio_model, summary.radiation_mode)
    if model is None or model.model == "none":
        return []
    try:
        return list(model.check_tissues(summary, cst, engine=engine_capability_quietly(engine)))
    except Exception:  # noqa: BLE001 - the dose engine runs the same check
        logger.debug("Could not check the tissues for model '%s'", model.model, exc_info=True)
        return []


def tissue_errors(issues: Iterable[TissueIssue]) -> list[TissueIssue]:
    """Return the issues preventing the biological model from being evaluated."""
    return [issue for issue in issues if issue.is_error]


def tissue_warnings(issues: Iterable[TissueIssue]) -> list[TissueIssue]:
    """Return the issues reported while the biological model is evaluated nonetheless."""
    return [issue for issue in issues if not issue.is_error]


def tissue_issue_text(issue: TissueIssue) -> str:
    """One issue as shown to the user, naming the closest tissue class if there is one."""
    text = str(issue)
    if issue.is_error and issue.suggestion is not None and "closest" not in issue.message.lower():
        text += f" Closest tissue class: {tissue_summary(issue.suggestion)}."
    return text


def tissue_issues_text(issues: Iterable[TissueIssue]) -> str:
    """Bulleted list of issues, one per line."""
    return "\n".join(f"• {tissue_issue_text(issue)}" for issue in issues)


def compare_tissues_quietly(
    machine: Any, bio_model: Any, structures: Any, engine: Any = None
) -> list[TissueComparison]:
    """
    Compare the structures' tissues parameter by parameter with the base data's classes.

    Empty without a biological model (or ``none``), machine or structures, or if the
    comparison fails; see :meth:`~pyRadPlan.biology.models.BiologicalModel.compare_tissues`.
    Pass every structure, since together they select the kernel set.
    """
    summary = machine_summary(machine)
    if summary is None or structures is None:
        return []
    model = bio_model_quietly(bio_model, summary.radiation_mode)
    if model is None or model.model == "none":
        return []
    try:
        return list(
            model.compare_tissues(summary, structures, engine=engine_capability_quietly(engine))
        )
    except Exception:  # noqa: BLE001 - highlighting is a convenience only
        logger.debug("Could not compare the tissues for model '%s'", model.model, exc_info=True)
        return []


def _format_value(value: Any) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def mismatch_note(parameter: ParameterComparison) -> str:
    """Tooltip of a parameter differing from the base data, e.g. ``base data: D_t = 30``."""
    return f"base data: {param_label(parameter.name)} = {_format_value(parameter.declared)}"


def comparison_row_note(comparison: TissueComparison) -> str:
    """Tooltip of a structure naming its matching or closest tissue class of the base data."""
    if comparison.tissue_class is None:
        return ""
    summary = tissue_summary(comparison.tissue_class)
    if comparison.matched:
        lines = [f"Matches the base data class {summary}"]
        if comparison.provenance_mismatch:
            lines.append(provenance_mismatch_note(comparison))
        return "\n".join(lines)
    lines = [f"Closest base data class: {summary}"]
    if comparison.issue is not None:
        lines.insert(0, str(comparison.issue))
    return "\n".join(lines)


#: Second line of the note on a provenance label differing from the base data's.
PROVENANCE_WARNING_NOTE = "Evaluated with the base data's model; relabelling is optional."


def provenance_mismatch_note(comparison: TissueComparison) -> str:
    """
    Tooltip of a tissue model differing from the provenance of the base data.

    For a tissue matching the base data by its parameters (a warning), it adds that the
    model is evaluated with the base data's provenance.
    """
    if comparison.tissue_class is None:
        return "The tissue model differs from the base data's."
    note = (
        f"base data: {provenance_label(tissue_provenance(comparison.tissue_class))} "
        f"(tissue: {provenance_label(tissue_provenance(comparison.tissue))})"
    )
    return f"{note}\n{PROVENANCE_WARNING_NOTE}" if comparison.matched else note


def provenance_field_notes(comparison: Optional[TissueComparison]) -> dict[str, str]:
    """
    Return notes on provenance labels (e.g. the LEM version) differing from the base data.

    Only for a tissue matching its class by the parameters, i.e. a warning; keyed by field.
    """
    if comparison is None or not comparison.matched or not comparison.provenance_mismatch:
        return {}
    tissue, tissue_class = comparison.tissue, comparison.tissue_class
    return {
        name: f"base data: {param_label(name)} = {_format_value(getattr(tissue_class, name))}"
        for name in type(tissue).provenance_fields
        if name != "model"
        and name in type(tissue_class).model_fields
        and getattr(tissue, name) != getattr(tissue_class, name)
    }


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


def relabel_issues(comparisons: Iterable[Optional[TissueComparison]]) -> list[TissueIssue]:
    """Warnings of comparisons offering to relabel a tissue with the base data's model."""
    return [
        c.issue
        for c in comparisons
        if c is not None and c.matched and c.fix_provenance and c.issue is not None
    ]


def relabel_text(provenances: Iterable[str]) -> Optional[str]:
    """Label of the action relabelling tissues, e.g. ``"Relabel as mMKM"``; None if none."""
    labels = list(dict.fromkeys(provenance_label(p) for p in provenances if p))
    if not labels:
        return None
    return f"Relabel as {' / '.join(labels)}"


#: Tooltip of the action relabelling tissues with the provenance of the base data.
RELABEL_TOOLTIP = (
    "The tissue matches the base data by its parameters but is labelled with another model "
    "(or LEM version), so the base data's model is evaluated. Relabelling it as that model "
    "keeps every parameter and only silences the warning; it is optional."
)

#: Buttons of the dialog asking whether to accept tissues failing the check.
APPLY_ANYWAY_TEXT = "Apply anyway"
KEEP_EDITING_TEXT = "Keep editing"

#: Answers of :func:`ask_tissue_issues`.
APPLY_ANYWAY = "apply"
KEEP_EDITING = "keep"

#: Label and tooltip of the button matching tissues to their closest base data class.
MATCH_BASE_DATA_TEXT = "Match base data"
MATCH_BASE_DATA_TOOLTIP = (
    "Apply the closest tissue class of the selected model's base data: its model and the "
    "parameters identifying it. The parameters describing the tissue itself (e.g. α_0 / "
    "β_0 of MKM) and explicitly set reference parameters keep their value."
)


def provenance_fixes(issues: Iterable[TissueIssue]) -> list[str]:
    """Distinct provenances the relabel quick fixes of tissue issues adopt, in issue order."""
    fixes: list[str] = []
    for issue in issues:
        fix = issue.fix_provenance
        if fix and issue.structure is not None and fix not in fixes:
            fixes.append(fix)
    return fixes


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


def param_summary(tissue: TissueProperties) -> str:
    """Comma separated list of the parameters a tissue has set, empty if it has none."""
    return ", ".join(
        f"{param_label(field)} = {getattr(tissue, field):g}"
        for field in tissue_param_fields(type(tissue))
        if not math.isnan(getattr(tissue, field))
    )


def tissue_summary(tissue: TissueProperties) -> str:
    """Return a one-line summary of a tissue class: model, cell line and set parameters."""
    model = tissue.display_name
    name = f"{model} / {tissue.cell_line}" if tissue.cell_line else model
    params = param_summary(tissue)
    return f"{name}: {params}" if params else name


def cell_line_presets() -> list[tuple[CellLinePreset, TissueProperties]]:
    """
    Return the registered cell line presets with their tissues, as (preset, tissue) pairs.

    Every preset of a cell line and provenance is listed, in registration order (see
    :func:`~pyRadPlan.biology.tissues.registered_cell_line_presets`), which groups the
    models of one cell line together. A preset that does not validate is skipped, so a
    broken entry never keeps the editors from opening.
    """
    presets: list[tuple[CellLinePreset, TissueProperties]] = []
    for preset in registered_cell_line_presets():
        try:
            presets.append((preset, preset.tissue()))
        except Exception:  # noqa: BLE001 - convenience only, must never block an editor
            logger.warning("Skipping invalid tissue preset %s", preset.key, exc_info=True)
    return presets


def tissue_preset(tissue: TissueProperties) -> Optional[CellLinePreset]:
    """
    Cell line preset a tissue was built from, by its cell line and source (citation).

    A preset of the tissue's own provenance is preferred, else one it was converted from
    (e.g. an MKM preset adopted as mMKM); None if the tissue names no registered preset.
    """
    if not tissue.cell_line or not tissue.cell_line_source:
        return None
    try:
        return get_cell_line_preset(
            tissue.cell_line, tissue_provenance(tissue), tissue.cell_line_source
        )
    except ValueError:
        pass
    return next(
        (
            preset
            for preset in registered_cell_line_presets(tissue.cell_line)
            if preset.source.citation == tissue.cell_line_source
        ),
        None,
    )


def published_note(tissue: TissueProperties) -> str:
    """Note of a reference parameter given as published, naming the publication."""
    preset = tissue_preset(tissue)
    if preset is not None:
        return f"published ({preset.reference_label})"
    return f"published ({tissue.cell_line_source})" if tissue.cell_line_source else "published"


def preset_label(preset: CellLinePreset, tissue: TissueProperties) -> str:
    """
    Return the label a cell line preset is offered under, e.g. ``HSG · LEM II (Manganaro 2018)``.

    Names the preset's provenance ("LEM II"), which is how the LEM versions are published,
    and its source in short (``"unverified"`` for a preset without a confirmed one), which
    tells several presets of one cell line and provenance apart, followed by its parameters.
    """
    provenance = provenance_label(tissue_provenance(tissue))
    name = f"{preset.cell_line} · {provenance} ({preset.label})"
    params = param_summary(tissue)
    return f"{name}: {params}" if params else name


def preset_tooltip(preset: CellLinePreset, tissue: TissueProperties) -> str:
    """Tooltip of a cell line preset: its full source, notes and reference parameters."""
    source = preset.source
    lines = [f"Source: {source.citation}"]
    if not source.verified:
        lines.append("The origin of these values has not been confirmed.")
    if source.note:
        lines.append(source.note)
    reference = ", ".join(
        f"{param_label(name)} = {getattr(tissue, name):.4g}" for name in REFERENCE_FIELDS
    )
    if tissue.reference_source == "published":
        how = f"published, {preset.reference_label}"
    elif tissue.reference_source == "estimated":
        how = "derived from the model"
        radiation = getattr(tissue, REFERENCE_RADIATION_FIELD, None)
        if radiation is not None:
            how += f" for {reference_radiation_text(radiation)}"
    else:
        how = "given"
    lines.append(f"Reference radiation (for RBE): {reference} ({how})")
    return "\n".join(lines)


def class_label(tissue: TissueProperties) -> str:
    """Label of a tissue class of base data, e.g. ``Generic_MKM · mMKM (base data): ...``."""
    provenance = provenance_label(tissue_provenance(tissue))
    name = f"{tissue.cell_line or 'unnamed'} · {provenance} (base data)"
    params = param_summary(tissue)
    return f"{name}: {params}" if params else name


#: Marks a picker entry whose tissue matches a tissue class of the base data exactly.
EXACT_MARK = "✓ "


@dataclass(frozen=True)
class TissuePick:
    """
    One entry of the tissue picker: a tissue class of the base data or a cell line preset.

    Attributes
    ----------
    kind : str
        ``"class"`` for a tissue class of the base data, ``"preset"`` for a preset.
    index : int
        Position of the class or preset in the list it was taken from.
    tissue : TissueProperties
        The class or preset.
    provenance : str
        Canonical provenance of the tissue.
    label : str
        Text offered in the picker.
    preferred : bool
        Offered above the separator: a class of the base data, or a preset of its
        provenance.
    exact : bool
        Whether applying it yields a tissue matching a class of the base data exactly.
    tooltip : str
        Tooltip of the entry, e.g. the full source of a preset.
    """

    kind: str
    index: int
    tissue: TissueProperties
    provenance: str
    label: str
    preferred: bool = False
    exact: bool = False
    tooltip: str = ""


def _provenance_rank(provenance: str) -> int:
    names = [registered.name for registered in registered_provenances()]
    return names.index(provenance) if provenance in names else len(names)


def ranked_tissue_picks(
    classes: list[TissueProperties],
    presets: list[tuple[CellLinePreset, TissueProperties]],
    provenance: Optional[str] = None,
) -> list[TissuePick]:
    """
    Order the tissue classes of the base data and the cell line presets for the picker.

    The classes come first, followed by the presets of the base data's `provenance`: these
    are :attr:`~TissuePick.preferred`. The other presets follow, grouped by provenance in
    registration order. Every class is marked :attr:`~TissuePick.exact`, as is each
    preferred preset matching one of the classes. Several presets of one cell line and
    provenance are each listed, labelled by their source.
    """
    picks = [
        TissuePick(
            "class",
            index,
            tissue,
            tissue_provenance(tissue),
            EXACT_MARK + class_label(tissue),
            preferred=True,
            exact=True,
        )
        for index, tissue in enumerate(classes)
    ]
    others: list[TissuePick] = []
    for index, (preset, tissue) in enumerate(presets):
        tissue_prov = tissue_provenance(tissue)
        preferred = provenance is not None and tissue_prov == provenance
        exact = preferred and any(tissue.matches(cls, ignore_undeclared=True) for cls in classes)
        label = preset_label(preset, tissue)
        pick = TissuePick(
            "preset",
            index,
            tissue,
            tissue_prov,
            EXACT_MARK + label if exact else label,
            preferred=preferred,
            exact=exact,
            tooltip=preset_tooltip(preset, tissue),
        )
        (picks if preferred else others).append(pick)
    others.sort(key=lambda pick: _provenance_rank(pick.provenance))
    return picks + others


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


def picker_title(machine: Any, bio_model: Any, engine: Any = None) -> str:
    """Header of the preferred picks: the base data the selected model reads."""
    option = selected_bio_option(machine, bio_model, engine)
    return f"Base data · {option.label}" if option is not None else "Base data"


def target_provenance(machine: Any, bio_model: Any, engine: Any = None) -> Optional[str]:
    """
    Provenance of the base data the selected biological model reads, None if none.

    See :func:`selected_bio_option`.
    """
    option = selected_bio_option(machine, bio_model, engine)
    return option.provenance if option is not None else None


def adopt_preset(
    tissue: TissueProperties, provenance: Optional[str]
) -> tuple[TissueProperties, str]:
    """
    Convert a preset to the base data's `provenance`.

    A preset keeps its own provenance if there is none or the base data does not declare
    it. Returns the tissue to apply and a note on where its values came from, e.g.
    ``"values from HSG · MKM; adopted as mMKM"``, see
    :func:`~pyRadPlan.biology.tissues.adopt_provenance`.
    """
    if provenance is None or not get_provenance(provenance).declared:
        provenance = tissue_provenance(tissue)
    adoption = adopt_provenance(tissue, provenance)
    return adoption.tissue, adoption.describe()


def apply_base_data_class(
    tissue: TissueProperties, tissue_class: TissueProperties
) -> tuple[TissueProperties, Optional[str]]:
    """
    Apply a tissue class of base data to a tissue, see :func:`apply_tissue_class`.

    Returns the new tissue and the name of the class if the reference parameters may
    have been taken from it, for :func:`reference_sources`; None otherwise.
    """
    result = apply_tissue_class(tissue, tissue_class)
    sets_reference = bool(declared_reference_parameters(tissue_class)) or any(
        name in type(tissue_class).identifying_fields for name in REFERENCE_FIELDS
    )
    name = tissue_class.cell_line or tissue_summary(tissue_class)
    return result, name if sets_reference else None


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

#: Note of a reference parameter holding the generic default.
DEFAULT_NOTE = f"default {DEFAULT_ALPHA_X:g} / {DEFAULT_BETA_X:g}"


def derived_note(
    name: str, model: Optional[str] = None, reference_radiation: Optional[str] = None
) -> str:
    """
    Note of a reference parameter derived from the MKM model parameters.

    E.g. ``"derived: α_0 + β_0 · z_1D,ref of 200 kV X-rays (y_D = 4.2 keV/µm)"``, with
    the saturation-corrected z*_1D,ref for mMKM.
    """
    if name != "alpha_x":
        return "derived: β_0"
    saturated = MKMTissueProperties.saturation_corrected(model or "MKM")
    z_ref = "z*_1D,ref" if saturated else "z_1D,ref"
    return f"derived: α_0 + β_0 · {z_ref} of {reference_radiation_text(reference_radiation)}"


def base_data_note(class_name: Optional[str] = None) -> str:
    """Note of a reference parameter taken over from a tissue class of base data."""
    return f"from base data class {class_name}" if class_name else "from base data"


def reference_sources(
    tissue: TissueProperties, class_name: Optional[str] = None
) -> dict[str, tuple[str, str]]:
    """
    Source and note of each reference parameter of a tissue, by parameter name.

    The source is the one the tissue records for the parameter
    (:attr:`~TissueProperties.reference_sources`); the note says where the value came
    from, e.g. the publication of a published value. `class_name` names the tissue class a
    ``"base_data"`` value came from, if known.
    """
    sources: dict[str, tuple[str, str]] = {}
    for name in REFERENCE_FIELDS:
        source = tissue.reference_sources.get(name, "explicit")
        if source == "estimated":
            note = derived_note(
                name, getattr(tissue, "model", None), getattr(tissue, "reference_radiation", None)
            )
        elif source == "published":
            note = published_note(tissue)
        elif source == "default":
            note = DEFAULT_NOTE
        elif source == "base_data":
            note = base_data_note(class_name)
        else:
            note = ""
        sources[name] = (source, note)
    return sources


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


def build_tissue_properties(
    tissue_cls: type[TissueProperties],
    values: Mapping[str, Any],
    current: Optional[TissueProperties] = None,
    reference_sources: Optional[Mapping[str, str]] = None,
    model: Optional[str] = None,
) -> TissueProperties:
    """
    Build tissue properties of `tissue_cls` from a parameter mapping.

    Parameters the class does not know and NaN values are dropped, so unset parameters
    keep the class default. If `current` is of the same class, its model name is carried
    over, and so is its cell line (with its source) as long as every parameter still
    matches it: once the values were changed, they no longer describe that cell line. The
    reference radiation is taken from `values` if given there, else kept from `current`.

    Parameters
    ----------
    tissue_cls : type[TissueProperties]
        Class to build.
    values : Mapping[str, Any]
        Parameter values by field name.
    current : TissueProperties, optional
        The tissue being edited.
    reference_sources : Mapping[str, str], optional
        Where each reference parameter in `values` came from (see
        :func:`reference_source_of`), so that e.g. an overridden α_x is kept while β_x is
        still derived. A parameter it does not name keeps the source of `current` as long as
        it equals its value, and is explicit otherwise.
    model : str, optional
        Model name to give the tissue (see :func:`model_label_name`), instead of the one
        of `current` or the class default.
    """
    params: dict[str, Any] = {
        name: value
        for name, value in values.items()
        if name in tissue_cls.model_fields and not (isinstance(value, float) and math.isnan(value))
    }
    sources = dict(reference_sources or {})
    for name in REFERENCE_FIELDS:
        if name not in sources and current is not None:
            if _same_value(params.get(name), getattr(current, name)):
                sources[name] = current.reference_sources[name]
    params["reference_sources"] = sources
    same_class = current is not None and type(current) is tissue_cls
    if model is not None:
        params["model"] = model
    elif same_class:
        params["model"] = current.model
    for name in _KEPT_SETTINGS:
        if current is not None and name in tissue_cls.model_fields:
            if name in type(current).model_fields:
                params.setdefault(name, getattr(current, name))
    tissue = tissue_cls(**params)
    if (
        same_class
        and current.cell_line
        and all(
            _same_value(getattr(tissue, name), getattr(current, name))
            for name in tissue_edit_fields(tissue_cls) + reference_radiation_fields(tissue_cls)
        )
    ):
        tissue = tissue_cls(
            **params, cell_line=current.cell_line, cell_line_source=current.cell_line_source
        )
    return tissue


# Settings kept from the tissue being edited unless the editor gives them.
_KEPT_SETTINGS = (REFERENCE_RADIATION_FIELD,)


def _same_value(a: Any, b: Any) -> bool:
    """Whether two parameter values are equal, two unset (NaN) values included."""
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b

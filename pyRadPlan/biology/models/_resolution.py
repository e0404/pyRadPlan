"""
Resolution of a biological model against a machine and the structures' tissues.

A formalism reading tissue-dependent base data (α/β or z* kernels, RBE tables) needs one
kernel set, chosen by deterministic rules (:func:`select_kernel_set`), and one tissue class
of that set per structure. :func:`resolve_bio_model` performs both and reports every
problem as a :class:`TissueIssue`, so that a user interface and a dose engine apply the
same check.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from pyRadPlan.biology.tissues import (
    TissueProperties,
    get_provenance,
    provenance_label,
    tissue_provenance,
)

from ._engine import engine_bio_capability
from ._kernel_sets import (
    KERNEL_SET_KIND_LABELS,
    BioMachineSummary,
    KernelSetInfo,
    TissueClassInfo,
)

if TYPE_CHECKING:  # pragma: no cover
    from ._base import BiologicalModel

#: Kinds of :class:`TissueIssue`, in the order they are checked.
ISSUE_KINDS: tuple[str, ...] = (
    "model_unavailable",
    "base_data",
    "invalid_reference",
    "reference_domain",
    "type_mismatch",
    "no_matching_class",
    "provenance_differs",
    "missing_parameter",
)

#: Severities of a :class:`TissueIssue`, see :attr:`TissueIssue.severity`.
ISSUE_SEVERITIES: tuple[str, ...] = ("error", "warning")


@dataclass(frozen=True, eq=False)
class TissueIssue:
    """
    One problem evaluating a biological model for a machine and structure set.

    Attributes
    ----------
    kind : str
        One of :data:`ISSUE_KINDS`:

        - ``model_unavailable``: the machine cannot run the model at all (radiation mode,
          missing quantities, no or an ambiguous kernel set).
        - ``base_data``: the kernel set is inconsistent (e.g. differing tissue classes per
          energy, missing kernel fields).
        - ``invalid_reference``: a structure's reference α_x / β_x is negative, infinite or
          NaN. Dose engines always raise for it.
        - ``reference_domain``: a structure's reference α_x / β_x is outside the model's
          domain (e.g. β_x = 0 for McNamara). An error, since the model cannot be
          evaluated there at all.
        - ``type_mismatch``: the tissue's parameter class cannot describe the classes of
          the kernel set, so the two share no identifying parameter (e.g. LEM parameters
          for MKM z* kernels).
        - ``no_matching_class``: no tissue class of the kernel set matches; ``suggestion``
          holds the closest one.
        - ``provenance_differs`` (warning): the tissue matches a class of the kernel set by
          its parameters, but is labelled with another provenance (e.g. MKM for mMKM z*
          kernels, LEM I for LEM IV α/β kernels). The model is evaluated with the base
          data, i.e. the base data's provenance; ``fix_provenance`` offers to relabel.
        - ``missing_parameter``: a tissue parameter the model reads at runtime is unset.
    message : str
        Human-readable description.
    structure : str or None
        Name of the affected structure; None for issues of the model or machine.
    tissue : TissueProperties or None
        The structure's tissue properties.
    kernel_set : str or None
        Id of the kernel set (or table) concerned.
    suggestion : TissueProperties or None
        Closest tissue class of the kernel set, for ``no_matching_class`` and
        ``type_mismatch``; the matched class for ``provenance_differs``.
    severity : str
        One of :data:`ISSUE_SEVERITIES`. ``"error"``: the model cannot be evaluated; dose
        engines raise (``calc_bio_dose=True``) or fall back to the physical dose
        (``"auto"``). ``"warning"``: the model is evaluated, and dose engines log and
        report the issue (``provenance_differs``).
    fix_provenance : str or None
        Optional quick fix of a ``provenance_differs`` warning: the canonical provenance
        of the matched class to relabel the tissue with, e.g. ``"mMKM"``. Applied by
        :func:`apply_provenance_fixes` or ``tissue.adopt_provenance(issue.fix_provenance)``;
        set only if the tissue's parameter class is that provenance's, so that relabelling
        keeps every parameter.
    """

    kind: str
    message: str
    structure: Optional[str] = None
    tissue: Optional[TissueProperties] = None
    kernel_set: Optional[str] = None
    suggestion: Optional[TissueProperties] = None
    severity: str = "error"
    fix_provenance: Optional[str] = None

    def __post_init__(self) -> None:
        if self.severity not in ISSUE_SEVERITIES:
            raise ValueError(
                f"TissueIssue severity must be one of {ISSUE_SEVERITIES}, not {self.severity!r}"
            )

    @property
    def is_error(self) -> bool:
        """Whether the issue prevents evaluating the model."""
        return self.severity == "error"

    @property
    def is_warning(self) -> bool:
        """Whether the issue is reported, but the model is evaluated nonetheless."""
        return self.severity == "warning"

    def __str__(self) -> str:
        return f"{self.structure}: {self.message}" if self.structure else self.message


@dataclass(frozen=True, eq=False)
class StructureResolution:
    """
    Where the base data of one structure's tissue lives.

    Attributes
    ----------
    structure : str
        Structure name.
    tissue : TissueProperties
        The structure's tissue properties.
    kernel_set : str or None
        Id of the kernel set (or table) holding its base data; None if the formalism reads
        no tissue-dependent data or the tissue does not resolve.
    column : int or None
        Zero-based column of that kernel set.
    tissue_class : TissueClassInfo or None
        The matched tissue class of the kernel set.
    issues : tuple[TissueIssue, ...]
        Problems of this structure.
    """

    structure: str
    tissue: TissueProperties
    kernel_set: Optional[str] = None
    column: Optional[int] = None
    tissue_class: Optional[TissueClassInfo] = None
    issues: tuple[TissueIssue, ...] = ()

    @property
    def resolved(self) -> bool:
        """Whether the structure has no error."""
        return not any(issue.is_error for issue in self.issues)


@dataclass(frozen=True, eq=False)
class BioModelResolution:
    """
    Result of resolving a biological model against a machine and structure set.

    Shaped per structure, so that structures may later resolve to different kernel sets
    (mixed provenances, biological scenarios); dose calculations currently require all of
    them to use the same one.

    Attributes
    ----------
    model : str
        Identifier of the formalism.
    kernel_set : KernelSetInfo or None
        The kernel set (or RBE table) selected for the formalism; None if it reads no
        tissue-dependent data or none could be selected.
    structures : tuple[StructureResolution, ...]
        One entry per structure, in structure order.
    issues : tuple[TissueIssue, ...]
        Every issue: those of the model and machine first, then per structure.
    """

    model: str
    kernel_set: Optional[KernelSetInfo]
    structures: tuple[StructureResolution, ...]
    issues: tuple[TissueIssue, ...]

    @property
    def errors(self) -> tuple[TissueIssue, ...]:
        """Issues preventing the evaluation."""
        return tuple(issue for issue in self.issues if issue.is_error)

    @property
    def warnings(self) -> tuple[TissueIssue, ...]:
        """Issues reported while the model is evaluated nonetheless."""
        return tuple(issue for issue in self.issues if issue.is_warning)

    @property
    def ok(self) -> bool:
        """Whether every structure resolves and the model can be evaluated."""
        return not self.errors

    @property
    def kernel_set_ids(self) -> tuple[str, ...]:
        """Distinct kernel sets the structures resolve to, in structure order."""
        ids: list[str] = []
        for entry in self.structures:
            if entry.kernel_set is not None and entry.kernel_set not in ids:
                ids.append(entry.kernel_set)
        return tuple(ids)

    def pairs(self) -> dict[str, tuple[Optional[str], Optional[int]]]:
        """``{structure: (kernel set id, column)}`` for every structure."""
        return {entry.structure: (entry.kernel_set, entry.column) for entry in self.structures}

    def raise_for_errors(self) -> None:
        """
        Raise ValueError describing every error, if there is one.

        Unmatched tissues are reported together, as "No matching tissue class in the
        machine base data for CST tissue(s): [...]".
        """
        errors = self.errors
        if not errors:
            return
        unmatched = [i for i in errors if i.kind == "no_matching_class"]
        messages = [str(i) for i in errors if i.kind != "no_matching_class"]
        if unmatched:
            tissues = []
            for issue in unmatched:
                if not any(issue.tissue is t or issue.tissue == t for t in tissues):
                    tissues.append(issue.tissue)
            available = list(self.kernel_set.tissues) if self.kernel_set is not None else []
            messages.append(
                "No matching tissue class in the machine base data for CST tissue(s): "
                f"{tissues}. Available classes: {available}"
            )
        raise ValueError("; ".join(messages))


# ----------------------------------------------------------------- structures
def structure_tissues(cst: Any) -> list[tuple[str, TissueProperties]]:
    """
    ``(name, tissue)`` pairs of a structure set.

    Accepts a :class:`~pyRadPlan.cst.StructureSet`, a sequence of VOIs, a mapping of
    names to tissue properties, a sequence of ``(name, tissue)`` pairs, or a sequence of
    tissue properties (named ``"tissue class <i>"``).
    """
    if cst is None:
        return []
    vois = getattr(cst, "vois", None)
    if vois is not None:
        cst = vois
    if isinstance(cst, Mapping):
        return [(str(name), tissue) for name, tissue in cst.items()]
    pairs = []
    for position, entry in enumerate(list(cst)):
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[1], TissueProperties):
            pairs.append((str(entry[0]), entry[1]))
            continue
        tissue = getattr(entry, "tissue_properties", None)
        if tissue is not None:
            pairs.append((str(getattr(entry, "name", f"structure {position + 1}")), tissue))
        elif isinstance(entry, TissueProperties):
            pairs.append((f"tissue class {position + 1}", entry))
    return pairs


# ----------------------------------------------------------------- matching
def _compatible_type(tissue: TissueProperties, kernel_set: KernelSetInfo) -> bool:
    """
    Whether a tissue's parameter class can describe a class of the kernel set at all.

    Only the parameter class counts, not the provenance label: MKM parameters can describe
    mMKM classes, LEM I parameters LEM IV ones.
    """
    if type(tissue) is TissueProperties:
        return True
    provenance = get_provenance(kernel_set.provenance)
    if provenance.declared:
        return isinstance(tissue, provenance.tissue_type)
    classes = kernel_set.tissues
    return not classes or any(tissue._same_model_as(cls) for cls in classes)


def _class_provenance(info: TissueClassInfo, kernel_set: KernelSetInfo) -> str:
    """Provenance of a tissue class, the kernel set's if the class declares none."""
    provenance = info.provenance
    return provenance if get_provenance(provenance).declared else kernel_set.provenance


def _provenance_differs(tissue: TissueProperties, provenance: str) -> bool:
    """Whether a tissue is labelled with another provenance; undeclared ones never are."""
    own = get_provenance(tissue_provenance(tissue))
    other = get_provenance(provenance)
    return own.declared and other.declared and own.name != other.name


def _match(tissue: TissueProperties, kernel_set: KernelSetInfo) -> Optional[TissueClassInfo]:
    """Return the matching class of the kernel set, preferring the tissue's provenance."""
    matching = [
        info
        for info in kernel_set.tissue_classes
        if tissue.matches(info.tissue, ignore_undeclared=True)
    ]
    return next(
        (
            info
            for info in matching
            if not _provenance_differs(tissue, _class_provenance(info, kernel_set))
        ),
        matching[0] if matching else None,
    )


def _distance(tissue: TissueProperties, candidate: TissueProperties) -> float:
    names = [
        name
        for name in type(tissue).model_fields
        if name in type(candidate).model_fields
        and isinstance(getattr(tissue, name), float)
        and isinstance(getattr(candidate, name), float)
    ]
    total = 0.0
    for name in names:
        a, b = getattr(tissue, name), getattr(candidate, name)
        # A parameter the class leaves undeclared is not compared, as in matching.
        if math.isnan(b):
            continue
        if math.isnan(a):
            total += 1.0
            continue
        total += abs(a - b) / max(abs(a), abs(b), 1e-12)
    return total


def closest_tissue_class(
    tissue: TissueProperties, kernel_set: KernelSetInfo
) -> Optional[TissueProperties]:
    """
    Tissue class of the kernel set whose parameters are closest to ``tissue``'s.

    Classes of a compatible parameter class are preferred; of equally close ones, one of
    the tissue's own provenance.
    """
    infos = [
        info for info in kernel_set.tissue_classes if tissue._same_model_as(info.tissue)
    ] or list(kernel_set.tissue_classes)
    if not infos:
        return None
    closest = min(
        infos,
        key=lambda info: (
            _distance(tissue, info.tissue),
            _provenance_differs(tissue, _class_provenance(info, kernel_set)),
        ),
    )
    return closest.tissue


def resolves_in(
    tissues: Sequence[TissueProperties],
    kernel_set: KernelSetInfo,
    same_provenance: bool = False,
) -> bool:
    """
    Whether every tissue resolves to a class of the kernel set.

    With `same_provenance`, every tissue must moreover resolve to a class of its own
    provenance (tissues of undeclared provenance resolve to any).
    """
    if kernel_set.problems:
        return False
    for tissue in tissues:
        info = _match(tissue, kernel_set) if _compatible_type(tissue, kernel_set) else None
        if info is None:
            return False
        if same_provenance and _provenance_differs(tissue, _class_provenance(info, kernel_set)):
            return False
    return True


# ----------------------------------------------------------------- set selection
class KernelSetSelectionError(ValueError):
    """No kernel set, or several equally fitting ones, for a formalism."""


def select_kernel_set(
    sets: Sequence[KernelSetInfo],
    *,
    kind: str,
    model_name: str = "",
    kernel_set: Optional[str] = None,
    data_model: Optional[str] = None,
    variant: Optional[str] = None,
    tissues: Optional[Sequence[TissueProperties]] = None,
) -> KernelSetInfo:
    """
    Select the kernel set a formalism reads, by the resolution rules of the design.

    1. An explicit ``kernel_set`` id wins.
    2. Otherwise the sets of the right kind matching ``data_model`` (provenance) and
       ``variant``.
    3. Otherwise the sets that all ``tissues`` resolve to, by their parameters; of several,
       those the tissues resolve to with their own provenance label.
    4. Otherwise the set marked ``default``, or the only set of that kind.
    5. Still ambiguous: an error listing the candidates and how to choose one.

    Raises
    ------
    KernelSetSelectionError
        No set fits, or the choice is ambiguous.
    """
    kind_label = KERNEL_SET_KIND_LABELS.get(kind, kind)
    subject = f"Biological model '{model_name}'" if model_name else "The biological model"
    of_kind = [s for s in sets if s.kind == kind]
    if kernel_set is not None:
        chosen = next((s for s in sets if s.id == kernel_set), None)
        if chosen is None or chosen.kind != kind:
            raise KernelSetSelectionError(
                f"{subject} is configured for kernel set '{kernel_set}', but the machine has "
                f"no {kind_label} set of that id. Available: {_describe(of_kind)}."
            )
        return chosen

    if not of_kind:
        raise KernelSetSelectionError(
            f"{subject} needs {kind_label}, but the machine carries none."
        )

    candidates = of_kind
    if data_model:
        provenance = get_provenance(data_model)
        candidates = [s for s in candidates if s.provenance == provenance.name]
        if not candidates:
            raise KernelSetSelectionError(
                f"{subject} is configured for data model '{provenance.label}', but the "
                f"machine has no {kind_label} of that provenance. Available: "
                f"{_describe(of_kind)}."
            )
    if variant:
        candidates = [s for s in candidates if (s.variant or "").casefold() == variant.casefold()]
        if not candidates:
            raise KernelSetSelectionError(
                f"{subject} is configured for the {kind_label} variant '{variant}', which the "
                f"machine does not provide. Available: {_describe(of_kind)}."
            )

    if len(candidates) > 1 and tissues:
        resolving = [s for s in candidates if resolves_in(tissues, s)]
        if resolving:
            candidates = resolving
        if len(candidates) > 1:
            labelled = [s for s in candidates if resolves_in(tissues, s, same_provenance=True)]
            if labelled:
                candidates = labelled
    if len(candidates) > 1:
        defaults = [s for s in candidates if s.default]
        if len(defaults) == 1:
            candidates = defaults
    if len(candidates) == 1:
        return candidates[0]

    hints = [f"kernel_set='{candidates[0].id}'"]
    if len({s.provenance for s in candidates}) == len(candidates):
        hints.append(f"data_model='{candidates[0].provenance}'")
    raise KernelSetSelectionError(
        f"{subject} fits several {kind_label} sets of the machine: {_describe(candidates)}. "
        f"Choose one with the model parameter {' or '.join(hints)} (or mark one set as "
        "default in the machine's kernelSets)."
    )


def _describe(sets: Sequence[KernelSetInfo]) -> str:
    if not sets:
        return "none"
    return ", ".join(f"'{s.id}' ({s.label})" for s in sets)


# ----------------------------------------------------------------- resolution
def _reference_issues(
    model: "BiologicalModel", name: str, tissue: TissueProperties
) -> list[TissueIssue]:
    issues = []
    values = {}
    for parameter in ("alpha_x", "beta_x"):
        value = getattr(tissue, parameter, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = math.nan
        values[parameter] = value
        if not (math.isfinite(value) and value >= 0.0):
            issues.append(
                TissueIssue(
                    "invalid_reference",
                    f"The reference photon parameter {parameter} must be finite and "
                    f"non-negative, but is {value}.",
                    structure=name,
                    tissue=tissue,
                )
            )
    if issues:
        return issues
    defined = any(v != 0.0 for v in values.values())
    for parameter in model.requires_positive_reference:
        if defined and not values.get(parameter, 0.0) > 0.0:
            issues.append(
                TissueIssue(
                    "reference_domain",
                    f"Biological model '{model.model}' is only defined for {parameter} > 0, "
                    f"but the structure has {parameter} = {values.get(parameter)}.",
                    structure=name,
                    tissue=tissue,
                )
            )
    return issues


def _quantity_issue(
    model: "BiologicalModel", summary: Optional[BioMachineSummary], provided: Optional[list]
) -> Optional[TissueIssue]:
    messages = []
    mode = summary.radiation_mode if summary is not None else ""
    if mode and mode not in model.possible_radiation_modes:
        messages.append(f"Radiation mode '{mode}' is invalid for model '{model.model}'.")
    quantities = provided if provided is not None else (summary.quantities if summary else ())
    if quantities:
        missing = [q for q in model.required_quantities if q not in quantities]
        if missing:
            messages.append(
                "Required quantities not provided by Dose Engine or selected Machine Dataset: "
                f"{', '.join(missing)}."
            )
    if not messages:
        return None
    return TissueIssue(
        "model_unavailable",
        f"Biological model '{model.model}' not available: {' '.join(messages)}",
    )


def resolve_bio_model(
    model: "BiologicalModel",
    machine: Any = None,
    cst: Any = None,
    *,
    provided_quantities: Optional[Sequence[str]] = None,
    engine: Any = None,
) -> BioModelResolution:
    """
    Resolve a biological model against a machine and the tissues of a structure set.

    Parameters
    ----------
    model : BiologicalModel
        The formalism with its configuration.
    machine : Machine, BioMachineSummary or None
        The machine; without one, only the structure-level checks independent of base data
        (reference parameters, runtime tissue parameters) are made.
    cst : StructureSet, sequence or mapping, optional
        The structures, see :func:`structure_tissues`.
    provided_quantities : sequence of str, optional
        Quantities available to the model; defaults to the machine's as the dose engine
        sees them (:meth:`EngineBioCapability.quantities`).
    engine : str, type, DoseEngineBase or EngineBioCapability, optional
        The dose engine. A model the engine cannot compute is unavailable (e.g. α/β kernels
        with a Monte Carlo engine not using the machine's kernels), and the kernel sets are
        those the engine reads or scores (:meth:`EngineBioCapability.machine_view`). This
        part of the check needs no machine.

    Returns
    -------
    BioModelResolution
    """
    structures = structure_tissues(cst)
    model_issues, kernel_set, unavailable = _select_base_data(
        model, machine, structures, provided_quantities, engine
    )

    tissue_dependent = _tissue_dependent(kernel_set)
    resolved: list[StructureResolution] = []
    structure_issues: list[TissueIssue] = []
    for name, tissue in structures:
        # One error per structure: invalid reference parameters first, then the tissue
        # class, then the parameters read at runtime; a model that cannot run at all has
        # no per-structure reasons beyond the reference parameters. A differing provenance
        # label is a warning and does not hide the errors after it.
        issues = _reference_issues(model, name, tissue)
        info = None
        if not issues and tissue_dependent and not unavailable:
            issue, info = _tissue_class_issue(model, name, tissue, kernel_set)
            if issue is not None:
                issues.append(issue)
        missing = model.missing_tissue_parameters(tissue)
        if missing and not any(i.is_error for i in issues) and not unavailable:
            issues.append(
                TissueIssue(
                    "missing_parameter",
                    f"Biological model '{model.model}' reads the tissue parameter(s) "
                    f"{', '.join(missing)} of every structure, but {tissue!r} lacks "
                    f"{', '.join(missing)}.",
                    structure=name,
                    tissue=tissue,
                )
            )
        resolved.append(
            StructureResolution(
                structure=name,
                tissue=tissue,
                kernel_set=info.kernel_set if info is not None else None,
                column=info.column if info is not None else None,
                tissue_class=info,
                issues=tuple(issues),
            )
        )
        structure_issues.extend(issues)

    return BioModelResolution(
        model=model.model,
        kernel_set=kernel_set,
        structures=tuple(resolved),
        issues=tuple(model_issues + structure_issues),
    )


def _tissue_dependent(kernel_set: Optional[KernelSetInfo]) -> bool:
    return kernel_set is not None and kernel_set.tissue_dependent and not kernel_set.problems


def _select_base_data(
    model: "BiologicalModel",
    machine: Any,
    structures: Sequence[tuple[str, TissueProperties]],
    provided_quantities: Optional[Sequence[str]],
    engine: Any,
) -> tuple[list[TissueIssue], Optional[KernelSetInfo], bool]:
    """
    Issues of the model and machine, the selected kernel set and whether it is unavailable.

    The first, structure-independent half of :func:`resolve_bio_model`, shared with
    :func:`compare_tissue_parameters` so that both judge the same kernel set.
    """
    capability = engine_bio_capability(engine)
    summary = BioMachineSummary.from_machine(machine) if machine is not None else None
    if capability is not None and summary is not None:
        capability = capability.for_radiation_mode(summary.radiation_mode)
    provided = list(provided_quantities) if provided_quantities is not None else None

    model_issues: list[TissueIssue] = []
    kernel_set: Optional[KernelSetInfo] = None
    unavailable = False
    if capability is not None:
        reason = capability.model_reason(model)
        if reason:
            model_issues.append(
                TissueIssue(
                    "model_unavailable",
                    f"Biological model '{model.model}' not available: {reason}",
                )
            )
            unavailable = True
        elif summary is not None:
            summary = capability.machine_view(summary)
    if summary is not None and not unavailable:
        issue = _quantity_issue(model, summary, provided)
        if issue is not None:
            model_issues.append(issue)
            unavailable = True
        else:
            try:
                kernel_set = model.select_kernel_set(summary, [t for _, t in structures])
            except KernelSetSelectionError as exc:
                model_issues.append(TissueIssue("model_unavailable", str(exc)))
                unavailable = True
        if kernel_set is not None:
            model_issues.extend(
                TissueIssue("base_data", problem, kernel_set=kernel_set.id)
                for problem in kernel_set.problems
            )
            extra = model.kernel_set_issue(kernel_set)
            if extra is not None:
                model_issues.append(extra)
                unavailable = True
    return model_issues, kernel_set, unavailable


def provenance_fix(
    tissue: TissueProperties,
    kernel_set: KernelSetInfo,
    tissue_class: Optional[TissueClassInfo] = None,
) -> Optional[str]:
    """
    Provenance to relabel a tissue with so that it names the base data's, if any.

    The provenance of `tissue_class` (the class the tissue matches), else of the kernel
    set. Set if that provenance is declared and registered, the tissue is of its parameter
    class (so relabelling keeps every parameter) but labelled with another one, e.g. an MKM
    tissue for mMKM z* kernels or LEM I for LEM IV; None otherwise.
    """
    name = (
        _class_provenance(tissue_class, kernel_set)
        if tissue_class is not None
        else kernel_set.provenance
    )
    provenance = get_provenance(name)
    if not (provenance.declared and provenance.registered):
        return None
    if type(tissue) is not provenance.tissue_type or not _provenance_differs(tissue, name):
        return None
    return provenance.name


def _tissue_label(tissue: TissueProperties) -> str:
    """Name the tissue in an issue, e.g. ``"The tissue (HSG)"``."""
    return f"The tissue ({tissue.cell_line})" if tissue.cell_line else "The tissue"


def _tissue_class_issue(
    model: "BiologicalModel", name: str, tissue: TissueProperties, kernel_set: KernelSetInfo
) -> tuple[Optional[TissueIssue], Optional[TissueClassInfo]]:
    """
    Return the tissue-class issue of one structure and the class it resolves to.

    A ``provenance_differs`` warning comes with the class; errors come without one.
    """
    if not _compatible_type(tissue, kernel_set):
        return (
            TissueIssue(
                "type_mismatch",
                f"The tissue parameters are {provenance_label(tissue_provenance(tissue))} "
                f"({type(tissue).__name__}), but {kernel_set.label} (kernel set "
                f"'{kernel_set.id}') is tabulated for "
                f"{provenance_label(kernel_set.provenance)} tissue classes, which share no "
                "identifying parameter with them.",
                structure=name,
                tissue=tissue,
                kernel_set=kernel_set.id,
                suggestion=closest_tissue_class(tissue, kernel_set),
            ),
            None,
        )
    info = _match(tissue, kernel_set)
    if info is not None:
        class_provenance = _class_provenance(info, kernel_set)
        if not _provenance_differs(tissue, class_provenance):
            return None, info
        fix = provenance_fix(tissue, kernel_set, info)
        base_label = provenance_label(class_provenance)
        variant = f", {kernel_set.variant}" if kernel_set.variant else ""
        hint = (
            f" Relabel it as {provenance_label(fix)} to silence this warning "
            f'(tissue.adopt_provenance("{fix}")).'
            if fix
            else ""
        )
        return (
            TissueIssue(
                "provenance_differs",
                f"{_tissue_label(tissue)} is labelled "
                f"{provenance_label(tissue_provenance(tissue))}; the base data is "
                f"{base_label} ({kernel_set.label}{variant}, kernel set '{kernel_set.id}', "
                f"tissue class {info.name}). Evaluated with the base data's model.{hint}",
                structure=name,
                tissue=tissue,
                kernel_set=kernel_set.id,
                suggestion=info.tissue,
                severity="warning",
                fix_provenance=fix,
            ),
            info,
        )
    suggestion = closest_tissue_class(tissue, kernel_set)
    hint = f" Closest class: {suggestion!r}." if suggestion is not None else ""
    return (
        TissueIssue(
            "no_matching_class",
            f"No matching tissue class in the machine base data for {tissue!r} in "
            f"{kernel_set.label} (kernel set '{kernel_set.id}'). Available classes: "
            f"{list(kernel_set.tissues)}.{hint}",
            structure=name,
            tissue=tissue,
            kernel_set=kernel_set.id,
            suggestion=suggestion,
        ),
        None,
    )


def lookup_rows(resolution: BioModelResolution) -> np.ndarray:
    """Kernel column of every resolved structure, for an exact class lookup."""
    return np.asarray([entry.column or 0 for entry in resolution.structures], dtype=int)


# ----------------------------------------------------------------- quick fixes
def apply_provenance_fixes(cst: Any, issues: Sequence[TissueIssue]) -> Any:
    """
    Apply the ``fix_provenance`` quick fixes of tissue issues to a structure set.

    Every structure named by an issue carrying a ``fix_provenance`` (a ``provenance_differs``
    warning) gets its tissue relabelled with the base data's provenance by
    :meth:`TissueProperties.adopt_provenance`, keeping all its parameters, as only the model
    name or LEM version changes. Other structures are left as they are. Relabelling is
    optional: the model is evaluated with the base data either way, and it only silences
    the warning.

    Parameters
    ----------
    cst : StructureSet, sequence of VOIs or mapping of names to TissueProperties
        The structures. Not modified.
    issues : sequence of TissueIssue
        Issues of :meth:`BiologicalModel.check_tissues`; those without a fix are ignored.

    Returns
    -------
    StructureSet, list of VOIs or dict
        A copy of `cst` of the same kind with the fixed tissues; fixed VOIs are copies
        (:meth:`VOI.model_copy`), unchanged ones are shared with `cst`.
    """
    fixes = {
        issue.structure: issue.fix_provenance
        for issue in issues
        if issue.fix_provenance and issue.structure is not None
    }

    def fixed_voi(voi: Any) -> Any:
        provenance = fixes.get(getattr(voi, "name", None))
        if provenance is None:
            return voi
        tissue = voi.tissue_properties.adopt_provenance(provenance)
        return voi.model_copy(update={"tissue_properties": tissue})

    if isinstance(cst, Mapping):
        return {
            name: tissue.adopt_provenance(fixes[name]) if name in fixes else tissue
            for name, tissue in cst.items()
        }
    vois = getattr(cst, "vois", None)
    if vois is not None:
        return cst.model_copy(update={"vois": [fixed_voi(voi) for voi in vois]})
    return [fixed_voi(voi) for voi in cst]


# ----------------------------------------------------------------- comparison
@dataclass(frozen=True)
class ParameterComparison:
    """
    One identifying parameter of a tissue compared with the tissue class's.

    Attributes
    ----------
    name : str
        Parameter name, e.g. ``"r_domain"``.
    declared : Any
        Value the tissue class declares.
    value : Any
        Value of the structure's tissue (NaN if its parameter class lacks the parameter).
    matches : bool
        Whether the two are equal, as base-data matching compares them.
    """

    name: str
    declared: Any
    value: Any
    matches: bool


@dataclass(frozen=True, eq=False)
class TissueComparison:
    """
    A structure's tissue compared parameter by parameter with its (closest) tissue class.

    Attributes
    ----------
    structure : str
        Structure name.
    tissue : TissueProperties
        The structure's tissue.
    kernel_set : str or None
        Id of the kernel set (or table) compared against; None if the model reads no
        tissue-dependent base data or none could be selected.
    tissue_class : TissueProperties or None
        The matching class, else the closest one (the ``suggestion`` of the structure's
        :class:`TissueIssue`).
    matched : bool
        Whether the tissue resolves to `tissue_class`, i.e. its parameters match; its
        provenance label may still differ, see `provenance_mismatch`.
    provenance_mismatch : bool
        Whether the tissue's provenance differs from the base data's: with `matched` a
        warning (``provenance_differs``), otherwise an error as the parameter classes
        share no identifying parameter (``type_mismatch``).
    fix_provenance : str or None
        Provenance to relabel the tissue with, see :attr:`TissueIssue.fix_provenance`.
    parameters : tuple[ParameterComparison, ...]
        The parameters matching compares, those the class leaves undeclared excluded; the
        provenance labels are not among them.
    issue : TissueIssue or None
        The tissue-class issue :meth:`BiologicalModel.check_tissues` reports for the
        structure (``type_mismatch``, ``no_matching_class`` or the warning
        ``provenance_differs``), if any.
    """

    structure: str
    tissue: TissueProperties
    kernel_set: Optional[str] = None
    tissue_class: Optional[TissueProperties] = None
    matched: bool = False
    provenance_mismatch: bool = False
    fix_provenance: Optional[str] = None
    parameters: tuple[ParameterComparison, ...] = ()
    issue: Optional[TissueIssue] = None

    @property
    def mismatched(self) -> tuple[str, ...]:
        """Names of the compared parameters that differ."""
        return tuple(p.name for p in self.parameters if not p.matches)


def _parameter_comparisons(
    tissue: TissueProperties, tissue_class: TissueProperties
) -> tuple[ParameterComparison, ...]:
    comparisons = []
    for name in tissue._compare_fields_with(tissue_class):
        declared = getattr(tissue_class, name)
        if isinstance(declared, float) and math.isnan(declared):
            continue
        value = getattr(tissue, name, math.nan)
        comparisons.append(
            ParameterComparison(
                name=name,
                declared=declared,
                value=value,
                matches=TissueProperties._values_equal(value, declared),
            )
        )
    return tuple(comparisons)


def compare_tissue_parameters(
    model: "BiologicalModel",
    machine: Any = None,
    cst: Any = None,
    *,
    provided_quantities: Optional[Sequence[str]] = None,
    engine: Any = None,
) -> list[TissueComparison]:
    """
    Compare every structure's tissue parameter by parameter with its tissue class.

    Selects the kernel set exactly as :func:`resolve_bio_model` does and judges each tissue
    with the same matching and closest-class logic, so the comparison agrees with
    :meth:`BiologicalModel.check_tissues`: ``matched`` is False exactly for the structures
    the check reports a ``type_mismatch`` or ``no_matching_class`` for (unless a
    reference-parameter issue takes precedence there), and ``provenance_mismatch`` is set
    with ``matched`` exactly for those it warns about (``provenance_differs``). Pass the
    whole structure set, since
    the structures' tissues take part in selecting the kernel set.

    Parameters
    ----------
    model : BiologicalModel
        The formalism with its configuration.
    machine : Machine, BioMachineSummary or None
        The machine.
    cst : StructureSet, sequence or mapping, optional
        The structures, see :func:`structure_tissues`.
    provided_quantities, engine
        As for :func:`resolve_bio_model`.

    Returns
    -------
    list[TissueComparison]
        One entry per structure, in structure order. Without tissue-dependent base data
        (no machine, a model reading none, or one unavailable) the entries carry no class.
    """
    structures = structure_tissues(cst)
    kernel_set, unavailable = None, True
    if machine is not None:
        _issues, kernel_set, unavailable = _select_base_data(
            model, machine, structures, provided_quantities, engine
        )
    if unavailable or not _tissue_dependent(kernel_set):
        return [TissueComparison(structure=name, tissue=tissue) for name, tissue in structures]

    comparisons = []
    for name, tissue in structures:
        issue, info = _tissue_class_issue(model, name, tissue, kernel_set)
        tissue_class = info.tissue if info is not None else issue.suggestion
        comparisons.append(
            TissueComparison(
                structure=name,
                tissue=tissue,
                kernel_set=kernel_set.id,
                tissue_class=tissue_class,
                matched=info is not None,
                provenance_mismatch=issue is not None
                and issue.kind in ("type_mismatch", "provenance_differs"),
                fix_provenance=issue.fix_provenance if issue is not None else None,
                parameters=(
                    _parameter_comparisons(tissue, tissue_class)
                    if tissue_class is not None
                    else ()
                ),
                issue=issue,
            )
        )
    return comparisons

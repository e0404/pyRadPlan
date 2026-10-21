"""Biological model options: the (formalism, base data) combinations a machine offers."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional

from pyRadPlan.biology.tissues import TissueProperties

from ._base import BiologicalModel
from ._engine import EngineBioCapability, engine_bio_capability
from ._factory import available_bio_models, create_bio_model
from ._kernel_sets import BioMachineSummary, KernelSetInfo


@dataclass(frozen=True, eq=False)
class BioModelOption:
    """
    One biological model a user can choose for a machine: a formalism and its base data.

    Attributes
    ----------
    model : str
        Identifier of the formalism (e.g. ``"alpha_beta_kernels"``).
    parameters : Mapping[str, Any]
        Model parameters selecting the base data (e.g. ``{"data_model": "LEM_I"}``); empty
        when the machine offers only one choice.
    label : str
        Display label "formalism · provenance", e.g. ``"α/β kernels · LEM I"``.
    formalism : str
        Display name of the formalism.
    description : str
        One-line description of the formalism.
    provenance : str or None
        Canonical provenance of the base data, None for formalisms without tissue-dependent
        base data.
    kernel_set : str or None
        Id of the kernel set (or RBE table) the option reads.
    tissue_classes : tuple[TissueProperties, ...]
        Tissue classes of that base data, i.e. the tissues structures may resolve to.
    available : bool
        Whether the machine can run the option.
    reason : str
        Why the option is unavailable; empty if it is available.
    """

    model: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    label: str = ""
    formalism: str = ""
    description: str = ""
    provenance: Optional[str] = None
    kernel_set: Optional[str] = None
    tissue_classes: tuple[TissueProperties, ...] = ()
    available: bool = True
    reason: str = ""

    @property
    def spec(self) -> dict[str, Any]:
        """Plan specification of the option: ``{"model": ..., **parameters}``."""
        return {"model": self.model, **dict(self.parameters)}

    def create(self) -> BiologicalModel:
        """Instantiate the biological model of this option."""
        return create_bio_model(self.spec)


def _option(
    cls: type[BiologicalModel],
    *,
    parameters: Optional[dict[str, Any]] = None,
    kernel_set: Optional[KernelSetInfo] = None,
    suffix: Optional[str] = None,
    available: bool = True,
    reason: str = "",
) -> BioModelOption:
    formalism = cls.display_name()
    if suffix is None and kernel_set is not None and kernel_set.tissue_dependent:
        suffix = kernel_set.provenance_label
    return BioModelOption(
        model=cls.model,
        parameters=MappingProxyType(dict(parameters or {})),
        label=f"{formalism} · {suffix}" if suffix else formalism,
        formalism=formalism,
        description=cls.description,
        provenance=kernel_set.provenance
        if kernel_set is not None and kernel_set.tissue_dependent
        else None,
        kernel_set=kernel_set.id if kernel_set is not None else None,
        tissue_classes=kernel_set.tissues if kernel_set is not None else (),
        available=available,
        reason=reason,
    )


def _model_reason(
    cls: type[BiologicalModel], parameters: dict[str, Any], summary: BioMachineSummary
) -> str:
    """Why the model cannot run on the machine at all, empty if it can."""
    try:
        model = cls(**parameters)
    except Exception:  # noqa: BLE001 - a model needing arguments is judged by its declarations
        available, message = BiologicalModel.is_available(
            cls, summary.radiation_mode, list(summary.quantities) or None
        )
        return "" if available else message
    resolution = model.resolve(summary)
    return " ".join(issue.message for issue in resolution.errors)


def _selecting_parameters(kernel_set: KernelSetInfo, of_kind: list[KernelSetInfo]) -> dict:
    if len(of_kind) == 1:
        return {}
    provenances = [s.provenance for s in of_kind]
    if provenances.count(kernel_set.provenance) == 1:
        return {"data_model": kernel_set.provenance}
    return {"kernel_set": kernel_set.id}


def _kernel_options(
    cls: type[BiologicalModel], summary: BioMachineSummary
) -> list[BioModelOption]:
    kind = cls.kernel_set_kind
    of_kind = [s for s in summary.kernel_sets.values() if s.kind == kind]
    if kind == "let":
        if len(of_kind) <= 1:
            reason = _model_reason(cls, {}, summary)
            return [_option(cls, kernel_set=None, available=not reason, reason=reason)]
        options = []
        for kernel_set in of_kind:
            if kernel_set.variant:
                parameters = {"quantity_variants": {"let": kernel_set.variant}}
            elif kernel_set.default:
                parameters = {}
            else:
                options.append(
                    _option(
                        cls,
                        kernel_set=kernel_set,
                        suffix=kernel_set.id,
                        available=False,
                        reason=f"LET kernel set '{kernel_set.id}' declares no variant to "
                        "select it by.",
                    )
                )
                continue
            reason = _model_reason(cls, parameters, summary)
            options.append(
                _option(
                    cls,
                    parameters=parameters,
                    kernel_set=kernel_set,
                    suffix=kernel_set.variant or kernel_set.id,
                    available=not reason,
                    reason=reason,
                )
            )
        return options

    if not of_kind:
        reason = _model_reason(cls, {}, summary)
        return [_option(cls, available=False, reason=reason)]
    options = []
    for kernel_set in of_kind:
        parameters = _selecting_parameters(kernel_set, of_kind)
        reason = _model_reason(cls, parameters, summary)
        options.append(
            _option(
                cls,
                parameters=parameters,
                kernel_set=kernel_set,
                available=not reason,
                reason=reason,
            )
        )
    return options


def _table_options(cls: Any, summary: BioMachineSummary) -> list[BioModelOption]:
    tables = cls.quantity_table_kernel_sets()
    missing = [q for q in cls.required_quantities if q not in summary.quantities]
    base_reason = ""
    if missing and not summary.has_fluence_spectrum:
        base_reason = (
            f"The machine provides no {', '.join(missing)} data: averaging the tables needs "
            "fragment fluence spectra in the pencil-beam kernels."
        )
    if summary.radiation_mode and summary.radiation_mode not in cls.possible_radiation_modes:
        base_reason = f"Radiation mode '{summary.radiation_mode}' is invalid for '{cls.model}'."
    if not tables:
        return [_option(cls, available=False, reason="No quantity table is available.")]
    default = getattr(cls, "DEFAULT_QUANTITY_TABLE", None)
    options = []
    for table in tables:
        parameters = {} if table.id == default else {"quantity_table_name": table.id}
        reason = " ".join([base_reason, *table.problems]).strip()
        options.append(
            _option(
                cls,
                parameters=parameters,
                kernel_set=table,
                available=not reason,
                reason=reason,
            )
        )
    return options


def _restricted_by_engine(
    option: BioModelOption, cls: type[BiologicalModel], capability: EngineBioCapability
) -> BioModelOption:
    reason = capability.model_reason(
        cls, quantity_variants=dict(option.parameters).get("quantity_variants") or {}
    )
    if not reason:
        return option
    return replace(option, available=False, reason=reason)


def bio_model_options(
    radiation_mode: str, machine: Any = None, engine: Any = None
) -> list[BioModelOption]:
    """
    Biological models available for a radiation mode and, optionally, a machine and engine.

    With a machine, there is one entry per (formalism, kernel set or RBE table) the machine
    could offer, labelled "formalism · provenance" (e.g. ``"α/β kernels · LEM I"``,
    ``"α/β kernels · model not declared"``); entries the machine cannot run are included
    with ``available=False`` and the reason. Without a machine, the plain formalism list.

    With a dose engine, an entry is also unavailable if the engine cannot compute it, e.g.
    the kernel formalisms with a Monte Carlo engine that does not use the machine's
    pencil-beam kernels, and the reason names the engine. Formalisms reading a quantity the
    engine scores itself (e.g. LET) are listed for what the engine scores, independently of
    the machine's kernels.

    Parameters
    ----------
    radiation_mode : str
        Radiation mode (``"protons"``, ``"carbon"``, ...).
    machine : Machine, BioMachineSummary or None
        The machine, or a cacheable summary of it (:meth:`BioMachineSummary.from_machine`).
    engine : str, type, DoseEngineBase, EngineBioCapability or None
        The dose engine (short name, class or instance), see
        :meth:`~pyRadPlan.dose.engines.DoseEngineBase.bio_capability`. None: the machine
        alone decides.

    Returns
    -------
    list[BioModelOption]
        ``none`` first, then by formalism identifier and base data.
    """
    capability = engine_bio_capability(engine)
    if capability is not None:
        capability = capability.for_radiation_mode(radiation_mode)
    summary = BioMachineSummary.from_machine(machine) if machine is not None else None
    view = (
        capability.machine_view(summary)
        if capability is not None and summary is not None
        else summary
    )
    options: list[BioModelOption] = []
    for cls in available_bio_models(radiation_mode):
        if summary is None:
            entries = [_option(cls)]
        else:
            # Formalisms the engine supplies are offered for what it reads or scores; the
            # others for the machine's base data, unavailable because of the engine.
            supplied = capability is None or all(
                capability.supplies(q) for q in cls.required_quantities
            )
            source = view if supplied else summary
            if cls.kernel_set_kind == "rbe_table" and hasattr(cls, "quantity_table_kernel_sets"):
                entries = _table_options(cls, source)
            elif cls.kernel_set_kind is not None:
                entries = _kernel_options(cls, source)
            else:
                reason = _model_reason(cls, {}, source)
                entries = [_option(cls, available=not reason, reason=reason)]
        if capability is not None:
            entries = [_restricted_by_engine(o, cls, capability) for o in entries]
        options.extend(entries)
    return options

"""
What a dose engine can supply to a biological model.

A pencil-beam engine evaluates the biological kernel data tabulated in the machine (α/β
kernels, z* kernels, LET, fluence spectra); a Monte Carlo engine does not use these kernels
and supplies only what it scores itself (e.g. LET). Whether a formalism can be computed
therefore depends on the dose engine as well as on the machine.
:class:`EngineBioCapability` declares this per engine class, and
:func:`~pyRadPlan.biology.models.bio_model_options` and
:meth:`~pyRadPlan.biology.models.BiologicalModel.check_tissues` take it into account.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Optional

from ._kernel_sets import KERNEL_SET_QUANTITIES, BioMachineSummary, KernelSetInfo

#: How a machine quantity is named in messages: its kernel data, and the quantity if an
#: engine may score it itself.
_QUANTITY_LABELS: dict[str, tuple[str, Optional[str]]] = {
    "alpha": ("α/β kernels", None),
    "beta": ("α/β kernels", None),
    "zs": ("z* kernels", None),
    "let": ("LET kernels", "LET"),
    "fluence": ("fluence spectra", None),
}


def _base(quantity: str) -> str:
    """Quantity without its variant: ``"let:LETt"`` -> ``"let"``."""
    return quantity.split(":", 1)[0]


def _kind_of(quantity: str) -> Optional[str]:
    return next((kind for kind, names in KERNEL_SET_QUANTITIES.items() if quantity in names), None)


@dataclass(frozen=True)
class EngineBioCapability:
    """
    Quantities a dose engine can supply to a biological model.

    Declared by each dose engine class, see
    :meth:`~pyRadPlan.dose.engines.DoseEngineBase.bio_capability`.

    Attributes
    ----------
    engine : str
        Name of the engine, used in messages (e.g. ``"FRED"``).
    kernel_quantities : frozenset[str] or None
        Machine quantities (as in :meth:`Machine.provided_quantities`, without variant) the
        engine evaluates from the machine's pencil-beam kernels, e.g.
        ``{"alpha", "beta", "zs", "let", "fluence"}``. None: every quantity the machine
        tabulates. Empty for engines not using the machine's kernels (Monte Carlo).
    scored_quantities : tuple[str, ...]
        Quantities the engine computes itself, independently of the machine, as
        ``"<quantity>"`` or ``"<quantity>:<variant>"`` (e.g. ``"let:LETd"``). A scored
        quantity replaces the machine's kernel data of that quantity.
    computes_bio_influence : bool
        Whether the engine evaluates biological models into influence matrices. Without,
        only models without outputs (``none``, a constant RBE) can be computed.
    scored_radiation_modes : frozenset[str] or None
        Radiation modes the engine scores ``scored_quantities`` for (e.g. TOPAS scores the
        LET of protons only); None for every mode. See :meth:`for_radiation_mode`.
    radiation_mode : str
        The radiation mode a capability was narrowed to by :meth:`for_radiation_mode`,
        empty if it was not.
    """

    engine: str = ""
    kernel_quantities: Optional[frozenset[str]] = None
    scored_quantities: tuple[str, ...] = ()
    computes_bio_influence: bool = True
    scored_radiation_modes: Optional[frozenset[str]] = None
    radiation_mode: str = ""

    def for_radiation_mode(self, radiation_mode: Optional[str]) -> EngineBioCapability:
        """
        Narrow the capability to plans of one radiation mode.

        Drops the scored quantities if the engine does not score them for `radiation_mode`;
        unchanged for an unknown (empty) mode.
        """
        if (
            not radiation_mode
            or not self.scored_quantities
            or self.scored_radiation_modes is None
            or radiation_mode in self.scored_radiation_modes
        ):
            return self
        return replace(self, scored_quantities=(), radiation_mode=radiation_mode)

    def reads(self, quantity: str) -> bool:
        """Whether the engine evaluates ``quantity`` from the machine's kernels."""
        base = _base(quantity)
        if base in self.scored:
            return False
        return self.kernel_quantities is None or base in self.kernel_quantities

    def _reads_kind(self, kind: str) -> bool:
        names = KERNEL_SET_QUANTITIES.get(kind)
        if not names:
            return self.kernel_quantities is None
        return all(self.reads(name) for name in names)

    @property
    def scored(self) -> dict[str, Optional[str]]:
        """Scored quantities (without variant) and the variant each is scored as."""
        scored: dict[str, Optional[str]] = {}
        for quantity in self.scored_quantities:
            base, _, variant = quantity.partition(":")
            scored[base] = variant or scored.get(base)
        return scored

    def supplies(self, quantity: str) -> bool:
        """Whether the engine can supply ``quantity`` (given a machine tabulating it)."""
        base = _base(quantity)
        return base == "physical_dose" or base in self.scored or self.reads(base)

    @property
    def label(self) -> str:
        """``"The dose engine <name>"``, the subject of messages."""
        return f"The dose engine {self.engine}" if self.engine else "The dose engine"

    # ------------------------------------------------------------ machine view
    def quantities(self, machine: Any = None) -> list[str]:
        """
        Quantities available to a biological model with this engine and machine.

        The machine's quantities the engine reads from its kernels (including variants,
        ``"let:LETt"``), followed by the ones it scores; always ``physical_dose``.
        """
        return list(self.machine_view(machine).quantities)

    def machine_view(self, machine: Any = None) -> BioMachineSummary:
        """
        Return the machine as the engine sees it.

        Keeps the kernel sets the engine reads; a quantity the engine scores is described
        by a kernel set of its own (id ``"scored_<kind>"``, variant as scored) instead of
        the machine's.

        Parameters
        ----------
        machine : Machine, BioMachineSummary or None
            The machine; None for an engine that reads no machine data.
        """
        summary = (
            BioMachineSummary.from_machine(machine) if machine is not None else BioMachineSummary()
        )
        narrowed = self.for_radiation_mode(summary.radiation_mode)
        if narrowed is not self:
            return narrowed.machine_view(summary)
        sets: dict[str, KernelSetInfo] = {
            set_id: kernel_set
            for set_id, kernel_set in summary.kernel_sets.items()
            if self._reads_kind(kernel_set.kind)
        }
        quantities = ["physical_dose"]
        quantities += [q for q in summary.quantities if q not in quantities and self.reads(q)]
        for quantity, variant in self.scored.items():
            if quantity not in quantities:
                quantities.append(quantity)
            kind = _kind_of(quantity)
            if kind is None:
                continue
            set_id = f"scored_{kind}"
            sets[set_id] = KernelSetInfo(
                id=set_id,
                kind=kind,
                fields=KERNEL_SET_QUANTITIES[kind],
                variant=variant,
                reference=f"Scored by the dose engine {self.engine}.",
                default=True,
                declared=True,
            )
        fluence = (summary.has_fluence_spectrum and self.reads("fluence")) or (
            "fluence" in self.scored
        )
        return replace(
            summary,
            kernel_sets=MappingProxyType(sets),
            quantities=tuple(quantities),
            has_fluence_spectrum=fluence,
        )

    # ------------------------------------------------------------ models
    def model_reason(self, model: Any, quantity_variants: Optional[dict] = None) -> str:
        """
        Why the engine cannot compute a biological model, empty if it can.

        Judged by the model's declarations alone: the quantities it requires, whether it has
        outputs, and the variants it is configured for. Whether the machine tabulates what
        the engine reads is a question of the machine (see :meth:`machine_view`).

        Parameters
        ----------
        model : BiologicalModel or type[BiologicalModel]
            The model (an instance is needed to judge its ``quantity_variants``).
        quantity_variants : dict, optional
            Variants to judge instead of the model's ``quantity_variants``.
        """
        reasons: list[str] = []
        for quantity in dict.fromkeys(_base(q) for q in model.required_quantities):
            if self.supplies(quantity):
                continue
            kernels, scorable = _QUANTITY_LABELS.get(quantity, (f"{quantity} kernels", None))
            if scorable and self.radiation_mode and self.scored_radiation_modes is not None:
                modes = ", ".join(sorted(self.scored_radiation_modes))
                reason = (
                    f"{self.label} neither uses the machine's {kernels} nor scores "
                    f"{scorable} for {self.radiation_mode} (only for {modes})."
                )
            elif scorable:
                reason = (
                    f"{self.label} neither uses the machine's {kernels} nor scores {scorable}."
                )
            else:
                reason = f"{self.label} does not use the machine's {kernels}."
            if reason not in reasons:
                reasons.append(reason)

        if quantity_variants is None and not isinstance(model, type):
            quantity_variants = getattr(model, "quantity_variants", None)
        for quantity, wanted in (quantity_variants or {}).items():
            if quantity not in self.scored or not wanted:
                continue
            scored = self.scored[quantity]
            if (scored or "").casefold() != str(wanted).casefold():
                noun = _QUANTITY_LABELS.get(quantity, ("", None))[1] or quantity
                reasons.append(
                    f"{self.label} scores {noun} as {scored or 'an undeclared variant'}, "
                    f"not {wanted}."
                )

        if not reasons and model.output_quantities and not self.computes_bio_influence:
            reasons.append(f"{self.label} does not compute biological influence matrices.")
        return " ".join(reasons)


def engine_bio_capability(engine: Any) -> Optional[EngineBioCapability]:
    """
    Biological capability of a dose engine.

    Parameters
    ----------
    engine : str, type, DoseEngineBase, EngineBioCapability or None
        A registered engine's short name, an engine class or instance (anything with a
        ``bio_capability()`` method), or a capability.

    Returns
    -------
    EngineBioCapability or None
        None for ``engine=None``.

    Raises
    ------
    ValueError
        ``engine`` names no registered dose engine.
    TypeError
        ``engine`` declares no capability.
    """
    if engine is None or isinstance(engine, EngineBioCapability):
        return engine
    if isinstance(engine, str):
        from pyRadPlan.dose.engines._factory import DOSE_ENGINES  # noqa: PLC0415 - cycle

        if engine not in DOSE_ENGINES:
            raise ValueError(
                f"Unknown dose engine '{engine}'. Registered engines: {sorted(DOSE_ENGINES)}."
            )
        engine = DOSE_ENGINES[engine]
    capability = getattr(engine, "bio_capability", None)
    if not callable(capability):
        raise TypeError(f"{engine!r} declares no biological capability (bio_capability()).")
    return capability()

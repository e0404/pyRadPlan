"""Resolver that builds the dependency graph of fluence-dependent quantities."""

import inspect
from typing import Iterable, Any

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend

from ._base import FluenceDependentQuantity


class QuantityResolver:
    """
    Build and deduplicate the graph of :class:`FluenceDependentQuantity` instances for a Dij.

    Each quantity (by identifier) is instantiated at most once per resolver.
    Sub-quantities shared between parents (e.g. ``alpha_dose`` requested both directly and via ``effect``)
    are produced as a single shared instance. The resolver also performs the namespace conversion of the dij
    once so that every quantity in the graph shares the converted dij.

    Dynamic Dependency Injection (via Signature Inspection):
    When instantiating a quantity class, the resolver inspects the quantity's ``__init__`` signature.
    It automatically supplies required dependencies from:
    1. Problem-level context objects (e.g., ``cst``, ``pln``, ``stf``, ``ct``).
    2. Objective-level parameters that the objective explicitly opts into via its
       ``quantity_params`` ClassVar (e.g., ``risk_voi_name``).
       Each quantity is instantiated once, so all objectives targeting it must agree on those
       values; conflicting values raise. If you need multiple parameterizations, subclass the
       quantity with a unique identifier.
    3. Explicit ``quantity_kwargs`` passed to the resolver.

    Mode resolution per class:
    - If the dij has an attribute matching the class ``identifier``, mode is ``"direct"``
      and dependencies are not constructed.
    - Otherwise mode is ``"indirect"``; both ``required_dependencies`` and
      ``optional_dependencies`` are resolved recursively.
    - If neither path is available, a ``ValueError`` is raised.
    """

    def __init__(
        self,
        dij: Dij,
        *,
        _dij_already_in_namespace: bool = False,
        quantity_kwargs: dict | None = None,
        context: dict | None = None,
        objective_list: list | None = None,
    ):
        xp = compute_backend.choose_array_api_namespace()
        device = compute_backend.choose_device(xp)

        if _dij_already_in_namespace:
            self._dij = dij
        else:
            self._dij = dij.to_namespace(xp, device=device)

        self.quantity_kwargs = quantity_kwargs or {}
        self.context = context or {}
        self.objective_list = objective_list or []
        self._instances: dict[str, FluenceDependentQuantity] = {}
        # In-progress identifiers used for cycle detection.
        self._resolving: set[str] = set()

    @property
    def dij(self) -> Dij:
        """The namespace-converted dij that every resolved quantity will share."""
        return self._dij

    @property
    def instances(self) -> dict[str, FluenceDependentQuantity]:
        """All quantity instances resolved so far, keyed by identifier."""
        return self._instances

    def get(self, identifier: str) -> FluenceDependentQuantity:
        """Resolve a single quantity by identifier, using dynamic dependency injection."""
        if identifier in self._instances:
            return self._instances[identifier]

        # Lazy import to avoid a circular import at module load time.
        from . import QUANTITIES  # noqa: PLC0415

        try:
            cls = QUANTITIES[identifier]
        except KeyError as exc:
            raise ValueError(f"Unknown quantity identifier: {identifier!r}.") from exc

        # 1. Dynamically discover keyword arguments required by cls.__init__ via signature inspection
        sig = inspect.signature(cls.__init__)
        kwargs: dict[str, Any] = {}

        # A. Inject from problem-level context (e.g. cst, pln, stf, ct) if accepted by __init__
        for param in sig.parameters:
            if param in self.context and param not in kwargs:
                kwargs[param] = self.context[param]

        # B. Inject parameters the targeting objectives explicitly opt into (e.g. risk_voi_name).
        # The quantity is built once, so disagreeing objectives cannot both be honoured.
        from_objectives: dict[str, Any] = {}
        for _, objs in self.objective_list:
            for obj in objs:
                if getattr(obj, "quantity", None) != identifier:
                    continue
                for param in type(obj).quantity_params:
                    if param not in sig.parameters or param in kwargs:
                        continue
                    value = getattr(obj, param)
                    if param in from_objectives and from_objectives[param] != value:
                        raise ValueError(
                            f"Objectives on quantity {identifier!r} disagree on {param!r}: "
                            f"{from_objectives[param]!r} vs {value!r}. The quantity is "
                            "instantiated once per identifier, so only one value can apply. "
                            "Give all objectives on this quantity the same value, or subclass "
                            "the quantity with its own identifier."
                        )
                    from_objectives[param] = value
        kwargs.update(from_objectives)

        # C. Explicit overrides passed directly to QuantityResolver
        if identifier in self.quantity_kwargs:
            kwargs.update(self.quantity_kwargs[identifier])

        if identifier in self._resolving:
            raise ValueError(
                f"Cyclic dependency detected while resolving quantity {identifier!r}."
            )

        self._resolving.add(identifier)
        try:
            mode = self._choose_mode(cls, identifier)
            deps = self._resolve_dependencies(cls, mode)
            inst = cls(self._dij, mode=mode, dependencies=deps, **kwargs)
        finally:
            self._resolving.discard(identifier)

        self._instances[identifier] = inst
        return inst

    def resolve(self, identifiers: Iterable[str]) -> list[FluenceDependentQuantity]:
        """Resolve every identifier and return them in input order."""
        return [self.get(i) for i in identifiers]

    def _choose_mode(self, cls: type[FluenceDependentQuantity], identifier: str) -> str:
        if getattr(self._dij, identifier, None) is not None:
            return "direct"
        if cls.required_dependencies or cls.optional_dependencies:
            return "indirect"
        raise ValueError(
            f"Cannot resolve quantity {identifier!r}: dij has no attribute "
            f"{identifier!r} and no dependencies are declared."
        )

    def _resolve_dependencies(
        self, cls: type[FluenceDependentQuantity], mode: str
    ) -> dict[str, FluenceDependentQuantity]:
        if mode == "direct":
            return {}
        deps: dict[str, FluenceDependentQuantity] = {}
        for dep_id in cls.required_dependencies:
            deps[dep_id] = self.get(dep_id)
        for dep_id in cls.optional_dependencies:
            deps[dep_id] = self.get(dep_id)
        return deps

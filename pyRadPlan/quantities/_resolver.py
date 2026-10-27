"""Resolver that builds the dependency graph of fluence-dependent quantities."""

import inspect
import logging
from typing import Any, Iterable, Optional

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend

from ._base import FluenceDependentQuantity


logger = logging.getLogger(__name__)


class QuantityResolver:
    """
    Build and deduplicate the graph of :class:`FluenceDependentQuantity` instances for a Dij.

    Each quantity is instantiated at most once per resolver and parameterization (see 2.).
    Sub-quantities shared between parents (e.g. ``alpha_dose`` requested both directly and via ``effect``)
    are produced as a single shared instance. The resolver also performs the namespace conversion of the dij
    once so that every quantity in the graph shares the converted dij.

    Dynamic Dependency Injection (via Signature Inspection):
    When instantiating a quantity class, the resolver inspects the quantity's ``__init__`` signature.
    It automatically supplies required dependencies from:
    1. Problem-level context (e.g., ``cst``, ``ct``, ``stf``, ``num_of_fractions``). Planning
       problems pass the ``cst`` as given, before overlap priorities are applied.
    2. Objective-level parameters that the objective explicitly opts into via its
       ``quantity_params`` ClassVar (e.g., ``risk_voi_name``), passed to :meth:`get`.
       Each distinct parameterization is its own instance under its own :meth:`key`
       (e.g. ``"polo[risk_voi_name='Core']"``); its dependencies are still shared.
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
        scenarios=None,
        _dij_already_in_namespace: bool = False,
        quantity_kwargs: dict | None = None,
        context: dict | None = None,
    ):
        xp = compute_backend.choose_array_api_namespace()
        device = compute_backend.choose_device(xp)

        if _dij_already_in_namespace:
            self._dij = dij
        else:
            self._dij = dij.to_namespace(xp, device=device)
        # Scenario indices every resolved quantity (and its dependencies) computes.
        self._scenarios = scenarios

        self.quantity_kwargs = quantity_kwargs or {}
        self.context = context or {}
        self._instances: dict[str, FluenceDependentQuantity] = {}
        # In-progress identifiers used for cycle detection.
        self._resolving: set[str] = set()

    @property
    def dij(self) -> Dij:
        """The namespace-converted dij that every resolved quantity will share."""
        return self._dij

    @property
    def instances(self) -> dict[str, FluenceDependentQuantity]:
        """All quantity instances resolved so far, keyed by :meth:`key`."""
        return self._instances

    def key(self, identifier: str, params: Optional[dict[str, Any]] = None) -> str:
        """
        Key of the instance :meth:`get` returns for *identifier* and objective *params*.

        Parameters the quantity does not accept, or that equal its default, do not distinguish
        instances, so the key is the bare identifier unless an objective sets one differently.
        """
        params = self._own_params(self._class(identifier), params)
        if not params:
            return identifier
        return f"{identifier}[{', '.join(f'{p}={v!r}' for p, v in sorted(params.items()))}]"

    def get(
        self, identifier: str, params: Optional[dict[str, Any]] = None
    ) -> FluenceDependentQuantity:
        """
        Resolve a quantity by identifier, using dynamic dependency injection.

        *params* are the objective-level parameters of the quantity (see the class docstring);
        each distinct parameterization is resolved into its own instance.
        """
        key = self.key(identifier, params)
        if key in self._instances:
            return self._instances[key]

        cls = self._class(identifier)
        sig = inspect.signature(cls.__init__)

        # A. Inject problem-level context (e.g. cst, num_of_fractions) if accepted by __init__
        kwargs: dict[str, Any] = {p: self.context[p] for p in sig.parameters if p in self.context}

        # B. Parameters the targeting objectives explicitly opt into (e.g. risk_voi_name)
        kwargs.update(self._own_params(cls, params))

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
            inst = cls(
                self._dij, mode=mode, dependencies=deps, scenarios=self._scenarios, **kwargs
            )
        finally:
            self._resolving.discard(identifier)

        self._instances[key] = inst
        return inst

    def resolve(self, identifiers: Iterable[str]) -> list[FluenceDependentQuantity]:
        """Resolve every identifier and return them in input order."""
        return [self.get(i) for i in identifiers]

    def reachable(
        self, roots: Iterable[FluenceDependentQuantity]
    ) -> list[FluenceDependentQuantity]:
        """
        Return the instances ``roots`` depend on (including themselves), in resolution order.

        A quantity may drop a resolved optional dependency once it has chosen its computation
        path (e.g. the ``effect`` of a constant-RBE ``rbe_x_dose``). Such instances remain in
        :attr:`instances`, but are not reachable from any root and need not be evaluated.
        """
        seen: set[int] = set()
        stack = list(roots)
        while stack:
            quantity = stack.pop()
            if id(quantity) not in seen:
                seen.add(id(quantity))
                stack.extend(quantity.dependencies.values())
        return [q for q in self._instances.values() if id(q) in seen]

    def try_get(self, identifier: str) -> Optional[FluenceDependentQuantity]:
        """Resolve ``identifier`` or return ``None`` if it cannot be computed for this dij."""
        try:
            return self.get(identifier)
        except ValueError as exc:
            logger.debug("Optional quantity %r not resolvable: %s", identifier, exc)
            return None

    @staticmethod
    def _class(identifier: str) -> type[FluenceDependentQuantity]:
        # Lazy import to avoid a circular import at module load time.
        from . import QUANTITIES  # noqa: PLC0415

        try:
            return QUANTITIES[identifier]
        except KeyError as exc:
            raise ValueError(f"Unknown quantity identifier: {identifier!r}.") from exc

    def _own_params(
        self, cls: type[FluenceDependentQuantity], params: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """The objective parameters *cls* accepts and that differ from its defaults."""
        sig = inspect.signature(cls.__init__).parameters
        return {
            p: v
            for p, v in (params or {}).items()
            if p in sig and p not in self.context and v != sig[p].default
        }

    def _choose_mode(self, cls: type[FluenceDependentQuantity], identifier: str) -> str:
        if getattr(self._dij, identifier, None) is not None:
            return "direct"
        if cls.has_indirect_path(self._dij):
            return "indirect"
        raise ValueError(
            f"Cannot resolve quantity {identifier!r}: dij has no attribute "
            f"{identifier!r} and no usable dependency path."
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
            dep = self.try_get(dep_id)
            if dep is not None:
                deps[dep_id] = dep
        return deps

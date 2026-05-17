"""Resolver that builds the dependency graph of fluence-dependent quantities."""

from typing import Iterable

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend

from ._base import FluenceDependentQuantity


class QuantityResolver:
    """
    Build and deduplicate the graph of :class:`FluenceDependentQuantity` instances for a Dij.

    Each identifier is instantiated at most once per resolver. Sub-quantities shared between
    parents (e.g. ``alpha_dose`` requested both directly and via ``effect``) are produced as
    a single shared instance. The resolver also performs the namespace conversion of the dij
    once so that every quantity in the graph shares the converted dij.

    Mode resolution per class:

    - If the dij has an attribute matching the class ``identifier``, mode is ``"direct"``
      and dependencies are not constructed.
    - Otherwise mode is ``"indirect"``; both ``required_dependencies`` and
      ``optional_dependencies`` are resolved recursively.
    - If neither path is available, a ``ValueError`` is raised.
    """

    def __init__(self, dij: Dij, *, _dij_already_in_namespace: bool = False):
        xp = compute_backend.choose_array_api_namespace()
        device = compute_backend.choose_device(xp)

        if _dij_already_in_namespace:
            self._dij = dij
        else:
            self._dij = dij.to_namespace(xp, device=device)
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
        """Resolve a single quantity by identifier, reusing the cached instance if present."""
        if identifier in self._instances:
            return self._instances[identifier]
        if identifier in self._resolving:
            raise ValueError(
                f"Cyclic dependency detected while resolving quantity {identifier!r}."
            )

        # Lazy import to avoid a circular import at module load time.
        from . import QUANTITIES  # noqa: PLC0415

        try:
            cls = QUANTITIES[identifier]
        except KeyError as exc:
            raise ValueError(f"Unknown quantity identifier: {identifier!r}.") from exc

        self._resolving.add(identifier)
        try:
            mode = self._choose_mode(cls, identifier)
            deps = self._resolve_dependencies(cls, mode)
            inst = cls(self._dij, mode=mode, dependencies=deps)
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

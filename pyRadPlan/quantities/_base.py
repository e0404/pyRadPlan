from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Union

import logging

import numpy as np
import array_api_compat

from ..core.xp_utils.typing import Array, ArrayNamespace


import pint

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend


logger = logging.getLogger(__name__)


class RTQuantity(ABC):
    # TODO: missing description
    name: ClassVar[str]
    identifier: ClassVar[str]
    unit: ClassVar[pint.Unit]
    dim: ClassVar[int]  # To differentiate between scalar and vector quantities
    #: Whether the quantity describes a biological effect (e.g. RBE-weighted dose or LQ
    #: effect) rather than a physical one.
    biological: ClassVar[bool] = False
    #: Exponent ``p`` with which the quantity accumulates over ``n`` identical fractions: the
    #: total-course value is ``n**p`` times the per-fraction value (1 for additive quantities
    #: such as dose or effect, 0 for intensive ones such as LET). ``None`` (the default for
    #: custom quantities) declares no rule, and converting between per-fraction and
    #: total-course values then raises instead of guessing.
    fraction_exponent: ClassVar[Optional[float]] = None
    precision: str

    def __init__(self, dij: Dij, scenarios=None):
        if scenarios is None:
            scenarios = self.default_scenarios(dij)
        self.scenarios = np.asarray(scenarios, dtype=np.int64)

    @staticmethod
    def default_scenarios(dij: Dij) -> list[int]:
        """
        Scenario indices a quantity evaluates when none are given.

        With a scenario set on the dij, every scenario of the set (computed and derived).
        Without, every non-empty slot of the ``physical_dose`` container (``[0]`` if there is
        none).
        """
        scenario_set = getattr(dij, "scenarios", None)
        if scenario_set is not None:
            return list(range(len(scenario_set)))
        container = getattr(dij, "physical_dose", None)
        if container is None:
            return [0]
        filled = [i for i in range(container.size) if container.flat[i] is not None]
        return filled or [0]

    @classmethod
    def total_course_factor(cls, num_of_fractions: int) -> float:
        """
        Factor from a per-fraction value of the quantity to its total-course value.

        Parameters
        ----------
        num_of_fractions : int
            Number of identical fractions in the course.

        Returns
        -------
        float
            ``num_of_fractions ** fraction_exponent``.

        Raises
        ------
        ValueError
            If ``num_of_fractions`` is not positive, or if it is not 1 and the quantity does
            not declare a :attr:`fraction_exponent`.
        """
        if num_of_fractions <= 0:
            raise ValueError("Number of fractions must be greater than zero.")
        if num_of_fractions == 1:
            return 1.0
        if cls.fraction_exponent is None:
            raise ValueError(
                f"Quantity {cls.identifier!r} declares no fraction_exponent, so its values "
                "cannot be converted between one fraction and the total course."
            )
        return float(num_of_fractions) ** cls.fraction_exponent


class FluenceDependentQuantity(RTQuantity, ABC):
    """
    Base class for quantities that depend on fluence distributions.

    Concrete subclasses declare how they relate to a :class:`Dij` via two ClassVars:

    - :attr:`required_dependencies`: identifiers of upstream quantities that are always
      needed to compute this quantity. The resolver must produce them; failure raises.
    - :attr:`optional_dependencies`: identifiers of upstream quantities used as a fallback
      computation path when no direct Dij matrix is available.
    - :attr:`indirect_dij_fields`: Dij attributes (e.g. ``alphax``) the indirect path reads
      directly; if any is missing, the indirect path is unavailable.

    The resolution mode (``"direct"`` vs ``"indirect"``) is decided at instantiation time:
    if the dij carries an attribute whose name matches this quantity's ``identifier``, the
    direct path wins; otherwise the indirect path is used and ``required + optional`` deps
    must resolve.  This enforces a consistent naming convention: a Dij matrix attribute must
    share its name with the quantity identifier it represents.
    """

    # ClassVar metadata describing how this quantity is computed.
    required_dependencies: ClassVar[tuple[str, ...]] = ()
    optional_dependencies: ClassVar[tuple[str, ...]] = ()
    indirect_dij_fields: ClassVar[tuple[str, ...]] = ()

    @classmethod
    def has_indirect_path(cls, dij: Dij) -> bool:
        """Whether the dependency-based path is available for ``dij``."""
        if not (cls.required_dependencies or cls.optional_dependencies):
            return False
        return all(getattr(dij, field, None) is not None for field in cls.indirect_dij_fields)

    array_backend: ArrayNamespace

    def __init__(
        self,
        dij: Dij,
        *,
        mode: Optional[str] = None,
        dependencies: Optional[dict[str, "FluenceDependentQuantity"]] = None,
        scenarios=None,
    ):
        super().__init__(dij, scenarios=scenarios)
        xp = compute_backend.choose_array_api_namespace()
        device = compute_backend.choose_device(xp)

        logger.info("Optimization will use array backend: %s", xp.__name__)

        self.array_backend: ArrayNamespace = xp

        # The resolver pre-converts the dij to the target namespace and supplies
        # `mode` + `dependencies`. When invoked directly (e.g. from tests) we do the
        # conversion ourselves and build deps lazily via the resolver.
        if mode is None:
            self._dij = dij.to_namespace(xp, device=device)
            self.device = device
            self._mode = self._choose_mode()
            self._deps: dict[str, FluenceDependentQuantity] = dict(dependencies or {})
            if self._mode == "indirect" and not self._deps:
                # Lazy import to avoid circular dependency at module import time.
                from ._resolver import QuantityResolver  # noqa: PLC0415

                resolver = QuantityResolver(
                    self._dij, scenarios=self.scenarios, _dij_already_in_namespace=True
                )
                for dep_id in self.required_dependencies:
                    self._deps[dep_id] = resolver.get(dep_id)
                for dep_id in self.optional_dependencies:
                    dep = resolver.try_get(dep_id)
                    if dep is not None:
                        self._deps[dep_id] = dep
        else:
            if mode not in ("direct", "indirect"):
                raise ValueError(f"Unknown quantity mode: {mode!r}")
            self._dij = dij
            self._mode = mode
            self._deps = dict(dependencies or {})
            self.device = device

        self._validate_dependencies()

        # dtype is inferred from the matrix actually driving the computation in
        # direct mode, and from the canonical `physical_dose` container otherwise.
        self._dtype = self._infer_dtype(xp)

        # Fluence cache for derivative calculation
        self._w_cache: Union[Array, None] = None
        self._w_grad_cache: Union[Array, None] = None
        # Quantity vector cache (shape mirrors the scenario container layout)
        self._q_cache = np.empty_like(getattr(self._dij, "physical_dose"), dtype=object)
        self._qgrad_cache = np.empty_like(self._q_cache)
        self._evaluation_plan_cache: Optional[list] = None

    @property
    def mode(self) -> str:
        """The resolution mode chosen for this instance: ``"direct"`` or ``"indirect"``."""
        return self._mode

    @property
    def dependencies(self) -> dict[str, "FluenceDependentQuantity"]:
        """Mapping of identifier to resolved upstream quantity instance."""
        return self._deps

    def _choose_mode(self) -> str:
        """Pick the resolution mode for this dij; direct wins when both are available."""
        if getattr(self._dij, self.identifier, None) is not None:
            return "direct"
        if self.has_indirect_path(self._dij):
            return "indirect"
        raise ValueError(
            f"Quantity {type(self).__name__!r} cannot be computed: dij has no attribute "
            f"{self.identifier!r} and no usable dependency path."
        )

    def _validate_dependencies(self) -> None:
        if self._mode == "indirect":
            for dep_id in self.required_dependencies:
                if dep_id not in self._deps:
                    raise ValueError(
                        f"Required dependency {dep_id!r} missing for {type(self).__name__!r}."
                    )

    def _infer_dtype(self, xp: ArrayNamespace):
        if self._mode == "direct":
            container = getattr(self._dij, self.identifier)
            sample = container.flat[0]
        else:
            sample = self._dij.physical_dose.flat[0]
        try:
            return getattr(xp, sample.dtype.name)
        except AttributeError:
            return sample.dtype

    def __call__(self, fluence: Array) -> Array:
        """
        Make the quantity callable by calling the compute method.

        Parameters
        ----------
        fluence : Array
            Fluence vector.

        Returns
        -------
        NDArray
            Quantity vector.
        """

        return self.compute(fluence)

    def compute(self, fluence: Array) -> Array:
        """
        Forward calculation of the quantity from the fluence.

        Parameters
        ----------
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Quantity vector.
        """

        xp = array_api_compat.array_namespace(fluence)

        if not xp.isdtype(fluence.dtype, self._dtype):
            fluence = xp.asarray(fluence, dtype=self._dtype)

        # check if we need to update the cache
        if self._w_cache is None or not xp.all(self._w_cache == fluence):
            self._w_cache = xp.asarray(fluence, copy=True)
            self._compute_quantity_cache()

        return self._q_cache

    def compute_chain_derivative(self, d_quantity: Array, fluence: Array) -> Array:
        """
        Fluence Derivative of the quantity w.r.t. to the quantity derivative.

        Parameters
        ----------
        d_quantity : ArrayLike or object ndarray
            Derivative w.r.t. the quantity, either one array applied to every scenario or an
            object array in the scenario container layout holding one cotangent per slot.
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Derivative of the quantity w.r.t. the fluence per scenario slot.
        """

        per_slot = isinstance(d_quantity, np.ndarray) and d_quantity.dtype == np.dtype(object)
        if per_slot:
            xp = array_api_compat.array_namespace(fluence)
        else:
            xp = array_api_compat.array_namespace(d_quantity, fluence)

        if not xp.isdtype(fluence.dtype, self._dtype):
            fluence = xp.asarray(fluence, dtype=self._dtype)

        if self._w_grad_cache is None or not xp.all(self._w_grad_cache == fluence):
            self._w_grad_cache = xp.asarray(fluence, copy=True)
            self._compute_chain_derivative_cache(d_quantity)

        return self._qgrad_cache

    def uses_voxel_params(self) -> bool:
        """
        Whether the quantity reads per-voxel parameters (``alphax``, ``betax``, ...).

        Scenarios sharing their influence matrices through an alias basis evaluate to the same
        result unless the quantity (or one of its dependencies) uses per-voxel parameters,
        which may differ between such scenarios.
        """
        return any(dep.uses_voxel_params() for dep in self._deps.values())

    @property
    def _evaluation_plan(self) -> list[tuple[int, str, Optional[dict[int, float]]]]:
        # Built lazily: subclasses settle their computation path after the base __init__.
        if self._evaluation_plan_cache is None:
            self._evaluation_plan_cache = self._plan_evaluation()
        return self._evaluation_plan_cache

    def _plan_evaluation(self) -> list[tuple[int, str, Optional[dict[int, float]]]]:
        """
        Order in which the scenario slots of the cache are filled.

        Each entry is ``(index, kind, coefficients)`` with kind ``"compute"`` (evaluate the
        scenario), ``"alias"`` (reuse the referenced slot) or ``"combine"`` (linear combination
        of computed slots). Computed slots come first, including the basis scenarios a
        requested derived scenario needs.
        """
        scenario_set = getattr(self._dij, "scenarios", None)
        requested = [int(i) for i in self.scenarios]
        if scenario_set is None:
            return [(i, "compute", None) for i in requested]

        uses_voxel_params = self.uses_voxel_params()
        computed: list[int] = []
        derived: list[tuple[int, str, Optional[dict[int, float]]]] = []
        for i in requested:
            basis = scenario_set.bases[i]
            if basis is None:
                computed.append(i)
                continue
            if basis.is_alias:
                ref = next(iter(basis.coefficients))
                same_params = self._dij.voxel_param_column(i) == self._dij.voxel_param_column(ref)
                if uses_voxel_params and not same_params:
                    computed.append(i)
                    continue
            derived.append((i, "alias" if basis.is_alias else "combine", dict(basis.coefficients)))
        for _, _, coefficients in derived:
            computed.extend(k for k in coefficients if k not in computed)
        unique = list(dict.fromkeys(computed))
        return [(i, "compute", None) for i in unique] + derived

    def _compute_quantity_cache(self):
        """Compute the quantity from the fluence and write it into the cache."""

        for scenario_index, kind, coefficients in self._evaluation_plan:
            if kind == "compute":
                self._q_cache.flat[scenario_index] = self._compute_quantity_single_scenario(
                    scenario_index
                )
            else:
                self._q_cache.flat[scenario_index] = self._combine(self._q_cache, coefficients)

    def _compute_chain_derivative_cache(self, d_quantity: Any) -> None:
        """
        Compute the fluence derivative from the quantity derivative into the cache.

        With one cotangent per slot (object array), a derived slot is
        ``sum(c_k * chain_k(cotangent_s))`` with its own cotangent; with a single cotangent the
        combination of the computed slots' derivatives is used instead.
        """
        per_slot = isinstance(d_quantity, np.ndarray) and d_quantity.dtype == np.dtype(object)

        def cotangent(index: int) -> Array:
            return d_quantity.flat[index] if per_slot else d_quantity

        for scenario_index, kind, coefficients in self._evaluation_plan:
            if kind == "compute":
                self._qgrad_cache.flat[scenario_index] = (
                    self._compute_chain_derivative_single_scenario(
                        cotangent(scenario_index), scenario_index
                    )
                )
            elif not per_slot:
                self._qgrad_cache.flat[scenario_index] = self._combine(
                    self._qgrad_cache, coefficients
                )
            else:
                result = None
                for k, c in coefficients.items():
                    term = self._compute_chain_derivative_single_scenario(
                        cotangent(scenario_index), k
                    )
                    term = term if c == 1.0 else c * term
                    result = term if result is None else result + term
                self._qgrad_cache.flat[scenario_index] = result

    @staticmethod
    def _combine(cache: np.ndarray, coefficients: dict[int, float]) -> Array:
        """``sum(c_k * cache[k])`` over the basis of a derived scenario."""
        items = list(coefficients.items())
        if len(items) == 1 and items[0][1] == 1.0:
            return cache.flat[items[0][0]]
        result = None
        for k, c in items:
            term = c * cache.flat[k]
            result = term if result is None else result + term
        return result

    def _backproject(self, matrix, d_quantity: Array) -> Array:
        """Compute ``d_quantity @ matrix`` for a sparse influence matrix."""
        if array_api_compat.is_jax_array(d_quantity):
            # JAX sparse matrices do not implement __rmatmul__
            product = (matrix.T @ d_quantity.T).T
        else:
            # Called on the matrix so scipy accepts array_api_strict operands
            product = matrix.__rmatmul__(d_quantity)
        return self.array_backend.asarray(product, copy=False)

    @abstractmethod
    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        """
        Calculate the quantity in a specific scenario.

        Parameters
        ----------
        scenario_index : int
            Scenario index.

        Returns
        -------
        Array
            Quantity in the scenario.
        """

    @abstractmethod
    def _compute_chain_derivative_single_scenario(
        self, d_quantity: Array, scenario_index: int
    ) -> Array:
        """
        Calculate the derivative of the quantity w.r.t. the fluence in a specific scenario.

        Parameters
        ----------
        d_quantity : Array
            Derivative w.r.t. to the quantity.
        scenario_index : int
            Scenario index.

        Returns
        -------
        Array
            Derivative of the quantity w.r.t. the fluence in the scenario.
        """

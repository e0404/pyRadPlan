from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Union

import logging

import numpy as np
import array_api_compat

from ..core.xp_utils.typing import Array, ArrayNamespace


import pint

from pyRadPlan.dij import Dij
from pyRadPlan.core import xp_utils as compute_backend

ureg = pint.UnitRegistry()

logger = logging.getLogger(__name__)


class RTQuantity(ABC):
    name: ClassVar[str]
    identifier: ClassVar[str]
    unit: ClassVar[pint.Unit]
    dim: ClassVar[int]  # To differentiate between scalar and vector quantities
    precision: str

    def __init__(self, dij: Dij, scenarios=None):
        if scenarios is None:
            scenarios = [0]
        self.scenarios = np.asarray(scenarios, dtype=np.int64)


class FluenceDependentQuantity(RTQuantity, ABC):
    """
    Base class for quantities that depend on fluence distributions.

    Concrete subclasses declare how they relate to a :class:`Dij` via two ClassVars:

    - :attr:`required_dependencies`: identifiers of upstream quantities that are always
      needed to compute this quantity. The resolver must produce them; failure raises.
    - :attr:`optional_dependencies`: identifiers of upstream quantities used as a fallback
      computation path when no direct Dij matrix is available.

    The resolution mode (``"direct"`` vs ``"indirect"``) is decided at instantiation time:
    if the dij carries an attribute whose name matches this quantity's ``identifier``, the
    direct path wins; otherwise the indirect path is used and ``required + optional`` deps
    must resolve.  This enforces a consistent naming convention: a Dij matrix attribute must
    share its name with the quantity identifier it represents.
    """

    # ClassVar metadata describing how this quantity is computed.
    required_dependencies: ClassVar[tuple[str, ...]] = ()
    optional_dependencies: ClassVar[tuple[str, ...]] = ()

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
        self.array_backend: ArrayNamespace = xp

        # The resolver pre-converts the dij to the target namespace and supplies
        # `mode` + `dependencies`. When invoked directly (e.g. from tests) we do the
        # conversion ourselves and build deps lazily via the resolver.
        if mode is None:
            self._dij = dij.to_namespace(xp)
            self._mode = self._choose_mode()
            self._deps: dict[str, FluenceDependentQuantity] = dict(dependencies or {})
            if self._mode == "indirect" and not self._deps:
                # Lazy import to avoid circular dependency at module import time.
                from ._resolver import QuantityResolver  # noqa: PLC0415

                resolver = QuantityResolver(self._dij, _dij_already_in_namespace=True)
                for dep_id in self._dep_ids_for_indirect():
                    self._deps[dep_id] = resolver.get(dep_id)
        else:
            if mode not in ("direct", "indirect"):
                raise ValueError(f"Unknown quantity mode: {mode!r}")
            self._dij = dij
            self._mode = mode
            self._deps = dict(dependencies or {})

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
        if self.required_dependencies or self.optional_dependencies:
            return "indirect"
        raise ValueError(
            f"Quantity {type(self).__name__!r} cannot be computed: dij has no attribute "
            f"{self.identifier!r} and no dependencies are declared."
        )

    def _dep_ids_for_indirect(self) -> tuple[str, ...]:
        """Return identifiers that must be resolved to compute this quantity indirectly."""
        return tuple(self.required_dependencies) + tuple(self.optional_dependencies)

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
            if self._w_cache is None:
                self._w_cache = xp.asarray(fluence, copy=True)
            else:
                self._w_cache[:] = fluence
            self._compute_quantity_cache()

        return self._q_cache

    def compute_chain_derivative(self, d_quantity: Array, fluence: Array) -> Array:
        """
        Fluence Derivative of the quantity w.r.t. to the quantity derivative.

        Parameters
        ----------
        d_quantity : ArrayLike
            Derivative of w.r.t. to the quantity.
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Derivative of the quantity w.r.t. the fluence.
        """

        xp = array_api_compat.array_namespace(d_quantity, fluence)

        if not xp.isdtype(fluence.dtype, self._dtype):
            fluence = xp.asarray(fluence, dtype=self._dtype)

        if self._w_grad_cache is None or not xp.all(self._w_grad_cache == fluence):
            if self._w_grad_cache is None:
                self._w_grad_cache = xp.asarray(fluence, copy=True)
            else:
                self._w_grad_cache[:] = fluence
            self._compute_chain_derivative_cache(d_quantity)

        return self._qgrad_cache

    def _compute_quantity_cache(self):
        """Compute the quantity from the fluence and write it into the cache."""

        for scenario_index in self.scenarios:
            self._q_cache.flat[scenario_index] = self._compute_quantity_single_scenario(
                scenario_index
            )

    def _compute_chain_derivative_cache(self, d_quantity: Array) -> Array:
        """Compute the fluence derivative from the quantity derivative into the cache."""

        for scenario_index in self.scenarios:
            self._qgrad_cache.flat[scenario_index] = (
                self._compute_chain_derivative_single_scenario(d_quantity, scenario_index)
            )

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

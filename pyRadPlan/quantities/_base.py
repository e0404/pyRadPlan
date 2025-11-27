from abc import ABC, abstractmethod
from typing import ClassVar, Union

import logging

import numpy as np
import array_api_compat

try:
    import cupy as cp
    import cupyx.scipy.sparse as cusparse
except ImportError:
    cp = None
    cusparse = None

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

    def __init__(self, scenarios=None):
        if scenarios is None:
            scenarios = [0]
        self.scenarios = np.asarray(scenarios, dtype=np.int64)


class FluenceDependentQuantity(RTQuantity, ABC):
    """Base class for quantities that depend on fluence distributions."""

    array_backend: ArrayNamespace

    def __init__(self, dij: Dij, **kwargs):
        super().__init__(**kwargs)

        # TODO: This backend check and conversion should be a part of the dij
        xp = compute_backend.choose_array_api_namespace()

        influence_matrix = getattr(dij, self.identifier, None)
        if influence_matrix is None:
            raise ValueError(f"Influence matrix {self.identifier} not available in Dij object.")

        self._dij = dij.to_namespace(xp)

        try:
            typename = getattr(self._dij, self.identifier).flat[0].dtype.name
            self._dtype = getattr(xp, typename)
        except AttributeError:
            self._dtype = getattr(self._dij, self.identifier).flat[0].dtype

        logger.info("Optimization uses array backend: %s", xp.__name__)

        self.array_backend: ArrayNamespace = xp

        # Fluence cache for derivative calculation
        self._w_cache: Union[Array, None] = None
        self._w_grad_cache: Union[Array, None] = None
        # Quantity vector cache
        self._q_cache = np.empty_like(getattr(self._dij, self.identifier), dtype=object)
        self._qgrad_cache = np.empty_like(self._q_cache)

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
        """
        Protected function to compute the quantity from the fluence and write it into the cache.

        Parameters
        ----------
        fluence : Array
            Fluence distribution.
        """

        for scenario_index in self.scenarios:
            self._q_cache.flat[scenario_index] = self._compute_quantity_single_scenario_from_cache(
                scenario_index
            )

    def _compute_chain_derivative_cache(self, d_quantity: Array) -> Array:
        """
        Protected interface for calculating the fluence derivative from quantity derivative.

        Parameters
        ----------
        d_quantity : Array
            Derivative w.r.t. to the quantity.

        Returns
        -------
        Array
            Derivative of the quantity w.r.t. the fluence.
        """

        for scenario_index in self.scenarios:
            self._qgrad_cache.flat[scenario_index] = (
                self._compute_chained_fluence_gradient_single_scenario_from_cache(
                    d_quantity, scenario_index
                )
            )

    @abstractmethod
    def _compute_quantity_single_scenario_from_cache(self, scenario_index: int) -> Array:
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
    def _compute_chained_fluence_gradient_single_scenario_from_cache(
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

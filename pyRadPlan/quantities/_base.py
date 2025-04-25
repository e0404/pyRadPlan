from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np
from numpy.typing import NDArray, ArrayLike
import pint

from pyRadPlan.dij import Dij, validate_dij

ureg = pint.UnitRegistry()


class RTQuantity(ABC):
    name: ClassVar[str]
    identifier: ClassVar[str]
    unit: ClassVar[pint.Unit]
    dim: ClassVar[int]  # To differentiate between scalar and vector quantities

    scenarios: NDArray[np.int64]  # Scenarios the quantity is calculated / defined for

    def __init__(self, scenarios=None):
        if scenarios is None:
            scenarios = [0]
        self.scenarios = np.asarray(scenarios, dtype=np.int64)


class FluenceDependentQuantity(RTQuantity, ABC):
    """Base class for quantities that depend on fluence distributions."""

    def __init__(self, dij: Dij, **kwargs):
        super().__init__(**kwargs)
        self._dij = validate_dij(dij)

        # Initialize Caches
        # Fluence cache for forward calculation
        self._w_cache = np.nan * np.ones((self._dij.total_num_of_bixels), dtype=np.float64)
        # Fluence cache for derivative calculation
        self._w_grad_cache = self._w_cache.copy()
        # Quantity vector cache
        self._q_cache = np.empty_like(getattr(self._dij, self.identifier), dtype=object)
        self._qgrad_cache = np.empty_like(self._q_cache)

    def __call__(self, fluence: ArrayLike) -> NDArray:
        """
        Make the quantity callable by calling the compute method.

        Parameters
        ----------
        fluence : ArrayLike
            Fluence vector.

        Returns
        -------
        NDArray
            Quantity vector.
        """

        return self.compute(fluence)

    def compute(self, fluence: ArrayLike) -> NDArray:
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

        _fluence = np.asarray(fluence)

        if not np.array_equal(self._w_cache, _fluence):
            self._w_cache = _fluence.copy()
            self._compute_quantity_cache()

        return self._q_cache

    def compute_chain_derivative(self, d_quantity: ArrayLike, fluence: ArrayLike) -> NDArray:
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

        _d_quantity = np.asarray(d_quantity)
        _fluence = np.asarray(fluence)

        if not np.array_equal(self._w_grad_cache, _d_quantity):
            self._w_grad_cache = _fluence.copy()
            self._compute_chain_derivative_cache(_d_quantity)

        return self._qgrad_cache

    def _compute_quantity_cache(self):
        """
        Protected function to compute the quantity from the fluence and write it into the cache.

        Parameters
        ----------
        fluence : NDArray
            Fluence distribution.
        """

        for scenario_index in self.scenarios:
            self._q_cache.flat[scenario_index] = self._compute_quantity_single_scenario_from_cache(
                scenario_index
            )

    def _compute_chain_derivative_cache(self, d_quantity: NDArray) -> NDArray:
        """
        Protected interface for calculating the fluence derivative from quantity derivative.

        Parameters
        ----------
        d_quantity : NDArray
            Derivative w.r.t. to the quantity.

        Returns
        -------
        NDArray
            Derivative of the quantity w.r.t. the fluence.
        """

        for scenario_index in self.scenarios:
            self._qgrad_cache.flat[scenario_index] = (
                self._compute_chained_fluence_gradient_single_scenario_from_cache(
                    d_quantity, scenario_index
                )
            )

    @abstractmethod
    def _compute_quantity_single_scenario_from_cache(self, scenario_index: int) -> NDArray:
        """
        Calculate the quantity in a specific scenario.

        Parameters
        ----------
        scenario_index : int
            Scenario index.

        Returns
        -------
        NDArray
            Quantity in the scenario.
        """

    @abstractmethod
    def _compute_chained_fluence_gradient_single_scenario_from_cache(
        self, d_quantity: NDArray, scenario_index: int
    ) -> NDArray:
        """
        Calculate the derivative of the quantity w.r.t. the fluence in a specific scenario.

        Parameters
        ----------
        d_quantity : NDArray
            Derivative w.r.t. to the quantity.
        scenario_index : int
            Scenario index.

        Returns
        -------
        NDArray
            Derivative of the quantity w.r.t. the fluence in the scenario.
        """

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

from pyRadPlan.core._units import ureg


class Dose(FluenceDependentQuantity):
    """Physical dose quantity backed directly by the dij.physical_dose matrix."""

    unit = ureg.gray
    dim = 1
    identifier = "physical_dose"
    name = "dose"
    fraction_exponent = 1.0

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        return self.array_backend.asarray(
            self._dij.physical_dose.flat[scenario_index] @ self._w_cache, copy=False
        )

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        return self._backproject(self._dij.physical_dose.flat[scenario_index], d_quantity)

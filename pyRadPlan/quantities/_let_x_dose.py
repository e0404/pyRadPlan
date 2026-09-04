import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class LETxDose(FluenceDependentQuantity):
    """LET-weighted dose quantity backed directly by the dij.let_dose matrix."""

    unit = ureg.gray * ureg.keV / ureg.micrometer
    dim = 1
    identifier = "let_dose"
    name = "LETxDose"

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        return self.array_backend.asarray(
            self._dij.let_dose.flat[scenario_index] @ self._w_cache, copy=False
        )

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # Transpose form is array-api compliant (scipy / array_api_strict compatibility).
        return self.array_backend.asarray(
            (self._dij.let_dose.flat[scenario_index].T @ d_quantity.T).T, copy=False
        )

import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class AlphaDose(FluenceDependentQuantity):
    """Alpha-dose quantity; uses dij.alpha_dose directly, falls back to alpha_x * Dose."""

    unit = []
    dim = 1
    identifier = "alpha_dose"
    name = "AlphaDose"
    optional_dependencies = ("physical_dose",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        if self._mode == "direct":
            return self.array_backend.asarray(
                self._dij.alpha_dose.flat[scenario_index] @ self._w_cache, copy=False
            )
        # Indirect path: alpha_x * physical_dose
        dose = self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index]
        return self.array_backend.asarray(self._dij.alphax * dose, copy=False)

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        if self._mode == "direct":
            # Transpose form is array-api compliant (scipy / array_api_strict compatibility).
            return self.array_backend.asarray(
                self._dij.alpha_dose.flat[scenario_index].__rmatmul__(d_quantity), copy=False
            )
        # Chain rule for alpha_x * Dose: d/dw = alpha_x * dDose/dw, scaled by d_quantity.
        scaled = self._dij.alphax * d_quantity
        return self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            scaled, scenario_index
        )

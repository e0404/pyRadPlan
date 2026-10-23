from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity


class SqrtBetaDose(FluenceDependentQuantity):
    """Square-root beta-dose quantity; uses dij.sqrt_beta_dose, falls back to sqrt(beta_x) * Dose."""

    unit = []
    dim = 1
    identifier = "sqrt_beta_dose"
    biological = True
    name = "SqrtBetaDose"
    optional_dependencies = ("physical_dose",)
    indirect_dij_fields = ("betax",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        if self._mode == "direct":
            return self.array_backend.asarray(
                self._dij.sqrt_beta_dose.flat[scenario_index] @ self._w_cache, copy=False
            )
        # Indirect path: sqrt_beta_x * physical_dose
        dose = self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index]
        return self.array_backend.asarray(
            self.array_backend.sqrt(self._dij.betax[:, scenario_index]) * dose, copy=False
        )

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # Transpose form is array-api compliant (scipy / array_api_strict compatibility).
        if self._mode == "direct":
            return self.array_backend.asarray(
                self._dij.sqrt_beta_dose.flat[scenario_index].__rmatmul__(d_quantity), copy=False
            )
        # Chain rule for sqrt_beta_x * Dose: d/dw = sqrt_beta_x * dDose/dw, scaled by d_quantity.
        scaled = self.array_backend.sqrt(self._dij.betax[:, scenario_index]) * d_quantity
        return self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            scaled, scenario_index
        )

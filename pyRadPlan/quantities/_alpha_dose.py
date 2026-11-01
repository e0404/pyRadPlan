from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity


class AlphaDose(FluenceDependentQuantity):
    """Alpha-dose quantity; uses dij.alpha_dose directly, falls back to alpha_x * Dose."""

    unit = []
    dim = 1
    identifier = "alpha_dose"
    biological = True
    name = "AlphaDose"
    fraction_exponent = 1.0
    optional_dependencies = ("physical_dose",)
    indirect_dij_fields = ("alphax",)

    def uses_voxel_params(self) -> bool:
        return self._mode == "indirect" or super().uses_voxel_params()

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        if self._mode == "direct":
            return self.array_backend.asarray(
                self._dij.alpha_dose.flat[scenario_index] @ self._w_cache, copy=False
            )
        # Indirect path: alpha_x * physical_dose
        dose = self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index]
        return self.array_backend.asarray(
            self._dij.alphax[:, self._dij.voxel_param_column(scenario_index)] * dose, copy=False
        )

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        if self._mode == "direct":
            return self._backproject(self._dij.alpha_dose.flat[scenario_index], d_quantity)
        # Chain rule for alpha_x * Dose: d/dw = alpha_x * dDose/dw, scaled by d_quantity.
        column = self._dij.voxel_param_column(scenario_index)
        scaled = self._dij.alphax[:, column] * d_quantity
        return self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            scaled, scenario_index
        )

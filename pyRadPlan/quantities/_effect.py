import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class Effect(FluenceDependentQuantity):
    """Linear-quadratic effect, computed from alpha_dose + sqrt_beta_dose squared."""

    unit = []
    dim = 1
    identifier = "effect"
    name = "Effect"
    required_dependencies = ("alpha_dose", "sqrt_beta_dose")

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        alpha = self._deps["alpha_dose"].compute(self._w_cache).flat[scenario_index]
        sb = self._deps["sqrt_beta_dose"].compute(self._w_cache).flat[scenario_index]
        return alpha + sb**2

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        alpha_grad = self._deps["alpha_dose"]._compute_chain_derivative_single_scenario(
            d_quantity, scenario_index
        )
        sqrt_beta_dose = self._deps["sqrt_beta_dose"].compute(self._w_cache)
        fgrad_beta = 2 * d_quantity * sqrt_beta_dose.flat[scenario_index]
        beta_grad = self._deps["sqrt_beta_dose"]._compute_chain_derivative_single_scenario(
            fgrad_beta, scenario_index
        )
        return alpha_grad + beta_grad

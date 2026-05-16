import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class DoseWeightedLET(FluenceDependentQuantity):
    """Dose-weighted LET; routed through the shared LETxDose quantity instance."""

    unit = ureg.micrometer / ureg.keV
    dim = 1
    identifier = "let"
    name = "LETd"
    required_dependencies = ("let_dose",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        return self._deps["let_dose"].compute(self._w_cache).flat[scenario_index]

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        return self._deps["let_dose"]._compute_chain_derivative_single_scenario(
            d_quantity, scenario_index
        )

import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class LETxDose(FluenceDependentQuantity):
    """LETxDose quantity depending on fluence."""

    unit = ureg.gray * ureg.micrometer / ureg.keV
    dim = 1
    identifier = "let_dose"
    name = "LETxDose"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.identifier not in self._dij.model_fields or self._dij.let_dose is None:
            raise ValueError(f"Quantity {self.identifier} not available in Dij object.")

    def _compute_quantity_single_scenario_from_cache(self, scenario_index: int) -> Array:
        return self.array_backend.asarray(
            self._dij.let_dose.flat[scenario_index] @ self._w_cache, copy=False
        )

    def _compute_chained_fluence_gradient_single_scenario_from_cache(
        self, d_quantity, scenario_index: int
    ) -> Array:
        # Implemented as transpose to be array api compliant in case of scipy / array_api_strict
        # compatibility
        return self.array_backend.asarray(
            (self._dij.let_dose.flat[scenario_index].T @ d_quantity.T).T, copy=False
        )

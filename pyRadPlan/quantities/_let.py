import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class DoseWeightedLET(FluenceDependentQuantity):
    """
    Dose-weighted LET (LETd).

    Computes physical dose-weighted LET as the quotient of LETxDose (let_dose)
    and physical dose (physical_dose).
    """

    unit = ureg.keV / ureg.micrometer
    dim = 1
    identifier = "let"
    name = "LETd"
    required_dependencies = ("let_dose", "physical_dose")

    def _quotient_terms(self, fluence: Array, scenario_index: int) -> tuple[Array, Array, Array]:
        """
        LETxDose, a divisor that is safe to divide by, and the positive-dose mask.

        The divisor is the physical dose where that is positive and one elsewhere, so the
        quotient never evaluates a division by zero. Callers mask the result back to zero
        where the dose vanishes, which keeps those entries exactly zero instead of
        the very large values an epsilon-regularised denominator would produce.
        """

        xp = self.array_backend
        let_dose = self._deps["let_dose"].compute(fluence).flat[scenario_index]
        dose = self._deps["physical_dose"].compute(fluence).flat[scenario_index]
        d_positive = dose > 0.0
        return let_dose, xp.where(d_positive, dose, xp.ones_like(dose)), d_positive

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        xp = self.array_backend
        let_dose, safe_dose, d_positive = self._quotient_terms(self._w_cache, scenario_index)
        return xp.where(d_positive, let_dose / safe_dose, xp.zeros_like(safe_dose))

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        xp = self.array_backend
        let_dose, safe_dose, d_positive = self._quotient_terms(self._w_grad_cache, scenario_index)
        zero = xp.zeros_like(safe_dose)

        grad_let_dose = xp.where(d_positive, d_quantity / safe_dose, zero)
        grad_phys_dose = xp.where(d_positive, -d_quantity * let_dose / safe_dose**2, zero)

        return self._deps["let_dose"]._compute_chain_derivative_single_scenario(
            grad_let_dose, scenario_index
        ) + self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            grad_phys_dose, scenario_index
        )

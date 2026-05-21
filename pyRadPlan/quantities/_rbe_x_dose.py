import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class RBExDose(FluenceDependentQuantity):
    """RBE-weighted dose computed from the linear-quadratic effect."""

    unit = ureg.gray
    dim = 1
    identifier = "rbe_x_dose"
    name = "RBExDose"
    required_dependencies = ("effect",)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        # TODO: correct handling of ct scenario
        effect = self._deps["effect"].compute(self._w_cache)
        ix = self._dij.betax > 0
        gamma = self.array_backend.zeros_like(self._dij.betax)
        gamma[ix] = self._dij.alphax[ix] / self._dij.betax[ix] / 2
        rbe_x_dose = self.array_backend.zeros_like(effect.flat[scenario_index])
        rbe_x_dose[ix] = (
            self.array_backend.sqrt(
                gamma[ix] ** 2 + effect.flat[scenario_index][ix] / self._dij.betax[ix]
            )
            - gamma[ix]
        )
        return rbe_x_dose

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # TODO: correct handling of ct scenarios
        xp = self.array_backend
        d_quantity = xp.reshape(d_quantity, (-1,))
        ix = self._dij.betax > 0
        gamma = xp.zeros_like(self._dij.betax)
        gamma[ix] = self._dij.alphax[ix] / self._dij.betax[ix] / 2
        # Sync effect's fluence cache to ours, then route the gradient through it.
        effect = self._deps["effect"].compute(self._w_cache)
        effect.flat[scenario_index][ix] = effect.flat[scenario_index][ix] + gamma[ix]
        fgrad = xp.zeros_like(d_quantity)
        fgrad[ix] = d_quantity[ix] / (2 * self._dij.betax[ix] * effect.flat[scenario_index][ix])
        fgrad = xp.reshape(fgrad, (1, -1))
        return self._deps["effect"]._compute_chain_derivative_single_scenario(
            fgrad, scenario_index
        )

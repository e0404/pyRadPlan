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
        xp = self.array_backend
        dtype_xp = self._dtype

        effect = self._deps["effect"].compute(self._w_cache)

        # Wrap all numpy slices upfront
        alphax = xp.asarray(self._dij.alphax[:, scenario_index])
        betax = xp.asarray(self._dij.betax[:, scenario_index])
        ix = xp.asarray(self._dij.betax[:, scenario_index] > 0)
        effect_slice = xp.asarray(effect.flat[scenario_index])

        gamma = xp.zeros(betax.shape, dtype=dtype_xp)
        gamma[ix] = alphax[ix] / betax[ix] / 2

        rbe_x_dose = xp.zeros(effect_slice.shape, dtype=dtype_xp)
        rbe_x_dose[ix] = xp.sqrt(gamma[ix] ** 2 + effect_slice[ix] / betax[ix]) - gamma[ix]
        return rbe_x_dose

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # TODO: correct handling of ct scenarios
        xp = self.array_backend
        dtype_xp = self._dtype

        d_quantity = xp.reshape(d_quantity, (-1,))

        # Wrap all numpy slices upfront
        alphax = xp.asarray(self._dij.alphax[:, scenario_index])
        betax = xp.asarray(self._dij.betax[:, scenario_index])
        ix = xp.asarray(self._dij.betax[:, scenario_index] > 0)

        gamma = xp.zeros(betax.shape, dtype=dtype_xp)
        gamma[ix] = alphax[ix] / betax[ix] / 2

        # Sync effect's fluence cache to ours, then route the gradient through it.
        effect = self._deps["effect"].compute(self._w_cache)
        effect_slice = xp.asarray(effect.flat[scenario_index])
        effect_slice[ix] = effect_slice[ix] + gamma[ix]

        fgrad = xp.zeros(d_quantity.shape, dtype=dtype_xp)
        fgrad[ix] = d_quantity[ix] / (2 * betax[ix] * effect_slice[ix])

        fgrad = xp.reshape(fgrad, (1, -1))
        return self._deps["effect"]._compute_chain_derivative_single_scenario(
            fgrad, scenario_index
        )

import pint

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._polo_base import POLOBase

ureg = pint.UnitRegistry()


class LinearPOLO(POLOBase):
    """
    Linear Probability of Lesion Origin (POLO) model quantity (eta).

    Exposes the linear predictor eta before the sigmoid transformation, as used by the
    convexified objectives in Ortkamp et al. ([2], Sec. 2.2.2). The volume correction of
    :class:`~pyRadPlan.quantities.POLO` is deliberately not applied here: it acts on the
    probability, not on eta.

    Voxels below :attr:`dose_threshold` fall outside the dose domain the model was fitted on
    and are returned as ``-inf``, which maps to a probability of zero. Objectives on eta are
    expected to filter these out (see :class:`HellingerConvPOLO`).

    For coefficients and references, see :class:`~pyRadPlan.quantities.POLOBase`.
    """

    unit = ureg.dimensionless
    dim = 1
    identifier = "linear_polo"
    name = "Linear POLO"

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        eta, d = self._linear_predictor(self._w_cache, scenario_index)

        # -inf is the only correct sentinel in eta-space: it maps to p = 0, whereas a finite
        # 0 would mean p = 0.5.
        return self.array_backend.where(
            self._in_dose_domain(d), eta, self.array_backend.full_like(eta, float("-inf"))
        )

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # d(eta)/d(dose) and d(eta)/d(let_dose) are the model coefficients, so the incoming
        # gradient only needs masking to the dose domain before back-propagation.
        d_quantity = self.array_backend.where(
            self._in_dose_domain(self._dose(self._w_grad_cache, scenario_index)),
            d_quantity,
            self.array_backend.zeros_like(d_quantity),
        )
        return self._backpropagate(d_quantity, scenario_index)

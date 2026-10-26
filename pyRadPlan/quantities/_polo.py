from typing import Optional

import numpy as np
import pint
import SimpleITK as sitk

from ..core.resample import resample_image
from ..core.xp_utils import to_numpy
from ..core.xp_utils.typing import Array
from ._polo_base import POLOBase, total_ntcp

ureg = pint.UnitRegistry()


class POLO(POLOBase):
    """
    Probability of Lesion Origin (POLO) model quantity.

    Predicts the probability of lesion origin from physical dose, dose-weighted LET and a
    binary risk factor (periventricular fringe <= 4 mm), volume-corrected by
    ``k = v_voxel / 1.08 mm^3`` relative to the grid the model was fitted on
    (0.6 x 0.6 x 3.0 mm^3).

    Voxels below :attr:`dose_threshold` fall outside the dose domain the model was fitted on
    and are returned as zero.

    For coefficients and references, see :class:`~pyRadPlan.quantities.POLOBase`.
    """

    unit = ureg.dimensionless
    dim = 1
    identifier = "polo"
    name = "POLO"

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        xp = self.array_backend
        eta, d = self._linear_predictor(self._w_cache, scenario_index)

        # Sigmoid probability, then volume correction q = 1 - (1 - p)^k
        p = 1.0 / (1.0 + xp.exp(-eta))
        q = -xp.expm1(self.k * xp.log1p(-p))

        return xp.where(self._in_dose_domain(d), q, xp.zeros_like(q))

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        xp = self.array_backend
        eta, d = self._linear_predictor(self._w_grad_cache, scenario_index)
        p = 1.0 / (1.0 + xp.exp(-eta))

        # Numerically stable gradient of the volume-scaled probability w.r.t. eta:
        # dq/deta = k * p * (1 - p)^k
        dq_deta = self.k * p * xp.exp(self.k * xp.log1p(-p))

        # Sub-threshold voxels are clamped to a constant, hence carry no gradient.
        dq_deta = xp.where(self._in_dose_domain(d), dq_deta, xp.zeros_like(dq_deta))

        return self._backpropagate(d_quantity * dq_deta, scenario_index)

    def compute_total_ntcp(
        self, fluence: Array, mask: Optional[Array] = None, scenario_index: int = 0
    ) -> float:
        """
        Compute the total Normal Tissue Complication Probability (NTCP).

        Parameters
        ----------
        fluence : Array
            The optimized fluence vector.
        mask : Array or sitk.Image, optional
            Binary mask restricting the evaluation. A SimpleITK image is resampled to the
            dose grid; an array is used as a flat boolean index. Evaluates every voxel when
            omitted.
        scenario_index : int, optional
            Scenario index to compute the NTCP for, by default 0.

        Returns
        -------
        float
            The total NTCP.
        """
        # Bring the result to the host before masking and aggregating. The quantity may hold
        # its dij on an accelerator, and a device array neither converts implicitly to NumPy
        # nor indexes with a NumPy boolean mask.
        polo_flat = to_numpy(self.compute(fluence).flat[scenario_index])

        if mask is not None:
            if hasattr(mask, "GetSize"):
                resampled = resample_image(
                    mask,
                    target_grid=self._dij.dose_grid,
                    interpolator=sitk.sitkNearestNeighbor,
                    extrapolate=0.0,
                )
                mask_idx = sitk.GetArrayFromImage(resampled).astype(bool).ravel()
            else:
                mask_idx = np.asarray(mask, dtype=bool).ravel()

            polo_flat = polo_flat[mask_idx]

        return float(total_ntcp(polo_flat))

import pint

from typing import Optional
from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class LinearPOLO(FluenceDependentQuantity):
    """
    Linear Probability of Lesion Origin (POLO) model quantity (eta).

    Predicts the linear predictor (eta) before the sigmoid transformation.

    References
    ----------
    .. [1] Bahn, E., Bauer, J., Harrabi, S. B., et al. (2020). Late contrast enhancing brain lesions
           in proton-treated patients with low-grade glioma. Int J Radiat Oncol Biol Phys, 107(3),
           571–578. https://doi.org/10.1016/j.ijrobp.2020.03.013
    .. [2] Ortkamp, T., Sallem, H., Harrabi, S., Frank, M., Jäkel, O., Bauer, J., & Wahl, N. (2026).
           Direct optimization of the probability of lesion origin in proton treatment planning for
           low-grade glioma patients. Medical Physics, 53, e70395. https://doi.org/10.1002/mp.70395
    """

    required_dependencies = ("physical_dose", "let_dose")
    unit = ureg.dimensionless
    dim = 1
    identifier = "linear_polo"
    name = "Linear POLO"

    def __init__(
        self,
        dij,
        binary_risk_mask: Optional[Array] = None,
        cst=None,
        risk_voi_name: str = "Ventricles_4mm",
        mode=None,
        dependencies=None,
        scenarios=None,
    ):
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        self.risk_voi_name = risk_voi_name

        if binary_risk_mask is None and cst is not None:
            # Look up requested risk VOI in CST
            for voi in getattr(cst, "vois", []):
                if voi.name == risk_voi_name:
                    binary_risk_mask = voi.mask
                    break

        if binary_risk_mask is None:
            binary_risk_mask = getattr(dij, "binary_risk_mask", None)

        if hasattr(binary_risk_mask, "mask") or hasattr(binary_risk_mask, "GetSize"):
            import SimpleITK as sitk
            from pyRadPlan.core.resample import resample_image

            img = binary_risk_mask.mask if hasattr(binary_risk_mask, "mask") else binary_risk_mask
            resampled = resample_image(
                img,
                target_grid=dij.dose_grid,
                extrapolate=0.0,
                interpolator=sitk.sitkNearestNeighbor,
            )
            binary_risk_mask = sitk.GetArrayViewFromImage(resampled).ravel() > 0.5

        if binary_risk_mask is None:
            num_vox = getattr(dij, "num_of_voxels", getattr(dij, "total_voxels", 0))
            binary_risk_mask = self.array_backend.zeros((num_vox,), dtype=self._dtype)

        self.binary_risk_mask = self.array_backend.asarray(binary_risk_mask, dtype=self._dtype)

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        d = self.array_backend.asarray(
            self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index],
            dtype=self._dtype,
        )
        d_x_ld = self.array_backend.asarray(
            self._deps["let_dose"].compute(self._w_cache).flat[scenario_index], dtype=self._dtype
        )
        eta = -26.3 + 0.19 * d + 0.018 * d_x_ld + 1.19 * self.binary_risk_mask
        return eta

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        # Gradient of eta w.r.t physical_dose is 0.19
        grad_d = d_quantity * 0.19

        # Gradient of eta w.r.t let_dose is 0.018
        grad_d_x_ld = d_quantity * 0.018

        chain_d = (
            self._deps["physical_dose"]
            .compute_chain_derivative(grad_d, self._w_cache)
            .flat[scenario_index]
        )
        chain_d_x_ld = (
            self._deps["let_dose"]
            .compute_chain_derivative(grad_d_x_ld, self._w_cache)
            .flat[scenario_index]
        )

        return chain_d + chain_d_x_ld

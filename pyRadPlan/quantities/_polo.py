import pint

from typing import Optional
from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class POLO(FluenceDependentQuantity):
    """
    Probability of Lesion Origin (POLO) model quantity.

    Predicts the probability of lesion origin based on Physical Dose, LETxDose,
    and a binary risk factor mask.
    """

    required_dependencies = ("physical_dose", "let_dose")
    unit = ureg.dimensionless
    dim = 1
    identifier = "polo"
    name = "POLO"

    def __init__(
        self,
        dij,
        binary_risk_mask: Optional[Array] = None,
        mode=None,
        dependencies=None,
        scenarios=None,
    ):
        """
        Initialize the POLO quantity.

        Args:
            dij: Dose influence matrix object.
            binary_risk_mask: Array of shape (n_voxels,) containing 1 for risk
                              voxels (e.g., near ventricles) and 0 otherwise.
            mode: Compute mode.
            dependencies: Dictionary of dependency quantity objects.
            scenarios: Scenario indices.
        """
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        if binary_risk_mask is None:
            binary_risk_mask = getattr(dij, "binary_risk_mask", None)
        if hasattr(binary_risk_mask, "mask") or hasattr(binary_risk_mask, "GetSize"):
            import SimpleITK as sitk
            from pyRadPlan.core.resample import resample_image

            img = binary_risk_mask.mask if hasattr(binary_risk_mask, "mask") else binary_risk_mask
            resampled = resample_image(
                img, dij.dose_grid, default_pixel_value=0.0, interpolator=sitk.sitkNearestNeighbor
            )
            binary_risk_mask = sitk.GetArrayViewFromImage(resampled).ravel() > 0.5
        if binary_risk_mask is None:
            binary_risk_mask = self.array_backend.zeros((dij.total_voxels,), dtype=self._dtype)
        self.binary_risk_mask = self.array_backend.asarray(binary_risk_mask, dtype=self._dtype)

        # Calculate volume correction factor k = v_new / v_old
        # v_old is 0.6 * 0.6 * 3.0 = 1.08 mm^3 (from the original paper fit)
        v_old = 1.08
        res = self._dij.dose_grid.resolution
        v_new = res["x"] * res["y"] * res["z"]
        self.k = v_new / v_old

    def _compute_quantity_single_scenario(self, scenario_index: int) -> Array:
        d = self.array_backend.asarray(
            self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index],
            dtype=self._dtype,
        )
        d_x_ld = self.array_backend.asarray(
            self._deps["let_dose"].compute(self._w_cache).flat[scenario_index], dtype=self._dtype
        )
        eta = -26.3 + 0.19 * d + 0.018 * d_x_ld + 1.19 * self.binary_risk_mask

        # Sigmoid probability
        p = 1.0 / (1.0 + self.array_backend.exp(-eta))

        # Volume correction
        return 1.0 - (1.0 - p) ** self.k

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        d = self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index]
        d_x_ld = self._deps["let_dose"].compute(self._w_cache).flat[scenario_index]

        eta = -26.3 + 0.19 * d + 0.018 * d_x_ld + 1.19 * self.binary_risk_mask
        p = 1.0 / (1.0 + self.array_backend.exp(-eta))

        # Numerically stable gradient of volume-scaled probability with respect to eta:
        # d/deta (1 - (1-p)^k) = k * (1-p)^(k-1) * p * (1-p) = k * p * (1-p)^k
        dp_k_deta = self.k * p * ((1.0 - p) ** self.k)

        # Chain rule back to linear components
        grad_d = d_quantity * dp_k_deta * 0.19
        grad_d_x_ld = d_quantity * dp_k_deta * 0.018

        # Delegate to dependencies to push gradient all the way back to fluence (w)
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

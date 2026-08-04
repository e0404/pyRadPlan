import pint

from typing import Optional
from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

ureg = pint.UnitRegistry()


class POLO(FluenceDependentQuantity):
    """
    Probability of Lesion Origin (POLO) model quantity.

    Predicts the probability of lesion origin based on Physical Dose, LETxDose,
    and a binary risk factor mask (periventricular fringe <= 4mm), automatically
    scaled by grid voxel volume ratio k = v_new / 1.08 mm^3 relative to the reference
    fit grid (0.6 x 0.6 x 3.0 mm^3).

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
    identifier = "polo"
    name = "POLO"

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
        """
        Initialize the POLO quantity.

        Args:
            dij: Dose influence matrix object.
            binary_risk_mask: Optional array of shape (n_voxels,) containing 1 for risk voxels.
            cst: Optional Clinical Structure Set object to extract the risk mask from.
            risk_voi_name: VOI name in CST for the periventricular risk structure (default: 'Ventricles_4mm').
            mode: Compute mode.
            dependencies: Dictionary of dependency quantity objects.
            scenarios: Scenario indices.
        """
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        self.risk_voi_name = risk_voi_name

        if binary_risk_mask is None and cst is not None:
            # Look up requested risk VOI in CST
            vois = cst.vois if hasattr(cst, "vois") else (cst if isinstance(cst, list) else [])
            for voi in vois:
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
            if risk_voi_name is not None:
                raise ValueError(
                    f"POLO risk VOI '{risk_voi_name}' not found in CST and no binary_risk_mask provided. "
                    "The POLO model was originally fitted using a 'Ventricles_4mm' risk mask. "
                    "If you intentionally want to evaluate POLO without the periventricular risk component, "
                    "explicitly set `risk_voi_name=None`."
                )
            num_vox = getattr(dij, "num_of_voxels", getattr(dij, "total_voxels", 0))
            binary_risk_mask = self.array_backend.zeros((num_vox,), dtype=self._dtype)

        self.binary_risk_mask = self.array_backend.asarray(binary_risk_mask, dtype=self._dtype)

        # Calculate volume correction factor k = v_new / v_old
        # v_old is 0.6 * 0.6 * 3.0 = 1.08 mm^3 (from the original paper ([1]) fit)
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
        return -self.array_backend.expm1(self.k * self.array_backend.log1p(-p))

    def _compute_chain_derivative_single_scenario(self, d_quantity, scenario_index: int) -> Array:
        d = self.array_backend.asarray(
            self._deps["physical_dose"].compute(self._w_cache).flat[scenario_index],
            dtype=self._dtype,
        )
        d_x_ld = self.array_backend.asarray(
            self._deps["let_dose"].compute(self._w_cache).flat[scenario_index], dtype=self._dtype
        )

        eta = -26.3 + 0.19 * d + 0.018 * d_x_ld + 1.19 * self.binary_risk_mask
        p = 1.0 / (1.0 + self.array_backend.exp(-eta))

        # Numerically stable gradient of volume-scaled probability with respect to eta:
        dp_k_deta = self.k * p * self.array_backend.exp(self.k * self.array_backend.log1p(-p))

        # Chain rule back to linear components
        grad_d = d_quantity * dp_k_deta * 0.19
        grad_d_x_ld = d_quantity * dp_k_deta * 0.018

        # Delegate to dependencies to push gradient all the way back to fluence (w)
        chain_d = self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            grad_d, scenario_index
        )
        chain_d_x_ld = self._deps["let_dose"]._compute_chain_derivative_single_scenario(
            grad_d_x_ld, scenario_index
        )

        return chain_d + chain_d_x_ld

    def compute_total_ntcp(
        self, fluence: Array, mask: Optional[Array] = None, scenario_index: int = 0
    ) -> float:
        """
        Compute the total Normal Tissue Complication Probability (NTCP).

        Args:
            fluence: The optimized fluence array.
            mask: Optional binary mask (as a SimpleITK image or 1D boolean array) to restrict the evaluation.
                  If provided as a SimpleITK image, it will be resampled to the dose grid.
            scenario_index: Scenario index to compute the NTCP for (default: 0).

        Returns
        -------
            The total NTCP as a float.
        """
        import numpy as np

        polo_flat = self.compute(fluence).flat[scenario_index]

        if mask is not None:
            if hasattr(mask, "GetSize"):
                import SimpleITK as sitk
                from pyRadPlan.core.resample import resample_image

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

        polo_flat_np = np.asarray(polo_flat)
        return float(1.0 - np.exp(np.sum(np.log1p(-np.clip(polo_flat_np, 0.0, 1.0 - 1e-6)))))

"""Shared base for the POLO family of fluence-dependent quantities."""

from typing import Optional

import array_api_compat

from ..core.xp_utils.typing import Array
from pyRadPlan.quantities._base import FluenceDependentQuantity

# Clip bound for POLO probabilities. Keeps a saturated voxel from driving the product to
# exactly zero; 1e-6 stays representable in float32. Public because the NTCPPOLO objective
# clips identically when forming the gradient.
P_MAX = 1.0 - 1e-6


def total_ntcp(values: Array) -> Array:
    """
    Aggregate voxel-wise POLO values into a patient-level NTCP.

    ``NTCP = 1 - prod_i (1 - p_i)``, evaluated in log space (Ortkamp et al., Equation 5).

    Parameters
    ----------
    values : Array
        Voxel-wise POLO probabilities. An empty input yields 0.0 (an empty product is 1).

    Returns
    -------
    Array
        The aggregated NTCP, as a 0-d array in the namespace of *values*.
    """
    xp = array_api_compat.array_namespace(values)
    vals = xp.clip(values, 0.0, P_MAX)
    return 1.0 - xp.exp(xp.sum(xp.log1p(-vals)))


class POLOBase(FluenceDependentQuantity):
    """
    Common machinery for quantities derived from the POLO logistic model.

    Subclasses differ only in what they expose: the linear predictor eta itself
    (:class:`~pyRadPlan.quantities.LinearPOLO`) or the volume-corrected probability derived
    from it (:class:`~pyRadPlan.quantities.POLO`). Everything else -- the model coefficients,
    resolution of the binary risk mask, the volume correction factor and the dose domain of
    the fit -- lives here.

    References
    ----------
    .. [1] Bahn, E., Bauer, J., Harrabi, S. B., et al. (2020). Late contrast enhancing brain lesions
           in proton-treated patients with low-grade glioma. Int J Radiat Oncol Biol Phys, 107(3),
           571-578. https://doi.org/10.1016/j.ijrobp.2020.03.013
    .. [2] Ortkamp, T., Sallem, H., Harrabi, S., Frank, M., Jaekel, O., Bauer, J., & Wahl, N. (2026).
           Direct optimization of the probability of lesion origin in proton treatment planning for
           low-grade glioma patients. Medical Physics, 53, e70395. https://doi.org/10.1002/mp.70395
    """

    required_dependencies = ("physical_dose", "let_dose")

    # Logistic regression coefficients of the POLO model, from Bahn et al. ([1], Table 2) and
    # restated as Equation 1 in Ortkamp et al. ([2]).
    INTERCEPT = -26.3
    BETA_DOSE = 0.19  # beta_1, per Gy
    BETA_LET_DOSE = 0.018  # beta_2, per keV / (Gy * um)
    BETA_RISK = 1.19  # beta_3, ventricular proximity (<= 4 mm)

    # Voxel volume of the planning CT the model was fitted on: 0.6 x 0.6 x 3.0 mm^3 ([2], Sec 2.1.2)
    FIT_VOXEL_VOLUME = 1.08  # mm^3

    # Physical dose domain of the fit: Bahn et al. included voxels receiving at least 2 Gy ([1]).
    DEFAULT_DOSE_THRESHOLD = 2.0  # Gy

    def __init__(
        self,
        dij,
        binary_risk_mask: Optional[Array] = None,
        cst=None,
        risk_voi_name: str = "Ventricles_4mm",
        dose_threshold: float = DEFAULT_DOSE_THRESHOLD,
        mode=None,
        dependencies=None,
        scenarios=None,
    ):
        """
        Initialize a POLO-family quantity.

        Args:
            dij: Dose influence matrix object.
            binary_risk_mask: Optional array of shape (n_voxels,) containing 1 for risk voxels.
            cst: Optional Clinical Structure Set object to extract the risk mask from.
            risk_voi_name: VOI name in CST for the periventricular risk structure
                (default: 'Ventricles_4mm'). Pass None to evaluate without the risk component.
            dose_threshold: Physical dose in Gy below which the model is treated as undefined
                (default: 2.0 Gy). Set to 0.0 to disable.
            mode: Compute mode.
            dependencies: Dictionary of dependency quantity objects.
            scenarios: Scenario indices.
        """
        super().__init__(dij, mode=mode, dependencies=dependencies, scenarios=scenarios)
        self.risk_voi_name = risk_voi_name
        self.dose_threshold = dose_threshold
        self.binary_risk_mask = self._resolve_risk_mask(binary_risk_mask, cst, risk_voi_name)

        # Volume correction factor k = v_new / v_old ([2], Equation 4). Applied by subclasses
        # that expose a probability; the linear predictor is defined before the correction.
        res = self._dij.dose_grid.resolution
        self.k = (res["x"] * res["y"] * res["z"]) / self.FIT_VOXEL_VOLUME

    def _resolve_risk_mask(self, binary_risk_mask, cst, risk_voi_name) -> Array:
        """Resolve the binary risk mask from an explicit array, a CST lookup, or the dij."""
        if binary_risk_mask is None and cst is not None:
            # Look up requested risk VOI in CST
            vois = cst.vois if hasattr(cst, "vois") else (cst if isinstance(cst, list) else [])
            for voi in vois:
                if voi.name == risk_voi_name:
                    binary_risk_mask = voi.mask
                    break

        if binary_risk_mask is None:
            binary_risk_mask = getattr(self._dij, "binary_risk_mask", None)

        if hasattr(binary_risk_mask, "mask") or hasattr(binary_risk_mask, "GetSize"):
            import SimpleITK as sitk  # noqa: PLC0415
            from pyRadPlan.core.resample import resample_image  # noqa: PLC0415

            img = binary_risk_mask.mask if hasattr(binary_risk_mask, "mask") else binary_risk_mask
            resampled = resample_image(
                img,
                target_grid=self._dij.dose_grid,
                extrapolate=0.0,
                interpolator=sitk.sitkNearestNeighbor,
            )
            binary_risk_mask = sitk.GetArrayViewFromImage(resampled).ravel() > 0.5

        if binary_risk_mask is None:
            if risk_voi_name is not None:
                raise ValueError(
                    f"POLO risk VOI '{risk_voi_name}' not found in CST and no binary_risk_mask "
                    "provided. The POLO model was originally fitted using a 'Ventricles_4mm' "
                    "risk mask. If you intentionally want to evaluate POLO without the "
                    "periventricular risk component, explicitly set `risk_voi_name=None`."
                )
            binary_risk_mask = self.array_backend.zeros(
                (self._dij.dose_grid.num_voxels,), dtype=self._dtype
            )

        return self.array_backend.asarray(binary_risk_mask, dtype=self._dtype)

    def _dose(self, fluence: Array, scenario_index: int) -> Array:
        """
        Physical dose for one scenario, in the quantity's namespace and dtype.

        *fluence* is passed in rather than read from a cache: the forward pass must evaluate
        at ``self._w_cache`` and the chain derivative at ``self._w_grad_cache``, and those are
        not guaranteed to hold the same vector.
        """
        return self.array_backend.asarray(
            self._deps["physical_dose"].compute(fluence).flat[scenario_index],
            dtype=self._dtype,
        )

    def _linear_predictor(self, fluence: Array, scenario_index: int) -> tuple[Array, Array]:
        """Return ``(eta, d)``: the linear predictor and the physical dose it was built from."""
        d = self._dose(fluence, scenario_index)
        d_x_ld = self.array_backend.asarray(
            self._deps["let_dose"].compute(fluence).flat[scenario_index], dtype=self._dtype
        )
        eta = (
            self.INTERCEPT
            + self.BETA_DOSE * d
            + self.BETA_LET_DOSE * d_x_ld
            + self.BETA_RISK * self.binary_risk_mask
        )
        return eta, d

    def _in_dose_domain(self, d: Array) -> Array:
        """Boolean mask of voxels inside the dose domain the model was fitted on."""
        return d >= self.dose_threshold

    def _backpropagate(self, grad_eta: Array, scenario_index: int) -> Array:
        """Push a gradient w.r.t. eta back to fluence through both dose dependencies."""
        chain_d = self._deps["physical_dose"]._compute_chain_derivative_single_scenario(
            grad_eta * self.BETA_DOSE, scenario_index
        )
        chain_d_x_ld = self._deps["let_dose"]._compute_chain_derivative_single_scenario(
            grad_eta * self.BETA_LET_DOSE, scenario_index
        )
        return chain_d + chain_d_x_ld

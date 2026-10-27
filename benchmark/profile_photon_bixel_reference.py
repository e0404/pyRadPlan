"""Frozen photon bixel evaluation from 9d140807 for paired comparisons."""

from typing import Any, cast

import array_api_compat
import array_api_extra as xpx

from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator, to_namespace
from pyRadPlan.machines import PhotonSVDKernel


def _compute_bixel(self, curr_ray: dict[str], k: int) -> dict[str, Any]:
    """
    PyRadPlan photon dose calculation for a single bixel.

    call
        bixel = self.computeBixel(currRay,k)
    """

    xp = self.xp
    bixel = {}

    kernel = cast(PhotonSVDKernel, curr_ray["kernel"])

    m = kernel.m
    betas = kernel.kernel_betas
    rd = curr_ray["rad_depths"]
    interpolators = cast(ArrayBackendGridInterpolator, curr_ray["kernel_interpolators"])
    iso_lat_dists = curr_ray["iso_lat_dists"]
    geo_depths = curr_ray["geo_depths"]
    sad = curr_ray["sad"]

    # Reshape for broadcasting: betas (n_components, 1), rd (1, n_voxels)
    device = array_api_compat.device(rd)
    betas = xp.reshape(xp.asarray(betas, dtype=rd.dtype, device=device), (-1, 1))
    rd = xp.reshape(rd, (1, -1))
    m_arr = xp.asarray(m, dtype=rd.dtype, device=device)

    dose_component = betas / (betas - m_arr) * (xp.exp(-m_arr * rd) - xp.exp(-betas * rd))

    iso_lat_dists = to_namespace(xp, iso_lat_dists, device=self.device)
    interpolated_kernels = interpolators(iso_lat_dists)

    for c in range(interpolated_kernels.shape[1]):
        dose_component = xpx.at(dose_component, (c, slice(None))).set(
            dose_component[c, :] * interpolated_kernels[:, c]
        )

    bixel_dose = xp.sum(dose_component, axis=0)

    bixel_dose = bixel_dose * ((sad / geo_depths) ** 2)

    bixel["physical_dose"] = bixel_dose
    bixel["weight"] = curr_ray["beamlets"][k]["weight"]
    bixel["ix"] = curr_ray["ix"]

    if self.enable_dij_sampling:
        bixel["ix"], bixel["physical_dose"] = self._sample_dij(
            curr_ray["ix"],
            bixel_dose,
            curr_ray["rad_depths"],
            curr_ray["radial_dist_sq"],
            bixel_width=curr_ray["bixel_width"],
        )

    return bixel

"""Frozen photon hot paths from 0c298044 for paired performance comparisons."""

from typing import Any, Callable, cast

import numpy as np

import array_api_compat

import array_api_extra as xpx

from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator

from pyRadPlan.core.xp_utils.helpers import to_namespace, scatter

from pyRadPlan.core.xp_utils.typing import Array

from pyRadPlan.core.xp_utils.compat import _fft2, _ifft2

from pyRadPlan.machines import PhotonSVDKernel

from typing import Optional


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
    interpolators = cast(list[Callable], curr_ray["kernel_interpolators"])
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
    interpolated_kernels = [interp(iso_lat_dists) for interp in interpolators]

    for c, interp in enumerate(interpolated_kernels):
        dose_component = xpx.at(dose_component, (c, slice(None))).set(
            dose_component[c, :] * xp.asarray(interp)
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


def _get_kernel_interpolators(self, beam_info: dict[str], f: Array) -> list[Callable]:
    """Prepare interpolators for this fluence's convolved photon kernels."""

    xp = self.xp

    num_kernels = cast(PhotonSVDKernel, beam_info["kernel"]).num_kernel_components
    conv_size = beam_info["kernel_conv_size"]

    f = to_namespace(xp, f, device=self.device)
    kernel_mxs = to_namespace(xp, beam_info["kernel_mxs"], device=self.device)
    conv_grid = to_namespace(xp, beam_info["kernel_conv_grid"], device=self.device)

    interpolators = [None] * num_kernels
    for c in range(num_kernels):
        conv_mx = xp.real(
            _ifft2(
                _fft2(f, s=(conv_size, conv_size))
                * _fft2(kernel_mxs[c, ...], s=(conv_size, conv_size))
            )
        )

        # A query outside the convolution grid indicates incorrect grid sizing.
        interpolators[c] = ArrayBackendGridInterpolator(
            (conv_grid, conv_grid), conv_mx, bounds_error=True
        )

    return interpolators


def _sample_dij(
    self,
    ix: Array,
    bixel_dose: Array,
    rad_depth_v: Array,
    rad_distances_sq: Array,
    bixel_width: Optional[float] = None,
) -> tuple[Array, Array]:
    """
    Sample the lateral dose tail of a bixel to sparsify the dose influence matrix.

    Voxels inside the sampling core (either within a radius around the central ray or
    above a relative dose threshold) are kept untouched. The remaining voxels are
    clustered in radiological depth, and within each cluster a voxel is accepted with
    a probability proportional to its dose. An accepted voxel then carries the maximum
    dose of its cluster, which keeps the sampled tail dose unbiased in expectation
    while reducing the number of non-zeros [1]_.

    Parameters
    ----------
    ix : Array
        Linear indices (or a boolean mask) of the voxels the dose was computed on.
    bixel_dose : Array
        Dose at these voxels.
    rad_depth_v : Array
        Radiological depth at these voxels.
    rad_distances_sq : Array
        Squared radial distance of these voxels to the central ray.
    bixel_width : float, optional
        Bixel width [mm]. If given, half of the bixel diagonal is added to the
        sampling radius, since dose is deposited over the whole bixel and not only
        around its central ray.

    Returns
    -------
    tuple[Array, Array]
        The reduced voxel indices and their sampled dose values, in the order of the
        passed indices.

    References
    ----------
    .. [1] https://doi.org/10.1118/1.1469633
    """

    # Infer the array namespace and device from the dose vector
    xp = array_api_compat.array_namespace(bixel_dose)
    device = array_api_compat.device(bixel_dose)

    rel_dose_threshold = self.dij_sampling["rel_dose_threshold"]
    lat_cut_off = self.dij_sampling["lat_cut_off"]
    sample_type = self.dij_sampling["type"]
    delta_rad_depth = self.dij_sampling["delta_rad_depth"]

    # If the input index vector is boolean, convert it to linear indices
    if ix.dtype == xp.bool:
        ix = xp.nonzero(ix)[0]

    if bixel_dose.shape[0] == 0:
        return ix, bixel_dose

    # Increase sample cut-off by bixel width if given. Half of the bixel width diagonal
    # is the max. field size radius for sampling. Kept a Python float, since mixing a
    # backend scalar into the comparison below is not allowed for all namespaces.
    if bixel_width is not None:
        lat_cut_off = float(lat_cut_off + bixel_width / np.sqrt(2.0))

    # Identify core voxels
    if sample_type == "radius":
        ix_core = rad_distances_sq < lat_cut_off**2
    elif sample_type == "dose":
        ix_core = bixel_dose > rel_dose_threshold * xp.max(bixel_dose)
    else:
        raise ValueError(f"Dij Sampling mode '{sample_type}' not known!")

    # All voxels are in the core - exit with the unsampled dose
    if bool(xp.all(ix_core)):
        return ix, bixel_dose

    ix_tail = ~ix_core
    dose_tail = xp.where(ix_tail, bixel_dose, xp.zeros_like(bixel_dose))

    # Cluster the radiological depths to reduce computations. Clipping guards against
    # the negative depths range shift scenarios can produce, which would otherwise
    # give negative cluster indices.
    cluster = xp.astype(xp.clip(rad_depth_v, 0.0) / delta_rad_depth, xp.int64)
    num_clusters = int(xp.max(cluster)) + 1

    # Maximum tail dose per cluster. The Array API offers no scatter-maximum, so the
    # voxels are sorted by cluster with the dose as tie-breaker, which puts the maximum
    # last in each cluster's block. Core voxels enter with zero dose and cannot win.
    order = xp.argsort(dose_tail, stable=True)
    order = xp.take(order, xp.argsort(xp.take(cluster, order), stable=True))
    sorted_cluster = xp.take(cluster, order)
    block_end = xp.concat(
        (
            sorted_cluster[:-1] != sorted_cluster[1:],
            xp.ones((1,), dtype=xp.bool, device=device),
        )
    )

    cluster_dose = xp.zeros((num_clusters,), dtype=bixel_dose.dtype, device=device)
    cluster_dose = scatter(
        cluster_dose, sorted_cluster[block_end], xp.take(dose_tail, order)[block_end]
    )
    threshold_dose = xp.take(cluster_dose, cluster)

    # Get random samples. The Array API has no random number generation, so they are
    # drawn on the host and moved to the namespace of the dose.
    r = xp.asarray(
        self._rng.random(int(bixel_dose.shape[0])), dtype=bixel_dose.dtype, device=device
    )

    # r <= dose/threshold, as a product to not divide by a cluster without tail dose
    ix_samp = (r * threshold_dose <= dose_tail) & (threshold_dose > 0.0)

    # Masking the input instead of concatenating core and sampled tail keeps the
    # indices ascending, as promised by _dij_guarantee_canonical
    keep = ix_core | ix_samp

    return ix[keep], xp.where(ix_tail, threshold_dose, bixel_dose)[keep]

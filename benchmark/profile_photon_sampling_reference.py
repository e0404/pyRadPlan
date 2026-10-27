"""Sampling from 5d7f6266 with an optional supplied count for isolated comparisons.

With num_clusters=None this is the previous algorithm. Supplying the prepared
count isolates the old Boolean output selection from cluster-count preparation.
"""

from typing import Optional

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import grouped_max
from pyRadPlan.core.xp_utils.typing import Array


def _sample_dij(
    self,
    ix: Array,
    bixel_dose: Array,
    rad_depth_v: Array,
    rad_distances_sq: Array,
    bixel_width: Optional[float] = None,
    *,
    num_clusters: Optional[int] = None,
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
    if num_clusters is None:
        num_clusters = int(xp.max(cluster)) + 1

    cluster_dose = grouped_max(dose_tail, cluster, num_clusters, initial=0.0)

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

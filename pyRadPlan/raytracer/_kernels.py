"""
Backend-specialized computational kernels for the ray tracers.

Each kernel is generic array-API code wrapped by
:func:`pyRadPlan.core.xp_utils.jittable`: backends enabled in
``settings.xp.jit_backends`` run a jit-compiled version (the numba
implementations from :mod:`._kernels_numba` for NumPy when numba is installed,
the backend's own jit elsewhere), every other backend runs the generic code
unchanged.
"""

from contextlib import nullcontext

import array_api_compat
import numpy as np

from ..core import xp_utils
from ..core.xp_utils.jittable import jittable
from ..core.xp_utils.typing import Array


@jittable(backends=("jax", "torch"))
def compute_plane_alphas(
    dim_min: Array,
    dim_max: Array,
    planes: Array,
    source: Array,
    ray: Array,
    plane_ix: Array,
) -> Array:
    """
    Compute the ray parameters of the intersections with one axis' voxel planes.

    Plane indices outside ``[dim_min, dim_max]`` (per ray), degenerate single-plane
    ranges and NaN limits are masked with NaN.
    """
    xp = array_api_compat.array_namespace(dim_min, dim_max, planes, source, ray)

    plane_ix = plane_ix[: planes.shape[0]][None, :]
    low = plane_ix < dim_min[:, None]
    high = plane_ix > dim_max[:, None]
    deg = (plane_ix == dim_min[:, None]) & (plane_ix == dim_max[:, None])
    nanm = xp.isnan(dim_min)[:, None] | xp.isnan(dim_max)[:, None]
    mask_invalid = low | high | deg | nanm

    if array_api_compat.is_numpy_array(ray):
        errstate = np.errstate(divide="ignore", invalid="ignore")
    else:
        errstate = nullcontext()
    with errstate:
        alphas = (planes[None, :] - source[:, None]) / ray[:, None]

    return xp.where(mask_invalid, xp.nan, alphas)


@jittable(backends=("jax", "torch"))
def merge_sorted_unique(
    alpha_limits: Array, alpha_x: Array, alpha_y: Array, alpha_z: Array
) -> Array:
    """
    Merge the per-axis plane alphas into row-wise sorted, deduplicated sets.

    Duplicates are pushed to the end of each row as inf (a second sort after
    masking), so every row is sorted, unique and front-compacted.
    """
    xp = array_api_compat.array_namespace(alpha_limits, alpha_x, alpha_y, alpha_z)

    alphas = xp.concat((alpha_limits, alpha_x, alpha_y, alpha_z), axis=1)
    alphas = xp.sort(alphas, axis=1)
    mask = xp.diff(alphas, axis=1, prepend=xp.full_like(alphas[:, :1], xp.inf)) == 0
    return xp.sort(xp.where(mask, xp.inf, alphas), axis=1)


@jittable(backends=("jax", "torch"))
def compute_indices_from_alpha(
    source_points: Array,
    ray_vec: Array,
    alphas_mid: Array,
    cube_origin: Array,
    resolution: Array,
    cube_dim: Array,
) -> tuple[Array, Array]:
    """
    Convert segment-midpoint ray parameters to voxel indices with a validity mask.

    Returns ``(val_ix, ijk)`` with ``ijk`` of shape ``(rays, 3, segments)`` in
    int32 and ``val_ix`` marking midpoints inside the cube.
    """
    xp = array_api_compat.array_namespace(source_points, ray_vec, alphas_mid)

    sp_scaled = (source_points - cube_origin) / resolution
    rv_scaled = ray_vec / resolution

    ijk = sp_scaled[:, :, None] + rv_scaled[:, :, None] * alphas_mid[:, None, :]
    ijk = xp.where(xp.isfinite(ijk), ijk, -1.0)
    ijk = xp.astype(xp.round(ijk), xp.int32)

    val_ix = xp.all((ijk >= 0) & (ijk < cube_dim[None, :, None]), axis=1)
    return val_ix, ijk


@jittable(backends=())  # nonzero/boolean gathers have data-dependent shapes
def select_rad_depth_segments(
    ix: Array,
    index_to_bev: Array,
    bev_offset: Array,
    ray_x: Array,
    ray_z: Array,
    scale_num: float,
    ray_selection: float,
    num_voxels: int,
    ny: int,
    nz: int,
) -> Array:
    """
    Mark the traced segments whose voxel receives its radiological depth value.

    A segment is selected when its voxel center, mapped to BEV through the
    composed affine ``index_to_bev``/``bev_offset`` and projected to the
    ray-matrix plane, falls within the half-open selection square of its ray
    position ``(ray_x, ray_z)``.
    """
    xp = array_api_compat.array_namespace(ix)
    device = array_api_compat.device(ix)
    precision = index_to_bev.dtype
    # We don't want -1 to be counted as "valid" or else gathering at ix silently
    # reads the last element; indices past the end would raise instead
    valid_ix = (ix >= 0) & (ix < num_voxels)

    # Work only on the valid segments and scatter the selection back into the full
    # (rays, segments) shape. Index decode is linear and the image and beam
    # transforms are affine, so voxel index -> BEV coordinates is a single affine
    # map, applied in working precision
    rows = xp.nonzero(valid_ix)[0]
    gathered_ix = ix[valid_ix]
    if gathered_ix.dtype != xp.int32 and num_voxels < np.iinfo(np.int32).max:
        gathered_ix = xp.astype(gathered_ix, xp.int32)

    # "sitk" linear index layout is z-fastest: z + nz * (y + ny * x)
    tmp = gathered_ix // nz
    i_f = xp.astype(tmp // ny, precision)
    j_f = xp.astype(tmp % ny, precision)
    k_f = xp.astype(gathered_ix % nz, precision)

    if xp_utils.openblas_has_gemm_race() and array_api_compat.is_numpy_namespace(xp):
        # elementwise fallback: this OpenBLAS silently produces wrong, run-to-run
        # varying elements for tall-skinny (M, 3) @ (3, 3) matmuls
        bev_x, bev_y, bev_z = (
            i_f * float(index_to_bev[0, c])
            + j_f * float(index_to_bev[1, c])
            + k_f * float(index_to_bev[2, c])
            + float(bev_offset[c])
            for c in range(3)
        )
    else:
        coords_bev = xp.stack((i_f, j_f, k_f), axis=1) @ index_to_bev + bev_offset
        bev_x, bev_y, bev_z = coords_bev[:, 0], coords_bev[:, 1], coords_bev[:, 2]

    scale_factor = scale_num / bev_y
    x_dist = bev_x * scale_factor - xp.take(ray_x, rows)
    z_dist = bev_z * scale_factor - xp.take(ray_z, rows)

    selection_mask = xp.zeros(ix.shape, dtype=xp.bool, device=device)
    return xp_utils.scatter(
        selection_mask,
        valid_ix,
        (x_dist > -ray_selection)
        & (x_dist <= ray_selection)
        & (z_dist > -ray_selection)
        & (z_dist <= ray_selection),
    )


try:
    from . import _kernels_numba  # noqa: F401  (registers the numba implementations)
except ImportError:
    pass

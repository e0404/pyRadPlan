"""
Numba implementations of the ray tracer kernels for the NumPy backend.

Importing this module registers the implementations on the dispatchable kernels
in :mod:`._kernels`; it is only imported when numba is installed.
"""

import numpy as np
from numba import njit, prange

from ._kernels import compute_indices_from_alpha, select_rad_depth_segments


@njit(parallel=True, cache=True)
def _indices_from_alpha_njit(
    source_points, ray_vec, alphas_mid, cube_origin, resolution, cube_dim
):
    num_rays, num_segments = alphas_mid.shape
    ijk = np.empty((num_rays, 3, num_segments), np.int32)
    val_ix = np.empty((num_rays, num_segments), np.bool_)
    d0, d1, d2 = cube_dim[0], cube_dim[1], cube_dim[2]

    for r in prange(num_rays):
        sp0 = (np.float64(source_points[r, 0]) - cube_origin[0]) / resolution[0]
        sp1 = (np.float64(source_points[r, 1]) - cube_origin[1]) / resolution[1]
        sp2 = (np.float64(source_points[r, 2]) - cube_origin[2]) / resolution[2]
        rv0 = np.float64(ray_vec[r, 0]) / resolution[0]
        rv1 = np.float64(ray_vec[r, 1]) / resolution[1]
        rv2 = np.float64(ray_vec[r, 2]) / resolution[2]

        for s in range(num_segments):
            alpha = np.float64(alphas_mid[r, s])
            x = sp0 + rv0 * alpha
            y = sp1 + rv1 * alpha
            z = sp2 + rv2 * alpha
            if not np.isfinite(x):
                x = -1.0
            if not np.isfinite(y):
                y = -1.0
            if not np.isfinite(z):
                z = -1.0
            xi = np.int32(np.rint(x))
            yi = np.int32(np.rint(y))
            zi = np.int32(np.rint(z))
            ijk[r, 0, s] = xi
            ijk[r, 1, s] = yi
            ijk[r, 2, s] = zi
            val_ix[r, s] = 0 <= xi < d0 and 0 <= yi < d1 and 0 <= zi < d2

    return val_ix, ijk


@compute_indices_from_alpha.register("numpy")
def _indices_from_alpha_numpy(
    source_points, ray_vec, alphas_mid, cube_origin, resolution, cube_dim
):
    return _indices_from_alpha_njit(
        source_points,
        ray_vec,
        alphas_mid,
        np.asarray(cube_origin, dtype=np.float64),
        np.asarray(resolution, dtype=np.float64),
        np.asarray(cube_dim, dtype=np.int64),
    )


@njit(parallel=True, cache=True)
def _select_segments_njit(
    ix, index_to_bev, bev_offset, ray_x, ray_z, scale_num, ray_selection, num_voxels, ny, nz
):
    num_rays, num_segments = ix.shape
    out = np.zeros((num_rays, num_segments), np.bool_)

    m00, m01, m02 = index_to_bev[0, 0], index_to_bev[0, 1], index_to_bev[0, 2]
    m10, m11, m12 = index_to_bev[1, 0], index_to_bev[1, 1], index_to_bev[1, 2]
    m20, m21, m22 = index_to_bev[2, 0], index_to_bev[2, 1], index_to_bev[2, 2]
    b0, b1, b2 = bev_offset[0], bev_offset[1], bev_offset[2]
    scale_num32 = np.float32(scale_num)
    selection32 = np.float32(ray_selection)

    for r in prange(num_rays):
        rx = ray_x[r]
        rz = ray_z[r]
        for s in range(num_segments):
            voxel = ix[r, s]
            if voxel < 0 or voxel >= num_voxels:
                continue
            # "sitk" linear index layout is z-fastest: z + nz * (y + ny * x)
            tmp = voxel // nz
            # the per-axis indices are bounded by the cube dimensions, so the
            # float32 casts are exact
            i_f = np.float32(tmp // ny)
            j_f = np.float32(tmp % ny)
            k_f = np.float32(voxel % nz)

            bev_x = i_f * m00 + j_f * m10 + k_f * m20 + b0
            bev_y = i_f * m01 + j_f * m11 + k_f * m21 + b1
            bev_z = i_f * m02 + j_f * m12 + k_f * m22 + b2

            scale_factor = scale_num32 / bev_y
            x_dist = bev_x * scale_factor - rx
            z_dist = bev_z * scale_factor - rz
            out[r, s] = (
                -selection32 < x_dist <= selection32 and -selection32 < z_dist <= selection32
            )

    return out


@select_rad_depth_segments.register("numpy")
def _select_segments_numpy(
    ix, index_to_bev, bev_offset, ray_x, ray_z, scale_num, ray_selection, num_voxels, ny, nz
):
    # the kernel hardcodes float32 working precision
    if index_to_bev.dtype != np.float32:
        return NotImplemented
    return _select_segments_njit(
        ix,
        index_to_bev,
        np.asarray(bev_offset, dtype=np.float32),
        np.ascontiguousarray(ray_x),
        np.ascontiguousarray(ray_z),
        float(scale_num),
        float(ray_selection),
        num_voxels,
        ny,
        nz,
    )

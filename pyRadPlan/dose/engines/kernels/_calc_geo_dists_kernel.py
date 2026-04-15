from ....core.xp_utils import cupy_available

import numpy as np
import math


class _NotAvailableKernel:
    """Placeholder for kernels that are not available."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(f"{self._name}-kernel for calc_geo_dists is not available.")


# Numba CUDA kernel
try:
    from numba import cuda

    @cuda.jit
    def _calc_geo_dists_numba_kernel(
        rot_coords_bev,
        source_point_bev,
        target_point_bev,
        sad,
        lateral_cutoff,
        nb_rays,
        m,
        rot_coords_temp,
        subset_mask,
        rad_distances_sq,
        lat_dists,
    ):
        i = cuda.grid(1)
        if i >= m:
            return
        a0 = -source_point_bev[0]
        a1 = -source_point_bev[1]
        a2 = -source_point_bev[2]
        norm_a = math.sqrt(a0 * a0 + a1 * a1 + a2 * a2)
        a0 /= norm_a
        a1 /= norm_a
        a2 /= norm_a
        bx = target_point_bev[0] - source_point_bev[0]
        by = target_point_bev[1] - source_point_bev[1]
        bz = target_point_bev[2] - source_point_bev[2]
        norm_b = math.sqrt(bx * bx + by * by + bz * bz)
        bx /= norm_b
        by /= norm_b
        bz /= norm_b
        cx = a1 * bz - a2 * by
        cy = a2 * bx - a0 * bz
        cz = a0 * by - a1 * bx
        cnorm = math.sqrt(cx * cx + cy * cy + cz * cz)

        ##################################################
        tolerance = 1e-7
        if abs(a0 - bx) < tolerance and abs(a1 - by) < tolerance and abs(a2 - bz) < tolerance:
            rot_coords_temp[i, 0] = rot_coords_bev[i][0]
            rot_coords_temp[i, 1] = rot_coords_bev[i][1]
            rot_coords_temp[i, 2] = rot_coords_bev[i][2]
        else:
            cross = cuda.local.array((3, 3), dtype=np.float32)
            ################################################## skew matrix cross product building
            cross[0, 0] = 0.0
            cross[0, 1] = -cz
            cross[0, 2] = cy
            cross[1, 0] = cz
            cross[1, 1] = 0.0
            cross[1, 2] = -cx
            cross[2, 0] = -cy
            cross[2, 1] = cx
            cross[2, 2] = 0.0
            ##################################################
            cross2 = cuda.local.array((3, 3), dtype=np.float32)
            ##################################################
            cross2[0, 0] = (
                cross[0, 0] * cross[0, 0] + cross[0, 1] * cross[1, 0] + cross[0, 2] * cross[2, 0]
            )
            cross2[0, 1] = (
                cross[0, 0] * cross[0, 1] + cross[0, 1] * cross[1, 1] + cross[0, 2] * cross[2, 1]
            )
            cross2[0, 2] = (
                cross[0, 0] * cross[0, 2] + cross[0, 1] * cross[1, 2] + cross[0, 2] * cross[2, 2]
            )
            cross2[1, 0] = (
                cross[1, 0] * cross[0, 0] + cross[1, 1] * cross[1, 0] + cross[1, 2] * cross[2, 0]
            )
            cross2[1, 1] = (
                cross[1, 0] * cross[0, 1] + cross[1, 1] * cross[1, 1] + cross[1, 2] * cross[2, 1]
            )
            cross2[1, 2] = (
                cross[1, 0] * cross[0, 2] + cross[1, 1] * cross[1, 2] + cross[1, 2] * cross[2, 2]
            )
            cross2[2, 0] = (
                cross[2, 0] * cross[0, 0] + cross[2, 1] * cross[1, 0] + cross[2, 2] * cross[2, 0]
            )
            cross2[2, 1] = (
                cross[2, 0] * cross[0, 1] + cross[2, 1] * cross[1, 1] + cross[2, 2] * cross[2, 1]
            )
            cross2[2, 2] = (
                cross[2, 0] * cross[0, 2] + cross[2, 1] * cross[1, 2] + cross[2, 2] * cross[2, 2]
            )
            #################################################
            offset = 1 - (a0 * bx + a1 * by + a2 * bz)
            derived_rot_mat = cuda.local.array((3, 3), dtype=np.float32)
            cnorm *= cnorm
            ##################################################
            derived_rot_mat[0, 0] = 1.0 + cross[0, 0] + (cross2[0, 0] * offset / (cnorm))
            derived_rot_mat[0, 1] = cross[0, 1] + (cross2[0, 1] * offset / (cnorm))
            derived_rot_mat[0, 2] = cross[0, 2] + (cross2[0, 2] * offset / (cnorm))
            derived_rot_mat[1, 0] = cross[1, 0] + (cross2[1, 0] * offset / (cnorm))
            derived_rot_mat[1, 1] = 1.0 + cross[1, 1] + (cross2[1, 1] * offset / (cnorm))
            derived_rot_mat[1, 2] = cross[1, 2] + (cross2[1, 2] * offset / (cnorm))
            derived_rot_mat[2, 0] = cross[2, 0] + (cross2[2, 0] * offset / (cnorm))
            derived_rot_mat[2, 1] = cross[2, 1] + (cross2[2, 1] * offset / (cnorm))
            derived_rot_mat[2, 2] = 1.0 + cross[2, 2] + (cross2[2, 2] * offset / (cnorm))
            ##################################################
            rot_coords_temp[i, 0] = (
                rot_coords_bev[i, 0] * derived_rot_mat[0, 0]
                + rot_coords_bev[i, 1] * derived_rot_mat[1, 0]
                + rot_coords_bev[i, 2] * derived_rot_mat[2, 0]
            )
            rot_coords_temp[i, 1] = (
                rot_coords_bev[i, 0] * derived_rot_mat[0, 1]
                + rot_coords_bev[i, 1] * derived_rot_mat[1, 1]
                + rot_coords_bev[i, 2] * derived_rot_mat[2, 1]
            )
            rot_coords_temp[i, 2] = (
                rot_coords_bev[i, 0] * derived_rot_mat[0, 2]
                + rot_coords_bev[i, 1] * derived_rot_mat[1, 2]
                + rot_coords_bev[i, 2] * derived_rot_mat[2, 2]
            )

        ##################################################
        lat_dists[i, 0] = rot_coords_temp[i, 0] + source_point_bev[0]
        lat_dists[i, 1] = rot_coords_temp[i, 2] + source_point_bev[2]
        rad_distances_sq[i] = lat_dists[i, 0] ** 2 + lat_dists[i, 1] ** 2
        subset_mask[i] = (
            rad_distances_sq[i] <= (lateral_cutoff / sad) ** 2 * rot_coords_temp[i, 1] ** 2
        )

except ImportError:
    _calc_geo_dists_numba_kernel = _NotAvailableKernel("Numba CUDA")


# CuPy RawKernel
if cupy_available():
    import cupy as cp

    _calc_geo_dists_cupy_raw_kernel2 = cp.RawKernel(
        r"""
    extern "C" __global__ void geo_dist(float* rot_coords_bev, float* source_point_bev, float* target_point_bev, float sad, float lateral_cutoff,int nb_rays,int m,float* rot_coords_temp,int* subset_mask, float* rad_distances_sq, float* lat_dists)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= m)
         return;
        float a0 = -source_point_bev[0];
        float a1 = -source_point_bev[1];
        float a2 = -source_point_bev[2];
        float norm_a = rsqrtf(a0 * a0 + a1 * a1 + a2 * a2);
        a0 *= norm_a;
        a1 *= norm_a;
        a2 *= norm_a;
        float bx = target_point_bev[0] - source_point_bev[0];
        float by = target_point_bev[1] - source_point_bev[1];
        float bz = target_point_bev[2] - source_point_bev[2];
        float norm_b = rsqrtf(bx*bx + by*by + bz*bz);
        bx *= norm_b;
        by *= norm_b;
        bz *= norm_b;
        float cx = a1 * bz - a2 * by;
        float cy = a2 * bx - a0 * bz;
        float cz = a0 * by - a1 * bx;
        float cnorm=cx * cx + cy * cy + cz * cz;
        float tolerance = 1e-7f;
        float cross[9];
        float cross2[9];
        float derived_rot_mat[9];

        if (fabsf(a0 - bx) < tolerance && fabsf(a1 - by) < tolerance && fabsf(a2 - bz) < tolerance)
        {
            rot_coords_temp[i*3+0]=rot_coords_bev[i*3+0];
            rot_coords_temp[i*3+1]=rot_coords_bev[i*3+1];
            rot_coords_temp[i*3+2]=rot_coords_bev[i*3+2];
        }
        else
        {
            cross[0*3+0] = 0.0f;
            cross[0*3+1] = -cz;
            cross[0*3+2] = cy;
            cross[1*3+0] = cz;
            cross[1*3+1] = 0.0f;
            cross[1*3+2] = -cx;
            cross[2*3+0] = -cy;
            cross[2*3+1] = cx;
            cross[2*3+2] = 0.0f;
            cross2[0*3+0] = cross[0*3+0] * cross[0*3+0]+cross[0*3+1] * cross[1*3+0]+cross[0*3+2] * cross[2*3+0];
            cross2[0*3+1] = cross[0*3+0] * cross[0*3+1]+cross[0*3+1] * cross[1*3+1]+cross[0*3+2] * cross[2*3+1];
            cross2[0*3+2] = cross[0*3+0] * cross[0*3+2]+cross[0*3+1] * cross[1*3+2]+cross[0*3+2] * cross[2*3+2];
            cross2[1*3+0] = cross[1*3+0] * cross[0*3+0]+cross[1*3+1] * cross[1*3+0]+cross[1*3+2] * cross[2*3+0];
            cross2[1*3+1] = cross[1*3+0] * cross[0*3+1]+cross[1*3+1] * cross[1*3+1]+cross[1*3+2] * cross[2*3+1];
            cross2[1*3+2] = cross[1*3+0] * cross[0*3+2]+cross[1*3+1] * cross[1*3+2]+cross[1*3+2] * cross[2*3+2];
            cross2[2*3+0] = cross[2*3+0] * cross[0*3+0]+cross[2*3+1] * cross[1*3+0]+cross[2*3+2] * cross[2*3+0];
            cross2[2*3+1] = cross[2*3+0] * cross[0*3+1]+cross[2*3+1] * cross[1*3+1]+cross[2*3+2] * cross[2*3+1];
            cross2[2*3+2] = cross[2*3+0] * cross[0*3+2]+cross[2*3+1] * cross[1*3+2]+cross[2*3+2] * cross[2*3+2];
            float offset=1-(a0*bx+a1*by+a2*bz);
            derived_rot_mat[0*3+0] = 1.0f + cross[0*3+0] + (cross2[0*3+0] *offset/(cnorm));
            derived_rot_mat[0*3+1] =  cross[0*3+1] + (cross2[0*3+1] * offset/(cnorm));
            derived_rot_mat[0*3+2] =  cross[0*3+2] + (cross2[0*3+2] * offset/(cnorm));
            derived_rot_mat[1*3+0] =  cross[1*3+0] + (cross2[1*3+0] * offset/(cnorm));
            derived_rot_mat[1*3+1] = 1.0f + cross[1*3+1] + (cross2[1*3+1] * offset/(cnorm));
            derived_rot_mat[1*3+2] =  cross[1*3+2] + (cross2[1*3+2] * offset/(cnorm));
            derived_rot_mat[2*3+0] =  cross[2*3+0] + (cross2[2*3+0] * offset/(cnorm));
            derived_rot_mat[2*3+1] =  cross[2*3+1] + (cross2[2*3+1] * offset/(cnorm));
            derived_rot_mat[2*3+2] =  1.0f + cross[2*3+2] + (cross2[2*3+2] * offset/(cnorm));
            rot_coords_temp[i*3+0] = rot_coords_bev[i*3+0] * derived_rot_mat[0*3+0]+rot_coords_bev[i*3+1] * derived_rot_mat[1*3+0]+rot_coords_bev[i*3+2] * derived_rot_mat[2*3+0];
            rot_coords_temp[i*3+1] = rot_coords_bev[i*3+0] * derived_rot_mat[0*3+1]+rot_coords_bev[i*3+1] * derived_rot_mat[1*3+1]+rot_coords_bev[i*3+2] * derived_rot_mat[2*3+1];
            rot_coords_temp[i*3+2] = rot_coords_bev[i*3+0] * derived_rot_mat[0*3+2]+rot_coords_bev[i*3+1] * derived_rot_mat[1*3+2]+rot_coords_bev[i*3+2] * derived_rot_mat[2*3+2];
        }
        lat_dists[i*2+0]=rot_coords_temp[i*3+0]+source_point_bev[0];
        lat_dists[i*2+1]=rot_coords_temp[i*3+2]+source_point_bev[2];
        rad_distances_sq[i]=lat_dists[i*2+0]*lat_dists[i*2+0]+lat_dists[i*2+1]*lat_dists[i*2+1];
        subset_mask[i]=rad_distances_sq[i]<=((lateral_cutoff/sad)*(lateral_cutoff/sad)*rot_coords_temp[i*3+1]*rot_coords_temp[i*3+1]);

    }
    """,
        "geo_dist",
    )
else:
    _calc_geo_dists_cupy_raw_kernel2 = _NotAvailableKernel("CuPy")

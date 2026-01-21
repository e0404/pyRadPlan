from ....core.xp_utils import numba_cuda_available, pytorch_gpu_available

if numba_cuda_available():
    from numba import cuda

    @cuda.jit
    def _calc_geo_dists_cupy_kernel(
        coords,
        rot_mat,
        source_point,
        lateral_cutoff_sq_over_sad_sq,
        out_coords_temp,
        out_lat_dists,
        out_rad_dist_sq,
        out_mask,
    ):
        i = cuda.grid(1)
        if i < coords.shape[0]:
            # Matrix multiplication: coords[i] @ rot_mat
            # Unroll for 3x3
            x = coords[i, 0]
            y = coords[i, 1]
            z = coords[i, 2]

            rx = x * rot_mat[0, 0] + y * rot_mat[1, 0] + z * rot_mat[2, 0]
            ry = x * rot_mat[0, 1] + y * rot_mat[1, 1] + z * rot_mat[2, 1]
            rz = x * rot_mat[0, 2] + y * rot_mat[1, 2] + z * rot_mat[2, 2]

            out_coords_temp[i, 0] = rx
            out_coords_temp[i, 1] = ry
            out_coords_temp[i, 2] = rz

            # lat_dists
            lx = rx + source_point[0]
            lz = rz + source_point[2]

            out_lat_dists[i, 0] = lx
            out_lat_dists[i, 1] = lz

            # rad_dist_sq
            r_sq = lx * lx + lz * lz
            out_rad_dist_sq[i] = r_sq

            # mask
            # (lateral_cutoff / sad) ** 2 * ry ** 2
            limit = lateral_cutoff_sq_over_sad_sq * ry * ry

            out_mask[i] = r_sq <= limit


if pytorch_gpu_available():
    import torch

    @torch.jit.script
    def _calc_geo_dists_torch_kernel(
        coords: torch.Tensor,
        rot_mat: torch.Tensor,
        source_point: torch.Tensor,
        lateral_cutoff_sq_over_sad_sq: float,
    ):
        # Matrix multiplication: coords @ rot_mat
        out_coords_temp = torch.mm(coords, rot_mat)

        # lat_dists
        lx = out_coords_temp[:, 0] + source_point[0]
        lz = out_coords_temp[:, 2] + source_point[2]

        out_lat_dists = torch.stack((lx, lz), dim=1)

        # rad_dist_sq
        r_sq = lx * lx + lz * lz

        # mask
        ry = out_coords_temp[:, 1]
        limit = lateral_cutoff_sq_over_sad_sq * ry * ry

        out_mask = r_sq <= limit

        return out_coords_temp, out_lat_dists, r_sq, out_mask
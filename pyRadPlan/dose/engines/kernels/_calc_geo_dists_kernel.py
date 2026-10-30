from collections.abc import Callable
from typing import Optional

from ....core.xp_utils import cupy_available, pytorch_gpu_available


class _NotAvailableKernel:
    """Placeholder for kernels that are not available."""

    def __init__(self, name: str):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise ImportError(f"{self._name}-kernel for calc_geo_dists is not available.")


class _LazyKernel:
    """Kernel built on its first call, so importing this module loads no GPU backend."""

    def __init__(self, build: Callable[[], Optional[Callable]], name: str):
        self._build = build
        self._name = name
        self._kernel = None

    def __call__(self, *args, **kwargs):
        if self._kernel is None:
            self._kernel = self._build() or _NotAvailableKernel(self._name)
        return self._kernel(*args, **kwargs)


def _build_cupy_kernel():
    if not cupy_available():
        return None
    import cupy as cp  # noqa: PLC0415

    return cp.ElementwiseKernel(
        # Input params
        in_params=(
            "raw T coords, raw T rot_mat, raw T source_point, "
            "float64 lateral_cutoff_sq_over_sad_sq"
        ),
        # Output params
        out_params="T out_coords_x, T out_coords_y, T out_coords_z, T out_lat_x, T out_lat_z, T out_rad_dist_sq, bool out_mask",
        # Kernel operation
        operation="""
            // Matrix multiplication: coords[i] @ rot_mat (unrolled for 3x3)
            T x = coords[i * 3 + 0];
            T y = coords[i * 3 + 1];
            T z = coords[i * 3 + 2];

            T rx = x * rot_mat[0] + y * rot_mat[3] + z * rot_mat[6];
            T ry = x * rot_mat[1] + y * rot_mat[4] + z * rot_mat[7];
            T rz = x * rot_mat[2] + y * rot_mat[5] + z * rot_mat[8];

            out_coords_x = rx;
            out_coords_y = ry;
            out_coords_z = rz;

            // lat_dists
            T lx = rx + source_point[0];
            T lz = rz + source_point[2];

            out_lat_x = lx;
            out_lat_z = lz;

            // rad_dist_sq
            T r_sq = lx * lx + lz * lz;
            out_rad_dist_sq = r_sq;

            // mask: (lateral_cutoff / sad) ** 2 * ry ** 2
            T limit = lateral_cutoff_sq_over_sad_sq * ry * ry;
            out_mask = r_sq <= limit;
        """,
        name="calc_geo_dists_kernel",
    )


def _build_cupy_raw_kernel():
    if not cupy_available():
        return None
    import cupy as cp  # noqa: PLC0415

    # RawKernel version - Numba-like calling convention with pre-allocated outputs
    return cp.RawKernel(
        r"""
    extern "C" __global__
    void calc_geo_dists_raw_kernel(
        const double* coords,
        const double* rot_mat,
        const double* source_point,
        double lateral_cutoff_sq_over_sad_sq,
        double* out_coords_temp,
        double* out_lat_dists,
        double* out_rad_dist_sq,
        bool* out_mask,
        int n
    ) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i >= n) return;

        // Matrix multiplication: coords[i] @ rot_mat (unrolled for 3x3)
        double x = coords[i * 3 + 0];
        double y = coords[i * 3 + 1];
        double z = coords[i * 3 + 2];

        double rx = x * rot_mat[0] + y * rot_mat[3] + z * rot_mat[6];
        double ry = x * rot_mat[1] + y * rot_mat[4] + z * rot_mat[7];
        double rz = x * rot_mat[2] + y * rot_mat[5] + z * rot_mat[8];

        out_coords_temp[i * 3 + 0] = rx;
        out_coords_temp[i * 3 + 1] = ry;
        out_coords_temp[i * 3 + 2] = rz;

        // lat_dists
        double lx = rx + source_point[0];
        double lz = rz + source_point[2];

        out_lat_dists[i * 2 + 0] = lx;
        out_lat_dists[i * 2 + 1] = lz;

        // rad_dist_sq
        double r_sq = lx * lx + lz * lz;
        out_rad_dist_sq[i] = r_sq;

        // mask
        double limit = lateral_cutoff_sq_over_sad_sq * ry * ry;
        out_mask[i] = r_sq <= limit;
    }
    """,
        "calc_geo_dists_raw_kernel",
    )


def _build_torch_kernel():
    if not pytorch_gpu_available():
        return None
    import torch  # noqa: PLC0415

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

    return _calc_geo_dists_torch_kernel


_calc_geo_dists_cupy_kernel = _LazyKernel(_build_cupy_kernel, "CuPy")
_calc_geo_dists_cupy_raw_kernel = _LazyKernel(_build_cupy_raw_kernel, "CuPy")
_calc_geo_dists_torch_kernel = _LazyKernel(_build_torch_kernel, "PyTorch with GPU support")

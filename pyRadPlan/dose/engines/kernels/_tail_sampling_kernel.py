"""Fused kernels of the lateral tail sampling of the particle pencil beam engines.

:func:`thin_tail_candidates` runs once per ray and drops the voxels no bixel of the
ray can accept, :func:`select_tail_samples` then selects the voxels of each bixel.

The generic array-API code is wrapped by :func:`pyRadPlan.core.xp_utils.jittable`;
NumPy runs the numba implementation from :mod:`._tail_sampling_kernel_numba` when numba
is installed and enabled in ``settings.xp.jit_backends``. CuPy arrays run one fused
elementwise kernel each, since the generic code is bound by its kernel launches there.
"""

import array_api_compat

from ....core.xp_utils import cupy_available
from ....core.xp_utils.compat import interp1d
from ....core.xp_utils.jittable import jittable
from ....core.xp_utils.typing import Array
from ._calc_geo_dists_kernel import _LazyKernel

GOLDEN_RATIO_FRACTION = 0.6180339887498949
# Keeps voxels whose envelope equals the smallest probability up to rounding
_ENVELOPE_MARGIN = 0.999


@jittable(backends=())
def _thin_tail_candidates(  # noqa: PLR0913, PLR0917 - flat kernel signature
    offsets: Array,
    ix: Array,
    rad_depths: Array,
    radial_dist_sq: Array,
    grid: Array,
    core_sq: Array,
    slope: Array,
    depth_scale: Array,
    depth_shift: Array,
    query_shift: float,
    rotation: float,
    min_probability: float,
) -> tuple[Array, Array, Array]:
    xp = array_api_compat.array_namespace(rad_depths, radial_dist_sq)
    numbers = xp.take(offsets, ix, axis=0) + rotation
    numbers = xp.where(numbers >= 1.0, numbers - 1.0, numbers)
    envelope = xp.zeros_like(numbers)
    for i in range(depth_scale.shape[0]):
        depths = float(depth_scale[i]) * rad_depths + float(depth_shift[i])
        query = xp.where(depths < 0.0, xp.zeros_like(depths), depths) + query_shift
        tables = interp1d(query, grid, {"core_sq": core_sq, "slope": slope})
        excess = radial_dist_sq - tables["core_sq"]
        excess = xp.where(excess > 0.0, excess, xp.zeros_like(excess))
        probability = xp.exp(-excess * tables["slope"])
        inside = (query >= grid[0]) & (query <= grid[-1])
        probability = xp.where(inside, probability, xp.zeros_like(probability))
        envelope = xp.maximum(envelope, xp.astype(probability, envelope.dtype, copy=False))
    keep = xp.nonzero((numbers < envelope) & (envelope >= _ENVELOPE_MARGIN * min_probability))[0]
    envelope = xp.take(envelope, keep, axis=0)
    # Uniform again among the kept voxels
    numbers = xp.take(numbers, keep, axis=0) / envelope
    return keep, numbers, envelope


@jittable(backends=())
def _select_tail_samples(  # noqa: PLR0913, PLR0917 - flat kernel signature
    rad_depths: Array,
    radial_dist_sq: Array,
    numbers: Array,
    envelope: Array,
    grid: Array,
    core_sq: Array,
    slope: Array,
    fac: Array,
    query_shift: float,
    rotation: float,
    min_probability: float,
    depth_min: float,
    depth_max: float,
) -> tuple[Array, Array]:
    xp = array_api_compat.array_namespace(rad_depths, radial_dist_sq)
    query = rad_depths + query_shift
    tables = interp1d(query, grid, {"core_sq": core_sq, "slope": slope}, left=0.0, right=0.0)
    probability = xp.exp((tables["core_sq"] - radial_dist_sq) * tables["slope"])
    rotated = numbers + rotation
    rotated = xp.where(rotated >= 1.0, rotated - 1.0, rotated)
    keep = (
        (rotated * envelope < probability)
        & (probability >= min_probability)
        & (rad_depths >= depth_min)
        & (rad_depths <= depth_max)
        & (query >= grid[0])
        & (query <= grid[-1])
    )
    indices = xp.nonzero(keep)[0]
    # A voxel is never accepted more often than it passed the ray envelope
    probability = xp.minimum(
        xp.take(probability, indices, axis=0), xp.take(envelope, indices, axis=0)
    )
    factor = interp1d(xp.take(query, indices, axis=0), grid, fac)
    weight = xp.where(probability < 1.0, factor / probability, xp.ones_like(probability))
    return indices, xp.astype(weight, radial_dist_sq.dtype, copy=False)


def _build_cupy_kernel():
    if not cupy_available():
        return None
    import cupy as cp  # noqa: PLC0415

    return cp.ElementwiseKernel(
        in_params=(
            "D rad_depth, R radial_dist_sq, V number, E envelope, "
            "raw float64 grid, raw float64 core_sq, raw float64 slope, raw float64 fac, "
            "int32 num_nodes, float64 query_shift, float64 rotation, "
            "float64 min_probability, float64 depth_min, float64 depth_max"
        ),
        out_params="bool keep, R weight",
        operation="""
            keep = false;
            weight = 1;
            double depth = (double)rad_depth;
            double query = depth + query_shift;
            if (depth >= depth_min && depth <= depth_max
                    && query >= grid[0] && query <= grid[num_nodes - 1]) {
                int low = 0;
                int high = num_nodes - 1;
                while (high - low > 1) {
                    int mid = (low + high) / 2;
                    if (grid[mid] <= query) { low = mid; } else { high = mid; }
                }
                double t = (query - grid[low]) / (grid[high] - grid[low]);
                double c = core_sq[low] + t * (core_sq[high] - core_sq[low]);
                double s = slope[low] + t * (slope[high] - slope[low]);
                double p = exp((c - (double)radial_dist_sq) * s);
                double rotated = (double)number + rotation;
                if (rotated >= 1.0) { rotated -= 1.0; }
                double env = (double)envelope;
                if (rotated * env < p && p >= min_probability) {
                    keep = true;
                    if (env < p) { p = env; }
                    if (p < 1.0) {
                        weight = (fac[low] + t * (fac[high] - fac[low])) / p;
                    }
                }
            }
        """,
        name="select_tail_samples_kernel",
    )


_select_tail_samples_cupy_kernel = _LazyKernel(_build_cupy_kernel, "CuPy")


def _build_cupy_thinning_kernel():
    if not cupy_available():
        return None
    import cupy as cp  # noqa: PLC0415

    return cp.ElementwiseKernel(
        in_params=(
            "raw V offsets, I ix, D rad_depth, R radial_dist_sq, "
            "raw float64 grid, raw float64 core_sq, raw float64 slope, "
            "raw float64 depth_scale, raw float64 depth_shift, "
            "int32 num_nodes, int32 num_shifts, float64 query_shift, float64 rotation, "
            "float64 min_probability"
        ),
        out_params="bool keep, V number, V envelope",
        operation="""
            double env = 0.0;
            double r_sq = (double)radial_dist_sq;
            for (int k = 0; k < num_shifts; ++k) {
                double depth = depth_scale[k] * (double)rad_depth + depth_shift[k];
                if (depth < 0.0) { depth = 0.0; }
                double query = depth + query_shift;
                if (query < grid[0] || query > grid[num_nodes - 1]) { continue; }
                int low = 0;
                int high = num_nodes - 1;
                while (high - low > 1) {
                    int mid = (low + high) / 2;
                    if (grid[mid] <= query) { low = mid; } else { high = mid; }
                }
                double t = (query - grid[low]) / (grid[high] - grid[low]);
                double excess = r_sq - (core_sq[low] + t * (core_sq[high] - core_sq[low]));
                double p = 1.0;
                if (excess > 0.0) {
                    p = exp(-excess * (slope[low] + t * (slope[high] - slope[low])));
                }
                if (p > env) { env = p; }
            }
            double u = (double)offsets[ix] + rotation;
            if (u >= 1.0) { u -= 1.0; }
            keep = (u < env) && (env >= min_probability);
            envelope = env;
            number = keep ? u / env : 0.0;
        """,
        name="thin_tail_candidates_kernel",
    )


_thin_tail_candidates_cupy_kernel = _LazyKernel(_build_cupy_thinning_kernel, "CuPy")


def thin_tail_candidates(  # noqa: PLR0913, PLR0917 - flat kernel signature
    offsets: Array,
    ix: Array,
    rad_depths: Array,
    radial_dist_sq: Array,
    grid: Array,
    core_sq: Array,
    slope: Array,
    depth_scale: Array,
    depth_shift: Array,
    query_shift: float,
    rotation: float,
    min_probability: float,
) -> tuple[Array, Array, Array]:
    """Drop the voxels of a ray that none of its bixels can accept.

    The envelope of a voxel is the largest acceptance probability any bixel of the
    ray can have there, over all range shifts. A voxel survives with that probability.

    Parameters
    ----------
    offsets : Array
        Sampling numbers in [0, 1) of all voxels of the dose grid.
    ix, rad_depths, radial_dist_sq : Array
        Dose grid index, radiological depth and squared radial distance of the
        candidate voxels of the ray.
    grid, core_sq, slope : Array
        Depth nodes and envelope tables of the squared core radius (largest) and the
        fall-off (slowest) of all bixels of the ray (float64, at least two nodes).
    depth_scale, depth_shift : Array
        Range shifts as ``depth_scale * rad_depths + depth_shift``, clipped at zero
        (float64, one entry per shift).
    query_shift : float
        Added to the shifted radiological depths to obtain the table coordinate.
    rotation : float
        Rotation of the sampling numbers in [0, 1) for this ray.
    min_probability : float
        Voxels with a smaller envelope are dropped.

    Returns
    -------
    tuple[Array, Array, Array]
        Ascending positions of the surviving candidates, their sampling numbers
        (uniform in [0, 1) again) and their envelope, in the dtype of ``offsets``.
    """
    args = (
        offsets,
        ix,
        rad_depths,
        radial_dist_sq,
        grid,
        core_sq,
        slope,
        depth_scale,
        depth_shift,
        float(query_shift),
        float(rotation),
        float(min_probability),
    )
    if array_api_compat.is_cupy_array(rad_depths) and rad_depths.shape[0] > 0:
        xp = array_api_compat.array_namespace(rad_depths)
        keep, numbers, envelope = _thin_tail_candidates_cupy_kernel(
            *args[:9],
            grid.shape[0],
            depth_scale.shape[0],
            args[9],
            args[10],
            _ENVELOPE_MARGIN * args[11],
        )
        keep = xp.nonzero(keep)[0]
        return keep, xp.take(numbers, keep, axis=0), xp.take(envelope, keep, axis=0)
    return _thin_tail_candidates(*args)


def select_tail_samples(  # noqa: PLR0913, PLR0917 - flat kernel signature
    rad_depths: Array,
    radial_dist_sq: Array,
    numbers: Array,
    envelope: Array,
    grid: Array,
    core_sq: Array,
    slope: Array,
    fac: Array,
    query_shift: float,
    rotation: float,
    min_probability: float,
    depth_min: float,
    depth_max: float,
) -> tuple[Array, Array]:
    """Select the voxels a bixel computes under tail sampling and their weights.

    Parameters
    ----------
    rad_depths, radial_dist_sq : Array
        Radiological depth and squared radial distance of the candidate voxels.
    numbers, envelope : Array
        Sampling number in [0, 1) of each candidate and the probability with which
        it passed the envelope of the ray.
    grid, core_sq, slope, fac : Array
        Depth nodes and tables of the squared core radius, the fall-off and the tail
        factor (float64, at least two nodes).
    query_shift : float
        Added to the radiological depths to obtain the table coordinate.
    rotation : float
        Rotation of the sampling numbers in [0, 1) for this bixel.
    min_probability : float
        Acceptance probability below which the tail is cut.
    depth_min, depth_max : float
        Limits of the radiological depth the bixel covers.

    Returns
    -------
    tuple[Array, Array]
        Ascending indices of the selected candidates and their weights (1 inside the
        core), in the dtype of ``radial_dist_sq``.
    """
    args = (
        rad_depths,
        radial_dist_sq,
        numbers,
        envelope,
        grid,
        core_sq,
        slope,
        fac,
        float(query_shift),
        float(rotation),
        float(min_probability),
        float(depth_min),
        float(depth_max),
    )
    if array_api_compat.is_cupy_array(rad_depths) and rad_depths.shape[0] > 0:
        xp = array_api_compat.array_namespace(rad_depths)
        keep, weight = _select_tail_samples_cupy_kernel(*args[:8], grid.shape[0], *args[8:])
        indices = xp.nonzero(keep)[0]
        return indices, xp.take(weight, indices, axis=0)
    return _select_tail_samples(*args)


try:
    from . import _tail_sampling_kernel_numba  # noqa: F401  (registers the numba implementation)
except ImportError:
    pass

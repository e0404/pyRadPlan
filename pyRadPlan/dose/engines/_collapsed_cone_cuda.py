"""
CUDA implementation of the collapsed cone transport (port of ``matRad_collapsedConeMex``).

The kernel is compiled at runtime with CuPy. Each thread processes one transport line of a
direction; the lines of a direction write disjoint voxels, and the directions are launched
one after the other on the same stream, so no atomic operations are needed.
"""

import numpy as np

from pyRadPlan.core.xp_utils import cupy_available, import_backend

KERNEL_NAME = "ccc_transport_direction"

# The line function is plain C++ so that the same source can be compiled for the host
# (``CCC_HOST``) to check it against the CPU implementations without a GPU.
KERNEL_SOURCE = r"""
typedef long long ll;

#ifdef CCC_HOST
#include <cmath>
#define CCC_DEVICE static inline
using std::expm1;
#else
#define CCC_DEVICE __device__ __forceinline__
#endif

CCC_DEVICE ll ccc_max(ll a, ll b) { return a > b ? a : b; }
CCC_DEVICE ll ccc_min(ll a, ll b) { return a < b ? a : b; }

// Planes [first, last] with lo <= sh[p] <= hi for a monotone shift sequence
CCC_DEVICE void ccc_monotone_range(const ll* sh, ll n, ll lo, ll hi, ll* first, ll* last)
{
    ll a, b;
    if (sh[0] <= sh[n - 1]) {
        a = 0; b = n;
        while (a < b) { ll m = (a + b) / 2; if (sh[m] < lo) a = m + 1; else b = m; }
        *first = a;
        a = 0; b = n;
        while (a < b) { ll m = (a + b) / 2; if (sh[m] <= hi) a = m + 1; else b = m; }
        *last = a - 1;
    } else {
        a = 0; b = n;
        while (a < b) { ll m = (a + b) / 2; if (sh[m] > hi) a = m + 1; else b = m; }
        *first = a;
        a = 0; b = n;
        while (a < b) { ll m = (a + b) / 2; if (sh[m] >= lo) a = m + 1; else b = m; }
        *last = a - 1;
    }
}

CCC_DEVICE void ccc_line(
    ll line, const double* terma, const double* rho, double* dose,
    const ll* sh1, const ll* sh2, const ll* bounds, ll nk, ll n1, ll n2,
    ll stride_k, ll stride_1, ll stride_2,
    ll union_lo1, ll union_hi1, ll union_lo2, ll union_hi2,
    ll min_line1, ll min_line2, ll num_lines1,
    double step, int forward, int num_comps, const double* weights, const double* att)
{
    const ll i0 = min_line1 + line % num_lines1;
    const ll j0 = min_line2 + line / num_lines1;
    ll a1, b1, c1, d1, a2, b2, c2, d2;
    ccc_monotone_range(sh2, nk, -j0, n2 - 1 - j0, &a2, &b2);
    ccc_monotone_range(sh2, nk, union_lo2 - j0, union_hi2 - j0, &c2, &d2);
    ccc_monotone_range(sh1, nk, -i0, n1 - 1 - i0, &a1, &b1);
    const ll valid_first = ccc_max(a1, a2);
    const ll valid_last = ccc_min(b1, b2);
    ccc_monotone_range(sh1, nk, union_lo1 - i0, union_hi1 - i0, &c1, &d1);
    const ll cand_first = ccc_max(ccc_max(c1, c2), valid_first);
    const ll cand_last = ccc_min(ccc_min(d1, d2), valid_last);
    if (cand_first > cand_last) {
        return;
    }
    // bounds: lo1, hi1, lo2, hi2 of the TERMA in each plane, 4 x nk
    ll start = -1;
    for (ll q = 0; q <= cand_last - cand_first; ++q) {
        const ll p = forward ? cand_first + q : cand_last - q;
        const ll i = i0 + sh1[p];
        const ll j = j0 + sh2[p];
        if (i >= bounds[p] && i <= bounds[nk + p] && j >= bounds[2 * nk + p] &&
            j <= bounds[3 * nk + p]) {
            start = p;
            break;
        }
    }
    if (start < 0) {
        return;
    }
    double flow[8];
    for (int c = 0; c < num_comps; ++c) {
        flow[c] = 0.0;
    }
    const ll count = forward ? valid_last - start + 1 : start - valid_first + 1;
    const ll inc = forward ? 1 : -1;
    ll p = start;
    for (ll q = 0; q < count; ++q, p += inc) {
        const ll ix = p * stride_k + (i0 + sh1[p]) * stride_1 + (j0 + sh2[p]) * stride_2;
        const double dt = step * rho[ix];
        const double t = terma[ix];
        double d = 0.0;
        for (int c = 0; c < num_comps; ++c) {
            const double w = weights[c];
            if (w == 0.0) {
                continue;
            }
            const double x = att[c] * dt;
            double one_minus_e, g;
            if (x > 0.0) {
                one_minus_e = -expm1(-x);
                g = one_minus_e / x;
            } else {
                one_minus_e = 0.0;
                g = 1.0;
            }
            const double source = t * w;
            d += flow[c] * g + source * (1.0 - g);
            flow[c] = flow[c] * (1.0 - one_minus_e) + source * one_minus_e;
        }
        dose[ix] += d;
    }
}

#ifndef CCC_HOST
extern "C" __global__ void ccc_transport_direction(
    ll num_lines, const double* terma, const double* rho, double* dose,
    const ll* sh1, const ll* sh2, const ll* bounds, ll nk, ll n1, ll n2,
    ll stride_k, ll stride_1, ll stride_2,
    ll union_lo1, ll union_hi1, ll union_lo2, ll union_hi2,
    ll min_line1, ll min_line2, ll num_lines1,
    double step, int forward, int num_comps, const double* weights, const double* att)
{
    const ll line = (ll)blockIdx.x * blockDim.x + threadIdx.x;
    if (line >= num_lines) {
        return;
    }
    ccc_line(line, terma, rho, dose, sh1, sh2, bounds, nk, n1, n2, stride_k, stride_1,
             stride_2, union_lo1, union_hi1, union_lo2, union_hi2, min_line1, min_line2,
             num_lines1, step, forward, num_comps, weights, att);
}
#endif
"""

MAX_COMPONENTS = 8
THREADS_PER_BLOCK = 128

_kernel = None


def gpu_available() -> bool:
    """CuPy is installed and a CUDA device is present."""
    try:
        return bool(cupy_available())
    except Exception:  # noqa: BLE001 - any CuPy/CUDA error means no usable GPU
        return False


def _get_kernel():
    """Compile the CUDA kernel on first use."""
    global _kernel  # noqa: PLW0603
    if _kernel is None:
        _kernel = import_backend("cupy").RawKernel(KERNEL_SOURCE, KERNEL_NAME)
    return _kernel


def source_bounds(subs: np.ndarray, shape: tuple) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Bounding boxes of the TERMA in the planes perpendicular to each axis.

    Parameters
    ----------
    subs : np.ndarray
        ``(3, n)`` subscripts of the voxels with non-zero TERMA.
    shape : tuple
        Array shape.

    Returns
    -------
    tuple[list[np.ndarray], np.ndarray]
        Per axis k a ``(4, shape[k])`` array with lo1, hi1, lo2, hi2 (lo > hi: no TERMA in
        the plane) and the ``(3, 4)`` union of the boxes.
    """
    bounds = []
    union = np.empty((3, 4), dtype=np.int64)
    for k in range(3):
        t1, t2 = (k + 1) % 3, (k + 2) % 3
        b = np.empty((4, shape[k]), dtype=np.int64)
        b[0] = shape[t1]
        b[1] = -1
        b[2] = shape[t2]
        b[3] = -1
        np.minimum.at(b[0], subs[k], subs[t1])
        np.maximum.at(b[1], subs[k], subs[t1])
        np.minimum.at(b[2], subs[k], subs[t2])
        np.maximum.at(b[3], subs[k], subs[t2])
        bounds.append(b)
        union[k] = (subs[t1].min(), subs[t1].max(), subs[t2].min(), subs[t2].max())
    return bounds, union


def direction_setup(
    shape: tuple, voxel_size: np.ndarray, origin: np.ndarray, direction: np.ndarray, union
) -> dict:
    """
    Set up the transport lattice of one direction on the host.

    Parameters
    ----------
    shape : tuple
        Array shape.
    voxel_size : np.ndarray
        Voxel size along the array axes [mm].
    origin : np.ndarray
        Zero based index of the first voxel in the global grid.
    direction : np.ndarray
        Unit transport direction in array axes.
    union : np.ndarray
        ``(3, 4)`` union of the TERMA bounding boxes (see ``source_bounds``).

    Returns
    -------
    dict
        Dominant axis ``k``, shifts ``sh1``, ``sh2`` of the lines per plane, sizes and
        strides along the dominant and the sheared axes, the range of lines crossing the
        TERMA box, the step length ``step`` [mm] and the transport direction ``forward``.
    """
    k = int(np.argmax(np.abs(direction) / voxel_size))
    t1, t2 = (k + 1) % 3, (k + 2) % 3
    nk = shape[k]
    sigma1 = direction[t1] * voxel_size[k] / (direction[k] * voxel_size[t1])
    sigma2 = direction[t2] * voxel_size[k] / (direction[k] * voxel_size[t2])
    plane = np.arange(nk, dtype=np.float64) + origin[k]
    sh1 = np.floor(plane * sigma1 + 0.5).astype(np.int64) - origin[t1]
    sh2 = np.floor(plane * sigma2 + 0.5).astype(np.int64) - origin[t2]
    min_line1 = int(union[k, 0] - sh1.max())
    min_line2 = int(union[k, 2] - sh2.max())
    num_lines1 = int(union[k, 1] - sh1.min()) - min_line1 + 1
    num_lines2 = int(union[k, 3] - sh2.min()) - min_line2 + 1
    strides = (shape[1] * shape[2], shape[2], 1)
    return {
        "k": k,
        "sh1": sh1,
        "sh2": sh2,
        "nk": nk,
        "n1": shape[t1],
        "n2": shape[t2],
        "strides": (strides[k], strides[t1], strides[t2]),
        "min_line1": min_line1,
        "min_line2": min_line2,
        "num_lines1": num_lines1,
        "num_lines2": num_lines2,
        "step": float(voxel_size[k] / abs(direction[k])),
        "forward": int(direction[k] > 0.0),
    }


def transport(terma, rho, voxel_size, dirs, weights, att, origin):
    """
    Collapsed cone transport on the GPU, see ``_collapsed_cone.transport``.

    Parameters
    ----------
    terma, rho : cupy.ndarray
        3D arrays of TERMA and relative electron density on the GPU.
    voxel_size : np.ndarray
        Voxel size along the array axes [mm].
    dirs : np.ndarray
        ``(3, n_dirs)`` unit transport directions in array axes.
    weights, att : np.ndarray
        ``(n_components, n_dirs)`` kernel weights and attenuation coefficients.
    origin : np.ndarray
        Zero based index of the first voxel in the global grid.

    Returns
    -------
    cupy.ndarray
        Deposited dose on the GPU.
    """
    cupy = import_backend("cupy")

    terma = cupy.ascontiguousarray(terma, dtype=cupy.float64)
    rho = cupy.ascontiguousarray(rho, dtype=cupy.float64)
    shape = tuple(int(n) for n in terma.shape)
    dose = cupy.zeros_like(terma)
    num_comps = weights.shape[0]
    if num_comps > MAX_COMPONENTS:
        raise ValueError(f"At most {MAX_COMPONENTS} kernel components are supported.")

    # TERMA bounding boxes per plane, computed on the host from the source voxels
    subs = cupy.stack(cupy.nonzero(terma)).get()
    if subs.shape[1] == 0:
        return dose
    bounds, union = source_bounds(subs, shape)
    bounds_dev = [cupy.asarray(b.ravel()) for b in bounds]
    weights_dev = cupy.asarray(np.ascontiguousarray(weights.T, dtype=np.float64))
    att_dev = cupy.asarray(np.ascontiguousarray(att.T, dtype=np.float64))

    # One launch per direction with one thread per line; the launches run in order on the
    # same stream, so the dose of a direction is complete before the next one starts
    kernel = _get_kernel()
    i64, f64, i32 = np.int64, np.float64, np.int32
    for d in range(dirs.shape[1]):
        s = direction_setup(shape, voxel_size, origin, dirs[:, d], union)
        num_lines = s["num_lines1"] * s["num_lines2"]
        if num_lines <= 0:
            continue
        k = s["k"]
        blocks = (num_lines + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        kernel(
            (blocks,),
            (THREADS_PER_BLOCK,),
            (
                i64(num_lines),
                terma,
                rho,
                dose,
                cupy.asarray(s["sh1"]),
                cupy.asarray(s["sh2"]),
                bounds_dev[k],
                i64(s["nk"]),
                i64(s["n1"]),
                i64(s["n2"]),
                i64(s["strides"][0]),
                i64(s["strides"][1]),
                i64(s["strides"][2]),
                i64(union[k, 0]),
                i64(union[k, 1]),
                i64(union[k, 2]),
                i64(union[k, 3]),
                i64(s["min_line1"]),
                i64(s["min_line2"]),
                i64(s["num_lines1"]),
                f64(s["step"]),
                i32(s["forward"]),
                i32(num_comps),
                weights_dev[d],
                att_dev[d],
            ),
        )
    return dose

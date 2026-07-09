"""Provide Array API compliant functions and utilities (e.g. from numpy)."""

import logging
from typing import Literal, Union, Sequence, Any

from .typing import Array
from functools import lru_cache, partial
from . import to_numpy
import importlib.util
import array_api_compat
import numpy as np

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    jax = None
    jnp = None

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None

if torch is not None:

    @torch.jit.script
    def _interp1d_torch_kernel(xq, x, y):
        # xq: (N_out,)
        # x:  (N_in,)
        # y:  (N_in,) or (C, N_in)
        xq = torch.clamp(xq, min=x[0], max=x[-1])
        idx = torch.searchsorted(x, xq, right=False) - 1
        idx = torch.clamp(idx, min=0, max=x.shape[0] - 2)
        i0 = idx
        i1 = idx + 1
        x0 = x[i0]
        t = (xq - x0) / (x[i1] - x0)
        if y.ndim == 1:
            y0 = y[i0]
            y1 = y[i1]
            return y0 + (y1 - y0) * t
        else:
            y0 = y[:, i0]
            y1 = y[:, i1]
            return y0 + (y1 - y0) * t.unsqueeze(0)

else:
    _interp1d_torch_kernel = None

try:
    _HAS_TRITON = importlib.util.find_spec("triton") is not None
except (ImportError, ValueError):
    _HAS_TRITON = False

try:
    import scipy.interpolate
except ImportError:
    scipy = None

try:
    import cupy
    import cupyx
    import cupyx.scipy.interpolate
except ImportError:
    cupy = None
    cupyx = None

try:
    import torch
except ImportError:
    torch = None
try:
    from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
except ModuleNotFoundError:
    CupyRegularGridInterpolator = None

from scipy.interpolate import RegularGridInterpolator as NumpyRegularGridInterpolator


# %%
def quantile(
    x: Array,
    p: float,
    *,
    axis: int = -1,
    method: Literal["nearest", "linear"] = "nearest",
    is_sorted: bool = False,
) -> Array:
    """
    Array API compliant quantile function.

    Computes the q-th quantile of the data along the specified axis. Reproduces numpy.quantile
    behavior for method="nearest" and method="linear", but will always return an Array in the
    respective namespace, even if the result is a scalar.

    Parameters
    ----------
    x : Array
        Input array.
    p : float
        Quantile to compute (between 0 and 1).
    method : Literal["nearest","linear"], optional
        Method to use for interpolation. Default is "nearest".
    axis : int, optional
        Axis along which to compute the quantile. Default is -1 (last axis).
    is_sorted : bool, optional
        Whether the input array is already sorted along the specified axis. Default is False.
        This will skip the sorting step if True. If the input is not sorted and this is set to
        True, the result will be incorrect.

    Returns
    -------
    Array
        The computed quantile(s) along the specified axis. Always returns an Array, even for scalar
        results.
    """

    xp = array_api_compat.array_namespace(x)
    device = array_api_compat.device(x)

    if not is_sorted:
        x_sorted = xp.sort(x, axis=axis, stable=False)
    else:
        x_sorted = x

    float_ix = p * (x.shape[axis] - 1)

    if method == "nearest" or float_ix == round(float_ix):
        k = int(round(float_ix))
        return xp.take(x_sorted, xp.reshape(xp.asarray(k, device=device), shape=(1,)), axis=axis)

    # Note that the case float_ix == round(float_ix) is already handled above, so we can assume
    # that our float_ix lies between two indices
    if method == "linear":
        k_lower = int(float_ix)
        k_upper = k_lower + 1

        assert k_lower != k_upper  # This should be captured above

        # We try to do as much in place on x_upper to save memory allocations
        x_upper = xp.take(
            x_sorted, xp.reshape(xp.asarray(k_upper, device=device), shape=(1,)), axis=axis
        )
        weight_upper = float_ix - k_lower

        x_upper *= weight_upper

        x_upper += (1.0 - weight_upper) * xp.take(
            x_sorted, xp.reshape(xp.asarray(k_lower, device=device), shape=(1,)), axis=axis
        )

        # Squeeze might not work correctly with pytorch?
        return xp.squeeze(x_upper, axis=axis)

    # interp_core.py


# %%
def _interp1d_core(xp, xq: Array, x: Array, y: Array) -> Array:
    """Perform generic core implementation of 1d interpolation."""

    xq_clip = xp.clip(xq, x[0], x[-1])

    # find interval indices and handle x coordinates
    idx = xp.searchsorted(x, xq_clip, side="left") - 1

    i0 = xp.clip(idx, 0, x.shape[0] - 2)
    i1 = i0 + 1

    x0 = x[i0]
    x1 = x[i1]
    t = (xq_clip - x0) / (x1 - x0)

    y0 = xp.take(y, i0, axis=-1)
    y1 = xp.take(y, i1, axis=-1)

    dy = y1 - y0
    if dy.ndim > 1:
        t = xp.expand_dims(t, axis=0)

    return y0 + dy * t


@lru_cache(maxsize=1)
def _get_jax_interp1d(xp):
    """Get JAX implementation of 1D interpolation, compiled with jax.jit."""
    return jax.jit(lambda xq, x, y: _interp1d_core(xp, xq, x, y))


@lru_cache(maxsize=1)
def _get_torch_interp1d(xp):
    """Get PyTorch implementation of 1D interpolation.

    Uses ``torch.compile`` when Triton is available,
    otherwise falls back to a ``@torch.jit.script`` kernel. Both vectorize 1D
    and stacked-2D y.
    """
    if _HAS_TRITON:

        def fn(xq, x, y):
            return _interp1d_core(xp, xq, x, y)

        return torch.compile(fn)

    logger.info("Triton not available; using @torch.jit.script kernel for interp1d.")
    return _interp1d_torch_kernel


def interp1d(
    xq: Array, x: Array, y: Union[Array, Sequence[Array], dict[Any, Array]], *, stack: bool = False
) -> Union[Array, Sequence[Array], dict[Any, Array]]:
    """
    Array API Conform 1D interpolation.

    Will perform 1D array interpolation with array API conformal arrays.
    If a dedicated implementation is found, it will be used (if implemented).

    Parameters
    ----------
    xq : Array
        Interpolation coordinates as 1D Array
    x : Array
        Coordinates of the array(s) to interpolate as 1D Array
    y : Union[Array, Sequence[Array]]
        Array(s) to interpolate. Can be a 1D array matching the size of x/xq.
        Can be a 2D array of shape (N, x.size) if to interpolate n arrays.
        Can also be a sequence of arrays, which will either be stacked or
        looped, depending on the value of stack.
    stack: bool, optional
        Whether to stack multiple input arrays if a sequence is provided.
        Will propagate to the output

    Returns
    -------
    Array
        Interpolated values at the specified xq-coordinates. If a Sequence
        was passed, a sequence will be returned if stack was False.
        In all other cases, an Array will be returned.

    Note
    ----
        Feel free to improve this, as interpolation is performance sensitive.
        Improvements can defer to better implementations, or have additional
        custom implementations (like using CUDA texture memory, for exaxmple)
    """

    xp = array_api_compat.array_namespace(x, xq)

    # Fast path, if xp.interp is available (e.g. np, jnp, cupy)
    if hasattr(xp, "interp"):
        if not isinstance(y, (list, tuple, dict)) and y.ndim == 1:
            return xp.interp(xq, x, y)
        if isinstance(y, (list, tuple)) and all(v.ndim == 1 for v in y):
            res = [xp.interp(xq, x, v) for v in y]
            if stack:
                return xp.stack(res, axis=0)
            return tuple(res) if isinstance(y, tuple) else res
        if isinstance(y, dict) and all(v.ndim == 1 for v in y.values()):
            res = {k: xp.interp(xq, x, v) for k, v in y.items()}
            if stack:
                return xp.stack(list(res.values()), axis=0)
            return res

    # Numpy only 2D fast path
    if array_api_compat.is_numpy_namespace(xp):
        if not isinstance(y, (list, tuple, dict)) and y.ndim == 2:
            return np.apply_along_axis(lambda ytmp: np.interp(xq, x, ytmp), axis=-1, arr=y)

    # Stack if requested and no fast path matched
    if isinstance(y, (list, tuple)) and stack:
        y = xp.stack(y, axis=0)

    if isinstance(y, dict) and stack:
        y = xp.stack([*y.values()], axis=0)

    # Backend implementation of fallback interpolation
    if array_api_compat.is_jax_namespace(xp):
        _interpolation = _get_jax_interp1d(xp)

    elif array_api_compat.is_torch_namespace(xp):
        _interpolation = _get_torch_interp1d(xp)

    else:
        _interpolation = partial(_interp1d_core, xp)

    # Interpolation
    if isinstance(y, (list, tuple)):
        res = [_interpolation(xq, x, y_arr) for y_arr in y]
        return tuple(res) if isinstance(y, tuple) else res
    elif isinstance(y, dict):
        return {key: _interpolation(xq, x, ytmp) for key, ytmp in y.items()}

    return _interpolation(xq, x, y)


# %% N-D Interpolation
def _flip_axis(values: Array, axis: int) -> Array:
    """Flip an array along one axis using basic slicing."""
    sl = [slice(None)] * values.ndim
    sl[axis] = slice(None, None, -1)
    return values[tuple(sl)]


def _prepare_rectilinear_grid(
    xp, xq: Array, x: tuple[Array, ...], y: Array
) -> tuple[Array, tuple[Array, ...], Array]:
    """Ensure ascending grid axes and clip query points to grid bounds."""
    grids = list(x)

    for axis, g in enumerate(grids):
        if g[0] > g[-1]:
            grids[axis] = g[::-1]
            y = _flip_axis(y, axis)

    xq = xp.stack(
        [xp.clip(xq[:, axis], g[0], g[-1]) for axis, g in enumerate(grids)],
        axis=1,
    )

    return xq, tuple(grids), y


def _interp2d_core(xp, xq: Array, gx: Array, gy: Array, y: Array) -> Array:
    """Perform generic core implementation of 2D interpolation."""
    qx = xq[:, 0]
    qy = xq[:, 1]

    nx = gx.shape[0]
    ny = gy.shape[0]

    ix = xp.clip(xp.searchsorted(gx, qx, side="left") - 1, 0, nx - 2)
    iy = xp.clip(xp.searchsorted(gy, qy, side="left") - 1, 0, ny - 2)

    gx0 = gx[ix]
    gy0 = gy[iy]

    tx = (qx - gx0) / (gx[ix + 1] - gx0)
    ty = (qy - gy0) / (gy[iy + 1] - gy0)

    y_flat = xp.reshape(y, (nx * ny,))

    stride_x = ny
    base = ix * stride_x + iy

    i00 = base
    i01 = base + 1
    i10 = base + stride_x
    i11 = base + stride_x + 1

    c00 = xp.take(y_flat, i00, axis=0)
    c01 = xp.take(y_flat, i01, axis=0)
    c10 = xp.take(y_flat, i10, axis=0)
    c11 = xp.take(y_flat, i11, axis=0)

    ux = 1.0 - tx
    uy = 1.0 - ty

    return ux * uy * c00 + ux * ty * c01 + tx * uy * c10 + tx * ty * c11


def _interp3d_core(xp, xq: Array, gx: Array, gy: Array, gz: Array, y: Array) -> Array:  # noqa: PLR0913
    """Perform generic core implementation of 3d interpolation.

    Basic trilinear interpolation implementation, which can be used for any array API conform namespace.
    """
    qx = xq[:, 0]
    qy = xq[:, 1]
    qz = xq[:, 2]

    nx = gx.shape[0]
    ny = gy.shape[0]
    nz = gz.shape[0]

    # Find enclosing cell indices
    ix = xp.clip(xp.searchsorted(gx, qx, side="left") - 1, 0, nx - 2)
    iy = xp.clip(xp.searchsorted(gy, qy, side="left") - 1, 0, ny - 2)
    iz = xp.clip(xp.searchsorted(gz, qz, side="left") - 1, 0, nz - 2)

    # Local coordinates within cell
    gx0 = gx[ix]
    gy0 = gy[iy]
    gz0 = gz[iz]

    tx = (qx - gx0) / (gx[ix + 1] - gx0)
    ty = (qy - gy0) / (gy[iy + 1] - gy0)
    tz = (qz - gz0) / (gz[iz + 1] - gz0)

    # Flatten y for fast linear indexing
    y_flat = xp.reshape(y, (nx * ny * nz,))

    stride_x = ny * nz
    stride_y = nz

    base = ix * stride_x + iy * stride_y + iz

    i000 = base
    i001 = base + 1
    i010 = base + stride_y
    i011 = base + stride_y + 1
    i100 = base + stride_x
    i101 = base + stride_x + 1
    i110 = base + stride_x + stride_y
    i111 = base + stride_x + stride_y + 1

    c000 = xp.take(y_flat, i000, axis=0)
    c001 = xp.take(y_flat, i001, axis=0)
    c010 = xp.take(y_flat, i010, axis=0)
    c011 = xp.take(y_flat, i011, axis=0)
    c100 = xp.take(y_flat, i100, axis=0)
    c101 = xp.take(y_flat, i101, axis=0)
    c110 = xp.take(y_flat, i110, axis=0)
    c111 = xp.take(y_flat, i111, axis=0)

    # Trilinear weights
    ux = 1.0 - tx
    uy = 1.0 - ty
    uz = 1.0 - tz

    return (
        ux * uy * uz * c000
        + ux * uy * tz * c001
        + ux * ty * uz * c010
        + ux * ty * tz * c011
        + tx * uy * uz * c100
        + tx * uy * tz * c101
        + tx * ty * uz * c110
        + tx * ty * tz * c111
    )


@lru_cache(maxsize=1)
def _get_jax_interpnd():
    """Get JAX implementation of rectilinear N-D interpolation, compiled with jax.jit."""
    return jax.jit(lambda xq, x, y: jax.scipy.interpolate.RegularGridInterpolator(x, y)(xq))


def interpnd(xq: Array, x: tuple[Array, ...], y: Array) -> Array:
    """Array API conform N-D interpolation on rectilinear grids.

    Will perform interpolation on a rectilinear grid using array API conformal arrays.
    Dedicated backend implementations are used if available. The generic fallback
    implementation currently supports only 2D and 3D interpolation.

    Parameters
    ----------
    xq : Array
        Interpolation coordinates as an array of shape (N, ndim), where N is the
        number of query points and ndim == len(x).
    x : tuple[Array, ...]
        Tuple of 1D coordinate arrays defining the rectilinear grid, e.g.
        (gx, gy) for 2D or (gx, gy, gz) for 3D.
    y : Array
        Grid values with shape matching the grid axes, i.e.
        (len(x[0]), len(x[1]), ...) .
        Query points outside the grid bounds are clipped to the valid range.

    Returns
    -------
    Array
        Interpolated values at the specified query coordinates.

    Notes
    -----
    Dedicated implementations based on ``RegularGridInterpolator`` are used for
    NumPy/SciPy, JAX, and CuPy when available.

    The generic fallback currently supports only 2D and 3D interpolation.
    """

    xp = array_api_compat.array_namespace(*x, xq, y)

    xq, x, y = _prepare_rectilinear_grid(xp, xq, x, y)

    if array_api_compat.is_numpy_namespace(xp) and scipy is not None:
        return scipy.interpolate.RegularGridInterpolator(x, y)(xq)

    if array_api_compat.is_jax_namespace(xp) and jax is not None:
        return _get_jax_interpnd()(xq, x, y)

    if array_api_compat.is_cupy_namespace(xp) and cupy is not None:
        return cupyx.scipy.interpolate.RegularGridInterpolator(x, y)(xq)

    # In the future,we may implement gpu kernel fused torch implementation, but for now we can just use the generic fallback for CPU tensors. Note, that torch does not have
    # a direct equivalent to scipy's RegularGridInterpolator.

    if len(x) == 2:
        return _interp2d_core(xp, xq, *x, y)
    elif len(x) == 3:
        return _interp3d_core(xp, xq, *x, y)
    else:
        raise NotImplementedError(
            "Only 2D and 3D interpolation is currently implemented for the generic fallback. Note, that torch does use the generic fallback."
        )


if torch is not None:

    class TorchRegularGridInterpolator:
        """
        Torch-based RegularGridInterpolator implementation.
        supports batched interpolation
        """

        def __init__(
            self,
            points,
            values,
            method="linear",
            bounds_error=True,
            fill_value=float("nan"),
        ):
            self.points = points
            # grid_sample requires float input (float32 or float64)
            if not values.is_floating_point():
                values = values.to(dtype=torch.float32)
            self.values = values
            self.method = method
            self.bounds_error = bounds_error
            self.fill_value = fill_value

            self.dims = [len(p) for p in points]

            # Check for batch dimension
            if list(values.shape) == self.dims:
                # Single volume (D, H, W)
                self.vol = values[None, None, ...]  # (1, 1, D, H, W)
                self.is_batched = False
            elif list(values.shape)[1:] == self.dims:
                # Batched volume (B, D, H, W)
                self.vol = values.unsqueeze(0)  # (1, B, D, H, W) -> channels=B
                self.is_batched = True
            else:
                raise ValueError(
                    f"Incompatible shapes: values {values.shape} vs points {self.dims}"
                )

            # Store limits
            self.limits = []
            for p in points:
                # Store limits as tensors/arrays to avoid host-device synchronization
                self.limits.append((p[0], p[-1]))

        def __call__(self, xi):
            # some preprocessing is needed to use grid_sample
            if self.values.device != xi.device or self.values.dtype != xi.dtype:
                xi = xi.to(device=self.values.device, dtype=self.values.dtype)

            N = xi.shape[0]
            grid_norm = torch.zeros_like(xi)
            mapping = [(2, 0), (1, 1), (0, 2)]

            for src_i, dst_i in mapping:
                vmin, vmax = self.limits[src_i]
                grid_norm[:, dst_i] = 2 * (xi[:, src_i] - vmin) / (vmax - vmin) - 1

            grid = grid_norm.view(1, 1, 1, N, 3)

            mode = "bilinear" if self.method == "linear" else "nearest"

            # Handle align_corners=True to match scipy behavior with points definition
            res = torch.nn.functional.grid_sample(
                self.vol, grid, mode=mode, padding_mode="zeros", align_corners=True
            )

            # res: (1, C, 1, 1, N)
            res = res.view(self.vol.shape[1], N)  # (C, N)

            if not self.is_batched:
                res = res.squeeze(0)  # (N,)

            # Bounds check
            if self.bounds_error or self.fill_value != 0.0:
                # Check normalized coords
                oob_mask = (grid_norm < -1) | (grid_norm > 1)
                oob_mask = oob_mask.any(dim=-1)  # (N,)

                if self.bounds_error and oob_mask.any():
                    raise ValueError("Out of bounds")

                if self.fill_value != 0.0:
                    if self.is_batched:
                        res[:, oob_mask] = self.fill_value
                    else:
                        res[oob_mask] = self.fill_value

            return res


class _BatchedInterpolator:
    """Wraps B independent interpolators so non-batched backends can handle (B, D, H, W) values."""

    def __init__(self, interpolators, xp):
        self._interps = interpolators
        self._xp = xp

    def __call__(self, xi):
        results = [interp(xi) for interp in self._interps]
        return self._xp.stack(results, axis=0)


def batched_interp3d(
    points, values, method="linear", bounds_error=True, fill_value=np.nan
) -> Union[Array, Sequence[Array], dict[Any, Array]]:
    """
    Array API conform batched 3D interpolation.

    """
    # points is a tuple/list of grid vectors; expand it so array_namespace
    # sees actual array objects instead of the container itself.
    xp = array_api_compat.array_namespace(values, *points)

    is_batched = values.ndim == len(points) + 1

    if array_api_compat.is_numpy_namespace(xp):
        if is_batched:
            interps = [
                NumpyRegularGridInterpolator(
                    points,
                    values[b],
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=float(fill_value),
                )
                for b in range(values.shape[0])
            ]
            return _BatchedInterpolator(interps, xp)
        return NumpyRegularGridInterpolator(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=float(fill_value),
        )

    if array_api_compat.is_array_api_strict_namespace(xp):
        # Treat strict Array API tensors by converting to numpy underneath
        points_np = tuple(to_numpy(p) for p in points)
        values_np = to_numpy(values)
        if is_batched:
            interps = [
                NumpyRegularGridInterpolator(
                    points_np,
                    values_np[b],
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=float(fill_value),
                )
                for b in range(values_np.shape[0])
            ]
            return _BatchedInterpolator(interps, np)
        return NumpyRegularGridInterpolator(
            points_np,
            values_np,
            method=method,
            bounds_error=bounds_error,
            fill_value=float(fill_value),
        )

    if array_api_compat.is_cupy_namespace(xp):
        if is_batched:
            interps = [
                CupyRegularGridInterpolator(
                    points,
                    values[b],
                    method=method,
                    bounds_error=bounds_error,
                    fill_value=float(fill_value),
                )
                for b in range(values.shape[0])
            ]
            return _BatchedInterpolator(interps, xp)
        return CupyRegularGridInterpolator(
            points, values, method=method, bounds_error=bounds_error, fill_value=float(fill_value)
        )

    if array_api_compat.is_torch_namespace(xp):
        return TorchRegularGridInterpolator(
            points,
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=float(fill_value),
        )

    raise NotImplementedError("3D interpolation is not implemented for this backend.")


def array_meshgrid(
    *arrays: Array, indexing: Literal["xy", "ij"] = "xy"
) -> Union[Array, Sequence[Array], dict[Any, Array]]:
    """
    Meshgrid indexing isn't the same for cupy/numpy and torch when using Array API, use this wrapper as workaround

    """
    xp = array_api_compat.array_namespace(*arrays)

    if array_api_compat.is_torch_namespace(xp):
        output = torch.meshgrid(*arrays, indexing=indexing)

    elif array_api_compat.is_cupy_namespace(xp):
        output = cp.meshgrid(*arrays, indexing=indexing)

    elif array_api_compat.is_numpy_namespace(xp):
        output = np.meshgrid(*arrays, indexing=indexing)

    elif array_api_compat.is_array_api_strict_namespace(xp):
        output = xp.meshgrid(*arrays, indexing=indexing)

    else:
        raise NotImplementedError("array_meshgrid not supported for this Backend")

    return output

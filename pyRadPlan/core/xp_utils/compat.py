"""Provide Array API compliant functions and utilities (e.g. from numpy)."""

from typing import Literal, Union, Sequence, Any

from .typing import Array
from functools import lru_cache, partial

import array_api_compat
import numpy as np

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

if torch is not None:

    @torch.jit.script
    def _interp1d_torch_kernel(xq, x, y, left, right):
        # xq: (N_out,)
        # x: (N_in,)
        # y: (N_in,) or (C, N_in)
        # left, right: (C,) or (1,) or scalar
        below = xq < x[0]
        above = xq > x[-1]
        xq = torch.clamp(xq, min=x[0], max=x[-1])
        idx = torch.searchsorted(x, xq, right=False) - 1
        idx = torch.clamp(idx, min=0, max=x.shape[0] - 2)
        i0 = idx
        i1 = idx + 1
        x0 = x[i0]
        x1 = x[i1]
        t = (xq - x0) / (x1 - x0)
        if y.ndim == 1:
            y0 = y[i0]
            y1 = y[i1]
            result = y0 + (y1 - y0) * t
        else:
            y0 = y[:, i0]
            y1 = y[:, i1]
            result = y0 + (y1 - y0) * t.unsqueeze(0)

        # Handle out-of-bounds values
        if result.ndim > xq.ndim:
            below = below.unsqueeze(0)
            above = above.unsqueeze(0)
        while left.ndim < result.ndim:
            left = left.unsqueeze(-1)
        while right.ndim < result.ndim:
            right = right.unsqueeze(-1)
        result = torch.where(below, left, result)
        result = torch.where(above, right, result)
        return result

else:
    _interp1d_torch_kernel = None

if jax is not None:

    @jax.jit
    def _interp1d_jax_kernel(xq, x, y, left, right):
        below = xq < x[0]
        above = xq > x[-1]

        xq = jnp.clip(xq, x[0], x[-1])
        idx = jnp.searchsorted(x, xq, side="left") - 1
        idx = jnp.clip(idx, 0, x.shape[0] - 2)

        i0 = idx
        i1 = i0 + 1

        x0 = x[i0]
        x1 = x[i1]
        t = (xq - x0) / (x1 - x0)

        y0 = jnp.take(y, i0, axis=-1)
        y1 = jnp.take(y, i1, axis=-1)

        if y.ndim == 1:
            result = y0 + (y1 - y0) * t
        else:
            result = y0 + (y1 - y0) * jnp.expand_dims(t, axis=0)

        if result.ndim > below.ndim:
            below = jnp.expand_dims(below, axis=0)
            above = jnp.expand_dims(above, axis=0)

        while left.ndim < result.ndim:
            left = jnp.expand_dims(left, axis=-1)

        while right.ndim < result.ndim:
            right = jnp.expand_dims(right, axis=-1)

        result = jnp.where(below, left, result)
        result = jnp.where(above, right, result)

        return result


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
def _interp1d_core(
    xp,
    xq: Array,
    x: Array,
    y: Array,
    left: Union[float, Array, None] = None,
    right: Union[float, Array, None] = None,
) -> Array:
    """Perform generic core implementation of 1d interpolation."""
    # Flatten N-D query points; the standard only guarantees 1-D indexing/take.
    xq_shape = xq.shape
    if xq.ndim > 1:
        xq = xp.reshape(xq, (-1,))

    below = xq < x[0]
    above = xq > x[-1]
    xq = xp.clip(xq, x[0], x[-1])
    idx = xp.searchsorted(x, xq, side="left") - 1
    idx = xp.clip(idx, 0, x.shape[0] - 2)
    i0 = idx
    i1 = i0 + 1
    x0 = x[i0]
    x1 = x[i1]
    t = (xq - x0) / (x1 - x0)
    y0 = xp.take(y, i0, axis=-1)
    y1 = xp.take(y, i1, axis=-1)

    dy = y1 - y0
    if dy.ndim > 1:
        t = xp.expand_dims(t, axis=0)

    result = y0 + dy * t

    device = array_api_compat.device(y)
    left = y[..., 0] if left is None else xp.asarray(left, dtype=y.dtype, device=device)
    right = y[..., -1] if right is None else xp.asarray(right, dtype=y.dtype, device=device)

    while below.ndim < result.ndim:
        below = xp.expand_dims(below, axis=0)
        above = xp.expand_dims(above, axis=0)
    while left.ndim < result.ndim:
        left = xp.expand_dims(left, axis=-1)
    while right.ndim < result.ndim:
        right = xp.expand_dims(right, axis=-1)

    result = xp.where(below, left, result)
    result = xp.where(above, right, result)

    if len(xq_shape) > 1:
        result = xp.reshape(result, y.shape[:-1] + xq_shape)

    return result


@lru_cache(maxsize=1)
def _get_jax_interp1d(xp):
    """Get JAX implementation of 1D interpolation, compiled with jax.jit."""

    def fn(xq, x, y, left, right):
        left = jnp.take(y, 0, axis=-1) if left is None else jnp.asarray(left, dtype=y.dtype)
        right = jnp.take(y, -1, axis=-1) if right is None else jnp.asarray(right, dtype=y.dtype)

        return _interp1d_jax_kernel(xq, x, y, left, right)

    return fn


@lru_cache(maxsize=1)
def _get_torch_interp1d(xp):
    """Get PyTorch implementation of 1D interpolation."""

    def fn(xq, x, y, left, right):
        left = y[..., 0] if left is None else torch.as_tensor(left, dtype=y.dtype, device=y.device)
        right = (
            y[..., -1] if right is None else torch.as_tensor(right, dtype=y.dtype, device=y.device)
        )
        return _interp1d_torch_kernel(xq, x, y, left, right)

    return fn


def interp1d(
    xq: Array,
    x: Array,
    y: Union[Array, Sequence[Array], dict[Any, Array]],
    *,
    stack: bool = False,
    left: Union[float, Array, None] = None,
    right: Union[float, Array, None] = None,
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
    left : float or Array, optional
        Value to return for xq values below x[0]. If None, uses y[:, 0].
        For 2D y, an Array must contain exactly one value per row.
        Lists, tuples, and dictionaries of y arrays only support scalar boundary values.
    right : float or Array, optional
        Value to return for xq values above x[-1]. If None, uses y[:, -1].
        For 2D y, an Array must contain exactly one value per row.
        Lists, tuples, and dictionaries of y arrays only support scalar boundary values.

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

    y_is_container = isinstance(y, (list, tuple, dict))
    boundary_arrays = [
        boundary for boundary in (left, right) if array_api_compat.is_array_api_obj(boundary)
    ]
    row_boundaries = [boundary for boundary in boundary_arrays if boundary.ndim > 0]

    if row_boundaries and (y_is_container or y.ndim != 2):
        raise ValueError("Non-scalar left and right arrays require y to be a single 2D array.")

    namespace_inputs = [x, xq]
    if isinstance(y, dict):
        namespace_inputs.extend(y.values())
    elif isinstance(y, (list, tuple)):
        namespace_inputs.extend(y)
    else:
        namespace_inputs.append(y)

    namespace_inputs.extend(boundary_arrays)
    xp = array_api_compat.array_namespace(*namespace_inputs)

    # Fast path, if xp.interp is available (e.g. np, jnp, cupy)
    if hasattr(xp, "interp"):
        if not isinstance(y, (list, tuple, dict)) and y.ndim == 1:
            return xp.interp(xq, x, y, left=left, right=right)
        if isinstance(y, (list, tuple)) and all(v.ndim == 1 for v in y):
            res = [xp.interp(xq, x, v, left=left, right=right) for v in y]
            if stack:
                return xp.stack(res, axis=0)
            return tuple(res) if isinstance(y, tuple) else res
        if isinstance(y, dict) and all(v.ndim == 1 for v in y.values()):
            res = {k: xp.interp(xq, x, v, left=left, right=right) for k, v in y.items()}
            if stack:
                return xp.stack(list(res.values()), axis=0)
            return res

    # Numpy only 2D fast path
    if (
        array_api_compat.is_numpy_namespace(xp)
        and not isinstance(y, (list, tuple, dict))
        and y.ndim == 2
    ):
        left_is_array = left is not None and np.ndim(left) > 0
        right_is_array = right is not None and np.ndim(right) > 0

        if left_is_array or right_is_array:
            if left_is_array and np.shape(left) != (y.shape[0],):
                raise ValueError("left must contain exactly one value per row of y.")
            if right_is_array and np.shape(right) != (y.shape[0],):
                raise ValueError("right must contain exactly one value per row of y.")

            return np.stack(
                [
                    np.interp(
                        xq,
                        x,
                        row,
                        left=left[i] if left_is_array else left,
                        right=right[i] if right_is_array else right,
                    )
                    for i, row in enumerate(y)
                ],
                axis=0,
            )

        return np.apply_along_axis(
            lambda ytmp: np.interp(xq, x, ytmp, left=left, right=right), axis=-1, arr=y
        )

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
        res = [_interpolation(xq, x, y_arr, left, right) for y_arr in y]
        return tuple(res) if isinstance(y, tuple) else res
    elif isinstance(y, dict):
        return {key: _interpolation(xq, x, ytmp, left, right) for key, ytmp in y.items()}

    return _interpolation(xq, x, y, left, right)


# %% N-D Interpolation
def _flip_axis(values: Array, axis: int) -> Array:
    """Flip an array along one axis using basic slicing."""
    sl = [slice(None)] * values.ndim
    sl[axis] = slice(None, None, -1)
    return values[tuple(sl)]


def _prepare_rectilinear_grid(
    xp, xq: Array, x: tuple[Array, ...], y: Array, bounds_error: bool = False
) -> tuple[Array, tuple[Array, ...], Array]:
    """Ensure ascending grid axes and clip query points to grid bounds."""
    grids = list(x)

    for axis, g in enumerate(grids):
        if g[0] > g[-1]:
            grids[axis] = g[::-1]
            y = _flip_axis(y, axis)

    if bounds_error:
        for axis, g in enumerate(grids):
            col = xq[:, axis]
            if bool(xp.any(col < g[0])) or bool(xp.any(col > g[-1])):
                raise ValueError(
                    f"Query points along axis {axis} lie outside the grid bounds "
                    f"[{float(g[0])}, {float(g[-1])}]."
                )

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


def interpnd(xq: Array, x: tuple[Array, ...], y: Array, *, bounds_error: bool = False) -> Array:
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
    bounds_error : bool, optional
        If False (default), query points outside the grid bounds are clipped to
        the valid range. If True, a ValueError is raised instead, which is useful
        when an out-of-grid query indicates a bug in the caller rather than an
        expected extrapolation.

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

    xq, x, y = _prepare_rectilinear_grid(xp, xq, x, y, bounds_error=bounds_error)

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


# %% Meshgrid
def array_meshgrid(*arrays: Array, indexing: Literal["xy", "ij"] = "xy") -> Sequence[Array]:
    """Array API compatible meshgrid.

    Thin wrapper around the namespace's ``meshgrid``. PyTorch is dispatched directly
    because ``array_api_compat.torch.meshgrid`` does not honor ``indexing="ij"``.

    Parameters
    ----------
    *arrays : Array
        One-dimensional coordinate arrays used to construct the meshgrid.
    indexing : {"xy", "ij"}, default="xy"
        Cartesian ("xy") or matrix ("ij") indexing convention.

    Returns
    -------
    Sequence[Array]
        Coordinate arrays forming the meshgrid.
    """
    xp = array_api_compat.array_namespace(*arrays)
    if array_api_compat.is_torch_namespace(xp):
        return torch.meshgrid(*arrays, indexing=indexing)
    return xp.meshgrid(*arrays, indexing=indexing)


# %% Fast-Fourier-Transformation
def _as_complex(xp, x: Array) -> Array:
    """Cast to the matching complex dtype, as the fft extension requires complex input."""
    if x.dtype in (xp.complex64, xp.complex128):
        return x
    return xp.astype(x, xp.complex64 if x.dtype == xp.float32 else xp.complex128)


def _fft2(x: Array, s: tuple[int, int]) -> Array:
    """Backend-specific 2D fast Fourier transform.

    Computes the two-dimensional discrete Fourier transform using the FFT
    implementation of the input array backend.

    Parameters
    ----------
    x : Array
        Input array to transform.
    s : tuple[int, int]
        Shape of the transformed axes.

    Returns
    -------
    Array
        Complex-valued 2D Fourier transform of the input array.
    """
    xp = array_api_compat.array_namespace(x)

    if array_api_compat.is_cupy_namespace(xp):
        return cupyx.scipy.fft.fft2(x, s=s)
    if array_api_compat.is_torch_namespace(xp):
        return torch.fft.fft2(x, s=s)
    if array_api_compat.is_jax_namespace(xp):
        return jnp.fft.fft2(x, s=s)

    # Standard fft extension only exposes fftn, and requires complex input
    return xp.fft.fftn(_as_complex(xp, x), s=s, axes=(-2, -1))


def _ifft2(x: Array) -> Array:
    """Backend-specific inverse 2D fast Fourier transform.

    Computes the inverse two-dimensional discrete Fourier transform using the FFT
    implementation of the input array backend.

    Parameters
    ----------
    x : Array
        Complex-valued frequency-domain input array.

    Returns
    -------
    Array
        Inverse 2D Fourier transform of the input array.
    """
    xp = array_api_compat.array_namespace(x)

    if array_api_compat.is_cupy_namespace(xp):
        return cupyx.scipy.fft.ifft2(x)
    if array_api_compat.is_torch_namespace(xp):
        return torch.fft.ifft2(x)
    if array_api_compat.is_jax_namespace(xp):
        return jnp.fft.ifft2(x)

    return xp.fft.ifftn(_as_complex(xp, x), axes=(-2, -1))

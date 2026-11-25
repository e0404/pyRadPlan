"""Provide Array API compliant functions and utilities (e.g. from numpy)."""

from typing import Literal, Union, Sequence, Any

from .typing import Array
from functools import lru_cache, partial

import array_api_compat
import numpy as np

from ._backends import loaded_backend


from ._interpolation import interpnd  # noqa: F401 -- preserve the compatibility import


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


@lru_cache(maxsize=None)
def _compiled_interp1d_kernel(backend: str):
    """Compile the 1D interpolation kernel once per backend."""
    if backend == "jax":
        import jax  # noqa: PLC0415
        from jax import numpy as jnp  # noqa: PLC0415

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

        return jax.jit(_interp1d_jax_kernel)

    import torch  # noqa: PLC0415

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

    return torch.jit.script(_interp1d_torch_kernel)


@lru_cache(maxsize=1)
def _get_jax_interp1d(xp):
    """Get JAX implementation of 1D interpolation, compiled with jax.jit."""
    jnp = loaded_backend("jax.numpy")
    kernel = _compiled_interp1d_kernel("jax")

    def fn(xq, x, y, left, right):
        left = jnp.take(y, 0, axis=-1) if left is None else jnp.asarray(left, dtype=y.dtype)
        right = jnp.take(y, -1, axis=-1) if right is None else jnp.asarray(right, dtype=y.dtype)

        return kernel(xq, x, y, left, right)

    return fn


@lru_cache(maxsize=1)
def _get_torch_interp1d(xp):
    """Get PyTorch implementation of 1D interpolation."""
    torch = loaded_backend("torch")
    kernel = _compiled_interp1d_kernel("torch")

    def fn(xq, x, y, left, right):
        left = y[..., 0] if left is None else torch.as_tensor(left, dtype=y.dtype, device=y.device)
        right = (
            y[..., -1] if right is None else torch.as_tensor(right, dtype=y.dtype, device=y.device)
        )
        return kernel(xq, x, y, left, right)

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
    torch = loaded_backend("torch")
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
    torch = loaded_backend("torch")
    jnp = loaded_backend("jax.numpy")
    xp = array_api_compat.array_namespace(x)

    if array_api_compat.is_cupy_namespace(xp):
        from cupyx.scipy import fft  # noqa: PLC0415

        return fft.fft2(x, s=s)
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
    torch = loaded_backend("torch")
    jnp = loaded_backend("jax.numpy")
    xp = array_api_compat.array_namespace(x)

    if array_api_compat.is_cupy_namespace(xp):
        from cupyx.scipy import fft  # noqa: PLC0415

        return fft.ifft2(x)
    if array_api_compat.is_torch_namespace(xp):
        return torch.fft.ifft2(x)
    if array_api_compat.is_jax_namespace(xp):
        return jnp.fft.ifft2(x)

    return xp.fft.ifftn(_as_complex(xp, x), axes=(-2, -1))

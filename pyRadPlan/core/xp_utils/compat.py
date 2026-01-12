"""Provide Array API compliant functions and utilities (e.g. from numpy)."""

from typing import Literal, Union, Sequence, Any

from .typing import Array

import array_api_compat
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None


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

    if not is_sorted:
        x_sorted = xp.sort(x, axis=axis, stable=False)
    else:
        x_sorted = x

    float_ix = p * (x.shape[axis] - 1)

    if method == "nearest" or float_ix == round(float_ix):
        k = int(round(float_ix))
        return xp.take(x_sorted, xp.reshape(xp.asarray(k), shape=(1,)), axis=axis)

    # Note that the case float_ix == round(float_ix) is already handled above, so we can assume
    # that our float_ix lies between two indices
    if method == "linear":
        k_lower = int(float_ix)
        k_upper = k_lower + 1

        assert k_lower != k_upper  # This should be captured above

        # We try to do as much in place on x_upper to save memory allocations
        x_upper = xp.take(x_sorted, xp.reshape(xp.asarray(k_upper), shape=(1,)), axis=axis)
        weight_upper = float_ix - k_lower

        x_upper *= weight_upper

        x_upper += (1.0 - weight_upper) * xp.take(
            x_sorted, xp.reshape(xp.asarray(k_lower), shape=(1,)), axis=axis
        )

        # Squeeze might not work correctly with pytorch?
        return xp.squeeze(x_upper, axis=axis)

    # interp_core.py


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

    if isinstance(y, (list, tuple)) and stack:
        y = xp.stack(y, axis=0)

    if isinstance(y, dict) and stack:
        y = xp.stack([*y.values()], axis=0)

    # for numpy, calling np.interp is usually much faster than our code
    if array_api_compat.is_numpy_namespace(xp):
        if not isinstance(y, (list, tuple, dict)) and y.ndim == 1:
            return np.interp(xq, x, y)

        if not isinstance(y, (list, tuple, dict)) and y.ndim == 2:
            return np.apply_along_axis(lambda ytmp: np.interp(xq, x, ytmp), axis=-1, arr=y)

        if isinstance(y, (list, tuple)):
            return [np.interp(xq, x, ytmp) for ytmp in y]

        if isinstance(y, dict) and all(ytmp.ndim == 1 for ytmp in y.values()):
            return {key: np.interp(xq, x, ytmp) for key, ytmp in y.items()}

    if (
        not isinstance(y, (list, tuple, dict))
        and y.ndim == 1
        and array_api_compat.is_cupy_namespace(xp)
    ):
        return cp.interp(xq, x, y)

    xq = xp.clip(xq, x[0], x[-1])

    # find interval indices and handle x coordinates
    idx = xp.searchsorted(x, xq, side="left")
    idx -= 1

    i0 = xp.clip(idx, 0, x.shape[0] - 2)
    i1 = i0 + 1

    x0 = x[i0]
    xq -= x0
    xq /= x[i1] - x0

    def _final_y_interp(y: Array, i0: Array, i1: Array, t: Array):
        y0 = xp.take(y, i0, axis=-1)
        y1 = xp.take(y, i1, axis=-1)

        # y_interp = y0 + (y1 - y0) * t
        y1 -= y0
        if y1.ndim > 1:
            y1 *= xp.expand_dims(t, axis=0)
        else:
            y1 *= t
        y1 += y0
        return y1

    # Perform interpolation on y
    if isinstance(y, (list, tuple)):
        result = [_final_y_interp(y_arr, i0, i1, xq) for y_arr in y]
    elif isinstance(y, dict):
        result = {key: _final_y_interp(ytmp, i0, i1, xq) for key, ytmp in y.items()}
    else:
        result = _final_y_interp(y, i0, i1, xq)
    return result

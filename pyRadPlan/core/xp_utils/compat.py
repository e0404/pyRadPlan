"""Provide Array API compliant functions and utilities (e.g. from numpy)."""

from typing import Literal, Union, Sequence, Any

from .typing import Array
from . import pytorch_gpu_available
from . import to_numpy
import array_api_compat
import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None
try:
    from cupyx.scipy.interpolate import RegularGridInterpolator as CupyRegularGridInterpolator
except ModuleNotFoundError:
    CupyRegularGridInterpolator = None

from scipy.interpolate import RegularGridInterpolator as NumpyRegularGridInterpolator


if pytorch_gpu_available():

    @torch.jit.script
    def _interp1d_torch_kernel(xq, x, y):
        # xq: (N_out)
        # x: (N_in)
        # y: (N_channels, N_in) or (N_in)

        # Clip xq
        xq = torch.clamp(xq, min=x[0], max=x[-1])

        # Searchsorted
        idx = torch.searchsorted(x, xq, right=False)
        idx = idx - 1
        idx = torch.clamp(idx, min=0, max=x.shape[0] - 2)

        i0 = idx
        i1 = idx + 1

        x0 = x[i0]
        x1 = x[i1]

        # Compute t
        t = (xq - x0) / (x1 - x0)

        # y interpolation
        # Handle y dim
        if y.ndim == 1:
            y0 = y[i0]
            y1 = y[i1]
            return y0 + (y1 - y0) * t
        else:
            # y is (C, N_in) (stacked)
            # We need y[:, i0]
            # i0 is (N_out). y is (C, N_in).
            # We want (C, N_out).
            y0 = y[:, i0]
            y1 = y[:, i1]

            # t is (N_out).
            # y1-y0 is (C, N_out)
            # We need to unsqueeze t to (1, N_out) to broadcast against C
            return y0 + (y1 - y0) * t.unsqueeze(0)


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

    if torch is not None and array_api_compat.is_torch_namespace(xp):
        # Optimized PyTorch implementation
        y_stacked = None
        is_seq = isinstance(y, (list, tuple))
        is_dict = isinstance(y, dict)

        keys = None
        if is_dict:
            keys = list(y.keys())
            try:
                # Assuming all values in dict are tensors of same shape
                y_stacked = torch.stack([y[k] for k in keys], dim=0)
            except Exception:
                # Fallback if stack fails (e.g. different shapes/devices)
                pass
        elif is_seq:
            try:
                y_stacked = torch.stack(y, dim=0)
            except Exception:
                pass
        elif isinstance(y, torch.Tensor):
            y_stacked = y

        if y_stacked is not None:
            # Run JIT optimized kernel
            # Ensure dtypes match where needed or let JIT handle it
            res = _interp1d_torch_kernel(xq, x, y_stacked)

            if stack:
                return res

            if is_dict:
                return {k: res[i] for i, k in enumerate(keys)}
            elif is_seq:
                return [res[i] for i in range(len(y))]
            else:
                return res

    # Optimization: Use xp.interp if available (e.g. numpy, cupy) for list/dict inputs
    # This avoids stacking inputs and using the slower manual implementation
    interp_func = None
    if array_api_compat.is_numpy_namespace(xp):
        interp_func = np.interp
    elif array_api_compat.is_cupy_namespace(xp):
        interp_func = cp.interp
    elif hasattr(xp, "interp"):
        interp_func = xp.interp

    if interp_func is not None:
        if isinstance(y, dict) and all(v.ndim == 1 for v in y.values()):
            res = {k: interp_func(xq, x, v) for k, v in y.items()}
            if stack:
                return xp.stack(list(res.values()), axis=0)
            return res

        if isinstance(y, (list, tuple)) and all(v.ndim == 1 for v in y):
            res = [interp_func(xq, x, v) for v in y]
            if stack:
                return xp.stack(res, axis=0)
            return res

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

    is_sequence = isinstance(y, (list, tuple))
    is_dict = isinstance(y, dict)

    if is_sequence:
        result = [_final_y_interp(y_arr, i0, i1, xq) for y_arr in y]
    elif is_dict:
        result = {key: _final_y_interp(ytmp, i0, i1, xq) for key, ytmp in y.items()}
    else:
        result = _final_y_interp(y, i0, i1, xq)
    return result


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


def interp3d(
    points, values, method="linear", bounds_error=True, fill_value=np.nan
) -> Union[Array, Sequence[Array], dict[Any, Array]]:
    """
    Array API Conform 3D interpolation.

    """
    # points is a tuple/list of grid vectors; expand it so array_namespace
    # sees actual array objects instead of the container itself.
    xp = array_api_compat.array_namespace(values, *points)

    if array_api_compat.is_numpy_namespace(xp):
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
        return NumpyRegularGridInterpolator(
            points_np,
            values_np,
            method=method,
            bounds_error=bounds_error,
            fill_value=float(fill_value),
        )

    if array_api_compat.is_cupy_namespace(xp):
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

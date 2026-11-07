"""Backend reductions missing from the Array API standard."""

import array_api_compat

from .typing import Array


def grouped_max(values: Array, groups: Array, num_groups: int, *, initial: float) -> Array:
    """Reduce one-dimensional values by integer group, on their original device.

    Parameters
    ----------
    values : Array
        One-dimensional real numeric values. NaNs propagate within a group.
    groups : Array
        Integer group indices, one per value, in ``[0, num_groups)``.
    num_groups : int
        Output length, including any empty groups.
    initial : float
        Initial maximum for every group, including empty groups. Converted to
        the values dtype; use a representable lower bound for signed integers.

    Returns
    -------
    Array
        Maximum of initial and each group's values, with the values dtype.

    Notes
    -----
    NumPy and CuPy use indexed maximum reduction. Other namespaces use a
    sorting fallback entirely on the input device, without host array copies.
    """
    xp = array_api_compat.array_namespace(values, groups)
    device = array_api_compat.device(values)
    if values.ndim != 1 or groups.shape != values.shape:
        raise ValueError("Values and groups must be one-dimensional with equal shapes.")
    if not xp.isdtype(groups.dtype, "integral"):
        raise TypeError("Group indices must have an integer dtype.")
    if not xp.isdtype(values.dtype, ("real floating", "integral")):
        raise TypeError("Values must have a real numeric dtype.")
    if array_api_compat.device(groups) != device:
        raise ValueError("Values and groups must share a device.")
    if num_groups < 0:
        raise ValueError("Number of groups must be nonnegative.")
    if bool(xp.any((groups < 0) | (groups >= num_groups))):
        raise ValueError("Group indices must lie in [0, num_groups).")
    result = xp.full((num_groups,), initial, dtype=values.dtype, device=device)
    if values.shape[0] == 0:
        return result
    # CuPy atomic reductions support fewer dtypes than its sorting kernels.
    # Keep untested narrow/integer types on the portable fallback.
    if array_api_compat.is_numpy_namespace(xp) or (
        array_api_compat.is_cupy_namespace(xp)
        and values.dtype in (xp.float32, xp.float64, xp.int32)
    ):
        xp.maximum.at(result, groups, values)
        if array_api_compat.is_cupy_namespace(xp) and xp.isdtype(values.dtype, "real floating"):
            # CUDA atomicMax ignores NaNs. Track them separately to retain the
            # same propagation contract as NumPy and the sorting fallback.
            nan_groups = xp.zeros((num_groups,), dtype=xp.int32, device=device)
            xp.maximum.at(nan_groups, groups, xp.astype(xp.isnan(values), xp.int32))
            result = xp.where(
                nan_groups > 0, xp.asarray(float("nan"), dtype=values.dtype, device=device), result
            )
            result = xp.maximum(result, xp.asarray(initial, dtype=values.dtype, device=device))
        return result

    # Sort values first, then stably group them: the last value of each block
    # is its maximum (or NaN). Search every output group, including empty ones.
    order = xp.argsort(values, stable=True)
    order = xp.take(order, xp.argsort(xp.take(groups, order, axis=0), stable=True), axis=0)
    sorted_groups = xp.take(groups, order, axis=0)
    output_groups = xp.arange(num_groups, dtype=groups.dtype, device=device)
    ends = xp.searchsorted(sorted_groups, output_groups, side="right") - 1
    ends = xp.maximum(ends, xp.zeros_like(ends))
    present = xp.take(sorted_groups, ends, axis=0) == output_groups
    maxima = xp.take(values, xp.take(order, ends, axis=0), axis=0)
    return xp.where(present, xp.maximum(result, maxima), result)

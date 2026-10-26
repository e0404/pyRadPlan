"""Frozen NumPy/CuPy reference paths from the pencil-beam performance audit.

These retain the pre-refactor interpolation work for reproducible comparisons;
they are not used by production code. The legacy path matches compat.interpnd
at c1ee82ae. The direct native path matches the engine integration at 9a045eeb.
"""

from functools import partial

import array_api_compat


def native_grid_interpolator(points, values, *, bounds_error=False):
    """Construct the direct native interpolator used by the first engine patch."""
    if array_api_compat.is_cupy_array(values):
        from cupyx.scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415
    else:
        from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415
    return RegularGridInterpolator(points, values, bounds_error=bounds_error)


def legacy_interpnd(xq, x, y, *, bounds_error=False):
    """Repeat grid normalization, query checks/clipping, and native construction."""
    xp = array_api_compat.array_namespace(*x, xq, y)
    if not (array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_cupy_namespace(xp)):
        raise NotImplementedError("This benchmark reference supports NumPy/CuPy only.")
    grids = list(x)
    for axis, g in enumerate(grids):
        if g[0] > g[-1]:
            grids[axis] = g[::-1]
            sl = [slice(None)] * y.ndim
            sl[axis] = slice(None, None, -1)
            y = y[tuple(sl)]
    if bounds_error:
        for axis, g in enumerate(grids):
            col = xq[:, axis]
            if bool(xp.any(col < g[0])) or bool(xp.any(col > g[-1])):
                raise ValueError("Query points lie outside the grid bounds.")
    xq = xp.stack([xp.clip(xq[:, axis], g[0], g[-1]) for axis, g in enumerate(grids)], axis=1)
    return native_grid_interpolator(tuple(grids), y, bounds_error=True)(xq)


def legacy_grid_interpolator(points, values, *, bounds_error=False):
    """Bind data without preparing it, matching the original photon call site."""
    return partial(legacy_interpnd, x=points, y=values, bounds_error=bounds_error)

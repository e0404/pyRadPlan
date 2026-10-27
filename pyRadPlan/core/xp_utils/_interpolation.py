"""Reusable Array API interpolation on rectilinear grids."""

from functools import lru_cache, partial

import array_api_compat

from ._backends import loaded_backend
from .typing import Array


class ArrayBackendGridInterpolator:
    """Reusable linear interpolation on a rectilinear grid of backend arrays.

    Parameters
    ----------
    points : tuple[Array, ...]
        One-dimensional, strictly ascending or descending coordinate arrays.
        Spacing may be nonuniform. All arrays must share a namespace and device.
    values : Array
        Grid values with shape ``(*grid_shape, *value_shape)``. Optional trailing
        dimensions represent fields evaluated with shared coordinate searches.
    bounds_error : bool, optional
        Raise ValueError for queries outside the grid if True. Otherwise clip
        coordinates to the boundary (default), rather than extrapolating.
    prefer_native : bool or None, optional
        None (default) selects the measured faster implementation: generic for
        NumPy multi-value 2D grids, native where available otherwise. True
        explicitly prefers SciPy, CuPy, or JAX. False selects the generic Array
        API implementation (2D/3D grids with at least two points per axis).

    Notes
    -----
    Preparation and backend selection happen once at construction. The object
    retains grid/value arrays and native state without a global data cache or
    implicit device transfers. Do not modify the input arrays during its lifetime;
    construct another interpolator when the grid or values change.

    Queries must have shape ``(N, len(points))`` and share the input namespace and
    device. Result dtype follows the selected implementation, as with ``interpnd``.
    Grid validation occurs eagerly; constructing this object inside a traced JAX
    function is not supported. JAX evaluation reuses a cached compiled function
    with arrays passed as arguments, avoiding compilation per object instance.
    """

    def __init__(
        self,
        points: tuple[Array, ...],
        values: Array,
        *,
        bounds_error: bool = False,
        prefer_native: bool | None = None,
    ):
        self._xp = xp = array_api_compat.array_namespace(*points, values)
        self._array_type = type(values)
        self._device = array_api_compat.device(values)
        self._ndim = len(points)
        self._value_shape = values.shape[self._ndim :]
        self._bounds_error = bounds_error
        self._native_bounds = False
        self._implementation = "array_api"
        if not points or any(p.ndim != 1 or p.shape[0] == 0 for p in points):
            raise ValueError("Grid axes must be nonempty one-dimensional arrays.")
        if values.shape[: self._ndim] != tuple(p.shape[0] for p in points):
            raise ValueError("Values shape must match the grid axes.")
        if any(array_api_compat.device(p) != self._device for p in points):
            raise ValueError("Grid axes and values must share a device.")

        # Keep bounds in the input precision even if a native implementation
        # converts its own axes to float64. No host copies are needed.
        if prefer_native is None:
            # Shared searches pay off on both CPU and GPU, but the native
            # vector-value evaluator wins only on CuPy in measured 2D workloads.
            # Keep native scalar / singleton / higher-dimensional support.
            prefer_native = not (
                array_api_compat.is_numpy_namespace(xp)
                and bool(self._value_shape)
                and self._ndim == 2
                and all(p.shape[0] >= 2 for p in points)
            )
        native = _native_interpolator_class(xp) if prefer_native else None
        self._bounds = (
            tuple((xp.minimum(p[0], p[-1]), xp.maximum(p[0], p[-1])) for p in points)
            if not bounds_error or native is None
            else ()
        )
        if native is not None:
            # CuPy's NaN mask supports only one trailing value axis. Flatten
            # multiple value dimensions internally and restore them on return.
            native_values = values
            if len(self._value_shape) > 1:
                native_values = xp.reshape(values, values.shape[: self._ndim] + (-1,))
            self._evaluate = native(points, native_values, bounds_error=bounds_error)
            self._native_bounds = True
            self._implementation = "scipy" if array_api_compat.is_numpy_namespace(xp) else "cupy"
            return

        points, values = _normalize_grid(xp, points, values)
        if prefer_native and array_api_compat.is_jax_namespace(xp):
            self._evaluate = partial(_get_jax_interpnd(), x=points, y=values)
            self._implementation = "jax"
        else:
            if len(points) not in (2, 3):
                raise NotImplementedError(
                    "Only 2D and 3D interpolation is currently implemented for the generic "
                    "fallback. Note, that torch does use the generic fallback."
                )
            if any(p.shape[0] < 2 for p in points):
                raise ValueError("Generic interpolation requires at least two points per axis.")
            core = _interp2d_core if len(points) == 2 else _interp3d_core
            options = {}
            if len(points) == 2 and self._value_shape:
                # Pack fields once, with grid samples contiguous within each field.
                # Evaluation then broadcasts weights along long query rows, avoiding
                # repeated short-row broadcasting for (N, number_of_fields) arrays.
                flat = xp.reshape(values, (points[0].shape[0] * points[1].shape[0], -1))
                # Flatten in logical (row-major) order; asarray(copy=True) alone
                # preserves NumPy's transposed strides and makes take copy per call.
                packed = xp.reshape(xp.permute_dims(flat, (1, 0)), (-1,))
                values = xp.reshape(packed, (flat.shape[1], flat.shape[0]))
                options["values_first"] = True
            self._evaluate = partial(
                core, xp, **dict(zip(("gx", "gy", "gz"), points)), y=values, **options
            )

    @property
    def implementation(self) -> str:
        """Selected implementation: scipy, cupy, jax, or array_api (read-only)."""
        return self._implementation

    def __call__(self, xi: Array) -> Array:
        """Return shape ``(N, *value_shape)`` for queries of shape ``(N, ndim)``."""
        xp = self._xp
        # The usual query has the same array type as the prepared values. Only
        # unusual/subclass inputs need full namespace discovery again.
        if type(xi) is not self._array_type and array_api_compat.array_namespace(xi) != xp:
            raise TypeError("Queries must share the interpolator's array namespace.")
        if array_api_compat.device(xi) != self._device:
            raise ValueError("Queries must share the interpolator's device.")
        if xi.ndim != 2 or xi.shape[1] != self._ndim:
            raise ValueError(f"Queries must have shape (N, {self._ndim}).")
        if self._bounds_error:
            # Native NumPy/CuPy already check the queries. Duplicating this check
            # would add GPU reductions and synchronization on every evaluation.
            if not self._native_bounds:
                for axis, (lower, upper) in enumerate(self._bounds):
                    col = xi[:, axis]
                    if bool(xp.any(col < lower)) or bool(xp.any(col > upper)):
                        raise ValueError("Query points lie outside the grid bounds.")
        else:
            xi = xp.stack(
                [xp.clip(xi[:, axis], lo, hi) for axis, (lo, hi) in enumerate(self._bounds)],
                axis=1,
            )
        result = self._evaluate(xi)
        return xp.reshape(result, (xi.shape[0],) + self._value_shape)


def _native_interpolator_class(xp):
    """Load only the native implementation belonging to this namespace."""
    try:
        if array_api_compat.is_numpy_namespace(xp):
            from scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

            return RegularGridInterpolator
        if array_api_compat.is_cupy_namespace(xp):
            from cupyx.scipy.interpolate import RegularGridInterpolator  # noqa: PLC0415

            return RegularGridInterpolator
    except ImportError:
        pass  # The Array API implementation remains available without the library.
    return None


def _normalize_grid(xp, points, values):
    """Validate grid ordering and normalize descending axes once."""
    ascending = []
    for axis, p in enumerate(points):
        grid = p
        if p.shape[0] > 1:
            if bool(xp.all(p[1:] < p[:-1])):
                grid = xp.flip(p, axis=0)
                values = xp.flip(values, axis=axis)
            elif not bool(xp.all(p[1:] > p[:-1])):
                raise ValueError("Grid axes must be strictly ascending or descending.")
        ascending.append(grid)
    return tuple(ascending), values


@lru_cache(maxsize=1)
def _get_jax_interpnd():
    """Share JAX compilation across objects without caching their grid arrays."""
    jax = loaded_backend("jax")
    interpolate = loaded_backend("jax.scipy.interpolate")
    return jax.jit(lambda xq, x, y: interpolate.RegularGridInterpolator(x, y)(xq))


def interpnd(
    xq: Array,
    x: tuple[Array, ...],
    y: Array,
    *,
    bounds_error: bool = False,
    prefer_native: bool | None = None,
) -> Array:
    """Interpolate once on a rectilinear grid of Array API arrays.

    Parameters
    ----------
    xq : Array
        Query coordinates with shape ``(N, len(x))``.
    x : tuple[Array, ...]
        One-dimensional grid axes.
    y : Array
        Grid values with optional trailing value dimensions after the grid axes.
    bounds_error : bool, optional
        Raise for out-of-grid queries if True; otherwise clip to the grid (default).
    prefer_native : bool or None, optional
        None selects the measured backend policy. True prefers native libraries;
        False selects generic 2D/3D interpolation. See the reusable class.

    Returns
    -------
    Array
        Interpolated values on the input device.

    See Also
    --------
    ArrayBackendGridInterpolator : Retains preparation for repeated queries.
    """
    return ArrayBackendGridInterpolator(
        x, y, bounds_error=bounds_error, prefer_native=prefer_native
    )(xq)


def _interp2d_core(
    xp, xq: Array, gx: Array, gy: Array, y: Array, *, values_first: bool = False
) -> Array:
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

    if values_first:
        y_flat = y
        weight_shape = (1, -1)
        gather_axis = 1
    else:
        y_flat = xp.reshape(y, (nx * ny,) + y.shape[2:])
        weight_shape = (-1,) + (1,) * (y.ndim - 2)
        gather_axis = 0
    tx, ty = xp.reshape(tx, weight_shape), xp.reshape(ty, weight_shape)

    stride_x = ny
    base = ix * stride_x + iy

    i00 = base
    i01 = base + 1
    i10 = base + stride_x
    i11 = base + stride_x + 1

    c00 = xp.take(y_flat, i00, axis=gather_axis)
    c01 = xp.take(y_flat, i01, axis=gather_axis)
    c10 = xp.take(y_flat, i10, axis=gather_axis)
    c11 = xp.take(y_flat, i11, axis=gather_axis)

    ux = 1.0 - tx
    uy = 1.0 - ty

    result = ux * uy * c00 + ux * ty * c01 + tx * uy * c10 + tx * ty * c11
    return xp.permute_dims(result, (1, 0)) if values_first else result


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
    y_flat = xp.reshape(y, (nx * ny * nz,) + y.shape[3:])
    weight_shape = (-1,) + (1,) * (y.ndim - 3)
    tx, ty, tz = (xp.reshape(t, weight_shape) for t in (tx, ty, tz))

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

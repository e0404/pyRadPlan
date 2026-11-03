"""Prepared interpolation of owner-tagged queries on multiple one-dimensional grids."""

import math

import array_api_compat

from ._interpolation1d import ArrayBackend1DInterpolator


class ArrayBackend1DInterpolatorBank:
    """Prepare compatible fields sampled on different one-dimensional grids.

    Parameters
    ----------
    points : sequence of Array
        Nonempty, finite, strictly increasing grids of possibly different lengths.
        All grids use the same float32/float64 dtype, namespace and device.
    values : sequence of Array or sequence of dict
        One value array or field dictionary per grid. Arrays have shape
        ``(n_points,)`` or ``(n_rows, n_points)``. Field keys, outer shapes and
        per-field float32/float64 dtypes must agree across tables. Different
        fields may have different dtypes. Inputs must remain unchanged.
    left, right : float or None, optional
        Constant extrapolation values, defaulting to each table's endpoints.
    prefer_native : bool or None, optional
        None groups NumPy queries through the existing prepared interpolator
        and uses portable execution elsewhere. True prefers backend-specific
        execution, including fused CuPy searches/gathers for float64 tables.
        False uses Array API binary searches and gathers on every backend.
        Unsupported specialized combinations retain the portable implementation.

    Notes
    -----
    Calls accept one table index per query, in arbitrary order, and optionally
    one grid translation per query. Translations move the grid, not the queries.
    All tables must fit in memory; callers bound query batches independently.

    The portable formula and the CuPy specialization follow generic prepared
    interpolation, including singleton constant extension and NaN propagation.
    NumPy's native preference retains its existing prepared dispatch, including
    float64 native results. As with the single-grid interpolator, changing native
    preference may change output precision and last-bit rounding.

    Construction validates grids. Calls validate index bounds with one scalar
    decision, which synchronizes device execution; no query arrays are downloaded.
    This class does not change dispatch of existing single-grid interpolators.
    """

    def __init__(self, points, values, *, left=None, right=None, prefer_native=None):
        points, values = tuple(points), tuple(values)
        if not points or len(points) != len(values):
            raise ValueError("Provide equally many nonempty grids and value tables.")
        self._is_dict = isinstance(values[0], dict)
        if any(isinstance(value, dict) != self._is_dict for value in values):
            raise TypeError("All value tables must use the same container kind.")
        fields = values if self._is_dict else tuple({None: value} for value in values)
        self._keys = tuple(fields[0])
        if not self._keys or any(set(f) != set(self._keys) for f in fields):
            raise ValueError("Every table must have the same nonempty set of fields.")
        xp = array_api_compat.array_namespace(*points, *(v for f in fields for v in f.values()))
        self._xp = xp
        self._device = array_api_compat.device(points[0])
        self._dtype = points[0].dtype
        self._count = len(points)
        lengths = []
        for x, table in zip(points, fields):
            if x.ndim != 1 or x.shape[0] == 0:
                raise ValueError("Each grid must be nonempty and one-dimensional.")
            if x.dtype != self._dtype or x.dtype not in (xp.float32, xp.float64):
                raise TypeError("Grids must share a float32 or float64 dtype.")
            if array_api_compat.device(x) != self._device:
                raise ValueError("All tables must share a device.")
            if not bool(xp.all(xp.isfinite(x))) or not bool(xp.all(x[1:] > x[:-1])):
                raise ValueError("Grids must be finite and strictly increasing.")
            for key, value in table.items():
                reference = fields[0][key]
                if value.ndim not in (1, 2) or value.shape[-1] != x.shape[0]:
                    raise ValueError("Fields must have shape (n_points,) or (n_rows, n_points).")
                if value.shape[:-1] != reference.shape[:-1] or value.dtype != reference.dtype:
                    raise ValueError("Field outer shapes and dtypes must agree across tables.")
                if value.dtype not in (xp.float32, xp.float64):
                    raise TypeError("Fields must have a float32 or float64 dtype.")
                if array_api_compat.device(value) != self._device:
                    raise ValueError("All tables must share a device.")
            lengths.append(x.shape[0])
        for boundary in (left, right):
            if boundary is not None and not isinstance(boundary, (int, float)):
                raise TypeError("Boundary values must be scalar numbers or None.")
        self._left, self._right = left, right
        self._steps = math.ceil(math.log2(max(lengths) + 1))
        self._all_singletons = max(lengths) == 1
        starts, total = [], 0
        for length in lengths:
            starts.append(total)
            total += length
        self._starts = xp.asarray(starts, dtype=xp.int64, device=self._device)
        self._lengths = xp.asarray(lengths, dtype=xp.int64, device=self._device)
        self._points = xp.concat(points)
        self._fields = {key: xp.concat([f[key] for f in fields], axis=-1) for key in self._keys}
        self._prepared = None
        if prefer_native is not False and array_api_compat.is_numpy_namespace(xp):
            self._prepared = [
                ArrayBackend1DInterpolator(
                    x, y, left=left, right=right, prefer_native=prefer_native
                )
                for x, y in zip(points, values)
            ]
        self._fused = (
            prefer_native is True
            and array_api_compat.is_cupy_namespace(xp)
            and self._dtype == xp.float64
            and all(value.dtype == xp.float64 for value in self._fields.values())
        )

    @property
    def implementation(self):
        """Report the prepared execution policy without importing optional backends."""
        if self._prepared is not None:
            return "numpy_prepared"
        return "cupy_fused" if self._fused else "array_api"

    def __call__(self, queries, table_indices, *, grid_offsets=None):
        """Evaluate 1D queries using integer table IDs and optional grid translations.

        ``grid_offsets`` is None, a finite Python scalar, or a same-device array
        with the grid dtype and shape ``()`` or ``queries.shape``. Indices must
        have the query shape and lie in ``[0, number_of_tables)``. Queries may
        use float32 or float64 independently of the grid dtype.
        """
        xp = self._xp
        arrays = [queries, table_indices]
        if array_api_compat.is_array_api_obj(grid_offsets):
            arrays.append(grid_offsets)
            if grid_offsets.dtype != self._dtype or grid_offsets.shape not in ((), queries.shape):
                raise ValueError("Grid offsets must have the grid dtype and scalar/query shape.")
        elif grid_offsets is not None and (
            not isinstance(grid_offsets, (int, float)) or not math.isfinite(grid_offsets)
        ):
            raise ValueError("Grid offsets must be finite numbers or backend arrays.")
        if array_api_compat.array_namespace(*arrays) != xp:
            raise TypeError("Queries, indices and offsets must share the table namespace.")
        if any(array_api_compat.device(a) != self._device for a in arrays):
            raise ValueError("Queries, indices and offsets must share the table device.")
        if queries.ndim != 1 or table_indices.shape != queries.shape:
            raise ValueError("Queries and table indices must be one-dimensional and equal-sized.")
        if queries.dtype not in (xp.float32, xp.float64):
            raise TypeError("Queries must have a float32 or float64 dtype.")
        if not xp.isdtype(table_indices.dtype, "integral"):
            raise TypeError("Table indices must have an integer dtype.")
        shifts = xp.asarray(
            0.0 if grid_offsets is None else grid_offsets, dtype=self._dtype, device=self._device
        )
        invalid = (table_indices < 0) | (table_indices >= self._count)
        if bool(xp.any(invalid) | xp.any(~xp.isfinite(shifts))):
            raise ValueError("Table indices are out of bounds or grid offsets are nonfinite.")
        if self._prepared is not None:
            return self._numpy_grouped(queries, table_indices, shifts)
        if self._fused:
            from ._interpolation_bank_cupy import evaluate_bank  # noqa: PLC0415

            result = evaluate_bank(self, queries, table_indices, shifts)
        else:
            result = self._generic(queries, table_indices, shifts)
        return result if self._is_dict else result[None]

    def _numpy_grouped(self, queries, indices, shifts):
        xp = self._xp
        offsets = xp.broadcast_to(shifts, queries.shape)
        order = xp.lexsort((offsets, indices))
        ids, delta = indices[order], offsets[order]
        edges = xp.nonzero((ids[1:] != ids[:-1]) | (delta[1:] != delta[:-1]))[0] + 1
        result = {
            key: xp.empty(
                value.shape[:-1] + queries.shape,
                dtype=xp.float64
                if self._prepared[0]._native
                else value.dtype
                if self._all_singletons
                else xp.result_type(value.dtype, self._dtype, queries.dtype),
            )
            for key, value in self._fields.items()
        }
        for group in xp.split(order, edges):
            if group.size == 0:
                continue
            first = group[0]
            values = self._prepared[int(indices[first])](
                queries[group], grid_offset=offsets[first]
            )
            fields = values if self._is_dict else {None: values}
            for key, value in fields.items():
                result[key][..., group] = value
        return result if self._is_dict else result[None]

    def _generic(self, q, ids, shifts):
        xp = self._xp
        start, length = xp.take(self._starts, ids), xp.take(self._lengths, ids)
        lower = xp.take(self._points, start) + shifts
        upper = xp.take(self._points, start + length - 1) + shifts
        if array_api_compat.is_torch_namespace(xp):
            # Match weak promotion of the single-table 0-D clamp boundaries.
            lower, upper = xp.astype(lower, q.dtype), xp.astype(upper, q.dtype)
        below, above = q < lower, q > upper
        clipped = xp.minimum(xp.maximum(q, lower), upper)
        lo, hi = xp.zeros_like(start), length
        for _ in range(self._steps):
            mid = (lo + hi) // 2
            value = xp.take(self._points, start + xp.minimum(mid, length - 1)) + shifts
            right = value < clipped
            lo, hi = xp.where(right, mid + 1, lo), xp.where(right, hi, mid)
        i0 = start + xp.maximum(xp.zeros_like(lo), xp.minimum(lo - 1, length - 2))
        i1 = xp.minimum(i0 + 1, start + length - 1)
        x0, x1 = xp.take(self._points, i0) + shifts, xp.take(self._points, i1) + shifts
        weight = (clipped - x0) / xp.where(length == 1, 1.0, x1 - x0)
        result = {}
        for key, value in self._fields.items():
            y0, y1 = xp.take(value, i0, axis=-1), xp.take(value, i1, axis=-1)
            out = (
                y0 if self._all_singletons else xp.where(length == 1, y0, y0 + (y1 - y0) * weight)
            )
            left = (
                xp.take(value, start, axis=-1)
                if self._left is None
                else xp.asarray(self._left, dtype=value.dtype, device=self._device)
            )
            right = (
                xp.take(value, start + length - 1, axis=-1)
                if self._right is None
                else xp.asarray(self._right, dtype=value.dtype, device=self._device)
            )
            result[key] = xp.where(above, right, xp.where(below, left, out))
        return result

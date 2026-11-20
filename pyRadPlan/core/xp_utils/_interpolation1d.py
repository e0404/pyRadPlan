"""Reusable one-dimensional interpolation with backend-owned preparation."""

from functools import partial

import array_api_compat

from .compat import _get_jax_interp1d, _get_torch_interp1d, _interp1d_core
from .typing import Array


class ArrayBackend1DInterpolator:
    """Prepare linear interpolation of one or more fields on a shared axis.

    Parameters
    ----------
    x : Array
        Strictly increasing, nonempty, one-dimensional coordinates. A singleton
        axis returns its single value, subject to the boundary policy.
    y : Array or dict of Array
        Values shaped ``(n_points,)`` or ``(n_fields, n_points)``. A dictionary
        may mix these shapes and dtypes; its keys and individual shapes survive
        evaluation. Inputs must remain unchanged for the object's lifetime.
    left, right : float or None, optional
        Constant out-of-domain values, defaulting to each field's endpoint.
    prefer_native : bool or None, optional
        None retains the one-shot ``interp1d`` native dispatch policy. True
        prefers the namespace's ``interp`` when applicable. False uses shared
        searches through the backend-compatible kernel. PyTorch and JAX reuse
        the existing compiled kernels when no native path applies.

    Notes
    -----
    Preparation retains dispatch, packs compatible fields once, and preserves
    per-field dtypes. Unlike ``ArrayBackendGridInterpolator``, the interpolation
    axis is last, matching ``interp1d``. No device transfers occur during calls.
    """

    def __init__(self, x: Array, y: Array | dict, *, left=None, right=None, prefer_native=None):
        self._is_dict = isinstance(y, dict)
        fields = y if self._is_dict else {None: y}
        if not fields:
            raise ValueError("At least one interpolation field is required.")
        xp = array_api_compat.array_namespace(x, *fields.values())
        self._xp = xp
        self._x = x
        self._array_type = type(x)
        self._device = array_api_compat.device(x)
        self._keys = tuple(fields)
        if x.ndim != 1 or x.shape[0] == 0:
            raise ValueError("x must be nonempty and one-dimensional.")
        if not bool(xp.all(x[1:] > x[:-1])):
            raise ValueError("x must be strictly increasing.")
        for value in fields.values():
            if value.ndim not in (1, 2) or value.shape[-1] != x.shape[0]:
                raise ValueError("Fields must have shape (n_points,) or (n_fields, n_points).")
            if array_api_compat.device(value) != self._device:
                raise ValueError("Coordinates and values must share a device.")
        for boundary in (left, right):
            if boundary is not None and not isinstance(boundary, (int, float)):
                raise TypeError("left and right must be scalar numbers or None.")

        native = prefer_native is not False and hasattr(xp, "interp")
        # Preserve one-shot dispatch, including NumPy's float64 native outputs.
        self._native = native and (
            all(v.ndim == 1 for v in fields.values())
            or (not self._is_dict and array_api_compat.is_numpy_namespace(xp))
            or prefer_native is True
        )
        self._left, self._right = left, right
        self._fields = fields
        self._groups = []
        if self._native:
            return

        if x.shape[0] == 1:
            self._interpolate = partial(_interp1d_singleton, xp)
        elif array_api_compat.is_torch_namespace(xp):
            self._interpolate = _get_torch_interp1d(xp)
        elif array_api_compat.is_jax_namespace(xp):
            self._interpolate = _get_jax_interp1d(xp)
        else:
            self._interpolate = partial(_interp1d_core, xp)

        # Different dtypes must not silently promote one another. Pack rows of
        # the same dtype so they share search indices and interpolation weights.
        groups = {}
        for key, value in fields.items():
            groups.setdefault(value.dtype, []).append((key, value))
        for entries in groups.values():
            if len(entries) == 1:
                key, value = entries[0]
                self._groups.append((value, [(key, None)]))
                continue
            rows, layout, start = [], [], 0
            for key, value in entries:
                count = 1 if value.ndim == 1 else value.shape[0]
                rows.append(xp.reshape(value, (count, x.shape[0])))
                selection = start if value.ndim == 1 else slice(start, start + count)
                layout.append((key, selection))
                start += count
            self._groups.append((xp.concat(rows, axis=0), layout))

    def __call__(self, xq: Array, *, grid_offset=None):
        """Evaluate queries, optionally translating the grid by ``grid_offset``.

        ``grid_offset`` is a scalar added to the stored coordinates, without
        rebuilding field preparation. Shifting the grid (rather than queries)
        preserves the arithmetic of callers using ``interp1d(xq, x + offset, y)``.
        Query coordinates must be one-dimensional and share the input device.
        """
        xp = self._xp
        if type(xq) is not self._array_type and array_api_compat.array_namespace(xq) != xp:
            raise TypeError("Queries must share the interpolator's array namespace.")
        if array_api_compat.device(xq) != self._device:
            raise ValueError("Queries must share the interpolator's device.")
        if xq.ndim != 1:
            raise ValueError("Queries must be one-dimensional.")
        x = self._x if grid_offset is None else self._x + grid_offset
        result = {}
        if self._native:
            for key, value in self._fields.items():
                if value.ndim == 1:
                    result[key] = xp.interp(xq, x, value, left=self._left, right=self._right)
                else:
                    result[key] = xp.stack(
                        [
                            xp.interp(xq, x, row, left=self._left, right=self._right)
                            for row in value
                        ],
                        axis=0,
                    )
        else:
            for values, layout in self._groups:
                interpolated = self._interpolate(xq, x, values, self._left, self._right)
                for key, selection in layout:
                    result[key] = (
                        interpolated if selection is None else interpolated[selection, ...]
                    )
        if self._is_dict:
            return {key: result[key] for key in self._keys}
        return result[None]


def _interp1d_singleton(xp, xq, x, y, left, right):
    """Extend a single sample with NumPy's singleton and boundary semantics."""
    device = array_api_compat.device(y)
    left = y[..., :1] if left is None else xp.asarray(left, dtype=y.dtype, device=device)
    right = y[..., :1] if right is None else xp.asarray(right, dtype=y.dtype, device=device)
    result = xp.where(xq < x[0], left, y)
    return xp.where(xq > x[0], right, result)

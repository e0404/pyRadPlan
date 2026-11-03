"""Experimental table-bank interpolation; no production/backend dispatch changes.

The generic implementation searches different grids for owner-tagged queries in
one bounded vector operation. Preparation and backend semantics live here rather
than in the dose-engine prototype. CPU/CuPy benchmarking is driven by the pipeline
harness; this module also supplies independent boundary checks.
"""

import math
import importlib

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import ArrayBackend1DInterpolator, to_numpy


def scalar_bound(xp, bound, queries):
    """Preserve torch's weak promotion of a per-table zero-dimensional bound."""
    if array_api_compat.is_torch_namespace(xp):
        return xp.astype(bound, queries.dtype)
    return bound


class TableBank:
    """Immutable compatible fields on different grids, concatenated once per beam."""

    def __init__(self, points, fields):
        xp = array_api_compat.array_namespace(points[0])
        self.xp = xp
        self.device = array_api_compat.device(points[0])
        lengths = [x.shape[0] for x in points]
        if not lengths or min(lengths) < 1:
            raise ValueError("Every table must have at least one sample")
        self.starts = xp.asarray(np.cumsum([0, *lengths[:-1]]), device=self.device)
        self.lengths = xp.asarray(lengths, dtype=xp.int64, device=self.device)
        self.steps = math.ceil(math.log2(max(lengths) + 1))
        self.has_singletons = min(lengths) == 1
        self.points = xp.concat(points)
        self.fields = {key: xp.concat([f[key] for f in fields], axis=-1) for key in fields[0]}
        self.native_arithmetic = array_api_compat.is_numpy_namespace(xp) and all(
            f.ndim == 1 for f in fields[0].values()
        )
        self.cupy_native_nan = array_api_compat.is_cupy_namespace(xp) and all(
            f.ndim == 1 for f in fields[0].values()
        )
        # CuPy's interpolation arithmetic is checked experimentally, not assumed
        # to match NumPy. It keeps the generic interpolation order in this prototype.

    def __call__(self, q, table_ids, shifts, *, left=None, right=None):
        """Evaluate owner-tagged queries against their translated table grids."""
        xp = self.xp
        start = xp.take(self.starts, table_ids)
        length = xp.take(self.lengths, table_ids)
        lower = xp.take(self.points, start) + shifts
        upper = xp.take(self.points, start + length - 1) + shifts
        lo_bound, hi_bound = scalar_bound(xp, lower, q), scalar_bound(xp, upper, q)
        below, above = q < lo_bound, q > hi_bound
        clipped = xp.minimum(xp.maximum(q, lo_bound), hi_bound)
        lo, hi = xp.zeros_like(start), length
        for _ in range(self.steps):
            mid = (lo + hi) // 2
            value = xp.take(self.points, start + xp.minimum(mid, length - 1)) + shifts
            move_right = value < clipped
            lo = xp.where(move_right, mid + 1, lo)
            hi = xp.where(move_right, hi, mid)
        i0 = xp.maximum(xp.zeros_like(lo), xp.minimum(lo - 1, length - 2))
        i1 = xp.minimum(i0 + 1, length - 1)
        x0 = xp.take(self.points, start + i0) + shifts
        x1 = xp.take(self.points, start + i1) + shifts
        denominator = xp.where(length == 1, 1.0, x1 - x0)
        t = (clipped - x0) / denominator
        result = {}
        for key, values in self.fields.items():
            y0, y1 = xp.take(values, start + i0, axis=-1), xp.take(values, start + i1, axis=-1)
            if self.native_arithmetic:
                interpolated = y0 + ((y1 - y0) / denominator) * (clipped - x0)
                interpolated = xp.where(clipped == x0, y0, interpolated)
                interpolated = xp.where(clipped == x1, y1, interpolated)
                interpolated = xp.where(xp.isnan(interpolated) & (y0 == y1), y0, interpolated)
            else:
                interpolated = y0 + (y1 - y0) * t
            if self.has_singletons:
                # CuPy's native scalar-field interp propagates a NaN query even
                # for one sample; the existing generic singleton path does not.
                singleton = xp.where(xp.isnan(q), q, y0) if self.cupy_native_nan else y0
                interpolated = xp.where(length == 1, singleton, interpolated)
            left_value = xp.take(values, start, axis=-1) if left is None else left
            right_value = xp.take(values, start + length - 1, axis=-1) if right is None else right
            interpolated = xp.where(below, left_value, interpolated)
            result[key] = xp.where(above, right_value, interpolated)
        return result


def self_check(backend="numpy"):
    """Compare knots, boundaries, mixed precision, offsets, empties and ragged tables."""
    module = importlib.import_module(backend)
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    count = 0
    for query_dtype in (xp.float32, xp.float64):
        for rows in (False, True):
            points = [
                xp.asarray([0.1, 0.4, 0.9, 1.0 - 1e-10], dtype=xp.float64),
                xp.asarray([0.2, 0.8], dtype=xp.float64),
                xp.asarray([0.35], dtype=xp.float64),
            ]
            fields = [
                {"a": x * x + 0.3, "b": xp.stack([x + 1, x + 2]) if rows else x + 1}
                for x in points
            ]
            bank = TableBank(points, fields)
            for offset in (0.0, -0.12345, 10.1):
                queries, ids, expected = [], [], []
                for i, (x, values) in enumerate(zip(points, fields)):
                    q = xp.asarray(
                        np.r_[np.linspace(-0.5, 1.5, 51), 0.1, 1.0, np.nan, np.inf, -np.inf],
                        dtype=query_dtype,
                    )
                    q = q + offset
                    queries.append(q)
                    ids.append(xp.full(q.shape, i, dtype=xp.int64, device=bank.device))
                    expected.append(ArrayBackend1DInterpolator(x, values)(q, grid_offset=offset))
                q, owners = xp.concat(queries), xp.concat(ids)
                shifts = xp.full(q.shape, offset, dtype=xp.float64, device=bank.device)
                actual = bank(q, owners, shifts)
                for key in actual:
                    reference = xp.concat([e[key] for e in expected], axis=-1)
                    np.testing.assert_allclose(
                        to_numpy(actual[key]),
                        to_numpy(reference),
                        rtol=3e-6,
                        atol=1e-12,
                        equal_nan=True,
                    )
                empty = bank(q[:0], owners[:0], shifts[:0])
                assert all(a.shape[-1] == 0 for a in empty.values())
                count += 1
    print(f"{backend}: {count} grouped interpolation checks passed", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", default="numpy", choices=["numpy", "torch", "array_api_strict", "cupy"]
    )
    self_check(parser.parse_args().backend)

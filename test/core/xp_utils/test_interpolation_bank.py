"""Multi-table interpolation preserves independent prepared-table evaluations."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import (
    ArrayBackend1DInterpolator,
    ArrayBackend1DInterpolatorBank,
    to_numpy,
)


@pytest.fixture(params=["numpy", "torch", "array_api_strict", "cupy"])
def xp(request):
    """Exercise portable execution and installed native dispatch."""
    module = pytest.importorskip(request.param)
    if request.param == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return array_api_compat.array_namespace(module.asarray([0.0]))


@pytest.mark.parametrize("grid_dtype", ["float32", "float64"])
@pytest.mark.parametrize("query_dtype", ["float32", "float64"])
@pytest.mark.parametrize("kind", ["array", "rows", "mixed"])
@pytest.mark.parametrize("native", [None, False, True])
@pytest.mark.parametrize("boundary", [None, -3.25])
def test_matches_separate_tables(xp, grid_dtype, query_dtype, kind, native, boundary):
    """Cover ragged grids, query order, translations, dtypes, fields and boundaries."""
    grids = [
        xp.asarray(v, dtype=getattr(xp, grid_dtype))
        for v in ([0.1, 0.4, 0.9, 1.0 - 1e-10], [0.2, 0.8], [0.35])
    ]
    fields = []
    for x in grids:
        y = x * x + 0.3
        if kind == "rows":
            y = xp.stack([y, y + 1])
        elif kind == "mixed":
            y = {"a": y, "b": xp.astype(xp.stack([y + 1, y + 2]), xp.float32)}
        fields.append(y)
    bank = ArrayBackend1DInterpolatorBank(
        grids, fields, prefer_native=native, left=boundary, right=boundary
    )
    if array_api_compat.is_cupy_namespace(xp):
        fused = native is True and grid_dtype == "float64" and kind != "mixed"
        assert bank.implementation == ("cupy_fused" if fused else "array_api")
    expected, queries, owners, offsets = [], [], [], []
    for i, (grid, values) in enumerate(zip(grids, fields)):
        for delta in (0.0, -0.12345, 10.1):
            shift = xp.asarray(delta, dtype=grid.dtype)
            q = (
                xp.asarray(
                    np.r_[np.linspace(-0.5, 1.5, 15), to_numpy(grid), np.nan, np.inf, -np.inf],
                    dtype=getattr(xp, query_dtype),
                )
                + delta
            )
            reference_native = native if array_api_compat.is_numpy_namespace(xp) else False
            reference = ArrayBackend1DInterpolator(
                grid, values, prefer_native=reference_native, left=boundary, right=boundary
            )(q, grid_offset=shift)
            expected.append(reference if isinstance(reference, dict) else {None: reference})
            queries.append(q)
            owners.append(xp.full(q.shape, i, dtype=xp.int32))
            offsets.append(xp.full(q.shape, delta, dtype=grid.dtype))
    q, ids, shifts = xp.concat(queries), xp.concat(owners), xp.concat(offsets)
    # Interleave different tables/offsets; no sorted-owner assumption is allowed.
    order = xp.asarray(np.random.default_rng(27).permutation(q.shape[0]), dtype=xp.int64)
    q, ids, shifts = (xp.take(a, order) for a in (q, ids, shifts))
    actual = bank(q, ids, grid_offsets=shifts)
    actual = actual if isinstance(actual, dict) else {None: actual}
    for key, values in actual.items():
        reference = xp.take(xp.concat([e[key] for e in expected], axis=-1), order, axis=-1)
        assert values.dtype == reference.dtype
        assert array_api_compat.device(values) == array_api_compat.device(q)
        np.testing.assert_allclose(to_numpy(values), to_numpy(reference), rtol=3e-6, atol=1e-12)
    empty = bank(q[:0], ids[:0], grid_offsets=shifts[:0])
    for value in empty.values() if isinstance(empty, dict) else [empty]:
        assert value.shape[-1] == 0


@pytest.mark.parametrize("native", [False, True])
def test_singletons_nonfinite_values_and_scalar_shifts(xp, native):
    """Singletons retain value dtype; generic/fused paths retain NaN sample semantics."""
    points = [xp.asarray([0.0], dtype=xp.float64), xp.asarray([2.0], dtype=xp.float64)]
    values = [xp.asarray([3.0], dtype=xp.float32), xp.asarray([np.nan], dtype=xp.float32)]
    bank = ArrayBackend1DInterpolatorBank(points, values, prefer_native=native)
    q = xp.asarray([np.nan, 3.0, -np.inf, 1.0, np.inf], dtype=xp.float32)
    ids = xp.asarray([0, 1, 0, 1, 0], dtype=xp.int64)
    for offset in (None, 0.25, xp.asarray(0.25, dtype=xp.float64)):
        out = bank(q, ids, grid_offsets=offset)
        np.testing.assert_array_equal(to_numpy(out), [3.0, np.nan, 3.0, np.nan, 3.0])
        expected_dtype = (
            xp.float64 if native and array_api_compat.is_numpy_namespace(xp) else xp.float32
        )
        assert out.dtype == expected_dtype
    x = xp.asarray([0.0, 1.0, 2.0], dtype=xp.float64)
    for samples in ([1.0, np.nan, 3.0], [np.inf, np.inf, -np.inf]):
        y = xp.asarray(samples, dtype=xp.float64)
        bank = ArrayBackend1DInterpolatorBank([x], [y], prefer_native=False)
        q = xp.asarray([-np.inf, 0.0, 0.25, 1.0, 2.0, np.inf, np.nan], dtype=xp.float64)
        ids = xp.zeros(q.shape, dtype=xp.int64)
        reference = ArrayBackend1DInterpolator(x, y, prefer_native=False)(q)
        np.testing.assert_array_equal(to_numpy(bank(q, ids)), to_numpy(reference))
        if array_api_compat.is_cupy_namespace(xp):
            fused = ArrayBackend1DInterpolatorBank([x], [y], prefer_native=True)
            np.testing.assert_array_equal(to_numpy(fused(q, ids)), to_numpy(reference))


def test_rejects_invalid_tables_and_queries(xp):
    """Reject malformed tables and unsafe indices before backend gathers."""
    x = xp.asarray([0.0, 1.0], dtype=xp.float64)
    y = xp.asarray([1.0, 2.0], dtype=xp.float64)
    for points, values in (([], []), ([x], []), ([x[:0]], [y[:0]]), ([x, x], [y, y[:1]])):
        with pytest.raises(ValueError):
            ArrayBackend1DInterpolatorBank(points, values)
    for points in ([0.0, 0.0], [1.0, 0.0], [0.0, np.nan], [0.0, np.inf]):
        with pytest.raises(ValueError, match="finite and strictly"):
            ArrayBackend1DInterpolatorBank([xp.asarray(points)], [y])
    with pytest.raises(ValueError, match="fields"):
        ArrayBackend1DInterpolatorBank([x, x], [{"a": y}, {"b": y}])
    with pytest.raises(ValueError, match="dtypes"):
        ArrayBackend1DInterpolatorBank([x, x], [y, xp.astype(y, xp.float32)])
    bank = ArrayBackend1DInterpolatorBank([x], [y])
    q = xp.asarray([0.1, 0.5], dtype=xp.float32)
    for ids in ([-1, 0], [0, 1]):
        with pytest.raises(ValueError, match="out of bounds"):
            bank(q, xp.asarray(ids))
    with pytest.raises(TypeError, match="integer"):
        bank(q, xp.asarray([0.0, 0.0]))
    with pytest.raises(ValueError, match="equal-sized"):
        bank(q, xp.asarray([0], dtype=xp.int64))
    ids = xp.zeros(q.shape, dtype=xp.int64)
    for offset in (float("nan"), xp.asarray([0.0, float("inf")], dtype=xp.float64)):
        with pytest.raises(ValueError, match="finite"):
            bank(q, ids, grid_offsets=offset)
    with pytest.raises(ValueError, match="dtype"):
        bank(q, ids, grid_offsets=xp.asarray([0.0, 1.0], dtype=xp.float32))
    with pytest.raises(ValueError, match="nonfinite"):
        bank(q[:0], ids[:0], grid_offsets=xp.asarray(float("nan"), dtype=xp.float64))


def test_singleton_first_and_strided_reuse(xp):
    """Table order cannot truncate promoted results; strided queries are supported."""
    points = [xp.asarray([0.0], dtype=xp.float64), xp.asarray([0.0, 1.0], dtype=xp.float64)]
    fields = [
        {"a": xp.asarray([1.0], dtype=xp.float32), "b": xp.asarray([[2.0]], dtype=xp.float32)},
        {
            "a": xp.asarray([0.0, 1.0], dtype=xp.float32),
            "b": xp.asarray([[2.0, 4.0]], dtype=xp.float32),
        },
    ]
    for native in (None, False):
        bank = ArrayBackend1DInterpolatorBank(points, fields, prefer_native=native)
        q = xp.asarray([0.25, 99.0, 0.75, 99.0, -1.0, 99.0], dtype=xp.float32)[::2]
        for ids in ([1, 0, 1], [0, 1, 0]):
            actual = bank(q, xp.asarray(ids, dtype=xp.int64))
            for key in actual:
                expected = []
                for j, i in enumerate(ids):
                    result = ArrayBackend1DInterpolator(points[i], fields[i], prefer_native=False)(
                        q[j : j + 1]
                    )
                    expected.append(result[key])
                reference = xp.concat(expected, axis=-1)
                assert actual[key].dtype == reference.dtype
                np.testing.assert_allclose(to_numpy(actual[key]), to_numpy(reference), rtol=1e-12)

"""CuPy specialization for float64 table banks; imported only for CuPy calls."""

from functools import lru_cache

from ._backends import loaded_backend


@lru_cache(maxsize=1)
def _kernels():
    cp = loaded_backend("cupy")
    search = cp.ElementwiseKernel(
        "Q q, I owner, float64 shift, raw float64 points, raw int64 starts, raw int64 lengths",
        "int64 i0, int64 i1, float64 weight, int8 boundary",
        r"""
        const long long start = starts[owner], n = lengths[owner];
        const double lower = points[start] + shift;
        const double upper = points[start + n - 1] + shift;
        boundary = q < lower ? -1 : (q > upper ? 1 : 0);
        const double clipped = isnan((double)q) ? (double)q :
                               fmin(fmax((double)q, lower), upper);
        long long lo = 0, hi = n;
        while (lo < hi) {
            const long long mid = (lo + hi) / 2;
            if (points[start + mid] + shift < clipped) lo = mid + 1;
            else hi = mid;
        }
        i0 = start + max(0LL, min(lo - 1, n - 2));
        i1 = min(i0 + 1, start + n - 1);
        const double x0 = points[i0] + shift, x1 = points[i1] + shift;
        weight = (clipped - x0) / (n == 1 ? 1.0 : x1 - x0);
        """,
        "pyradplan_bank_search_f64",
        options=("--fmad=false",),
    )
    evaluate = cp.ElementwiseKernel(
        "raw float64 values, raw int64 i0, raw int64 i1, raw float64 weight, "
        "raw int8 boundary, int64 queries, int64 samples, "
        "bool use_left, float64 left, bool use_right, float64 right",
        "float64 out",
        r"""
        const long long q = i % queries, row = i / queries;
        const double y0 = values[row * samples + i0[q]];
        const double y1 = values[row * samples + i1[q]];
        if (boundary[q] < 0) out = use_left ? left : y0;
        else if (boundary[q] > 0) out = use_right ? right : y1;
        else out = i0[q] == i1[q] ? y0 : y0 + (y1 - y0) * weight[q];
        """,
        "pyradplan_bank_evaluate_f64",
        options=("--fmad=false",),
    )
    return search, evaluate


def evaluate_bank(bank, queries, indices, shifts):
    """Share one interval search across all fields and preserve field row layouts."""
    with bank._device:
        return _evaluate_bank(bank, queries, indices, shifts)


def _evaluate_bank(bank, queries, indices, shifts):
    cp = loaded_backend("cupy")
    if queries.size == 0:
        return {
            key: cp.empty(value.shape[:-1] + queries.shape, dtype=cp.float64)
            for key, value in bank._fields.items()
        }
    search, evaluate = _kernels()
    i0, i1, weight, boundary = search(
        queries, indices, shifts, bank._points, bank._starts, bank._lengths
    )
    result = {}
    for key, values in bank._fields.items():
        out = cp.empty(values.shape[:-1] + queries.shape, dtype=cp.float64)
        if out.size:
            evaluate(
                values,
                i0,
                i1,
                weight,
                boundary,
                queries.size,
                bank._points.size,
                bank._left is not None,
                0.0 if bank._left is None else bank._left,
                bank._right is not None,
                0.0 if bank._right is None else bank._right,
                out,
            )
        result[key] = out
    return result

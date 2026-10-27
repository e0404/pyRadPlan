"""Compare three scalar queries with a shared trailing value dimension.

Use final production code to reproduce the contiguous layout. Older result files
label the unpacked and rejected transposed-copy experiments explicitly; those
intermediate implementations are not production options.
"""

import argparse
import json
import time
from pathlib import Path

import array_api_compat
import numpy as np

from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator, to_numpy


def main():
    """Time cached evaluation, including device synchronization and clipping checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend == "cupy":
        import cupy as module  # noqa: PLC0415
    else:
        module = np
    xp = array_api_compat.array_namespace(module.asarray([0.0]))

    def sync():
        if args.backend == "cupy":
            module.cuda.get_current_stream().synchronize()

    rng = np.random.default_rng(1234)
    results = []
    for dtype_name in ("float32", "float64"):
        dtype = getattr(xp, dtype_name)
        p = xp.asarray(np.linspace(-100, 100, 800), dtype=dtype)
        values = xp.asarray(rng.random((800, 800, 3)), dtype=dtype)
        scalar = [
            ArrayBackendGridInterpolator((p, p), values[:, :, i], bounds_error=True)
            for i in range(3)
        ]
        native = ArrayBackendGridInterpolator(
            (p, p), values, bounds_error=True, prefer_native=True
        )
        generic = ArrayBackendGridInterpolator(
            (p, p), values, bounds_error=True, prefer_native=False
        )
        for count in (10000, 50000, 150000):
            q = xp.asarray(rng.uniform(-99, 99, (count, 2)), dtype=dtype)
            reference = np.column_stack([to_numpy(f(q)) for f in scalar])
            for name, call in [
                ("three_native", lambda: xp.stack([f(q) for f in scalar], axis=1)),
                ("shared_native", lambda: native(q)),
                ("shared_generic", lambda: generic(q)),
            ]:
                result = call()
                np.testing.assert_allclose(to_numpy(result), reference, rtol=3e-6, atol=1e-7)
                sync()
                samples = []
                for _ in range(7):
                    start = time.perf_counter()
                    for _ in range(10):
                        call()
                    sync()
                    samples.append((time.perf_counter() - start) / 10)
                results.append(
                    dict(
                        dtype=dtype_name,
                        queries=count,
                        path=name,
                        median_s=float(np.median(samples)),
                        max_abs=float(np.max(np.abs(to_numpy(result) - reference))),
                    )
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

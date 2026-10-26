"""Compare cached native/generic interpolation on captured photon inputs.

Run separately for each backend, e.g.::

    python benchmark/profile_interpnd_selection.py --backend cupy

The dose run only captures representative inputs; its timings are not comparison
results. Evaluation excludes construction, alternates order, and synchronizes GPU
work. Both bounds policies are measured on identical, in-bounds queries.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np

import profile_pb_audit as audit
from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator, to_numpy


def main():
    """Capture one kernel's queries, then compare prepared evaluators."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--out", default="benchmark/pb_audit_results/cached_native_generic")
    args = parser.parse_args()
    captured = {}

    def capture_factory(points, values, **kwargs):
        interpolator = ArrayBackendGridInterpolator(points, values, **kwargs)
        if captured:
            return interpolator
        captured.update(points=points, values=values, queries=[])

        def evaluate(xi):
            captured["queries"].append(xi)
            return interpolator(xi)

        return evaluate

    original = audit.SVD._get_kernel_interpolators
    original_argv = sys.argv
    original.__globals__["_selection_capture_factory"] = capture_factory
    audit.replace_method(
        audit.SVD,
        "_get_kernel_interpolators",
        [("ArrayBackendGridInterpolator(", "_selection_capture_factory(")],
    )
    try:
        sys.argv = [
            "profile_pb_audit.py",
            "--backend",
            args.backend,
            "--mode",
            "photons",
            "--runs",
            "1",
            "--out",
            str(Path(args.out) / "capture"),
        ]
        audit.main()
    finally:
        audit.SVD._get_kernel_interpolators = original
        original.__globals__.pop("_selection_capture_factory")
        sys.argv = original_argv

    queries = sorted(captured["queries"], key=lambda q: q.shape[0])
    rows = []
    for bounds_error in (True, False):
        evaluators = {
            name: ArrayBackendGridInterpolator(
                captured["points"],
                captured["values"],
                bounds_error=bounds_error,
                prefer_native=prefer_native,
            )
            for name, prefer_native in (("native", True), ("generic", False))
        }
        for label, xi in zip(
            ("min", "median", "max"), (queries[0], queries[len(queries) // 2], queries[-1])
        ):
            outputs = {name: to_numpy(fn(xi)) for name, fn in evaluators.items()}
            np.testing.assert_allclose(
                outputs["native"], outputs["generic"], rtol=1e-12, atol=1e-12
            )
            for fn in evaluators.values():
                for _ in range(3):
                    fn(xi)
            audit.sync(args.backend)
            timings = {name: [] for name in evaluators}
            for i in range(args.rounds):
                order = ("native", "generic") if i % 2 == 0 else ("generic", "native")
                for name in order:
                    audit.sync(args.backend)
                    start = time.perf_counter()
                    evaluators[name](xi)
                    audit.sync(args.backend)
                    timings[name].append((time.perf_counter() - start) * 1000)
            medians = {name: statistics.median(times) for name, times in timings.items()}
            row = dict(
                bounds_error=bounds_error,
                query_sample=label,
                query_shape=list(xi.shape),
                query_dtype=str(xi.dtype),
                milliseconds=timings,
                median_ms=medians,
                generic_over_native=medians["generic"] / medians["native"],
                max_abs=float(np.max(np.abs(outputs["native"] - outputs["generic"]))),
            )
            rows.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "milliseconds"}), flush=True)
    result = dict(
        backend=args.backend,
        rounds=args.rounds,
        grid_shape=list(captured["values"].shape),
        values_dtype=str(captured["values"].dtype),
        axes_dtype=[str(p.dtype) for p in captured["points"]],
        captured_queries=len(queries),
        rows=rows,
    )
    (Path(args.out) / f"{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

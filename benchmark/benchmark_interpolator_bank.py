"""Synchronized preparation/evaluation timings for prepared multi-table interpolation."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from pyRadPlan.core.xp_utils import (
    ArrayBackend1DInterpolator,
    ArrayBackend1DInterpolatorBank,
    choose_array_api_namespace,
    to_numpy,
)


def main():
    """Compare native prepared segments, portable bank and backend specialization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "torch", "cupy"], default="cupy")
    parser.add_argument("--queries", nargs="+", type=int, default=[1024, 16384, 65536])
    parser.add_argument("--segments", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--out", default="benchmark/pb_audit_results/interpolator_bank")
    args = parser.parse_args()
    xp = choose_array_api_namespace(args.backend)
    sync = xp.cuda.get_current_stream().synchronize if args.backend == "cupy" else lambda: None
    grids, fields = [], []
    for table in range(18):
        x = xp.asarray(np.linspace(0, 180 + table * 7, 184 + table * 9), dtype=xp.float64)
        grids.append(x)
        fields.append({"idd": x * x * 0.01, "sigma": x * 0.02 + 2, "let": 4 - x * 0.002})
    preparation, banks = {}, {}
    for label, native in (("portable", False), ("specialized", True)):
        sync()
        start = time.perf_counter()
        banks[label] = ArrayBackend1DInterpolatorBank(grids, fields, prefer_native=native)
        sync()
        preparation[label] = time.perf_counter() - start
    sync()
    start = time.perf_counter()
    prepared = [ArrayBackend1DInterpolator(x, f) for x, f in zip(grids, fields)]
    sync()
    preparation["segments"] = time.perf_counter() - start
    results = {}
    rng = np.random.default_rng(29)
    for n in args.queries:
        cuts = np.linspace(0, n, args.segments + 1).astype(np.int64)
        table_ids = np.arange(args.segments) % len(grids)
        shifts = (np.arange(args.segments) % 7) * 0.11
        q = xp.asarray(rng.uniform(-2, 320, n).astype(np.float32))
        owner = xp.asarray(np.repeat(table_ids, np.diff(cuts)), dtype=xp.int64)
        offsets = xp.asarray(np.repeat(shifts, np.diff(cuts)), dtype=xp.float64)

        def segments():
            parts = [
                prepared[int(table_ids[i])](
                    q[int(cuts[i]) : int(cuts[i + 1])], grid_offset=float(shifts[i])
                )
                for i in range(args.segments)
            ]
            return {key: xp.concat([p[key] for p in parts]) for key in fields[0]}

        calls = {"segments": segments}
        for label, bank in banks.items():
            calls[label] = lambda bank=bank: bank(q, owner, grid_offsets=offsets)
        first_call, samples = {}, {label: [] for label in calls}
        reference = segments()
        for label, call in calls.items():
            sync()
            start = time.perf_counter()
            out = call()
            sync()
            first_call[label] = time.perf_counter() - start
            for key in reference:
                np.testing.assert_allclose(
                    to_numpy(out[key]), to_numpy(reference[key]), rtol=3e-6, atol=1e-12
                )
        for repeat in range(args.repeats):
            for label in list(calls)[:: 1 if repeat % 2 == 0 else -1]:
                sync()
                start = time.perf_counter()
                calls[label]()
                sync()
                samples[label].append(time.perf_counter() - start)
        results[n] = dict(first_call_s=first_call, samples_s=samples)
        print(
            n,
            {k: round(float(np.median(v)) * 1000, 3) for k, v in samples.items()},
            "ms",
            flush=True,
        )
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"micro_{args.backend}.json").write_text(
        json.dumps(dict(config=vars(args), preparation_s=preparation, results=results), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Compare boolean, integer-bracket, and Array API take on real ray inputs.

Captures three beamlets from the TG119 proton fixture. Includes nonzero in every
timed integer/take evaluation; transfers and capture instrumentation are excluded.
Run after (not alongside) the full-dose benchmark to avoid resource contention.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import array_api_compat
import numpy as np

import profile_pb_audit as audit
from pyRadPlan.core.xp_utils import to_numpy


def gather(xp, mask, arrays, method):
    """Select the same rows, paying compaction once or once per array."""
    if method == "boolean":
        return [a[mask] for a in arrays]
    selected = xp.nonzero(mask)[0]
    if method == "integer":
        return [a[selected] for a in arrays]
    return [xp.take(a, selected, axis=0) for a in arrays]


def check_strict():
    """Verify portable row gathering, including multidimensional optional fields."""
    import array_api_strict as xp  # noqa: PLC0415

    checks = 0
    for version in ("2022.12", "2023.12", "2024.12", "2025.12"):
        with xp.ArrayAPIStrictFlags(api_version=version):
            for values in ([], [False] * 6, [True] * 6, [True, False, True, False, False, True]):
                mask = xp.asarray(values, dtype=xp.bool)
                n = len(values)
                arrays = [
                    xp.arange(n, dtype=xp.int64),
                    xp.arange(n, dtype=xp.float32),
                    xp.reshape(xp.arange(n * 2, dtype=xp.float64), (n, 2)),
                ]
                expected = gather(xp, mask, arrays, "boolean")
                with xp.ArrayAPIStrictFlags(boolean_indexing=False):
                    actual = gather(xp, mask, arrays, "take")
                for a, b in zip(expected, actual):
                    assert a.shape == b.shape and a.dtype == b.dtype
                    assert bool(xp.all(a == b))
                checks += 1
    return checks


def main():  # noqa: PLR0915
    """Capture representative inputs and time just their indexing operations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--out", default="benchmark/pb_audit_results/particle_indexing/micro")
    args = parser.parse_args()
    strict_checks = check_strict()
    captured = []
    calls = 0

    def capture(mask, ray):
        nonlocal calls
        if calls in (0, 450, 907):
            fields = [k for k in ("ix", "radial_dist_sq", "rad_depths", "lat_dists") if k in ray]
            captured.append((calls, mask, [ray[k] for k in fields], fields))
        calls += 1

    original = audit.PP._get_bixel_indices_on_ray
    original_argv = sys.argv
    original.__globals__["_indexing_capture"] = capture
    audit.replace_method(
        audit.PP,
        "_get_bixel_indices_on_ray",
        [("# Apply mask", "_indexing_capture(curr_ix, curr_ray)\n    # Apply mask")],
    )
    try:
        sys.argv = [
            "profile_pb_audit.py",
            "--backend",
            args.backend,
            "--mode",
            "protons",
            "--runs",
            "1",
            "--out",
            str(Path(args.out) / "capture"),
        ]
        audit.main()
    finally:
        audit.PP._get_bixel_indices_on_ray = original
        original.__globals__.pop("_indexing_capture")
        sys.argv = original_argv

    methods = ["boolean", "integer", "take"]
    rows = []
    for call, mask, arrays, fields in captured:
        xp = array_api_compat.array_namespace(mask)
        expected = [to_numpy(a) for a in gather(xp, mask, arrays, "boolean")]
        for method in methods:
            actual = gather(xp, mask, arrays, method)
            for a, b in zip(expected, actual):
                np.testing.assert_array_equal(a, to_numpy(b))
            for _ in range(3):
                gather(xp, mask, arrays, method)
        times = {m: [] for m in methods}
        for i in range(args.rounds):
            order = methods[i % 3 :] + methods[: i % 3]
            for method in order:
                audit.sync(args.backend)
                start = time.perf_counter()
                gather(xp, mask, arrays, method)
                audit.sync(args.backend)
                times[method].append((time.perf_counter() - start) * 1000)
        row = dict(
            beamlet=call,
            candidates=mask.shape[0],
            selected=expected[0].shape[0],
            fields=fields,
            shapes=[list(a.shape) for a in arrays],
            milliseconds=times,
            median_ms={k: statistics.median(v) for k, v in times.items()},
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "milliseconds"}), flush=True)
    result = dict(backend=args.backend, rounds=args.rounds, strict_checks=strict_checks, rows=rows)
    (Path(args.out) / f"{args.backend}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Profile identical pipeline workloads with legacy and shared batch scheduling.

Dose options are forwarded to benchmark_particle_pipeline; this profiler compares
the bank and bank_hooks influence-calculation variants only.
First calls (scalar reference and each variant's correctness warm-up) are excluded.
Profiles perturb GPU submission timing: use the unprofiled harness for speed claims.
"""

import argparse
from collections import defaultdict
import cProfile
import json
from pathlib import Path
import pstats
import sys
from unittest.mock import patch

import benchmark_particle_pipeline as pipeline


def main():
    """Aggregate warmed cProfile call counts, self times and cumulative times."""
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument(
        "--profile-out", default="benchmark/pb_audit_results/hook_overhead/profile"
    )
    parser.add_argument(
        "--variants", nargs="+", default=["bank", "bank_hooks"], choices=["bank", "bank_hooks"]
    )
    parser.add_argument("--forward", action="store_true")
    args, forwarded = parser.parse_known_args()
    if args.forward or len(args.variants) != 2 or len(set(args.variants)) != 2:
        parser.error("Profile both bank and bank_hooks influence variants exactly once")
    forwarded += ["--variants", *args.variants]
    original = pipeline.Hong.calc_dose_influence
    seen = defaultdict(int)
    records = defaultdict(list)

    def measured(engine, *positional, **kwargs):
        label = type(engine).__name__
        seen[label] += 1
        if type(engine) is pipeline.Hong or seen[label] == 1:
            return original(engine, *positional, **kwargs)
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            result = original(engine, *positional, **kwargs)
            pipeline.sync("cupy" if "cupy" in engine.xp.__name__ else "numpy")
        finally:
            profiler.disable()
        records[label].append(pstats.Stats(profiler).stats)
        return result

    with patch.object(sys, "argv", [pipeline.__file__, *forwarded]):
        with patch.object(pipeline.Hong, "calc_dose_influence", measured):
            pipeline.main()

    output = {}
    root = Path(__file__).resolve().parents[1]
    for label, runs in records.items():
        combined = defaultdict(lambda: [0, 0, 0.0, 0.0])
        for run in runs:
            for key, (primitive, calls, own, cumulative, _) in run.items():
                for i, value in enumerate((primitive, calls, own, cumulative)):
                    combined[key][i] += value / len(runs)
        rows = []
        for (filename, line, function), values in combined.items():
            try:
                display_path = Path(filename).relative_to(root).as_posix()
            except ValueError:
                display_path = filename.replace(str(root), ".")
            rows.append(
                dict(
                    file=display_path,
                    line=line,
                    function=function,
                    primitive_calls=values[0],
                    calls=values[1],
                    self_s=values[2],
                    cumulative_s=values[3],
                )
            )
        # Keep all project/CuPy rows and expensive external routines. Tiny stdlib
        # helpers add substantial report size without explaining the comparison.
        retained = [
            row
            for row in rows
            if row["file"].startswith(("pyRadPlan/", "benchmark/"))
            or "cupy" in row["function"]
            or row["self_s"] >= 0.005
            or row["cumulative_s"] >= 0.01
        ]
        output[label] = dict(
            runs=len(runs),
            omitted_functions=len(rows) - len(retained),
            profile_self_s=sum(row["self_s"] for row in rows),
            functions=sorted(retained, key=lambda r: -r["self_s"]),
        )
        print(
            label,
            [(r["function"], round(r["self_s"], 4)) for r in output[label]["functions"][:12]],
        )
    out = Path(args.profile_out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "profile.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()

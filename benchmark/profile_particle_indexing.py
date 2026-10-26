"""Measure indexing alone, without hoisting particle kernel conversion.

Runs paired full-dose comparisons in isolated processes, sequentially to avoid
CPU/GPU contention. The compact_take variant only changes the mask application
in _get_bixel_indices_on_ray. Dose values and sparse structure must match exactly.
"""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(out):
    """Record medians and exact parity, then reject any numerical mismatch."""
    rows = []
    for path in sorted(out.glob("*_paired.json")):
        paired = json.loads(path.read_text())
        metadata = json.loads(path.with_name(path.name.replace("_paired", "")).read_text())
        ratios = [b / v for b, v in zip(paired["baseline"], paired["variant"])]
        rows.append(
            dict(
                backend=metadata["args"]["backend"],
                grid_mm=metadata["args"]["grid"],
                selected_voxels=metadata["effective"]["selected_voxels"],
                beamlets=metadata["beamlets"],
                nnz=metadata["output"]["nnz"],
                baseline_seconds=statistics.median(paired["baseline"]),
                take_seconds=statistics.median(paired["variant"]),
                median_pair_speedup=statistics.median(ratios),
                pair_speedup_range=[min(ratios), max(ratios)],
                exact_pairs=sum(
                    c["max_abs"] == 0 and c["identical_indices"] for c in paired["comparisons"]
                ),
                pairs=len(ratios),
                source=path.name,
            )
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2), flush=True)
    assert rows and all(r["exact_pairs"] == r["pairs"] for r in rows)


def main():
    """Run the requested backend/resolution matrix and summarize each case."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["numpy", "cupy"])
    parser.add_argument("--grids", nargs="+", type=float, default=[8, 5, 3])
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--out", default="benchmark/pb_audit_results/particle_indexing")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for backend in args.backends:
        for grid in args.grids:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("profile_pb_audit.py")),
                    "--backend",
                    backend,
                    "--mode",
                    "protons",
                    "--grid",
                    str(grid),
                    "--variant",
                    "compact_take",
                    "--runs",
                    "1",
                    "--paired",
                    "--pairs",
                    str(args.pairs),
                    "--out",
                    str(out),
                ],
                check=True,
            )
            summarize(out)


if __name__ == "__main__":
    main()

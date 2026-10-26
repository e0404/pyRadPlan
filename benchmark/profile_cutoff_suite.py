"""Full-dose cutoff comparisons against the frozen pre-refactor implementation."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys


def summarize(out):
    """Record paired speedups and exact physical-dose/sparse-structure parity."""
    rows = []
    for path in sorted(out.glob("*_paired.json")):
        paired = json.loads(path.read_text())
        meta = json.loads(path.with_name(path.name.replace("_paired", "")).read_text())
        # baseline = current backend-native cutoff; variant = frozen legacy or NumPy cutoff override.
        current = paired["baseline"]
        variant = paired["variant"]
        rows.append(
            dict(
                mode=meta["args"]["mode"],
                dose_backend=meta["args"]["backend"],
                variant=meta["args"]["variant"],
                current_median=statistics.median(current),
                variant_median=statistics.median(variant),
                current_speedup=statistics.median(v / c for v, c in zip(variant, current)),
                exact_pairs=sum(
                    c["max_abs"] == 0 and c["identical_indices"] for c in paired["comparisons"]
                ),
                pairs=len(current),
                source=path.name,
            )
        )
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    assert rows and all(row["exact_pairs"] == row["pairs"] for row in rows), rows
    print(f"Validated {sum(row['exact_pairs'] for row in rows)} exact pairs.", flush=True)


def main():
    """Run cases in fresh, sequential processes to avoid timing contention."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", default="benchmark/pb_audit_results/cutoff_native/full_dose")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("protons", "VHEE", "carbon"):
        for backend in ("numpy", "cupy"):
            variants = (
                ("legacy_cutoff", "numpy_cutoff") if backend == "cupy" else ("legacy_cutoff",)
            )
            for variant in variants:
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("profile_pb_audit.py")),
                        "--backend",
                        backend,
                        "--mode",
                        mode,
                        "--variant",
                        variant,
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

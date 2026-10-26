"""Compare production photon interpolator reuse with the legacy evaluation path.

Run with the repository venv from the repository root. Results use ``baseline``
for current production, and ``variant`` for the selected legacy or direct-native reference.
Each case includes six alternating pairs and an output comparison per pair.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys

from profile_pb_audit import main
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine


def run():
    """Validate ordinary influence, sampled forward dose, and shaped-field dose."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", choices=["legacy", "native"], default="legacy")
    parser.add_argument("--out", default="benchmark/pb_audit_results/array_backend_interpolator")
    parser.add_argument("--influence-only", action="store_true")
    args = parser.parse_args()
    original = PhotonPencilBeamSVDEngine._get_kernel_interpolators
    out = Path(args.out)
    for backend in ("numpy", "cupy"):
        for mode, flags in (
            ("photons", []),
            ("photons", ["--sampling", "--forward"]),
            ("field", ["--forward"]),
        ):
            if args.influence_only and flags:
                continue
            sys.argv = [
                __file__,
                "--backend",
                backend,
                "--mode",
                mode,
                "--variant",
                args.reference + "_interpolators",
                "--paired",
                "--runs",
                "2",
                "--out",
                str(out),
                *flags,
            ]
            try:
                main()
            finally:
                PhotonPencilBeamSVDEngine._get_kernel_interpolators = original

    summarize_results(out)


def summarize_results(out):
    """Require exact output parity and summarize the paired measurements."""
    results = []
    for path in sorted(out.glob("*_paired.json")):
        paired = json.loads(path.read_text())
        exact = all(
            c["max_abs"] == 0 and c.get("identical_indices", True) for c in paired["comparisons"]
        )
        current = statistics.median(paired["baseline"])
        legacy = statistics.median(paired["variant"])
        results.append(
            dict(
                case=path.stem,
                current_seconds=current,
                reference_seconds=legacy,
                speedup=legacy / current,
                exact_output=exact,
                max_abs=max(c["max_abs"] for c in paired["comparisons"]),
                max_abs_over_peak=max(c["max_abs_over_peak"] for c in paired["comparisons"]),
                pairs=len(paired["comparisons"]),
            )
        )
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    assert all(r["exact_output"] for r in results), "Dose mismatch; see summary and paired records"


if __name__ == "__main__":
    run()

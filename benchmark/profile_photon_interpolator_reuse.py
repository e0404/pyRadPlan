"""Compare production photon interpolator reuse with the legacy evaluation path.

Run with the repository venv from the repository root. Results use ``baseline``
for current production (reused interpolators), and ``variant`` for legacy.
Each case includes six alternating pairs and an output comparison per pair.
"""

import json
from pathlib import Path
import statistics
import sys

from profile_pb_audit import main
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine


def run():
    """Validate ordinary influence, sampled forward dose, and shaped-field dose."""
    original = PhotonPencilBeamSVDEngine._get_kernel_interpolators
    out = Path("benchmark/pb_audit_results/interpolator_reuse")
    for backend in ("numpy", "cupy"):
        for mode, flags in (
            ("photons", []),
            ("photons", ["--sampling", "--forward"]),
            ("field", ["--forward"]),
        ):
            sys.argv = [
                __file__,
                "--backend",
                backend,
                "--mode",
                mode,
                "--variant",
                "legacy_interpolators",
                "--paired",
                "--runs",
                "2",
                "--out",
                "benchmark/pb_audit_results/interpolator_reuse",
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
        assert all(c["max_abs"] == 0 for c in paired["comparisons"]), path
        assert all(c.get("identical_indices", True) for c in paired["comparisons"]), path
        current = statistics.median(paired["baseline"])
        legacy = statistics.median(paired["variant"])
        results.append(
            dict(
                case=path.stem,
                current_seconds=current,
                legacy_seconds=legacy,
                speedup=legacy / current,
                exact_output=True,
                pairs=len(paired["comparisons"]),
            )
        )
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()

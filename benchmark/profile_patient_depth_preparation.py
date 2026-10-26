"""Paired full-dose and isolated tissue benchmarks for backend depth preparation."""

import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

from pyRadPlan import load_tg119
from profile_patient_depth_reference import legacy_tissue_data


def main():
    """Run workloads sequentially and preserve raw comparisons."""
    out = Path("benchmark/pb_audit_results/patient_depth")
    out.mkdir(exist_ok=True)
    ct, cst = load_tg119()
    grid = ct.grid.resample(dict(x=5.0, y=5.0, z=5.0))
    times = {"current": [], "legacy": []}
    for i in range(6):
        outputs = {}
        for name in list(times) if i % 2 else list(reversed(times)):
            start = time.perf_counter()
            outputs[name] = (
                cst.get_tissue_data(grid) if name == "current" else legacy_tissue_data(cst, grid)
            )
            if i:
                times[name].append(time.perf_counter() - start)
        for a, b in zip(outputs["current"][:3], outputs["legacy"][:3]):
            np.testing.assert_array_equal(a, b)
    (out / "tissue.json").write_text(json.dumps(times, indent=2) + "\n")
    for mode in ("photons", "protons", "VHEE"):
        for backend in ("numpy", "cupy"):
            for trace_ct in (False, True):
                command = [
                    sys.executable,
                    str(Path(__file__).with_name("profile_pb_audit.py")),
                    "--backend",
                    backend,
                    "--mode",
                    mode,
                    "--variant",
                    "legacy_patient_depth",
                    "--paired",
                    "--pairs",
                    "3",
                    "--runs",
                    "1",
                    "--out",
                    str(out),
                ]
                if trace_ct:
                    command.append("--trace-on-ct-grid")
                subprocess.run(command, check=True)
    rows = []
    for path in sorted(out.glob("*_paired.json")):
        pair = json.loads(path.read_text())
        meta = json.loads(path.with_name(path.name.replace("_paired", "")).read_text())
        max_error = max(c["max_abs_over_peak"] for c in pair["comparisons"])
        same_indices = all(c["identical_indices"] for c in pair["comparisons"])
        rows.append(
            dict(
                mode=meta["args"]["mode"],
                backend=meta["args"]["backend"],
                trace_on_ct=meta["args"]["trace_on_ct_grid"],
                current=statistics.median(pair["baseline"]),
                legacy=statistics.median(pair["variant"]),
                speedup=statistics.median(
                    b / a for a, b in zip(pair["baseline"], pair["variant"])
                ),
                max_peak_relative_error=max_error,
                identical_indices=same_indices,
            )
        )
        assert same_indices and max_error <= 1e-5, path
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()

"""Compare production voxel preparation with the frozen 4e747717 implementation.

Run sequentially: stage timings exclude full dose evaluation, and tracemalloc
memory measurements are separate from timings. Traced peaks include Python and
NumPy allocations, not SimpleITK's native allocations or process RSS.
"""

import argparse
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import tracemalloc

import numpy as np

from profile_voxel_preparation_reference import legacy_voxel_selection
from pyRadPlan import load_tg119
from pyRadPlan.dose.engines import DoseEngineBase, ParticleHongPencilBeamEngine


def microbench(rounds, grids):
    """Compare complete selection/resampling/coordinate preparation at several grids."""
    ct, cst = load_tg119()
    methods = dict(legacy=legacy_voxel_selection, current=DoseEngineBase._init_voxel_selection)
    outputs = (
        "_vct_grid",
        "_vdose_grid",
        "_vct_grid_scen_ix",
        "_vdose_grid_scen_ix",
        "_vox_world_coords",
        "_vox_world_coords_dose_grid",
        "_vct_grid_mask",
        "_vdose_grid_mask",
    )
    subset = cst.vois[0].indices_numpy[::20]
    selections = {"automatic": None, "custom": np.concatenate((subset[::-1], subset[:100]))}
    rows = []
    for resolution in grids:
        for kind, selection in selections.items():

            def run(name):
                engine = object.__new__(ParticleHongPencilBeamEngine)
                engine._ct_grid = ct.grid
                engine.dose_grid = ct.grid.resample(dict.fromkeys(("x", "y", "z"), resolution))
                engine.voxel_sub_ix = None if selection is None else selection.copy()
                methods[name](engine, ct, cst)
                return engine

            expected = run("legacy")
            timings = {name: [] for name in methods}
            for i in range(rounds + 1):
                for name in list(methods) if i % 2 else list(reversed(methods)):
                    start = time.perf_counter()
                    actual = run(name)
                    elapsed = time.perf_counter() - start
                    for field in outputs:
                        np.testing.assert_array_equal(
                            getattr(actual, field), getattr(expected, field)
                        )
                    if i:
                        timings[name].append(elapsed)
            peaks = {}
            for name in methods:
                tracemalloc.start()
                actual = run(name)
                _, peaks[name] = tracemalloc.get_traced_memory()
                tracemalloc.stop()
            row = dict(
                resolution=resolution,
                selection=kind,
                ct_voxels=ct.grid.num_voxels,
                dose_voxels=actual.dose_grid.num_voxels,
                selected_ct=actual._vct_grid.size,
                selected_dose=actual._vdose_grid.size,
                seconds=timings,
                medians={n: statistics.median(t) for n, t in timings.items()},
                traced_peak_bytes=peaks,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def summarize(folder):
    """Report paired current-versus-original full-dose timings and exactness."""
    rows = []
    for path in sorted(folder.glob("*_paired.json")):
        pairs = json.loads(path.read_text())
        meta = json.loads(path.with_name(path.name.replace("_paired", "")).read_text())
        assert all(c["max_abs"] == 0 and c["identical_indices"] for c in pairs["comparisons"])
        rows.append(
            dict(
                mode=meta["args"]["mode"],
                backend=meta["args"]["backend"],
                resolution=meta["args"]["grid"],
                current_median=statistics.median(pairs["baseline"]),
                legacy_median=statistics.median(pairs["variant"]),
                paired_speedup=statistics.median(
                    v / b for v, b in zip(pairs["variant"], pairs["baseline"])
                ),
                exact_pairs=len(pairs["comparisons"]),
                source=path.name,
            )
        )
    (folder / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main():
    """Measure host preparation, then sequential CPU/GPU full-dose comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--grids", nargs="+", type=float, default=[3, 5, 10])
    parser.add_argument("--out", default="benchmark/pb_audit_results/voxel_preparation")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    micro = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        args=vars(args),
        rows=microbench(args.rounds, args.grids),
    )
    (out / "preparation.json").write_text(json.dumps(micro, indent=2) + "\n")
    full = out / "full_dose"
    full.mkdir(exist_ok=True)
    for mode in ("photons", "protons", "VHEE"):
        for grid in args.grids if mode == "photons" else [5]:
            for backend in ("numpy", "cupy"):
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("profile_pb_audit.py")),
                        "--mode",
                        mode,
                        "--backend",
                        backend,
                        "--grid",
                        str(grid),
                        "--variant",
                        "legacy_voxel_selection",
                        "--runs",
                        "1",
                        "--paired",
                        "--pairs",
                        str(args.pairs),
                        "--out",
                        str(full),
                    ],
                    check=True,
                )
                records = summarize(full)
                print(
                    f"Validated {sum(r['exact_pairs'] for r in records)} exact pairs.", flush=True
                )


if __name__ == "__main__":
    main()

"""Compare end-to-end TG119 influence-matrix calculation between git revisions.

Each revision's ``pyRadPlan`` package is exported with ``git archive`` and imported
in a fresh process with the installed dependencies; no checkout is modified.
The pseudo-revision ``worktree`` imports the current checkout, uncommitted changes included.
Revisions alternate within each round. Outputs are compared exactly: dose values,
row indices and column pointers of the sorted CSC matrix (index dtype ignored).

The workloads mirror the GUI defaults: 3 mm dose grid, 5 mm bixels; photons use
five fields with dose sampling (fixed seed), protons three fields, physical dose.

    python benchmark/benchmark_dose_revisions.py develop worktree --mode photons --backend cupy
"""

# ruff: noqa: PLC0415

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark" / "results" / "revisions"


def export_revision(revision: str) -> Path:
    """Export the package of one revision once and return the import root."""
    if revision == "worktree":
        return ROOT
    sha = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
    snapshot = RESULTS / "snapshots" / sha
    if not snapshot.exists():
        archive = subprocess.check_output(["git", "archive", sha, "pyRadPlan"], cwd=ROOT)
        snapshot.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(snapshot, filter="data")
    return snapshot


def worker(snapshot: str, mode: str, backend: str, runs: int) -> None:
    """Time the influence calculation of one snapshot and print a JSON record."""
    sys.path.insert(0, snapshot)
    import logging
    import warnings

    import numpy as np
    import pyRadPlan
    from pyRadPlan import IonPlan, PhotonPlan, calc_dose_influence, generate_stf, load_tg119
    from pyRadPlan import settings

    assert Path(pyRadPlan.__file__).is_relative_to(Path(snapshot)), pyRadPlan.__file__
    settings.xp.prefer_gpu = backend == "cupy"
    settings.xp.preferred_cpu_array_backend = "numpy"
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")

    ct, cst = load_tg119()
    angles = [0, 72, 144, 216, 288] if mode == "photons" else [0, 120, 240]
    common = dict(
        machine="Generic",
        prop_stf=dict(gantry_angles=angles, couch_angles=[0] * len(angles), bixel_width=5.0),
        prop_dose_calc=dict(dose_grid=dict(resolution=[3.0] * 3), console_progress=False),
    )
    if mode == "photons":
        common["prop_dose_calc"]["random_seed"] = 1234
        plan = PhotonPlan(**common)
    else:
        plan = IonPlan(radiation_mode=mode, bio_model="none", **common)
    stf = generate_stf(ct, cst, plan)

    def synchronize():
        if backend == "cupy":
            import cupy

            cupy.cuda.Device().synchronize()

    times = []
    for _ in range(runs + 1):
        synchronize()
        start = time.perf_counter()
        result = calc_dose_influence(ct, cst, stf, plan)
        synchronize()
        times.append(time.perf_counter() - start)

    matrix = result.physical_dose.flat[0]
    if hasattr(matrix, "get"):
        matrix = matrix.get()
    matrix = matrix.tocsc()
    matrix.sort_indices()
    digest = hashlib.sha256()
    for array in (matrix.data, matrix.indices.astype(np.int64), matrix.indptr.astype(np.int64)):
        digest.update(memoryview(np.ascontiguousarray(array)))
    print(
        json.dumps(
            dict(
                cold=times[0],
                warm=times[1:],
                nnz=int(matrix.nnz),
                index_dtype=str(matrix.indices.dtype),
                sha256=digest.hexdigest(),
            )
        )
    )


def main() -> None:
    """Alternate revisions per round and summarize warm medians and output equality."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "revisions",
        nargs="*",
        help="Git revisions, or 'worktree' for uncommitted changes; the first is the baseline",
    )
    parser.add_argument("--mode", choices=["photons", "protons", "carbon"], default="photons")
    parser.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    parser.add_argument("--runs", type=int, default=2, help="Warm runs after one cold run")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.mode, args.backend, args.runs)
        return
    if not args.revisions:
        parser.error("at least one revision is required")

    snapshots = {revision: export_revision(revision) for revision in args.revisions}
    records = {revision: [] for revision in args.revisions}
    for round_ix in range(args.rounds):
        for revision, snapshot in snapshots.items():
            process = subprocess.run(
                [sys.executable, __file__, "--worker", str(snapshot)]
                + ["--mode", args.mode, "--backend", args.backend, "--runs", str(args.runs)],
                capture_output=True,
                text=True,
                check=False,
            )
            if process.returncode:
                sys.exit(f"{revision} failed:\n{process.stderr}")
            output = process.stdout
            record = json.loads(output.strip().splitlines()[-1])
            records[revision].append(record)
            print(f"round {round_ix}: {revision}: warm {statistics.median(record['warm']):.2f} s")

    baseline = statistics.median(t for r in records[args.revisions[0]] for t in r["warm"])
    print(f"\n{args.mode} / {args.backend}, warm medians over {args.rounds} rounds:")
    for revision, runs in records.items():
        median = statistics.median(t for r in runs for t in r["warm"])
        identical = {r["sha256"] for r in runs} == {records[args.revisions[0]][0]["sha256"]}
        print(
            f"  {revision:>20}: {median:7.2f} s  {baseline / median:5.2f}x  "
            f"nnz {runs[0]['nnz']}  indices {runs[0]['index_dtype']}  "
            f"{'identical' if identical else 'DIFFERENT'} output"
        )
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{args.mode}_{args.backend}_{int(time.time())}.json"
    out.write_text(json.dumps(dict(args=vars(args), records=records), indent=2) + "\n")
    print(f"Saved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

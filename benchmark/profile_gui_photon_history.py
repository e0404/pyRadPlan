"""Recheck GUI-default TG119 IMRT against complete, isolated package revisions.

No checkout or installed package is changed. Git archives are cached locally.
"""

# ruff: noqa: PLC0415, PLR0915

import argparse
from contextlib import nullcontext
import cProfile
import io
import hashlib
import json
import logging
from pathlib import Path
import pstats
import subprocess
import sys
import time
import warnings


def compare_saved_outputs(out):
    """Hash complete sparse outputs without building dense or difference matrices."""
    from scipy import sparse

    records = []
    for path in sorted(out.glob("*.npz")):
        matrix = sparse.load_npz(path)
        record = dict(file=path.name, shape=matrix.shape, nnz=matrix.nnz)
        for key in ("data", "indices", "indptr"):
            array = getattr(matrix, key)
            record[key] = dict(
                dtype=str(array.dtype), sha256=hashlib.sha256(memoryview(array)).hexdigest()
            )
        records.append(record)
        del matrix
        print(path.name, record, flush=True)
    assert records, "No saved dose matrices found"
    comparison = dict(
        outputs=records,
        exact_match=all(
            all(r[key] == records[0][key] for key in ("shape", "nnz", "data", "indices", "indptr"))
            for r in records
        ),
    )
    (out / "output_comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    assert comparison["exact_match"], "Dose outputs differ; inspect the saved matrices"


def main():
    """Run the GUI-equivalent compute callback and record its actual backend."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default="135da7a1")
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--blas-threads", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--variant", choices=["baseline", "scattermax"], default="baseline")
    parser.add_argument("--out", default="benchmark/pb_audit_results/gui_photon_recheck")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.compare_only:
        compare_saved_outputs(root / args.out)
        return
    cache = root / ".benchmarks" / "gui_photon_history"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ".gitignore").write_text("*\n")
    git = ["git", "-c", f"safe.directory={root.as_posix()}"]
    revision = subprocess.check_output(
        [*git, "rev-parse", args.revision], cwd=root, text=True
    ).strip()
    snapshot = cache / revision
    if not snapshot.exists():
        archive = subprocess.check_output([*git, "archive", revision, "pyRadPlan"], cwd=root)
        snapshot.mkdir()
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(snapshot, filter="data")
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import pyRadPlan
    from pyRadPlan import PhotonPlan, calc_dose_influence, generate_stf, load_tg119, settings
    from scipy import sparse
    from threadpoolctl import threadpool_info, threadpool_limits

    assert Path(pyRadPlan.__file__).is_relative_to(snapshot), pyRadPlan.__file__
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    ct, cst = load_tg119()
    plan = PhotonPlan(
        machine="Generic",
        prop_stf=dict(gantry_angles=[0, 72, 144, 216, 288], couch_angles=[0] * 5, bixel_width=5.0),
        prop_dose_calc=dict(
            dose_grid=dict(resolution=[3.0] * 3), random_seed=1234, console_progress=False
        ),
    )
    if args.variant != "baseline":
        from profile_pb_audit import apply_variant

        apply_variant(args.variant)
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    label = f"{args.revision}_{args.variant}_blas{args.blas_threads or 'default'}"
    times = []
    effective = {}

    def run():
        start = time.perf_counter()
        stf = generate_stf(ct, cst, plan)
        generated = time.perf_counter()
        result = calc_dose_influence(ct, cst, stf, plan)
        end = time.perf_counter()
        effective.update(
            rays=stf.num_of_rays,
            beamlets=stf.total_number_of_bixels,
            shape=result.physical_dose.flat[0].shape,
            nnz=result.physical_dose.flat[0].nnz,
        )
        return result, dict(
            stf_seconds=generated - start, dose_seconds=end - generated, total_seconds=end - start
        )

    limit = (
        threadpool_limits(limits=args.blas_threads, user_api="blas")
        if args.blas_threads
        else nullcontext()
    )
    with limit:
        for i in range(args.runs + 1):
            result, elapsed = run()
            times.append(elapsed)
            print(label, i, elapsed, flush=True)
            (out / (label + "_times.json")).write_text(json.dumps(times, indent=2) + "\n")
        matrix = result.physical_dose.flat[0]
        sparse.save_npz(out / (label + ".npz"), matrix)
        record = dict(
            revision=revision,
            imported_from=pyRadPlan.__file__,
            numpy=np.__version__,
            module_sources={
                name: module.__file__
                for name, module in sys.modules.items()
                if name.startswith("pyRadPlan.") and getattr(module, "__file__", None)
            },
            args=vars(args),
            times=times,
            effective=effective,
            threadpools=threadpool_info(),
            settings=settings.xp.model_dump(mode="json"),
            plan=dict(
                radiation_mode=plan.radiation_mode,
                machine="Generic",
                prop_stf=plan.prop_stf,
                prop_dose_calc=plan.prop_dose_calc,
            ),
        )
        (out / (label + ".json")).write_text(json.dumps(record, indent=2, default=str) + "\n")
        if args.profile:
            profiler = cProfile.Profile()
            profiler.runcall(run)
            with (out / (label + "_profile.txt")).open("w") as stream:
                pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                    "cumulative"
                ).print_stats(80)
        print("Saved", out / (label + ".json"), flush=True)


if __name__ == "__main__":
    main()

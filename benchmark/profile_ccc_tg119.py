"""Time and peak memory of the collapsed cone engine on TG119.

Runs the dose influence matrix and/or the forward dose of a TG119 plan with the collapsed
cone engine ("CCC") on one array backend and prints the wall time and the peak memory: host
memory (tracemalloc peak of Python and NumPy allocations, peak RSS of the process) and, on
a GPU, the CuPy memory pool (used / total bytes) or ``torch.cuda.max_memory_allocated``.

Examples
--------
CPU, a single beam of the 7 beam plan to estimate the full plan::

    python benchmark/profile_ccc_tg119.py --backend numpy --compute-beams 1

Google Colab with a GPU runtime (after ``pip install -e .`` of the branch)::

    !python benchmark/profile_ccc_tg119.py --backend cupy
    !python benchmark/profile_ccc_tg119.py --backend torch
"""

import argparse
import json
import resource
import time
import tracemalloc

import numpy as np


def _gpu_memory(backend: str) -> dict:
    if backend == "cupy":
        import cupy  # noqa: PLC0415

        pool = cupy.get_default_memory_pool()
        return {"pool_used_bytes": pool.used_bytes(), "pool_total_bytes": pool.total_bytes()}
    if backend == "torch":
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return {"max_memory_allocated": torch.cuda.max_memory_allocated()}
    return {}


def _reset_gpu_memory(backend: str) -> None:
    if backend == "cupy":
        import cupy  # noqa: PLC0415

        cupy.get_default_memory_pool().free_all_blocks()
    elif backend == "torch":
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()


def _synchronize(backend: str) -> None:
    if backend == "cupy":
        import cupy  # noqa: PLC0415

        cupy.cuda.Device().synchronize()
    elif backend == "torch":
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            torch.cuda.synchronize()


def main() -> None:
    """Run the requested dose calculations and print the measurements as JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="numpy", choices=["numpy", "torch", "cupy"])
    parser.add_argument("--gpu", action=argparse.BooleanOptionalAction, default=None,
                        help="Run on the GPU (default: for cupy, and for torch with CUDA)")  # fmt: skip
    parser.add_argument("--beams", type=int, default=7, help="Beams of the plan")
    parser.add_argument("--compute-beams", type=int, default=None,
                        help="Compute only the first N beams (estimate of the full plan)")  # fmt: skip
    parser.add_argument("--mode", default="both", choices=["dij", "forward", "both"])
    parser.add_argument("--resolution", type=float, default=3.0, help="Dose grid [mm]")
    parser.add_argument("--no-tracemalloc", action="store_true",
                        help="Skip tracemalloc (removes its overhead from the times)")  # fmt: skip
    args = parser.parse_args()

    from pyRadPlan import (  # noqa: PLC0415
        PhotonPlan,
        calc_dose_forward,
        calc_dose_influence,
        generate_stf,
        load_tg119,
        settings,
    )

    gpu = args.gpu
    if gpu is None:
        gpu = args.backend == "cupy"
        if args.backend == "torch":
            import torch  # noqa: PLC0415

            gpu = torch.cuda.is_available()
    settings.xp.prefer_gpu = gpu
    if gpu:
        settings.xp.preferred_gpu_array_backend = args.backend
    else:
        settings.xp.preferred_cpu_array_backend = args.backend

    ct, cst = load_tg119()
    pln = PhotonPlan(machine="Generic")
    angles = np.linspace(0, 360, args.beams, endpoint=False).tolist()[: args.compute_beams]
    pln.prop_stf = {"gantry_angles": angles, "couch_angles": [0.0] * len(angles)}
    res = args.resolution
    pln.prop_dose_calc = {
        "engine": "CCC",
        "dose_grid": {"resolution": {"x": res, "y": res, "z": res}},
    }
    stf = generate_stf(ct, cst, pln)

    results = {
        "backend": args.backend,
        "gpu": gpu,
        "beams_in_plan": args.beams,
        "beams_computed": len(stf.beams),
        "bixels": int(stf.total_number_of_bixels),
        "dose_grid_mm": res,
    }
    modes = ["dij", "forward"] if args.mode == "both" else [args.mode]
    for mode in modes:
        _reset_gpu_memory(args.backend)
        if not args.no_tracemalloc:
            tracemalloc.start()
        t = time.perf_counter()
        if mode == "dij":
            calc_dose_influence(ct, cst, stf, pln)
        else:
            calc_dose_forward(ct, cst, stf, pln, np.ones(stf.total_number_of_bixels))
        _synchronize(args.backend)
        entry = {"time_s": round(time.perf_counter() - t, 2)}
        if not args.no_tracemalloc:
            entry["tracemalloc_peak_bytes"] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        entry.update(_gpu_memory(args.backend))
        entry["max_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        results[mode] = entry
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

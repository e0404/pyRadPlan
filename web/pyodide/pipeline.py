"""Standard TG119 planning pipeline, shared by the native run and the Pyodide harness.

Run natively (from the repository root)::

    python web/pyodide/pipeline.py --mode photons --preset demo --out native_photons_demo.npz

Under Pyodide the harness imports this file and calls :func:`run_pipeline` directly.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np

#: Plan settings per preset. ``default`` mirrors ``examples/intro_minimal_*.py`` (pyRadPlan
#: defaults: one beam at 0 deg, 5 mm dose grid, bixel width of the stf generator); ``demo``
#: is the 5-beam setup of ``examples/pencilbeam_photon.py``; ``fast`` is a coarse
#: configuration aimed at interactive use in the browser.
PRESETS = {
    "default": {"num_beams": 1, "bixel_width": None, "dose_resolution": None},
    "demo": {"num_beams": 5, "bixel_width": None, "dose_resolution": None},
    "fast": {"num_beams": 5, "bixel_width": 10.0, "dose_resolution": 8.0},
}


def _array(image) -> np.ndarray:
    import SimpleITK as sitk  # noqa: PLC0415

    return np.asarray(sitk.GetArrayFromImage(image))


def _rss_peak_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1024 if sys.platform != "darwin" else peak / 1024**2
    except Exception:  # noqa: BLE001
        pass
    try:
        import psutil  # noqa: PLC0415

        info = psutil.Process().memory_info()
        return getattr(info, "peak_wset", info.rss) / 1024**2
    except Exception:  # noqa: BLE001
        return None


def run_pipeline(  # noqa: PLR0913, PLR0915
    mode: str = "photons",
    preset: str = "default",
    solver: str = "scipy",
    max_iter: int = 200,
    num_beams: int | None = None,
    bixel_width: float | None = None,
    dose_resolution: float | None = None,
    dij_sampling: bool = True,
    dij_seed: int | None = None,
) -> dict:
    """Run TG119 -> plan -> stf -> dij -> optimization -> CT-grid dose.

    Returns
    -------
    dict
        ``info`` (JSON-able timings and sanity numbers) and ``arrays`` (numpy arrays: the
        optimized dose on the CT grid, the dose of unit bixel weights on the CT grid, the
        optimized weights and the target mask).
    """
    import pyRadPlan  # noqa: PLC0415

    pyRadPlan.settings.xp.prefer_gpu = False
    pyRadPlan.settings.xp.preferred_cpu_array_backend = "numpy"
    pyRadPlan.settings.xp.jit_backends = ""

    from pyRadPlan import (  # noqa: PLC0415
        IonPlan,
        PhotonPlan,
        calc_dose_influence,
        fluence_optimization,
        generate_stf,
        load_tg119,
    )

    cfg = dict(PRESETS[preset])
    for key, value in (
        ("num_beams", num_beams),
        ("bixel_width", bixel_width),
        ("dose_resolution", dose_resolution),
    ):
        if value is not None:
            cfg[key] = value

    timings: dict[str, float] = {}

    def tick(name, func, *args):
        t0 = time.perf_counter()
        out = func(*args)
        timings[name] = time.perf_counter() - t0
        return out

    ct, cst = tick("load_tg119", load_tg119)

    pln = PhotonPlan(machine="Generic") if mode == "photons" else IonPlan(radiation_mode=mode)
    n = int(cfg["num_beams"])
    prop_stf = {
        "gantry_angles": list(np.linspace(0.0, 360.0, n, endpoint=False)),
        "couch_angles": [0.0] * n,
    }
    if cfg["bixel_width"] is not None:
        prop_stf["bixel_width"] = float(cfg["bixel_width"])
    pln.prop_stf = prop_stf
    prop_dose_calc = {}
    if cfg["dose_resolution"] is not None:
        r = float(cfg["dose_resolution"])
        prop_dose_calc["dose_grid"] = {"resolution": {"x": r, "y": r, "z": r}}
    if not dij_sampling and mode == "photons":
        prop_dose_calc["enable_dij_sampling"] = False
    if dij_seed is not None and mode == "photons":
        prop_dose_calc["random_seed"] = int(dij_seed)
    pln.prop_dose_calc = prop_dose_calc
    pln.prop_opt = {"solver": solver, "max_iter": int(max_iter)}

    stf = tick("generate_stf", generate_stf, ct, cst, pln)
    dij = tick("calc_dose_influence", calc_dose_influence, ct, cst, stf, pln)
    weights = tick("fluence_optimization", fluence_optimization, ct, cst, stf, dij, pln)
    weights = np.asarray(weights)
    result = tick("compute_result_ct_grid", dij.compute_result_ct_grid, weights)
    unit = dij.compute_result_ct_grid(np.ones_like(weights))

    dose = _array(result["physical_dose"]).astype(np.float64)
    unit_dose = _array(unit["physical_dose"]).astype(np.float64)

    target = None
    for voi in cst.vois:
        if voi.voi_type == "TARGET":
            target = _array(voi.mask) > 0
            break
    tdose = dose[target] if target is not None else dose

    matrix = dij.physical_dose.flat[0]
    info = {
        "mode": mode,
        "preset": preset,
        "config": cfg,
        "solver": solver,
        "max_iter": int(max_iter),
        "dij_sampling": bool(dij_sampling),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pyradplan": getattr(pyRadPlan, "__version__", "unknown"),
        "numpy": np.__version__,
        "ct_size": list(ct.size),
        "dose_grid": list(dij.dose_grid.dimensions),
        "num_bixels": int(weights.size),
        "dij_nnz": int(getattr(matrix, "nnz", 0)),
        "timings_s": timings,
        "total_s": sum(timings.values()),
        "dose_max": float(dose.max()),
        "dose_mean": float(dose.mean()),
        "target_dose_max": float(tdose.max()),
        "target_dose_mean": float(tdose.mean()),
        "target_dose_min": float(tdose.min()),
        "target_d95": float(np.percentile(tdose, 5)),
        "target_d5": float(np.percentile(tdose, 95)),
        "unit_dose_max": float(unit_dose.max()),
        "peak_rss_mb": _rss_peak_mb(),
    }
    arrays = {
        "dose": dose.astype(np.float32),
        "unit_dose": unit_dose.astype(np.float32),
        "weights": weights.astype(np.float64),
        "iso_centers": np.array([np.asarray(b.iso_center) for b in stf.beams]),
        "ray_positions": np.concatenate(
            [np.array([np.asarray(r.ray_pos_bev) for r in b.rays]) for b in stf.beams]
        ),
        "bixel_sums": np.asarray(matrix.sum(axis=0)).ravel(),
    }
    if target is not None:
        arrays["target"] = target
    return {"info": info, "arrays": arrays}


def save_result(result: dict, path: str) -> None:
    """Write ``arrays`` and the JSON ``info`` of :func:`run_pipeline` to an ``.npz`` file."""
    np.savez_compressed(path, info=np.array(json.dumps(result["info"])), **result["arrays"])


def main(argv=None) -> None:
    """Run the pipeline from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="photons", choices=["photons", "protons"])
    parser.add_argument("--preset", default="default", choices=sorted(PRESETS))
    parser.add_argument("--solver", default="scipy")
    parser.add_argument("--max-iter", type=int, default=200)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-dij-sampling", action="store_true")
    parser.add_argument("--dij-seed", type=int, default=None)
    parser.add_argument(
        "--sitk-shim",
        action="store_true",
        help="import the Pyodide SimpleITK shim (web/pyodide/shims) instead of SimpleITK",
    )
    args = parser.parse_args(argv)
    if args.sitk_shim:
        from pathlib import Path  # noqa: PLC0415

        sys.path.insert(0, str(Path(__file__).resolve().parent / "shims"))
        import SimpleITK  # noqa: PLC0415

        assert getattr(SimpleITK, "IS_PYRADPLAN_SHIM", False)
    result = run_pipeline(
        args.mode,
        args.preset,
        args.solver,
        args.max_iter,
        dij_sampling=not args.no_dij_sampling,
        dij_seed=args.dij_seed,
    )
    print(json.dumps(result["info"], indent=2))
    if args.out:
        save_result(result, args.out)


if __name__ == "__main__":
    main()

"""Compare original, hoisted and batched cutoff preparation including transfers.

Single/double cases use a real Generic proton kernel. Multi and singleXY cases
are explicitly synthetic extensions of that kernel to exercise their workload.
Timings include quadrature construction, kernel conversion/upload and result
download. Warm GPU allocation/JIT caches are retained, but no prepared data is
reused between timed calls. Run sequentially without other benchmarks.
"""

# ruff: noqa: PLC0415

import argparse
from copy import deepcopy
import json
from pathlib import Path
import platform
import statistics
import time

import array_api_compat
import numpy as np

from profile_cutoff_reference import legacy_particle_bixel, legacy_quadrature
from pyRadPlan.core.xp_utils import to_namespace, to_numpy
from pyRadPlan.dose.engines import DoseEngineBase, ParticleHongPencilBeamEngine
from pyRadPlan.dose.engines._particle_cutoff import CutoffQuadrature, evaluate_cutoff
from pyRadPlan.machines.particles import LateralCutOff


def make_case(machine, model):
    """Create a real or explicitly synthetic kernel and its scalar reference depths."""
    kernel = deepcopy(machine.get_kernel_by_index(len(machine.energies) // 2))
    kernel.lateral_cut_off = LateralCutOff()
    if model == "multi":
        kernel.sigma_multi = np.stack((kernel.sigma, 2 * kernel.sigma, 4 * kernel.sigma))
        kernel.weight_multi = np.full((2, kernel.depths.size), 0.1)
    elif model == "singleXY":
        kernel.sigma_x = 0.7 * kernel.sigma
        kernel.sigma_y = 1.3 * kernel.sigma
    depths = np.linspace(kernel.depths[0], kernel.depths[-1] * 0.95, 35)
    idd = 1.6021766208e-02 * np.interp(depths, kernel.depths, kernel.idd)
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.lateral_model = model
    engine._use_let_kernel = engine._calc_let = engine._calc_bio_dose = False
    return engine, kernel, depths, idd


def legacy_evaluate(engine, kernel, depths, idd, grid, *, hoist=False, method="integral"):
    """Execute the original per-depth dose/integration operations."""
    prepared = kernel.to_namespace(np) if hoist else None
    result = []
    for depth, reference_idd in zip(depths, idd):
        bixel = dict(
            kernel=prepared if hoist else kernel.to_namespace(np),
            radial_dist_sq=grid.radial_dist_sq,
            lat_dists=grid.lat_dists,
            sigma_ini_sq=4.0,
            rad_depths=(depth + kernel.offset) * np.ones_like(grid.radii),
            rad_depth_offset=0,
            add_sigma_sq=0,
        )
        legacy_particle_bixel(engine, bixel)
        dose = bixel["physical_dose"]
        area = np.cumsum(grid.radii * dose * grid.dr * grid.angular_weights)
        if method == "integral":
            ix = (area >= reference_idd * 0.995).argmax()
            comp = 0.995**-1
        else:
            ix = (dose <= (1 - 0.995) * dose.max()).argmax()
            comp = (area[ix] / area[-1]) ** -1
        result.append([grid.radii[ix], comp, abs(area[-1] / reference_idd - 1) * 100])
    return np.asarray(result)


def measure(case, variant, cp, method):
    """Time preparation, numerical evaluation and download as separate stages."""
    engine, kernel, depths, idd = case
    sync = cp.cuda.get_current_stream().synchronize if variant == "cupy" else lambda: None
    sync()
    start = time.perf_counter()
    if variant in ("legacy", "hoisted"):
        grid = legacy_quadrature(engine.lateral_model)
        prepared = time.perf_counter()
        result = legacy_evaluate(
            engine, kernel, depths, idd, grid, hoist=variant == "hoisted", method=method
        )
        evaluated = finished = time.perf_counter()
    else:
        xp = array_api_compat.array_namespace(cp.empty(0) if variant == "cupy" else np.empty(0))
        grid = CutoffQuadrature.create(engine.lateral_model, xp=xp)
        data = kernel.to_namespace(xp)
        batch_depths = to_namespace(xp, depths)
        batch_idd = to_namespace(xp, idd)
        sync()
        prepared = time.perf_counter()
        result = evaluate_cutoff(
            batch_depths,
            batch_idd,
            grid,
            lambda d, g: engine._cutoff_dose_profile(d, data, 4.0, g),
            cutoff=0.995,
            method=method,
        )
        sync()
        evaluated = time.perf_counter()
        result = to_numpy(result)
        sync()
        finished = time.perf_counter()
    return result, dict(
        total=finished - start,
        preparation=prepared - start,
        evaluation=evaluated - prepared,
        download=finished - evaluated,
    )


def main():
    """Compare all strategies with paired, rotating-order measurements."""
    import cupy as cp
    import scipy

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--models", nargs="+", default=["single", "double", "multi", "singleXY"])
    parser.add_argument("--method", choices=["integral", "relative"], default="integral")
    parser.add_argument("--out", default="benchmark/pb_audit_results/cutoff_preparation")
    args = parser.parse_args()
    cp.arange(1).sum()
    machine = DoseEngineBase.load_machine("protons", "Generic")
    variants = ["legacy", "hoisted", "numpy", "cupy"]
    rows = []
    for model in args.models:
        case = make_case(machine, model)
        expected, _ = measure(case, "legacy", cp, args.method)
        times = {v: [] for v in variants}
        max_abs = {v: np.zeros(3) for v in variants}
        for variant in variants:
            actual, _ = measure(case, variant, cp, args.method)
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
            # Native grid generation may differ by a few ulps; require the same radius bin.
            np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=2e-13, atol=1e-13)
        for i in range(args.rounds):
            for variant in variants[i % 4 :] + variants[: i % 4]:
                actual, elapsed = measure(case, variant, cp, args.method)
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
                np.testing.assert_allclose(actual[:, 0], expected[:, 0], rtol=2e-13, atol=1e-13)
                times[variant].append(elapsed)
                max_abs[variant] = np.maximum(
                    max_abs[variant], np.max(abs(actual - expected), axis=0)
                )
        row = dict(
            model=model,
            synthetic=model in ("multi", "singleXY"),
            method=args.method,
            depth_count=case[2].size,
            energy=float(case[1].energy),
            samples_per_depth=1200 * (360 if model == "singleXY" else 1),
            max_samples=1_000_000,
            seconds=times,
            medians={
                v: {k: statistics.median(t[k] for t in ts) for k in ts[0]}
                for v, ts in times.items()
            },
            max_abs={v: a.tolist() for v, a in max_abs.items()},
        )
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "seconds"}), flush=True)
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        scipy=scipy.__version__,
        cupy=cp.__version__,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        rounds=args.rounds,
        rows=rows,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"preparation_{args.method}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

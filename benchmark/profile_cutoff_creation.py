"""Backend-created quadrature and resident-kernel prototypes in actual dose runs.

Only process-local patches are used. Production modules are unchanged. Resident
variants share immutable device kernel arrays across cutoff and bixel preparation;
beam-dependent cutoff tables are always refreshed. CPU machine models remain for
validation/depth selection. A download variant additionally measures the cost of
recovering kernel arrays from the GPU instead of using the available host mirror.
"""

# ruff: noqa: PLC0415, PLR0915

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import platform
import statistics
import time
import warnings

import array_api_compat
import numpy as np

from profile_particle_cutoff import make_case
from profile_pb_audit import configure, sync
from pyRadPlan import IonPlan, generate_stf, load_tg119
from pyRadPlan.core.xp_utils import to_namespace, to_numpy
from pyRadPlan.dose.engines import DoseEngineBase, ParticleHongPencilBeamEngine
from pyRadPlan.dose.engines._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract
from pyRadPlan.dose.engines._particle_cutoff import CutoffQuadrature, evaluate_cutoff
from pyRadPlan.machines.particles.kernel import ParticlePencilBeamKernel


def create_native(model, xp):
    """Construct the historical grid with Array API operations on its target device."""
    edges = xp.concat((xp.zeros(1, dtype=xp.float64), 10 ** xp.linspace(-1, 3, 1200)))
    radii = 0.5 * (edges[:-1] + edges[1:])
    dr = edges[1:] - edges[:-1]
    if model == "singleXY":
        angles = xp.linspace(0, 2 * xp.pi, 361)[:-1]
        shape = (radii.shape[0], angles.shape[0])
        radii = xp.reshape(xp.broadcast_to(radii[:, None], shape), (-1,))
        dr = xp.reshape(xp.broadcast_to(dr[:, None], shape), (-1,))
        theta = xp.reshape(xp.broadcast_to(angles[None, :], shape), (-1,))
        angular = xp.full(radii.shape, angles[1] - angles[0], dtype=xp.float64)
        r2 = radii**2
        lat = xp.sqrt(r2)[:, None] * xp.stack((xp.cos(theta), xp.sin(theta)), axis=1)
    else:
        angular = xp.asarray(2 * xp.pi, dtype=xp.float64)
        r2 = radii**2
        lat = xp.broadcast_to(xp.sqrt(r2 / 2)[:, None], (radii.shape[0], 2))
    return CutoffQuadrature(radii, dr, angular, r2, lat)


def timed(fn):
    """Include all GPU work in wall-clock measurements."""
    sync("cupy")
    start = time.perf_counter()
    result = fn()
    sync("cupy")
    return result, time.perf_counter() - start


def medians(rows):
    """Summarize equally named timing fields."""
    return {key: statistics.median(row[key] for row in rows) for key in rows[0]}


def microbench(cp, rounds):
    """Isolate grid creation and cutoff evaluation with already-resident kernels."""
    machine = DoseEngineBase.load_machine("protons", "Generic")
    xp_cpu = array_api_compat.array_namespace(np.empty(0))
    xp_gpu = array_api_compat.array_namespace(cp.empty(0))
    records = []
    for model in ("single", "double", "singleXY"):
        engine, kernel, depths, idd = make_case(machine, model)
        data = {"numpy": kernel.to_namespace(xp_cpu), "cupy": kernel.to_namespace(xp_gpu)}
        arrays = {
            "numpy": (depths, idd),
            "cupy": (cp.asarray(depths), cp.asarray(idd)),
        }
        makers = {
            "numpy_host": lambda: CutoffQuadrature.create(model).to_namespace(xp_cpu),
            "numpy_native": lambda: create_native(model, xp_cpu),
            "cupy_host": lambda: CutoffQuadrature.create(model).to_namespace(xp_gpu),
            "cupy_native": lambda: create_native(model, xp_gpu),
        }
        reference = CutoffQuadrature.create(model)
        geometry_errors = {}
        for name, maker in makers.items():
            grid = maker()
            geometry_errors[name] = {}
            for field in vars(grid):
                actual = to_numpy(getattr(grid, field))
                expected = getattr(reference, field)
                np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
                geometry_errors[name][field] = float(np.max(abs(actual - expected)))
        samples = {name: [] for name in makers}
        cutoff_errors = {name: 0.0 for name in makers}
        expected = None
        for iteration in range(rounds + 1):
            names = list(makers)
            names = names[iteration % 4 :] + names[: iteration % 4]
            for name in names:
                backend = name.split("_")[0]
                grid, creation = timed(makers[name])
                ds, ids = arrays[backend]
                result, evaluation = timed(
                    lambda: evaluate_cutoff(
                        ds,
                        ids,
                        grid,
                        lambda d, g: engine._cutoff_dose_profile(d, data[backend], 4.0, g),
                        cutoff=0.995,
                        method="integral",
                    )
                )
                result, download = timed(lambda: to_numpy(result))
                if expected is None:
                    expected = result
                np.testing.assert_allclose(result, expected, rtol=2e-10, atol=2e-10)
                cutoff_errors[name] = max(
                    cutoff_errors[name], float(np.max(abs(result - expected)))
                )
                if iteration:
                    samples[name].append(
                        dict(
                            creation=creation,
                            evaluation=evaluation,
                            download=download,
                            total=creation + evaluation + download,
                        )
                    )
        records.append(
            dict(
                model=model,
                synthetic=model == "singleXY",
                samples=samples,
                medians={name: medians(rows) for name, rows in samples.items()},
                geometry_max_abs=geometry_errors,
                cutoff_max_abs=cutoff_errors,
            )
        )
    return records


@contextmanager
def prototype(variant, resident, xp_gpu, stages=None):
    """Install and restore experiment hooks; optional stage timing is a separate run."""
    owner = ParticlePencilBeamEngineAbstract
    original_create = CutoffQuadrature.__dict__["create"]
    create = CutoffQuadrature.create
    original_convert = ParticlePencilBeamKernel.to_namespace
    original_cutoff = owner._calc_lateral_particle_cut_off
    original_placement = owner._cutoff_execution
    gpu_cutoff = "gpu" in variant

    def record(key, fn):
        if stages is None:
            return fn()
        value, elapsed = timed(fn)
        item = stages.setdefault(key, dict(calls=0, seconds=0.0))
        item["calls"] += 1
        item["seconds"] += elapsed
        return value

    def grid(cls, model):
        return record(
            "grid_creation",
            lambda: create_native(model, xp_gpu) if "native" in variant else create(model),
        )

    def convert(kernel, xp, device=None):
        def run():
            use_resident = variant.startswith("resident") and (
                array_api_compat.is_cupy_namespace(xp) or "download" in variant
            )
            if not use_resident:
                return original_convert(kernel, xp, device=device)
            saved = resident[float(kernel.energy)]
            result = {
                name: to_namespace(xp, value, device=device)
                if isinstance(value, np.ndarray) or array_api_compat.is_cupy_array(value)
                else value
                for name, value in saved.items()
                if name != "lateral_cut_off"
            }
            # Cutoff tables are beam-dependent and must never come from the database cache.
            if kernel.lateral_cut_off is not None:
                cutoff = kernel.lateral_cut_off
                result["lateral_cut_off"] = dict(
                    comp_fac=float(cutoff.comp_fac),
                    cut_off=to_namespace(xp, cutoff.cut_off, device=device),
                    depths=to_namespace(xp, cutoff.depths, device=device),
                )
            return result

        return record("kernel_conversion", run)

    def cutoff(self, *args, **kwargs):
        return record("cutoff_total", lambda: original_cutoff(self, *args, **kwargs))

    CutoffQuadrature.create = classmethod(grid)
    ParticlePencilBeamKernel.to_namespace = convert
    owner._calc_lateral_particle_cut_off = cutoff
    if gpu_cutoff:
        owner._cutoff_execution = lambda self: (self.xp, self.device)
    try:
        yield
    finally:
        CutoffQuadrature.create = original_create
        ParticlePencilBeamKernel.to_namespace = original_convert
        owner._calc_lateral_particle_cut_off = original_cutoff
        owner._cutoff_execution = original_placement


def full_dose(cp, mode, beams, rounds):
    """Run actual TG119 influence calculations with rotating-order prototype policies."""
    configure("numpy")
    ct, cst = load_tg119()
    plan = IonPlan(radiation_mode=mode, machine="Generic", bio_model="none")
    plan.prop_stf = dict(
        bixel_width=10,
        gantry_angles=np.linspace(0, 360, beams, endpoint=False),
        couch_angles=np.zeros(beams),
    )
    if mode == "VHEE":
        plan.prop_stf["energy"] = 200
    plan.prop_dose_calc = dict(dose_grid=dict(resolution=[5] * 3), calc_let=False)
    stf = generate_stf(ct, cst, plan)
    configure("cupy")
    xp_gpu = array_api_compat.array_namespace(cp.empty(0))
    machine = DoseEngineBase.load_machine(mode, "Generic")

    def upload():
        return {
            float(kernel.energy): kernel.to_namespace(xp_gpu)
            for kernel in (machine.get_kernel_by_index(i) for i in range(len(machine.energies)))
        }

    resident, upload_cold = timed(upload)
    upload_times = []
    for _ in range(5):
        resident, elapsed = timed(upload)
        upload_times.append(elapsed)
    resident_bytes = sum(
        value.nbytes
        for data in resident.values()
        for value in data.values()
        if isinstance(value, cp.ndarray)
    )
    variants = [
        "cpu",
        "gpu_host",
        "gpu_native",
        "resident_cpu_mirror",
        "resident_cpu_download",
        "resident_gpu_host",
        "resident_gpu_native",
    ]

    def run(variant, stages=None):
        with prototype(variant, resident, xp_gpu, stages):
            engine = ParticleHongPencilBeamEngine(plan.model_copy(deep=True))
            engine.console_progress = False
            result = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
            return result.physical_dose.flat[0]

    rows = {name: [] for name in variants}
    parity = {name: [] for name in variants}
    expected = None
    for iteration in range(rounds + 1):
        names = variants[iteration % len(variants) :] + variants[: iteration % len(variants)]
        for name in names:
            actual, elapsed = timed(lambda: run(name))
            if expected is None:
                expected = actual.copy()
            assert np.array_equal(actual.indices, expected.indices)
            assert np.array_equal(actual.indptr, expected.indptr)
            np.testing.assert_allclose(actual.data, expected.data, rtol=1e-12, atol=1e-12)
            error = float(np.max(abs(actual.data - expected.data), initial=0))
            if iteration:
                rows[name].append(elapsed)
                parity[name].append(error)
            print(
                f"{mode} beams={beams} round={iteration} {name}: {elapsed:.4f}s error={error}",
                flush=True,
            )
    stages = {}
    for name in variants:
        stages[name] = {}
        run(name, stages[name])
    return dict(
        mode=mode,
        beams=beams,
        samples=rows,
        medians={name: statistics.median(values) for name, values in rows.items()},
        max_abs=parity,
        stages=stages,
        kernel_count=len(resident),
        resident_array_bytes=resident_bytes,
        upload_cold=upload_cold,
        upload_seconds=upload_times,
        upload_median=statistics.median(upload_times),
        beamlets=sum(len(ray.beamlets) for beam in stf.beams for ray in beam.rays),
    )


def main():
    """Run sequentially, persist raw measurements and record numerical differences."""
    import cupy as cp

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--micro-rounds", type=int, default=15)
    parser.add_argument("--beams", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--modes", nargs="+", default=["protons", "VHEE"])
    parser.add_argument("--out", default="benchmark/pb_audit_results/cutoff_creation/results.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    configure("cupy")
    result = dict(
        python=platform.python_version(),
        numpy=np.__version__,
        cupy=cp.__version__,
        gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
        args=vars(args),
        micro=microbench(cp, args.micro_rounds),
        full_dose=[],
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "micro": [
                    {"model": row["model"], "medians": row["medians"]} for row in result["micro"]
                ]
            }
        ),
        flush=True,
    )
    for mode in args.modes:
        for beams in args.beams:
            result["full_dose"].append(full_dose(cp, mode, beams, args.rounds))
            out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Real TG119 ray-geometry workloads, plus optional paired full-dose validation.

Run sequentially (GPU contention makes parallel timing misleading):
    python benchmark/profile_ray_geometry.py --full-dose
"""

# ruff: noqa: N814

import argparse
import gc
import json
import logging
from pathlib import Path
import subprocess
import sys
import time
from types import MethodType
import warnings

import numpy as np

from pyRadPlan import IonPlan, PhotonPlan, generate_stf, load_tg119
from pyRadPlan.core.xp_utils import to_namespace, to_numpy
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract as PB
from pyRadPlan.dose.engines._hongpb import ParticleHongPencilBeamEngine as Hong
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine as SVD
from profile_pb_audit import configure, sync
from profile_ray_geometry_reference import reference_methods


def measure_geometry(ct, cst, stf, plan, backend, rounds):
    """Measure complete ray extraction, including one preparation per beam."""
    engine = (SVD if plan.radiation_mode == "photons" else Hong)(plan)
    engine.console_progress = False
    dij = engine._init_dose_calc(ct, cst, stf)
    for key in ("_vox_world_coords", "_vox_world_coords_dose_grid"):
        setattr(
            engine,
            key,
            engine.xp.asarray(getattr(engine, key), dtype=engine.xp.float32, device=engine.device),
        )
    engine._vdose_grid = to_namespace(engine.xp, engine._vdose_grid, device=engine.device)
    beam = engine._init_beam(dij, ct, cst, stf, 0)
    frozen = reference_methods(PB)
    legacy = frozen["_get_ray_geometry_from_beam"]
    current_calc = engine.calc_geo_dists
    times = {"current": [], "legacy": []}

    def run(name, verify=False):
        engine.calc_geo_dists = (
            MethodType(frozen["calc_geo_dists"], engine) if name == "legacy" else current_calc
        )
        beam.pop("_ray_geometry", None)
        outputs = []
        for original in beam["beam"]["rays"]:
            ray = original.copy()
            ray["source_point_bev"] = beam["beam"]["source_point_bev"]
            ray["sad"] = beam["beam"]["sad"]
            if name == "legacy":
                legacy(engine, ray, beam)
            else:
                engine._get_ray_geometry_from_beam(ray, beam)
            if verify:
                outputs.append(
                    {
                        key: [to_numpy(v) for v in ray[key]]
                        for key in (
                            "ix",
                            "radial_dist_sq",
                            "lat_dists",
                            "iso_lat_dists",
                            "rad_depths",
                            "geo_depths",
                        )
                    }
                )
        sync(backend)
        return outputs

    a, b = run("current", True), run("legacy", True)
    for ra, rb in zip(a, b):
        for key in ra:
            for va, vb in zip(ra[key], rb[key]):
                np.testing.assert_array_equal(va, vb)
    del a, b
    for i in range(rounds):
        for name in times if i % 2 else reversed(times):
            sync(backend)
            start = time.perf_counter()
            run(name)
            times[name].append(time.perf_counter() - start)
    del engine.calc_geo_dists  # Do not leave a bound-method reference cycle.
    sync(backend)
    start = time.perf_counter()
    prepared = engine._prepare_beam_geometry(beam)
    sync(backend)
    prep_seconds = time.perf_counter() - start
    record = dict(
        mode=plan.radiation_mode,
        backend=backend,
        grid=plan.prop_dose_calc["dose_grid"]["resolution"][0],
        trace_on_dose_grid=engine.trace_on_dose_grid,
        rays=len(beam["beam"]["rays"]),
        selected_voxels=len(engine._vdose_grid),
        valid_voxels=prepared["coords"].shape[0],
        seconds=times,
        preparation_seconds=prep_seconds,
        prepared_bytes=sum(v.nbytes for v in (prepared["coords"], prepared["indices"])),
        exact_geometry=True,
    )
    record["speedup"] = float(np.median(np.array(times["legacy"]) / times["current"]))
    return record


def main():
    """Save isolated timings and optionally run full dose comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dose", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--skip-isolated", action="store_true")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out", default="benchmark/pb_audit_results/geometry")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for mode in () if args.skip_isolated else ("photons", "protons", "VHEE"):
        configure("numpy")
        ct, cst = load_tg119()
        plan = (
            PhotonPlan(machine="Generic")
            if mode == "photons"
            else IonPlan(radiation_mode=mode, machine="Generic", bio_model="none")
        )
        plan.prop_stf = dict(bixel_width=10, gantry_angles=[0], couch_angles=[0])
        if mode == "VHEE":
            plan.prop_stf["energy"] = 200
        stf = generate_stf(ct, cst, plan)
        for backend in ("numpy", "cupy"):
            configure(backend)
            for grid in (8, 5, 3):
                for trace_on_dose_grid in (True, False):
                    plan.prop_dose_calc = dict(
                        dose_grid=dict(resolution=[grid] * 3),
                        trace_on_dose_grid=trace_on_dose_grid,
                    )
                    if mode == "photons":
                        plan.prop_dose_calc.update(enable_dij_sampling=False, random_seed=1234)
                    else:
                        plan.prop_dose_calc["calc_let"] = False
                    record = measure_geometry(
                        ct,
                        cst,
                        stf.model_copy(deep=True),
                        plan.model_copy(deep=True),
                        backend,
                        args.rounds,
                    )
                    records.append(record)
                    (out / "isolated.json").write_text(json.dumps(records, indent=2) + "\n")
                    print(json.dumps(record), flush=True)
    if args.full_dose:
        # Release isolated workloads before timing independent full-dose processes.
        gc.collect()
        if "cupy" in sys.modules:
            cp = sys.modules["cupy"]
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
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
                        "legacy_geometry",
                        "--paired",
                        "--pairs",
                        str(args.pairs),
                        "--runs",
                        "1",
                        "--out",
                        str(out),
                    ]
                    if trace_ct:
                        command.append("--trace-on-ct-grid")
                    subprocess.run(command, check=True)
        for path in out.glob("*_paired.json"):
            pair = json.loads(path.read_text())
            assert all(
                c["max_abs"] == 0 and c["identical_indices"] for c in pair["comparisons"]
            ), path


if __name__ == "__main__":
    main()

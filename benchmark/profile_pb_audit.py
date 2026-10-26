"""Isolated pencil-beam audit; never edits production modules on disk."""

# Runtime patching and lazy backend imports keep each benchmark process isolated.
# ruff: noqa: N814, PLC0415, PLR0915

from __future__ import annotations

import argparse
import cProfile
import inspect
import json
import logging
import os
from pathlib import Path
import platform
import pstats
import re
import textwrap
import time
import warnings

import numpy as np
from scipy import sparse

from pyRadPlan import IonPlan, PhotonPlan, generate_stf, load_tg119, settings
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract as PB
from pyRadPlan.dose.engines._base_pencilbeam_particle import ParticlePencilBeamEngineAbstract as PP
from pyRadPlan.dose.engines._hongpb import ParticleHongPencilBeamEngine as Hong
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine as SVD
from pyRadPlan.raytracer import RayTracerSiddon
from pyRadPlan.dose.engines._base import DoseEngineBase


def replace_method(cls, name, substitutions):
    """Compile a process-local prototype from an existing method."""
    fn = getattr(cls, name)
    source = textwrap.dedent(inspect.getsource(fn))
    for old, new in substitutions:
        assert old in source, (name, old)
        source = source.replace(old, new)
    namespace = dict(fn.__globals__)
    exec(compile(source, "<audit prototype " + name + ">", "exec"), namespace)
    setattr(cls, name, namespace[name])


def apply_variant(name):
    """Install the selected experiment in this Python process only."""
    if name == "baseline":
        return
    if name in ("legacy_interpolators", "native_interpolators"):
        from profile_interpnd_reference import legacy_grid_interpolator, native_grid_interpolator

        factory = (
            legacy_grid_interpolator
            if name == "legacy_interpolators"
            else native_grid_interpolator
        )
        SVD._get_kernel_interpolators.__globals__["_audit_grid_interpolator"] = factory
        replace_method(
            SVD,
            "_get_kernel_interpolators",
            [("ArrayBackendGridInterpolator(", "_audit_grid_interpolator(")],
        )
        return
    if name == "compact_take":
        return  # Shared compaction with Array API take is now the production path.
    if name in ("compact", "legacy_indexing"):
        fields = (
            "ix",
            "radial_dist_sq",
            "rad_depths",
            "lat_dists",
            "v_alpha_x",
            "v_beta_x",
            "v_tissue_class_ix",
        )
        source = textwrap.dedent(inspect.getsource(PP._get_bixel_indices_on_ray))
        substitutions = []
        for field in fields:
            pattern = rf'xp\.take\(\s*curr_ray\["{field}"\],\s*curr_ix,\s*axis=0\s*\)'
            match = re.search(pattern, source)
            assert match is not None, field
            substitutions.append((match.group(), f'curr_ray["{field}"][curr_ix]'))
        replace_method(
            PP,
            "_get_bixel_indices_on_ray",
            ([("    curr_ix = xp.nonzero(curr_ix)[0]\n", "")] if name == "legacy_indexing" else [])
            + substitutions,
        )
    if name in ("hoist", "legacy_cutoff"):
        from profile_cutoff_reference import (
            legacy_cutoff,
            legacy_particle_bixel,
            make_hoisted_cutoff,
        )

        PP._calc_lateral_particle_cut_off = (
            make_hoisted_cutoff() if name == "hoist" else legacy_cutoff
        )
        Hong._calc_particle_bixel = legacy_particle_bixel
        return
    if name == "numpy_cutoff":
        import array_api_compat

        PP._cutoff_execution = lambda self: (array_api_compat.array_namespace(np.empty(0)), None)
        return
    if name == "device_cutoff":
        PP._cutoff_execution = lambda self: (self.xp, self.device)
        return
    if name in ("fft", "combined"):
        replace_method(
            SVD,
            "_get_kernel_interpolators",
            [
                ("_fft2(f, s=(conv_size, conv_size))", "f_fft"),
                (
                    "for c in range(num_kernels):",
                    "f_fft = _fft2(f, s=(conv_size, conv_size))\n    for c in range(num_kernels):",
                ),
            ],
        )
    if name == "scattermax":
        fn = SVD._sample_dij
        source = textwrap.dedent(inspect.getsource(fn))
        start = source.index("    order = xp.argsort(dose_tail, stable=True)")
        end = source.index("    threshold_dose =", start)
        source = (
            source[:start]
            + (
                "    cluster_dose = xp.zeros((num_clusters,), dtype=bixel_dose.dtype, device=device)\n"
                "    xp.maximum.at(cluster_dose, cluster, dose_tail)\n"
            )
            + source[end:]
        )
        namespace = dict(fn.__globals__)
        exec(compile(source, "<audit scatter maximum>", "exec"), namespace)
        SVD._sample_dij = namespace["_sample_dij"]
    if name == "union":
        return  # Nominal selection reuse is now implemented in production.
    if name == "legacy_voxel_selection":
        from profile_voxel_preparation_reference import legacy_voxel_selection

        DoseEngineBase._init_voxel_selection = legacy_voxel_selection
        return
    if name == "prepared":
        return  # Interpolator reuse is now provided by the production backend class.
    if name == "device_forward":
        original_fill = PB._fill_dij
        original_finalize = PB._finalize_dose

        def fill(self, bixel, dij, stf, scen, beam, ray, k, counter):
            if not self._calc_dose_direct:
                return original_fill(self, bixel, dij, stf, scen, beam, ray, k, counter)
            ix = tuple(np.unravel_index(scen, self.mult_scen.scen_mask.shape))
            buffers = self.__dict__.setdefault("_audit_device_buffers", {})
            for q in self._computed_quantities:
                key = (q, ix)
                if key not in buffers:
                    buffers[key] = self.xp.zeros(
                        dij[q][ix].shape, dtype=self.xp.float32, device=self.device
                    )
                buffers[key][bixel["ix"], beam] += bixel["weight"] * bixel[q]
            for key in ("beam_num", "ray_num", "bixel_num"):
                dij[key][beam] = beam

        def finalize(self, dij):
            from pyRadPlan.core.xp_utils import to_numpy

            for (q, ix), buffer in self.__dict__.get("_audit_device_buffers", {}).items():
                dij[q][ix] = to_numpy(buffer)
            return original_finalize(self, dij)

        PB._fill_dij = fill
        PB._finalize_dose = finalize


def configure(backend):
    """Select the requested backend and initialize its runtime."""
    settings.xp.prefer_gpu = backend == "cupy"
    settings.xp.preferred_cpu_array_backend = "numpy"
    settings.xp.preferred_gpu_array_backend = "cupy"
    if backend == "cupy":
        import cupy as cp

        cp.arange(8).sum()
        cp.cuda.get_current_stream().synchronize()


def sync(backend):
    """Wait for outstanding work on the current CUDA stream."""
    if backend == "cupy":
        import cupy as cp

        cp.cuda.get_current_stream().synchronize()


def save_mismatch(out, label, pair, a, b, forward):
    """Retain failing dose outputs locally for repeatability diagnosis."""
    if forward:
        np.savez_compressed(out / f"{label}_pair{pair}_mismatch.npz", baseline=a, variant=b)
    else:
        sparse.save_npz(out / f"{label}_pair{pair}_baseline.npz", a)
        sparse.save_npz(out / f"{label}_pair{pair}_variant.npz", b)


def main():
    """Run full dose calculations, optional profiles, and paired comparisons."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["numpy", "cupy"], required=True)
    parser.add_argument("--mode", default="protons")
    parser.add_argument("--machine", default="Generic")
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--grid", type=float, default=5)
    parser.add_argument("--bixel", type=float, default=10)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--let", dest="with_let", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--paired", action="store_true")
    parser.add_argument("--pairs", type=int, default=6)
    parser.add_argument("--out", default="benchmark/pb_audit_results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.ERROR)
    warnings.filterwarnings("ignore")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    configure("numpy")  # Identical, CPU-generated steering for both dose backends.
    ct, cst = load_tg119()
    photon = args.mode in ("photons", "field")
    plan = (
        PhotonPlan(machine=args.machine)
        if photon
        else IonPlan(radiation_mode=args.mode, machine=args.machine, bio_model="none")
    )
    plan.prop_stf = dict(
        bixel_width=args.bixel,
        gantry_angles=np.linspace(0, 360, args.beams, endpoint=False),
        couch_angles=np.zeros(args.beams),
    )
    if args.mode == "VHEE":
        plan.prop_stf["energy"] = 200
    if args.mode == "field":
        from pyRadPlan.machines import create_bld

        plan.prop_stf.update(
            generator="photonSingleBixel",
            field_based=True,
            blds=[
                create_bld(
                    dict(
                        device_type="JAW",
                        device_orientation="X",
                        positions=[-20, 20],
                        field_width=50,
                        leakage=0.01,
                    )
                )
            ],
            resolution=0.5,
            energy=6,
        )
    plan.prop_dose_calc = {"dose_grid": {"resolution": [args.grid] * 3}}
    if photon:
        plan.prop_dose_calc.update(enable_dij_sampling=args.sampling, random_seed=1234)
    else:
        plan.prop_dose_calc.update(calc_let=args.with_let)
    t = time.perf_counter()
    stf = generate_stf(ct, cst, plan)
    preparation = time.perf_counter() - t
    configure(args.backend)
    patch_targets = [
        (PP, "_get_bixel_indices_on_ray"),
        (PP, "_calc_lateral_particle_cut_off"),
        (PP, "_cutoff_execution"),
        (Hong, "_calc_particle_bixel"),
        (SVD, "_get_kernel_interpolators"),
        (SVD, "_sample_dij"),
        (DoseEngineBase, "_init_dose_calc"),
        (DoseEngineBase, "_init_voxel_selection"),
        (PB, "_fill_dij"),
        (PB, "_finalize_dose"),
    ]
    baseline_methods = [(c, n, getattr(c, n)) for c, n in patch_targets]
    apply_variant(args.variant)
    variant_methods = [(c, n, getattr(c, n)) for c, n in patch_targets]
    cls = SVD if photon else Hong
    effective = {}

    def run():
        engine = cls(plan.model_copy(deep=True))
        engine.console_progress = False
        if args.forward:
            result = engine.calc_dose_forward(ct, cst, stf.model_copy(deep=True), None)
        else:
            result = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
        sync(args.backend)
        effective.update(
            dose_resolution=engine.dose_grid.resolution,
            dose_dimensions=engine.dose_grid.dimensions,
            selected_voxels=int(engine._vdose_grid.size),
            lateral_model=getattr(engine, "lateral_model", None),
        )
        return result

    times = []
    for i in range(args.runs + 1):
        sync(args.backend)
        t = time.perf_counter()
        result = run()
        elapsed = time.perf_counter() - t
        print(f"{args.mode} {args.backend} {args.variant} run {i}: {elapsed:.4f}s", flush=True)
        times.append(elapsed)
    label = f"{args.mode}_{args.machine}_{args.backend}_{args.variant}_g{args.grid}_b{args.bixel}_n{args.beams}"
    label += "_forward" if args.forward else "_influence"
    label += "_sampled" if args.sampling else ""
    label += "_let" if args.with_let else ""
    if args.forward:
        import SimpleITK as sitk

        matrix = sitk.GetArrayFromImage(result["physical_dose"])
        np.save(out / (label + ".npy"), matrix)
        size = dict(shape=list(matrix.shape), bytes=matrix.nbytes)
    else:
        matrix = result.physical_dose.flat[0]
        sparse.save_npz(out / (label + ".npz"), matrix)
        size = dict(
            shape=list(matrix.shape),
            nnz=int(matrix.nnz),
            bytes=matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes,
            index_dtype=str(matrix.indices.dtype),
            data_dtype=str(matrix.data.dtype),
        )
        if args.with_let:
            let_matrix = result.let_dose.flat[0]
            sparse.save_npz(out / (label + "_let_dose.npz"), let_matrix)
            size["let_nnz"] = int(let_matrix.nnz)
            size["let_data_bytes"] = int(let_matrix.data.nbytes)
            size["let_shares_indices"] = bool(np.shares_memory(matrix.indices, let_matrix.indices))
            size["let_shares_indptr"] = bool(np.shares_memory(matrix.indptr, let_matrix.indptr))
    metadata = dict(
        args=vars(args),
        python=platform.python_version(),
        numpy=np.__version__,
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        stf_seconds=preparation,
        cold_seconds=times[0],
        warm_seconds=times[1:],
        median_seconds=float(np.median(times[1:])),
        effective=effective,
        jit_backends=settings.xp.jit_backends,
        rays=sum(len(b.rays) for b in stf.beams),
        beamlets=sum(len(r.beamlets) for b in stf.beams for r in b.rays),
        output=size,
    )
    if args.backend == "cupy":
        import cupy as cp

        metadata.update(
            cupy=cp.__version__,
            gpu=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
        )
    (out / (label + ".json")).write_text(json.dumps(metadata, indent=2))
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(run)
        profiler.dump_stats(str(out / (label + ".prof")))
        with (out / (label + "_profile.txt")).open("w") as stream:
            pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats(
                "cumulative"
            ).print_stats(65)
        # Separate synchronized, exclusive stage attribution. Not used for speedups.
        stages = {}
        stack = []
        originals = []
        targets = [
            (PB, "_init_dose_calc"),
            (PB, "_init_beam"),
            (PB, "_compute_ssd"),
            (PB, "calc_geo_dists"),
            (PB, "_fill_dij"),
            (PB, "_finalize_dose"),
            (RayTracerSiddon, "trace_cubes"),
            (PP, "_calc_lateral_particle_cut_off"),
            (PP, "_init_bixel"),
            (Hong, "_calc_particle_bixel"),
            (SVD, "_init_beam"),
            (SVD, "_get_kernel_interpolators"),
            (SVD, "_compute_bixel"),
        ]

        def wrap(fn, key):
            def wrapped(*a, **kw):
                sync(args.backend)
                frame = [time.perf_counter(), 0.0]
                stack.append(frame)
                try:
                    return fn(*a, **kw)
                finally:
                    sync(args.backend)
                    elapsed = time.perf_counter() - frame[0]
                    stack.pop()
                    record = stages.setdefault(
                        key, {"calls": 0, "exclusive_s": 0, "inclusive_s": 0}
                    )
                    record["calls"] += 1
                    record["exclusive_s"] += elapsed - frame[1]
                    record["inclusive_s"] += elapsed
                    if key == "PencilBeamEngineAbstract._fill_dij" and not a[0]._calc_dose_direct:
                        shared = a[2]["shared_structure"].flat[0]
                        record["peak_capacity"] = max(
                            record.get("peak_capacity", 0), int(shared["indices"].size)
                        )
                        record["final_nnz"] = int(shared["nnz"])
                    if stack:
                        stack[-1][1] += elapsed

            return wrapped

        for owner, name in targets:
            fn = getattr(owner, name)
            originals.append((owner, name, fn))
            setattr(owner, name, wrap(fn, owner.__name__ + "." + name))
        run()
        for owner, name, fn in originals:
            setattr(owner, name, fn)
        (out / (label + "_stages.json")).write_text(json.dumps(stages, indent=2))
    print(json.dumps(metadata), flush=True)
    if args.paired:
        paired = {"baseline": [], "variant": [], "comparisons": []}
        for pair in range(args.pairs):
            outputs = {}
            names = ["baseline", "variant"] if pair % 2 == 0 else ["variant", "baseline"]
            for name in names:
                methods = baseline_methods if name == "baseline" else variant_methods
                for c, n, fn in methods:
                    setattr(c, n, fn)
                sync(args.backend)
                t = time.perf_counter()
                outputs[name] = run()
                elapsed = time.perf_counter() - t
                paired[name].append(elapsed)
                print(f"PAIR {pair} {name}: {elapsed:.4f}s", flush=True)
            if args.forward:
                import SimpleITK as sitk

                a = sitk.GetArrayFromImage(outputs["baseline"]["physical_dose"])
                b = sitk.GetArrayFromImage(outputs["variant"]["physical_dose"])
                max_abs = float(np.max(np.abs(a - b), initial=0))
                comparison = dict(max_abs=max_abs, max_abs_over_peak=max_abs / float(a.max()))
            else:
                a = outputs["baseline"].physical_dose.flat[0]
                b = outputs["variant"].physical_dose.flat[0]
                diff = a - b
                max_abs = float(np.max(np.abs(diff.data), initial=0))
                comparison = dict(
                    max_abs=max_abs,
                    max_abs_over_peak=max_abs / float(np.max(np.abs(a.data))),
                    identical_indices=bool(
                        np.array_equal(a.indices, b.indices) and np.array_equal(a.indptr, b.indptr)
                    ),
                )
            paired["comparisons"].append(comparison)
            if comparison["max_abs"] != 0:
                save_mismatch(out, label, pair, a, b, args.forward)
        (out / (label + "_paired.json")).write_text(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()

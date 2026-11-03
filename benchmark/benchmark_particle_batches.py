"""CPU-only, bounded particle batching prototypes; production dispatch is unchanged.

Compare validation-only, ragged packed and padded physics batches. Interpolation
and support selection deliberately retain their existing per-beamlet algorithms.
Run --self-check for synthetic Array API checks, or run paired TG119 calculations.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["PYRADPLAN_XP_PREFER_GPU"] = "false"

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import platform
import time

import array_api_compat
import numpy as np

from pyRadPlan import IonPlan, generate_stf, load_tg119, settings
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine as Hong


def pack_arrays(xp, arrays, layout, *, axis=-1, fill=1):
    """Pack variable-length arrays; padded layout inserts its batch axis before voxels."""
    if layout == "packed":
        return xp.concat(arrays, axis=axis)
    axis = axis % arrays[0].ndim
    shape = list(arrays[0].shape)
    shape[axis] = max(a.shape[axis] for a in arrays)
    shape.insert(axis, len(arrays))
    result = xp.full(
        tuple(shape), fill, dtype=arrays[0].dtype, device=array_api_compat.device(arrays[0])
    )
    for i, array in enumerate(arrays):
        selection = [slice(None)] * result.ndim
        selection[axis] = i
        selection[axis + 1] = slice(0, array.shape[axis])
        result[tuple(selection)] = array
    return result


def schema(bixel, kernels):
    """Different Gaussian component counts/dtypes cannot share this simple packet."""
    return (
        tuple((key, a.dtype, a.shape[:-1]) for key, a in kernels.items()),
        bixel["radial_dist_sq"].dtype,
        bixel["sigma_ini_sq"].dtype,
        tuple(
            (key, bixel[key].dtype, bixel[key].shape[1:])
            for key in ("v_alpha_x", "v_beta_x", "v_tissue_class_ix", "lat_dists")
            if key in bixel
        ),
    )


def check_dose(xp, dose):
    """Retain the existing NaN/negative policy; one host decision per batch."""
    if bool(xp.any(xp.isnan(dose) | (dose < 0))):
        raise ValueError("Error in particle dose calculation (batch).")


def evaluate_batch(engine, bixels, kernels, layout):
    """Return outputs and accounted materialized batch-array bytes, without transfers."""
    xp = array_api_compat.array_namespace(bixels[0]["radial_dist_sq"])
    lengths = [b["radial_dist_sq"].shape[0] for b in bixels]
    if layout == "checks" or len(bixels) == 1:
        doses = []
        for bixel, fields in zip(bixels, kernels):
            lateral = engine._lateral_profile(
                bixel["radial_dist_sq"], bixel.get("lat_dists"), bixel["sigma_ini_sq"], fields
            )
            dose = bixel["kernel"]["lateral_cut_off"]["comp_fac"] * lateral * fields["idd"]
            bixel["physical_dose"] = dose
            doses.append(dose)
            if engine._calc_let:
                bixel["let_dose"] = dose * fields["let"]
        combined = doses[0] if len(doses) == 1 else xp.concat(doses)
        check_dose(xp, combined)
        for bixel, fields in zip(bixels, kernels):
            if engine._calc_bio_dose and "v_alpha_x" in bixel:
                bixel.update(engine._evaluate_bio_influence(bixel, fields))
        return 0 if len(doses) == 1 else array_nbytes(xp, combined)

    needed = ("idd", *engine._lateral_kernel_names(), *(("let",) if engine._calc_let else ()))
    # Biological channels are consumed per bixel below; do not copy them into
    # batch buffers that the physical evaluator never reads.
    fields = {key: pack_arrays(xp, [k[key] for k in kernels], layout) for key in needed}
    radial = pack_arrays(xp, [b["radial_dist_sq"] for b in bixels], layout)
    lat = (
        pack_arrays(xp, [b["lat_dists"] for b in bixels], layout, axis=0)
        if engine.lateral_model == "singleXY"
        else None
    )
    if layout == "padded":
        sigma = xp.stack([b["sigma_ini_sq"] for b in bixels])[:, None]
    else:
        sigma = pack_arrays(
            xp, [xp.broadcast_to(b["sigma_ini_sq"], (n,)) for b, n in zip(bixels, lengths)], layout
        )
    lateral = engine._lateral_profile(radial, lat, sigma, fields)
    # These are Python metadata floats in the converted kernel. Preserve the
    # scalar multiplication's precision instead of torch's default float32.
    factors = xp.asarray(
        [b["kernel"]["lateral_cut_off"]["comp_fac"] for b in bixels],
        dtype=lateral.dtype,
        device=array_api_compat.device(radial),
    )
    if layout == "padded":
        compensation = factors[:, None]
    else:
        compensation = pack_arrays(
            xp, [xp.broadcast_to(factors[i], (n,)) for i, n in enumerate(lengths)], layout
        )
    dose = compensation * lateral * fields["idd"]
    valid = None
    if layout == "padded":
        valid = (
            xp.arange(dose.shape[-1], device=array_api_compat.device(dose))[None, :]
            < xp.asarray(lengths, device=array_api_compat.device(dose))[:, None]
        )
        dose = xp.where(valid, dose, 0.0)
    check_dose(xp, dose)
    quantities = {"physical_dose": dose}
    if engine._calc_let:
        quantities["let_dose"] = dose * fields["let"]
    start = 0
    for i, (bixel, n) in enumerate(zip(bixels, lengths)):
        for key, value in quantities.items():
            bixel[key] = value[start : start + n] if layout == "packed" else value[i, :n]
        # Biology stays per-beamlet in this first prototype: custom evaluators may
        # inspect dimensionality or component counts. Output identities stay intact.
        if engine._calc_bio_dose and "v_alpha_x" in bixel:
            bixel.update(engine._evaluate_bio_influence(bixel, kernels[i]))
        start += n
    # Account for the main batch field arrays, not an allocator peak. Excludes
    # original bixel inputs, small metadata vectors and expression/JIT temporaries.
    arrays = [*fields.values(), radial, sigma, compensation, lateral, *quantities.values()]
    arrays += [] if lat is None else [lat]
    arrays += [] if valid is None else [valid]
    return sum(array_nbytes(xp, a) for a in arrays)


def array_nbytes(xp, array):
    """Account for array storage without downloading values or relying on nbytes."""
    if array.dtype == xp.bool:
        itemsize = 1
    elif xp.isdtype(array.dtype, "integral"):
        itemsize = xp.iinfo(array.dtype).bits // 8
    else:
        itemsize = xp.finfo(array.dtype).bits // 8
    return int(np.prod(array.shape)) * itemsize


class BatchPrototype(Hong):
    """Delay bounded physics work until output submission; never reorder columns.

    This experimental adapter only supports Hong's existing physics. A production
    implementation should use explicit prepare/evaluate/collect hooks instead of
    changing the meaning of _compute_bixel and _fill_dij as this adapter does.
    """

    def __init__(self, plan, *, layout, scope, batch_size, max_pairs):
        self.layout, self.scope = layout, scope
        self.batch_size, self.max_pairs = batch_size, max_pairs
        self.pending = []
        self.pending_pairs = self.pending_max = 0
        self.metrics = defaultdict(float)
        self.batch_counts = []
        super().__init__(plan)

    def _init_dose_calc(self, *args):
        # A failed or interrupted run must not leak pending work into engine reuse.
        self.pending.clear()
        self.pending_pairs = self.pending_max = 0
        self.metrics.clear()
        self.batch_counts.clear()
        return super()._init_dose_calc(*args)

    def _compute_bixel(self, ray, k):
        return self._init_bixel(ray, k)

    def _fill_dij(self, bixel, *args):
        # args: dij, stf, scenario, beam, ray, local bixel, delivery column.
        n = bixel["radial_dist_sq"].shape[0]
        size = len(self.pending) + 1
        capacity = (
            max(self.pending_max, n) * size if self.layout == "padded" else self.pending_pairs + n
        )
        boundary = self.pending and self.scope == "ray" and args[2:5] != self.pending[-1][1][2:5]
        if self.pending and (size > self.batch_size or capacity > self.max_pairs or boundary):
            self.flush()
        if n > self.max_pairs:
            # One large bixel is unavoidable, but never add packed copies to it.
            self._calc_particle_bixel(bixel)
            super()._fill_dij(bixel, *args)
            self.metrics["oversized_scalar_bixels"] += 1
            return
        self.pending.append((bixel, args))
        self.pending_pairs += n
        self.pending_max = max(self.pending_max, n)
        if len(self.pending) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        entries, self.pending = self.pending, []
        self.pending_pairs = self.pending_max = 0
        started = time.perf_counter()
        prepared = [(b, args, self._interpolate_kernels_in_depth(b)) for b, args in entries]
        self.metrics["interpolation_s"] += time.perf_counter() - started
        # Contiguous compatible runs preserve delivery/scenario ordering exactly.
        groups = []
        for entry in prepared:
            signature = schema(entry[0], entry[2])
            if not groups or signature != groups[-1][0]:
                groups.append((signature, []))
            groups[-1][1].append(entry)
        self.metrics["schema_splits"] += len(groups) - 1
        for _, group in groups:
            bixels = [b for b, _, _ in group]
            fields = [k for _, _, k in group]
            started = time.perf_counter()
            size = evaluate_batch(self, bixels, fields, self.layout)
            self.metrics["physics_packing_validation_s"] += time.perf_counter() - started
            self.metrics["max_accounted_batch_bytes"] = max(
                size, self.metrics["max_accounted_batch_bytes"]
            )
            lengths = [b["radial_dist_sq"].shape[0] for b in bixels]
            capacity = max(lengths) * len(lengths) if self.layout == "padded" else sum(lengths)
            self.metrics["selected_pairs"] += sum(lengths)
            self.metrics["evaluated_pairs"] += capacity
            self.metrics["max_batch_pairs"] = max(capacity, self.metrics["max_batch_pairs"])
            self.batch_counts.append(len(group))
            started = time.perf_counter()
            for bixel, args, _ in group:
                super()._fill_dij(bixel, *args)
            self.metrics["collection_s"] += time.perf_counter() - started

    def _init_beam(self, *args):
        self.flush()  # Before the machine's geometry-dependent cutoff data changes.
        return super()._init_beam(*args)

    def _init_geometry_group(self):
        self.flush()
        return super()._init_geometry_group()

    def _finalize_dose(self, dij):
        self.flush()
        return super()._finalize_dose(dij)


def compare_dose(a, b):
    """Compare sparse structure, every quantity/scenario and delivery identities."""
    if isinstance(a, dict):
        import SimpleITK as sitk

        assert a.keys() == b.keys()
        errors = {}
        for key in a:
            aa = a[key] if isinstance(a[key], list) else [a[key]]
            bb = b[key] if isinstance(b[key], list) else [b[key]]
            assert len(aa) == len(bb)
            for i, (x, y) in enumerate(zip(aa, bb)):
                if isinstance(x, sitk.Image):
                    assert x.GetOrigin() == y.GetOrigin()
                    assert x.GetSpacing() == y.GetSpacing()
                    assert x.GetDirection() == y.GetDirection()
                    x, y = sitk.GetArrayFromImage(x), sitk.GetArrayFromImage(y)
                    np.testing.assert_allclose(x, y, rtol=2e-6, atol=1e-12)
                    errors[f"{key}_{i}"] = float(np.max(np.abs(x - y), initial=0))
                else:
                    np.testing.assert_array_equal(x, y)
        return errors
    assert a.quantities == b.quantities
    for key in ("beam_num", "ray_num", "bixel_num", "num_of_scenarios"):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    errors = {}
    for key in a.quantities:
        aa, bb = getattr(a, key), getattr(b, key)
        assert aa.shape == bb.shape
        for i, (x, y) in enumerate(zip(aa.flat, bb.flat)):
            assert x.shape == y.shape
            np.testing.assert_array_equal(x.indptr, y.indptr)
            np.testing.assert_array_equal(x.indices, y.indices)
            np.testing.assert_allclose(x.data, y.data, rtol=2e-6, atol=1e-12)
            errors[f"{key}_{i}"] = float(np.max(np.abs(x.data - y.data), initial=0))
    return errors


def self_check():
    """Exercise ragged/empty batches, all lateral models and active invalid doses."""
    import importlib
    import copy

    checks = 0
    for backend in ("numpy", "torch", "array_api_strict"):
        module = importlib.import_module(backend)
        xp = array_api_compat.array_namespace(module.asarray([0.0]))
        for dtype_name in ("float32", "float64"):
            for model in ("single", "double", "multi", "singleXY"):
                for lengths in ([3, 0, 7, 2], [0, 0]):
                    engine = object.__new__(Hong)
                    engine.lateral_model, engine._calc_let, engine._calc_bio_dose = (
                        model,
                        True,
                        False,
                    )
                    bixels, kernels = [], []
                    for i, n in enumerate(lengths):
                        make = lambda a: xp.asarray(a, dtype=getattr(xp, dtype_name))
                        bixels.append(
                            dict(
                                radial_dist_sq=make(np.arange(n) + 1),
                                lat_dists=make(np.ones((n, 2))),
                                sigma_ini_sq=make(i + 0.5),
                                kernel={"lateral_cut_off": {"comp_fac": 1.123456789012345}},
                            )
                        )
                        fields = dict(idd=make(np.arange(n) + 1), let=make(np.ones(n) * 2))
                        for key in engine._lateral_kernel_names():
                            shape = (2, n) if key in ("sigma_multi", "weight_multi") else (n,)
                            fields[key] = make(np.full(shape, 0.2 if "weight" in key else 2.0))
                        if model == "multi":
                            fields["sigma_multi"] = make(np.full((3, n), 2.0))
                        kernels.append(fields)
                    for layout in ("checks", "packed", "padded"):
                        actual = [dict(b) for b in bixels]
                        evaluate_batch(engine, actual, kernels, layout)
                        for b, result, fields in zip(bixels, actual, kernels):
                            expected = (
                                b["kernel"]["lateral_cut_off"]["comp_fac"]
                                * engine._lateral_profile(
                                    b["radial_dist_sq"], b["lat_dists"], b["sigma_ini_sq"], fields
                                )
                                * fields["idd"]
                            )
                            np.testing.assert_allclose(
                                to_numpy(result["physical_dose"]),
                                to_numpy(expected),
                                rtol=2e-6 if dtype_name == "float32" else 1e-14,
                            )
                            np.testing.assert_allclose(
                                to_numpy(result["let_dose"]),
                                to_numpy(expected * fields["let"]),
                                rtol=2e-6 if dtype_name == "float32" else 1e-14,
                            )
                        checks += 1
                    if lengths[0]:
                        for invalid in (np.nan, -1.0):
                            for layout in ("checks", "packed", "padded"):
                                bad = copy.copy(kernels)
                                bad[0] = {**bad[0], "idd": make(np.full(lengths[0], invalid))}
                                try:
                                    evaluate_batch(engine, [dict(b) for b in bixels], bad, layout)
                                except ValueError:
                                    checks += 1
                                else:
                                    raise AssertionError((backend, model, layout, invalid))
    print(f"{checks} CPU synthetic batch checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "torch"], default="numpy")
    parser.add_argument("--mode", choices=["protons", "VHEE", "carbon"], default="protons")
    parser.add_argument("--machine", default="Generic")
    parser.add_argument("--bio-model", default="none")
    parser.add_argument("--grid", type=float, default=5)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--trace-on-ct-grid", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--layouts", nargs="+", default=["checks", "packed", "padded"])
    parser.add_argument("--sizes", nargs="+", type=int, default=[4, 16])
    parser.add_argument("--scope", choices=["ray", "beam"], default="beam")
    parser.add_argument("--max-pairs", type=int, default=65536)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--out", default="benchmark/pb_audit_results/batching_cpu")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if min(*args.sizes, args.max_pairs, args.pairs) < 1:
        raise ValueError("Batch sizes, pair budget and repeat counts must be positive")
    if not set(args.layouts) <= {"checks", "packed", "padded"}:
        raise ValueError("Unknown layout")
    logging.basicConfig(level=logging.ERROR)
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    ct, cst = load_tg119()
    plan = IonPlan(radiation_mode=args.mode, machine=args.machine, bio_model=args.bio_model)
    plan.prop_stf = dict(
        bixel_width=10,
        gantry_angles=np.linspace(0, 360, args.beams, endpoint=False),
        couch_angles=np.zeros(args.beams),
    )
    if args.mode == "VHEE":
        plan.prop_stf["energy"] = 200
    plan.prop_dose_calc = dict(
        dose_grid={"resolution": [args.grid] * 3},
        trace_on_dose_grid=not args.trace_on_ct_grid,
        calc_let=True,
    )
    if args.scenarios:
        from pyRadPlan.scenarios import (
            ScenarioSet,
            ExplicitScenarioModel,
            SetupUncertainty,
            RangeUncertainty,
        )

        scenarios = ScenarioSet.from_points(
            (SetupUncertainty(), RangeUncertainty()),
            {"setup": [[0, 0, 0], [0, 0, 0], [2, 0, 0]], "range": [[0, 0], [2, 0], [0, 0]]},
        )
        plan.scenarios = ExplicitScenarioModel(scenario_set=scenarios)
    stf = generate_stf(ct, cst, plan)
    settings.xp.preferred_cpu_array_backend = args.backend
    threads, torch_version = None, None
    if args.backend == "torch":
        import torch

        if args.torch_threads:
            torch.set_num_threads(args.torch_threads)
        threads, torch_version = torch.get_num_threads(), torch.__version__

    def run(layout=None, size=None):
        engine = (
            Hong(plan.model_copy(deep=True))
            if layout is None
            else BatchPrototype(
                plan.model_copy(deep=True),
                layout=layout,
                scope=args.scope,
                batch_size=size,
                max_pairs=args.max_pairs,
            )
        )
        engine.console_progress = False
        start = time.perf_counter()
        if args.forward:
            weights = np.linspace(0.2, 1.4, stf.total_number_of_bixels)
            dose = engine.calc_dose_forward(ct, cst, stf.model_copy(deep=True), weights)
        else:
            dose = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
        elapsed = time.perf_counter() - start
        info = {} if layout is None else dict(engine.metrics)
        if layout is not None:
            info["batch_count"] = len(engine.batch_counts)
            info["mean_bixels_per_batch"] = (
                float(np.mean(engine.batch_counts)) if engine.batch_counts else 0
            )
            assert info.get("max_batch_pairs", 0) <= args.max_pairs
        return elapsed, dose, info

    baseline, reference, _ = run()
    results = {}
    for layout in args.layouts:
        for size in args.sizes:
            label = f"{layout}_{size}"
            _, warm, _ = run(layout, size)
            compare_dose(reference, warm)
            samples = {"scalar": [], "batch": []}
            for i in range(args.pairs):
                outputs = {}
                for key in ("scalar", "batch") if i % 2 == 0 else ("batch", "scalar"):
                    elapsed, outputs[key], info = run(layout if key == "batch" else None, size)
                    samples[key].append(elapsed)
                    if key == "batch":
                        metrics = info
                errors = compare_dose(outputs["scalar"], outputs["batch"])
            results[label] = dict(samples_s=samples, metrics=metrics, max_abs_errors=errors)
            print(
                label, {k: round(float(np.median(v)), 4) for k, v in samples.items()}, flush=True
            )
    result = dict(
        reference="530209cc scalar production path",
        config=vars(args),
        numpy=np.__version__,
        torch=torch_version,
        torch_threads=threads,
        cpu=platform.processor(),
        beamlets=stf.total_number_of_bixels,
        rays=sum(len(beam.rays) for beam in stf.beams),
        scalar_first_s=baseline,
        results=results,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filename = f"{args.backend}_{args.mode}_{args.machine}_{args.grid:g}mm_{args.scope}_{args.bio_model}_{'scenarios' if args.scenarios else 'nominal'}_{'ct' if args.trace_on_ct_grid else 'dose'}.json"
    if args.forward:
        filename = filename.replace(".json", "_forward.json")
    (out / filename).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Bounded support/interpolation/physics/transfer batching, isolated from production.

Native grouping reuses prepared backend interpolators for identical energy/offset
tables. Binary grouping uses the experimental multi-table Array API bank. The
collector's existing sparse/forward assembly remains authoritative. CuPy timings
synchronize at full calculation boundaries, never between timed pipeline stages.
"""

# Optional GPU and scenario dependencies are loaded only by the requested experiment.
# ruff: noqa: PLC0415

import argparse
from collections import defaultdict
import json
import logging
from pathlib import Path
import platform
import time

import numpy as np

from pyRadPlan import IonPlan, generate_stf, load_tg119, settings
from pyRadPlan.core.xp_utils import (
    ArrayBackend1DInterpolator,
    ArrayBackend1DInterpolatorBank,
    to_numpy,
)
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine as Hong
from benchmark_particle_batches import BatchPrototype, compare_dose, check_dose
from benchmark_grouped_interp import TableBank, scalar_bound


def concatenate(xp, arrays, axis=0):
    """Avoid a copy when a work packet has only one constituent array."""
    return arrays[0] if len(arrays) == 1 else xp.concat(arrays, axis=axis)


class PipelinePrototype(BatchPrototype):
    """Candidate-bounded work packets with explicit original delivery arguments."""

    def __init__(self, plan, *, verify=False, two_stage=False, **kwargs):
        self.verify = verify
        self.two_stage = two_stage
        self.ready = []
        self.ready_pairs = 0
        self.energy_group = {}
        self.banks = {}
        self.depth_tables = {}
        self.cutoff_tables = {}
        super().__init__(plan, **kwargs)

    def _init_dose_calc(self, *args):
        self.ready.clear()
        self.ready_pairs = 0
        return super()._init_dose_calc(*args)

    def _init_geometry_group(self):
        self.flush()
        self.flush_ready()
        return Hong._init_geometry_group(self)

    def _finalize_dose(self, dij):
        self.flush()
        self.flush_ready()
        return Hong._finalize_dose(self, dij)

    def _get_bixel_indices_on_ray(self, bixel, ray):
        bixel["_reference_ray"] = ray
        if self.layout == "interpolation":
            Hong._get_bixel_indices_on_ray(self, bixel, ray)
            bixel["_source_ray"] = {key: bixel[key] for key in self.source_fields if key in bixel}
        else:
            bixel["_source_ray"] = ray
        return bixel

    @property
    def source_fields(self):
        """Return voxel fields carried through exact support selection."""
        return (
            "ix",
            "rad_depths",
            "radial_dist_sq",
            "lat_dists",
            "v_alpha_x",
            "v_beta_x",
            "v_tissue_class_ix",
        )

    def _prepare_bank(self, points, fields):
        if self.layout == "binary":
            return TableBank(points, fields)
        bank = ArrayBackend1DInterpolatorBank(
            points, fields, prefer_native=self.layout != "bank_generic"
        )
        return lambda q, ids, shifts: bank(q, ids, grid_offsets=shifts)

    @property
    def _uses_bank(self):
        return self.layout in ("binary", "bank", "bank_generic")

    def _init_beam(self, *args):
        self.flush()
        self.flush_ready()
        beam = Hong._init_beam(self, *args)
        start = time.perf_counter()
        self.energy_group.clear()
        self.banks.clear()
        self.depth_tables.clear()
        self.cutoff_tables.clear()
        xp = self.xp
        energies = sorted({b["energy"] for ray in beam["beam"]["rays"] for b in ray["beamlets"]})
        groups = defaultdict(list)
        for energy in energies:
            ix = int(self._machine.get_energy_index(energy, 4))
            cache_key = (np.int64(ix), xp)
            if cache_key not in self._kernel_cache:
                self._kernel_cache[cache_key] = self._machine.get_kernel_by_index(ix).to_namespace(
                    xp, device=self.device
                )
            kernel = self._kernel_cache[cache_key]
            stub = {"kernel": kernel}
            if self._calc_bio_dose:
                stub["v_alpha_x"] = None
            prepared = self._prepare_depth_interpolator(stub)
            self._depth_interpolator_cache[(id(kernel), bool(self._calc_bio_dose))] = (
                kernel,
                prepared,
            )
            self.depth_tables[ix] = prepared
            cutoff = kernel["lateral_cut_off"]
            if "cutoff_vals_sq" not in cutoff:
                cutoff["cutoff_vals_sq"] = cutoff["cut_off"] ** 2
            cx = cutoff["depths"][:1] if cutoff["cut_off"].shape[0] == 1 else cutoff["depths"]
            self.cutoff_tables[ix] = ArrayBackend1DInterpolator(cx, cutoff["cutoff_vals_sq"])
            # Private attributes are intentionally confined to this experiment.
            # A production backend API should expose table-bank preparation explicitly.
            signature = tuple((key, a.dtype, a.shape[:-1]) for key, a in prepared._fields.items())
            groups[signature].append((ix, kernel, prepared, cx))
        for group_id, entries in enumerate(groups.values()):
            for row, (ix, *_rest) in enumerate(entries):
                self.energy_group[ix] = (group_id, row)
            points = [p._x for _, _, p, _ in entries]
            fields = [p._fields for _, _, p, _ in entries]
            self.banks[group_id] = dict(
                depth=self._prepare_bank(points, fields) if self._uses_bank else None,
                cutoff=self._prepare_bank(
                    [cx for *_, cx in entries],
                    [{"cutoff": k["lateral_cut_off"]["cutoff_vals_sq"]} for _, k, _, _ in entries],
                )
                if self._uses_bank
                else None,
                end=xp.stack([k["depths"][-1] for _, k, _, _ in entries]),
                lower=xp.stack([cx[0] for *_, cx in entries]),
                upper=xp.stack([cx[-1] for *_, cx in entries]),
                constant=xp.asarray(
                    [k["lateral_cut_off"]["cut_off"].shape[0] == 1 for _, k, _, _ in entries],
                    device=self.device,
                ),
                constant_value=xp.stack(
                    [k["lateral_cut_off"]["cutoff_vals_sq"][0] for _, k, _, _ in entries]
                ),
            )
        self.metrics["table_preparation_submit_s"] += time.perf_counter() - start
        self.metrics["prepared_energies"] += len(energies)
        return beam

    def _fill_dij(self, bixel, *args):
        n = bixel["_source_ray"]["rad_depths"].shape[0]
        if self.pending:
            previous, previous_args = self.pending[-1]
            incompatible = (
                self.energy_group[int(previous["energy_ix"])][0]
                != self.energy_group[int(bixel["energy_ix"])][0]
            )
            boundary = self.scope == "ray" and args[2:5] != previous_args[2:5]
            if (
                len(self.pending) >= self.batch_size
                or self.pending_pairs + n > self.max_pairs
                or incompatible
                or boundary
            ):
                self.flush()
        if n > self.max_pairs:
            self.flush_ready()
            if self.layout != "interpolation":
                Hong._get_bixel_indices_on_ray(self, bixel, bixel["_reference_ray"])
            self._calc_particle_bixel(bixel)
            Hong._fill_dij(self, bixel, *args)
            self.metrics["oversized_scalar_bixels"] += 1
            return
        self.pending.append((bixel, args))
        self.pending_pairs += n
        if len(self.pending) >= self.batch_size:
            self.flush()

    def grouped_native(self, q, offsets, bixels, *, cutoff=False):
        """Reuse prepared interpolation for queries sharing an energy and offset."""
        xp = self.xp
        groups = defaultdict(list)
        for i, b in enumerate(bixels):
            shift = (
                b["kernel"]["offset"] - b["rad_depth_offset"] if cutoff else -b["rad_depth_offset"]
            )
            groups[(int(b["energy_ix"]), float(shift))].append(i)
        output = {}
        for (energy, shift), members in groups.items():
            queries = concatenate(xp, [q[int(offsets[i]) : int(offsets[i + 1])] for i in members])
            if cutoff:
                values = {"cutoff": self.cutoff_tables[energy](queries, grid_offset=shift)}
            else:
                values = self.depth_tables[energy](queries, grid_offset=shift)
            for key, value in values.items():
                if key not in output:
                    output[key] = xp.empty(
                        value.shape[:-1] + (q.shape[0],), dtype=value.dtype, device=self.device
                    )
                start = 0
                for i in members:
                    n = int(offsets[i + 1] - offsets[i])
                    output[key][..., int(offsets[i]) : int(offsets[i + 1])] = value[
                        ..., start : start + n
                    ]
                    start += n
        self.metrics["cutoff_native_groups" if cutoff else "depth_native_groups"] += len(groups)
        return output

    def select(self, bixels):
        """Compact exact support once for a bounded packet of candidate pairs."""
        xp = self.xp
        lengths = [b["_source_ray"]["rad_depths"].shape[0] for b in bixels]
        offsets = np.cumsum([0, *lengths], dtype=np.int64)
        unique, source_starts, ray_start = {}, [], 0
        for b in bixels:
            ray = b["_source_ray"]
            if id(ray) not in unique:
                unique[id(ray)] = (ray, ray_start)
                ray_start += ray["rad_depths"].shape[0]
            source_starts.append(unique[id(ray)][1])
        sources = [ray for ray, _ in unique.values()]
        source = {
            key: concatenate(xp, [ray[key] for ray in sources])
            for key in self.source_fields
            if key in sources[0]
        }
        positions = xp.arange(int(offsets[-1]), dtype=xp.int64, device=self.device)
        device_offsets = xp.asarray(offsets, device=self.device)
        owner = xp.searchsorted(device_offsets, positions, side="right") - 1
        source_ix = (
            positions
            - xp.take(device_offsets, owner)
            + xp.take(xp.asarray(source_starts, dtype=xp.int64, device=self.device), owner)
        )
        q, r2 = (
            xp.take(source["rad_depths"], source_ix),
            xp.take(source["radial_dist_sq"], source_ix),
        )
        rows = xp.asarray(
            [self.energy_group[int(b["energy_ix"])][1] for b in bixels],
            dtype=xp.int64,
            device=self.device,
        )
        table_ids = xp.take(rows, owner)
        group_id = self.energy_group[int(bixels[0]["energy_ix"])][0]
        bank = self.banks[group_id]
        shifts = xp.asarray(
            [b["kernel"]["offset"] - b["rad_depth_offset"] for b in bixels],
            dtype=bank["end"].dtype,
            device=self.device,
        )
        shift = xp.take(shifts, owner)
        if self.layout == "interpolation":
            keep = xp.ones_like(q, dtype=xp.bool)
        else:
            end = scalar_bound(xp, xp.take(bank["end"], table_ids) + shift, q)
            keep = q <= end
            if self.dosimetric_lateral_cutoff != 1:
                if self._uses_bank:
                    radius_sq = bank["cutoff"](q, table_ids, shift)["cutoff"]
                    self.metrics["cutoff_binary_calls"] += 1
                else:
                    radius_sq = self.grouped_native(q, offsets, bixels, cutoff=True)["cutoff"]
                lower = scalar_bound(xp, xp.take(bank["lower"], table_ids) + shift, q)
                upper = scalar_bound(xp, xp.take(bank["upper"], table_ids) + shift, q)
                constant = xp.take(bank["constant"], table_ids)
                constant_value = xp.astype(xp.take(bank["constant_value"], table_ids), r2.dtype)
                keep &= xp.where(
                    constant, constant_value >= r2, (radius_sq >= r2) & (q >= lower) & (q <= upper)
                )
        picked = xp.nonzero(keep)[0]
        selected_owner = xp.take(owner, picked)
        selected_source = xp.take(source_ix, picked)
        selected = {key: xp.take(value, selected_source, axis=0) for key, value in source.items()}
        # One small metadata download, not one synchronization per beamlet. It
        # supplies native grouping and the unchanged CPU sparse collector.
        cuts = to_numpy(
            xp.searchsorted(
                selected_owner, xp.arange(len(bixels) + 1, dtype=xp.int64, device=self.device)
            )
        )
        for i, b in enumerate(bixels):
            for key, value in selected.items():
                b[key] = value[int(cuts[i]) : int(cuts[i + 1]), ...]
            if self.verify:
                reference = dict(b)
                Hong._get_bixel_indices_on_ray(self, reference, b["_reference_ray"])
                for key in selected:
                    np.testing.assert_array_equal(to_numpy(b[key]), to_numpy(reference[key]))
        self.metrics["candidate_pairs"] += int(offsets[-1])
        self.metrics["selected_pairs"] += int(cuts[-1])
        self.metrics["max_batch_pairs"] = max(self.metrics["max_batch_pairs"], int(offsets[-1]))
        return selected, selected_owner, cuts, rows, bank

    def flush(self):
        """Select pending candidate work and schedule its surviving pairs."""
        if not self.pending:
            return
        entries, self.pending = self.pending, []
        self.pending_pairs = 0
        bixels = [b for b, _ in entries]
        xp = self.xp
        start = time.perf_counter()
        if self.two_stage and len(bixels) == 1:
            b = bixels[0]
            candidates = b["_source_ray"]["rad_depths"].shape[0]
            if self.layout != "interpolation":
                Hong._get_bixel_indices_on_ray(self, b, b["_reference_ray"])
            selected = {key: b[key] for key in self.source_fields if key in b}
            n = b["ix"].shape[0]
            owner = xp.zeros((n,), dtype=xp.int64, device=self.device)
            cuts = np.array([0, n], dtype=np.int64)
            group, row = self.energy_group[int(b["energy_ix"])]
            rows = xp.asarray([row], dtype=xp.int64, device=self.device)
            bank = self.banks[group]
            self.metrics["scalar_selection_packets"] += 1
            self.metrics["candidate_pairs"] += candidates
            self.metrics["selected_pairs"] += n
            self.metrics["max_batch_pairs"] = max(self.metrics["max_batch_pairs"], candidates)
        else:
            selected, owner, cuts, rows, bank = self.select(bixels)
        self.metrics["selection_submit_s"] += time.perf_counter() - start
        self.metrics["selection_packets"] += 1
        for b in bixels:
            # Do not retain whole candidate rays in the selected-work queue.
            b.pop("_source_ray", None)
            b.pop("_reference_ray", None)
        if not self.two_stage or (not self.ready and len(entries) >= self.batch_size):
            self.evaluate_selected(entries, selected, owner, cuts, rows, bank)
            return
        for b, args in entries:
            n = b["ix"].shape[0]
            group = self.energy_group[int(b["energy_ix"])][0]
            different = (
                self.ready and group != self.energy_group[int(self.ready[-1][0]["energy_ix"])][0]
            )
            if self.ready and (
                len(self.ready) >= self.batch_size
                or self.ready_pairs + n > self.max_pairs
                or different
            ):
                self.flush_ready()
            self.ready.append((b, args))
            self.ready_pairs += n
            if len(self.ready) >= self.batch_size:
                self.flush_ready()

    def flush_ready(self):
        """Pack the selected-work queue without retaining full candidate rays."""
        if not self.ready:
            return
        entries, self.ready = self.ready, []
        self.ready_pairs = 0
        xp = self.xp
        bixels = [b for b, _ in entries]
        selected = {
            key: concatenate(xp, [b[key] for b in bixels])
            for key in self.source_fields
            if key in bixels[0]
        }
        cuts = np.cumsum([0, *[b["ix"].shape[0] for b in bixels]], dtype=np.int64)
        owner = (
            xp.searchsorted(
                xp.asarray(cuts, device=self.device),
                xp.arange(int(cuts[-1]), dtype=xp.int64, device=self.device),
                side="right",
            )
            - 1
        )
        group = self.energy_group[int(bixels[0]["energy_ix"])][0]
        rows = xp.asarray(
            [self.energy_group[int(b["energy_ix"])][1] for b in bixels],
            dtype=xp.int64,
            device=self.device,
        )
        self.evaluate_selected(entries, selected, owner, cuts, rows, self.banks[group])

    def evaluate_selected(self, entries, selected, owner, cuts, rows, bank):
        """Interpolate, evaluate and collect one bounded selected-work packet."""
        xp = self.xp
        bixels = [b for b, _ in entries]
        self.metrics["max_selected_packet_pairs"] = max(
            self.metrics["max_selected_packet_pairs"], int(cuts[-1])
        )
        assert int(cuts[-1]) <= self.max_pairs
        start = time.perf_counter()
        if self.layout == "support":
            per_bixel = [self._interpolate_kernels_in_depth(b) for b in bixels]
            fields = {
                key: concatenate(xp, [f[key] for f in per_bixel], axis=-1) for key in per_bixel[0]
            }
            self.metrics["depth_native_groups"] += len(bixels)
        elif self._uses_bank:
            shifts = xp.asarray(
                [-b["rad_depth_offset"] for b in bixels],
                dtype=bank["end"].dtype,
                device=self.device,
            )
            fields = bank["depth"](
                selected["rad_depths"], xp.take(rows, owner), xp.take(shifts, owner)
            )
            self.metrics["depth_binary_calls"] += 1
        else:
            fields = self.grouped_native(selected["rad_depths"], cuts, bixels)
        self.metrics["interpolation_submit_s"] += time.perf_counter() - start
        start = time.perf_counter()
        sigma = xp.take(xp.stack([b["sigma_ini_sq"] for b in bixels]), owner)
        lateral = self._lateral_profile(
            selected["radial_dist_sq"], selected.get("lat_dists"), sigma, fields
        )
        factors = xp.asarray(
            [b["kernel"]["lateral_cut_off"]["comp_fac"] for b in bixels],
            dtype=lateral.dtype,
            device=self.device,
        )
        dose = xp.take(factors, owner) * lateral * fields["idd"]
        check_dose(xp, dose)
        quantities = {"physical_dose": dose}
        if self._calc_let:
            quantities["let_dose"] = dose * fields["let"]
        if self._calc_bio_dose:
            bio = []
            for i, b in enumerate(bixels):
                part = slice(int(cuts[i]), int(cuts[i + 1]))
                b["physical_dose"] = dose[part]
                bio.append(
                    self._evaluate_bio_influence(b, {k: v[..., part] for k, v in fields.items()})
                )
            for key in self._bio_influence_names:
                quantities[key] = concatenate(xp, [result[key] for result in bio])
        self.metrics["physics_submit_s"] += time.perf_counter() - start
        start = time.perf_counter()
        # Download each contiguous quantity and row vector once per packet.
        host = {key: to_numpy(value) for key, value in quantities.items()}
        host_ix = to_numpy(selected["ix"])
        for i, (b, args) in enumerate(entries):
            part = slice(int(cuts[i]), int(cuts[i + 1]))
            b["ix"] = host_ix[part]
            b.update({key: value[part] for key, value in host.items()})
            Hong._fill_dij(self, b, *args)
        self.metrics["collection_submit_s"] += time.perf_counter() - start
        self.batch_counts.append(len(bixels))


def sync(backend):
    """Synchronize only the full calculation timing boundaries."""
    if backend == "cupy":
        import cupy

        cupy.cuda.get_current_stream().synchronize()


def main():  # noqa: PLR0915
    """Run paired complete calculations and retain timings and comparison errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["numpy", "torch", "cupy"], default="numpy")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["native", "binary"],
        choices=[
            "physics",
            "support",
            "interpolation",
            "native",
            "binary",
            "bank",
            "bank_generic",
        ],
    )
    parser.add_argument("--mode", choices=["protons", "VHEE", "carbon"], default="protons")
    parser.add_argument("--machine", default="Generic")
    parser.add_argument("--bio-model", default="none")
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--max-pairs", type=int, default=65536)
    parser.add_argument("--grid", type=float, default=5)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--scenarios", action="store_true")
    parser.add_argument("--ct-phases", action="store_true")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--trace-on-ct-grid", action="store_true")
    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--out", default="benchmark/pb_audit_results/pipeline_batches")
    args = parser.parse_args()
    if min(args.size, args.max_pairs, args.pairs) < 1:
        raise ValueError("Positive batch size, budget and repeat count required")
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
        calc_let=True,
        trace_on_dose_grid=not args.trace_on_ct_grid,
    )
    if args.scenarios:
        from pyRadPlan.scenarios import (
            ScenarioSet,
            ExplicitScenarioModel,
            SetupUncertainty,
            RangeUncertainty,
        )

        plan.scenarios = ExplicitScenarioModel(
            scenario_set=ScenarioSet.from_points(
                (SetupUncertainty(), RangeUncertainty()),
                {"setup": [[0, 0, 0], [0, 0, 0], [2, 0, 0]], "range": [[0, 0], [2, 0], [0, 0]]},
            )
        )
    stf = generate_stf(ct, cst, plan)
    if args.ct_phases:
        # Generate identical steering on the original CT, then exercise two
        # different outlines/SSDs, range/setup realizations and a derived alias.
        import SimpleITK as sitk
        from pyRadPlan.ct import CT
        from pyRadPlan.scenarios import (
            ScenarioSet,
            ExplicitScenarioModel,
            SetupUncertainty,
            RangeUncertainty,
            CTPhaseUncertainty,
            ScenarioBasis,
        )

        moved = sitk.GetImageFromArray(
            np.roll(sitk.GetArrayFromImage(ct.cube_hu), 2, axis=(0, 1, 2))
        )
        moved.CopyInformation(ct.cube_hu)
        ct = CT(cube_hu=sitk.JoinSeries([ct.cube_hu, moved]))
        plan.scenarios = ExplicitScenarioModel(
            scenario_set=ScenarioSet.from_points(
                (
                    SetupUncertainty(),
                    RangeUncertainty(),
                    CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5]),
                ),
                {
                    "setup": [[0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0]],
                    "range": [[0, 0], [2, 0], [0, 0], [0, 0]],
                    "ct_phase": [0, 1, 1, 0],
                },
                bases=[None, None, None, ScenarioBasis(coefficients={0: 1.0})],
            )
        )
    settings.xp.prefer_gpu = args.backend == "cupy"
    settings.xp.preferred_cpu_array_backend = "torch" if args.backend == "torch" else "numpy"
    settings.xp.preferred_gpu_array_backend = "cupy"
    hardware = dict(cpu=platform.processor(), numpy=np.__version__)
    if args.backend == "cupy":
        import cupy

        cupy.arange(1)
        hardware.update(
            cupy=cupy.__version__, gpu=cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
        )
    elif args.backend == "torch":
        import torch

        hardware.update(torch=torch.__version__, threads=torch.get_num_threads())

    def run(variant=None, verify=False):
        if variant is None:
            engine = Hong(plan.model_copy(deep=True))
        elif variant == "physics":
            engine = BatchPrototype(
                plan.model_copy(deep=True),
                layout="packed",
                scope="beam",
                batch_size=args.size,
                max_pairs=args.max_pairs,
            )
        else:
            engine = PipelinePrototype(
                plan.model_copy(deep=True),
                layout=variant,
                scope="beam",
                batch_size=args.size,
                max_pairs=args.max_pairs,
                verify=verify,
                two_stage=args.two_stage,
            )
        engine.console_progress = False
        sync(args.backend)
        start = time.perf_counter()
        if args.forward:
            dose = engine.calc_dose_forward(
                ct,
                cst,
                stf.model_copy(deep=True),
                np.linspace(0.2, 1.4, stf.total_number_of_bixels),
            )
        else:
            dose = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
        sync(args.backend)
        elapsed = time.perf_counter() - start
        if args.ct_phases and not args.forward:
            for key in dose.quantities:
                slots = getattr(dose, key).ravel()
                assert slots[3] is slots[0], "Derived alias must retain shared storage"
        metrics = {} if variant is None else dict(engine.metrics)
        if variant is not None:
            metrics["batch_count"] = len(engine.batch_counts)
            metrics["batch_sizes"] = engine.batch_counts
            assert metrics.get("max_batch_pairs", 0) <= args.max_pairs
        return elapsed, dose, metrics

    first, reference, _ = run()
    results = {}
    for variant in args.variants:
        _, warm, _ = run(variant, verify=True)
        compare_dose(reference, warm)
        samples = {"scalar": [], "batch": []}
        for i in range(args.pairs):
            outputs = {}
            for name in ("scalar", "batch") if i % 2 == 0 else ("batch", "scalar"):
                elapsed, outputs[name], info = run(variant if name == "batch" else None)
                samples[name].append(elapsed)
                if name == "batch":
                    metrics = info
            errors = compare_dose(outputs["scalar"], outputs["batch"])
        results[variant] = dict(samples_s=samples, metrics=metrics, max_abs_errors=errors)
        print(variant, {k: round(float(np.median(v)), 4) for k, v in samples.items()}, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filename = f"{args.backend}_{args.mode}_{args.machine}_{args.grid:g}mm_b{args.size}_{args.bio_model}_{'scenarios' if args.scenarios else 'nominal'}_{'ct' if args.trace_on_ct_grid else 'dose'}{'_forward' if args.forward else ''}.json"
    if args.two_stage:
        filename = filename.replace(".json", "_staged.json")
    if args.ct_phases:
        filename = filename.replace(".json", "_phases.json")
    (out / filename).write_text(
        json.dumps(
            dict(
                reference="a780b35c scalar production path",
                config=vars(args),
                hardware=hardware,
                scalar_first_s=first,
                beamlets=stf.total_number_of_bixels,
                results=results,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Opt-in particle support, interpolation and physics packets on Array API backends."""

from collections import defaultdict
from dataclasses import dataclass
import math

import array_api_compat
import numpy as np

from ...core.xp_utils import ArrayBackend1DInterpolatorBank, to_numpy
from ._batching import BixelBatchCost, BixelBatchLimits, BixelBatchProcessor


def _concat(xp, arrays):
    return arrays[0] if len(arrays) == 1 else xp.concat(arrays, axis=0)


def _nbytes(xp, array):
    info = xp.iinfo(array.dtype) if xp.isdtype(array.dtype, "integral") else xp.finfo(array.dtype)
    return math.prod(array.shape) * (info.bits // 8)


def _bound(xp, value, queries):
    # A scalar torch bound is weakly promoted against float32 queries. Preserve
    # that scalar-selector behavior after gathering it into a non-scalar vector.
    return xp.astype(value, queries.dtype) if array_api_compat.is_torch_namespace(xp) else value


@dataclass
class _SelectedPacket:
    arrays: dict
    owner: object
    cuts: np.ndarray
    rows: object
    bank: dict


class ParticleBatchProcessor(BixelBatchProcessor):
    """Beam-scoped particle adapter; numerical specialization belongs to xp_utils.

    Selection and evaluation have independent budgets. A slice can retain a
    complete selected packet; byte costs deliberately count that full storage
    per queued item, plus conservative per-entry scratch estimates. Shared beam
    geometry/banks and allocator behavior remain outside these queue estimates.
    """

    def __init__(self, engine, dij, stf, energies):
        super().__init__(
            BixelBatchLimits(
                engine.batch_size, engine.batch_selection_max_entries, engine.batch_max_bytes
            ),
            BixelBatchLimits(
                engine.batch_size, engine.batch_evaluation_max_entries, engine.batch_max_bytes
            ),
        )
        self.engine, self.dij, self.stf = engine, dij, stf
        self.xp, self.device = engine.xp, engine.device
        self.groups, self.banks = {}, {}
        self.fields = ["ix", "rad_depths", "radial_dist_sq", "lat_dists"]
        if engine._calc_bio_dose:
            self.fields += ["v_alpha_x", "v_beta_x", "v_tissue_class_ix"]
        self._prepare_banks(energies)

    def _prepare_banks(self, energies):
        engine, xp = self.engine, self.xp
        grouped = defaultdict(list)
        for energy in energies:
            ix = np.int64(engine._machine.get_energy_index(energy, 4))
            key = (ix, xp)
            kernel = engine._kernel_cache.get(key)
            if kernel is None:
                kernel = engine._machine.get_kernel_by_index(ix).to_namespace(
                    xp, device=self.device
                )
                engine._kernel_cache[key] = kernel
            stub = {"kernel": kernel}
            if engine._calc_bio_dose:
                stub["v_alpha_x"] = None
            grid, fields = engine._depth_interpolation_data(stub)
            cutoff = kernel["lateral_cut_off"]
            if cutoff.get("cutoff_vals_sq") is None:
                cutoff["cutoff_vals_sq"] = cutoff["cut_off"] ** 2
            cx = cutoff["depths"][:1] if cutoff["cut_off"].shape[0] == 1 else cutoff["depths"]
            signature = (
                grid.dtype,
                cx.dtype,
                cutoff["cutoff_vals_sq"].dtype,
                tuple((k, v.dtype, v.shape[:-1]) for k, v in fields.items()),
            )
            grouped[signature].append((int(ix), kernel, grid, fields, cx))
        for group, entries in enumerate(grouped.values()):
            for row, (ix, *_) in enumerate(entries):
                self.groups[ix] = group, row
            fields = [entry[3] for entry in entries]
            self.banks[group] = dict(
                depth=ArrayBackend1DInterpolatorBank(
                    [e[2] for e in entries], fields, prefer_native=True
                ),
                cutoff=ArrayBackend1DInterpolatorBank(
                    [e[4] for e in entries],
                    [{"cutoff": e[1]["lateral_cut_off"]["cutoff_vals_sq"]} for e in entries],
                    prefer_native=True,
                ),
                end=xp.stack([e[1]["depths"][-1] for e in entries]),
                lower=xp.stack([e[4][0] for e in entries]),
                upper=xp.stack([e[4][-1] for e in entries]),
                constant=xp.asarray([e[4].shape[0] == 1 for e in entries], device=self.device),
                constant_value=xp.stack(
                    [e[1]["lateral_cut_off"]["cutoff_vals_sq"][0] for e in entries]
                ),
                scratch_per_entry=256
                + 16
                * sum(
                    math.prod(v.shape[:-1]) * xp.finfo(v.dtype).bits // 8
                    for v in fields[0].values()
                ),
            )

    def prepare(self, ray, bixel_index):
        """Snapshot scalar beamlet metadata and retain immutable scenario-ray arrays."""
        bixel = self.engine._prepare_bixel(ray, bixel_index)
        bixel["_batch_ray"] = ray
        return bixel

    def compatibility_key(self, work, stage):
        """Group only compatible fields and grid dtypes, using host metadata."""
        return self.groups[int(work.payload["energy_ix"])][0]

    def cost(self, work, stage):
        """Bound entry counts and optionally estimate retained plus scratch bytes."""
        bixel = work.payload
        source = bixel["_batch_ray"] if stage == "selection" else bixel
        count = source["rad_depths"].shape[0]
        nbytes = 0
        if self.engine.batch_max_bytes is not None:
            nbytes = sum(_nbytes(self.xp, source[k]) for k in self.fields if k in source)
            info = bixel.get("_batch_packet")
            if info is not None:
                packet = info[0]
                nbytes = max(
                    nbytes,
                    sum(_nbytes(self.xp, a) for a in packet.arrays.values())
                    + _nbytes(self.xp, packet.owner)
                    + _nbytes(self.xp, packet.rows),
                )
            bank = self.banks[self.compatibility_key(work, stage)]
            nbytes += count * bank["scratch_per_entry"]
        return BixelBatchCost(count, nbytes)

    def select_one(self, work):
        """Use the original selector for singleton or oversized candidate work."""
        bixel = work.payload
        self.engine._get_bixel_indices_on_ray(bixel, bixel.pop("_batch_ray"))

    def select(self, batch):
        """Select exact support for a bounded packet, compacting each field once."""
        xp = self.xp
        bixels = [work.payload for work in batch]
        offsets = np.cumsum(
            [0, *[b["_batch_ray"]["rad_depths"].shape[0] for b in bixels]], dtype=np.int64
        )
        unique, starts, size = {}, [], 0
        for b in bixels:
            ray = b["_batch_ray"]
            if id(ray) not in unique:
                unique[id(ray)] = ray, size
                size += ray["rad_depths"].shape[0]
            starts.append(unique[id(ray)][1])
        sources = [ray for ray, _ in unique.values()]
        source = {
            k: _concat(xp, [ray[k] for ray in sources]) for k in self.fields if k in sources[0]
        }
        positions = xp.arange(int(offsets[-1]), dtype=xp.int64, device=self.device)
        device_offsets = xp.asarray(offsets, device=self.device)
        owner = xp.searchsorted(device_offsets, positions, side="right") - 1
        source_ix = (
            positions
            - xp.take(device_offsets, owner)
            + xp.take(xp.asarray(starts, dtype=xp.int64, device=self.device), owner)
        )
        q, r2 = (
            xp.take(source["rad_depths"], source_ix),
            xp.take(source["radial_dist_sq"], source_ix),
        )
        rows = self._rows(bixels)
        table_ids = xp.take(rows, owner)
        bank = self.banks[self.groups[int(bixels[0]["energy_ix"])][0]]
        shifts = xp.asarray(
            [b["kernel"]["offset"] - b["rad_depth_offset"] for b in bixels],
            dtype=bank["end"].dtype,
            device=self.device,
        )
        shift = xp.take(shifts, owner)
        keep = q <= _bound(xp, xp.take(bank["end"], table_ids) + shift, q)
        if self.engine.dosimetric_lateral_cutoff != 1:
            radius_sq = bank["cutoff"](q, table_ids, grid_offsets=shift)["cutoff"]
            lower = _bound(xp, xp.take(bank["lower"], table_ids) + shift, q)
            upper = _bound(xp, xp.take(bank["upper"], table_ids) + shift, q)
            constant = xp.take(bank["constant"], table_ids)
            constant_value = xp.astype(xp.take(bank["constant_value"], table_ids), r2.dtype)
            keep &= xp.where(
                constant, constant_value >= r2, (radius_sq >= r2) & (q >= lower) & (q <= upper)
            )
        picked = xp.nonzero(keep)[0]
        selected_owner = xp.take(owner, picked)
        selected_source = xp.take(source_ix, picked)
        arrays = {k: xp.take(v, selected_source, axis=0) for k, v in source.items()}
        cuts = to_numpy(
            xp.searchsorted(
                selected_owner, xp.arange(len(bixels) + 1, dtype=xp.int64, device=self.device)
            )
        )
        packet = _SelectedPacket(arrays, selected_owner, cuts, rows, bank)
        for i, b in enumerate(bixels):
            b.update({k: v[int(cuts[i]) : int(cuts[i + 1]), ...] for k, v in arrays.items()})
            b.pop("_batch_ray")
            # Packet has no reference back to payloads: no reference cycle.
            b["_batch_packet"] = packet, i

    def _rows(self, bixels):
        return self.xp.asarray(
            [self.groups[int(b["energy_ix"])][1] for b in bixels],
            dtype=self.xp.int64,
            device=self.device,
        )

    def _selected_packet(self, bixels):
        info = [b.pop("_batch_packet", None) for b in bixels]
        if info[0] is not None:
            packet = info[0][0]
            if len(packet.cuts) == len(bixels) + 1 and all(
                entry is not None and entry[0] is packet and entry[1] == i
                for i, entry in enumerate(info)
            ):
                return packet
        xp = self.xp
        arrays = {k: _concat(xp, [b[k] for b in bixels]) for k in self.fields if k in bixels[0]}
        cuts = np.cumsum([0, *[b["ix"].shape[0] for b in bixels]], dtype=np.int64)
        owner = (
            xp.searchsorted(
                xp.asarray(cuts, device=self.device),
                xp.arange(int(cuts[-1]), dtype=xp.int64, device=self.device),
                side="right",
            )
            - 1
        )
        return _SelectedPacket(
            arrays,
            owner,
            cuts,
            self._rows(bixels),
            self.banks[self.groups[int(bixels[0]["energy_ix"])][0]],
        )

    def evaluate_one(self, work):
        """Avoid packet copies for singleton or oversized selected support."""
        work.payload.pop("_batch_packet", None)
        self.engine._calc_particle_bixel(work.payload)

    def evaluate(self, batch):
        """Evaluate shared interpolation/physics, retaining original per-beamlet biology."""
        engine, xp = self.engine, self.xp
        bixels = [work.payload for work in batch]
        packet = self._selected_packet(bixels)
        selected, owner, cuts, rows, bank = (
            packet.arrays,
            packet.owner,
            packet.cuts,
            packet.rows,
            packet.bank,
        )
        shifts = xp.asarray(
            [-b["rad_depth_offset"] for b in bixels], dtype=bank["end"].dtype, device=self.device
        )
        fields = bank["depth"](
            selected["rad_depths"], xp.take(rows, owner), grid_offsets=xp.take(shifts, owner)
        )
        sigma = xp.take(xp.stack([b["sigma_ini_sq"] for b in bixels]), owner)
        lateral = engine._lateral_profile(
            selected["radial_dist_sq"], selected.get("lat_dists"), sigma, fields
        )
        factors = xp.asarray(
            [b["kernel"]["lateral_cut_off"]["comp_fac"] for b in bixels],
            dtype=lateral.dtype,
            device=self.device,
        )
        dose = xp.take(factors, owner) * lateral * fields["idd"]
        if xp.any(xp.isnan(dose) | (dose < 0)):
            raise ValueError("Error in particle dose calculation.")
        quantities = {"physical_dose": dose}
        if engine._calc_let:
            quantities["let_dose"] = dose * fields["let"]
        if engine._calc_bio_dose:
            results = []
            for i, b in enumerate(bixels):
                part = slice(int(cuts[i]), int(cuts[i + 1]))
                b["physical_dose"] = dose[part]
                results.append(
                    engine._evaluate_bio_influence(b, {k: v[..., part] for k, v in fields.items()})
                )
            for k in engine._bio_influence_names:
                quantities[k] = _concat(xp, [result[k] for result in results])
        # Collection owns the packet download; do not download individual slices.
        bixels[0]["_batch_output"] = quantities, selected["ix"], cuts

    def collect(self, batch):
        """Download each packed quantity/row array once, then use existing assembly."""
        output = batch[0].payload.pop("_batch_output", None)
        if output is not None:
            quantities, indices, cuts = output
            host = {k: to_numpy(v) for k, v in quantities.items()}
            indices = to_numpy(indices)
            for i, work in enumerate(batch):
                part = slice(int(cuts[i]), int(cuts[i + 1]))
                work.payload["ix"] = indices[part]
                work.payload.update({k: v[part] for k, v in host.items()})
        self.engine._collect_bixel_batch(batch, self.dij, self.stf)

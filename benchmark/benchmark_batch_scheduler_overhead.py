"""Replay captured TG119 packet boundaries without numerical arrays or GPU work."""

import argparse
from collections import defaultdict
import json
import logging
from numbers import Integral
from pathlib import Path
import sys
from types import SimpleNamespace
import time
from unittest.mock import patch

import numpy as np

from pyRadPlan import IonPlan, generate_stf, load_tg119, settings
from pyRadPlan.dose.engines import (
    BixelBatchCost,
    BixelBatchLimits,
    BixelBatchProcessor,
    BixelDelivery,
)
from pyRadPlan.dose.engines import _batching
from benchmark_particle_pipeline import HookProcessor, HookPrototype, PipelinePrototype, Hong
import benchmark_particle_pipeline as pipeline
from benchmark_particle_batches import compare_dose


def checked_integer(value, name, minimum):
    """Freeze the original validation implementation from 582bb77a."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


class CheckedHookPrototype(HookPrototype):
    """Run the same hooks with the original integer validation for paired timings."""

    def calc_dose_influence(self, *args, **kwargs):
        """Replace only metadata validation during this complete calculation."""
        with patch.object(_batching, "_integer", checked_integer):
            return super().calc_dose_influence(*args, **kwargs)


class CaptureProcessor(HookProcessor):
    """Record host sizes and identities during one real numerical calculation."""

    def cost(self, work, stage):
        """Record both costs without downloading numerical values."""
        result = super().cost(work, stage)
        row = self.engine.records.setdefault(
            work.delivery.column,
            dict(
                delivery=list(vars(work.delivery).values()),
                group=self.compatibility_key(work, stage),
            ),
        )
        row[stage] = result.entries
        return result

    def evaluate(self, batch):
        """Record the actual selected-work packet identities."""
        self.engine.packets.append([work.delivery.column for work in batch])
        return super().evaluate(batch)

    def evaluate_one(self, work):
        """Record scalar evaluation too."""
        self.engine.packets.append([work.delivery.column])
        return super().evaluate_one(work)


class CaptureEngine(HookPrototype):
    """Use the real particle adapter with host metadata recording."""

    def _create_bixel_batch_processor(self, dij, stf):
        return CaptureProcessor(self, dij, stf)


def capture():
    """Capture the same nominal one-beam case as the GPU overhead comparison."""
    settings.xp.prefer_gpu = False
    settings.xp.preferred_cpu_array_backend = "numpy"
    ct, cst = load_tg119()
    plan = IonPlan(radiation_mode="protons")
    plan.prop_stf = dict(bixel_width=10, gantry_angles=[0], couch_angles=[0])
    plan.prop_dose_calc = dict(dose_grid={"resolution": [5] * 3}, calc_let=True)
    stf = generate_stf(ct, cst, plan)
    engine = CaptureEngine(plan, layout="bank", scope="beam", batch_size=64, max_pairs=65536)
    engine.selection_pairs, engine.batch_bytes = 65536, None
    engine.records, engine.packets = {}, []
    engine.console_progress = False
    actual = engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))
    scalar = Hong(plan)
    scalar.console_progress = False
    errors = compare_dose(actual, scalar.calc_dose_influence(ct, cst, stf.model_copy(deep=True)))
    return list(engine.records.values()), engine.packets, errors


def payload(row):
    """Replace arrays with shape metadata, sharing the same setup in both replays."""
    return dict(
        _source_ray={"rad_depths": row["candidate_shape"]},
        ix=row["selected_shape"],
        energy_ix=row["delivery"][-1],
    )


class LegacyReplay(PipelinePrototype):
    """Reuse the old prototype's actual queues; replace only numerical callbacks."""

    def __init__(self, rows):
        # Avoid dose initialization: only the original queue methods are used.
        self.batch_size, self.max_pairs, self.two_stage = 64, 65536, True
        self.pending, self.ready = [], []
        self.pending_pairs = self.ready_pairs = 0
        self.scope, self.xp = "beam", None
        self.energy_group = {r["delivery"][-1]: (r["group"], 0) for r in rows}
        self.metrics, self.packets = defaultdict(float), []

    def select(self, bixels):
        """Support sizes are already captured; all packets in this case are multi-item."""
        assert len(bixels) > 1
        return None, None, None, None, None

    def evaluate_entries(self, entries, *, collect=True):
        """Record evaluated/collected identities without physics or transfers."""
        self.packets.append([b["energy_ix"] for b, _ in entries])

    def evaluate_selected(self, entries, *args, collect=True):
        """Handle the old scheduler's immediate-evaluation shortcut."""
        self.evaluate_entries(entries)


class HookReplay(BixelBatchProcessor):
    """Exercise production scheduling and metadata validation with no numerical work."""

    def __init__(self, rows):
        super().__init__(BixelBatchLimits(64, 65536), BixelBatchLimits(64, 65536))
        self.groups = {r["delivery"][-1]: r["group"] for r in rows}
        self.packets = []

    def prepare(self, row, index):
        """Build the same fake payload as the old scheduler replay."""
        return payload(row)

    def cost(self, work, stage):
        """Keep the real adapter's shape-based cost construction."""
        bixel = work.payload
        n = (
            bixel["_source_ray"]["rad_depths"].shape[0]
            if stage == "selection"
            else bixel["ix"].shape[0]
        )
        return BixelBatchCost(n)

    def compatibility_key(self, work, stage):
        """Use the captured field-schema group."""
        return self.groups[work.payload["energy_ix"]]

    def select(self, batch):
        """Release candidate metadata just as the numerical adapter does."""
        for work in batch:
            del work.payload["_source_ray"]

    def evaluate(self, batch):
        """Omit numerical work for this host-only measurement."""

    def collect(self, batch):
        """Record ordered delivery identities."""
        self.packets.append([work.delivery.column for work in batch])


def main():
    """Compare queue costs on identical captured work, outside GPU timing noise."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=101)
    parser.add_argument("--full-backend", choices=["numpy", "cupy"])
    parser.add_argument("--pairs", type=int, default=11)
    parser.add_argument("--out", default="benchmark/pb_audit_results/hook_overhead/replay")
    args = parser.parse_args()
    if min(args.repeats, args.pairs) < 1:
        parser.error("Repeat and pair counts must be positive")
    if args.full_backend:
        # Both labels now use the same shared hooks. Only validation differs.
        argv = [
            pipeline.__file__,
            "--backend",
            args.full_backend,
            "--size",
            "64",
            "--pairs",
            str(args.pairs),
            "--two-stage",
            "--variants",
            "bank",
            "bank_hooks",
            "--paired-variants",
            "--out",
            args.out,
        ]
        with patch.object(sys, "argv", argv):
            with patch.object(pipeline, "PipelinePrototype", CheckedHookPrototype):
                pipeline.main()
        out = Path(args.out)
        (out / "comparison.json").write_text(
            json.dumps(
                dict(
                    bank="shared hooks with 582bb77a integer validation",
                    bank_hooks="shared hooks with built-in integer fast check",
                ),
                indent=2,
            )
            + "\n"
        )
        return
    logging.basicConfig(level=logging.ERROR)
    rows, packets, errors = capture()
    raw_rows = [dict(row) for row in rows]
    for row in rows:
        row["candidate_shape"] = SimpleNamespace(shape=(row["selection"],))
        row["selected_shape"] = SimpleNamespace(shape=(row["evaluation"],))

    fast_integer = _batching._integer

    def run(label):
        with patch.object(
            _batching, "_integer", fast_integer if label == "fast_int" else checked_integer
        ):
            start = time.perf_counter()
            if label == "legacy":
                processor = LegacyReplay(rows)
                for row in rows:
                    processor._fill_dij(payload(row))
                processor.flush()
                processor.flush_ready()
            else:
                processor = HookReplay(rows)
                with _batching.BixelBatchScheduler(processor) as scheduler:
                    for row in rows:
                        scheduler.submit(row, 0, BixelDelivery(*row["delivery"]))
            elapsed = time.perf_counter() - start
        assert processor.packets == packets
        return elapsed

    labels = ("legacy", "hooks", "fast_int")
    for _ in range(5):
        for label in labels:
            run(label)
    samples = {label: [] for label in labels}
    for i in range(args.repeats):
        for label in labels[:: 1 if i % 2 == 0 else -1]:
            samples[label].append(run(label))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "replay.json").write_text(
        json.dumps(
            dict(records=raw_rows, packets=packets, capture_errors=errors, samples_s=samples),
            indent=2,
        )
        + "\n"
    )
    print({k: float(np.median(v)) for k, v in samples.items()}, flush=True)


if __name__ == "__main__":
    main()

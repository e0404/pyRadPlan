"""Independent stage budgets, output identities and exception-safe queue lifetimes."""

from dataclasses import FrozenInstanceError

import pytest

from pyRadPlan.dose.engines import (
    BixelBatchProcessor,
    BixelBatchLimits,
    BixelBatchCost,
    BixelDelivery,
)
from pyRadPlan.dose.engines._batching import BixelBatchScheduler


class RecordingProcessor(BixelBatchProcessor):
    """Represent a voxel selection with host integers, independent of any modality."""

    def __init__(self, limits=None, evaluation=None, fail=None):
        limits = limits or BixelBatchLimits(16, 65536)
        super().__init__(limits, evaluation or limits)
        self.selected = []
        self.evaluated = []
        self.delivered = []
        self.fail = fail

    def prepare(self, ray, bixel_index):
        """Copy the fake beamlet metadata."""
        if self.fail == "prepare":
            raise RuntimeError("prepare")
        return dict(ray[bixel_index])

    def cost(self, work, stage):
        """Read the current fake storage size."""
        return BixelBatchCost(work.payload["entries"], work.payload.get("bytes", 0))

    def compatibility_key(self, work, stage):
        """Group fake schemas by their host key."""
        return work.payload.get("key", 0)

    def select(self, batch):
        """Replace candidate costs with selected costs."""
        self.selected.append(len(batch))
        for work in batch:
            work.payload["entries"] = work.payload.get("survivors", 0)
            work.payload["bytes"] = work.payload.get("selected_bytes", 0)
        if self.fail == "select":
            raise RuntimeError("select")

    def evaluate(self, batch):
        """Record the selected packet size."""
        if self.fail == "evaluate":
            raise RuntimeError("evaluate")
        self.evaluated.append(len(batch))

    def collect(self, batch):
        """Record original delivery identities."""
        if self.fail == "collect":
            raise RuntimeError("collect")
        self.delivered.extend(work.delivery for work in batch)


def submit(scheduler, records):
    """Include repeated columns in different scenarios and empty selected supports."""
    identities = [BixelDelivery(i % 3, 0, i // 3, i % 2, i // 3) for i in range(len(records))]
    for i, identity in enumerate(identities):
        scheduler.submit(records, i, identity)
    return identities


def test_selection_and_evaluation_have_independent_limits():
    """A VHEE-like candidate footprint must not serialize small selected packets."""
    processor = RecordingProcessor()
    with BixelBatchScheduler(processor) as scheduler:
        identities = submit(scheduler, [dict(entries=40000, survivors=2900)] * 121)
    assert processor.selected == [1] * 121
    assert processor.evaluated == [16] * 7 + [9]
    assert processor.delivered == identities
    assert scheduler.pending_counts == (0, 0)


def test_bytes_keys_oversized_items_and_empty_supports():
    """Preserve order through schema changes, oversized singleton work and byte limits."""
    processor = RecordingProcessor(BixelBatchLimits(10, 10, 10), BixelBatchLimits(10, 10, 5))
    records = [
        dict(entries=2, bytes=6, survivors=1, selected_bytes=3),
        dict(entries=2, bytes=6, survivors=1, selected_bytes=3),
        dict(entries=100, bytes=100, survivors=0),
        dict(entries=1, survivors=100, key=1),
        dict(entries=0, survivors=0, key=2),
        dict(entries=1, survivors=2, key=2),
    ]
    with BixelBatchScheduler(processor) as scheduler:
        expected = submit(scheduler, records)
    assert processor.delivered == expected
    assert sum(processor.selected) == sum(processor.evaluated) == len(records)
    assert processor.evaluated == [1, 2, 1, 2]
    with pytest.raises(FrozenInstanceError):
        expected[0].column = 9


@pytest.mark.parametrize("stage", ["prepare", "select", "evaluate", "collect"])
def test_failed_hooks_drop_both_queues(stage):
    """Never emit additional columns while unwinding a failed computation."""
    processor = RecordingProcessor(fail=stage)
    scheduler = BixelBatchScheduler(processor)
    with pytest.raises(RuntimeError, match=stage), scheduler:
        submit(scheduler, [dict(entries=1, survivors=1)] * 18)
    assert scheduler.pending_counts == (0, 0)
    assert not processor.delivered
    scheduler.close()
    with pytest.raises(RuntimeError, match="closed"):
        scheduler.submit([], 0, BixelDelivery(0, 0, 0, 0, 0))


def test_external_failure_discards_queued_payloads():
    """An interrupted ray/scenario loop must not flush partial pending work."""
    processor = RecordingProcessor(BixelBatchLimits(2, 20), BixelBatchLimits(8, 20))
    scheduler = BixelBatchScheduler(processor)
    with pytest.raises(KeyboardInterrupt), scheduler:
        submit(scheduler, [dict(entries=1, survivors=1)] * 3)
        assert scheduler.pending_counts == (1, 2)
        raise KeyboardInterrupt
    assert scheduler.pending_counts == (0, 0)
    assert not processor.delivered


@pytest.mark.parametrize(
    "kwargs", [{"max_bixels": 0}, {"max_entries": -1}, {"max_bytes": 0}, {"max_bixels": True}]
)
def test_invalid_budgets_rejected(kwargs):
    """Reject budgets that cannot bound a queue consistently."""
    with pytest.raises(ValueError):
        BixelBatchLimits(**kwargs)


def test_invalid_hook_results_are_not_silently_ignored():
    """Reject replacement/reordered outputs instead of silently dropping identities."""
    processor = RecordingProcessor(BixelBatchLimits(1, 20))
    processor.select = lambda batch: list(reversed(batch))
    scheduler = BixelBatchScheduler(processor)
    with pytest.raises(TypeError, match="return None"), scheduler:
        submit(scheduler, [dict(entries=1)])
    assert scheduler.pending_counts == (0, 0)


def test_oversized_work_uses_explicit_scalar_hooks():
    """Oversized selection and evaluation must never enter a packed callback."""
    processor = RecordingProcessor(BixelBatchLimits(4, 10))
    selected_one, evaluated_one = [], []

    def select_one(work):
        selected_one.append(work.delivery)
        work.payload["entries"] = work.payload["survivors"]

    def evaluate_one(work):
        evaluated_one.append(work.delivery)

    def reject_batch(batch):
        raise AssertionError("Oversized work must use its scalar hook")

    processor.select_one, processor.evaluate_one = select_one, evaluate_one
    processor.select = processor.evaluate = reject_batch
    with BixelBatchScheduler(processor) as scheduler:
        expected = submit(scheduler, [dict(entries=100, survivors=100)] * 3)
    assert selected_one == evaluated_one == processor.delivered == expected

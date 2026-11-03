"""Modality-independent, bounded scheduling for opt-in pencil-beam processors."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Integral
from typing import Any, Literal

BatchStage = Literal["selection", "evaluation"]


def _integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


@dataclass(frozen=True)
class BixelBatchLimits:
    """Bound one queue by beamlets, entries and optional estimated bytes.

    Entry/byte costs are supplied by the engine. Limits apply independently to
    candidate selection and selected-dose evaluation, not to total device memory.
    An individually oversized item uses the processor's single-item hook.
    """

    max_bixels: int = 64
    max_entries: int = 65536
    max_bytes: int | None = None

    def __post_init__(self):
        _integer(self.max_bixels, "max_bixels", 1)
        _integer(self.max_entries, "max_entries", 1)
        if self.max_bytes is not None:
            _integer(self.max_bytes, "max_bytes", 1)

    def exceeds(self, count, entries, nbytes):
        """Test accumulated costs without inspecting backend array values."""
        return (
            count > self.max_bixels
            or entries > self.max_entries
            or (self.max_bytes is not None and nbytes > self.max_bytes)
        )


@dataclass(frozen=True)
class BixelBatchCost:
    """Host-known work size and conservative retained/scratch byte estimate."""

    entries: int
    nbytes: int = 0

    def __post_init__(self):
        _integer(self.entries, "entries", 0)
        _integer(self.nbytes, "nbytes", 0)


@dataclass(frozen=True)
class BixelDelivery:
    """Original scenario and delivery-column identity, independent of packing."""

    scenario: int
    beam: int
    ray: int
    bixel: int
    column: int


@dataclass(frozen=True)
class BixelWork:
    """Engine-owned mutable payload attached to immutable delivery metadata."""

    payload: Any
    delivery: BixelDelivery


class BixelBatchProcessor(ABC):
    """Opt-in prepare/select/evaluate/collect hooks for one initialized beam.

    Hooks mutate payloads in place and return None, preserving tuple order and
    delivery identities. Selection must release unnecessary candidate references.
    Cost and compatibility keys use host metadata only. Keys must be hashable;
    include every schema/state distinction that prevents safe joint execution.

    Single-item hooks default to a singleton batch, but engines should override
    them when their scalar routines avoid expensive packing or handle oversized
    work. No generic scheduler can split arbitrary engine physics safely.
    """

    def __init__(self, selection_limits, evaluation_limits):
        if not isinstance(selection_limits, BixelBatchLimits) or not isinstance(
            evaluation_limits, BixelBatchLimits
        ):
            raise TypeError("Both stages require BixelBatchLimits.")
        self.selection_limits = selection_limits
        self.evaluation_limits = evaluation_limits

    @abstractmethod
    def prepare(self, ray, bixel_index):
        """Return an owned payload; snapshot mutable per-ray/scenario metadata."""

    @abstractmethod
    def cost(self, work: BixelWork, stage: BatchStage) -> BixelBatchCost:
        """Estimate entries and bytes from shapes/metadata, without device reductions."""

    def compatibility_key(self, work: BixelWork, stage: BatchStage):
        """Return a host key identifying items that this stage can process together."""
        return None

    @abstractmethod
    def select(self, batch: tuple[BixelWork, ...]) -> None:
        """Select exact support and release full candidate data when no longer needed."""

    @abstractmethod
    def evaluate(self, batch: tuple[BixelWork, ...]) -> None:
        """Evaluate selected work and retain results in each payload."""

    @abstractmethod
    def collect(self, batch: tuple[BixelWork, ...]) -> None:
        """Deliver results in original order; optionally transfer whole packets."""

    def select_one(self, work: BixelWork) -> None:
        """Select a singleton without requiring a multi-item allocation."""
        return self.select((work,))

    def evaluate_one(self, work: BixelWork) -> None:
        """Evaluate a singleton, including an inherently oversized beamlet."""
        return self.evaluate((work,))


class _Queue:
    def __init__(self, limits):
        self.limits = limits
        self.clear()

    def clear(self):
        self.items = []
        self.entries = self.nbytes = 0
        self.key = None

    def needs_flush(self, cost, key):
        return self.items and (
            key != self.key
            or self.limits.exceeds(
                len(self.items) + 1, self.entries + cost.entries, self.nbytes + cost.nbytes
            )
        )

    def append(self, work, cost, key):
        self.items.append(work)
        self.entries += int(cost.entries)
        self.nbytes += int(cost.nbytes)
        self.key = key

    def take(self):
        batch = tuple(self.items)
        self.clear()
        return batch


class BixelBatchScheduler:
    """Schedule two independent stages and discard pending work on failure.

    Use as a context manager within a beam's lifetime. Successful exit drains
    both queues before geometry/cache cleanup. Exceptional exit drops queued
    payloads without further numerical work or output collection. A closed or
    failed scheduler cannot be reused; the next beam/calculation creates a new one.
    """

    def __init__(self, processor: BixelBatchProcessor):
        self.processor = processor
        self._candidates = _Queue(processor.selection_limits)
        self._selected = _Queue(processor.evaluation_limits)
        self._closed = False

    @property
    def pending_counts(self):
        """Return the number of queued candidate and selected beamlets."""
        return len(self._candidates.items), len(self._selected.items)

    @staticmethod
    def _invoke(hook, batch):
        if hook(batch) is not None:
            raise TypeError("Batch hooks must mutate payloads in place and return None.")

    def _metadata(self, work, stage):
        cost = self.processor.cost(work, stage)
        if not isinstance(cost, BixelBatchCost):
            raise TypeError("Batch cost hooks must return BixelBatchCost.")
        key = self.processor.compatibility_key(work, stage)
        hash(key)  # Catch unhashable keys; processors must supply host metadata.
        return cost, key

    def submit(self, ray, bixel_index, delivery: BixelDelivery):
        """Prepare and enqueue one beamlet in its original execution order."""
        if self._closed:
            raise RuntimeError("Cannot submit to a closed batch scheduler.")
        try:
            work = BixelWork(self.processor.prepare(ray, bixel_index), delivery)
            cost, key = self._metadata(work, "selection")
            queue = self._candidates
            if queue.needs_flush(cost, key):
                self._flush_candidates()
            if queue.limits.exceeds(1, cost.entries, cost.nbytes):
                self._invoke(self.processor.select_one, work)
                self._enqueue_selected(work)
            else:
                queue.append(work, cost, key)
                if len(queue.items) == queue.limits.max_bixels:
                    self._flush_candidates()
        except BaseException:
            self.abort()
            raise

    def _flush_candidates(self):
        batch = self._candidates.take()
        if not batch:
            return
        if len(batch) == 1:
            self._invoke(self.processor.select_one, batch[0])
        else:
            self._invoke(self.processor.select, batch)
        for work in batch:
            self._enqueue_selected(work)

    def _enqueue_selected(self, work):
        cost, key = self._metadata(work, "evaluation")
        queue = self._selected
        if queue.needs_flush(cost, key):
            self._flush_selected()
        if queue.limits.exceeds(1, cost.entries, cost.nbytes):
            self._invoke(self.processor.evaluate_one, work)
            self._invoke(self.processor.collect, (work,))
        else:
            queue.append(work, cost, key)
            if len(queue.items) == queue.limits.max_bixels:
                self._flush_selected()

    def _flush_selected(self):
        batch = self._selected.take()
        if not batch:
            return
        if len(batch) == 1:
            self._invoke(self.processor.evaluate_one, batch[0])
        else:
            self._invoke(self.processor.evaluate, batch)
        self._invoke(self.processor.collect, batch)

    def close(self):
        """Finish pending work exactly once and release queue references."""
        if self._closed:
            return
        try:
            self._flush_candidates()
            self._flush_selected()
        finally:
            self.abort()

    def abort(self):
        """Discard pending work without evaluating or collecting it."""
        self._candidates.clear()
        self._selected.clear()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot enter a closed batch scheduler.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.abort()
        return False

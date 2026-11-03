"""The shared beam lifecycle preserves photon/particle sparse and forward output."""

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.dose.engines import (
    BixelBatchProcessor,
    BixelBatchLimits,
    BixelBatchCost,
    ParticleHongPencilBeamEngine,
    PhotonPencilBeamSVDEngine,
)
from pyRadPlan.scenarios import (
    ExplicitScenarioModel,
    ScenarioSet,
    SetupUncertainty,
    RangeUncertainty,
    ScenarioBasis,
)


class DeliveryProcessor(BixelBatchProcessor):
    """Delay collection of scalar-computed work to exercise a modality-neutral adapter."""

    def __init__(self, engine, dij, stf):
        super().__init__(BixelBatchLimits(3, 100000), BixelBatchLimits(7, 100000))
        self.engine, self.dij, self.stf = engine, dij, stf
        self.columns = []

    def prepare(self, ray, index):
        """Keep the existing modality-specific computation for this lifecycle check."""
        return self.engine._compute_bixel(ray, index)

    def cost(self, work, stage):
        """Use already-known support lengths, including empty columns."""
        return BixelBatchCost(work.payload["ix"].shape[0])

    def select(self, batch):
        """Keep the scalar engine's exact support."""

    def evaluate(self, batch):
        """Keep the scalar-computed quantities."""
        if getattr(self.engine, "fail_packet", False):
            raise RuntimeError("injected packet failure")

    def collect(self, batch):
        """Use immutable delivery identities through the original collector."""
        self.columns.extend(work.delivery for work in batch)
        self.engine._collect_bixel_batch(batch, self.dij, self.stf)


@pytest.mark.parametrize("modality", ["protons", "photons"])
@pytest.mark.parametrize("forward", [False, True])
def test_batch_lifecycle_matches_scalar(request, modality, forward):
    """Preserve scenario aliases, shifts, delivery columns and weighted dose."""
    plan, ct, cst, stf, *_ = request.getfixturevalue(f"test_data_{modality}")
    plan = plan.model_copy(deep=True)
    plan.scenarios = ExplicitScenarioModel(
        scenario_set=ScenarioSet.from_points(
            (SetupUncertainty(), RangeUncertainty()),
            {
                "setup": [[0, 0, 0], [0, 0, 0], [2, 0, 0], [0, 0, 0]],
                "range": [[0, 0], [2, 0], [0, 0], [0, 0]],
            },
            bases=[None, None, None, ScenarioBasis(coefficients={0: 1.0})],
        )
    )
    base = ParticleHongPencilBeamEngine if modality == "protons" else PhotonPencilBeamSVDEngine

    class HookEngine(base):
        def _create_bixel_batch_processor(self, dij, current_stf):
            processor = DeliveryProcessor(self, dij, current_stf)
            self.processors.append(processor)
            return processor

    scalar, hooked = base(plan), HookEngine(plan)
    hooked.processors = []
    for engine in (scalar, hooked):
        engine.console_progress = False
        if modality == "photons":
            engine.enable_dij_sampling = False

    def calculate(engine):
        if forward:
            return engine.calc_dose_forward(
                ct,
                cst,
                stf.model_copy(deep=True),
                np.linspace(0.2, 1.4, stf.total_number_of_bixels),
            )
        return engine.calc_dose_influence(ct, cst, stf.model_copy(deep=True))

    expected, actual = calculate(scalar), calculate(hooked)
    if forward:
        assert expected.keys() == actual.keys()
        for key, value in expected.items():
            values = value if isinstance(value, list) else [value]
            others = actual[key] if isinstance(actual[key], list) else [actual[key]]
            for a, b in zip(values, others):
                if isinstance(a, sitk.Image):
                    assert a.GetOrigin() == b.GetOrigin()
                    assert a.GetSpacing() == b.GetSpacing()
                    assert a.GetDirection() == b.GetDirection()
                    np.testing.assert_array_equal(
                        sitk.GetArrayFromImage(a), sitk.GetArrayFromImage(b)
                    )
                else:
                    np.testing.assert_array_equal(a, b)
    else:
        for key in expected.quantities:
            a, b = getattr(expected, key).flat, getattr(actual, key).flat
            assert b[3] is b[0]
            for x, y in zip(a, b):
                np.testing.assert_array_equal(x.indptr, y.indptr)
                np.testing.assert_array_equal(x.indices, y.indices)
                np.testing.assert_array_equal(x.data, y.data)
        for key in ("beam_num", "ray_num", "bixel_num"):
            np.testing.assert_array_equal(getattr(expected, key), getattr(actual, key))
    # Each geometry group/beam obtains a new processor, including engine reuse.
    first_count = len(hooked.processors)
    assert first_count == 2 * len(stf.beams)
    assert all(p.columns for p in hooked.processors)
    hooked.fail_packet = True
    with pytest.raises(RuntimeError, match="injected packet failure"):
        calculate(hooked)
    hooked.fail_packet = False
    calculate(hooked)
    assert len(hooked.processors) == 2 * first_count + 1

"""Tests for the outcome objectives.

``OutcomeModel`` is tested with a tiny torch model and a dummy preprocessor;
``load_model`` is patched, so no repo models are needed.
"""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.optimization.objectives import OutcomeObjective

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")

if _HAS_TORCH:
    # Limit torch threads to avoid Windows + CUDA-built torch crashes in pytest
    torch.set_num_threads(1)

    from pyRadPlan.ai.modelhub import OutcomePreprocessor

_MODULE = "pyRadPlan.optimization.objectives.outcome._outcome_model"

DIMS = (8, 8, 4)
N_VOXELS = int(np.prod(DIMS))


class FakeGrid:
    """Minimal Grid-like object exposing what set_dose_grid needs."""

    def __init__(self, dimensions=DIMS):
        self.dimensions = dimensions
        self.origin = np.zeros(3)
        self.resolution_vector = np.array([3.0, 3.0, 6.0])
        self.direction = np.eye(3)


if _HAS_TORCH:

    class DummyNet(torch.nn.Module):
        """A single logit depending smoothly on the dose."""

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([0.01]))

        def forward(self, dose):
            return (dose**2).mean() * self.w * 0.05

    class DummyPreprocessor(OutcomePreprocessor):
        """Model grid equals the dose grid."""

        def __init__(self, config=None):
            super().__init__(config)
            self.configured_with = None
            self.dims = None

        def configure(self, ct, cst, cst_masks=None, device=None):
            self.configured_with = (ct, cst, cst_masks, device)

        def set_dose_grid(self, dose_grid):
            self.dims = tuple(int(d) for d in dose_grid.dimensions)

        def preprocess(self, inputs):
            return (inputs,)

        def postprocess(self, outputs):
            return torch.sigmoid(outputs).sum()


def _make_obj(grid=None, **overrides):
    """Build an OutcomeModel with patched load_model and set it up like the optimizer."""
    from pyRadPlan.optimization.objectives import OutcomeModel

    dummies = (DummyNet().eval(), DummyPreprocessor())
    with patch(f"{_MODULE}.load_model", return_value=dummies):
        obj = OutcomeModel(**{"priority": 100.0, "model": "dummy", **overrides})
    obj.setup("ct", "cst")
    obj.preprocess_image_reference_parameters(target_grid=grid or FakeGrid())
    return obj


def _dose(seed=0):
    return np.random.default_rng(seed).random(N_VOXELS).astype(np.float32) * 50.0


class MeanDoseLogistic(OutcomeObjective):
    """Non-torch outcome model: logistic in mean dose."""

    name: str = "Mean Dose Logistic"

    def _predict(self, dose):
        return float(1.0 / (1.0 + np.exp(-(np.mean(dose) - 20.0) / 5.0)))

    def _predict_with_gradient(self, dose):
        p = self._predict(dose)
        return p, np.full_like(dose, p * (1.0 - p) / 5.0 / dose.size)


def test_base_class_works_without_torch_models():
    obj = MeanDoseLogistic(priority=1.0)
    index = np.arange(0, N_VOXELS, 2)
    obj.preprocess_image_reference_parameters(FakeGrid(), index_list=index)
    dose = np.full(index.size, 40.0)

    assert float(obj.compute_objective(dose)) == pytest.approx(1.0 / (1.0 + np.exp(0.0)))
    assert obj.compute_gradient(dose).shape == index.shape
    assert obj.report_info() == {"history": obj.history}


def test_import_objectives_does_not_load_torch():
    check = (
        "import sys, pyRadPlan.optimization.objectives\n"
        "assert 'torch' not in sys.modules, 'torch loaded'\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


@requires_torch
class TestOutcomeModel:
    def test_construction_freezes_weights(self):
        obj = _make_obj(warmup_iterations=2)
        assert obj.model == "dummy"
        assert not any(p.requires_grad for p in obj.nn_model.parameters())

    def test_setup_configures_preprocessor(self):
        obj = _make_obj(cst_masks=["Core"])
        ct, cst, masks, device = obj.preprocessor.configured_with
        assert (ct, cst, masks) == ("ct", "cst", ["Core"])
        assert device == next(obj.nn_model.parameters()).device
        assert obj.preprocessor.dims == DIMS

    def test_preloaded_model_tuple(self):
        from pyRadPlan.optimization.objectives import OutcomeModel

        model, pre = DummyNet(), DummyPreprocessor()
        obj = OutcomeModel(priority=1.0, model=(model, pre))
        obj.preprocess_image_reference_parameters(target_grid=FakeGrid())
        assert obj.model == "<preloaded>"
        assert obj.nn_model is model
        assert obj.preprocessor is pre
        assert obj.compute_objective(_dose()).ndim == 0

    def _load_device_used(self, **overrides):
        from pyRadPlan.optimization.objectives import OutcomeModel

        dummies = (DummyNet().eval(), DummyPreprocessor())
        with patch(f"{_MODULE}.load_model", return_value=dummies) as loader:
            OutcomeModel(priority=1.0, model="dummy", **overrides)
        return loader.call_args.kwargs["device"]

    def test_device_defaults_to_the_backend_preference(self, monkeypatch):
        import pyRadPlan

        monkeypatch.setattr(pyRadPlan.settings.xp, "prefer_gpu", False)
        assert self._load_device_used() == "cpu"

    def test_explicit_device_overrides_the_preference(self, monkeypatch):
        import pyRadPlan

        monkeypatch.setattr(pyRadPlan.settings.xp, "prefer_gpu", True)
        assert self._load_device_used(device="cpu") == "cpu"

    def test_preloaded_model_is_moved_only_when_a_device_is_given(self):
        from pyRadPlan.optimization.objectives import OutcomeModel

        moved = []

        def _make(**overrides):
            model = DummyNet().eval()
            original_to = model.to

            def _record(device):
                moved.append(str(device))
                return original_to(device)

            model.to = _record
            OutcomeModel(priority=1.0, model=(model, DummyPreprocessor()), **overrides)

        _make()
        assert moved == []
        _make(device="cpu")
        assert moved == ["cpu"]

    def test_compute_objective_returns_scalar_array(self):
        dose = _dose()
        val = _make_obj().compute_objective(dose)

        assert val.ndim == 0
        assert array_api_compat.array_namespace(val) is array_api_compat.array_namespace(dose)
        assert 0.0 <= float(val) <= 1.0

    def test_gradient_matches_analytic(self):
        obj = _make_obj()
        dose = _dose()
        grad = obj.compute_gradient(dose)

        c = 0.01 * 0.05
        p = 1.0 / (1.0 + np.exp(-c * np.mean(dose.astype(np.float64) ** 2)))
        expected = p * (1.0 - p) * c * 2.0 * dose / N_VOXELS
        np.testing.assert_allclose(grad, expected, rtol=1e-4)

    def test_gradient_leaves_no_weight_gradients(self):
        obj = _make_obj()
        obj.compute_gradient(_dose())
        assert all(p.grad is None for p in obj.nn_model.parameters())

    def test_monitor_only(self):
        obj = _make_obj(monitor_only=True)
        assert obj.compute_objective(_dose()) == 0.0
        assert np.all(obj.compute_gradient(_dose()) == 0.0)
        assert len(obj.history) == 1

    def test_warmup_then_active_gradient(self):
        obj = _make_obj(warmup_iterations=2)
        dose = _dose()

        assert np.all(obj.compute_gradient(dose) == 0.0)
        assert np.all(obj.compute_gradient(dose) == 0.0)
        grad = obj.compute_gradient(dose)
        assert grad.shape == (N_VOXELS,)
        assert grad.dtype == np.float32
        assert np.any(grad != 0.0)

    def test_voi_subset_is_scattered_with_zeros(self):
        obj = _make_obj()
        index = np.arange(0, N_VOXELS, 2)
        obj.preprocess_image_reference_parameters(target_grid=FakeGrid(), index_list=index)

        dose = _dose()
        full = np.zeros(N_VOXELS, dtype=np.float32)
        full[index] = dose[index]

        assert float(obj.compute_objective(dose[index])) == pytest.approx(
            float(_make_obj().compute_objective(full))
        )
        grad = obj.compute_gradient(dose[index])
        assert grad.shape == index.shape
        assert np.any(grad != 0.0)

    def test_num_of_fractions_scales_dose_and_gradient(self):
        obj = _make_obj(num_of_fractions=30)
        reference = _make_obj()
        dose = _dose() / 30.0

        assert float(obj.compute_objective(dose)) == pytest.approx(
            float(reference.compute_objective(dose * 30.0))
        )
        np.testing.assert_allclose(
            obj.compute_gradient(dose), 30.0 * reference.compute_gradient(dose * 30.0), rtol=1e-5
        )

    def test_history_restarts_per_optimization(self):
        obj = _make_obj(warmup_iterations=1)
        for _ in range(3):
            obj.compute_objective(_dose())
        obj.compute_gradient(_dose())
        assert len(obj.history) == 3
        assert obj.report_info() == {"history": obj.history}

        copy = obj.model_copy()
        copy.preprocess_image_reference_parameters(target_grid=FakeGrid())
        assert copy.history == []
        assert copy.report_info() == {}
        assert np.all(copy.compute_gradient(_dose()) == 0.0)  # warmup restarted
        assert len(obj.history) == 3


@requires_torch
def test_planning_problem_passes_planning_data(test_data_protons_raw):
    from pyRadPlan.ct import validate_ct
    from pyRadPlan.cst import validate_cst
    from pyRadPlan.dij import validate_dij
    from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
    from pyRadPlan.plan import validate_pln
    from pyRadPlan.stf import validate_stf

    tmp = test_data_protons_raw
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    for voi in cst.vois:
        voi.objectives = []
    cst.vois[0].objectives = [MeanDoseLogistic(priority=1.0)]

    seen = []
    with patch.object(MeanDoseLogistic, "setup", lambda self, ct, cst: seen.append((ct, cst))):
        prob = NonLinearFluencePlanningProblem(validate_pln(tmp["pln"]))
        prob._ct, prob._cst = ct, cst
        prob._dij, prob._stf = validate_dij(tmp["dij"]), validate_stf(tmp["stf"])
        prob._initialize()

    assert seen == [(ct, cst)]

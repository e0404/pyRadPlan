"""Tests for the OutcomeCNN objective.

The preprocessor now ships *with* the model (from HuggingFace), so these tests
use a tiny self-contained torch model and a dummy preprocessor implementing the
outcome contract, and patch ``load_model`` to return them. No repo models are
required.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

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

    from pyRadPlan.ai_models import BasePreprocessor

_CNN_MODULE = "pyRadPlan.optimization.objectives.model._cnn"

# Minimal preprocessing config (the dummy preprocessor barely uses it).
DUMMY_CONFIG = {
    "input_dimensions": [8, 8, 4],
    "input_spacing": [3.0, 3.0, 6.0],
    "center": "target",
    "type_order": ["dose"],
    "modality": {"dose": {"normalization_value": 70.0}},
}


# ---------------------------------------------------------------------------
# Fakes / dummies
# ---------------------------------------------------------------------------


class FakeGrid:
    """Minimal Grid-like object exposing what set_dose_grid needs."""

    def __init__(self, dimensions, origin=(0.0, 0.0, 0.0), resolution_vector=(3.0, 3.0, 6.0)):
        self.dimensions = dimensions
        self.origin = np.array(origin)
        self.resolution_vector = np.array(resolution_vector)
        self.direction = np.eye(3)


class FakeCT:
    def __init__(self, grid):
        self.grid = grid


class FakeCST:
    def __init__(self, target_com=(12.0, 12.0, 12.0)):
        self.vois = []
        self._target_com = np.array(target_com)

    def target_center_of_mass(self):
        return self._target_com


if _HAS_TORCH:

    class DummyNet(torch.nn.Module):
        """Trivial model: a single logit that depends linearly on the input.

        Uses the mean and a small weight so the logit stays in a range where the
        sigmoid gradient does not underflow to zero.
        """

        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.tensor([0.01]))

        def forward(self, dose):
            return dose.reshape(-1).mean() * self.w

    class DummyOutcomePreprocessor(BasePreprocessor):
        """Dummy preprocessor implementing the outcome contract.

        The "model grid" equals the dose grid, so preprocessing is just a
        namespace-agnostic conversion of the flat dose vector to a torch leaf.
        """

        def __init__(self, config=None):
            super().__init__(config)
            self.device = torch.device("cpu")
            self._dims = None
            self._dose_leaf = None

        def configure(self, ct, cst, cst_masks=None, device=None):
            self.device = torch.device(device) if device is not None else torch.device("cpu")

        def set_dose_grid(self, dose_grid):
            self._dims = tuple(int(d) for d in dose_grid.dimensions)

        def preprocess(self, inputs, requires_grad=False):
            from pyRadPlan.core import xp_utils

            t = xp_utils.to_namespace(torch, inputs).to(dtype=torch.float32).reshape(-1)
            if requires_grad:
                t = t.detach().clone().requires_grad_(True)
                self._dose_leaf = t
            else:
                self._dose_leaf = None
            return (t,)

        def postprocess(self, outputs):
            return torch.sigmoid(outputs).sum()

        @property
        def dose_leaf(self):
            return self._dose_leaf

        def gradient_to_dose_grid(self):
            return self._dose_leaf.grad.detach().reshape(-1).to(torch.float32)


N_VOXELS = 8 * 8 * 4


@pytest.fixture
def ct_and_cst():
    grid = FakeGrid(dimensions=(8, 8, 4))
    return FakeCT(grid), FakeCST()


def _make_obj(ct, cst, **overrides):
    """Build an OutcomeCNN with patched load_model returning dummies."""
    dummy_model = DummyNet()
    dummy_model.eval()
    dummy_pp = DummyOutcomePreprocessor(DUMMY_CONFIG)

    with patch(f"{_CNN_MODULE}.load_model", return_value=(dummy_model, dummy_pp)):
        from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

        params = dict(priority=100.0, model="dummy", mode="NTCP", ct=ct, cst=cst)
        params.update(overrides)
        obj = OutcomeCNN(**params)

    # Wire the dose grid the same way the optimizer would.
    obj.preprocess_image_reference_parameters(target_grid=FakeGrid(dimensions=(8, 8, 4)))
    return obj


@pytest.fixture
def outcome_obj(ct_and_cst):
    ct, cst = ct_and_cst
    return _make_obj(ct, cst, warmup_iterations=2)


def _dose():
    return np.random.rand(N_VOXELS).astype(np.float32) * 50.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@requires_torch
class TestOutcomeCNN:
    def test_construction(self, outcome_obj):
        assert outcome_obj.model == "dummy"
        assert outcome_obj.mode == "NTCP"
        assert outcome_obj.warmup_iterations == 2
        assert outcome_obj.nn_model is not None
        assert outcome_obj.preprocessor is not None

    def test_preloaded_model_tuple(self, ct_and_cst):
        """model= accepts a preloaded (model, preprocessor) tuple (no load_model)."""
        ct, cst = ct_and_cst
        dummy_model = DummyNet()
        dummy_pp = DummyOutcomePreprocessor(DUMMY_CONFIG)

        from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

        obj = OutcomeCNN(
            priority=1.0,
            model=(dummy_model, dummy_pp),
            mode="NTCP",
            ct=ct,
            cst=cst,
        )
        obj.set_dose_grid(FakeGrid(dimensions=(8, 8, 4)))
        assert obj.model == "<preloaded>"
        assert obj.nn_model is dummy_model
        assert obj.preprocessor is dummy_pp
        assert isinstance(obj.compute_objective(_dose()), float)

    def test_compute_objective_returns_float(self, outcome_obj):
        val = outcome_obj.compute_objective(_dose())
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_monitor_only_objective_is_zero(self, ct_and_cst):
        ct, cst = ct_and_cst
        obj = _make_obj(ct, cst, monitor_only=True)
        assert obj.compute_objective(_dose()) == 0.0
        # ...but the probability is still logged.
        assert len(obj.ntcp_history) == 1

    def test_monitor_only_gradient_is_zero(self, ct_and_cst):
        ct, cst = ct_and_cst
        obj = _make_obj(ct, cst, monitor_only=True)
        grad = obj.compute_gradient(_dose())
        assert np.all(grad == 0.0)

    def test_warmup_then_active_gradient(self, outcome_obj):
        dose = _dose()

        # First 2 calls are warmup -> zeros.
        grad1 = outcome_obj.compute_gradient(dose)
        assert grad1.shape == (N_VOXELS,)
        assert np.all(grad1 == 0.0)
        assert np.all(outcome_obj.compute_gradient(dose) == 0.0)

        # Third call: warmup over -> real, non-zero gradient.
        grad3 = outcome_obj.compute_gradient(dose)
        assert grad3.shape == (N_VOXELS,)
        assert grad3.dtype == np.float32
        assert np.any(grad3 != 0.0)

    def test_gradient_namespace_matches_input(self, ct_and_cst):
        """Gradient comes back in the same namespace as the dose input."""
        import array_api_compat

        ct, cst = ct_and_cst
        obj = _make_obj(ct, cst, warmup_iterations=0)
        dose = _dose()
        grad = obj.compute_gradient(dose)
        assert array_api_compat.is_numpy_array(grad)

    def test_tcp_mode_negates(self, ct_and_cst):
        ct, cst = ct_and_cst
        obj_ntcp = _make_obj(ct, cst, warmup_iterations=0, mode="NTCP")
        obj_tcp = _make_obj(ct, cst, warmup_iterations=0, mode="TCP")

        dose = _dose()
        assert obj_tcp.compute_objective(dose) == pytest.approx(
            -obj_ntcp.compute_objective(dose), abs=1e-6
        )

        grad_ntcp = obj_ntcp.compute_gradient(dose)
        grad_tcp = obj_tcp.compute_gradient(dose)
        np.testing.assert_allclose(grad_tcp, -grad_ntcp, atol=1e-6)

    def test_ntcp_history_and_reset(self, outcome_obj):
        dose = _dose()
        for _ in range(3):
            outcome_obj.compute_objective(dose)

        hist = outcome_obj.get_ntcp_history()
        assert len(hist) == 3
        assert all(isinstance(v, float) for v in hist)

        outcome_obj.reset_history()
        assert outcome_obj.ntcp_history == []
        assert outcome_obj.iteration_count == 0

    def test_report_info(self, outcome_obj):
        assert outcome_obj.report_info() == {}
        outcome_obj.compute_objective(_dose())
        assert "ntcp_history" in outcome_obj.report_info()

    def test_dose_grid_wiring(self, ct_and_cst):
        """preprocess_image_reference_parameters feeds the dose grid to the preprocessor."""
        ct, cst = ct_and_cst
        obj = _make_obj(ct, cst)
        obj.preprocess_image_reference_parameters(target_grid=FakeGrid(dimensions=(10, 10, 5)))
        assert obj.preprocessor._dims == (10, 10, 5)
        assert obj._dose_dims == (10, 10, 5)

    def test_not_initialised_raises(self):
        """Constructing without ct/cst then calling compute should raise."""
        with patch(f"{_CNN_MODULE}.load_model"):
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj = OutcomeCNN(model="dummy", mode="NTCP")

        with pytest.raises(RuntimeError, match="not initialised"):
            obj.compute_objective(np.zeros(N_VOXELS, dtype=np.float32))

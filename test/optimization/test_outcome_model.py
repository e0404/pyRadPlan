"""Tests for the outcome objectives.

``OutcomeModel`` is tested with a tiny torch model and a dummy preprocessor;
``load_model`` is patched, so no repo models are needed.
"""

from __future__ import annotations

import copy
import subprocess
import sys
from unittest.mock import patch

import array_api_compat
import array_api_strict
import numpy as np
import pytest

from pyRadPlan.core import xp_utils
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


class QuadraticOutcome(OutcomeObjective):
    """Backend-agnostic outcome model: ``1e-4 * sum(dose ** 2)``."""

    name: str = "Quadratic Outcome"

    def _predict(self, dose):
        xp = array_api_compat.array_namespace(dose)
        return 1e-4 * float(xp.sum(dose**2))

    def _predict_with_gradient(self, dose):
        return self._predict(dose), 2e-4 * dose


def _namespaces():
    namespaces = [np, array_api_strict]
    if _HAS_TORCH:
        import array_api_compat.torch as torch_xp

        namespaces.append(torch_xp)
    try:
        import jax.numpy as jnp
    except ImportError:
        pass
    else:
        namespaces.append(jnp)
    return namespaces


@pytest.fixture(params=_namespaces(), ids=lambda xp: xp.__name__)
def xp(request):
    return request.param


def test_partial_voi_value_and_gradient(xp):
    obj = QuadraticOutcome(priority=1.0)
    index = np.arange(1, N_VOXELS, 3)
    obj.preprocess_image_reference_parameters(FakeGrid(), index_list=index)
    values = np.linspace(1.0, 2.0, index.size, dtype=np.float32)
    dose = xp.asarray(values)

    value = obj.compute_objective(dose)
    grad = obj.compute_gradient(dose)

    assert array_api_compat.array_namespace(grad) is array_api_compat.array_namespace(dose)
    assert grad.dtype == dose.dtype
    assert array_api_compat.device(grad) == array_api_compat.device(dose)
    assert float(value) == pytest.approx(1e-4 * float(np.sum(values**2)), rel=1e-5)
    np.testing.assert_allclose(np.asarray(grad), 2e-4 * values, rtol=1e-5)


def test_partial_voi_with_fractions(xp):
    obj = QuadraticOutcome(priority=1.0, num_of_fractions=3)
    index = np.arange(0, N_VOXELS, 2)
    obj.preprocess_image_reference_parameters(FakeGrid(), index_list=index)
    values = np.linspace(1.0, 2.0, index.size, dtype=np.float32)
    dose = xp.asarray(values)

    assert float(obj.compute_objective(dose)) == pytest.approx(
        9e-4 * float(np.sum(values**2)), rel=1e-5
    )
    np.testing.assert_allclose(np.asarray(obj.compute_gradient(dose)), 18e-4 * values, rtol=1e-5)


def test_device_index_is_cached_until_the_dose_grid_is_set_again(xp):
    obj = QuadraticOutcome(priority=1.0)
    obj.preprocess_image_reference_parameters(FakeGrid(), index_list=np.arange(0, N_VOXELS, 2))
    dose = xp.asarray(np.ones(N_VOXELS // 2, dtype=np.float32))

    obj.compute_gradient(dose)
    (cached,) = obj._dose_index_cache.values()
    obj.compute_gradient(dose)
    assert next(iter(obj._dose_index_cache.values())) is cached

    obj.preprocess_image_reference_parameters(FakeGrid(), index_list=np.arange(0, N_VOXELS, 4))
    assert obj._dose_index_cache == {}
    smaller = xp.asarray(np.ones(N_VOXELS // 4, dtype=np.float32))
    assert obj.compute_gradient(smaller).shape == (N_VOXELS // 4,)


def test_warmup_value_and_gradient_are_consistent():
    obj = QuadraticOutcome(priority=1.0, warmup_iterations=2)
    obj.preprocess_image_reference_parameters(FakeGrid())
    dose = np.ones(N_VOXELS)

    for _ in range(2):
        assert float(obj.compute_objective(dose)) == 0.0
        assert float(obj.compute_objective(dose)) == 0.0  # line-search evaluation
        assert np.all(obj.compute_gradient(dose) == 0.0)

    assert float(obj.compute_objective(dose)) == pytest.approx(1e-4 * N_VOXELS)
    assert np.all(obj.compute_gradient(dose) != 0.0)
    assert obj.history == [pytest.approx(1e-4 * N_VOXELS)] * 5


def test_total_dose_convention_sets_the_fraction_number():
    user = QuadraticOutcome(priority=1.0)
    planned = user.model_copy()
    assert planned.normalize_to_fraction_num(30) is planned
    assert planned.num_of_fractions == 30
    assert user.num_of_fractions is None

    explicit = QuadraticOutcome(priority=1.0, num_of_fractions=5)
    explicit.normalize_to_fraction_num(30)
    assert explicit.num_of_fractions == 5


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


def test_planning_problem_passes_the_fraction_number_of_a_total_dose_plan(test_data_protons_raw):
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
    cst.vois[0].objectives = [QuadraticOutcome(priority=1.0)]

    prob = NonLinearFluencePlanningProblem(validate_pln(tmp["pln"]))
    prob._ct, prob._cst = ct, cst
    prob._dij, prob._stf = validate_dij(tmp["dij"]), validate_stf(tmp["stf"])
    prob._dose_convention, prob._num_of_fractions = "total", 30
    prob._initialize()

    ((_, (planned,)),) = prob._objective_list
    assert planned.num_of_fractions == 30
    assert cst.vois[0].objectives[0].num_of_fractions is None


@requires_torch
def test_unloadable_model_is_a_validation_error(tmp_path):
    from pydantic import ValidationError

    from pyRadPlan.optimization.objectives import OutcomeModel

    with pytest.raises(ValidationError, match="Cannot load outcome model"):
        OutcomeModel(priority=1.0, model=str(tmp_path / "missing"))


@requires_torch
def test_missing_or_mistyped_model_is_a_validation_error():
    from pydantic import ValidationError

    from pyRadPlan.optimization.objectives import OutcomeModel

    with pytest.raises(ValidationError, match="Field required"):
        OutcomeModel(priority=1.0)
    with pytest.raises(ValidationError, match="must be a model name"):
        OutcomeModel(priority=1.0, model=5)


@requires_torch
def test_validation_leaves_the_input_dict_untouched():
    from pyRadPlan.optimization.objectives import OutcomeModel

    preloaded = (DummyNet(), DummyPreprocessor())
    data = {"priority": 1.0, "model": preloaded}

    obj = OutcomeModel.model_validate(data)

    assert obj.model == "<preloaded>"
    assert data == {"priority": 1.0, "model": preloaded}


@requires_torch
def test_gradient_follows_dtype_and_device_of_the_dose():
    obj = _make_obj()
    dose = _dose().astype(np.float64)

    with patch(f"{_MODULE}.xp_utils.to_namespace", wraps=xp_utils.to_namespace) as convert:
        grad = obj.compute_gradient(dose)

    assert convert.call_args.kwargs["device"] == xp_utils.get_device_info(dose)
    assert grad.dtype == np.float64

    tensor = torch.as_tensor(dose)
    grad = obj.compute_gradient(tensor)
    assert grad.dtype == tensor.dtype
    assert grad.device == tensor.device


@requires_torch
def test_deepcopy_shares_the_model_but_not_the_data():
    obj = _make_obj(cst_masks=["Core"])
    clone = copy.deepcopy(obj)

    assert clone.nn_model is obj.nn_model
    assert clone.preprocessor is obj.preprocessor
    assert clone.cst_masks == ["Core"]
    assert clone.cst_masks is not obj.cst_masks


def test_objective_info_names_the_voi(test_data_protons_raw):
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

    prob = NonLinearFluencePlanningProblem(validate_pln(tmp["pln"]))
    prob._ct, prob._cst = ct, cst
    prob._dij, prob._stf = validate_dij(tmp["dij"]), validate_stf(tmp["stf"])
    prob._initialize()
    assert prob.collect_objective_info() == []

    obj = prob._objective_list[0][1][0]
    n = obj._dose_n if obj._dose_index is None else obj._dose_index.size
    obj.compute_objective(np.full(n, 20.0))

    assert prob.collect_objective_info() == [
        {"objective": "Mean Dose Logistic", "voi": cst.vois[0].name, "history": obj.history}
    ]


@requires_torch
def test_model_without_parameters_runs_on_the_cpu():
    class Parameterless(torch.nn.Module):
        def forward(self, dose):
            return dose.mean() * 0.01

    obj = _make_obj(model=(Parameterless(), DummyPreprocessor()))

    assert obj._model_device == torch.device("cpu")
    assert 0.0 < float(obj.compute_objective(_dose())) < 1.0

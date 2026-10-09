"""Tests for the OutcomeSklearn objective and the pyanno4rt outcome-model loader.

No repository model is required: the tests build a tiny two-segment geometry and a
LogisticRegression whose coefficients are set by hand, so every expected value can be
computed in closed form.
"""

from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.ai.modelhub import OutcomeFeature, OutcomeModelSpec, load_pyanno4rt_model
from pyRadPlan import settings
from pyRadPlan.core import Grid
from pyRadPlan.optimization.objectives import OutcomeSklearn
from pyRadPlan.optimization.objectives.model._features import (
    DOSE_FEATURES,
    FeatureExtractor,
    segment_area,
    segment_eccentricity,
    segment_sphericity,
    segment_volume,
)

sklearn = pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

GRID_SIZE = (6, 6, 4)  # sitk (x, y, z)
SPACING = (2.0, 2.0, 3.0)


def _image(array: np.ndarray) -> sitk.Image:
    """Wrap a (z, y, x) numpy array in an sitk image on the test grid."""
    image = sitk.GetImageFromArray(array.astype(np.uint8))
    image.SetSpacing(SPACING)
    image.SetOrigin((0.0, 0.0, 0.0))
    return image


class FakeVOI:
    def __init__(self, name, mask):
        self.name = name
        self.mask = mask


class FakeCST:
    def __init__(self, vois):
        self.vois = vois


class FakeCT:
    def __init__(self, grid):
        self.grid = grid


def _make_geometry():
    """Two disjoint segments: the left and right half of a small cube."""
    shape = GRID_SIZE[::-1]  # numpy (z, y, x)
    left = np.zeros(shape, dtype=np.uint8)
    right = np.zeros(shape, dtype=np.uint8)
    left[:, :, :3] = 1
    right[:, :, 3:] = 1

    reference = _image(np.zeros(shape))
    grid = Grid.from_sitk_image(reference)
    cst = FakeCST([FakeVOI("PAROTID_LT", _image(left)), FakeVOI("PAROTID_RT", _image(right))])
    return FakeCT(grid), cst, grid


def _features(functions=("Dose Mean", "Segment Volume")):
    """Feature list alternating over the two segments, mirroring the real model."""
    features = []
    for segment, prefix in (("PAROTID_LT", "left"), ("PAROTID_RT", "right")):
        for function in functions:
            features.append(
                OutcomeFeature(
                    name=f"{prefix}_{function.replace(' ', '')}",
                    segment=segment,
                    function=function,
                )
            )
    return features


def _logistic(coefficients, intercept=0.05):
    estimator = LogisticRegression()
    estimator.coef_ = np.asarray([coefficients], dtype=np.float64)
    estimator.intercept_ = np.asarray([intercept], dtype=np.float64)
    estimator.classes_ = np.asarray([0, 1])
    estimator.n_features_in_ = len(coefficients)
    return estimator


class WrappedEstimator:
    """Opaque estimator that forces the finite-difference gradient path."""

    def __init__(self, inner):
        self._inner = inner

    def predict_proba(self, features):
        return self._inner.predict_proba(features)


def _objective(estimator, features, **kwargs):
    kwargs.setdefault("num_of_fractions", 1)
    ct, cst, grid = _make_geometry()
    spec = OutcomeModelSpec(label="test", outcome_type="NTCP", features=features)
    objective = OutcomeSklearn(model=(estimator, spec), ct=ct, cst=cst, **kwargs)
    objective.preprocess_image_reference_parameters(grid, index_list=None)
    return objective, grid


def _dose(grid, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(5.0, 60.0, size=grid.num_voxels)


class FakePlan:
    def __init__(self, num_of_fractions):
        self.num_of_fractions = num_of_fractions


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _write_snapshot(tmp_path, estimator, columns, data_path, model_path="lrNTCP"):
    model_dir = tmp_path / "lrNTCP"
    model_dir.mkdir(exist_ok=True)
    with (model_dir / "predictor.save").open("wb") as handle:
        pickle.dump(estimator, handle)

    config = {
        "optimization": {
            "components": [
                {"Squared Overdosing": {"segment": "PAROTID_LT", "maximum_dose": 25}},
                {
                    "Logistic Regression Outcome": {
                        "segment": ["PAROTID_LT", "PAROTID_RT"],
                        "outcome_type": "NTCP",
                        "transform": True,
                        "weight": 1000,
                        "model": {
                            "label": "lrNTCP",
                            "model_path": model_path,
                            "dataset": {
                                "data_path": data_path,
                                "columns": [
                                    {
                                        "Dynamic Feature": {
                                            "column": name,
                                            "segment": segment,
                                            "function": function,
                                            "argument": None,
                                        }
                                    }
                                    for name, segment, function in columns
                                ]
                                + [{"Label": {"column": "grade"}}],
                            },
                        },
                    }
                },
            ]
        }
    }
    path = tmp_path / "input.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path, model_dir


JSON_COLUMNS = [
    ("left_volume", "PAROTID_LT", "Segment Volume"),
    ("right_volume", "PAROTID_RT", "Segment Volume"),
    ("left_dose", "PAROTID_LT", "Dose Mean"),
    ("right_dose", "PAROTID_RT", "Dose Mean"),
]


def test_loader_orders_features_by_training_table(tmp_path):
    """The estimator's coefficients bind to CSV column order, not JSON order."""
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("left_dose,left_volume,right_dose,right_volume,grade\n", encoding="utf-8")

    config, model_dir = _write_snapshot(
        tmp_path, _logistic([0.1, 0.2, 0.3, 0.4]), JSON_COLUMNS, str(csv_path)
    )

    estimator, spec = load_pyanno4rt_model(config)

    assert isinstance(estimator, LogisticRegression)
    assert [feature.name for feature in spec.features] == [
        "left_dose",
        "left_volume",
        "right_dose",
        "right_volume",
    ]
    assert spec.segments == ("PAROTID_LT", "PAROTID_RT")
    assert spec.transform is True
    assert spec.model_dir == model_dir


def test_loader_accepts_a_directory(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("left_dose,left_volume,right_dose,right_volume\n", encoding="utf-8")
    _write_snapshot(tmp_path, _logistic([0.1, 0.2, 0.3, 0.4]), JSON_COLUMNS, str(csv_path))

    _, spec = load_pyanno4rt_model(tmp_path)
    assert spec.features[0].name == "left_dose"


def test_loader_falls_back_to_json_order_with_warning(tmp_path, caplog):
    """A missing training table must be loud: the failure is otherwise silent."""
    config, _ = _write_snapshot(
        tmp_path, _logistic([0.1, 0.2, 0.3, 0.4]), JSON_COLUMNS, str(tmp_path / "absent.csv")
    )

    with caplog.at_level(logging.WARNING):
        _, spec = load_pyanno4rt_model(config)

    assert [feature.name for feature in spec.features] == [name for name, _, _ in JSON_COLUMNS]
    assert "feature order" in caplog.text


def test_loader_resolves_stale_model_path(tmp_path):
    """model_path is relative to pyanno4rt's old CWD; the snapshot layout must still work."""
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("left_dose,left_volume,right_dose,right_volume\n", encoding="utf-8")
    config, model_dir = _write_snapshot(
        tmp_path,
        _logistic([0.1, 0.2, 0.3, 0.4]),
        JSON_COLUMNS,
        str(csv_path),
        model_path="../gone/somewhere/lrNTCP",
    )

    _, spec = load_pyanno4rt_model(config)
    assert spec.model_dir == model_dir


def test_loader_rejects_feature_count_mismatch(tmp_path):
    config, _ = _write_snapshot(tmp_path, _logistic([0.1, 0.2]), JSON_COLUMNS, None)

    with pytest.raises(ValueError, match="expects 2 features"):
        load_pyanno4rt_model(config)


def test_loader_rejects_a_fitted_preprocessor(tmp_path):
    config, model_dir = _write_snapshot(
        tmp_path, _logistic([0.1, 0.2, 0.3, 0.4]), JSON_COLUMNS, None
    )
    (model_dir / "preprocessor.save").write_bytes(b"")

    with pytest.raises(NotImplementedError, match="feature preprocessor"):
        load_pyanno4rt_model(config)


def test_loader_requires_component_selection_when_ambiguous(tmp_path):
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("left_dose,left_volume,right_dose,right_volume\n", encoding="utf-8")
    config, _ = _write_snapshot(
        tmp_path, _logistic([0.1, 0.2, 0.3, 0.4]), JSON_COLUMNS, str(csv_path)
    )

    payload = json.loads(config.read_text(encoding="utf-8"))
    payload["optimization"]["components"].append(payload["optimization"]["components"][1])
    config.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="2 outcome components"):
        load_pyanno4rt_model(config)

    _, spec = load_pyanno4rt_model(config, component="lrNTCP")
    assert spec.label == "lrNTCP"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_dose_mean_value_and_gradient():
    values = np.array([1.0, 2.0, 3.0, 6.0])
    value_fn, gradient_fn = DOSE_FEATURES["Dose Mean"]

    assert value_fn(values, np, None) == pytest.approx(3.0)
    assert gradient_fn(values, np, None) == pytest.approx(np.full(4, 0.25))


def test_dose_order_statistics():
    values = np.array([10.0, 40.0, 20.0, 30.0])

    assert DOSE_FEATURES["Dose Maximum"][0](values, np, None) == pytest.approx(40.0)
    assert DOSE_FEATURES["Dose Minimum"][0](values, np, None) == pytest.approx(10.0)
    assert DOSE_FEATURES["Dose Maximum"][1](values, np, None) == pytest.approx([0, 1, 0, 0])
    # D50 of four voxels -> sorted index round(4 * 0.5) = 2 -> the value 30.
    assert DOSE_FEATURES["Dose Dx"][0](values, np, 50.0) == pytest.approx(30.0)
    assert DOSE_FEATURES["Dose Dx"][1](values, np, 50.0) == pytest.approx([0, 0, 0, 1])
    assert DOSE_FEATURES["Dose Vx"][0](values, np, 25.0) == pytest.approx(0.5)


def test_dose_deviation_gradient_matches_finite_differences():
    values = np.array([10.0, 25.0, 33.0, 47.0, 12.0])
    value_fn, gradient_fn = DOSE_FEATURES["Dose Deviation"]

    analytic = gradient_fn(values, np, None)
    numeric = np.empty_like(values)
    step = 1e-6
    for i in range(values.size):
        forward, backward = values.copy(), values.copy()
        forward[i] += step
        backward[i] -= step
        numeric[i] = (value_fn(forward, np, None) - value_fn(backward, np, None)) / (2 * step)

    assert analytic == pytest.approx(numeric, abs=1e-6)


def test_geometry_features_of_a_cube():
    """A 4x4x4 cube of 1 mm voxels has V=64, A=96 and the textbook sphericity 0.806."""
    mask = np.zeros((8, 8, 8), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    resolution = np.array([1.0, 1.0, 1.0])

    assert segment_volume(mask, resolution) == pytest.approx(64.0)
    assert segment_area(mask, resolution) == pytest.approx(96.0)
    assert segment_sphericity(mask, resolution) == pytest.approx(0.8060, abs=1e-3)
    # Isotropic extent -> equal principal moments -> zero eccentricity.
    assert segment_eccentricity(mask, resolution) == pytest.approx(0.0, abs=1e-9)


def test_segment_volume_scales_with_voxel_size():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True

    assert segment_volume(mask, np.array([2.0, 2.0, 3.0])) == pytest.approx(8 * 12.0)


def test_elongated_segment_has_non_zero_eccentricity():
    mask = np.zeros((10, 4, 4), dtype=bool)
    mask[1:9, 1:3, 1:3] = True

    assert segment_eccentricity(mask, np.array([1.0, 1.0, 1.0])) > 0.5


def test_extractor_reports_segment_means_and_geometry():
    ct, cst, grid = _make_geometry()
    extractor = FeatureExtractor(_features(), ct, cst)
    extractor.set_dose_grid(grid)

    dose = np.zeros(grid.num_voxels)
    volume = np.zeros(GRID_SIZE[::-1])
    volume[:, :, :3] = 10.0  # left segment
    volume[:, :, 3:] = 20.0  # right segment
    dose[:] = volume.reshape(-1)

    values = extractor.compute_values(dose, np)

    assert values.shape == (1, 4)
    assert values[0, 0] == pytest.approx(10.0)  # left mean dose
    assert values[0, 2] == pytest.approx(20.0)  # right mean dose
    # Each half holds 3 * 6 * 4 voxels of 2 x 2 x 3 mm.
    assert values[0, 1] == pytest.approx(72 * 12.0)
    assert extractor.dose_feature_indices == (0, 2)


def test_extractor_rejects_unknown_segment():
    ct, cst, _ = _make_geometry()
    features = [OutcomeFeature(name="x", segment="PAROTID_XX", function="Dose Mean")]

    with pytest.raises(ValueError, match="not in the structure set"):
        FeatureExtractor(features, ct, cst)


def test_extractor_maps_segment_names():
    ct, cst, grid = _make_geometry()
    features = [OutcomeFeature(name="x", segment="parotid_lt_alias", function="Dose Mean")]
    extractor = FeatureExtractor(features, ct, cst, segment_map={"parotid_lt_alias": "PAROTID_LT"})
    extractor.set_dose_grid(grid)

    assert extractor.compute_values(np.full(grid.num_voxels, 7.0), np)[0, 0] == pytest.approx(7.0)


def test_extractor_honours_static_feature_overrides():
    ct, cst, grid = _make_geometry()
    features = _features()
    extractor = FeatureExtractor(features, ct, cst, static_features={features[1].name: 1234.0})
    extractor.set_dose_grid(grid)

    assert extractor.compute_values(np.zeros(grid.num_voxels), np)[0, 1] == pytest.approx(1234.0)


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


def test_objective_matches_the_hand_computed_probability():
    coefficients = [0.03, -1e-4, 0.06, -1e-4]
    objective, grid = _objective(_logistic(coefficients), _features())
    dose = _dose(grid)

    probability = objective.compute_objective(dose)

    features = objective.extractor.compute_values(dose, np)
    logit = float((features @ np.asarray(coefficients))[0] + 0.05)
    assert probability == pytest.approx(1.0 / (1.0 + np.exp(-logit)))
    assert objective.ntcp_history == [pytest.approx(probability)]
    assert objective.report_info()["ntcp_history"] == [pytest.approx(probability)]


def test_gradient_matches_finite_differences_on_the_dose_vector():
    """The key correctness check: the analytic chain rule against a numeric derivative."""
    objective, grid = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), _features())
    dose = _dose(grid, seed=1)

    analytic = objective.compute_gradient(dose)

    step = 1e-4
    probed = np.flatnonzero(analytic)[::7]
    assert probed.size > 3
    for index in probed:
        forward, backward = dose.copy(), dose.copy()
        forward[index] += step
        backward[index] -= step
        numeric = (
            objective.compute_objective(forward) - objective.compute_objective(backward)
        ) / (2 * step)
        assert analytic[index] == pytest.approx(numeric, rel=1e-4, abs=1e-12)


def test_gradient_is_supported_only_on_the_segments():
    objective, grid = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), _features())
    gradient = objective.compute_gradient(_dose(grid))

    # Every voxel belongs to one of the two halves, so the gradient is dense and,
    # with positive dose coefficients, everywhere positive.
    assert gradient.shape == (grid.num_voxels,)
    assert np.all(gradient > 0.0)


def test_geometry_features_do_not_contribute_to_the_gradient():
    """Doubling the geometry coefficients must leave the dose gradient untouched."""
    objective_a, grid = _objective(_logistic([0.03, 0.0, 0.06, 0.0]), _features())
    dose = _dose(grid, seed=2)
    gradient_a = objective_a.compute_gradient(dose)

    objective_b, _ = _objective(_logistic([0.03, 0.0, 0.06, 0.0]), _features())
    objective_b.extractor._static = {1: 0.0, 3: 0.0}
    gradient_b = objective_b.compute_gradient(dose)

    assert gradient_a == pytest.approx(gradient_b)


def test_finite_difference_path_matches_the_analytic_one():
    coefficients = [0.03, -1e-4, 0.06, -1e-4]
    analytic, grid = _objective(_logistic(coefficients), _features())
    wrapped, _ = _objective(WrappedEstimator(_logistic(coefficients)), _features())
    dose = _dose(grid, seed=3)

    assert wrapped.compute_gradient(dose) == pytest.approx(
        analytic.compute_gradient(dose), rel=1e-5
    )
    assert wrapped.compute_objective(dose) == pytest.approx(analytic.compute_objective(dose))


def test_monitor_only_reports_but_does_not_steer():
    objective, grid = _objective(
        _logistic([0.03, -1e-4, 0.06, -1e-4]), _features(), monitor_only=True
    )
    dose = _dose(grid)

    assert objective.compute_objective(dose) == 0.0
    assert objective.ntcp_history  # still recorded
    assert np.all(objective.compute_gradient(dose) == 0.0)


def test_tcp_mode_flips_value_and_gradient():
    features = _features()
    ntcp, grid = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), features)
    tcp, _ = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), features, mode="TCP")
    dose = _dose(grid, seed=4)

    assert tcp.compute_objective(dose) == pytest.approx(-ntcp.compute_objective(dose))
    assert tcp.compute_gradient(dose) == pytest.approx(-ntcp.compute_gradient(dose))


def test_transform_switches_to_the_salu_branch():
    """Above p = 0.5 the value becomes 0.25 * logit(p) + 0.5 with a matching gradient."""
    coefficients = [0.03, -1e-4, 0.06, -1e-4]
    plain, grid = _objective(_logistic(coefficients), _features())
    transformed, _ = _objective(_logistic(coefficients), _features(), transform=True)

    # A high dose pushes the probability well above 0.5.
    high = np.full(grid.num_voxels, 60.0)
    probability = plain.compute_objective(high)
    assert probability > 0.5

    expected = 0.25 * np.log(probability / (1.0 - probability)) + 0.5
    assert transformed.compute_objective(high) == pytest.approx(expected)

    # In the linear branch the logistic derivative cancels: dvalue/dfeature = 0.25 * coef.
    ratio = transformed.compute_gradient(high) / plain.compute_gradient(high)
    assert ratio == pytest.approx(np.full_like(ratio, 0.25 / (probability - probability**2)))

    # Below the break point the transform is the identity.
    low = np.full(grid.num_voxels, 1.0)
    assert plain.compute_objective(low) < 0.5
    assert transformed.compute_objective(low) == pytest.approx(plain.compute_objective(low))


def test_uninitialised_objective_raises():
    spec = OutcomeModelSpec(label="test", features=_features())
    objective = OutcomeSklearn(model=(_logistic([0.1] * 4), spec))

    with pytest.raises(RuntimeError, match="not initialised"):
        objective.compute_objective(np.zeros(4))


def test_mode_defaults_to_the_spec_outcome_type():
    ct, cst, _ = _make_geometry()
    spec = OutcomeModelSpec(label="test", outcome_type="TCP", features=_features())

    objective = OutcomeSklearn(model=(_logistic([0.1] * 4), spec), ct=ct, cst=cst)
    assert objective.mode == "TCP"


def test_deepcopy_shares_the_estimator_and_extractor():
    """The optimizer deep-copies the whole cst; the runtime handles must survive that."""
    import copy

    objective, grid = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), _features())
    clone = copy.deepcopy(objective)

    assert clone.estimator is objective.estimator
    assert clone.extractor is objective.extractor
    assert clone.ntcp_history is not objective.ntcp_history

    dose = _dose(grid, seed=5)
    assert clone.compute_objective(dose) == pytest.approx(objective.compute_objective(dose))


def test_objective_serialises_without_runtime_handles():
    objective, _ = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), _features())
    dumped = objective.model_dump()

    assert dumped["model"] == "<preloaded>"
    assert "estimator" not in dumped
    assert "extractor" not in dumped


def test_loader_needs_no_trust_setting_for_a_direct_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.ai, "modelhub_trust_remote_code", False)
    config, _ = _write_snapshot(tmp_path, _logistic([0.1] * 4), JSON_COLUMNS, None)

    estimator, _ = load_pyanno4rt_model(config)
    assert estimator.n_features_in_ == 4


def test_loader_reads_the_number_of_fractions(tmp_path):
    config, _ = _write_snapshot(tmp_path, _logistic([0.1] * 4), JSON_COLUMNS, None)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["configuration"] = {"number_of_fractions": 35}
    config.write_text(json.dumps(raw), encoding="utf-8")

    _, spec = load_pyanno4rt_model(config)
    assert spec.num_of_fractions == 35


def test_loader_accepts_a_sav_predictor(tmp_path):
    config, model_dir = _write_snapshot(tmp_path, _logistic([0.1] * 4), JSON_COLUMNS, None)
    (model_dir / "predictor.save").rename(model_dir / "predictor.save")

    estimator, _ = load_pyanno4rt_model(config)
    assert estimator.n_features_in_ == 4


def test_model_sees_total_dose():
    """Per-fraction dose times the fraction count must give the same result as total dose."""
    coefficients = [0.03, -1e-4, 0.06, -1e-4]
    single, grid = _objective(_logistic(coefficients), _features())
    fractionated, _ = _objective(_logistic(coefficients), _features(), num_of_fractions=30)
    total = _dose(grid, seed=6)

    assert fractionated.compute_objective(total / 30) == pytest.approx(
        single.compute_objective(total)
    )
    assert fractionated.compute_gradient(total / 30) == pytest.approx(
        30 * single.compute_gradient(total)
    )


def test_fractionated_gradient_matches_finite_differences():
    objective, grid = _objective(
        _logistic([0.03, -1e-4, 0.06, -1e-4]), _features(), num_of_fractions=30
    )
    dose = _dose(grid, seed=7) / 30

    analytic = objective.compute_gradient(dose)
    step = 1e-5
    for index in np.flatnonzero(analytic)[::11]:
        forward, backward = dose.copy(), dose.copy()
        forward[index] += step
        backward[index] -= step
        numeric = (
            objective.compute_objective(forward) - objective.compute_objective(backward)
        ) / (2 * step)
        assert analytic[index] == pytest.approx(numeric, rel=1e-4, abs=1e-12)


def test_fraction_count_comes_from_pln_before_the_spec():
    ct, cst, _ = _make_geometry()
    spec = OutcomeModelSpec(label="test", features=_features(), num_of_fractions=30)

    from_spec = OutcomeSklearn(model=(_logistic([0.1] * 4), spec), ct=ct, cst=cst)
    from_pln = OutcomeSklearn(model=(_logistic([0.1] * 4), spec), ct=ct, cst=cst, pln=FakePlan(25))

    assert from_spec.num_of_fractions == 30
    assert from_pln.num_of_fractions == 25


def test_objective_value_is_an_array_like_the_dose():
    """The optimizer stacks objective values, which fails on plain floats with cupy."""
    objective, grid = _objective(_logistic([0.03, -1e-4, 0.06, -1e-4]), _features())
    dose = _dose(grid).astype(np.float32)

    value = objective.compute_objective(dose)

    assert isinstance(value, np.ndarray)
    assert value.shape == ()
    assert value.dtype == np.float32

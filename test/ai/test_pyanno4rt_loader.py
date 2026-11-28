"""Tests for the snapshot layouts understood by the pyanno4rt outcome-model loader."""

from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pytest

from pyRadPlan.ai.modelhub import load_pyanno4rt_model

pytest.importorskip("sklearn")
from sklearn.linear_model import LogisticRegression  # noqa: E402

LABEL = "lrNTCP"

JSON_COLUMNS = [
    ("left_volume", "PAROTID_LT", "Segment Volume"),
    ("right_volume", "PAROTID_RT", "Segment Volume"),
    ("left_dose", "PAROTID_LT", "Dose Mean"),
    ("right_dose", "PAROTID_RT", "Dose Mean"),
]
JSON_ORDER = [name for name, _, _ in JSON_COLUMNS]
TABLE_ORDER = ["left_dose", "left_volume", "right_dose", "right_volume"]


def _logistic(n_features=4, feature_names=None):
    estimator = LogisticRegression()
    estimator.coef_ = np.linspace(0.1, 0.4, n_features)[None, :]
    estimator.intercept_ = np.asarray([0.05])
    estimator.classes_ = np.asarray([0, 1])
    estimator.n_features_in_ = n_features
    if feature_names is not None:
        estimator.feature_names_in_ = np.asarray(feature_names, dtype=object)
    return estimator


def _write_snapshot(
    tmp_path,
    estimator=None,
    *,
    model_file="model.sav",
    configuration=None,
    input_steps=None,
    data_path=None,
):
    """Write a snapshot; ``configuration`` becomes ``<label>/configuration.json``."""
    model_dir = tmp_path / LABEL
    model_dir.mkdir(exist_ok=True)
    if model_file is not None:
        with (model_dir / model_file).open("wb") as handle:
            pickle.dump(estimator if estimator is not None else _logistic(), handle)
    if configuration is not None:
        (model_dir / "configuration.json").write_text(json.dumps(configuration), encoding="utf-8")

    model = {
        "label": LABEL,
        "model_path": LABEL,
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
                for name, segment, function in JSON_COLUMNS
            ]
            + [{"Label": {"column": "grade"}}],
        },
    }
    if input_steps is not None:
        model["preprocessing_steps"] = input_steps

    component = {"segment": ["PAROTID_LT", "PAROTID_RT"], "outcome_type": "NTCP", "model": model}
    config = {"optimization": {"components": [{"Logistic Regression Outcome": component}]}}
    (tmp_path / "input.json").write_text(json.dumps(config), encoding="utf-8")
    return model_dir


def _names(spec):
    return [feature.name for feature in spec.features]


def _public_configuration(feature_names=TABLE_ORDER, steps=("Identity",)):
    return {
        "feature_names": list(feature_names),
        "feature_values": [[0.0] * len(feature_names)],
        "label_name": "grade",
        "preprocessing_steps": steps,
    }


# --- model file -----------------------------------------------------------------------------


def test_loads_the_public_layout(tmp_path, caplog):
    model_dir = _write_snapshot(tmp_path, configuration=_public_configuration())

    with caplog.at_level(logging.WARNING):
        estimator, spec = load_pyanno4rt_model(tmp_path)

    assert isinstance(estimator, LogisticRegression)
    assert spec.model_dir == model_dir
    assert _names(spec) == TABLE_ORDER
    assert not caplog.records


def test_model_file_is_preferred_over_predictor_file(tmp_path):
    model_dir = _write_snapshot(tmp_path, configuration=_public_configuration())
    with (model_dir / "predictor.pkl").open("wb") as handle:
        pickle.dump(_logistic(2), handle)

    estimator, _ = load_pyanno4rt_model(tmp_path)
    assert estimator.n_features_in_ == 4


@pytest.mark.parametrize("name", ["model.h5", "predictor.onnx"])
def test_rejects_model_files_that_are_not_pickles(tmp_path, name):
    model_dir = _write_snapshot(tmp_path, model_file=None, configuration=_public_configuration())
    (model_dir / name).write_bytes(b"")

    with pytest.raises(NotImplementedError, match="not a pickled scikit-learn"):
        load_pyanno4rt_model(tmp_path)


def test_model_data_is_not_mistaken_for_the_model(tmp_path):
    model_dir = _write_snapshot(tmp_path, model_file=None)
    (model_dir / "model_data.csv").write_text("a,b\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="No model file"):
        load_pyanno4rt_model(tmp_path)


# --- preprocessing --------------------------------------------------------------------------


@pytest.mark.parametrize("steps", [["StandardScaler"], ["Identity", "Whitening"], "Whitening"])
def test_rejects_scaling_named_in_the_model_configuration(tmp_path, steps):
    _write_snapshot(tmp_path, configuration=_public_configuration(steps=steps))

    with pytest.raises(NotImplementedError, match=r"preprocessing steps \['(Standard|Whiten)"):
        load_pyanno4rt_model(tmp_path)


def test_rejects_scaling_named_in_the_plan_configuration(tmp_path):
    _write_snapshot(tmp_path, model_file="predictor.sav", input_steps=["StandardScaler"])

    with pytest.raises(NotImplementedError, match=r"\['StandardScaler'\].*input\.json"):
        load_pyanno4rt_model(tmp_path)


def test_rejects_scaling_if_the_sources_disagree(tmp_path):
    _write_snapshot(
        tmp_path, configuration=_public_configuration(), input_steps=["StandardScaler"]
    )

    with pytest.raises(NotImplementedError, match="StandardScaler"):
        load_pyanno4rt_model(tmp_path)


@pytest.mark.parametrize("steps", [[], ["Identity"]])
def test_identity_preprocessing_is_accepted_silently(tmp_path, caplog, steps):
    _write_snapshot(tmp_path, configuration=_public_configuration(steps=steps), input_steps=steps)

    with caplog.at_level(logging.WARNING):
        load_pyanno4rt_model(tmp_path)

    assert "scaling" not in caplog.text


def test_warns_if_no_source_states_the_preprocessing(tmp_path, caplog):
    _write_snapshot(tmp_path, model_file="predictor.sav")

    with caplog.at_level(logging.WARNING):
        load_pyanno4rt_model(tmp_path)

    assert "without feature scaling" in caplog.text


def test_preprocessor_file_is_rejected_in_the_public_layout(tmp_path):
    model_dir = _write_snapshot(tmp_path, configuration=_public_configuration())
    (model_dir / "preprocessor.sav").write_bytes(b"")

    with pytest.raises(NotImplementedError, match="feature preprocessor"):
        load_pyanno4rt_model(tmp_path)


# --- feature order --------------------------------------------------------------------------


def test_model_configuration_wins_over_the_training_table(tmp_path):
    table = tmp_path / "train.csv"
    table.write_text(",".join(JSON_ORDER[::-1]) + ",grade\n", encoding="utf-8")
    _write_snapshot(tmp_path, configuration=_public_configuration(), data_path=str(table))

    _, spec = load_pyanno4rt_model(tmp_path)
    assert _names(spec) == TABLE_ORDER


def test_estimator_feature_names_win_over_the_training_table(tmp_path):
    table = tmp_path / "train.csv"
    table.write_text(",".join(JSON_ORDER[::-1]) + ",grade\n", encoding="utf-8")
    _write_snapshot(
        tmp_path,
        _logistic(feature_names=TABLE_ORDER),
        model_file="predictor.sav",
        data_path=str(table),
    )

    _, spec = load_pyanno4rt_model(tmp_path)
    assert _names(spec) == TABLE_ORDER


def test_rejects_contradictory_recorded_feature_names(tmp_path):
    _write_snapshot(
        tmp_path,
        _logistic(feature_names=JSON_ORDER),
        configuration=_public_configuration(),
    )

    with pytest.raises(ValueError, match="contradictory"):
        load_pyanno4rt_model(tmp_path)


def test_rejects_recorded_names_without_a_configured_feature(tmp_path):
    names = ["left_dose", "age", "right_dose", "right_volume"]
    _write_snapshot(tmp_path, configuration=_public_configuration(names))

    with pytest.raises(
        ValueError, match=r"\['age'\].*configured but not listed \['left_volume'\]"
    ):
        load_pyanno4rt_model(tmp_path)


def test_filtered_features_are_dropped(tmp_path, caplog):
    names = ["right_dose", "left_dose"]
    _write_snapshot(tmp_path, _logistic(2), configuration=_public_configuration(names))

    with caplog.at_level(logging.WARNING):
        _, spec = load_pyanno4rt_model(tmp_path)

    assert _names(spec) == names
    assert "left_volume" in caplog.text


def test_training_table_with_byte_order_mark(tmp_path):
    table = tmp_path / "train.csv"
    table.write_text(",".join(TABLE_ORDER) + ",grade\n", encoding="utf-8-sig")
    _write_snapshot(tmp_path, model_file="predictor.sav", data_path=str(table))

    _, spec = load_pyanno4rt_model(tmp_path)
    assert _names(spec) == TABLE_ORDER


def test_rejects_a_training_table_without_the_configured_columns(tmp_path):
    table = tmp_path / "train.csv"
    table.write_text("left_dose;left_volume;right_dose;right_volume\n", encoding="utf-8")
    _write_snapshot(tmp_path, model_file="predictor.sav", data_path=str(table))

    with pytest.raises(ValueError, match="absent from the training table"):
        load_pyanno4rt_model(tmp_path)

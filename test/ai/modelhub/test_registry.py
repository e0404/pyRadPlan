import json

from pyRadPlan.ai.modelhub import (
    ModelTask,
    list_local_models,
    task_from_dir,
    task_from_name,
)
from pyRadPlan.ai.modelhub import _resolve, _registry


def _make_model_dir(path, metadata=None):
    path.mkdir(parents=True, exist_ok=True)
    for name in _resolve.REQUIRED_FILES:
        (path / name).write_text("placeholder", encoding="utf-8")
    if metadata is not None:
        (path / "model_config.json").write_text(
            json.dumps({"model_name": "X", "metadata": metadata}), encoding="utf-8"
        )


def test_task_from_name():
    assert task_from_name("dosecalc-bayes-protons") == ModelTask.DOSE_CALC
    assert task_from_name("outcome-ORPDenseNet-tg119") == ModelTask.OUTCOME
    assert task_from_name("something-else") is None


def test_task_from_name_ignores_the_organization():
    assert task_from_name("myfork/dosecalc-bayes-protons") == ModelTask.DOSE_CALC


def test_task_from_dir_prefers_declared_metadata(tmp_path):
    # the name says dose calc, the config says otherwise: the config wins
    model_dir = tmp_path / "dosecalc-mislabeled"
    _make_model_dir(model_dir, metadata={"task": "outcome"})
    assert task_from_dir(model_dir) == ModelTask.OUTCOME


def test_task_from_dir_accepts_spelling_variants(tmp_path):
    for declared in ("dose_calc", "dose-calc", "DoseCalc", " dose calc "):
        model_dir = tmp_path / f"m_{declared.strip().replace(' ', '_')}"
        _make_model_dir(model_dir, metadata={"task": declared})
        assert task_from_dir(model_dir) == ModelTask.DOSE_CALC


def test_radiation_modality_is_not_mistaken_for_a_task(tmp_path):
    """metadata.training.modality is the radiation type, not what the model does."""
    model_dir = tmp_path / "outcome-protons"
    _make_model_dir(model_dir, metadata={"training": {"modality": "protons"}})
    assert task_from_dir(model_dir) == ModelTask.OUTCOME


def test_task_from_dir_falls_back_to_name(tmp_path):
    model_dir = tmp_path / "outcome-undeclared"
    _make_model_dir(model_dir, metadata={"description": "no task here"})
    assert task_from_dir(model_dir) == ModelTask.OUTCOME

    unusable = tmp_path / "mystery-model"
    _make_model_dir(unusable, metadata={"task": "not-a-task"})
    assert task_from_dir(unusable) is None


def test_task_from_dir_survives_unreadable_config(tmp_path):
    model_dir = tmp_path / "dosecalc-broken"
    _make_model_dir(model_dir)
    (model_dir / "model_config.json").write_text("{not json", encoding="utf-8")
    assert task_from_dir(model_dir) == ModelTask.DOSE_CALC


def test_list_local_models_reports_full_ids(tmp_path, monkeypatch):
    _make_model_dir(tmp_path / "DKFZ-RadOpt" / "dosecalc-bayes-protons")
    _make_model_dir(tmp_path / "DKFZ-RadOpt" / "outcome-tg119")
    (tmp_path / "DKFZ-RadOpt" / "not-a-model").mkdir()  # missing the contract files

    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_registry, "_scan_hf_cache", lambda cfg: {})

    assert list_local_models() == [
        "DKFZ-RadOpt/dosecalc-bayes-protons",
        "DKFZ-RadOpt/outcome-tg119",
    ]
    assert list_local_models(ModelTask.DOSE_CALC) == ["DKFZ-RadOpt/dosecalc-bayes-protons"]
    assert list_local_models(ModelTask.OUTCOME) == ["DKFZ-RadOpt/outcome-tg119"]


def test_a_fork_does_not_shadow_its_upstream(tmp_path, monkeypatch):
    """Same repo name, different orgs: two entries, two directories."""
    _make_model_dir(tmp_path / "DKFZ-RadOpt" / "dosecalc-x", metadata={"task": "dose_calc"})
    _make_model_dir(tmp_path / "myfork" / "dosecalc-x", metadata={"task": "dose_calc"})

    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_registry, "_scan_hf_cache", lambda cfg: {})

    assert list_local_models() == ["DKFZ-RadOpt/dosecalc-x", "myfork/dosecalc-x"]


def test_hand_placed_flat_model_is_listed_under_its_bare_name(tmp_path, monkeypatch):
    _make_model_dir(tmp_path / "dosecalc-handplaced")

    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_registry, "_scan_hf_cache", lambda cfg: {})

    assert list_local_models() == ["dosecalc-handplaced"]


def test_list_local_models_filters_on_declared_task(tmp_path, monkeypatch):
    # a name that carries no prefix, but a config that declares the task
    _make_model_dir(tmp_path / "DKFZ-RadOpt" / "ORPDenseNet-tg119", metadata={"task": "outcome"})

    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    monkeypatch.setattr(_registry, "_scan_hf_cache", lambda cfg: {})

    assert list_local_models(ModelTask.OUTCOME) == ["DKFZ-RadOpt/ORPDenseNet-tg119"]
    assert list_local_models(ModelTask.DOSE_CALC) == []


def test_list_local_models_unset_dir_is_empty(monkeypatch):
    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", None)
    monkeypatch.setattr(_registry, "_scan_hf_cache", lambda cfg: {})
    assert list_local_models() == []


def test_list_local_models_includes_hf_cache(tmp_path, monkeypatch):
    _make_model_dir(tmp_path / "DKFZ-RadOpt" / "dosecalc-local")

    cfg = _registry.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    # a model that only exists in the HuggingFace cache (dedup + merge)
    monkeypatch.setattr(
        _registry,
        "_scan_hf_cache",
        lambda cfg: {
            "DKFZ-RadOpt/pyRadPlan-outcome-x": ModelTask.OUTCOME,
            "DKFZ-RadOpt/dosecalc-local": None,
        },
    )

    assert list_local_models() == [
        "DKFZ-RadOpt/dosecalc-local",
        "DKFZ-RadOpt/pyRadPlan-outcome-x",
    ]
    assert list_local_models(ModelTask.OUTCOME) == ["DKFZ-RadOpt/pyRadPlan-outcome-x"]
    # the local copy wins over the cached entry, so its task is the one used
    assert list_local_models(ModelTask.DOSE_CALC) == ["DKFZ-RadOpt/dosecalc-local"]

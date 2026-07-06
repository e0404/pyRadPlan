from pyRadPlan.ml import Modality, list_local_models, modality_from_name
from pyRadPlan.ml import _factory, _hub


def _make_model_dir(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in _hub.REQUIRED_FILES:
        (path / name).write_text("placeholder", encoding="utf-8")


def test_modality_from_name():
    assert modality_from_name("dosecalc-bayes-protons") == Modality.DOSE_CALC
    assert modality_from_name("outcome-ORPDenseNet-tg119") == Modality.OUTCOME
    assert modality_from_name("something-else") is None


def test_list_local_models(tmp_path, monkeypatch):
    _make_model_dir(tmp_path / "dosecalc-bayes-protons")
    _make_model_dir(tmp_path / "outcome-tg119")
    (tmp_path / "not-a-model").mkdir()  # missing the contract files

    cfg = _factory.get_ml_settings()
    monkeypatch.setattr(cfg, "local_models_dir", tmp_path)

    assert list_local_models() == ["dosecalc-bayes-protons", "outcome-tg119"]
    assert list_local_models(Modality.DOSE_CALC) == ["dosecalc-bayes-protons"]
    assert list_local_models(Modality.OUTCOME) == ["outcome-tg119"]


def test_list_local_models_unset_dir_is_empty(monkeypatch):
    cfg = _factory.get_ml_settings()
    monkeypatch.setattr(cfg, "local_models_dir", None)
    assert list_local_models() == []

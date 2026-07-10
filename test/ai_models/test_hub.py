import json
from pathlib import Path

import pytest

from pyRadPlan.ai_models import _hub, is_valid_model_dir, resolve_model_dir


def _make_model_dir(path: Path, metadata: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in _hub.REQUIRED_FILES:
        (path / name).write_text("placeholder", encoding="utf-8")
    if metadata is not None:
        (path / _hub.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


def test_is_valid_model_dir(tmp_path):
    assert not is_valid_model_dir(tmp_path)
    _make_model_dir(tmp_path)
    assert is_valid_model_dir(tmp_path)


def test_local_dir_used_without_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path)

    def _boom():
        raise AssertionError("snapshot_download must not be imported for a valid local dir")

    monkeypatch.setattr(_hub, "_import_snapshot_download", _boom)

    out = resolve_model_dir(local_dir=str(tmp_path))
    assert Path(out) == tmp_path


def test_version_dedup_skips_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path, metadata={"repo_id": "org/repo", "revision": "v1"})

    def _boom():
        raise AssertionError("matching version must not trigger a download")

    monkeypatch.setattr(_hub, "_import_snapshot_download", _boom)

    out = resolve_model_dir(repo_id="org/repo", revision="v1", local_dir=str(tmp_path))
    assert Path(out) == tmp_path


def test_offline_fallback_retries_local_only(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            raise RuntimeError("network unavailable")
        return str(tmp_path)

    monkeypatch.setattr(_hub, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo")
    assert Path(out) == tmp_path
    assert calls == [False, True]


def test_metadata_written_into_local_dir(tmp_path, monkeypatch):
    def fake_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        _make_model_dir(target)
        return str(target)

    monkeypatch.setattr(_hub, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo", revision="v2", local_dir=str(tmp_path))
    meta = json.loads((Path(out) / _hub.METADATA_FILENAME).read_text(encoding="utf-8"))
    assert meta["repo_id"] == "org/repo"
    assert meta["revision"] == "v2"
    assert "downloaded_at" in meta


def test_missing_local_dir_without_repo_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_model_dir(local_dir=str(tmp_path / "missing"))


def test_no_local_dir_no_repo_id_raises():
    with pytest.raises(ValueError):
        resolve_model_dir()


def test_local_models_dir_used_as_default_base(tmp_path, monkeypatch):
    model_dir = tmp_path / "repo"
    _make_model_dir(model_dir)

    cfg = _hub.get_ai_model_settings()
    monkeypatch.setattr(cfg, "local_models_dir", tmp_path)

    def _boom():
        raise AssertionError("a present local model must not trigger a download")

    monkeypatch.setattr(_hub, "_import_snapshot_download", _boom)

    out = resolve_model_dir(repo_id="org/repo")
    assert Path(out) == model_dir


def test_empty_huggingface_path_is_treated_as_unset(monkeypatch):
    from pyRadPlan.ai_models import AiModelSettings

    monkeypatch.setenv("PYRADPLAN_HUGGINGFACE_PATH", "")
    assert AiModelSettings().local_models_dir is None


def test_local_models_dir_defaults_under_data_root(tmp_path, monkeypatch):
    from pyRadPlan.ai_models import AiModelSettings

    monkeypatch.delenv("PYRADPLAN_HUGGINGFACE_PATH", raising=False)
    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    assert AiModelSettings().local_models_dir == tmp_path / "ai_models"


def test_settings_are_the_global_ai_models_subconfig():
    import pyRadPlan
    from pyRadPlan.ai_models import AiModelSettings, get_ai_model_settings

    cfg = get_ai_model_settings()
    assert isinstance(cfg, AiModelSettings)
    assert cfg is pyRadPlan.settings.ai_models


def test_runtime_mutation_is_observed():
    import pyRadPlan
    from pyRadPlan.ai_models import get_ai_model_settings

    original = pyRadPlan.settings.ai_models.device
    try:
        pyRadPlan.settings.ai_models.device = "cuda"
        assert get_ai_model_settings().device == "cuda"
    finally:
        pyRadPlan.settings.ai_models.device = original

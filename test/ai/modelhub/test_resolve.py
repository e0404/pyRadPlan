import json
from pathlib import Path

import pytest

from pyRadPlan.ai.modelhub import _resolve, is_valid_model_dir, resolve_model_dir


def _make_model_dir(path: Path, metadata: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in _resolve.REQUIRED_FILES:
        (path / name).write_text("placeholder", encoding="utf-8")
    if metadata is not None:
        (path / _resolve.METADATA_FILENAME).write_text(json.dumps(metadata), encoding="utf-8")


def _no_download(monkeypatch, why: str) -> None:
    def _boom():
        raise AssertionError(why)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", _boom)


def test_is_valid_model_dir(tmp_path):
    assert not is_valid_model_dir(tmp_path)
    _make_model_dir(tmp_path)
    assert is_valid_model_dir(tmp_path)


def test_local_dir_used_without_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path)
    _no_download(monkeypatch, "snapshot_download must not be imported for a valid local dir")

    out = resolve_model_dir(local_dir=str(tmp_path))
    assert Path(out) == tmp_path


def test_version_dedup_skips_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path, metadata={"repo_id": "org/repo", "revision": "v1"})
    _no_download(monkeypatch, "matching version must not trigger a download")

    out = resolve_model_dir(repo_id="org/repo", revision="v1", local_dir=str(tmp_path))
    assert Path(out) == tmp_path


def test_different_revision_triggers_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path, metadata={"repo_id": "org/repo", "revision": "v1"})

    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["revision"])
        return kwargs["local_dir"]

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    resolve_model_dir(repo_id="org/repo", revision="v2", local_dir=str(tmp_path))
    assert calls == ["v2"]


def test_unpinned_request_defers_to_hub(tmp_path, monkeypatch):
    """Without a pinned revision a present local copy is still checked for updates."""
    _make_model_dir(tmp_path)

    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["repo_id"])
        return kwargs["local_dir"]

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path))
    assert Path(out) == tmp_path
    assert calls == ["org/repo"]


def test_offline_uses_local_copy_without_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path)
    _no_download(monkeypatch, "offline mode must not reach for snapshot_download")

    out = resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path), offline=True)
    assert Path(out) == tmp_path


def test_unreachable_hub_falls_back_to_local_copy(tmp_path, monkeypatch, caplog):
    """A hand-placed copy carries no hub metadata, so it must survive an outage."""
    _make_model_dir(tmp_path)

    def fake_snapshot(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    with caplog.at_level("WARNING"):
        out = resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path))
    assert Path(out) == tmp_path
    assert "using the local copy" in caplog.text


def test_unreachable_hub_without_local_copy_raises(tmp_path, monkeypatch):
    def fake_snapshot(**kwargs):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    with pytest.raises(FileNotFoundError, match="network unavailable"):
        resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path / "empty"))


def test_offline_fallback_retries_local_only(tmp_path, monkeypatch):
    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            raise RuntimeError("network unavailable")
        return str(tmp_path)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo")
    assert Path(out) == tmp_path
    assert calls == [False, True]


def test_metadata_written_into_local_dir(tmp_path, monkeypatch):
    def fake_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        _make_model_dir(target)
        return str(target)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo", revision="v2", local_dir=str(tmp_path))
    meta = json.loads((Path(out) / _resolve.METADATA_FILENAME).read_text(encoding="utf-8"))
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
    model_dir = tmp_path / "org" / "repo"
    _make_model_dir(model_dir, metadata={"repo_id": "org/repo", "revision": "v1"})

    cfg = _resolve.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)
    _no_download(monkeypatch, "a present, pinned local model must not trigger a download")

    out = resolve_model_dir(repo_id="org/repo", revision="v1")
    assert Path(out) == model_dir


def test_a_fork_gets_its_own_directory(tmp_path, monkeypatch):
    """Same repo name under two orgs must not share (and clobber) one directory."""
    cfg = _resolve.get_settings().ai
    monkeypatch.setattr(cfg, "modelhub_local_models_dir", tmp_path)

    targets = []

    def fake_snapshot(**kwargs):
        target = Path(kwargs["local_dir"])
        targets.append(target)
        _make_model_dir(target)
        return str(target)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    resolve_model_dir(repo_id="DKFZ-RadOpt/dosecalc-x", revision="v1")
    resolve_model_dir(repo_id="myfork/dosecalc-x", revision="v1")

    assert targets == [tmp_path / "DKFZ-RadOpt" / "dosecalc-x", tmp_path / "myfork" / "dosecalc-x"]
    # and each keeps its own metadata, so neither re-downloads over the other
    _no_download(monkeypatch, "both pinned copies are present; neither may re-download")
    assert Path(resolve_model_dir(repo_id="DKFZ-RadOpt/dosecalc-x", revision="v1")) == targets[0]
    assert Path(resolve_model_dir(repo_id="myfork/dosecalc-x", revision="v1")) == targets[1]


def test_repo_subpath_rejects_traversal():
    from pyRadPlan.ai.modelhub import repo_subpath

    assert repo_subpath("org/repo") == Path("org") / "repo"
    for bad in ("../evil", "org/../..", "org//repo", ""):
        with pytest.raises(ValueError):
            repo_subpath(bad)


def test_empty_local_models_dir_is_treated_as_unset(monkeypatch):
    from pyRadPlan.ai.modelhub import AiSettings

    monkeypatch.setenv("PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR", "")
    assert AiSettings(_env_file=None).modelhub_local_models_dir is None


def test_local_models_dir_defaults_under_data_root(tmp_path, monkeypatch):
    from pyRadPlan.ai.modelhub import AiSettings

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path))
    assert AiSettings(_env_file=None).modelhub_local_models_dir == tmp_path / "ai_models"


def test_trust_remote_code_defaults_to_false():
    from pyRadPlan.ai.modelhub import AiSettings

    assert AiSettings(_env_file=None).modelhub_trust_remote_code is False


def test_settings_are_the_global_ai_subconfig():
    import pyRadPlan
    from pyRadPlan.ai.modelhub import AiSettings

    cfg = pyRadPlan.settings.ai
    assert isinstance(cfg, AiSettings)


def test_runtime_mutation_is_observed():
    import pyRadPlan

    original = pyRadPlan.settings.ai.modelhub_device
    try:
        # a value set on the singleton is seen by the hub, which reads it at call time
        pyRadPlan.settings.ai.modelhub_device = "cuda"
        assert _resolve.get_settings().ai.modelhub_device == "cuda"
    finally:
        pyRadPlan.settings.ai.modelhub_device = original

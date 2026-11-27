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
    """Without a pinned revision a downloaded local copy is still checked for updates."""
    _make_model_dir(tmp_path, metadata={"repo_id": "org/repo", "revision": None})

    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["repo_id"])
        return kwargs["local_dir"]

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    out = resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path))
    assert Path(out) == tmp_path
    assert calls == ["org/repo"]


def test_user_placed_folder_is_never_refreshed(tmp_path, monkeypatch):
    """A model the user put there is used as-is, even online and unpinned."""
    _make_model_dir(tmp_path / "explicit")
    _make_model_dir(tmp_path / "base" / "org" / "repo")
    monkeypatch.setattr(_resolve.get_settings().ai, "modelhub_local_models_dir", tmp_path / "base")
    _no_download(monkeypatch, "a user-placed model must not be refreshed from the hub")

    resolved = _resolve._resolve_model(repo_id="org/repo", local_dir=tmp_path / "explicit")
    assert resolved.path == tmp_path / "explicit"
    assert resolved.downloaded is False

    resolved = _resolve._resolve_model(repo_id="org/repo", revision="v9")
    assert resolved.path == tmp_path / "base" / "org" / "repo"
    assert resolved.downloaded is False
    assert not (resolved.path / _resolve.METADATA_FILENAME).exists()


def test_partial_folder_without_marker_is_refused(tmp_path, monkeypatch):
    """Files that are neither a complete model nor a managed download are left alone."""
    (tmp_path / "model.py").write_text("print('mine')", encoding="utf-8")
    _no_download(monkeypatch, "pyRadPlan must not download over the user's files")

    with pytest.raises(FileNotFoundError, match="neither a complete model"):
        resolve_model_dir(repo_id="org/repo", local_dir=tmp_path)
    assert (tmp_path / "model.py").read_text(encoding="utf-8") == "print('mine')"
    assert not (tmp_path / _resolve.METADATA_FILENAME).exists()


def test_offline_uses_local_copy_without_download(tmp_path, monkeypatch):
    _make_model_dir(tmp_path)
    _no_download(monkeypatch, "offline mode must not reach for snapshot_download")

    out = resolve_model_dir(repo_id="org/repo", local_dir=str(tmp_path), offline=True)
    assert Path(out) == tmp_path


def test_unreachable_hub_falls_back_to_local_copy(tmp_path, monkeypatch, caplog):
    """A downloaded copy must survive an outage."""
    _make_model_dir(tmp_path, metadata={"repo_id": "org/repo", "revision": None})

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
    # a download that fetched nothing leaves nothing behind
    assert not (tmp_path / "empty").exists()


def test_offline_fallback_retries_local_only(tmp_path, monkeypatch):
    cached = tmp_path / "models--org--repo" / "snapshots" / "abc"
    _make_model_dir(cached)
    monkeypatch.setattr(_resolve.get_settings().ai, "modelhub_local_models_dir", None)
    calls = []

    def fake_snapshot(**kwargs):
        calls.append(kwargs["local_files_only"])
        if not kwargs["local_files_only"]:
            raise RuntimeError("network unavailable")
        return str(cached)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    resolved = _resolve._resolve_model(repo_id="org/repo")
    assert resolved.path == cached
    assert resolved.downloaded is True  # whatever comes from the cache was downloaded
    assert calls == [False, True]


def test_marker_is_written_before_the_download(tmp_path, monkeypatch):
    target = tmp_path / "org" / "repo"
    seen = []

    def fake_snapshot(**kwargs):
        seen.append(_resolve._read_metadata(Path(kwargs["local_dir"])))
        _make_model_dir(Path(kwargs["local_dir"]))
        return kwargs["local_dir"]

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    resolve_model_dir(repo_id="org/repo", revision="v1", local_dir=target)
    assert seen[0]["state"] == "downloading"
    assert _resolve._read_metadata(target)["state"] == "complete"


@pytest.mark.parametrize("complete", [True, False])
def test_interrupted_download_never_passes_as_user_placed(tmp_path, monkeypatch, complete):
    """Files fetched before an interruption carry the marker, so they count as downloaded."""
    target = tmp_path / "org" / "repo"

    def fake_snapshot(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        if kwargs["local_files_only"]:
            return str(local_dir)  # like huggingface_hub: a non-empty local_dir is returned
        names = _resolve.REQUIRED_FILES if complete else _resolve.REQUIRED_FILES[:1]
        for name in names:
            (local_dir / name).write_text("fetched", encoding="utf-8")
        raise RuntimeError("connection reset")

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    if complete:
        resolved = _resolve._resolve_model(repo_id="org/repo", revision="v1", local_dir=target)
        assert resolved.downloaded is True
    else:
        with pytest.raises(FileNotFoundError, match=r"not \(fully\) downloaded"):
            resolve_model_dir(repo_id="org/repo", revision="v1", local_dir=target)
    assert _resolve._classify(target) == "marked"
    # the pinned revision never completed, so it does not count as present
    assert not _resolve._metadata_matches(target, "org/repo", "v1")
    if complete:
        _no_download(monkeypatch, "a local-only load never downloads")
        assert _resolve._resolve_model(local_dir=target).downloaded is True


def test_a_shipped_marker_cannot_make_a_folder_user_placed(tmp_path, monkeypatch):
    """A repository's own .pyradplan_model.json is just another marker."""
    target = tmp_path / "org" / "repo"

    def fake_snapshot(**kwargs):
        local_dir = Path(kwargs["local_dir"])
        _make_model_dir(local_dir, metadata={"state": "user", "downloaded": False})
        return str(local_dir)

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake_snapshot)

    resolved = _resolve._resolve_model(repo_id="org/repo", local_dir=target)
    assert resolved.downloaded is True
    assert _resolve._read_metadata(target)["repo_id"] == "org/repo"
    assert _resolve._resolve_model(local_dir=target, offline=True).downloaded is True


def test_hf_cache_snapshot_counts_as_downloaded(tmp_path, monkeypatch):
    """A cache snapshot passed as local_dir is downloaded content, never refreshed in place."""
    snapshot = tmp_path / "hub" / "models--some-org--dosecalc--x" / "snapshots" / "abc123"
    _make_model_dir(snapshot)
    _no_download(monkeypatch, "a cache snapshot must not be downloaded into")

    assert _resolve._classify(snapshot) == "cache"
    resolved = _resolve._resolve_model(local_dir=snapshot)
    assert resolved.downloaded is True
    assert resolved.repo_id == "some-org/dosecalc--x"
    assert _resolve._resolve_model(repo_id="a/b", local_dir=snapshot).downloaded is True


def test_configured_cache_dir_counts_as_downloaded(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    _make_model_dir(cache / "elsewhere")
    monkeypatch.setattr(_resolve.get_settings().ai, "modelhub_cache_dir", cache)
    assert _resolve._classify(cache / "elsewhere") == "cache"


def test_symlink_into_the_hf_cache_counts_as_downloaded(tmp_path):
    snapshot = tmp_path / "models--org--repo" / "snapshots" / "abc"
    _make_model_dir(snapshot)
    link = tmp_path / "base" / "org" / "repo"
    link.parent.mkdir(parents=True)
    try:
        link.symlink_to(snapshot, target_is_directory=True)
    except OSError:
        pytest.skip("creating symlinks is not permitted here")
    assert _resolve._classify(link) == "cache"


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
    assert meta["state"] == "complete"
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

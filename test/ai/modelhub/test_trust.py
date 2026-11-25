"""Which model code may run: user-placed folders, downloads and their trust records.

The Hub is replaced by a fake ``snapshot_download`` that writes a repository's files
into the requested ``local_dir`` (or a fake HuggingFace cache), and every model's
``model.py`` leaves a file behind when it is imported, so a test can tell whether any
of a repository's code ran.
"""

import json
from pathlib import Path

import pytest

from pyRadPlan.ai.modelhub import (
    UntrustedModelError,
    _load_model,
    _resolve,
    _trust,
    is_trusted,
    load_model,
    load_model_with_dir,
    trust_model,
    untrust_model,
)

MODEL_PY = """
from pathlib import Path

Path({sentinel!r}).write_text("imported", encoding="utf-8")


class FakeNet:
    version = {version!r}

    def __init__(self):
        self.training = True

    def eval(self):
        self.training = False
        return self
"""

PREPROCESSOR_PY = """
from pyRadPlan.ai.modelhub import BasePreprocessor


class FakePreprocessor(BasePreprocessor):
    def preprocess(self, inputs):
        return inputs
"""


@pytest.fixture(autouse=True)
def no_weight_loading(monkeypatch):
    monkeypatch.setattr(_load_model, "_load_weights", lambda model, path, device: None)


class FakeHub:
    """Stand-in for ``snapshot_download``; repositories are dicts of file name to text."""

    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.cache = tmp_path / "hf_cache"
        self.repos = {}
        self.calls = []

    def publish(self, repo_id, version="1", **extra_files):
        sentinel = self.tmp_path / f"imported-{repo_id.replace('/', '--')}-{version}"
        self.repos[repo_id] = {
            "model.py": MODEL_PY.format(sentinel=str(sentinel), version=version),
            "preprocessor.py": PREPROCESSOR_PY,
            "weights.safetensors": "",
            "model_config.json": '{"model_name": "FakeNet"}',
            **extra_files,
        }
        return sentinel

    def __call__(self, *, repo_id, revision, cache_dir, local_dir, local_files_only):
        self.calls.append((repo_id, revision, local_files_only))
        if local_dir is not None:
            dest = Path(local_dir)
        else:
            folder = "models--" + repo_id.replace("/", "--")
            dest = self.cache / folder / "snapshots" / (revision or "main")
        if local_files_only:
            if dest.is_dir() and any(dest.iterdir()):
                return str(dest)
            raise FileNotFoundError(f"'{repo_id}' is not cached")
        dest.mkdir(parents=True, exist_ok=True)
        for name, text in self.repos[repo_id].items():
            (dest / name).write_text(text, encoding="utf-8")
        return str(dest)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    fake = FakeHub(tmp_path)
    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: fake)
    return fake


@pytest.fixture
def models_base(isolated_model_settings):
    return Path(isolated_model_settings.modelhub_local_models_dir)


def _make_user_model(path):
    path.mkdir(parents=True)
    (path / "model.py").write_text(
        "class FakeNet:\n    def eval(self):\n        return self\n", encoding="utf-8"
    )
    (path / "preprocessor.py").write_text(PREPROCESSOR_PY, encoding="utf-8")
    (path / "weights.safetensors").write_text("", encoding="utf-8")
    (path / "model_config.json").write_text('{"model_name": "FakeNet"}', encoding="utf-8")
    return path


def _trust_file():
    return _trust._trust_file()


# --- user-placed folders ------------------------------------------------------------


def test_user_placed_folder_loads_without_opt_in_and_is_never_downloaded(
    tmp_path, hub, models_base
):
    explicit = _make_user_model(tmp_path / "mine")
    by_id = _make_user_model(models_base / "org" / "placed")

    assert load_model(local_dir=explicit)[0] is not None
    assert load_model(repo_id="org/whatever", local_dir=explicit)[0] is not None
    _, _, model_dir = load_model_with_dir("org/placed")

    assert model_dir == by_id
    assert hub.calls == []  # online and unpinned, yet nothing was fetched
    assert not _trust_file().exists()
    assert is_trusted("org/placed")


def test_trust_model_records_nothing_for_a_user_placed_folder(hub, models_base):
    _make_user_model(models_base / "org" / "placed")
    trusted = trust_model("org/placed")
    assert trusted.model_dir == models_base / "org" / "placed"
    assert not _trust_file().exists()


# --- downloaded models ---------------------------------------------------------------


def test_downloaded_model_is_refused_before_any_code_runs(hub, models_base):
    sentinel = hub.publish("org/remote")

    with pytest.raises(UntrustedModelError) as excinfo:
        load_model("org/remote")

    message = str(excinfo.value)
    assert 'pyRadPlan.ai.modelhub.trust_model("org/remote")' in message
    assert "pyRadPlan.settings.ai.modelhub_trust_remote_code = True" in message
    assert "PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1" in message
    assert isinstance(excinfo.value, ValueError)
    assert (models_base / "org" / "remote" / "model.py").is_file()  # downloaded ...
    assert not sentinel.exists()  # ... but never imported


def test_bare_name_is_refused_with_the_full_id(hub, isolated_model_settings):
    sentinel = hub.publish(f"{isolated_model_settings.modelhub_hf_org}/dosecalc-x")
    with pytest.raises(UntrustedModelError, match="dosecalc-x"):
        load_model("dosecalc-x")
    assert not sentinel.exists()


def test_trust_lifecycle(hub, isolated_model_settings, monkeypatch):
    sentinel_v1 = hub.publish("org/remote", version="1")

    # the global opt-in trusts whatever is downloaded, and records it
    monkeypatch.setattr(isolated_model_settings, "modelhub_trust_remote_code", True)
    model, _ = load_model("org/remote")
    assert model.version == "1" and sentinel_v1.exists()
    records = json.loads(_trust_file().read_text(encoding="utf-8"))
    assert list(records) == ["org/remote"]

    # without it, the recorded code keeps running
    monkeypatch.setattr(isolated_model_settings, "modelhub_trust_remote_code", False)
    assert load_model("org/remote")[0].version == "1"

    # an unpinned load refreshes the copy; changed code is refused before it runs
    sentinel_v2 = hub.publish("org/remote", version="2")
    with pytest.raises(UntrustedModelError, match="changed since you trusted it") as excinfo:
        load_model("org/remote")
    assert 'trust_model("org/remote")' in str(excinfo.value)
    assert not sentinel_v2.exists()

    # trusting it again lets it run, without executing anything itself
    trusted = trust_model("org/remote")
    assert not sentinel_v2.exists()
    assert trusted.code_hash == _trust.code_hash(trusted.model_dir)
    _load_model._loaded_repos.pop(trusted.model_dir.resolve(), None)
    assert load_model("org/remote")[0] is not None
    assert sentinel_v2.exists()


def test_trust_remote_code_argument(hub):
    hub.publish("org/remote")
    assert load_model("org/remote", trust_remote_code=True)[0] is not None
    assert is_trusted("org/remote")


def test_trust_remote_code_false_overrides_the_setting(hub, isolated_model_settings, monkeypatch):
    sentinel = hub.publish("org/remote")
    monkeypatch.setattr(isolated_model_settings, "modelhub_trust_remote_code", True)
    with pytest.raises(UntrustedModelError):
        load_model("org/remote", trust_remote_code=False)
    assert not sentinel.exists()


def test_trust_model_downloads_without_running(hub, models_base):
    sentinel = hub.publish("org/remote")
    trusted = trust_model("org/remote")
    assert trusted.model_dir == models_base / "org" / "remote"
    assert not sentinel.exists()
    assert is_trusted("org/remote")

    assert load_model("org/remote")[0] is not None
    assert sentinel.exists()


def test_untrust_model(hub):
    hub.publish("org/remote")
    trust_model("org/remote")
    assert untrust_model("org/remote") is True
    assert untrust_model("org/remote") is False
    assert not is_trusted("org/remote")
    with pytest.raises(UntrustedModelError, match="was downloaded"):
        load_model("org/remote")


def test_hf_cache_model_counts_as_downloaded(hub, isolated_model_settings, monkeypatch):
    monkeypatch.setattr(isolated_model_settings, "modelhub_local_models_dir", None)
    sentinel = hub.publish("org/cached")

    with pytest.raises(UntrustedModelError):
        load_model("org/cached")
    assert not sentinel.exists()
    snapshot = hub.cache / "models--org--cached" / "snapshots" / "main"
    assert (snapshot / "model.py").is_file()

    # the snapshot passed as a directory is still downloaded code
    with pytest.raises(UntrustedModelError):
        load_model(local_dir=snapshot)

    trust_model("org/cached")
    assert load_model(local_dir=snapshot)[0] is not None


def test_marked_folder_needs_trust_as_local_dir(hub, models_base):
    """The old bypass: download by id, then load the folder as a plain directory."""
    sentinel = hub.publish("org/remote")
    folder = trust_model("org/remote").model_dir
    untrust_model("org/remote")

    with pytest.raises(UntrustedModelError, match="org/remote"):
        load_model(local_dir=folder)
    assert not sentinel.exists()

    trust_model("org/remote", local_dir=folder)
    assert load_model(local_dir=folder)[0] is not None


def test_interrupted_download_is_not_user_placed(hub, models_base, monkeypatch):
    """Files that arrived before an interruption are downloaded code, not the user's."""
    sentinel = hub.publish("org/remote")

    def interrupted(**kwargs):
        if kwargs["local_files_only"]:
            return hub(**kwargs)
        hub(**kwargs)
        raise ConnectionError("connection reset")

    monkeypatch.setattr(_resolve, "_import_snapshot_download", lambda: interrupted)

    with pytest.raises(UntrustedModelError):
        load_model("org/remote")
    with pytest.raises(UntrustedModelError):
        load_model(local_dir=models_base / "org" / "remote")
    assert not sentinel.exists()


@pytest.mark.parametrize("in_cache", [False, True])
@pytest.mark.parametrize(
    "shipped_marker",
    [
        {"repo_id": "org/trusted", "revision": None, "state": "complete"},
        {"user_placed": True, "trusted": True},
    ],
)
def test_a_repository_cannot_ship_its_way_into_trust(
    hub, models_base, isolated_model_settings, monkeypatch, shipped_marker, in_cache
):
    if in_cache:
        monkeypatch.setattr(isolated_model_settings, "modelhub_local_models_dir", None)
    hub.publish("org/trusted")
    trust_model("org/trusted")
    sentinel = hub.publish(
        "org/evil",
        **{
            _resolve.METADATA_FILENAME: json.dumps(shipped_marker),
            _trust.TRUST_FILENAME: json.dumps({"org/evil": {"hashes": {"x": {}}}}),
        },
    )

    with pytest.raises(UntrustedModelError):
        load_model("org/evil")
    folder = (
        hub.cache / "models--org--evil" / "snapshots" / "main"
        if in_cache
        else models_base / "org" / "evil"
    )
    with pytest.raises(UntrustedModelError):
        load_model(local_dir=folder)
    assert not sentinel.exists()


def test_sibling_modules_are_part_of_the_code_hash(hub, isolated_model_settings, monkeypatch):
    hub.publish("org/remote", **{"helpers.py": "X = 1"})
    folder = trust_model("org/remote").model_dir

    (folder / "helpers.py").write_text("X = 2", encoding="utf-8")
    with pytest.raises(UntrustedModelError, match="changed"):
        load_model("org/remote", offline=True)

    cache = folder / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-313.pyc").write_bytes(b"\0")
    trusted_hash = _trust.code_hash(folder)
    (cache / "model.cpython-313.pyc").write_bytes(b"\1")
    assert _trust.code_hash(folder) != trusted_hash


def test_weights_and_bookkeeping_are_not_hashed(tmp_path):
    folder = _make_user_model(tmp_path / "model")
    digest = _trust.code_hash(folder)

    (folder / "weights.safetensors").write_text("other weights", encoding="utf-8")
    (folder / _resolve.METADATA_FILENAME).write_text("{}", encoding="utf-8")
    (folder / ".cache" / "huggingface").mkdir(parents=True)
    (folder / ".cache" / "huggingface" / "x.metadata").write_text("1", encoding="utf-8")
    assert _trust.code_hash(folder) == digest

    (folder / "model_config.json").write_text('{"model_name": "Other"}', encoding="utf-8")
    assert _trust.code_hash(folder) != digest


# --- offline, revisions --------------------------------------------------------------


def test_offline_loads_trusted_download_and_user_placed(hub, models_base):
    hub.publish("org/remote")
    trust_model("org/remote")
    _make_user_model(models_base / "org" / "placed")
    hub.calls.clear()

    assert load_model("org/remote", offline=True)[0] is not None
    assert load_model("org/placed", offline=True)[0] is not None
    assert all(local_only for _, _, local_only in hub.calls)


def test_pinned_revision_is_not_downloaded_twice(hub):
    hub.publish("org/remote")
    trust_model("org/remote", revision="v1")
    hub.calls.clear()

    assert load_model("org/remote", revision="v1")[0] is not None
    assert hub.calls == []


# --- trust file ----------------------------------------------------------------------


def test_trust_file_lives_in_the_data_dir(hub, tmp_path):
    hub.publish("org/remote")
    trust_model("org/remote")
    assert _trust_file() == tmp_path / "data_root" / _trust.TRUST_FILENAME
    assert _trust_file().is_file()


@pytest.mark.parametrize("content", ["{not json", "[]", '{"org/remote": 3}', "\udcff"])
def test_corrupt_trust_file_is_tolerated(hub, content):
    hub.publish("org/remote")
    _trust_file().parent.mkdir(parents=True, exist_ok=True)
    _trust_file().write_text(content, encoding="utf-8", errors="surrogateescape")

    with pytest.raises(UntrustedModelError):
        load_model("org/remote")
    trust_model("org/remote")  # rewrites the file
    assert "org/remote" in json.loads(_trust_file().read_text(encoding="utf-8"))
    assert load_model("org/remote")[0] is not None


def test_trust_records_are_written_atomically(hub, monkeypatch):
    hub.publish("org/one")
    hub.publish("org/two")
    trust_model("org/one")
    before = _trust_file().read_text(encoding="utf-8")

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(_trust.os, "replace", failing_replace)
    with pytest.raises(OSError, match="disk full"):
        trust_model("org/two")
    assert _trust_file().read_text(encoding="utf-8") == before
    assert not list(_trust_file().parent.glob(f".{_trust.TRUST_FILENAME}.*"))

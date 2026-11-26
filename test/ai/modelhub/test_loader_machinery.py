"""Loader tests that do not need torch.

The end-to-end load of a real model lives in ``test_load_model.py`` and is
skipped without torch/safetensors. Everything the loader does *around* the
weights -- importing a repository's code, reading its config -- is exercised
here against a synthetic model folder, so it stays covered in an environment
without the ``ai`` extra installed. Which code may run at all is covered in
``test_trust.py``.
"""

import sys

import pytest

from pyRadPlan.ai.modelhub import (
    BasePreprocessor,
    _load_model,
    _resolve,
    load_model,
    load_model_with_dir,
)

MODEL_PY = """
class FakeNet:
    def __init__(self, width=2):
        self.width = width
        self.device = None
        self.training = True

    def to(self, device):
        self.device = device
        return self

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

CONFIG_JSON = '{"model_name": "FakeNet", "model_params": {"width": 4}}'


def _make_fake_model(path, config=CONFIG_JSON, model_py=MODEL_PY):
    path.mkdir(parents=True, exist_ok=True)
    (path / "model.py").write_text(model_py, encoding="utf-8")
    (path / "preprocessor.py").write_text(PREPROCESSOR_PY, encoding="utf-8")
    (path / "weights.safetensors").write_bytes(b"")
    (path / "model_config.json").write_text(config, encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def no_weight_loading(monkeypatch):
    """Skip the safetensors/torch step; these tests are about everything else."""
    monkeypatch.setattr(_load_model, "_load_weights", lambda model, path, device: None)


@pytest.fixture
def fake_model_dir(tmp_path):
    return _make_fake_model(tmp_path / "dosecalc-fake")


def test_loads_model_and_preprocessor(fake_model_dir):
    model, preprocessor = load_model(local_dir=str(fake_model_dir))

    assert type(model).__name__ == "FakeNet"
    assert model.width == 4  # model_params reached the constructor
    assert model.device == "cpu"  # settings default
    assert model.training is False  # eval() was called
    assert isinstance(preprocessor, BasePreprocessor)


def test_device_argument_is_applied(fake_model_dir):
    model, _ = load_model(local_dir=str(fake_model_dir), device="cuda")
    assert model.device == "cuda"


def test_classes_keep_a_resolvable_module(fake_model_dir):
    model, preprocessor = load_model(local_dir=str(fake_model_dir))

    for obj in (model, preprocessor):
        module_name = type(obj).__module__
        assert module_name.startswith("pyRadPlan._ai_model_repos.")
        # resolvable, which is what pickle/torch.save/spawn workers rely on
        assert sys.modules[module_name] is not None

    # the generic names must not be left behind
    assert "model" not in sys.modules
    assert "preprocessor" not in sys.modules


def test_two_model_folders_do_not_shadow_each_other(tmp_path):
    first = _make_fake_model(tmp_path / "dosecalc-a")
    second = _make_fake_model(
        tmp_path / "dosecalc-b", model_py=MODEL_PY.replace("width=2", "width=99")
    )

    model_a, _ = load_model(local_dir=str(first))
    model_b, _ = load_model(local_dir=str(second))

    assert type(model_a) is not type(model_b)
    assert type(model_a).__module__ != type(model_b).__module__


def test_repeated_load_reuses_the_module(fake_model_dir):
    first, _ = load_model(local_dir=str(fake_model_dir))
    second, _ = load_model(local_dir=str(fake_model_dir))
    assert type(first) is type(second)


def test_import_failure_leaves_no_partial_module(tmp_path):
    broken = _make_fake_model(tmp_path / "dosecalc-broken", model_py="raise RuntimeError('boom')")

    with pytest.raises(RuntimeError, match="boom"):
        load_model(local_dir=str(broken))

    assert broken.resolve() not in _load_model._loaded_repos
    assert not [n for n in sys.modules if n.endswith(".model") and "dosecalc_broken" in n]


def test_explicit_local_dir_needs_no_trust_opt_in(fake_model_dir):
    model, _ = load_model(local_dir=str(fake_model_dir))
    assert model is not None


def test_trust_remote_code_false_still_runs_a_folder_the_user_placed(fake_model_dir):
    """The argument is about downloaded code; a user's own folder is always theirs to run."""
    model, _ = load_model(local_dir=str(fake_model_dir), trust_remote_code=False)
    assert model is not None


def test_user_placed_folder_by_id_needs_no_trust_opt_in(
    tmp_path, monkeypatch, isolated_model_settings
):
    _make_fake_model(tmp_path / "base" / "org" / "dosecalc-x")
    monkeypatch.setattr(isolated_model_settings, "modelhub_local_models_dir", tmp_path / "base")

    model, _, model_dir = load_model_with_dir(repo_id="org/dosecalc-x")
    assert model is not None
    assert model_dir == tmp_path / "base" / "org" / "dosecalc-x"


def _capture_resolve(monkeypatch, model_dir):
    captured = {}

    def fake_resolve(**kwargs):
        captured.update(kwargs)
        return _resolve.ResolvedModel(model_dir, downloaded=False, repo_id=kwargs["repo_id"])

    monkeypatch.setattr(_load_model, "_resolve_model", fake_resolve)
    return captured


def test_load_model_with_dir_reports_the_resolved_directory(fake_model_dir, monkeypatch):
    """Callers reading the repository's metadata get the directory the load resolved."""
    _capture_resolve(monkeypatch, fake_model_dir)

    model, preprocessor, model_dir = load_model_with_dir(repo_id="org/dosecalc-x")
    assert model is not None and preprocessor is not None
    assert model_dir == fake_model_dir


def test_name_resolves_to_hf_org_repo_id(fake_model_dir, monkeypatch):
    captured = _capture_resolve(monkeypatch, fake_model_dir)

    load_model("dosecalc-x")
    hf_org = _load_model.get_settings().ai.modelhub_hf_org
    assert captured["repo_id"] == f"{hf_org}/dosecalc-x"


def test_full_id_as_name_is_used_verbatim(fake_model_dir, monkeypatch):
    """list_local_models() returns org/repo, so it must feed straight back in."""
    captured = _capture_resolve(monkeypatch, fake_model_dir)

    load_model("myfork/dosecalc-x")
    assert captured["repo_id"] == "myfork/dosecalc-x"


def test_missing_model_name_key_reports_the_config(tmp_path):
    model_dir = _make_fake_model(tmp_path / "dosecalc-nokey", config="{}")
    with pytest.raises(KeyError, match="model_name"):
        load_model(local_dir=str(model_dir))


def test_unknown_model_class_reports_the_config(tmp_path):
    model_dir = _make_fake_model(tmp_path / "dosecalc-noclass", config='{"model_name": "NoSuch"}')
    with pytest.raises(AttributeError, match="NoSuch"):
        load_model(local_dir=str(model_dir))


def test_invalid_json_config_reports_the_path(tmp_path):
    model_dir = _make_fake_model(tmp_path / "dosecalc-badjson", config="{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_model(local_dir=str(model_dir))


def test_named_preprocessor_must_exist(tmp_path):
    model_dir = _make_fake_model(
        tmp_path / "dosecalc-nopre",
        config='{"model_name": "FakeNet", "preprocessor_name": "Missing"}',
    )
    with pytest.raises(ImportError, match="preprocessor_name"):
        load_model(local_dir=str(model_dir))


def test_named_preprocessor_must_be_a_base_preprocessor(tmp_path):
    model_dir = _make_fake_model(
        tmp_path / "dosecalc-badpre",
        config='{"model_name": "FakeNet", "preprocessor_name": "NotAPreprocessor"}',
    )
    (model_dir / "preprocessor.py").write_text(
        PREPROCESSOR_PY + "\n\nNotAPreprocessor = object\n", encoding="utf-8"
    )

    with pytest.raises(TypeError, match="BasePreprocessor subclass"):
        load_model(local_dir=str(model_dir))


def test_ambiguous_preprocessor_requires_a_name(tmp_path):
    model_dir = _make_fake_model(tmp_path / "dosecalc-twopre")
    (model_dir / "preprocessor.py").write_text(
        PREPROCESSOR_PY
        + "\n\nclass OtherPreprocessor(BasePreprocessor):\n"
        + "    def preprocess(self, inputs):\n        return inputs\n",
        encoding="utf-8",
    )

    with pytest.raises(ImportError, match="multiple BasePreprocessor"):
        load_model(local_dir=str(model_dir))

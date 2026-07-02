import pytest

from pyRadPlan.ml import (
    Modality,
    ModelSpec,
    get_model_spec,
    list_models,
    modality_from_name,
    resolve_repo_id,
)
from pyRadPlan.ml import _factory


def test_modality_from_name():
    assert modality_from_name("dosecalc-bayes-protons") == Modality.DOSE_CALC
    assert modality_from_name("outcome-ORPDenseNet-tg119") == Modality.OUTCOME
    assert modality_from_name("something-else") is None


def test_latest_version_ordering():
    # numeric-aware: 1.0.10 must beat 1.0.2, and 'v' prefixes are handled
    assert max(["v1.0.0", "v1.0.10", "v1.0.2"], key=_factory._version_key) == "v1.0.10"


def test_has_model_contract():
    contract = set(_factory.REQUIRED_FILES)
    # full contract at the repo root (plus junk)
    assert _factory._has_model_contract(contract | {".gitattributes", "README.md"})
    # incomplete contract -> not a model
    assert not _factory._has_model_contract({"model_config.json", "model.py"})


def _fake_discovery(monkeypatch, local=None, remote=None):
    _factory.clear_model_cache()  # remote results are cached by org across calls
    monkeypatch.setattr(_factory, "_discover_local", lambda cfg: local or {})

    def _remote(org):
        if remote is None:
            raise AssertionError("remote discovery must not run")
        return remote

    monkeypatch.setattr(_factory, "_discover_remote", _remote)


def test_list_models_merges_local_and_remote(monkeypatch):
    _fake_discovery(
        monkeypatch,
        local={
            "outcome-tg119": {
                "repo": "outcome-tg119",
                "versions": ["v1.0.0", "v1.0.1"],
            }
        },
        remote={
            "outcome-tg119": {"repo": "outcome-tg119", "version": "v1.0.1"},
            "dosecalc-bayes-protons": {
                "repo": "dosecalc-bayes-protons",
                "version": "main",
            },
        },
    )

    specs = {s.name: s for s in list_models(offline=False)}
    assert set(specs) == {"outcome-tg119", "dosecalc-bayes-protons"}

    merged = specs["outcome-tg119"]
    assert merged.local_versions == ("v1.0.0", "v1.0.1")
    assert merged.remote_version == "v1.0.1"
    assert merged.is_local and merged.is_remote
    assert merged.modality == Modality.OUTCOME

    remote_only = specs["dosecalc-bayes-protons"]
    assert remote_only.repo == "dosecalc-bayes-protons"
    assert remote_only.local_versions == ()
    assert not remote_only.is_local
    assert remote_only.modality == Modality.DOSE_CALC


def test_offline_skips_remote(monkeypatch):
    _fake_discovery(
        monkeypatch,
        local={"outcome-tg119": {"repo": "outcome-tg119", "versions": ["local"]}},
        remote=None,  # would raise if the remote path is taken
    )

    specs = list_models(offline=True)
    assert [s.name for s in specs] == ["outcome-tg119"]
    assert specs[0].is_local and not specs[0].is_remote


def test_list_models_modality_filter(monkeypatch):
    _fake_discovery(
        monkeypatch,
        local={
            "outcome-a": {"repo": "outcome-a", "versions": ["local"]},
            "dosecalc-b": {"repo": "dosecalc-b", "versions": ["local"]},
        },
    )
    assert [s.name for s in list_models(Modality.OUTCOME, offline=True)] == ["outcome-a"]
    assert [s.name for s in list_models(Modality.DOSE_CALC, offline=True)] == ["dosecalc-b"]


def test_get_model_spec(monkeypatch):
    _fake_discovery(
        monkeypatch,
        local={
            "outcome-tg119": {
                "repo": "outcome-tg119",
                "versions": ["v1.0.0"],
            }
        },
    )
    spec = get_model_spec("outcome-tg119", offline=True)
    assert isinstance(spec, ModelSpec)
    assert spec.repo == "outcome-tg119"
    assert spec.modality == Modality.OUTCOME


def test_get_model_spec_unknown_raises(monkeypatch):
    _fake_discovery(monkeypatch, local={})
    with pytest.raises(KeyError):
        get_model_spec("does-not-exist", offline=True)


def test_resolve_repo_id():
    spec = ModelSpec(name="outcome-tg119", repo="outcome-tg119", modality=Modality.OUTCOME)
    assert resolve_repo_id(spec, "my-org") == "my-org/outcome-tg119"


def test_remote_discovery_is_cached(monkeypatch):
    _factory.clear_model_cache()
    calls = {"n": 0}

    def _remote(org):
        calls["n"] += 1
        return {"outcome-a": {"repo": "outcome-a", "version": "v1"}}

    monkeypatch.setattr(_factory, "_discover_local", lambda cfg: {})
    monkeypatch.setattr(_factory, "_discover_remote", _remote)

    list_models(offline=False)
    list_models(offline=False)
    list_models(offline=False)
    assert calls["n"] == 1  # subsequent calls served from cache

    list_models(offline=False, refresh=True)
    assert calls["n"] == 2  # refresh bypasses the cache

    _factory.clear_model_cache()
    list_models(offline=False)
    assert calls["n"] == 3  # cleared cache re-queries

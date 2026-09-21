"""Tests for the global pydantic-settings configuration."""

import pytest

import pyRadPlan
from pyRadPlan import xp_utils
from pyRadPlan._settings import AiSettings, PyRadPlanSettings, XpSettings, get_settings


@pytest.fixture
def restore_settings():
    """Snapshot the global backend settings and restore them after the test."""
    settings = get_settings()
    snapshot = settings.xp.model_dump()
    yield settings
    for name, value in snapshot.items():
        setattr(settings.xp, name, value)


def test_defaults():
    settings = PyRadPlanSettings(_env_file=None)
    assert isinstance(settings.xp, XpSettings)
    assert isinstance(settings.ai, AiSettings)
    assert settings.xp.prefer_gpu is True
    assert settings.xp.preferred_cpu_array_backend == "numpy"


def test_xp_env_override(monkeypatch):
    monkeypatch.setenv("PYRADPLAN_XP_PREFER_GPU", "false")
    monkeypatch.setenv("PYRADPLAN_XP_PREFERRED_CPU_ARRAY_BACKEND", "array_api_strict")
    monkeypatch.setenv("PYRADPLAN_XP_PREFERRED_GPU_ARRAY_BACKEND", "torch")
    settings = PyRadPlanSettings(_env_file=None)
    assert settings.xp.prefer_gpu is False
    assert settings.xp.preferred_cpu_array_backend == "array_api_strict"
    assert settings.xp.preferred_gpu_array_backend == "torch"


def test_ai_env_override(monkeypatch):
    monkeypatch.delenv("PYRADPLAN_AI_MODEL", raising=False)
    monkeypatch.setenv("PYRADPLAN_AI_AGENTS_MODEL", "my-test-model")
    settings = PyRadPlanSettings(_env_file=None)
    assert settings.ai.agents_model == "my-test-model"


def test_legacy_ai_env_names_still_read(monkeypatch):
    """PYRADPLAN_AI_MODEL / _DISPLAY_USAGE shipped in 0.4.0/0.4.1."""
    monkeypatch.delenv("PYRADPLAN_AI_AGENTS_MODEL", raising=False)
    monkeypatch.delenv("PYRADPLAN_AI_AGENTS_DISPLAY_USAGE", raising=False)
    monkeypatch.setenv("PYRADPLAN_AI_MODEL", "legacy-model")
    monkeypatch.setenv("PYRADPLAN_AI_DISPLAY_USAGE", "false")

    settings = PyRadPlanSettings(_env_file=None)
    assert settings.ai.agents_model == "legacy-model"
    assert settings.ai.agents_display_usage is False


def test_canonical_ai_env_name_wins_over_legacy(monkeypatch):
    monkeypatch.setenv("PYRADPLAN_AI_MODEL", "legacy-model")
    monkeypatch.setenv("PYRADPLAN_AI_AGENTS_MODEL", "canonical-model")
    settings = PyRadPlanSettings(_env_file=None)
    assert settings.ai.agents_model == "canonical-model"


def test_ai_section_covers_agents_and_modelhub(monkeypatch):
    """Agent and modelhub fields live side by side in the unified ai section."""
    monkeypatch.setenv("PYRADPLAN_AI_AGENTS_MODEL", "an-llm")
    monkeypatch.setenv("PYRADPLAN_AI_MODELHUB_DEVICE", "cuda")
    settings = PyRadPlanSettings(_env_file=None)
    assert settings.ai.agents_model == "an-llm"
    assert settings.ai.modelhub_device == "cuda"


def test_assignment_is_validated():
    settings = PyRadPlanSettings(_env_file=None)
    settings.xp.prefer_gpu = "no"
    assert settings.xp.prefer_gpu is False
    with pytest.raises(ValueError):
        settings.xp.prefer_gpu = "not-a-bool"


def test_singleton_exposed():
    assert pyRadPlan.settings is get_settings()


def test_deprecated_alias_read(restore_settings):
    restore_settings.xp.prefer_gpu = False
    with pytest.deprecated_call():
        assert xp_utils.PREFER_GPU is False
    with pytest.deprecated_call():
        assert (
            xp_utils.PREFERRED_CPU_ARRAY_BACKEND == restore_settings.xp.preferred_cpu_array_backend
        )


def test_deprecated_alias_write(restore_settings):
    with pytest.deprecated_call():
        xp_utils.PREFER_GPU = False
    assert restore_settings.xp.prefer_gpu is False
    with pytest.deprecated_call():
        xp_utils.PREFERRED_GPU_ARRAY_BACKEND = "array_api_strict"
    assert restore_settings.xp.preferred_gpu_array_backend == "array_api_strict"


def test_choose_namespace_follows_settings(restore_settings):
    restore_settings.xp.prefer_gpu = False
    restore_settings.xp.preferred_cpu_array_backend = "numpy"
    assert "numpy" in xp_utils.choose_array_api_namespace().__name__

    restore_settings.xp.prefer_gpu = True
    restore_settings.xp.preferred_gpu_array_backend = "array_api_strict"
    assert "array_api_strict" in xp_utils.choose_array_api_namespace().__name__


def test_unknown_xp_utils_attribute_raises():
    with pytest.raises(AttributeError):
        xp_utils.DOES_NOT_EXIST

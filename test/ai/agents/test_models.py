"""Tests for AI model discovery from configured API keys."""

import os

import pytest

pytest.importorskip("pydantic_ai")

from pyRadPlan._settings import get_settings
from pyRadPlan.ai.agents import available_models


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch):
    # Keep these tests hermetic: don't let a local .env leak provider keys in.
    monkeypatch.setattr("pyRadPlan.ai.agents._models.load_ai_env", lambda *a, **k: None)
    monkeypatch.setattr(get_settings().ai, "agents_model", "claude-sonnet-4-5")
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env, raising=False)


def test_available_models_lists_configured_default_first(monkeypatch):
    monkeypatch.setattr(get_settings().ai, "agents_model", "my-default-model")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    models = available_models()
    assert models[0] == "my-default-model"


def test_available_models_empty_without_provider_keys(monkeypatch):
    # Without any provider key not even the default model is usable.
    monkeypatch.setattr(get_settings().ai, "agents_model", "my-default-model")

    assert available_models() == []


def test_available_models_adds_provider_when_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")

    models = available_models()
    assert any("claude" in m for m in models)
    # No OpenAI key -> no OpenAI suggestions.
    assert not any("gpt" in m for m in models)


def test_load_ai_env_populates_environment(tmp_path, monkeypatch):
    from pyRadPlan.ai.agents import load_ai_env

    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    load_ai_env()
    assert os.environ.get("OPENAI_API_KEY") == "from-dotenv"


@pytest.mark.parametrize(
    "default, missing_provider",
    [
        ("claude-sonnet-4-5", "ANTHROPIC_API_KEY"),
        ("anthropic:claude-custom", "ANTHROPIC_API_KEY"),
        ("gpt-4o", "OPENAI_API_KEY"),
        ("openai:gpt-custom", "OPENAI_API_KEY"),
        ("gemini-custom", "GEMINI_API_KEY"),
        ("google-gla:gemini-custom", "GEMINI_API_KEY"),
    ],
)
def test_default_requires_its_provider_key(monkeypatch, default, missing_provider):
    monkeypatch.setattr(get_settings().ai, "agents_model", default)
    other_key = "ANTHROPIC_API_KEY" if missing_provider == "OPENAI_API_KEY" else "OPENAI_API_KEY"
    monkeypatch.setenv(other_key, "dummy")
    assert default not in available_models()
    monkeypatch.setenv(missing_provider, "dummy")
    assert available_models()[0] == default


def test_google_default_accepts_google_api_key_alias(monkeypatch):
    monkeypatch.setattr(get_settings().ai, "agents_model", "google-gla:gemini-custom")
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    assert available_models()[0] == "google-gla:gemini-custom"

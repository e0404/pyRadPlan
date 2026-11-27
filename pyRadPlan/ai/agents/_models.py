"""Discover which AI models are usable given the configured API keys."""

from __future__ import annotations

import os
from typing import Any

from pyRadPlan._settings import get_settings

from ._settings import load_ai_env

# Provider API-key env var -> a few suggested pydantic-ai model names. These are
# suggestions only; callers may type any model string the backend understands.
_PROVIDER_MODELS: dict[str, tuple[str, ...]] = {
    "ANTHROPIC_API_KEY": (
        "anthropic:claude-opus-4-8",
        "anthropic:claude-sonnet-4-6",
        "anthropic:claude-haiku-4-5",
    ),
    "OPENAI_API_KEY": ("openai:gpt-4o", "openai:gpt-4o-mini"),
    "GEMINI_API_KEY": ("google-gla:gemini-2.0-flash", "google-gla:gemini-1.5-pro"),
    "GOOGLE_API_KEY": ("google-gla:gemini-2.0-flash", "google-gla:gemini-1.5-pro"),
    # Local Ollama server: needs no API key, only its OpenAI-compatible base URL
    # (e.g. ``http://localhost:11434/v1``).
    "OLLAMA_BASE_URL": ("ollama:gpt-oss:20b",),
}


def _default_model_available(model: str) -> bool:
    """Check credentials for recognized providers, leaving custom defaults usable."""
    provider, separator, _ = model.partition(":")
    if not separator:
        if model.startswith("claude"):
            provider = "anthropic"
        elif model.startswith(("gpt", "chatgpt", "o1", "o3", "o4")):
            provider = "openai"
        elif model.startswith("gemini"):
            provider = "google-gla"
    keys = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "google-gla": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        "ollama": ("OLLAMA_BASE_URL",),
    }.get(provider)
    return keys is None or any(os.environ.get(key) for key in keys)


def available_models() -> list[str]:
    """Return suggested model names for which an API key is configured.

    The configured default (:attr:`AiSettings.agents_model`) is listed first if
    its recognized provider has credentials; custom provider defaults are retained.
    If none of the supported provider API keys is configured, no suggestions are
    returned, including the default.

    Returns
    -------
    list of str
        Suggested model identifiers, without duplicates. Empty if no provider
        API key is configured.
    """
    load_ai_env()
    models: list[str] = []
    for env_var, names in _PROVIDER_MODELS.items():
        if os.environ.get(env_var):
            for name in names:
                if name not in models:
                    models.append(name)
    if not models:
        return []
    default = get_settings().ai.agents_model
    if default and _default_model_available(default):
        if default in models:
            models.remove(default)
        models.insert(0, default)
    return models


def resolve_model(model: Any = None) -> Any:
    """Return the model the agents run with.

    ``None`` means :attr:`AiSettings.agents_model`. An Anthropic model name
    (``"anthropic:claude-..."`` or ``"claude-..."``) becomes a model that does not force
    a tool call, which newer Claude models (e.g. Sonnet 5.5) reject with an error; the
    model then calls the output tool by itself. Other names and model objects are
    returned unchanged.
    """
    if model is not None and not isinstance(model, str):
        return model
    load_ai_env()
    model = model or get_settings().ai.agents_model
    provider, separator, name = model.partition(":")
    if not separator:
        provider, name = ("anthropic" if model.startswith("claude") else ""), model
    if provider != "anthropic":
        return model

    from pydantic_ai.models.anthropic import AnthropicModel  # noqa: PLC0415
    from pydantic_ai.profiles import ModelProfile  # noqa: PLC0415

    # A partial profile is merged on top of the provider's profile for this model
    return AnthropicModel(name, profile=ModelProfile(supports_forced_tool_choice=False))

"""Discover which AI models are usable given the configured API keys."""

from __future__ import annotations

import os

from ._settings import AiSettings, load_ai_env

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
}


def available_models() -> list[str]:
    """Return suggested model names for which an API key is configured.

    The configured default (:attr:`AiSettings.model`) is listed first so a
    sensible option is preselected.  When no provider API key is configured at
    all, an empty list is returned — including the default, since it could not
    be used either.

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
    default = AiSettings().model
    if default:
        if default in models:
            models.remove(default)
        models.insert(0, default)
    return models

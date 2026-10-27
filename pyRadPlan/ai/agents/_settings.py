"""Pydantic settings for the ai.agents module.

:class:`AiSettings` is defined in :mod:`pyRadPlan._settings` as part of the
global pyRadPlan configuration and re-exported here for backward
compatibility.
"""

import os

from pyRadPlan._settings import ENV_FILE, AiSettings, find_env_files

__all__ = ["ENV_FILE", "AiSettings", "load_ai_env"]


def load_ai_env(override: bool = False) -> None:
    """Load pyRadPlan's ``.env`` files into ``os.environ``.

    pydantic-ai and :func:`~pyRadPlan.ai.agents.available_models` read provider
    API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...) from the *process*
    environment.  pydantic-settings only feeds the ``.env`` files into
    :class:`AiSettings` fields, so the API keys would otherwise never reach the
    environment.  This makes them visible by loading the files once: the
    per-user ``<data_dir>/.env`` and the working directory's ``.env``, the
    latter taking precedence (see :func:`pyRadPlan._settings.find_env_files`).

    Existing environment variables take precedence unless *override* is set, and
    a missing ``python-dotenv`` or ``.env`` file is silently ignored.
    """
    try:
        from dotenv import dotenv_values  # noqa: PLC0415
    except ImportError:  # python-dotenv is an optional convenience here
        return
    values: dict[str, str] = {}
    for path in find_env_files():
        values.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value

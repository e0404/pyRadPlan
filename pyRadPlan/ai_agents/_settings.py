"""Pydantic settings for the ai_agents module.

:class:`AiSettings` is defined in :mod:`pyRadPlan._settings` as part of the
global pyRadPlan configuration and re-exported here for backward
compatibility.
"""

from .._settings import ENV_FILE, AiSettings

__all__ = ["ENV_FILE", "AiSettings", "load_ai_env"]


def load_ai_env(override: bool = False) -> None:
    """Load a local ``.env`` file into ``os.environ``.

    pydantic-ai and :func:`~pyRadPlan.ai_agents.available_models` read provider
    API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …) from the *process*
    environment.  pydantic-settings only feeds the ``.env`` into
    :class:`AiSettings` fields, so the API keys would otherwise never reach the
    environment.  This makes them visible by loading the ``.env`` once.

    Existing environment variables take precedence unless *override* is set, and
    a missing ``python-dotenv`` or ``.env`` file is silently ignored.
    """
    try:
        from dotenv import find_dotenv, load_dotenv  # noqa: PLC0415
    except ImportError:  # python-dotenv is an optional convenience here
        return
    # ``usecwd=True`` mirrors pydantic-settings resolving ``env_file`` relative
    # to the working directory.
    path = find_dotenv(ENV_FILE, usecwd=True)
    if path:
        load_dotenv(path, override=override)

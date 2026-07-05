"""Pydantic settings for the ai_agents module."""

from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``.env`` file consulted both by :class:`AiSettings` and :func:`load_ai_env`.
ENV_FILE = ".env"


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


class AiSettings(BaseSettings):
    """Global defaults for pyRadPlan AI agents.

    Values are read from environment variables prefixed with ``PYRADPLAN_AI_``
    (e.g. ``PYRADPLAN_AI_MODEL``) or from a ``.env`` file in the working
    directory.

    API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``,
    etc.) are read directly by *pydantic-ai* from the environment — they do
    not need to be set here.

    Examples
    --------
    Configure via environment variable::

        export PYRADPLAN_AI_MODEL=gpt-4o-mini

    Or override per call::

        ai_agents.generate_beam_angles(pln, "prostate", model="claude-opus-4-6")
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRADPLAN_AI_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "claude-sonnet-4-5"
    display_usage: bool = True

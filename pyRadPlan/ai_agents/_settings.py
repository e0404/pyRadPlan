"""Pydantic settings for the ai_agents module."""

from pydantic_settings import BaseSettings, SettingsConfigDict


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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "claude-sonnet-4-5"
    display_usage: bool = True

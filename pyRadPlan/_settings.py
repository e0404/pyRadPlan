"""Global pydantic-settings configuration for pyRadPlan.

All configuration is read from environment variables prefixed with
``PYRADPLAN_`` and/or a ``.env`` file in the working directory. Sub-configurations
extend the prefix (e.g. ``PYRADPLAN_AI_`` for :class:`AiSettings`).

The module-level singleton :data:`settings` (also exposed as
``pyRadPlan.settings``) holds the runtime configuration and can be mutated to
change behavior of a running session::

    import pyRadPlan

    pyRadPlan.settings.xp.prefer_gpu = False
    pyRadPlan.settings.xp.preferred_cpu_array_backend = "numpy"
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``.env`` file consulted by all pyRadPlan settings classes.
ENV_FILE = ".env"


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
        validate_assignment=True,
    )

    model: str = "claude-sonnet-4-5"
    display_usage: bool = True


class XpSettings(BaseSettings):
    """Array-API backend ("xp") preferences.

    Values are read from environment variables prefixed with ``PYRADPLAN_XP_``
    (e.g. ``PYRADPLAN_XP_PREFER_GPU``) or from a ``.env`` file in the working
    directory.

    The values are consulted lazily by
    :func:`pyRadPlan.core.xp_utils.choose_array_api_namespace` and
    :func:`pyRadPlan.core.xp_utils.choose_device`, so mutating them on the
    :data:`settings` singleton takes effect for subsequent computations. The
    backends are *preferred* rather than mandatory — algorithms may locally
    enforce or keep a different backend.

    Examples
    --------
    Configure via environment variables or ``.env``::

        export PYRADPLAN_XP_PREFER_GPU=false
        export PYRADPLAN_XP_PREFERRED_GPU_ARRAY_BACKEND=torch

    Or at runtime::

        import pyRadPlan

        pyRadPlan.settings.xp.prefer_gpu = True
        pyRadPlan.settings.xp.preferred_gpu_array_backend = "cupy"
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRADPLAN_XP_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    prefer_gpu: bool = Field(
        default=True,
        description="Prefer a GPU array backend whenever one is available.",
    )
    preferred_cpu_array_backend: str = Field(
        default="numpy",
        description="Preferred Array-API namespace for CPU computations. "
        "Algorithms may locally enforce a different backend.",
    )
    preferred_gpu_array_backend: Optional[str] = Field(
        default=None,
        description="Preferred Array-API namespace for GPU computations. "
        "None selects the best available backend automatically. "
        "Algorithms may locally enforce a different backend.",
    )


class PyRadPlanSettings(BaseSettings):
    """Top-level pyRadPlan configuration.

    Values are read from environment variables prefixed with ``PYRADPLAN_``
    or from a ``.env`` file in the working directory. Sub-configurations use
    extended prefixes (e.g. ``PYRADPLAN_XP_PREFER_GPU`` configures :attr:`xp`,
    ``PYRADPLAN_AI_MODEL`` configures :attr:`ai`).

    Note that the AI agents read :class:`AiSettings` freshly from the
    environment at call time, so :attr:`ai` reflects the environment at
    construction of this object rather than live agent configuration. The
    :attr:`xp` backend preferences, in contrast, are read from the
    :data:`settings` singleton at runtime.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRADPLAN_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    xp: XpSettings = Field(
        default_factory=XpSettings,
        description="Array-API backend sub-configuration (PYRADPLAN_XP_* variables).",
    )
    ai: AiSettings = Field(
        default_factory=AiSettings,
        description="AI agent sub-configuration (PYRADPLAN_AI_* variables).",
    )


settings = PyRadPlanSettings()


def get_settings() -> PyRadPlanSettings:
    """Return the global :class:`PyRadPlanSettings` singleton."""
    return settings

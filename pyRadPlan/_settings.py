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

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: ``.env`` file consulted by all pyRadPlan settings classes.
ENV_FILE = ".env"


def _default_ai_models_dir() -> Path:
    """Default base directory for local AI models: ``<data_dir>/ai_models``.

    The import is deferred to keep the *textual* dependency out of this module's
    import block, but note it still runs while :mod:`pyRadPlan._settings` is
    being imported, because the :data:`settings` singleton below is constructed
    at module scope. It is safe only as long as nothing reachable from
    ``pyRadPlan/core/__init__.py`` imports :mod:`pyRadPlan._settings` back
    (:mod:`pyRadPlan.core.xp_utils` does, but is not imported there). Keep it
    that way, or make the singleton lazy.
    """
    from pyRadPlan.core import get_data_dir  # noqa: PLC0415

    return get_data_dir() / "ai_models"


class AiSettings(BaseSettings):
    """Global configuration of pyRadPlan's AI features.

    One class covers both AI subsystems, kept apart by the field-name prefix:
    the ``agents_*`` fields configure the LLM-powered planning agents
    (:mod:`pyRadPlan.ai.agents`), the ``modelhub_*`` fields configure model
    loading (:mod:`pyRadPlan.ai.modelhub`). Values are read from environment
    variables prefixed with ``PYRADPLAN_AI_`` (e.g. ``PYRADPLAN_AI_AGENTS_MODEL``,
    ``PYRADPLAN_AI_MODELHUB_DEVICE``) or from a ``.env`` file in the working
    directory. The pre-0.4.2 names ``PYRADPLAN_AI_MODEL`` and
    ``PYRADPLAN_AI_DISPLAY_USAGE`` are still read as legacy aliases.

    API keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``,
    etc.) are read directly by *pydantic-ai* from the environment — they do
    not need to be set here.

    Attributes
    ----------
    agents_model : str
        pydantic-ai model string the planning agents query (an LLM identifier
        such as ``"claude-sonnet-4-5"`` or ``"openai:gpt-4o-mini"``, not a
        modelhub model). Overridable per call via the agents' ``model=``
        keyword.
    agents_display_usage : bool
        Whether agent runs log token usage and estimated cost.
    modelhub_hf_org : str
        HuggingFace namespace/organization the model repositories live under.
        Combined with a friendly model name to form the full ``"<org>/<repo>"``
        repository id.
    modelhub_local_models_dir : Optional[Path]
        Base directory for local models. Defaults to ``<data_dir>/ai_models``
        (see :func:`pyRadPlan.core.get_data_dir`) and can be overridden via
        ``PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR``. Models are downloaded into /
        loaded from ``"<local_models_dir>/<org>/<repo>"``, which is used as the
        default ``local_dir``. Set it to ``""`` to opt out and use the
        HuggingFace cache only.
    modelhub_cache_dir : Optional[Path]
        Override for the HuggingFace cache directory. ``None`` uses the default
        HuggingFace cache location (or ``HF_HOME``).
    modelhub_offline : bool
        Force offline mode (``local_files_only``). When ``True``, no network
        access is attempted and only cached / local files are used.
    modelhub_trust_remote_code : bool
        Whether loading a model resolved from the HuggingFace Hub is allowed to
        execute the ``model.py`` and ``preprocessor.py`` shipped with it. This
        runs arbitrary Python from the model repository, so it defaults to
        ``False`` and must be opted into per source you trust. Loading from a
        directory passed explicitly as ``local_dir`` is exempt: that code is
        already under the caller's control.
    modelhub_device : str
        Default device a loaded model is moved to (e.g. ``"cpu"``, ``"cuda"``).

    Examples
    --------
    Configure via environment variables::

        export PYRADPLAN_AI_AGENTS_MODEL=gpt-4o-mini
        export PYRADPLAN_AI_MODELHUB_OFFLINE=1
        export PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1

    Or at runtime::

        pyRadPlan.settings.ai.modelhub_device = "cuda"
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRADPLAN_AI_",
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        validate_assignment=True,
    )

    # The canonical, prefix-conforming names come first; the second entries are
    # the names these settings shipped under up to 0.4.1.
    agents_model: str = Field(
        default="claude-sonnet-4-5",
        validation_alias=AliasChoices("PYRADPLAN_AI_AGENTS_MODEL", "PYRADPLAN_AI_MODEL"),
    )
    agents_display_usage: bool = Field(
        default=True,
        validation_alias=AliasChoices(
            "PYRADPLAN_AI_AGENTS_DISPLAY_USAGE",
            "PYRADPLAN_AI_DISPLAY_USAGE",
        ),
    )

    modelhub_hf_org: str = "DKFZ-RadOpt"
    modelhub_local_models_dir: Optional[Path] = Field(default_factory=_default_ai_models_dir)
    modelhub_cache_dir: Optional[Path] = None
    modelhub_offline: bool = False
    modelhub_trust_remote_code: bool = False
    modelhub_device: str = "cpu"

    @field_validator("modelhub_local_models_dir", "modelhub_cache_dir", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        """Treat an empty/blank string (e.g. ``PATH=""``) as unset."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


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
    jit_backends: str = Field(
        default="numpy,jax",
        description="Comma-separated backends that run jit-compiled kernel paths "
        "(numba implementations for numpy, the backend's own jit like jax.jit or "
        "torch.compile elsewhere; e.g. 'numpy,jax,torch'). An empty string "
        "disables jitting entirely.",
    )


class PyRadPlanSettings(BaseSettings):
    """Top-level pyRadPlan configuration.

    Values are read from environment variables prefixed with ``PYRADPLAN_``
    or from a ``.env`` file in the working directory. Sub-configurations use
    extended prefixes (e.g. ``PYRADPLAN_XP_PREFER_GPU`` configures :attr:`xp`,
    ``PYRADPLAN_AI_AGENTS_MODEL`` configures :attr:`ai`).

    All sub-configurations are consulted from the :data:`settings` singleton at
    runtime, so mutating them takes effect for subsequent computations.
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
        description="AI sub-configuration (PYRADPLAN_AI_* variables).",
    )


settings = PyRadPlanSettings()


def get_settings() -> PyRadPlanSettings:
    """Return the global :class:`PyRadPlanSettings` singleton."""
    return settings

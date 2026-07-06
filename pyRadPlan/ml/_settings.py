"""Pydantic settings for the ml module."""

from typing import Optional
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class MlSettings(BaseSettings):
    """Global defaults for pyRadPlan ML model handling.

    Values are read from environment variables prefixed with ``PYRADPLAN_ML_``
    (e.g. ``PYRADPLAN_ML_OFFLINE``) or from a ``.env`` file in the working
    directory.

    Attributes
    ----------
    hf_org : str
        HuggingFace namespace/organization the model repositories live under.
        Combined with a :class:`~pyRadPlan.ml.ModelSpec` to form the full
        ``"<org>/<repo>"`` repository id.
    local_models_dir : Optional[Path]
        Base directory for local models, read from the
        ``PYRADPLAN_HUGGINGFACE_PATH`` environment variable. When set, models
        are downloaded into / loaded from ``"<local_models_dir>/<repo>"`` and it
        is used as the default ``local_dir``.
    cache_dir : Optional[Path]
        Override for the HuggingFace cache directory. ``None`` uses the default
        HuggingFace cache location (or ``HF_HOME``).
    offline : bool
        Force offline mode (``local_files_only``). When ``True``, no network
        access is attempted and only cached / local files are used.
    trust_remote_code : bool
        Whether loading a model is allowed to execute the ``model.py``,
        ``preprocessor.py`` and ``init_param.py`` shipped with the model. This
        runs arbitrary Python from the model repository, so only enable it for
        sources you trust.
    device : str
        Default device the loaded model is moved to (e.g. ``"cpu"``,
        ``"cuda"``).

    Examples
    --------
    Configure via environment variable::

        export PYRADPLAN_ML_OFFLINE=1
        export PYRADPLAN_ML_HF_ORG=my-org
    """

    model_config = SettingsConfigDict(
        env_prefix="PYRADPLAN_ML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    hf_org: str = "DKFZ-RadOpt"
    local_models_dir: Optional[Path] = Field(
        default=None, validation_alias="PYRADPLAN_HUGGINGFACE_PATH"
    )
    cache_dir: Optional[Path] = None
    offline: bool = False
    trust_remote_code: bool = True
    device: str = "cpu"

    @field_validator("local_models_dir", "cache_dir", mode="before")
    @classmethod
    def _empty_str_to_none(cls, v):
        """Treat an empty/blank string (e.g. ``PATH=""``) as unset."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v


settings = MlSettings()


def get_ml_settings() -> MlSettings:
    """Return the global :class:`MlSettings` instance.

    Returns
    -------
    MlSettings
        The process-wide ML settings.
    """
    return settings

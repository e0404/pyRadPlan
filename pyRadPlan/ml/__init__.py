"""HuggingFace-based ML model handling for pyRadPlan.

Loads models (and their preprocessors) either from the HuggingFace Hub or from a
local directory, with version dedup and offline support. See ``README.md`` for
the model-repository contract.
"""

from ._settings import MlSettings, get_ml_settings
from ._preprocessor import BasePreprocessor
from ._factory import (
    Modality,
    ModelSpec,
    discover_models,
    list_models,
    get_model_spec,
    modality_from_name,
    resolve_repo_id,
    clear_model_cache,
)
from ._hub import resolve_model_dir, is_valid_model_dir
from ._load_model import load_model

__all__ = [
    "MlSettings",
    "get_ml_settings",
    "BasePreprocessor",
    "Modality",
    "ModelSpec",
    "discover_models",
    "list_models",
    "get_model_spec",
    "modality_from_name",
    "resolve_repo_id",
    "clear_model_cache",
    "resolve_model_dir",
    "is_valid_model_dir",
    "load_model",
]

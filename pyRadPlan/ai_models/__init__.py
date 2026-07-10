"""HuggingFace-based AI model handling for pyRadPlan.

Loads models (and their preprocessors) either from the HuggingFace Hub or from a
local directory, with version dedup and offline support. See ``README.md`` for
the model-repository contract.
"""

from ._settings import AiModelSettings, get_ai_model_settings
from ._preprocessor import BasePreprocessor
from ._factory import (
    Modality,
    list_local_models,
    modality_from_name,
)
from ._hub import resolve_model_dir, is_valid_model_dir
from ._load_model import load_model

__all__ = [
    "AiModelSettings",
    "get_ai_model_settings",
    "BasePreprocessor",
    "Modality",
    "list_local_models",
    "modality_from_name",
    "resolve_model_dir",
    "is_valid_model_dir",
    "load_model",
]

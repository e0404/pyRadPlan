"""HuggingFace-based AI model handling for pyRadPlan.

Loads models (and their preprocessors) either from the HuggingFace Hub or from a
local directory, with version dedup and offline support. See the "AI Model Hub"
section of the user guide for the configuration and the model-repository
contract.
"""

from ._settings import AiSettings
from ._preprocessor import BasePreprocessor
from ._registry import (
    ModelTask,
    list_local_models,
    task_from_dir,
    task_from_name,
)
from ._resolve import resolve_model_dir, is_valid_model_dir, repo_subpath
from ._load_model import load_model

__all__ = [
    "AiSettings",
    "BasePreprocessor",
    "ModelTask",
    "list_local_models",
    "task_from_dir",
    "task_from_name",
    "resolve_model_dir",
    "is_valid_model_dir",
    "repo_subpath",
    "load_model",
]

"""HuggingFace-based AI model handling for pyRadPlan.

Loads models (and their preprocessors) either from the HuggingFace Hub or from a
local directory, with version dedup and offline support. See ``README.md`` for
the model-repository contract.
"""

from ._settings import AiModelSettings
from ._preprocessor import BasePreprocessor
from ._registry import (
    ModelTask,
    list_local_models,
    task_from_dir,
    task_from_name,
)
from ._hub import resolve_model_dir, is_valid_model_dir, repo_subpath
from ._load_model import load_model
from ._pyanno4rt import OutcomeFeature, OutcomeModelSpec, load_pyanno4rt_model

__all__ = [
    "AiModelSettings",
    "BasePreprocessor",
    "ModelTask",
    "list_local_models",
    "task_from_dir",
    "task_from_name",
    "resolve_model_dir",
    "is_valid_model_dir",
    "repo_subpath",
    "load_model",
    "OutcomeFeature",
    "OutcomeModelSpec",
    "load_pyanno4rt_model",
]

from __future__ import annotations
import logging
import warnings
from typing import List
from pyRadPlan.bio_models._base import BiologicalModel

BIO_MODELS = {}

logger = logging.getLogger(__name__)

"""
Biological model registry for pyRadPlan.

This module maintains a global registry of available biological models and
provides utilities for registering, querying, and instantiating them.

The registry maps model names (and their aliases) to their corresponding
:class:`~pyRadPlan.bio_models._base.BiologicalModel` subclasses. Models
are registered via :func:`register_model` and can be retrieved by name using
:func:`get_bio_model`. Only models compatible with a given radiation mode and
the quantities provided by the dose engine are considered available.

Registry
--------
BIO_MODELS : dict[str, type[BiologicalModel]]
    Global mapping of model name / alias strings to model classes. Populated
    at import time as models are registered with :func:`register_model`.

Functions
---------
register_model
    Add a new biological model class to the registry.
get_available_models
    Filter the registry to models compatible with a given radiation mode and
    set of provided quantities.
get_bio_model
    Look up and instantiate a biological model by name, falling back to
    :class:`~pyRadPlan.bio_models._base.EmptyModel` when no match is found.
"""


def register_model(model_cls: BiologicalModel) -> None:
    """
    Register a new model.

    Parameters
    ----------
    model_cls : type
        A Biological Model class.
    """
    if not issubclass(model_cls, BiologicalModel):
        raise ValueError("Model must be a subclass of BiologicalModel.")

    if model_cls.model is None:
        raise ValueError("Model must have a 'model' attribute.")

    if model_cls.possible_radiation_modes is None:
        raise ValueError("Model must have a 'possible_radiation_modes' attribute.")

    for name in [model_cls.model, *model_cls.model_aliases]:
        if name in BIO_MODELS:
            warnings.warn(f"Model '{name}' is already registered.")
        else:
            BIO_MODELS[name] = model_cls


def get_available_models(
    radiation_mode: str,
    provided_quantities: List[str],
) -> dict[str, type[BiologicalModel]]:
    """
    Return all registered models given the radiaiton mode and provided quantities.
    """
    result = {}
    for cls in set(BIO_MODELS.values()):
        if radiation_mode in cls.possible_radiation_modes and all(
            q in provided_quantities for q in cls.required_quantities
        ):
            result[cls.model] = cls
            for alias in cls.model_aliases:
                result[alias] = cls
    return result


def get_bio_model(
    model_id: str, radiation_mode: str, provided_quantities: List[str]
) -> BiologicalModel:
    """
    Instantiate a biological model by name.

    Raises
    ------
    ValueError
        If the model is unknown, or registered but not available for the given
        radiation mode / provided quantities.
    """
    if model_id not in BIO_MODELS:
        raise ValueError(
            f"Unknown biological model '{model_id}'. Registered models: {sorted(BIO_MODELS)}"
        )
    available = get_available_models(radiation_mode, provided_quantities)
    if model_id not in available:
        _, msg = BIO_MODELS[model_id]().is_available(radiation_mode, provided_quantities)
        raise ValueError(f"Biological model '{model_id}' not available: {msg}")
    return available[model_id]()

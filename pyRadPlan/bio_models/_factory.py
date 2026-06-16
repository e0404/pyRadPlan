from __future__ import annotations
import logging
import warnings
from typing import List
from pyRadPlan.bio_models._base import BiologicalModelBase, EmptyModel

BIO_MODELS = {}

logger = logging.getLogger(__name__)

"""
Biological model registry for pyRadPlan.

This module maintains a global registry of available biological models and
provides utilities for registering, querying, and instantiating them.

The registry maps model names (and their aliases) to their corresponding
:class:`~pyRadPlan.bio_models._base.BiologicalModelBase` subclasses. Models
are registered via :func:`register_model` and can be retrieved by name using
:func:`get_bio_model`. Only models compatible with a given radiation mode and
the quantities provided by the dose engine are considered available.

Registry
--------
BIO_MODELS : dict[str, type[BiologicalModelBase]]
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


def register_model(model_cls: BiologicalModelBase) -> None:
    """
    Register a new model.

    Parameters
    ----------
    model_cls : type
        A Biological Model class.
    """
    if not issubclass(model_cls, BiologicalModelBase):
        raise ValueError("Model must be a subclass of BiologicalModelBase.")

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
) -> dict[str, type[BiologicalModelBase]]:
    """
    Return all registered models compatible with the given runtime context.
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
) -> BiologicalModelBase:
    """
    Instantiate a biological model by name, with a safe fallback.

    """
    try:
        class_list = get_available_models(radiation_mode, provided_quantities)
    except RuntimeError:
        warnings.warn(
            "Biological Model not found, creating Empty Model!",
            Warning,
            stacklevel=0,
        )
        return EmptyModel()

    model_names = [info for info in class_list]
    if model_id not in model_names:
        warnings.warn(
            f"Biological model '{model_id}' not found. Creating EmptyBiologicalModel.",
            Warning,
            stacklevel=0,
        )
        return EmptyModel()
    return class_list[model_id]()

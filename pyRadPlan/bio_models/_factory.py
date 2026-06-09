from __future__ import annotations
import logging
import warnings
from typing import List
from pyRadPlan.bio_models._base import BiologicalModelBase, EmptyModel

BIO_MODELS = {}

logger = logging.getLogger(__name__)


def register_model(model_cls: Type[BiologicalModelBase]) -> None:
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

    model_name = model_cls.model
    if model_name in BIO_MODELS:
        warnings.warn(f"Model '{model_name}' is already registered.")
    else:
        BIO_MODELS[model_name] = model_cls


def get_available_models(
    radiation_mode: str,
    provided_quantities: List[str],
) -> dict[str, type[BiologicalModelBase]]:
    return {
        name: cls
        for name, cls in BIO_MODELS.items()
        if radiation_mode in cls.possible_radiation_modes
        and all(q in provided_quantities for q in cls.required_quantities)
    }


def get_bio_model(
    model_id: str, radiation_mode: str, provided_quantities: List[str]
) -> BiologicalModelBase:
    """
    Instantiate a biological model by MODEL_ID string.

    Args:
        model_id: Short model identifier, e.g. 'MCN', 'LEM', 'none'.
        radiation_mode: The radiation mode for which to create the model.
        provided_quantities: The list of provided quantities.

    Returns
    -------
        A concrete BiologicalModel instance.
    """
    try:
        class_list = get_available_models(radiation_mode, provided_quantities)
    except RuntimeError:
        warnings.warn(
            "Biological Model not found, creating Empty Model!",
            UserWarning,
            stacklevel=2,
        )
        return EmptyModel()

    model_names = [info for info in class_list]
    if model_id not in model_names:
        warnings.warn(
            f"Biological model '{model_id}' not found. Creating EmptyBiologicalModel.",
            UserWarning,
            stacklevel=2,
        )
        return EmptyModel()
    return class_list[model_id]()

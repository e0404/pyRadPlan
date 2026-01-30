"""AI dose model base classes and registry.

This module defines the lightweight AI model interface used by AI-enabled dose engines
and provides a simple registry to discover and instantiate models based on the Plan.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, ClassVar, Optional, Type, Union, Dict

from pyRadPlan.plan import Plan, validate_pln

logger = logging.getLogger(__name__)


class AIDoseModelBase:
    """
    Lightweight AI dose model interface.

    Subclasses must:
    - set `model_name` (str) and `possible_radiation_modes` (list[str])
    - implement `_predict_dose(ct_cuboid, energy)` returning a 3D array with
      shape (depth, lateral, lateral) and dtype float32.

    The engine injects `xp` (array namespace) and `grid_information` before calls.
    """

    model_name: ClassVar[Optional[str]] = None
    possible_radiation_modes: ClassVar[Optional[list[str]]] = None

    # Injected by engine at runtime
    xp: Any = None
    grid_information: Dict[str, Any] | None = None

    def __init__(self, pln: Optional[Union[Plan, dict]] = None) -> None:
        self.pln = pln

    def _predict_dose(self, ct_cuboid: Any, energy: Any):
        raise NotImplementedError("AI model must implement _predict_dose(ct_cuboid, energy)")


AI_MODELS: dict[str, Type[AIDoseModelBase]] = {}


def register_model(models_cls: Type[AIDoseModelBase]) -> None:
    """
    Register a new AI model class.

    Parameters
    ----------
    model_cls : Type[AIDoseModelBase]
            An AI model class.
    """
    if not issubclass(models_cls, AIDoseModelBase):
        raise ValueError("Model must be a subclass of AIDoseModelBase.")

    if models_cls.model_name is None:
        raise ValueError("Model must have a 'model_name' attribute.")

    if models_cls.possible_radiation_modes is None:
        raise ValueError("Model must have a 'possible_radiation_modes' attribute.")

    model_name = models_cls.model_name
    if model_name in AI_MODELS:
        warnings.warn(f"Model '{model_name}' is already registered.")
    else:
        AI_MODELS[model_name] = models_cls


def get_available_models(pln: Union[Plan, dict[str]]) -> dict[str, Type[AIDoseModelBase]]:
    """
    Get a list of available models based on the plan.

    Parameters
    ----------
    pln : Union[Plan,dict]
            A Plan object.

    Returns
    -------
    dict[str, Type[AIDoseModelBase]]
            Mapping of model names to model classes available for the plan's radiation mode.
    """
    pln = validate_pln(pln)
    return {
        name: cls
        for name, cls in AI_MODELS.items()
        if pln.radiation_mode in cls.possible_radiation_modes
    }


def get_model(pln: Union[Plan, dict]) -> AIDoseModelBase:
    """
    Get the appropriate AI model based on the plan.

    Parameters
    ----------
    pln : Plan | dict
            A Plan object or dict.

    Returns
    -------
    AIDoseModelBase
            An AI model instance implementing `_predict_dose`.
    """
    pln = validate_pln(pln)

    # Available models
    models = get_available_models(pln)
    if len(models) <= 0:
        raise ValueError(f"No model available for radiation mode '{pln.radiation_mode}'.")

    model_names = list(models.keys())

    # Did the user provide a model in the pln?
    if isinstance(pln.prop_dose_calc, AIDoseModelBase):
        model_name = pln.prop_dose_calc.model_name
        if model_name not in models:
            warnings.warn(f"Model '{model_name}' seems not to be valid for Plan setup.")
        return pln.prop_dose_calc

    if isinstance(pln.prop_dose_calc, dict):
        # The user provided a dictionary with model parameters, so we need to find the model name
        if "model" in pln.prop_dose_calc:
            if pln.prop_dose_calc["model"] in models:
                return models[pln.prop_dose_calc["model"]](pln)
            warnings.warn(f"Model '{pln.prop_dose_calc['model']}' not available for Plan.")
        # Fallback to engine if present: deprecated alias
        if "engine" in pln.prop_dose_calc:
            if pln.prop_dose_calc["engine"] in models:
                warnings.warn(
                    "Using 'engine' key to select an AI model is deprecated; use 'model' instead."
                )
                return models[pln.prop_dose_calc["engine"]](pln)
            warnings.warn(
                f"Engine '{pln.prop_dose_calc['engine']}' is not an AI model name and will be ignored."
            )

        # If no model name was found, we choose the first as default
        logger.warning(
            "No model specified in Plan. Using first available model %s.", model_names[0]
        )
        return models[model_names[0]](pln)

    raise ValueError(f"No model available for radiation mode '{pln.radiation_mode}'.")


__all__ = [
    "AIDoseModelBase",
    "register_model",
    "get_available_models",
    "get_model",
]

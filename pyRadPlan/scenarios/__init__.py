"""Scenario Models for uncertainty quantification, robust optimization and robustness analysis."""

from typing import Union, Optional, Type
from pyRadPlan.ct import CT, validate_ct
from ._base import ScenarioModel
from ._nominal import NominalScenario
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


def available_scenario_models() -> list[str]:
    """Return a list of available scenario models."""

    return ["nomScen"]


def create_scenario_model(
    model_def: Union[ScenarioModel, Union[str, dict]] = "nomScen",
    ct: Optional[Union[CT, dict]] = None,
) -> ScenarioModel:
    """
    Return a scenario model object.

    Parameters
    ----------
    model_name : str
        The name of the scenario model.

    Returns
    -------
    ScenarioModel
        The scenario model object.
    """

    if isinstance(ct, dict):
        ct = validate_ct(ct)

    if isinstance(model_def, str):
        model_class = _get_model_class_by_name(model_def)
        return model_class(ct)

    if isinstance(model_def, ScenarioModel):
        return model_def

    if isinstance(model_def, dict):
        model_name = model_def.pop("model", None)
        if model_name is None:
            raise ValueError("Scenario model name not provided in the model definition")

        model_class = _get_model_class_by_name(model_name)
        return model_class(ct, **model_def)

    raise ValueError("Invalid scenario model definition")


def _get_model_class_by_name(model_name: str) -> Type[ScenarioModel]:
    """Return the scenario model class by name."""

    if model_name == "nomScen":
        return NominalScenario
    if model_name == "wcScen":
        raise NotImplementedError("Worst Case Scenarios are not implemented yet")
    if model_name == "impScen":
        raise NotImplementedError("Gridded Importance Weighted Scenarios are not implemented yet")
    if model_name == "rndScen":
        raise NotImplementedError("Random Scenarios are not implemented yet")
    raise ValueError(f"Unknown scenario model: {model_name}")


def validate_scenario_model(
    model_def: Union[ScenarioModel, Union[str, dict]], ct: Union[CT, dict] = None
) -> ScenarioModel:
    """
    Validate a scenario model input and returns a scenario model object.

    Parameters
    ----------
    model_def : Union[ScenarioModel, Union[str, dict]]
        The scenario model object.

    Returns
    -------
    ScenarioModel
        The scenario model object.
    """
    model_def = deepcopy(model_def)
    if "ctScenProb" in model_def:
        if all(a > 0 for a, b in model_def["ctScenProb"]):
            model_def["ctScenProb"] = [(a - 1, b) for (a, b) in model_def["ctScenProb"]]
            logger.info(
                "Assuming scenario model from matRad. Converting 1-based indexing to 0-based "
                "indexing..."
            )
        else:
            logger.info(
                "CamelCase was used, but assuming 0-based indexing, since some values are less "
                "than 1. No index conversion applied"
            )

    return create_scenario_model(model_def, ct)


__all__ = ["ScenarioModel", "NominalScenario", "create_scenario_model"]

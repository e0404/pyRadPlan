"""Factory methods to manage available objective implementations."""

import warnings
import logging
from typing import Union, Type
from ._objective import Objective

__matrad_name_map__ = {
    "DoseObjectives.matRad_SquaredDeviation": "Squared Deviation",
    "DoseObjectives.matRad_SquaredUnderdosing": "Squared Underdosing",
    "DoseObjectives.matRad_SquaredOverdosing": "Squared Overdosing",
    "DoseObjectives.matRad_MeanDose": "Mean Dose",
    "DoseObjectives.matRad_EUD": "EUD",
    "DoseObjectives.matRad_MinDVH": "MinDVH",
    "DoseObjectives.matRad_MaxDVH": "MaxDVH",
}

OBJECTIVES = {}

logger = logging.getLogger(__name__)


def register_objective(obj_cls: Type[Objective]) -> None:
    """
    Register a new objective.

    Parameters
    ----------
    obj_cls : type
        An Objective class.
    """
    if not issubclass(obj_cls, Objective):
        raise ValueError("Objective must be a subclass of Objective.")

    if obj_cls.name is None:
        raise ValueError("Objective must have a 'name' attribute.")

    obj_name = obj_cls.name
    if obj_name in OBJECTIVES:
        warnings.warn(f"Objective '{obj_name}' is already registered.")
    else:
        OBJECTIVES[obj_name] = obj_cls


def get_available_objectives() -> dict[str, Type[Objective]]:
    """
    Get a list of available objectives.

    Returns
    -------
    list
        A list of available objectives.
    """
    return OBJECTIVES


def get_objective(objective_desc: Union[str, dict, Objective]):
    """
    Return a objective instance based on a descriptive parameter.

    Parameters
    ----------
    objective_desc : Union[str, dict, Objective]
        A string with the objective name, a dictionary with the objective configuration or a
        objective instance

    Returns
    -------
    Objective
        A objective instance
    """
    if isinstance(objective_desc, str):
        objective = OBJECTIVES[objective_desc]()
    elif isinstance(objective_desc, dict):
        if "name" not in objective_desc:
            logger.debug("Objective not found, trying matRad-like objective.")
            if "className" not in objective_desc:
                raise ValueError(f"Invalid objective description: {objective_desc}")
            objective_name = __matrad_name_map__.get(objective_desc["className"], None)
            if objective_name is None:
                raise ValueError(f"Invalid objective description: {objective_desc}")
        else:
            objective_name = objective_desc["name"]

        objective_model = OBJECTIVES[objective_name]
        objective = objective_model.model_validate(objective_desc)
    elif isinstance(objective_desc, Objective):
        objective = objective_desc
    else:
        raise ValueError(f"Invalid objective description: {objective_desc}")

    return objective

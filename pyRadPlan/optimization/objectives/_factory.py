"""Factory methods to manage available objective implementations."""

import warnings
import logging
from typing import Annotated, Any, Optional, Union, Type

from pydantic import Field

from ._objective import Objective

__matrad_name_map__ = {
    "DoseObjectives.matRad_SquaredDeviation": "Squared Deviation",
    "DoseObjectives.matRad_SquaredUnderdosing": "Squared Underdosing",
    "DoseObjectives.matRad_SquaredOverdosing": "Squared Overdosing",
    "DoseObjectives.matRad_MeanDose": "Mean Dose",
    "DoseObjectives.matRad_EUD": "EUD",
    "DoseObjectives.matRad_MinDVH": "Min DVH",
    "DoseObjectives.matRad_MaxDVH": "Max DVH",
}

#: Inverse of ``__matrad_name_map__`` (pyRadPlan objective name -> matRad className).
__matrad_class_map__ = {name: class_name for class_name, name in __matrad_name_map__.items()}

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

    obj_name = obj_cls.model_fields["name"].default
    if not isinstance(obj_name, str):
        raise ValueError("Objective must define a default 'name'.")
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


def get_objectives_union(exclude_image_references: bool = False) -> Any:
    """
    Build a discriminated union type of all registered objectives.

    The union is tagged on the ``name`` field and can be used as a pydantic
    annotation wherever serialized objectives need to be (re-)validated into
    their concrete classes, e.g. in structure sets or LLM output schemas.

    Parameters
    ----------
    exclude_image_references : bool, optional
        Exclude objectives with image-reference parameters (e.g. a reference
        dose image), which cannot be represented in a JSON schema.

    Returns
    -------
    Any
        A type annotation for the discriminated union of registered objectives.
    """
    obj_classes = [
        obj_cls
        for obj_cls in OBJECTIVES.values()
        if not (exclude_image_references and "image_reference" in obj_cls._parameter_types())
    ]
    if len(obj_classes) == 1:
        return obj_classes[0]
    return Annotated[Union[tuple(obj_classes)], Field(discriminator="name")]


def get_matrad_class_name(objective: Union[str, Objective]) -> Optional[str]:
    """
    Return the matRad className for an objective, or ``None`` if it has no equivalent.

    Parameters
    ----------
    objective : Union[str, Objective]
        An objective instance or its pyRadPlan name.

    Returns
    -------
    Optional[str]
        The matRad className (e.g. ``"DoseObjectives.matRad_MeanDose"``) or ``None``
        when the objective cannot be represented in matRad.
    """
    name = objective.name if isinstance(objective, Objective) else objective
    return __matrad_class_map__.get(name)


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

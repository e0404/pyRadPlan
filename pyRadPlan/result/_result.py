"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

import os
from abc import ABC
from typing import Annotated, Any, Union
import SimpleITK as sitk
from pydantic import (
    Field,
    model_validator,
)
from pyRadPlan.result._quantity_cube import QuantityCube
from pyRadPlan.core import PyRadPlanBaseModel


class PlanResult(PyRadPlanBaseModel, ABC):
    """
    A class representing the result of a dose calculation.

    This class is used to store and do further sequencing/analysis of the dose cubes.
    """

    physical_dose: Annotated[QuantityCube, Field(default=None)]
    let: Annotated[QuantityCube, Field(default=None)]
    alpha_dose: Annotated[QuantityCube, Field(default=None)]
    sqrt_beta_dose: Annotated[QuantityCube, Field(default=None)]

    info: dict[str, Any] = Field(default_factory=dict)
    # TODO: used optimizer...

    @model_validator(mode="before")
    @classmethod
    def wrap_and_check_dose_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            dose_fields = ("physical_dose", "let", "alpha_dose", "sqrt_beta_dose")
            provided = False
            for field in dose_fields:
                if field in data and data[field] is not None:
                    provided = True
                    if isinstance(data[field], sitk.Image):
                        data[field] = {"cube": data[field]}
            if not provided:
                raise ValueError("At least one dose field is required.")
            if "info" in data and not isinstance(data["info"], dict):
                data["info"] = vars(data["info"])
        return data

    def sequencing(self) -> None:
        """Perform sequencing on the dose cube."""
        # TODO
        pass

    def analyse_plan(self) -> None:
        """Perform analysis on the plan."""
        # TODO
        pass


def create_result(
    data: Union[dict[str, Any], PlanResult, os.PathLike, str, None] = None, **kwargs
) -> PlanResult:
    """
    Create a Result object from various input types.

    Parameters
    ----------
    data : Union[dict[str, Any], Result, os.PathLike, str, None], optional
        The input data to create the Result object from. Can be a dictionary,
        existing Result object, file path, or None.
    **kwargs
        Additional keyword arguments to create the Result object.

    Returns
    -------
    Result
        A Result object created from the input data or keyword arguments.
    """
    if data:
        if isinstance(data, PlanResult):
            return data

        if isinstance(data, dict):
            # Merge additional key-value pairs (like info) into the dictionary.
            data.update(kwargs)
            return PlanResult.model_validate(data)
        return PlanResult.model_validate(data)
    return PlanResult(**kwargs)


def validate_result(
    cube: Union[dict[str, Any], PlanResult, os.PathLike, None] = None, **kwargs
) -> PlanResult:
    """
    Validate the cube input and return a Result object.

    Parameters
    ----------
    cube : Union[dict[str, Any], Result, os.PathLike, None]
        The input cube to validate.

    Returns
    -------
    Result
        A validated Result object.
    """

    return create_result(cube, **kwargs)

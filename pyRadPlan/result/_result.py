"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import Annotated, Any, Union
import SimpleITK as sitk
import numpy as np
from pydantic import (
    Field,
    SerializationInfo,
    model_validator,
    field_serializer,
)
from pyRadPlan.result._quantity_cube import QuantityCube
from pyRadPlan.core import PyRadPlanBaseModel


class PlanResult(PyRadPlanBaseModel):
    """
    A class representing the result of a dose calculation.

    This class is used to store and do further sequencing/analysis of the dose cubes.
    """

    physical_dose: Annotated[QuantityCube, Field(default=None)]
    physical_dose_beam: Annotated[list[QuantityCube], Field(default=None)]

    let: Annotated[QuantityCube, Field(default=None)]
    let_beam: Annotated[list[QuantityCube], Field(default=None)]

    alpha_dose: Annotated[QuantityCube, Field(default=None)]
    # TODO: add alpha_dose_beams
    sqrt_beta_dose: Annotated[QuantityCube, Field(default=None)]
    # TODO: add sqrt_beta_dose_beam

    fluence: Annotated[np.ndarray, Field(default=None)]
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

    @model_validator(mode="before")
    @classmethod
    def wrap_and_check_beam_fields(cls, data: Any) -> Any:
        """Validate and process beam fields (e.g., physical_dose_beam, let_dose_beam)."""
        if isinstance(data, dict):
            # List of beam fields to process
            beam_fields = ["physical_dose_beam", "let_beam"]

            for field_name in beam_fields:
                if field_name in data and data[field_name] is not None:
                    processed_beams = []
                    for beam in data[field_name]:
                        if isinstance(beam, sitk.Image):
                            processed_beams.append({"cube": beam})
                        elif isinstance(beam, QuantityCube):
                            processed_beams.append(beam)
                        else:
                            raise ValueError(
                                f"Invalid type for {field_name} entry: {type(beam)}. "
                                "Expected SimpleITK.Image or QuantityCube."
                            )
                    data[field_name] = processed_beams
        return data

    @field_serializer("physical_dose", "let", "alpha_dose", "sqrt_beta_dose")
    def custom_dose_serializer(self, v: QuantityCube, info: SerializationInfo) -> Any:
        """Serialize the dose fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            if v:
                numpy_array = np.transpose(sitk.GetArrayFromImage(v.cube), (1, 2, 0))

                wrapped_array = np.ndarray(shape=(1,), dtype=object)
                wrapped_array[0] = numpy_array
                return wrapped_array
            return None
        return v.model_dump(by_alias=info.by_alias) if v else None

    @field_serializer("physical_dose_beam", "let_beam")
    def custom_dose_beam_serializer(self, v: list[QuantityCube], info: SerializationInfo) -> Any:
        """Serialize the dose beam fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            if v and len(v) > 0:
                # Create a flat dictionary to hold individual beam fields
                flat_beam_dict = {}
                camel_case_field_name = to_camel_case(
                    info.field_name
                )  # Convert field name to camelCase
                for idx, beam in enumerate(v, start=1):
                    numpy_array = np.transpose(sitk.GetArrayFromImage(beam.cube), (1, 2, 0))
                    wrapped_array = np.ndarray(shape=(1,), dtype=object)
                    wrapped_array[0] = numpy_array

                    # Add the serialized beam directly to the flat dictionary with a dynamic key
                    flat_beam_dict[f"{camel_case_field_name}{idx}"] = wrapped_array
                return flat_beam_dict
            return None
        # Default serializatio
        return (
            {
                f"{info.field_name}{idx}": dose.model_dump(by_alias=info.by_alias)
                for idx, dose in enumerate(v, start=1)
            }
            if v
            else None
        )

    def sequencing(self) -> None:
        """Perform sequencing on the dose cube."""
        # TODO
        pass

    def analyse_plan(self) -> None:
        """Perform analysis on the plan."""
        # TODO
        pass

    def to_matrad(self, context: str = "mat-file") -> Any:
        export = super().to_matrad(context=context)
        if "physicalDoseBeam" in export and isinstance(export["physicalDoseBeam"], dict):
            physical_dose_beam_dict = export.pop("physicalDoseBeam")
            export.update(physical_dose_beam_dict)
        # Remove keys with None values from the export dictionary
        # (savemat cant handle "None")
        export = {key: value for key, value in export.items() if value is not None}
        return export  # ["resultGUI"]


def create_result(
    data: Union[dict[str, Any], PlanResult, str, None] = None, **kwargs
) -> PlanResult:
    """
    Create a Result object from various input types.

    Parameters
    ----------
    physical_dose : Union[dict[str, Any], PlanResult, None]
        The physical dose cube
    physical_dose_beam : Union[dict[str, Any], PlanResult, None]
        The physical dose beam cube
    let : Union[dict[str, Any], PlanResult, None]
        The LET cube
    let_dose_beam : Union[dict[str, Any], PlanResult, None]
        The LET dose beam cube
    alpha_dose : Union[dict[str, Any], PlanResult, None]
        The alpha dose cube
    sqrt_beta_dose : Union[dict[str, Any], PlanResult, None]
        The square root of beta dose cube
    fluence : Union[dict[str, Any], PlanResult, None]
        The fluence cube
    info : dict[str, Any], optional
        Additional information about the result. Passed on e.g. the optimier
    **kwargs
        Additional keyword arguments to create the Result object.

    Returns
    -------
    Result
        A PlanResult object created from the input data or keyword arguments.
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


def validate_result(cube: Union[dict[str, Any], PlanResult, None] = None, **kwargs) -> PlanResult:
    """
    Validate a PlanResult object or create a new one.

    Parameters
    ----------
    physical_dose : Union[dict[str, Any], PlanResult, None]
        The physical dose cube
    physical_dose_beam : Union[dict[str, Any], PlanResult, None]
        The physical dose beam cube
    let : Union[dict[str, Any], PlanResult, None]
        The LET cube
    let_dose_beam : Union[dict[str, Any], PlanResult, None]
        The LET dose beam cube
    alpha_dose : Union[dict[str, Any], PlanResult, None]
        The alpha dose cube
    sqrt_beta_dose : Union[dict[str, Any], PlanResult, None]
        The square root of beta dose cube
    fluence : Union[dict[str, Any], PlanResult, None]
        The fluence cube
    info : dict[str, Any], optional
        Additional information about the result. Passed on e.g. the optimier
    **kwargs
        Additional keyword arguments to create the Result object.

    Returns
    -------
    Result
        A PlanResult object created from the input data or keyword arguments.
    """

    return create_result(cube, **kwargs)


# TODO: might be placed somewhere else
def to_camel_case(snake_str: str) -> str:
    """Convert a snake_case string to camelCase."""
    # necessary since iteration over every beam creates a new instance which will not be handled by pydantic.
    # So custom camel_case function is needed here.
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])

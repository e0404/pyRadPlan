"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import List, Any, Union, Dict
import numpy as np
from pydantic import (
    Field,
    PrivateAttr,
    # SerializationInfo,
    model_validator,
    # field_serializer,
)
from pyRadPlan.result._quantity_distribution import QuantityDistribution
from pyRadPlan.core import PyRadPlanBaseModel


class BeamAccessor:
    def __init__(self, beam_quantities: dict):
        self._beam_quantities = beam_quantities

    def __getattr__(self, attr):
        if attr in self._beam_quantities:
            return self._beam_quantities[attr]
        raise AttributeError(f"'BeamAccessor' object has no attribute '{attr}'")


class PlanResult(PyRadPlanBaseModel):
    fluence: np.ndarray = Field(default=None)
    info: dict[str, Any] = Field(default_factory=dict)
    used_solver: str = Field(default="")

    # Declare private attributes; they are not part of the public validated model!
    _quantities: Dict[str, QuantityDistribution] = PrivateAttr(default_factory=dict)
    _beam_quantities: Dict[str, List[QuantityDistribution]] = PrivateAttr(default_factory=dict)

    class Config:
        extra = "allow"

    @property
    def stored_quantities(self) -> List[str]:
        """Return the list of stored quantities."""
        return list(self._quantities.keys()) if self._quantities else []

    @property
    def stored_beam_quantities(self) -> List[str]:
        """Return the list of stored beam quantities."""
        return list(self._beam_quantities.keys()) if self._beam_quantities else []

    @model_validator(mode="before")
    @classmethod
    def aggregate_dynamic_quantities(cls, data: Any) -> Any:
        if data is None or data == {}:
            raise ValueError("No data provided for PlanResult creation.")
        if isinstance(data, dict):
            reserved = {"fluence", "info", "used_solver", "beams"}
            # Place dynamic values into temporary public keys.
            quantities_tmp = {}
            beam_quantities_tmp = {}
            for key, value in list(data.items()):
                if key in reserved:
                    continue
                if not isinstance(value, list):
                    # The Cubes have to be validated because validation does not work via private attributes.
                    quantities_tmp[key] = QuantityDistribution.model_validate(value)
                else:
                    new_key = key.replace("_beam", "") if "_beam" in key else key
                    beam_quantities_tmp[new_key] = [
                        QuantityDistribution.model_validate(v) for v in value
                    ]
                    # Remove the key
                data.pop(key)
            data["_quantities"] = quantities_tmp
            data["_beam_quantities"] = beam_quantities_tmp

            return data

        return data

    @model_validator(mode="after")
    def set_dynamic_quantities(self, data: Any):
        # this workaround is needed because _quantities are filled AFTER the model is created.
        # Without __pydantic_extra__ there would be no data to set.
        # also by that, the logic in the before-validator is kept simple.
        extra = self.__pydantic_extra__ if hasattr(self, "__pydantic_extra__") else {}
        quantities = extra.get("_quantities", {})
        beam_quantities = extra.get("_beam_quantities", {})

        # these setters are needed to avoid infinite recursion by making the quantities an attribute.
        # Otherwise its not possible to call e.g. "physical_dose"
        object.__setattr__(self, "_quantities", quantities)
        object.__setattr__(self, "_beam_quantities", beam_quantities)

        # removing the extra attributes - if quantities should be visible,
        # remove these lines again.
        extra.pop("_quantities", None)
        extra.pop("_beam_quantities", None)
        object.__setattr__(self, "__pydantic_extra__", extra)

        return self

    def __getattr__(self, item):
        # First try standard dynamic quantities.
        quantities = object.__getattribute__(self, "_quantities")
        if item in quantities:
            return quantities[item]
        # If 'beam' is requested, return a BeamAccessor instance.
        if item == "beam":
            beam_quantities = object.__getattribute__(self, "_beam_quantities")

            return BeamAccessor(beam_quantities)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, name, value):
        if name == "_quantities":
            raise Warning("Direct modification of '_quantities' is not allowed. Input ignored.")
        if name == "_beam_quantities":
            raise Warning(
                "Direct modification of '_beam_quantities' is not allowed. Input ignored."
            )
        super().__setattr__(name, value)

    def add_quantity(self, key: str, value: Union[dict, QuantityDistribution]) -> None:
        """Add or update a quantity entry."""
        quantities = object.__getattribute__(self, "_quantities")
        quantities[key] = QuantityDistribution.model_validate(value)

    def add_beam_quantity(self, key: str, value: Union[dict, QuantityDistribution]) -> None:
        """Add or update a beam quantity entry."""
        beam_quantities = object.__getattribute__(self, "_beam_quantities")
        if key not in beam_quantities:
            beam_quantities[key] = []
        beam_quantities[key].append(QuantityDistribution.model_validate(value))

    # @field_serializer("physical_dose", "let", "alpha_dose", "sqrt_beta_dose")
    # def custom_dose_serializer(self, v: QuantityDistribution, info: SerializationInfo) -> Any:
    #     """Serialize the dose fields in various contexts."""
    #     context = info.context
    #     if context and context.get("matRad") == "mat-file":
    #         if v:
    #             numpy_array = np.transpose(sitk.GetArrayFromImage(v.cube), (1, 2, 0))

    #             wrapped_array = np.ndarray(shape=(1,), dtype=object)
    #             wrapped_array[0] = numpy_array
    #             return wrapped_array
    #         return None
    #     return v.model_dump(by_alias=info.by_alias) if v else None

    # @field_serializer("physical_dose_beam", "let_beam")
    # def custom_dose_beam_serializer(self, v: list[QuantityDistribution], info: SerializationInfo) -> Any:
    #     """Serialize the dose beam fields in various contexts."""
    #     context = info.context
    #     if context and context.get("matRad") == "mat-file":
    #         if v and len(v) > 0:
    #             # Create a flat dictionary to hold individual beam fields
    #             flat_beam_dict = {}
    #             camel_case_field_name = to_camel_case(
    #                 info.field_name
    #             )  # Convert field name to camelCase
    #             for idx, beam in enumerate(v, start=1):
    #                 numpy_array = np.transpose(sitk.GetArrayFromImage(beam.cube), (1, 2, 0))
    #                 wrapped_array = np.ndarray(shape=(1,), dtype=object)
    #                 wrapped_array[0] = numpy_array

    #                 # Add the serialized beam directly to the flat dictionary with a dynamic key
    #                 flat_beam_dict[f"{camel_case_field_name}{idx}"] = wrapped_array
    #             return flat_beam_dict
    #         return None
    #     # Default serializatio
    #     return (
    #         {
    #             f"{info.field_name}{idx}": dose.model_dump(by_alias=info.by_alias)
    #             for idx, dose in enumerate(v, start=1)
    #         }
    #         if v
    #         else None
    #     )

    def sequencing(self) -> None:
        """Perform sequencing on the dose cube."""
        # TODO
        pass

    def analyse_plan(self) -> None:
        """Perform analysis on the plan."""
        # TODO
        pass

    # def to_matrad(self, context: str = "mat-file") -> Any:
    #     export = super().to_matrad(context=context)
    #     if "physicalDoseBeam" in export and isinstance(export["physicalDoseBeam"], dict):
    #         physical_dose_beam_dict = export.pop("physicalDoseBeam")
    #         export.update(physical_dose_beam_dict)
    #     # Remove keys with None values from the export dictionary
    #     # (savemat cant handle "None")
    #     export = {key: value for key, value in export.items() if value is not None}
    #     return export  # ["resultGUI"]


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

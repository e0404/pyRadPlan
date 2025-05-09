"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import List, Any, Union, Dict
import numpy as np
from pydantic import (
    Field,
    model_validator,
    # field_serializer,
)
from pyRadPlan.result._quantity_distribution import QuantityDistribution
from pyRadPlan.core import PyRadPlanBaseModel


class BeamAccessor:
    """
    A class to access beam quantities in a PlanResult object.

    This class allows for dynamic access to beam quantities using dot notation.
    Dictionary methods are implemented to allow iteration and membership tests.
    """

    def __init__(self, beam_quantities: dict):
        self.beam_quantities = beam_quantities

    def __getattr__(self, attr):
        if attr in self.beam_quantities:
            return self.beam_quantities[attr]
        raise AttributeError(f"'BeamAccessor' object has no attribute '{attr}'")

    def __getitem__(self, key):
        return self.beam_quantities[key]

    def __iter__(self):
        return iter(self.beam_quantities)

    def __len__(self):
        return len(self.beam_quantities)

    def keys(self):
        return self.beam_quantities.keys()

    def items(self):
        return self.beam_quantities.items()

    def values(self):
        return self.beam_quantities.values()


class PlanResult(PyRadPlanBaseModel):
    """
    Stores and represents the result of the executed plan.

    Displayed are the calculated quantities, fluence and additional information.

    Attributes
    ----------
    fluence : np.ndarray
        The fluence array.
    info : dict[str, Any]
        Additional information passed by the solver used.
    used_solver : str
        The solver used for the calculation.
    quantities : Dict[str, QuantityDistribution]
        A dictionary of quantities, where the key is the quantity name and the value is a QuantityDistribution object.
        Quantities are called by their name, e.g. physical_dose, let, etc.
    beam_quantities : Dict[str, List[QuantityDistribution]]
        A dictionary of beam quantities, where the key is the quantity name and the value is a list of QuantityDistribution objects.
        Beam quantities are called by e.g. beam.physical_dose[i]

    Methods
    -------
    stored_quantities : List[str]
        Returns the list of stored quantities.
    stored_beam_quantities : List[str]
        Returns the list of stored beam quantities.
    """

    fluence: np.ndarray = Field(default=None)
    info: dict[str, Any] = Field(default_factory=dict)
    used_solver: str = Field(default="")

    quantities: Dict[str, QuantityDistribution] = Field(default_factory=dict)
    beam_quantities: Dict[str, List[QuantityDistribution]] = Field(default_factory=dict)

    class Config:
        extra = "allow"

    @property
    def stored_quantities(self) -> List[str]:
        """Return the list of stored quantities."""
        return list(self.quantities.keys()) if self.quantities else []

    @property
    def stored_beam_quantities(self) -> List[str]:
        """Return the list of stored beam quantities."""
        return list(self.beam_quantities.keys()) if self.beam_quantities else []

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
            data["quantities"] = quantities_tmp
            data["beam_quantities"] = beam_quantities_tmp

            return data

        return data

    # @model_validator(mode="after")
    # def set_dynamic_quantities(self, data: Any):
    #     # this workaround is needed because _quantities are filled AFTER the model is created.
    #     # Without __pydantic_extra__ there would be no data to set.
    #     # also by that, the logic in the before-validator is kept simple.
    #     extra = self.__pydantic_extra__ if hasattr(self, "__pydantic_extra__") else {}
    #     quantities = extra.get("_quantities", {})
    #     beam_quantities = extra.get("_beam_quantities", {})

    #     # these setters are needed to avoid infinite recursion by making the quantities an attribute.
    #     # Otherwise its not possible to call e.g. "physical_dose"
    #     object.__setattr__(self, "_quantities", quantities)
    #     object.__setattr__(self, "_beam_quantities", beam_quantities)

    #     # removing the extra attributes - if quantities should be visible,
    #     # remove these lines again.
    #     extra.pop("_quantities", None)
    #     extra.pop("_beam_quantities", None)
    #     object.__setattr__(self, "__pydantic_extra__", extra)

    #    return self

    def __dir__(self):
        # updated __dir__ to include dynamic attributes. These won't be highlighted,
        # but hovering in debug will give the correct information.
        # Highlighting will not be possible since attributes are dynamic.
        default = set(super().__dir__())
        # add dynamic attributes in _quantities and beam if available
        dynamic = set(self.quantities.keys())
        if self.beam_quantities:
            dynamic.add("beam")
        return list(default.union(dynamic))

    def __getattr__(self, item):
        # First try standard dynamic quantities.
        quantities = object.__getattribute__(self, "quantities")
        if item in quantities:
            return quantities[item]
        # If 'beam' is requested, return a BeamAccessor instance.
        if item == "beam":
            beam_quantities = object.__getattribute__(self, "beam_quantities")

            return BeamAccessor(beam_quantities)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, name, value):
        if name == "quantities":
            raise Warning("Direct modification of 'quantities' is not allowed. Input ignored.")
        if name == "beam_quantities":
            raise Warning(
                "Direct modification of 'beam_quantities' is not allowed. Input ignored."
            )
        super().__setattr__(name, value)

    def add_quantity(
        self, key: str = None, value: Union[dict, QuantityDistribution] = None, **kwargs
    ) -> None:
        """
        Add or update one or more quantity entries.

        This method can be used either by providing a key and value pair, e.g.:
            add_quantity("physical_dose", dose_array)
        or by specifying quantity names as keyword arguments, e.g.:
            add_quantity(physical_dose=dose_array)

        Parameters
        ----------
            key (str, optional): The name of the quantity. Defaults to None.
            value (Union[dict, QuantityDistribution], optional): The quantity data.
                Required if key is provided. Defaults to None.
            **kwargs: Additional quantity entries, where each key is the quantity name and each value is the quantity data.

        Raises
        ------
            ValueError: If neither a key/value pair nor keyword arguments are provided,
                or if both positional and keyword arguments are provided.
        """
        if key is not None and value is not None:
            if kwargs:
                raise ValueError("Cannot mix positional and keyword arguments in add_quantity.")
            self.quantities[key] = QuantityDistribution.model_validate(value)
        elif kwargs:
            for k, v in kwargs.items():
                self.quantities[k] = QuantityDistribution.model_validate(v)
        else:
            raise ValueError("No quantity provided to add_quantity.")

    def add_beam_quantity(
        self,
        key: str = None,
        value: Union[list[dict], list[QuantityDistribution]] = None,
        **kwargs,
    ) -> None:
        """
        Add or update one or more beam quantity entries.

        This method can be used either by providing a key/value pair, e.g.:
            add_beam_quantity("physical_dose", dose_array)
        or by specifying beam quantity names as keyword arguments, e.g.:
            add_beam_quantity(physical_dose=dose_array)

        Parameters
        ----------
            key (str, optional): The name of the beam quantity. Defaults to None.
            value (Union[list[dict], list[QuantityDistribution]], optional): The beam quantity data as list.
                Required if key is provided. Defaults to None.
            **kwargs: Additional beam quantity entries, where each key is the beam quantity name and
                each value is the beam quantity data (also list).

        Raises
        ------
            ValueError: If neither a key/value pair nor keyword arguments are provided, or if both positional
                and keyword arguments are provided.
        """

        if key is not None and value is not None:
            if kwargs:
                raise ValueError(
                    "Cannot mix positional and keyword arguments in add_beam_quantity."
                )
            if key not in self.beam_quantities:
                self.beam_quantities[key] = []
            self.beam_quantities[key] = [QuantityDistribution.model_validate(v) for v in value]
        elif kwargs:
            for k, v in kwargs.items():
                if k not in self.beam_quantities:
                    self.beam_quantities[k] = []
                self.beam_quantities[k] = [QuantityDistribution.model_validate(w) for w in v]
        else:
            raise ValueError("No beam quantity provided to add_beam_quantity.")

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
    Create a PlanResult object or create a new one.

    Parameters
    ----------
    data : Union[dict[str, Any], PlanResult, None]
        The quantities to validate or create a new PlanResult object.
        Either pass a dictionary with the quantities or a PlanResult object.

    fluence : Union[dict[str, Any], PlanResult, None]
        The fluence cube
    info : dict[str, Any], optional
        Additional information about the result. Passed on e.g. the optimier
    used_solver : str, optional
        The solver used for the calculation.
    **kwargs
        Additional keyword arguments to create the Result object.

    Returns
    -------
    Result
        A PlanResult object created from the input data or keyword arguments.

    Examples
    --------
    >>> result = create_result(custum_quantity_name=sample_image, custom_beam_quantity_name=[sample_image, sample_image])
    >>> result = create_result({"physical_dose": Union[dict, Array]}, info={"key": "value"}...)
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


def validate_result(data: Union[dict[str, Any], PlanResult, None] = None, **kwargs) -> PlanResult:
    """
    Validate a PlanResult object or create a new one.

    Parameters
    ----------
    data : Union[dict[str, Any], PlanResult, None]
        The quantities to validate or create a new PlanResult object.
        Either pass a dictionary with the quantities or a PlanResult object.

    fluence : Union[dict[str, Any], PlanResult, None]
        The fluence cube
    info : dict[str, Any], optional
        Additional information about the result. Passed on e.g. the optimier
    used_solver : str, optional
        The solver used for the calculation.
    **kwargs
        Additional keyword arguments to create the Result object.

    Returns
    -------
    Result
        A PlanResult object created from the input data or keyword arguments.

    Examples
    --------
    >>> result = validate_result(custum_quantity_name=sample_image, custom_beam_quantity_name=[sample_image, sample_image])
    >>> result = validate_result({"physical_dose": Union[dict, Array]}, info={"key": "value"}...)
    """

    return create_result(data, **kwargs)


# # TODO: might be placed somewhere else
# def to_camel_case(snake_str: str) -> str:
#     """Convert a snake_case string to camelCase."""
#     # necessary since iteration over every beam creates a new instance which will not be handled by pydantic.
#     # So custom camel_case function is needed here.
#     components = snake_str.split("_")
#     return components[0] + "".join(x.title() for x in components[1:])

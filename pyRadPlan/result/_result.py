"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import List, Any, Union, Dict
import numpy as np
import logging

from pydantic import (
    Field,
    model_validator,
    field_serializer,
    SerializationInfo,
    ValidationInfo,
)
from pydantic.alias_generators import to_camel, to_snake
import SimpleITK as sitk

from pyRadPlan.result._quantity_distribution import QuantityDistribution
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._sequence import Sequence

logger = logging.getLogger(__name__)


class BeamAccessor:
    """
    A class to access beam quantities in a PlanResult object.

    Dynamically injects type annotations for dynamic keys.
    """

    # physical_dose: list[QuantityDistribution]
    def __init__(self, beam_quantities: dict):
        self.beam_quantities = beam_quantities
        # Dynamically assign each key as an instance attribute and update annotations.
        # for key, value in beam_quantities.items():
        #     #setattr(self, key, value)
        #     self.__class__.__annotations__[key] = list[QuantityDistribution]

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

    # def __dir__(self):
    #     default = super().__dir__()
    #     return list(default) + list(self.beam_quantities.keys())


class PlanResult(PyRadPlanBaseModel):
    """
    Stores and represents the result of the executed plan.

    Displayed are the calculated quantities, fluence and additional information.

    Attributes
    ----------
    fluence : np.ndarray
        The fluence array (weights).
    info : dict[str, Any]
        The information given by the solver.
    used_solver : str
        The solver used for the calculation.
    MU : np.ndarray
        The monitor units array.
    quantities : Dict[str, QuantityDistribution]
        A dictionary of quantities, where the key is the quantity name and the value is a QuantityDistribution object.
        Quantities are called by their name, e.g. physical_dose, let, etc.
    beam_quantities : Dict[str, List[QuantityDistribution]]
        A dictionary of beam quantities, where the key is the quantity name and the value is a list of QuantityDistribution objects.
        Beam quantities are called by e.g. beam.physical_dose[i]
    sequence : Sequence
        The sequencing information.

    Methods
    -------
    stored_quantities : List[str]
        Returns the list of stored quantities.
    stored_beam_quantities : List[str]
        Returns the list of stored beam quantities.
    add_quantity(key: str, value: Union[dict, QuantityDistribution], **kwargs) -> None
        Adds or updates one quantity entry
    add_beam_quantity(key: str, value: Union[list[dict], list[QuantityDistribution]], **kwargs) -> None
        Adds or updates one or more beam quantity entries
    to_matrad(context: str = "mat-file") -> Any
        Converts the PlanResult object to a dictionary format for matRad.
    """

    quantities: Dict[str, QuantityDistribution] = Field(default_factory=dict)
    beam_quantities: Dict[str, List[QuantityDistribution]] = Field(default_factory=dict)

    fluence: np.ndarray = Field(default=None)
    mu: np.ndarray | None = Field(default=None, alias="MU")

    info: dict[str, Any] = Field(default_factory=dict)
    used_optimizer: Union[str, dict[str, any]] = Field(default="")

    sequence: Sequence = Field(default=None, alias="sequencing")

    @property
    def stored_quantities(self) -> List[str]:
        """Return the list of stored quantities."""
        return list(self.quantities.keys()) if self.quantities else []

    @property
    def stored_beam_quantities(self) -> List[str]:
        """Return the list of stored beam quantities."""
        return list(self.beam_quantities.keys()) if self.beam_quantities else []

    @property
    def beam(self):
        return BeamAccessor(self.beam_quantities)

    @classmethod
    def _process_sequencing_data(cls, data: Any, info: ValidationInfo) -> Any:
        pass

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad by grouping beam quantities."""
        # If the data is from matRad, we need to handle the beam quantities differently.
        # The keys are usually named with a "_beam" suffix.
        if "sequencing" in data:
            data["sequence"] = data.pop("sequencing")
        else:
            data["sequence"] = {}
        for key, value in list(data.items()):
            if key in ["w", "wUnsequenced"]:
                data["fluence"] = value
                data.pop(key)
            elif "_beam" in key:
                prefix, _, _ = key.partition("_beam")
                if to_snake(prefix) + "_beam" in data:
                    data[to_snake(prefix) + "_beam"].append(value)
                else:
                    data[to_snake(prefix) + "_beam"] = [value]
                # Remove the original key
                data.pop(key)
            elif key in ["w_sequenced", "wSequenced"]:
                data["sequence"]["fluence"] = value
                data.pop(key)
            elif key in ["apertureInfo", "aperture_info"]:
                data["sequence"]["aperture_info"] = data.pop(key)
            elif not to_snake(key) == key:
                data[to_snake(key)] = value
                data.pop(key)
        return data

    @model_validator(mode="before")
    @classmethod
    def aggregate_dynamic_quantities(cls, data: Any, info: ValidationInfo) -> Any:
        if data is None or data == {}:
            raise ValueError("No data provided for PlanResult creation.")
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)

        if isinstance(data, dict):
            # This here is necessary because of dynamic keys
            fields = cls.model_fields
            static_field_names = {n for n in fields if n not in {"quantities", "beam_quantities"}}
            static_aliases = {
                f.alias
                for f in fields.values()
                if f.alias and f.alias not in {"quantities", "beam_quantities"}
            }
            static_keys = static_field_names.union(static_aliases)

            # Place dynamic values into temporary public keys.
            quantities_tmp = {}
            beam_quantities_tmp = {}
            for key, value in list(data.items()):
                if key in static_keys:
                    continue
                if key == "quantities":
                    quantities_tmp.update(value)
                    data.pop(key)
                    continue
                elif key == "beam_quantities":
                    beam_quantities_tmp.update(value)
                    data.pop(key)
                    continue

                if not isinstance(value, list):
                    if isinstance(value, dict) and (
                        all(isinstance(value[v], QuantityDistribution) for v in value.keys())
                        or all(
                            isinstance(value[v], list)
                            and all(isinstance(w, QuantityDistribution) for w in value[v])
                            for v in value.keys()
                        )
                    ):
                        continue
                    # The Cubes have to be validated because validation does not work via private attributes.
                    quantities_tmp[key] = QuantityDistribution.model_validate(
                        value, context=info.context
                    )
                else:
                    new_key = key.replace("_beam", "") if "_beam" in key else key
                    beam_quantities_tmp[new_key] = [
                        QuantityDistribution.model_validate(v, context=info.context) for v in value
                    ]
                    # Remove the key
                data.pop(key)
            data["quantities"] = quantities_tmp
            data["beam_quantities"] = beam_quantities_tmp

            return data

        return data

    def __getattr__(self, item):
        quantities = object.__getattribute__(self, "quantities")
        if item in quantities:
            return quantities[item]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, name, value):
        if name == "quantities":
            raise Warning("Direct modification of 'quantities' is not allowed.")
        elif name == "beam_quantities":
            raise Warning("Direct modification of 'beam_quantities' is not allowed.")
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

    @field_serializer("quantities")
    def custom_quantities_serializer(self, v: Dict, info: SerializationInfo) -> Any:
        """Serialize the dose fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            if v:
                dict_tmp = {}
                for key, value in v.items():
                    numpy_array = np.transpose(
                        sitk.GetArrayFromImage(value.distribution), (1, 2, 0)
                    )
                    # TODO: multScen support
                    # wrapped_array = np.ndarray(shape=(1,), dtype=object)
                    # wrapped_array[0] = numpy_array
                    dict_tmp[to_camel(key)] = numpy_array  # wrapped_array
                return dict_tmp
            return None
        return v

    @field_serializer("beam_quantities")
    def custom_beam_quantities_serializer(self, v: Dict, info: SerializationInfo) -> Any:
        """Serialize the dose beam fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            if v:
                dict_tmp = {}
                for key, value in v.items():
                    for idx, beam in enumerate(value, start=1):
                        numpy_array = np.transpose(
                            sitk.GetArrayFromImage(beam.distribution), (1, 2, 0)
                        )
                        dict_tmp[f"{to_camel(key)}_beam{idx}"] = numpy_array
                return dict_tmp
            return None
        return v

    def analyse_plan(self) -> None:
        """Perform analysis on the plan."""
        # TODO
        pass

    def to_matrad(self, context: str = "mat-file") -> Any:
        """
        Convert the PlanResult object to a dictionary format for matRad.

        Parameters
        ----------
        context : str
            The context in which the object is being serialized. Default is "mat-file".

        Returns
        -------
        Any
            The serialized dictionary representation of the PlanResult object.
        """
        export = super().to_matrad(context=context)
        # removing Quantities and beamQuantities key from dict and passing content to top layer
        if "quantities" in export and isinstance(export["quantities"], dict):
            export.update(export.pop("quantities"))
        if "beamQuantities" in export and isinstance(export["beamQuantities"], dict):
            export.update(export.pop("beamQuantities"))
        # Remove keys with None values from the export dictionary
        # (savemat cant handle "None")
        export = {key: value for key, value in export.items() if value is not None}
        return export


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

        context = {"from_matRad": False}
        for key, value in data.items():
            if "beam1" in key:
                logger.info(
                    "matRad typical convention detected. Assuming matRad import. "
                    "If this is not the case, please rename variable '...beam1'"
                )
                context = {"from_matRad": True}

        return PlanResult.model_validate(data, context=context)
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

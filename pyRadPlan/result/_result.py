"""
Module for handling the result of a treatment plan computation.

Defines a result pydantic model and factories.
"""

from collections.abc import Mapping
from typing import List, Any, Optional, Union, Dict
import numpy as np
import logging

from pydantic import (
    Field,
    model_validator,
    field_serializer,
    field_validator,
    SerializationInfo,
    ValidationInfo,
)
from pydantic.alias_generators import to_camel, to_snake
import SimpleITK as sitk

from pyRadPlan.result._quantity_distribution import QuantityDistribution
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._sequence import Sequence

logger = logging.getLogger(__name__)


class BeamAccessor(Mapping):
    """
    A class to access beam (or scenario) quantities in a PlanResult object.

    Provides attribute and dict-like access to per-beam (per-scenario) quantity distributions.
    """

    def __init__(self, beam_quantities: dict):
        self.beam_quantities = beam_quantities

    def __getattr__(self, attr):
        # Only reached for names that are not real attributes, so the internal
        # ``beam_quantities`` lookup below cannot recurse.
        try:
            return self.__dict__["beam_quantities"][attr]
        except KeyError:
            raise AttributeError(f"'BeamAccessor' object has no attribute '{attr}'") from None

    def __getitem__(self, key):
        return self.beam_quantities[key]

    def __iter__(self):
        return iter(self.beam_quantities)

    def __len__(self):
        return len(self.beam_quantities)


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
    used_optimizer : Union[str, dict[str, Any]]
        The optimizer used for the calculation.
    mu : np.ndarray
        The monitor units array. Reserved: not computed yet.
    quantities : Dict[str, QuantityDistribution]
        A dictionary of quantities, where the key is the quantity name and the value is a QuantityDistribution object.
        Quantities are called by their name, e.g. physical_dose, let, etc.
    beam_quantities : Dict[str, List[QuantityDistribution]]
        A dictionary of beam quantities, where the key is the quantity name and the value is a list of QuantityDistribution objects.
        Beam quantities are called by e.g. beam.physical_dose[i]
    scenario_quantities : Dict[str, List[Optional[QuantityDistribution]]]
        Per-scenario quantities, one entry per scenario slot of the Dij (``None`` for scenarios
        not computed). Called by e.g. ``scenario.physical_dose[i]``
    scenario_labels : tuple[str, ...], optional
        The labels of the scenarios of ``scenario_quantities``, by flat scenario index.
    sequence : Sequence
        The sequencing information.

    Methods
    -------
    stored_quantities : List[str]
        Returns the list of stored quantities.
    stored_beam_quantities : List[str]
        Returns the list of stored beam quantities.
    stored_scenario_quantities : List[str]
        Returns the list of stored per-scenario quantities.
    add_quantity(key: str, value: Union[dict, QuantityDistribution], **kwargs) -> None
        Adds or updates one quantity entry
    add_beam_quantity(key: str, value: Union[list[dict], list[QuantityDistribution]], **kwargs) -> None
        Adds or updates one or more beam quantity entries
    to_matrad(context: str = "mat-file") -> Any
        Converts the PlanResult object to a dictionary format for matRad.
    """

    quantities: Dict[str, QuantityDistribution] = Field(default_factory=dict)
    beam_quantities: Dict[str, List[QuantityDistribution]] = Field(default_factory=dict)
    scenario_quantities: Dict[str, List[Optional[QuantityDistribution]]] = Field(
        default_factory=dict
    )
    scenario_labels: Optional[tuple[str, ...]] = None

    # "w" is matRad's name for the beamlet weights.
    fluence: Optional[np.ndarray] = Field(default=None, alias="w")
    mu: Optional[np.ndarray] = Field(default=None, alias="MU")

    info: dict[str, Any] = Field(default_factory=dict)
    used_optimizer: Union[str, dict[str, Any]] = Field(default="")

    sequence: Optional[Sequence] = Field(default=None, alias="sequencing")

    @field_validator("info", "used_optimizer", mode="before")
    @classmethod
    def _coerce_empty(cls, v: Any, info: ValidationInfo) -> Any:
        """Coerce empty MATLAB values back to this field's default.

        A ``.mat`` round-trip turns an empty dict or string into ``None`` or an
        empty array; accept those so a saved result re-validates.
        """
        if v is None or (hasattr(v, "__len__") and len(v) == 0):
            return {} if info.field_name == "info" else ""
        return v

    @property
    def stored_quantities(self) -> List[str]:
        """Return the list of stored quantities."""
        return list(self.quantities.keys()) if self.quantities else []

    @property
    def stored_beam_quantities(self) -> List[str]:
        """Return the list of stored beam quantities."""
        return list(self.beam_quantities.keys()) if self.beam_quantities else []

    @property
    def stored_scenario_quantities(self) -> List[str]:
        """Return the list of stored per-scenario quantities."""
        return list(self.scenario_quantities.keys()) if self.scenario_quantities else []

    @property
    def beam(self) -> BeamAccessor:
        """Per-beam quantities, accessible as ``result.beam.physical_dose[i]``."""
        # Not cached: ``model_copy(update=...)`` would carry over an accessor of the old dict.
        return BeamAccessor(self.beam_quantities)

    @property
    def scenario(self) -> BeamAccessor:
        """Per-scenario quantities, accessible as ``result.scenario.physical_dose[i]``."""
        return BeamAccessor(self.scenario_quantities)

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad by grouping beam quantities."""
        # If the data is from matRad, we need to handle the beam quantities differently.
        # The keys are usually named with a "_beam" suffix.
        # A new dictionary is built rather than mutating the input, because the
        # source is often a cached matRad mdict that the caller keeps using, and
        # because rewriting keys in place would reshuffle the quantity order.
        data = dict(data)
        sequence = dict(data.pop("sequencing", {}))
        out: dict[str, Any] = {}

        # Per-beam keys are collected with their beam number so that the beam order
        # follows the number in the key rather than the key order of the source file.
        beam_groups: dict[str, list[tuple[int, Any]]] = {}

        for key, value in data.items():
            if key in ["w", "wUnsequenced"]:
                out["fluence"] = value
            elif "_beam" in key:
                prefix, _, number = key.partition("_beam")
                group = beam_groups.setdefault(to_snake(prefix) + "_beam", [])
                group.append((int(number) if number.isdigit() else len(group), value))
            elif key in ["w_sequenced", "wSequenced"]:
                sequence["fluence"] = value
            elif key in ["apertureInfo", "aperture_info"]:
                sequence["aperture_info"] = value
            else:
                out[to_snake(key)] = value

        for key, group in beam_groups.items():
            out[key] = [value for _, value in sorted(group, key=lambda entry: entry[0])]

        out["sequence"] = sequence
        return out

    @model_validator(mode="before")
    @classmethod
    def aggregate_dynamic_quantities(cls, data: Any, info: ValidationInfo) -> Any:
        if data is None or data == {}:
            raise ValueError("No data provided for PlanResult creation.")
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)

        if not isinstance(data, dict):
            return data

        # Copy so that the caller's dictionary survives validation unchanged.
        data = dict(data)

        # This here is necessary because of dynamic keys
        fields = cls.model_fields
        containers = {"quantities", "beam_quantities", "scenario_quantities"}
        static_field_names = {n for n in fields if n not in containers}
        static_aliases = {
            f.alias for f in fields.values() if f.alias and f.alias not in containers
        }
        static_keys = static_field_names.union(static_aliases)

        # Place dynamic values into temporary public keys.
        containers_tmp: dict[str, dict] = {name: {} for name in containers}
        quantities_tmp = containers_tmp["quantities"]
        beam_quantities_tmp = containers_tmp["beam_quantities"]
        scenario_quantities_tmp = containers_tmp["scenario_quantities"]
        for key, value in list(data.items()):
            if key in static_keys:
                continue
            if key in containers_tmp:
                containers_tmp[key].update(value)
            elif isinstance(value, list) and key.endswith("_scenario"):
                scenario_quantities_tmp[key.removesuffix("_scenario")] = [
                    None
                    if v is None
                    else QuantityDistribution.model_validate(v, context=info.context)
                    for v in value
                ]
            elif isinstance(value, list):
                beam_quantities_tmp[key.replace("_beam", "")] = [
                    QuantityDistribution.model_validate(v, context=info.context) for v in value
                ]
            else:
                # The Cubes have to be validated here because dynamic keys are not fields.
                quantities_tmp[key] = QuantityDistribution.model_validate(
                    value, context=info.context
                )
            data.pop(key)

        data.update(containers_tmp)

        return data

    def __getattr__(self, item):
        # Stored quantities are reachable as attributes (result.physical_dose).
        # Everything else falls back to pydantic, which resolves private
        # attributes and raises a proper AttributeError for unknown names.
        quantities = self.__dict__.get("quantities")
        if quantities is not None and item in quantities:
            return quantities[item]
        return super().__getattr__(item)

    def __setattr__(self, name, value):
        # Rebinding the containers would bypass validation of their entries.
        # Their contents remain mutable through add_quantity/add_beam_quantity.
        if name == "quantities":
            raise AttributeError(
                "Direct modification of 'quantities' is not allowed. Use add_quantity() instead."
            )
        elif name == "beam_quantities":
            raise AttributeError(
                "Direct modification of 'beam_quantities' is not allowed. Use add_beam_quantity() instead."
            )
        elif name == "scenario_quantities":
            raise AttributeError(
                "Direct modification of 'scenario_quantities' is not allowed. "
                "Use add_scenario_quantity() instead."
            )
        super().__setattr__(name, value)

    def __getitem__(self, key: str) -> QuantityDistribution:
        """Return a stored quantity by name."""
        return self.quantities[key]

    def __contains__(self, key: str) -> bool:
        """Whether a quantity of that name is stored."""
        return key in self.quantities

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

    def add_scenario_quantity(
        self, key: str, value: list[Union[dict, QuantityDistribution, None]]
    ) -> None:
        """
        Add or update a per-scenario quantity.

        Parameters
        ----------
        key : str
            The name of the quantity.
        value : list
            One entry per scenario slot; ``None`` for scenarios without a result.
        """
        self.scenario_quantities[key] = [
            None if v is None else QuantityDistribution.model_validate(v) for v in value
        ]

    @field_serializer("quantities")
    def custom_quantities_serializer(self, v: Dict, info: SerializationInfo) -> Any:
        """Serialize the dose fields in various contexts."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            if v:
                dict_tmp = {}
                for key, value in v.items():
                    numpy_array = np.transpose(sitk.GetArrayFromImage(value.values), (1, 2, 0))
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
                        numpy_array = np.transpose(sitk.GetArrayFromImage(beam.values), (1, 2, 0))
                        dict_tmp[f"{to_camel(key)}_beam{idx}"] = numpy_array
                return dict_tmp
            return None
        return v

    @field_serializer("scenario_quantities", "scenario_labels")
    def custom_scenario_serializer(self, v: Any, info: SerializationInfo) -> Any:
        """Leave out the per-scenario results for matRad, whose layout differs."""
        context = info.context
        if context and context.get("matRad") == "mat-file":
            return None
        return v

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
        # (savemat can't handle "None")
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
        Additional information about the result. Passed on e.g. the optimizer
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
    if isinstance(data, PlanResult):
        return data

    # Reject non-dict inputs (bare arrays/images) explicitly. Checked before the
    # truthiness test below because ``if <ndarray>:`` raises an ambiguous-truth error.
    if data is not None and not isinstance(data, dict):
        raise TypeError(f"Expected dict or PlanResult, got {type(data).__name__}")

    if data:
        # Merge additional key-value pairs (like info) into the dictionary.
        data.update(kwargs)

        context = {"from_matRad": False}
        for key in data:
            if "beam1" in key:
                logger.info(
                    "matRad typical convention detected. Assuming matRad import. "
                    "If this is not the case, please rename variable '...beam1'"
                )
                context = {"from_matRad": True}
                break

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
        Additional information about the result. Passed on e.g. the optimizer
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

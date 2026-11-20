"""
Module for handling the result of a treatment plan computation.

Defines a result pydantic model and factories.
"""

from collections.abc import Mapping
from typing import List, Any, Optional, Union, Dict
from typing_extensions import Self
import numpy as np
import logging
import re

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
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.result._sequence import Sequence

logger = logging.getLogger(__name__)

_CONTAINERS = ("quantities", "beam_quantities", "scenario_quantities")

#: matRad result fields whose name does not follow from the camelCase conversion.
_MATRAD_QUANTITY_NAMES = {
    "RBExDose": "rbe_x_dose",
    "RBExD": "rbe_x_dose",
    "LET": "let",
    "BED": "bed",
    "RBE": "rbe",
}
_QUANTITY_NAMES_TO_MATRAD = {
    "rbe_x_dose": "RBExDose",
    "let": "LET",
    "bed": "BED",
    "rbe": "RBE",
}
#: Fields only matRad writes; their presence marks a dictionary as a matRad result.
_MATRAD_ONLY_KEYS = frozenset(
    {"physicalDose", "usedOptimizer", "wUnsequenced", "wSequenced", "apertureInfo"}
    | set(_MATRAD_QUANTITY_NAMES)
)
_MATRAD_BEAM_KEY = re.compile(r".+_beam\d+$")
#: Tolerance (in mm, and for direction cosines) up to which two grids count as identical.
_GEOMETRY_TOLERANCE = 1e-4


def _from_matrad_name(name: str) -> str:
    return _MATRAD_QUANTITY_NAMES.get(name) or to_snake(name)


def _to_matrad_name(name: str) -> str:
    return _QUANTITY_NAMES_TO_MATRAD.get(name) or to_camel(name)


def _is_matrad_cube(value: Any) -> bool:
    """Whether a matRad result field holds a cube (or a cell array of cubes)."""
    return isinstance(value, np.ndarray) and (value.dtype == object or value.ndim in (3, 4))


def _without_none(value: Any) -> Any:
    """Drop the ``None`` entries of (nested) dictionaries."""
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


def looks_like_matrad(data: Mapping) -> bool:
    """Whether a result dictionary uses matRad's field names."""
    return any(
        isinstance(key, str) and (key in _MATRAD_ONLY_KEYS or _MATRAD_BEAM_KEY.match(key))
        for key in data
    )


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

    @field_validator("fluence", "mu", mode="before")
    @classmethod
    def _to_numpy_vector(cls, v: Any) -> Any:
        """Accept weights from any array backend (and plain sequences) as numpy."""
        if v is None or isinstance(v, np.ndarray):
            return v
        if isinstance(v, (list, tuple)):
            return np.asarray(v, dtype=float)
        return to_numpy(v)

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
    def _static_keys(cls) -> set[str]:
        """Names and aliases of the fields that are no quantity containers."""
        fields = cls.model_fields
        names = {n for n in fields if n not in _CONTAINERS}
        aliases = {f.alias for f in fields.values() if f.alias and f.alias not in _CONTAINERS}
        return names | aliases

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
        static_keys = cls._static_keys()
        skipped: list[str] = []

        for key, value in data.items():
            if key in ["w", "wUnsequenced"]:
                out["fluence"] = value
            elif key in ["w_sequenced", "wSequenced"]:
                sequence["fluence"] = value
            elif key in ["apertureInfo", "aperture_info"]:
                sequence["aperture_info"] = value
            elif "_beam" in key:
                prefix, _, number = key.partition("_beam")
                if not _is_matrad_cube(value):
                    skipped.append(key)
                    continue
                group = beam_groups.setdefault(_from_matrad_name(prefix) + "_beam", [])
                group.append((int(number) if number.isdigit() else len(group), value))
            elif key in static_keys or _from_matrad_name(key) in static_keys:
                out[key if key in static_keys else _from_matrad_name(key)] = value
            elif _is_matrad_cube(value):
                out[_from_matrad_name(key)] = value
            else:
                # resultGUI also carries derived data (e.g. quality indicators or DVHs)
                # that is no distribution; it must not invalidate the cubes.
                skipped.append(key)

        if skipped:
            logger.warning(
                "Ignoring matRad result fields that are no quantity distributions: %s",
                ", ".join(skipped),
            )

        for key, group in beam_groups.items():
            out[key] = [value for _, value in sorted(group, key=lambda entry: entry[0])]

        if sequence:
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
        static_keys = cls._static_keys()

        # Place dynamic values into temporary public keys.
        containers_tmp: dict[str, dict] = {name: {} for name in _CONTAINERS}
        quantities_tmp = containers_tmp["quantities"]
        beam_quantities_tmp = containers_tmp["beam_quantities"]
        scenario_quantities_tmp = containers_tmp["scenario_quantities"]
        # Bare arrays carry no geometry and adopt the grid of the other quantities.
        bare: list[QuantityDistribution] = []
        placed: list[QuantityDistribution] = []

        def validate(v: Any) -> QuantityDistribution:
            distribution = QuantityDistribution.model_validate(v, context=info.context)
            (bare if isinstance(v, np.ndarray) else placed).append(distribution)
            return distribution

        for key, value in list(data.items()):
            if key in static_keys:
                continue
            if key in containers_tmp:
                containers_tmp[key].update(value)
            elif isinstance(value, list) and key.endswith("_scenario"):
                scenario_quantities_tmp[key.removesuffix("_scenario")] = [
                    None if v is None else validate(v) for v in value
                ]
            elif isinstance(value, list):
                beam_quantities_tmp[key.replace("_beam", "")] = [validate(v) for v in value]
            else:
                # The Cubes have to be validated here because dynamic keys are not fields.
                try:
                    quantities_tmp[key] = validate(value)
                except ValueError as exc:
                    raise ValueError(
                        f"'{key}' is neither a field of PlanResult nor a valid quantity "
                        f"distribution: {exc}"
                    ) from exc
            data.pop(key)

        if placed:
            for distribution in bare:
                cls._adopt_grid(distribution, placed[0])

        data.update(containers_tmp)

        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        """
        Check that the stored quantities can belong to one and the same result.

        Only the structure is checked (names, grid, number of beams and scenarios).
        Where a quantity comes from is not: an imported distribution need not know the
        model or the plan it was computed with.
        """
        reserved = self._static_keys() | set(_CONTAINERS)
        distributions: list[tuple[str, QuantityDistribution]] = []
        for container in _CONTAINERS:
            for name, value in getattr(self, container).items():
                if name in reserved or hasattr(type(self), name):
                    raise ValueError(
                        f"'{name}' cannot be used as a quantity name, it is an attribute "
                        "of PlanResult."
                    )
                entries = value if isinstance(value, list) else [value]
                distributions += [(name, entry) for entry in entries if entry is not None]

        self._check_common_grid(distributions)

        beam_counts = {name: len(v) for name, v in self.beam_quantities.items()}
        if len(set(beam_counts.values())) > 1:
            raise ValueError(f"Per-beam quantities differ in their number of beams: {beam_counts}")

        scenario_counts = {name: len(v) for name, v in self.scenario_quantities.items()}
        if len(set(scenario_counts.values())) > 1:
            raise ValueError(
                f"Per-scenario quantities differ in their number of scenarios: {scenario_counts}"
            )
        if self.scenario_labels is not None and scenario_counts:
            num_scenarios = next(iter(scenario_counts.values()))
            if len(self.scenario_labels) != num_scenarios:
                raise ValueError(
                    f"{len(self.scenario_labels)} scenario labels were given for "
                    f"{num_scenarios} scenarios."
                )
        return self

    @staticmethod
    def _check_common_grid(distributions: list[tuple[str, QuantityDistribution]]) -> None:
        if not distributions:
            return

        def geometry(image: sitk.Image) -> np.ndarray:
            return np.concatenate([image.GetSpacing(), image.GetOrigin(), image.GetDirection()])

        ref_name, ref = distributions[0]
        ref_size, ref_geometry = ref.values.GetSize(), geometry(ref.values)
        for name, distribution in distributions[1:]:
            image = distribution.values
            if image is ref.values:
                continue
            if image.GetSize() != ref_size or not np.allclose(
                geometry(image), ref_geometry, rtol=0.0, atol=_GEOMETRY_TOLERANCE
            ):
                raise ValueError(
                    f"All quantities of a result must share one grid, but '{name}' "
                    f"(size {image.GetSize()}, spacing {image.GetSpacing()}, origin "
                    f"{image.GetOrigin()}) differs from '{ref_name}' (size {ref_size}, "
                    f"spacing {ref.values.GetSpacing()}, origin {ref.values.GetOrigin()})."
                )

    @staticmethod
    def _adopt_grid(distribution: QuantityDistribution, reference: QuantityDistribution) -> None:
        """Place *distribution* on the grid of *reference* if their sizes agree."""
        if distribution.values.GetSize() == reference.values.GetSize():
            distribution.values.CopyInformation(reference.values)

    def _validate_entry(self, value: Any) -> QuantityDistribution:
        """Validate a quantity to add; a bare array adopts the grid of this result."""
        distribution = QuantityDistribution.model_validate(value)
        if isinstance(value, np.ndarray):
            for container in _CONTAINERS:
                for stored in getattr(self, container).values():
                    for entry in stored if isinstance(stored, list) else [stored]:
                        if entry is not None:
                            self._adopt_grid(distribution, entry)
                            return distribution
        return distribution

    def _store_checked(self, container: str, key: str, value: Any) -> None:
        """Store *value* under *key*, rolling back if the result becomes inconsistent."""
        entries = getattr(self, container)
        missing = object()
        previous = entries.get(key, missing)
        entries[key] = value
        try:
            self._check_consistency()
        except ValueError:
            if previous is missing:
                del entries[key]
            else:
                entries[key] = previous
            raise

    @classmethod
    def from_matrad(
        cls, data: Union[Mapping, "PlanResult"], reference_image: Optional[sitk.Image] = None
    ) -> Self:
        """
        Create a PlanResult from a matRad ``resultGUI`` structure.

        Parameters
        ----------
        data : Mapping
            The ``resultGUI`` structure as loaded from a ``.mat`` file.
        reference_image : sitk.Image, optional
            Image of the grid the result lives on, usually the CT cube. matRad stores
            its cubes without geometry, so without it the quantities end up with unit
            spacing at the origin. Cubes of another size are left untouched.

        Returns
        -------
        PlanResult
            The validated result.
        """
        if isinstance(data, cls):
            return data
        result = cls.model_validate(data, context={"from_matRad": True})
        if reference_image is not None and reference_image.GetDimension() == 3:
            for container in _CONTAINERS:
                for value in getattr(result, container).values():
                    for entry in value if isinstance(value, list) else [value]:
                        if entry is not None and (
                            entry.values.GetSize() == reference_image.GetSize()
                        ):
                            entry.values.CopyInformation(reference_image)
        return result

    def model_copy(self, *, update: Optional[Mapping[str, Any]] = None, deep: bool = False):
        """Copy the result; a shallow copy shares the images but not the containers.

        Adding or replacing a quantity of the copy therefore leaves the original as it is.
        """
        copy = super().model_copy(update=update, deep=deep)
        if deep:
            return copy
        for name in (*_CONTAINERS, "info"):
            if update is not None and name in update:
                continue
            entries = {
                key: list(value) if isinstance(value, list) else value
                for key, value in getattr(self, name).items()
            }
            copy.__dict__[name] = entries
        return copy

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
            self._store_checked("quantities", key, self._validate_entry(value))
        elif kwargs:
            for k, v in kwargs.items():
                self._store_checked("quantities", k, self._validate_entry(v))
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
            self._store_checked("beam_quantities", key, [self._validate_entry(v) for v in value])
        elif kwargs:
            for k, v in kwargs.items():
                self._store_checked("beam_quantities", k, [self._validate_entry(w) for w in v])
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
        self._store_checked(
            "scenario_quantities",
            key,
            [None if v is None else self._validate_entry(v) for v in value],
        )

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
                    dict_tmp[_to_matrad_name(key)] = numpy_array  # wrapped_array
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
                        dict_tmp[f"{_to_matrad_name(key)}_beam{idx}"] = numpy_array
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
        if self.sequence is None or not self.sequence.model_fields_set:
            export.pop("sequencing", None)
        # removing Quantities and beamQuantities key from dict and passing content to top layer
        if "quantities" in export and isinstance(export["quantities"], dict):
            export.update(export.pop("quantities"))
        if "beamQuantities" in export and isinstance(export["beamQuantities"], dict):
            export.update(export.pop("beamQuantities"))
        # savemat cannot handle None, neither at the top level nor in nested structs
        return _without_none(export)


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
        # Merge additional key-value pairs (like info) into a copy of the dictionary.
        data = {**data, **kwargs}

        # A dictionary read from a matRad file is recognized by its field names. Code
        # that knows its source should call ``PlanResult.from_matrad`` instead.
        from_matrad = looks_like_matrad(data)
        if from_matrad:
            logger.info("matRad field names detected. Assuming a matRad result.")

        return PlanResult.model_validate(data, context={"from_matRad": from_matrad})
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

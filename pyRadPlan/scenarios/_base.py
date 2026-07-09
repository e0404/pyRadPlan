"""
Abstract Interface for scenario models.

Classes
-------
- ScenarioModel: Abstract base class for scenario models.
"""

from abc import abstractmethod
from typing import Optional, Tuple, ClassVar, Any
from typing_extensions import Self  # python 3.9 & 3.10 compatibility

from pydantic import (
    Field,
    computed_field,
    model_validator,
    field_validator,
    AliasChoices,
    PrivateAttr,
    FieldSerializationInfo,
    field_serializer,
    SerializerFunctionWrapHandler,
    model_serializer,
    SerializationInfo,
)
from numpydantic import NDArray, Shape

import numpy as np
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.ct import validate_ct


class ScenarioModel(PyRadPlanBaseModel):
    """Base class representing a scenario model.

    Attributes
    ----------
    range_rel_sd : float
        The relative standard deviation for range.
    range_abs_sd : float
        The absolute standard deviation for range.
    shift_sd : Tuple[float]
        The standard deviation for shift in x, y, and z directions.
    wc_sigma : float
        The sigma value for the wc_factor.
    ct_scen_prob : list[Tuple[int,float]]
        The probability of each CT scenario.

    Methods
    -------
        update_scenarios() -> np.ndarray[float]
            Abstract method to update scenarios.
        extract_single_scenario(scen_num: int) -> ScenarioModel
            Extracts a single scenario based on its number.
        list_all_scenarios()
            Lists all scenarios.
        sub2scenIx(ctScen: int, shift_scen: int, range_shift_scen: int) -> int
            Converts CT scenario, shift scenario, and range shift scenario to scenario index.
        scenNum(fullScenIx: int) -> int
            Converts full scenario index to scenario number.
    """

    name: ClassVar[str]
    short_name: ClassVar[str]

    range_rel_sd: float = Field(
        default=3.5,
        description="The relative standard deviation for range.",
        ge=0,
        alias="rangeRelSD",
    )
    range_abs_sd: float = Field(
        default=1,
        description="The absolute standard deviation for range.",
        ge=0,
        alias="rangeAbsSD",
    )
    shift_sd: Tuple[float, float, float] = Field(
        default=(2.25, 2.25, 2.25),
        description="The standard deviation for shift in x, y, and z directions.",
        alias="shiftSD",
    )
    ct_scen_prob: list[Tuple[int, float]] = Field(
        default=[(0, 1.0)], description="The probability of each CT scenario."
    )
    wc_sigma: float = Field(
        1,
        description="Worst-case definition in multiples of standard deviation.",
        ge=0,
        validation_alias=AliasChoices("wcSigma", "wc_factor", "wcFactor"),
        serialization_alias="wcSigma",
    )

    # Other properties with protected access
    _num_of_available_ct_scen: int = PrivateAttr(default=1)
    _ct_scen_ix: np.ndarray[int] = PrivateAttr(default=np.array([0]))
    _iso_shift: np.ndarray[float] = PrivateAttr(default=np.array([[0.0, 0.0, 0.0]]))
    _rel_range_shift: np.ndarray[float] = PrivateAttr(default=0.0)
    _abs_range_shift: np.ndarray[float] = PrivateAttr(default=0.0)
    _tot_num_shift_scen: int = PrivateAttr(default=1)
    _tot_num_range_scen: int = PrivateAttr(default=1)
    _tot_num_scen: int = PrivateAttr(default=1)
    _scen_for_prob: np.ndarray[float] = PrivateAttr(
        default=np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    )
    _scen_prob: np.ndarray[float] = PrivateAttr(default=np.array([1.0]))
    _scen_weight: np.ndarray[float] = PrivateAttr(default=np.array([1.0]))
    _scen_mask: np.ndarray[bool] = PrivateAttr(default=np.ones((1, 1, 1), dtype=bool))
    _linear_mask: np.ndarray[int] = PrivateAttr(default=np.array([[0, 0, 0]]))

    # Constructor
    def __init__(self, ct: Optional[Any] = None, **kwargs):
        super(ScenarioModel, self).__init__(**kwargs)

        if ct is not None:
            ct = validate_ct(ct)
            self._num_of_available_ct_scen = ct.num_of_ct_scen
        else:
            self._num_of_available_ct_scen = 1

    @model_validator(mode="after")
    def _validate_model(self) -> Self:
        """
        Post-assignment validator.

        Mainly used to update Scenario Information.
        """
        self.update_scenarios()
        return self

    @field_validator("ct_scen_prob", mode="after")
    @classmethod
    def _validate_ct_scen_prob(
        cls, ct_scen_prob: list[Tuple[int, float]]
    ) -> list[Tuple[int, float]]:
        """
        Post-assignment validator for ct_scen_prob.

        Ensures sensible probabilities for ct scenarios.
        """
        if not all(0 <= x[1] <= 1 for x in ct_scen_prob):
            raise ValueError("All probabilities must be between 0 and 1.")
        if not all(0 <= x[0] for x in ct_scen_prob):
            raise ValueError("Scenario indices must be positive.")

        return ct_scen_prob

    @field_serializer("ct_scen_prob")
    def _serialize_ct_scen_prob(
        self, val: Any, info: FieldSerializationInfo
    ) -> list[Tuple[int, float]]:
        """Adapt for matRad if serialized for matRad."""

        if info.context is not None and "matRad" in info.context:
            return [(a + 1, b) for a, b in val]

        return val

    @computed_field
    @property
    def num_of_ct_scen(self) -> int:
        """Number of CT scenarios in the model."""
        return len(self.ct_scen_prob)

    @computed_field
    @property
    def num_of_available_ct_scen(self) -> int:
        """Number of totally available CT scenarios in the CT."""
        return self._num_of_available_ct_scen

    @computed_field
    @property
    def ct_scen_ix(self) -> NDArray[Shape["1-*"], int]:
        """CT scenario indices of the used CT scenarios."""
        return self._ct_scen_ix

    @computed_field
    @property
    def iso_shift(self) -> NDArray[Shape["1-*,3"], float]:
        """Isocenter shift values."""
        return self._iso_shift

    @computed_field
    @property
    def rel_range_shift(self) -> NDArray[Shape["1-*"], float]:
        """Relative range shift value."""
        return self._rel_range_shift

    @computed_field
    @property
    def abs_range_shift(self) -> NDArray[Shape["1-*"], float]:
        """Absolute range shift value."""
        return self._abs_range_shift

    @computed_field
    @property
    def max_abs_range_shift(self) -> float:
        """Maximum absolute range shift value."""
        return np.max(self.abs_range_shift)

    @computed_field
    @property
    def max_rel_range_shift(self) -> float:
        """Maximum relative range shift value."""
        return np.max(self.abs_range_shift)

    @computed_field
    @property
    def tot_num_shift_scen(self) -> int:
        """Total number of shift scenarios."""
        return self._tot_num_shift_scen

    @computed_field
    @property
    def tot_num_range_scen(self) -> int:
        """Total number of range shift scenarios."""
        return self._tot_num_range_scen

    @computed_field
    @property
    def tot_num_scen(self) -> int:
        """Total number of scenarios."""
        return self._tot_num_scen

    @computed_field
    @property
    def scen_for_prob(self) -> NDArray[Shape["1-*,1-*"], float]:
        """Scenarios organized for probability calculation."""
        return self._scen_for_prob

    @computed_field
    @property
    def scen_prob(self) -> NDArray[Shape["1-*,1-*"], float]:
        """Scenarios probability matrix."""
        return self._scen_prob

    @computed_field
    @property
    def scen_weight(self) -> NDArray[Shape["1-*,1-*"], float]:
        """Scenarios weight matrix."""
        return self._scen_weight

    @computed_field
    @property
    def scen_mask(self) -> NDArray[Shape["1-*,1-*"], bool]:
        """Scenarios mask matrix describing how scenarios should be stored."""
        return self._scen_mask

    @computed_field
    @property
    def linear_mask(self) -> NDArray[Shape["1-*,1-*"], int]:
        """Linear mask matrix for scenario selection."""
        return self._linear_mask

    @field_serializer("linear_mask")
    def _serialize_linear_mask(
        self, val: Any, info: FieldSerializationInfo
    ) -> NDArray[Shape["1-*,1-*"], int]:
        """Serialize the linear mask according to context."""

        if info.context is not None and "matRad" in info.context:
            return val + 1

        return val

    # Abstract methods
    @abstractmethod
    def update_scenarios(self) -> np.ndarray[float]:
        """Update scenario data from uncertainty model settings."""

    @abstractmethod
    def extract_single_scenario(self, scen_num: int) -> "ScenarioModel":
        """Extract a single scenario based on its number."""

    def list_all_scenarios(self):
        """
        Print a list of all scenarios.

        This method prints the details of all scenarios in a formatted table.
        The table includes the following columns:
        - xShift: The x-axis shift value
        - yShift: The y-axis shift value
        - zShift: The z-axis shift value
        - absRng: The absolute range value
        - relRng: The relative range value
        - prob: The probability value

        Example output:
        #   xShift   yShift   zShift   absRng   relRng   prob.
        1   0.000    0.000    0.000    0.000    0.000    0.000
        2   0.100    0.200    0.300    0.400    0.500    0.600
        3   0.700    0.800    0.900    1.000    1.100    1.200
        """

        print("Listing all scenarios...")
        print("#\tctScen\txShift\tyShift\tzShift\tabsRng\trelRng\tprob.")
        for s, row in enumerate(self.scen_for_prob):
            row_str = "\t".join([f"{int(row[0]):5d}"] + [f"{x:.3f}" for x in row[1:]])
            print(f"{s + 1}\t{row_str}\t{self.scen_prob[s]:.3f}")

    def sub2scen_ix(self, ct_scen: int, shift_scen: int, range_shift_scen: int) -> int:
        """
        Convert a subscript index to a linear scenario index.

        Parameters
        ----------
            ctScen (int): The sub-scenario index.
            shift_scen (int): The shift scenario value.
            range_shift_scen (int): The range shift scenario value.

        Returns
        -------
            int: The corresponding scenario index.

        Raises
        ------
            None

        Examples
        --------
            # Example usage
            sub2scenIx(1, 2, 3)  # Returns: 1
        """

        if self._scen_mask.shape[0] == 1:
            return self._ct_scen_ix[ct_scen]
        linear_index = np.ravel_multi_index(
            (ct_scen, shift_scen, range_shift_scen), self._scen_mask.shape
        )
        return linear_index

    def scen_num(self, full_scen_ix: int) -> int:
        """
        Return the scenario number given the scenario ray index.

        This corresponds to the first occurrence of `full_scen_ix` in
        `self.scen_mask`.

        Parameters
        ----------
        full_scen_ix : int
            The value to search for in `self.scen_mask`.

        Returns
        -------
        int
            The index of the first occurrence of `full_scen_ix` in `self.scen_mask`.
        """

        scen_indices = np.where(self._scen_mask.flatten())[0]
        return int(np.where(scen_indices == full_scen_ix)[0][0])

    @model_serializer(mode="wrap")
    def _serialize(self, wrap: SerializerFunctionWrapHandler, info: SerializationInfo) -> dict:
        """
        Serialize the model to a dictionary.

        Parameters
        ----------
        wrap : SerializerFunctionWrapHandler
            The serialization function wrapper.

        Returns
        -------
        dict
            The serialized model.
        """

        output = wrap(self, info)
        output["model"] = self.short_name
        return output

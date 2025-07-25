from typing import Optional, Annotated, Any
import warnings
from datetime import datetime
from pydantic import (
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
from numpydantic import NDArray, Shape
import numpy as np

from pyRadPlan.core import PyRadPlanBaseModel


class Machine(PyRadPlanBaseModel):
    """
    Base class for Machine objects.

    Defines minimum meta-data a machine must hold.

    Attributes
    ----------
    radiation_mode : str
        The radiation mode of the machine.
    description : str
        The description of the machine.
    machine : str
        The name of the machine.
    """

    radiation_mode: str = Field()
    description: str = Field(default="")
    name: Annotated[str, StringConstraints(min_length=1)] = Field(alias="machine")

    created_on: Optional[datetime] = Field(default=None)
    last_modified: Optional[datetime] = Field(default=None)
    created_by: Optional[str] = Field(default="")
    last_modified_by: Optional[str] = Field(default="")
    data_type: Optional[str] = Field(default="-")
    version: Annotated[str, StringConstraints(pattern=r"^\d+\.\d+\.\d+$")] = Field(
        default="0.0.1", validate_default=True
    )

    @field_validator("created_on", "last_modified", mode="before")
    @classmethod
    def validate_datetime_variants(cls, v):
        # If it is a string, we try some additional formats in addition to
        # pydantics accepted datetime values
        # For example, matRad macines use the format "%d-%b-%Y" for some dates
        if isinstance(v, str):
            try_formats = ["%d-%b-%Y"]

            for fmt in try_formats:
                try:
                    return datetime.strptime(v, fmt)
                except ValueError:
                    pass

        return v


class ExternalBeamMachine(Machine):
    """
    Base class for Machine used for external irradation.

    Attributes
    ----------
    sad : float
        The source-to-axis distance of the machine
    energies : list[float]
        List of available photon energies
    """

    energies: NDArray[Shape["1-*"], np.float64]
    sad: float = Field(ge=0.0, description="Source-to-axis distance", alias="SAD")

    # Model sanitation / validation for compatibility
    @model_validator(mode="before")
    @classmethod
    def validate_machine_input(cls, data: Any) -> Any:
        """Validate the input data for the machine model before passing to pydantic."""
        if isinstance(data, dict):
            if "meta" in data and "data" in data:
                # looks like we loaded a matRad machine from a mat file

                # assign meta fields to base dict
                for key, value in data["meta"].items():
                    if key in data:
                        warnings.warn(
                            f"Meta data key {key} is already present in the data dictionary.",
                            UserWarning,
                        )

                    if key == "data":
                        warnings.warn("data key in metadata not allowed. Ignoring!")
                    data[key] = value

                # Clean up meta
                data.pop("meta")

                # prepare data structure for validation
                tabulated_energy_data = data.pop("data")
                cls._parse_tabulated_energy_data_from_mat(tabulated_energy_data, data)

        return data

    @classmethod
    def _parse_tabulated_energy_data_from_mat(
        cls, tabulated_energy_data: dict, returned_data: dict
    ):
        """Parse the tabulated energy data from a matRad machine file."""
        # prepare data structure for validation
        if isinstance(tabulated_energy_data, dict):
            # We might have a dictionary of lists
            if "energy" not in tabulated_energy_data:
                raise ValueError("Energy could not be found!")

            returned_data["energies"] = np.array(tabulated_energy_data["energy"], dtype=np.float64)

            if returned_data["energies"].size == 1:
                returned_data["energies"] = returned_data["energies"].flatten()

    def get_closest_energy_index(self, energy: float) -> int:
        """
        Given an energy value, return the closest matching index.

        Parameters
        ----------
        energy : float
            Energy value to search for

        Returns
        -------
        int
            Index of the energy level
        """
        return np.argmin(np.abs(self.energies - energy))

    def get_energy_index(self, energy: float, round_decimals=4) -> Optional[int]:
        """
        Given an energy value, return the index of the exact matching.

        Parameters
        ----------
        energy : float
            Energy value to search for
        round_decimals : int, optional
            Number of decimals to round the energy values to, by default 4

        Returns
        -------
        Optional[int]
            Index of the energy level
        """
        ix = np.where(np.round(self.energies, round_decimals) == np.round(energy, round_decimals))[
            0
        ]
        if len(ix) == 0:
            return None
        if len(ix) > 1:
            raise ValueError("Multiple matching energies found.")
        return ix[0]

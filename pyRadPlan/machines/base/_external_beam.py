from typing import Optional, Any
import warnings
from pydantic import (
    Field,
    model_validator,
)
from numpydantic import NDArray, Shape
import numpy as np

from ._base import Machine


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

                # Lets make sure that np.arrays are converted to lists
                # Might happen when using scipy.io.savemat for the machine file.
                for k, v in list(tabulated_energy_data.items()):
                    if isinstance(v, np.ndarray):
                        tabulated_energy_data[k] = v.tolist()
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

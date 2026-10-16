from typing import Optional, Any
import warnings
from pydantic import (
    Field,
    model_validator,
)
from numpydantic import NDArray, Shape
import numpy as np
from pyRadPlan.biology.tissues._base import TissueProperties

from ._base import Machine


class ExternalBeamMachine(Machine):
    """
    Base class for Machine used for external irradiation.

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

                data_meta = data["meta"]

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
                cls._parse_meta_data_from_mat(data_meta, data)

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

    @classmethod
    def _parse_meta_data_from_mat(cls, meta_data: dict, returned_data: dict):
        """
        Parse the biological metadata: tissue classes and kernel sets.

        ``tissueParameters`` holds one tissue class per element, built through the
        provenance registry by its ``model`` (and ``version``); ``ref_Kernel`` and the
        one-based ``column`` assign it to a kernel set. ``kernelSets`` declares the kernel
        sets, see :func:`~pyRadPlan.biology.models.parse_kernel_set_declarations`.
        """
        from pyRadPlan.biology.models._kernel_sets import (  # noqa: PLC0415 - avoids an import cycle
            parse_kernel_set_declarations,
            tissue_class_reference,
        )

        tissue_params = meta_data.get("tissueParameters")
        if tissue_params is not None:
            rows = _tissue_parameter_rows(tissue_params)
            returned_data["tissue_properties"] = [
                TissueProperties.from_base_data(row) for row in rows
            ]
            refs = [tissue_class_reference(row) for row in rows]
            if any(ref["kernel_set"] is not None or ref["column"] is not None for ref in refs):
                returned_data["tissue_class_refs"] = refs

        returned_data.pop("kernelSets", None)
        kernel_sets = meta_data.get("kernelSets")
        if kernel_sets is not None:
            returned_data["kernel_set_declarations"] = parse_kernel_set_declarations(kernel_sets)

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


def _tissue_parameter_rows(tissue_params: dict) -> list[dict]:
    """
    Split a matRad ``tissueParameters`` struct (one column per field) into one dict per class.

    ``read_mat`` squeezes a struct describing a single tissue class to scalar / string
    values, while several classes arrive as one list or array per field. Both are accepted.

    Raises
    ------
    ValueError
        The struct is not a mapping, is empty, or its fields describe differing class counts.
    """
    if not isinstance(tissue_params, dict) or not tissue_params:
        raise ValueError(
            "Machine meta data 'tissueParameters' must be a non-empty struct of per-class "
            f"fields, got {tissue_params!r}."
        )

    def column(values: Any) -> list:
        if isinstance(values, (str, bytes)):
            return [values]
        if isinstance(values, np.ndarray):
            return np.reshape(values, (-1,)).tolist()
        if isinstance(values, (list, tuple)):
            return list(values)
        return [values]

    columns = {key: column(values) for key, values in tissue_params.items()}
    counts = {key: len(values) for key, values in columns.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(
            "Machine meta data 'tissueParameters' fields describe differing numbers of tissue "
            f"classes: {counts}."
        )
    n_classes = next(iter(counts.values()))
    return [{key: values[i] for key, values in columns.items()} for i in range(n_classes)]

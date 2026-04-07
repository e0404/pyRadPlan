from typing import Optional, Any, Union, Annotated
from typing_extensions import Self

from numpy.typing import NDArray
from pydantic import (
    Field,
    ConfigDict,
    StringConstraints,
    model_validator,
    field_validator,
)
from abc import abstractmethod
from copy import deepcopy

import numpy as np

from pyRadPlan.core import PyRadPlanBaseModel


class BeamLimitingDevice(PyRadPlanBaseModel):
    """Base class for beam limiting device objects.

    Defines ...:

    Attributes
    ----------
    device_type : str
        The type of device (X, Y, ASYMX, ASYMY, MLCX, MLCY). DICOM (300A,00B8)
    device_orientation : str
        The orientation of the device (X, Y) in IEC coordinates.
    number_of_elements : int
        The number of leaf or jaw pairs (1 for standard jaws). DICOM (300A,00BC)
    source_to_device_distance : float
        The distance between source and the beam limiting device (in mm). DICOM (300A,00BA)
    device_angle:
        The orientation of IEC BEAM LIMITING DEVICE coordinate system with respect to IEC GANTRY coordinate system (degrees). (300A,0120)
    """

    device_type: str = Field(default="")
    device_orientation: Annotated[str, StringConstraints(pattern="^(X|Y)$")]
    number_of_elements: int = 0
    source_to_device_distance: Optional[float] = 0
    device_angle: Optional[float] = 0

    @field_validator("device_type", "device_orientation", mode="before")
    @classmethod
    def _convert_to_uppercase(cls, v: str) -> str:
        return v.upper()

    # TODO: Does field_width need to be added here already, to include leakage
    # If the field width is given, then this could be nicely separated by letting this
    # function do preprocessing and calling a private interface for the actual mask calculation,
    # which gets a meshgrid as an input and can be shared between MLC and Jaw.
    @abstractmethod
    def calculate_transmission_mask(
        self,
        spacing: Union[float, tuple[NDArray, NDArray]],
    ) -> NDArray:
        """Return a 2D transmission map (0 = blocked, 1 = open)."""
        raise NotImplementedError("This method should be overridden in derived classes")


class MLC(BeamLimitingDevice):
    """Class for MLC objects.

    Defines ...:

    Attributes
    ----------
    leaf_position_boundaries : Optional[np.ndarray]
        Boundaries in mm (in isocenter plane) in IEC coordinates
    leaf_positions : Optional[np.ndarray]
        Leaf positions in mm (in isocenter plane) in IEC coordinates
    leaf_width: float
        Width of the leaves (mm in isocenter plane)
    leaf_leakage: float
        Leakage of the leaves relative to open field (0-1))
    interleaf_leakage: float
        Leakage between the leaves relative to open field (0-1))
    """

    # TODO: Should not overwrite Config?
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    leaf_position_boundaries: Optional[np.ndarray] = None
    leaf_positions: Optional[np.ndarray] = None

    leaf_width: float = Field(
        default=None, description="Leaf width in mm"
    )  # should be from BeamLimitingDevicePosition
    leaf_leakage: float = Field(default=0.0)
    interleaf_leakage: float = Field(default=0.0)

    @field_validator("leaf_position_boundaries", "leaf_positions", mode="before")
    @classmethod
    def _convert_to_array(cls, v):
        if v is None or isinstance(v, np.ndarray):
            return v
        else:
            return np.array(v)

    @model_validator(mode="before")
    @classmethod
    def validate_device_type(cls, data: dict) -> dict:
        device_orientation = data.get("device_orientation") or data.get("deviceOrientation")
        device_orientation = device_orientation.upper() if device_orientation else None
        device_type = data.get("device_type") or data.get("deviceType")
        device_type = device_type.upper() if device_type else None

        if device_orientation is None:
            if device_type == "MLCX":
                data["device_orientation"] = "X"
            elif device_type == "MLCY":
                data["device_orientation"] = "Y"
            else:
                raise ValueError(
                    "MLC is missing device orientation."
                    "Set device_type to 'MLCX' or 'MLCY' or device_orientation to 'X' or 'Y'"
                )
        else:
            expected = f"MLC{device_orientation}"
            if not device_type or device_type == "MLC":
                data["device_type"] = expected
            elif device_type != expected:
                raise ValueError(
                    f"Device type ({device_type}) doesn't match device orientation ({device_orientation})"
                )
        return data

    @model_validator(mode="after")
    def validate_boundaries_and_positions(self) -> Self:
        has_boundaries = self.leaf_position_boundaries is not None
        has_positions = self.leaf_positions is not None

        if has_boundaries != has_positions:
            raise ValueError("Leaf postions and boundaries must both be set or both be empty")

        if has_boundaries and has_positions:
            if len(self.leaf_positions) != len(self.leaf_position_boundaries):
                raise ValueError(
                    f"Number of leaf positions ({len(self.leaf_positions)}) must match "
                    f"number of boundaries ({len(self.leaf_position_boundaries)})"
                )
            if not np.array_equal(
                self.leaf_position_boundaries, np.sort(self.leaf_position_boundaries)
            ):
                raise ValueError(
                    f"Leaf boundary positions need to be sorted: {self.leaf_position_boundaries}"
                )
            self.number_of_elements = len(self.leaf_positions)
            boundary_diffs = np.unique(np.abs(np.diff(self.leaf_position_boundaries)))
            if len(boundary_diffs) != 1:
                raise ValueError(
                    f"Leaf boundary positions contain inconsistant leaf widths: {boundary_diffs}"
                )
            if self.leaf_width is None:
                self.leaf_width = boundary_diffs[0]
            elif self.leaf_width != boundary_diffs[0]:
                raise ValueError(
                    f"Distance between leaf boundary positions ({boundary_diffs[0]})"
                    f" differs from given leaf_width ({self.leaf_width})"
                )
        return self

    def calculate_transmission_mask(
        self, spacing: Union[float, tuple[NDArray, NDArray]]
    ) -> NDArray:
        """Calculate transmission mask from leaf boundaries and positions."""

        if (self.leaf_position_boundaries is not None) and (self.leaf_positions is not None):
            boundaries_max = np.abs(self.leaf_position_boundaries).max()
            positions_max = np.abs(self.leaf_positions).max()
            half_size = max(boundaries_max, positions_max)
            if isinstance(spacing, float):
                n = int(np.ceil(2 * half_size / spacing))
                # enforce odd mask width, so 0 is centered
                if n % 2 == 0:
                    n += 1
                spacing_vector = np.linspace(-half_size, half_size, n)
                grid_x, grid_y = np.meshgrid(spacing_vector, spacing_vector, indexing="xy")
            elif isinstance(spacing, tuple) and spacing[0].ndim == 1 and spacing[1].ndim == 1:
                grid_x, grid_y = np.meshgrid(spacing[0], spacing[1], indexing="xy")
            elif isinstance(spacing, tuple) and spacing[0].ndim == 2 and spacing[1].ndim == 2:
                grid_x, grid_y = spacing[0], spacing[1]
            else:
                raise ValueError(
                    "Spacing must be either a float or a tuple of two 1D or two 2D arrays"
                )

            mask = np.full_like(grid_x, self.leaf_leakage, dtype=np.float32)

            for i in range(len(self.leaf_position_boundaries)):
                leaf_boundary_start = self.leaf_position_boundaries[i]
                leaf_boundary_end = leaf_boundary_start + self.leaf_width
                if i + 1 < len(self.leaf_position_boundaries):
                    assert np.isclose(leaf_boundary_end, self.leaf_position_boundaries[i + 1])
                leaf_opening_start = self.leaf_positions[i, 0]
                leaf_opening_end = self.leaf_positions[i, 1]

                if self.device_orientation == "X":
                    leaf_mask = (
                        (grid_y >= leaf_boundary_start)
                        & (grid_y < leaf_boundary_end)
                        & (grid_x >= leaf_opening_start)
                        & (grid_x < leaf_opening_end)
                    )
                elif self.device_orientation == "Y":
                    leaf_mask = (
                        (grid_x >= leaf_boundary_start)
                        & (grid_x < leaf_boundary_end)
                        & (grid_y >= leaf_opening_start)
                        & (grid_y < leaf_opening_end)
                    )
                else:
                    raise ValueError(
                        f"Device orientation {self.device_orientation} not recognized. Must be 'X' or 'Y'."
                    )
                mask[leaf_mask] = 1
            return mask


class Jaw(BeamLimitingDevice):
    """Class for Jaw objects.

    Defines ...:

    Attributes
    ----------
    positions : list[float]
        ?? (in isocenter plane)
    field_width: float
        Width of field. ??
    leakage: float
        Leakage of the jaws. ??
    """

    number_of_elements: int = 1

    positions: list[float]
    field_width: float

    leakage: float = Field(default=0.0)

    @field_validator("positions", mode="before")
    @classmethod
    def _validate_positions(cls, v):
        if len(v) != 2:
            raise ValueError("Jaw positions must have 2 entries")
        return v

    @model_validator(mode="after")
    def validate_field_width(self) -> "Jaw":
        if self.field_width / 2 < np.abs(self.positions).max():
            raise ValueError(
                f"Field width ({self.field_width}) must be twice as large "
                f"as the maximum position {np.abs(self.positions).max()}"
            )
        return self

    @model_validator(mode="before")
    @classmethod
    def validate_device_type(cls, data: dict) -> dict:
        device_orientation = data.get("device_orientation") or data.get("deviceOrientation")
        device_orientation = device_orientation.upper() if device_orientation else None
        device_type = data.get("device_type") or data.get("deviceType")
        device_type = device_type.upper() if device_type else None

        if device_orientation is None:
            if device_type == "JAWX":
                data["device_orientation"] = "X"
            elif device_type == "JAWY":
                data["device_orientation"] = "Y"
            else:
                raise ValueError(
                    "Jaw is missing device orientation."
                    "Set device_type to 'JAWX' or 'JAWY' or device_orientation to 'X' or 'Y'"
                )
        else:
            expected = f"Jaw{device_orientation}"
            if not device_type or device_type == "JAW":
                data["device_type"] = expected
            elif device_type != expected:
                raise ValueError(
                    f"Device type ({device_type}) doesn't match device orientation ({device_orientation})"
                )

        return data

    def calculate_transmission_mask(
        self, spacing: Union[float, tuple[NDArray, NDArray]]
    ) -> NDArray:
        if isinstance(spacing, float):
            n = int(np.ceil(self.field_width / spacing))
            # enforce odd mask width, so 0 is centered
            if n % 2 == 0:
                n += 1
            spacing_vector = np.linspace(-self.field_width / 2, self.field_width / 2, n)
            grid_x, grid_y = np.meshgrid(spacing_vector, spacing_vector, indexing="xy")
        elif isinstance(spacing, tuple) and spacing[0].ndim == 1 and spacing[1].ndim == 1:
            grid_x, grid_y = np.meshgrid(spacing[0], spacing[1], indexing="xy")
        elif isinstance(spacing, tuple) and spacing[0].ndim == 2 and spacing[1].ndim == 2:
            grid_x, grid_y = spacing[0], spacing[1]

        mask = np.full_like(grid_x, self.leakage, dtype=np.float32)

        if self.device_orientation == "X":
            jaw_mask = (grid_x >= self.positions[0]) & (grid_x < self.positions[1])
        elif self.device_orientation == "Y":
            jaw_mask = (grid_y >= self.positions[0]) & (grid_y < self.positions[1])
        else:
            raise ValueError(
                f"Device orientation '{self.device_orientation}' not recognized. "
                "Must be 'X' or 'Y'."
            )
        mask[jaw_mask] = 1

        return mask


def create_bld(
    data: Union[dict[str, Any], BeamLimitingDevice, None] = None, **kwargs
) -> BeamLimitingDevice:
    """
    Create a BeamLimitingDevice object (factory function).

    Parameters
    ----------
    data : Union[dict[str, Any], None]
        Dictionary containing the data to create the BeamLimitingDevice object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    BeamLimitingDevice
        A BeamLimitingDevice object.

    Raises
    ------
    ValueError

    """
    data = deepcopy(data)
    if data:
        if isinstance(data, BeamLimitingDevice):
            return data

        if isinstance(data, dict):
            device_type = data.get("device_type")

            # Since we also allow camelCase, try to get radiationMode if radiation_mode is not set
            if device_type is None:
                device_type = data.get("deviceType")

            if device_type.upper() in ["MLC", "MLCX", "MLCY"]:
                return MLC.model_validate(data)

            if device_type.upper() in ["JAW", "JAWX", "JAWY"]:
                return Jaw.model_validate(data)

    device_type = kwargs.get("device_type", "")
    if device_type.upper() in ["MLC", "MLCX", "MLCY"]:
        return MLC(**kwargs)
    if device_type.upper() in ["JAW", "JAWX", "JAWY"]:
        return Jaw(**kwargs)

    raise ValueError(f"Unkown device type: {device_type}")


def validate_bld(
    beam_limiting_device: Union[dict[str, Any], BeamLimitingDevice, None] = None, **kwargs
) -> BeamLimitingDevice:
    """
    Validate and create a BeamLimitingDevice object (factory function).

    Parameters
    ----------
    beam_limiting_device : Union[dict[str, Any], BeamLimitingDevice, None], optional
        Dictionary containing the data to create the BeamLimitingDevice object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    BeamLimitingDevice
        A BeamLimitingDevice object.

    Raises
    ------
    ValueError

    """
    return create_bld(beam_limiting_device, **kwargs)

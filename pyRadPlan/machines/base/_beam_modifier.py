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


def _smooth_step(coord: NDArray, edge: float, dx: float) -> NDArray:
    """Linear ramp centered on *edge* with transition width *dx*.

    Returns 0 for ``coord <= edge - dx/2``, 1 for ``coord >= edge + dx/2``,
    and a linear interpolation in between
    """
    return np.clip((coord - edge) / dx + 0.5, 0.0, 1.0)


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
            raise ValueError("Leaf positions and boundaries must both be set or both be empty")

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
                    f"Leaf boundary positions contain inconsistent leaf widths: {boundary_diffs}"
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
        """Calculate transmission mask with smooth edge weighting.

        Each leaf opening contributes a coverage fraction computed from the
        product of two smooth step functions per axis.  The transition width
        equals one grid-cell width so values at the edge receive 0.5
        transmission, with a linear ramp over +/- 1/2 cell on either side.
        """
        if (self.leaf_position_boundaries is None) or (self.leaf_positions is None):
            return None

        boundaries_max = np.abs(self.leaf_position_boundaries).max()
        positions_max = np.abs(self.leaf_positions).max()
        half_size = max(boundaries_max, positions_max)

        if isinstance(spacing, float):
            n = int(np.ceil(2 * half_size / spacing))
            if n % 2 == 0:
                n += 1
            spacing_vector = np.linspace(-half_size, half_size, n)
            grid_x, grid_y = np.meshgrid(spacing_vector, spacing_vector, indexing="xy")
            dx_x = spacing
            dx_y = spacing
        elif isinstance(spacing, tuple) and spacing[0].ndim == 1 and spacing[1].ndim == 1:
            grid_x, grid_y = np.meshgrid(spacing[0], spacing[1], indexing="xy")
            dx_x = float(np.diff(spacing[0]).min())
            dx_y = float(np.diff(spacing[1]).min())
        elif isinstance(spacing, tuple) and spacing[0].ndim == 2 and spacing[1].ndim == 2:
            grid_x, grid_y = spacing[0], spacing[1]
            dx_x = float(np.diff(grid_x[0, :]).min())
            dx_y = float(np.diff(grid_y[:, 0]).min())
        else:
            raise ValueError(
                "Spacing must be either a float or a tuple of two 1D or two 2D arrays"
            )

        mask = np.full_like(grid_x, self.leaf_leakage, dtype=np.float32)
        open_weight = 1.0 - self.leaf_leakage

        for i in range(len(self.leaf_position_boundaries)):
            b_start = self.leaf_position_boundaries[i]
            b_end = b_start + self.leaf_width
            o_start = self.leaf_positions[i, 0]
            o_end = self.leaf_positions[i, 1]

            if o_end <= o_start:
                # Fully closed leaf — skip entirely
                continue

            if self.device_orientation == "X":
                # boundary runs along Y axis, opening runs along X axis
                cov_boundary = _smooth_step(grid_y, b_start, dx_y) * (
                    1.0 - _smooth_step(grid_y, b_end, dx_y)
                )
                cov_opening = _smooth_step(grid_x, o_start, dx_x) * (
                    1.0 - _smooth_step(grid_x, o_end, dx_x)
                )
            elif self.device_orientation == "Y":
                # boundary runs along X axis, opening runs along Y axis
                cov_boundary = _smooth_step(grid_x, b_start, dx_x) * (
                    1.0 - _smooth_step(grid_x, b_end, dx_x)
                )
                cov_opening = _smooth_step(grid_y, o_start, dx_y) * (
                    1.0 - _smooth_step(grid_y, o_end, dx_y)
                )
            else:
                raise ValueError(
                    f"Device orientation {self.device_orientation} not recognized. Must be 'X' or 'Y'."
                )
            mask += open_weight * cov_boundary * cov_opening

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
        """Calculate transmission mask with smooth edge weighting.

        The jaw opening contributes a coverage fraction computed from smooth
        step functions at each jaw edge.  The transition width equals one
        grid-cell width, giving 0.5 transmission exactly at the edge and a
        linear ramp over +/- 1/2 cell on either side.
        """
        if isinstance(spacing, float):
            n = int(np.ceil(self.field_width / spacing))
            if n % 2 == 0:
                n += 1
            spacing_vector = np.linspace(-self.field_width / 2, self.field_width / 2, n)
            grid_x, grid_y = np.meshgrid(spacing_vector, spacing_vector, indexing="xy")
            dx_x = spacing
            dx_y = spacing
        elif isinstance(spacing, tuple) and spacing[0].ndim == 1 and spacing[1].ndim == 1:
            grid_x, grid_y = np.meshgrid(spacing[0], spacing[1], indexing="xy")
            dx_x = float(np.diff(spacing[0]).min())
            dx_y = float(np.diff(spacing[1]).min())
        elif isinstance(spacing, tuple) and spacing[0].ndim == 2 and spacing[1].ndim == 2:
            grid_x, grid_y = spacing[0], spacing[1]
            dx_x = float(np.diff(grid_x[0, :]).min())
            dx_y = float(np.diff(grid_y[:, 0]).min())
        else:
            raise ValueError(
                "Spacing must be either a float or a tuple of two 1D or two 2D arrays"
            )

        if self.device_orientation == "X":
            cov = _smooth_step(grid_x, self.positions[0], dx_x) * (
                1.0 - _smooth_step(grid_x, self.positions[1], dx_x)
            )
        elif self.device_orientation == "Y":
            cov = _smooth_step(grid_y, self.positions[0], dx_y) * (
                1.0 - _smooth_step(grid_y, self.positions[1], dx_y)
            )
        else:
            raise ValueError(
                f"Device orientation '{self.device_orientation}' not recognized. "
                "Must be 'X' or 'Y'."
            )

        mask = (self.leakage + (1.0 - self.leakage) * cov).astype(np.float32)
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

    raise ValueError(f"Unknown device type: {device_type}")


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

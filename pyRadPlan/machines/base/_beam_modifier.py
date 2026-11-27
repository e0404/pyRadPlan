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
import math

from pyRadPlan.core import PyRadPlanBaseModel


def _smooth_step(coord: NDArray, edge: float, dx: float) -> NDArray:
    """Linear ramp centered on *edge* with transition width *dx*.

    Returns 0 for ``coord <= edge - dx/2``, 1 for ``coord >= edge + dx/2``,
    and a linear interpolation in between
    """
    return np.clip((coord - edge) / dx + 0.5, 0.0, 1.0)


def _centered_grid(half_size: float, spacing: float) -> NDArray:
    """Odd-length grid with exactly *spacing* pitch, centered on 0 and covering ±*half_size*.

    Must match the ``resolution * arange(-n // 2, n // 2 + 1)`` grid that
    ``FieldShapeAsBLD`` assigns to the mask, so the pitch has to be exact.
    """
    half_n = int(np.ceil(half_size / spacing - 1e-9))
    return spacing * np.arange(-half_n, half_n + 1)


class BeamLimitingDevice(PyRadPlanBaseModel):
    """Base class for beam limiting device objects.

    A device that shapes the field of a beam (an MLC or a pair of jaws).

    Notes
    -----
    Positions (jaw positions, MLC leaf positions and leaf-row boundaries) are in mm in the
    isocenter plane, in the beam's eye view frame of the ray matrix of the steering
    information: X is ``ray_pos_bev`` x and Y is ``ray_pos_bev`` z, which at gantry and couch
    angle 0 are LPS +x and LPS +z.

    For a head-first supine patient this frame coincides with the IEC 61217 BEAM LIMITING
    DEVICE coordinate system of DICOM RT at collimator angle 0: its X is ``ray_pos_bev`` x
    (the patient's left) and its Y is ``ray_pos_bev`` z (towards the gantry) at gantry and
    couch angle 0. Its Z points to the source, along ``-ray_pos_bev`` y, but enters no
    position, so DICOM positions are used without a change of sign. ``device_angle`` is
    stored but not applied yet.

    Attributes
    ----------
    device_type : str
        The type of device: MLCX, MLCY, JAWX or JAWY (MLC or JAW with device_orientation).
        DICOM X and ASYMX correspond to JAWX, Y and ASYMY to JAWY. DICOM (300A,00B8)
    device_orientation : str
        The axis (X, Y) along which the device's leaves or jaws travel: X along
        ``ray_pos_bev`` x (an MLC's leaf rows stacked along ``ray_pos_bev`` z), Y
        along ``ray_pos_bev`` z (rows stacked along ``ray_pos_bev`` x).
    number_of_elements : int
        The number of leaf or jaw pairs (1 for standard jaws). DICOM (300A,00BC)
    source_to_device_distance : float, optional
        The distance between source and the beam limiting device (in mm). DICOM (300A,00BA)
        Not used yet: all positions are given in the isocenter plane.
    device_angle : float, optional
        The orientation of IEC BEAM LIMITING DEVICE coordinate system with respect to IEC GANTRY
        coordinate system (degrees). DICOM (300A,0120)
        Stored but not applied yet. Mapping segments to beamlets (ApertureInfo) refuses a non-zero
        angle.
    """

    device_type: str = Field(default="")
    device_orientation: Annotated[str, StringConstraints(pattern="^(X|Y)$")]
    number_of_elements: int = 0
    source_to_device_distance: Optional[float] = None
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

    A multi-leaf collimator with rows of leaf pairs stacked across the field. Each pair opens
    between a left and right leaf position along the travel axis. DICOM (300A,00B8) MLCX or MLCY.

    Attributes
    ----------
    leaf_position_boundaries : np.ndarray, optional
        Lower edges of the leaf rows in mm (in isocenter plane), ascending. DICOM stores the
        ``n + 1`` Leaf Position Boundaries (300A,00BE) of ``n`` rows, which a future RT Plan
        importer has to convert.
    leaf_positions : np.ndarray, optional
        Leaf positions in mm (in isocenter plane), ``(rows, 2)``.
    leaf_width : float, optional
        Width of the leaves (mm in isocenter plane). Derived from leaf_position_boundaries when
        not given. Required for an MLC with one single leaf row.
    leaf_leakage : float
        Leakage of the leaves relative to open field (0-1).
    interleaf_leakage : float
        Leakage between the leaves relative to open field (0-1). Not used yet.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    leaf_position_boundaries: Optional[np.ndarray] = None
    leaf_positions: Optional[np.ndarray] = None

    leaf_width: Optional[float] = Field(default=None, description="Leaf width in mm")
    # TODO: Move leaf_leakage to BeamLimitingDevice as `leakage` (Jaw.leakage means the same),
    # accepting leaf_leakage as an alias since it is released. interleaf_leakage stays MLC-only.
    leaf_leakage: float = Field(default=0.0)
    interleaf_leakage: float = Field(default=0.0)

    @property
    def num_of_active_leaf_pairs(self) -> int:
        """Number of leaf rows in the field, closed rows included.

        Follows matRad's numOfActiveLeafPairs, where every row in the field counts as active.
        """
        if self.leaf_positions is None:
            return 0
        return len(self.leaf_positions)

    @property
    def central_leaf_pair(self) -> float:
        """1-based index of the leaf pair on isocenter, derived from the leaf rows.

        A device that is symmetric about the isocenter has a leaf boundary there for an
        even number of leaf pairs (``N / 2 + 0.5``) and a leaf center for an odd number.
        Leaf rows that put a leaf center on the isocenter of an even device, as sequencing
        on the ray grid does, give matRad's ``ceil(N / 2)``.

        Pairs are counted towards negative stack coordinates, as in matRad (DICOM counts the
        other way). ``k + 0.5`` means the isocenter lies between pairs ``k`` and ``k + 1``.
        Needs ``leaf_position_boundaries`` and ``leaf_width``.
        """
        fraction = (self.leaf_position_boundaries[0] / self.leaf_width + 0.5) % 1.0
        on_center = min(fraction, 1.0 - fraction) < 1e-6  # noqa: PLR2004
        num = self.number_of_elements
        return float(math.ceil(num / 2)) if on_center else num / 2 + 0.5

    def is_active_leaf_pair(self, central_leaf_pair: Optional[float] = None) -> np.ndarray:
        """Per leaf pair of the full MLC: 1.0 if a leaf row of the field falls on it, else 0.0.

        Parameters
        ----------
        central_leaf_pair : float, optional
            1-based index of the leaf pair centred on isocenter, with indices
            increasing towards negative stack coordinates. Fractional (``k + 0.5``)
            when isocenter falls on the boundary between pairs ``k`` and ``k + 1``.
            Defaults to :attr:`central_leaf_pair`.
        """
        if central_leaf_pair is None:
            central_leaf_pair = self.central_leaf_pair
        idx = self._device_leaf_indices(central_leaf_pair)
        is_active_leaf_pair = np.zeros(self.number_of_elements)
        is_active_leaf_pair[idx[(idx >= 0) & (idx < self.number_of_elements)]] = 1
        return is_active_leaf_pair

    def _device_leaf_indices(self, central_leaf_pair: float) -> np.ndarray:
        """0-based index in the full device of each leaf row, in the order of the rows.

        The rows are stored by ascending stack coordinate, the device counts its leaf
        pairs towards negative stack coordinates, so the indices descend.
        """
        # Leaf pairs are indexed by their centre, while leaf_position_boundaries
        # are the rows' lower edges
        centers = self.leaf_position_boundaries + self.leaf_width / 2.0
        return np.rint(central_leaf_pair - 1 - centers / self.leaf_width).astype(int)

    def full_leaf_positions(self, central_leaf_pair: Optional[float] = None) -> np.ndarray:
        """Leaf positions across the full physical device.

        Returns an ``(number_of_elements, 2)`` array with this shape's leaf
        openings placed at their physical leaf index; leaves outside the
        field span (unused by this shape) are closed at the center (0, 0).

        Parameters
        ----------
        central_leaf_pair : float, optional
            Index of the central leaf pair in the full MLC. Defaults to
            :attr:`central_leaf_pair`.
        """
        if central_leaf_pair is None:
            central_leaf_pair = self.central_leaf_pair
        idx = self._device_leaf_indices(central_leaf_pair)
        is_active = self.is_active_leaf_pair(central_leaf_pair).astype(bool)
        if is_active.sum() != len(self.leaf_positions):
            raise ValueError(
                f"This shape's {len(self.leaf_positions)} leaf rows do not fit within "
                f"the full device's {self.number_of_elements} leaf pairs at "
                f"central_leaf_pair={central_leaf_pair}: only {int(is_active.sum())} "
                "leaf pairs of the device fall within this shape's leaf-row range. "
                "The field does not fit within the physical collimator at this alignment."
            )
        full = np.zeros((self.number_of_elements, 2))
        full[idx] = self.leaf_positions
        return full

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

    @staticmethod
    def validate_leaf_width_consistency(
        leaf_position_boundaries: np.ndarray, leaf_width: Optional[float]
    ) -> Optional[float]:
        """Return the common width of the leaf rows, derived from their spacing if not given.

        Raises ValueError if the rows are spaced unevenly, if the spacing differs from
        ``leaf_width``, or if there is a single row and no ``leaf_width``.
        """
        if len(leaf_position_boundaries) > 1:
            boundary_diffs = np.unique(np.abs(np.diff(leaf_position_boundaries)))
            if len(boundary_diffs) != 1:
                raise ValueError(
                    f"Leaf boundary positions contain inconsistent leaf widths: {boundary_diffs}"
                )
            if leaf_width is None:
                leaf_width = boundary_diffs[0]
            elif not math.isclose(leaf_width, boundary_diffs[0]):
                raise ValueError(
                    f"Distance between leaf boundary positions ({boundary_diffs[0]})"
                    f" differs from given leaf_width ({leaf_width})"
                )
        elif len(leaf_position_boundaries) == 1 and leaf_width is None:
            raise ValueError(
                "leaf_width is required to place leaf rows when it cannot be derived "
                "from the leaf_position_boundaries (a single row)."
            )
        return leaf_width

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

            self.leaf_width = self.validate_leaf_width_consistency(
                self.leaf_position_boundaries, self.leaf_width
            )

            self.number_of_elements = max(
                self.number_of_elements, len(self.leaf_position_boundaries)
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

        boundaries_max = np.abs(
            np.concatenate(
                [self.leaf_position_boundaries, self.leaf_position_boundaries + self.leaf_width]
            )
        ).max()
        positions_max = np.abs(self.leaf_positions).max()
        half_size = max(boundaries_max, positions_max)

        if isinstance(spacing, float):
            spacing_vector = _centered_grid(half_size, spacing)
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
                # Boundary runs along Y axis, opening runs along X axis
                cov_boundary = _smooth_step(grid_y, b_start, dx_y) * (
                    1.0 - _smooth_step(grid_y, b_end, dx_y)
                )
                cov_opening = _smooth_step(grid_x, o_start, dx_x) * (
                    1.0 - _smooth_step(grid_x, o_end, dx_x)
                )
            elif self.device_orientation == "Y":
                # Boundary runs along X axis, opening runs along Y axis
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

    def calculate_from_shape(  # noqa: PLR0913
        self,
        shape_map: NDArray,
        min_travel: float,
        stack_ray_pos: np.ndarray,
        bixel_width: float,
        lim_l: np.ndarray,
        lim_r: np.ndarray,
        leaf_width: Optional[float] = None,
    ) -> Self:
        """Derive leaf boundaries and positions from a shape map and beam geometry.

        Parameters
        ----------
        shape_map : NDArray
            Open-bixel map for the shape, shape (dim_stack, dim_travel).
        min_travel : float
            Physical travel-axis coordinate of the corner bixel (mm).
        stack_ray_pos : np.ndarray
            Ray positions along the leaf-stack axis (mm), used for the leaf boundaries.
            These are already leaf-row-resolved when the caller resamples onto a
            physical MLC grid coarser than the ray/bixel grid.
        bixel_width : float
            Bixel width / spot spacing (mm), used for the leaf travel axis, which is
            never resampled.
        lim_l, lim_r : np.ndarray
            Left/right leaf travel limits per leaf row (mm), used for the closed-leaf
            fallback and to validate the resulting positions.
        leaf_width : float, optional
            The true row pitch (mm) of ``stack_ray_pos``. Needed to set
            ``self.leaf_width`` correctly when there's only a single leaf row
            (no boundary spacing to derive it from). Otherwise it is derived
            from/cross-checked against the boundary spacing as before.
        """
        # Get leaf position boundaries from unique ray positions (ray positions are at
        # row centres, while leaf_position_boundaries are the rows' lower edges)
        row_centers = np.unique(stack_ray_pos)
        leaf_width = self.validate_leaf_width_consistency(row_centers, leaf_width)
        self.leaf_width = leaf_width
        self.leaf_position_boundaries = row_centers - leaf_width / 2.0

        # Compute left/right leaf positions from open bixel columns in physical coordinates
        dim_stack = shape_map.shape[0]
        self.leaf_positions = np.full((dim_stack, 2), np.nan)
        for n in range(dim_stack):
            open_cols = np.where(shape_map[n, :] > 0)[0]
            if open_cols.size == 0:
                # If no bixel is open, use the midpoint of limits
                center_pos = (lim_l[n] + lim_r[n]) / 2
                self.leaf_positions[n] = [center_pos, center_pos]
            else:
                self.leaf_positions[n] = [
                    (open_cols[0]) * bixel_width + min_travel - 0.5 * bixel_width,
                    (open_cols[-1]) * bixel_width + min_travel + 0.5 * bixel_width,
                ]

        # Validate positions stay within travel limits
        if np.any(self.leaf_positions[:, 0] < lim_l):
            raise ValueError("Left leaf positions exceed left travel limits (lim_l)")
        if np.any(self.leaf_positions[:, 1] > lim_r):
            raise ValueError("Right leaf positions exceed right travel limits (lim_r)")

        self.validate_boundaries_and_positions()
        return self


class Jaw(BeamLimitingDevice):
    """Class for Jaw objects.

    A pair of jaws along one axis, open between two positions, within the maximum field width of
    the device.

    Attributes
    ----------
    positions : list[float]
        Jaw edge positions in mm (in isocenter plane).
    field_width : float
        Maximum field width of the device in mm (in isocenter plane). Positions must lie
        within ``±field_width / 2``. A single jaw collimates along its other axis too, unless
        ``field_width`` covers the whole field.
    leakage : float
        Leakage of the jaws relative to open field (0-1).
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
            expected = f"JAW{device_orientation}"
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
            spacing_vector = _centered_grid(self.field_width / 2, spacing)
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
    data : dict or BeamLimitingDevice, optional
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
    beam_limiting_device : dict or BeamLimitingDevice, optional
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

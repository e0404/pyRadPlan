"""Beamlet datamodels for particle and photon beamlets."""

from typing import Optional, Any, Union
from pydantic import (
    Field,
    field_serializer,
    field_validator,
    model_validator,
    SerializerFunctionWrapHandler,
    FieldSerializationInfo,
)
import logging
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pyRadPlan.stf._rangeshifter import RangeShifter
from pyRadPlan.core import PyRadPlanBaseModel

from pyRadPlan.machines import BeamLimitingDevice

# TODO: We need to figure out how we can do nested validation in pydantic and then differentiate
# between photon and particle beamlets. For now, we will use the Beamlet class for both.

logger = logging.getLogger(__name__)


class Beamlet(PyRadPlanBaseModel):
    """
    A class representing a single beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the bemamlet information, including properties like
    energy & monitor units.

    Attributes
    ----------
    energy : float
        The energy value for the beamlet
    num_particles_per_mu : float
        The number of particles per monitor unit
    relative_fluence : float
        The fluence of this beamlet relative to the central primary fluence.
        For example, due to a non-uniform primary fluence
    weight : float
        The applied fluence weight of the beamlet
    min_mu : float
        The minimum monitor unit
    max_mu : float
        The maximum monitor unit
    range_shifter : RangeShifter
        The range shifter applied for the beamlet.
    focus_ix : int
        The focus index identifying the focus setting for the beamlet.
    """

    energy: float
    num_particles_per_mu: float = Field(alias="numParticlesPerMU", default=1.0e6)
    min_mu: float = Field(alias="minMU", default=0.0)
    max_mu: float = Field(alias="maxMU", default=float("inf"))
    relative_fluence: float = Field(default=1.0)
    weight: float = Field(default=1.0)
    range_shifter: RangeShifter = Field(default_factory=RangeShifter)
    focus_ix: int = Field(default=0)

    @field_serializer(
        "energy",
        "num_particles_per_mu",
        "min_mu",
        "max_mu",
        "relative_fluence",
        "weight",
        "focus_ix",
        mode="wrap",
    )
    def field_typing(
        self, v: Any, handler: SerializerFunctionWrapHandler, info: FieldSerializationInfo
    ) -> Any:
        """Ensure correct serialization in various contexts."""

        context = info.context
        if context and context.get("matRad") == "mat-file":
            if info.field_name == "focus_ix":
                return np.float64(v + 1)  # Increment focus_ix by 1 for MATLAB/matRad
            return np.float64(v)  # Ensure double for MATLAB/matRad

        return handler(v, info)


# Do not use yet


class IonSpot(Beamlet):
    """
    A class representing a single beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information specific to particles, including properties like
    range shifter and focus index.

    Attributes
    ----------
    range_shifter : RangeShifter
        The range shifter applied for the beamlet.
    focus_ix : int
        The focus index identifying the focus setting for the beamlet.
    """


class PhotonBixel(Beamlet):
    """
    A class representing a single photon beamlet.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information for photons. Mainly the relative fluence of the beamlet /
    bixel due to its lateral position is stored in here.

    Attributes
    ----------
    relative_fluence : float
        The fluence of this beamlet relative to the central primary fluence.
    """


class FieldShape(Beamlet):
    """
    Base class representing a shape object.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information for a shape.

    Notes
    -----
    Coordinates in FieldShapes are defined in BEV in LPS.
    Note that Beam Limiting Devices from DICOM are typically defined in IEC with the
    X axis flipped compared to BEV convention (when gantry and couch angle are 0).
    """

    # TODO:
    # - forcing square mask correct? Should non-squared be padded automatically?
    # - forcing even mask shape (0 padding) correct?
    # - Should resolution and grid be top-level (+ grid validation)?

    is_field_based: bool = True
    resolution: Optional[float] = None
    grid: Optional[np.ndarray] = None
    field_width: Optional[float] = None

    @staticmethod
    def _resolve_resolution(resolution: Optional[float], grid: Optional[np.ndarray]) -> float:
        if resolution is None and grid is None:
            raise ValueError("Either resolution or grid must be provided")
        if grid is not None:
            grid_resolution = np.unique(np.round(np.diff(grid), decimals=10))
            if len(grid_resolution) != 1:
                raise ValueError("Grid must have uniform spacing")
            if resolution is not None:
                if not np.isclose(grid_resolution, resolution):
                    raise ValueError(
                        f"Grid spacing ({grid_resolution}mm) does not match "
                        f"resolution ({resolution}mm)"
                    )
            return float(grid_resolution[0])
        return float(resolution)

    @staticmethod
    def _resolve_field_width(field_width: Optional[float], grid: Optional[np.ndarray]) -> float:
        if field_width is None and grid is None:
            raise ValueError("Either resolution or grid must be provided")
        if grid is not None:
            grid_resolution = np.unique(np.round(np.diff(grid), decimals=10))
            if len(grid_resolution) != 1:
                raise ValueError("Grid must have uniform spacing")
            # TODO: field_width = #intervals not #points ?
            grid_field_width = float((len(grid) - 1) * grid_resolution)
            if field_width is not None:
                if not np.isclose(grid_field_width, field_width):
                    raise ValueError(
                        f"Grid field width ({grid_field_width}mm) does not match "
                        f"field width ({field_width}mm)"
                    )
            return grid_field_width
        return float(field_width)

    @staticmethod
    def _build_grid(resolution: float, field_width: float) -> np.ndarray:
        n = int(np.ceil(field_width / resolution))
        if n % 2 == 0:
            n += 1
        half_size = n // 2
        return resolution * np.arange(-half_size, half_size + 1)

    # TODO: Is the padding included in the grid?
    @staticmethod
    def _validate_mask_array(v: np.ndarray) -> np.ndarray:
        if len(v.shape) != 2:
            raise ValueError("Mask must be 2D")
        if v.shape[0] != v.shape[1]:
            raise ValueError("Mask must be quadratic")
        if v.shape[0] % 2 == 0:
            logger.warning(
                f"Mask size {v.shape[0]} is even. Padding with zeros to make it odd. "
                "This may slightly affect convolution results.",
                UserWarning,
            )
            v = np.pad(v, ((0, 1), (0, 1)), mode="constant", constant_values=0)
        return np.array(v) if not isinstance(v, np.ndarray) else v

    @model_validator(mode="after")
    def validate_spatial_consistency(self) -> "FieldShape":
        # skip if subclass hasn't set spatial parameters yet
        if self.grid is None and self.field_width is None:
            return self
        return self._validate_and_derive_spatial()

    def _validate_and_derive_spatial(self) -> "FieldShape":
        if self.grid is not None:
            resolution = self._resolve_resolution(self.resolution, self.grid)
            self.__dict__["resolution"] = resolution

            field_width = self._resolve_field_width(self.field_width, self.grid)
            self.__dict__["field_width"] = field_width
        else:
            if self.resolution is None or self.field_width is None:
                raise ValueError("Grid must be given or resolution with field width")
            self.__dict__["grid"] = self._build_grid(
                resolution=self.resolution, field_width=self.field_width
            )
        return self

    def _resample_to_grid(self, new_grid: np.ndarray) -> "FieldShapeAsMask":
        raise NotImplementedError("Subclasses must implement _resample_to_grid")

    def resample(
        self,
        new_resolution: Optional[float] = None,
        new_grid: Optional[np.array] = None,
        new_field_width: Optional[float] = None,
    ) -> "FieldShape":
        if new_resolution is None and new_grid is None and new_field_width is None:
            raise ValueError(
                "At least one of 'new_resolution', 'new_grid', or 'new_field_width' must be provided"
            )

        if new_resolution is not None or new_grid is not None:
            new_resolution = self._resolve_resolution(new_resolution, new_grid)
        else:
            new_resolution = self.resolution

        if new_field_width is not None or new_grid is not None:
            new_field_width = self._resolve_field_width(new_field_width, new_grid)
        else:
            new_field_width = self.field_width

        if new_grid is None:
            new_grid = self._build_grid(new_resolution, new_field_width)

        return self._resample_to_grid(new_grid=new_grid)

    # TODO: def set_weight()


class FieldShapeAsMask(FieldShape):
    """
    A class representing a shape object computed from a mask.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information for a shape.

    Attributes
    ----------
    mask : np.ndarray
        The mask representing the shape of the field.

    Notes
    -----
    The mask coordinates are defined in the beam's eye view (BEV) in LPS.
    """

    mask: np.ndarray

    @field_validator("mask", mode="before")
    @classmethod
    def validate_mask(cls, v):
        return cls._validate_mask_array(v)

    @model_validator(mode="after")
    def validate_grid(self) -> "FieldShapeAsMask":
        self.__dict__["resolution"] = self._resolve_resolution(self.resolution, self.grid)

        if self.grid is None and self.resolution is not None:
            # center grid around (0,0) (assume mask is centered at (0,0))
            half_size = self.mask.shape[0] // 2
            self.__dict__["grid"] = self.resolution * np.arange(-half_size, half_size + 1)

        return self._validate_and_derive_spatial()

    def _resample_to_grid(self, new_grid: np.ndarray) -> "FieldShapeAsMask":
        interpolator = RegularGridInterpolator(
            (self.grid, self.grid), self.mask, method="linear", bounds_error=False, fill_value=0.0
        )
        xx, yy = np.meshgrid(new_grid, new_grid, indexing="ij")
        new_mask = (
            interpolator(np.stack([xx.ravel(), yy.ravel()], axis=-1))
            .reshape(xx.shape)
            .astype(np.float32)
        )

        return FieldShapeAsMask(
            energy=self.energy, mask=new_mask, grid=new_grid, weight=self.weight
        )


class FieldShapeAsBLD(FieldShape):
    """
    A class representing a shape object computed from a beam limiting device.

    This class extends PyRadPlanBaseModel (pydantic) and provides functionality to
    handle the beamlet information for a shape.

    Attributes
    ----------
    bld : BeamLimitingDevice
        The beam limiting device from which to compute the field shape mask.

    """

    bld: BeamLimitingDevice

    @property
    def mask(self) -> np.ndarray:
        mask = self.bld.calculate_transmission_mask(spacing=self.resolution)
        # BLDs use IEC coordinates where X points to patient's right.
        # BEV-LPS X points to patient's left, so the X axis is inverted.
        mask = np.fliplr(mask)

        if self.field_width is not None:
            bld_field_width = (mask.shape[0] - 1) * self.resolution
            if self.field_width >= bld_field_width + self.resolution:
                n = int(np.ceil(self.field_width / self.resolution))
                if n % 2 == 0:
                    n += 1
                pad_pixels = (n - mask.shape[0]) // 2
                mask = np.pad(
                    mask, pad_pixels, mode="constant"
                )  # TODO: add leakage as "constant_values"?
        return self._validate_mask_array(mask)

    @model_validator(mode="after")
    def setup_spatial_parameters(self) -> "FieldShapeAsBLD":
        self.__dict__["resolution"] = self._resolve_resolution(self.resolution, self.grid)

        # TODO: add field_width here directly & build grid with _build_grid

        if self.grid is None:
            # center grid around (0,0) (assume mask is centered at (0,0))
            half_size = self.mask.shape[0] // 2
            self.__dict__["grid"] = self.resolution * np.arange(-half_size, half_size + 1)

        if len(self.grid) != self.mask.shape[0]:
            raise ValueError("Grid dimensions must match mask shape")

        return self._validate_and_derive_spatial()

    def _resample_to_grid(self, new_grid: np.ndarray) -> "FieldShapeAsBLD":
        return FieldShapeAsBLD(energy=self.energy, bld=self.bld, grid=new_grid, weight=self.weight)


class FieldShapeComposite(FieldShape):
    """
    A field shape composed of multiple shapes combined by element-wise mask multiplication.

    This class represents a combined field shape where the total transmission is the
    product of the individual shape transmissions (e.g., multiple beam limiting devices
    applied simultaneously). Spatial parameters (resolution, field_width) are derived
    from the child shapes when not explicitly provided.

    Attributes
    ----------
    shapes : list[Union[FieldShapeAsMask, FieldShapeAsBLD]]
        The child shapes whose masks are multiplied to form the composite mask.
    """

    shapes: list[Union[FieldShapeAsMask, FieldShapeAsBLD]]

    @property
    def mask(self) -> np.ndarray:
        combined = np.ones((len(self.grid), len(self.grid)), dtype=np.float32)
        for shape in self.shapes:
            combined *= shape.resample(new_grid=self.grid).mask
        return combined

    @model_validator(mode="after")
    def setup_spatial(self) -> "FieldShapeComposite":
        if self.resolution is None and self.grid is None:
            self.__dict__["resolution"] = min(s.resolution for s in self.shapes)
        if self.field_width is None and self.grid is None:
            self.__dict__["field_width"] = max(s.field_width for s in self.shapes)
        return self._validate_and_derive_spatial()

    def _resample_to_grid(self, new_grid: np.ndarray) -> "FieldShapeComposite":
        return FieldShapeComposite(
            energy=self.energy, shapes=self.shapes, grid=new_grid, weight=self.weight
        )

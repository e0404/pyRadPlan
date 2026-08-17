"""Sequenced beam object."""

from typing import Any, Optional
from numpy.typing import NDArray
import logging
import numpy as np

from pydantic import (
    Field,
    model_validator,
    field_validator,
    ValidationInfo,
    computed_field,
)
from pyRadPlan.core import PyRadPlanBaseModel

logger = logging.getLogger(__name__)


class BeamSequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    shapes: NDArray
    shapes_weight: NDArray
    bixel_ix: list[int]
    fluence: NDArray
    leaf_pair_pos: Optional[list[float]] = Field(default=None)
    is_active_leaf_pair: Optional[list[bool]] = Field(default=None)
    bixel_ind_map: Optional[NDArray] = Field(default=None)
    pos_of_corner_bixel: Optional[list[float]] = Field(default=None)
    mlc_window: Optional[list[float]] = Field(alias="MLCWindow", default=None)
    lim_l: Optional[list[float]] = Field(default=None)
    lim_r: Optional[list[float]] = Field(default=None)
    central_leaf_pair: Optional[float] = Field(default=None)
    left_leaf_pos: Optional[list[NDArray]] = Field(default=None)
    right_leaf_pos: Optional[list[NDArray]] = Field(default=None)
    weight: Optional[list[float]] = Field(default=None)
    shape_map: Optional[list[NDArray]] = Field(default=None)
    vector_offset: Optional[list[int]] = Field(default=None)
    num_of_active_leaf_pairs: Optional[int] = Field(default=None)

    @computed_field
    @property
    def sum_shapes(self) -> NDArray:
        """Sum of all shapes in the beam along the last dimension."""
        if self.shapes is None:
            return np.array([])
        if self.shapes.ndim == 3:
            return np.sum(self.shapes, axis=-1)  # Sum over the last dimension (e.g. n=1 => 3x3x1)
        elif self.shapes.ndim == 2:
            return self.shapes
        else:
            logger.warning("Shapes have an unsupported number of dimensions.")
            return np.array([])  # Return an empty array for unsupported cases

    @computed_field
    @property
    def num_of_shapes(self) -> int:
        """Total number of shapes in the beam."""
        if self.shapes is None:
            return 0
        return self.shapes.shape[2] if self.shapes.ndim == 3 else 1

    @field_validator("bixel_ind_map", mode="after")
    @classmethod
    def validate_bixel_ind_map(cls, v: Any, info: ValidationInfo) -> np.ndarray:
        """Rotate 2D index map and convert from 1-based to 0-based indexing."""

        if info.context and info.context.get("from_matRad"):
            arr = np.asarray(v, dtype=np.float64)
            mask = ~np.isnan(arr)
            arr[mask] -= 1  # Convert to 0-based indexing
            return arr.T

        return v

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad."""
        if data == {}:
            return data

        # add dimension because if num_of_shapes = 1, matLab saves only has 2d Shapes
        # TODO: In future we should add up all shapes along axis=0 (currently axis=2)
        if "shapes" in data:
            if data["shapes"].ndim == 2:
                shapes = data["shapes"][:, :, np.newaxis]
                data["shapes"] = shapes
        if "shapes_weight" in data:
            shapes_weight = np.array(data["shapes_weight"])
            data["shapes_weight"] = shapes_weight

        if "shape" in data:
            for key in data["shape"].keys():
                if data["num_of_shapes"] != 1:
                    data.update({key: data["shape"][key]})
                else:
                    data.update({key: [data["shape"][key]]})
            data.pop("shape", None)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)

        return data

    # shapes are binary aperture masks, so uint8 is lossless and compact
    @field_validator("shapes", mode="before")
    @classmethod
    def _to_uint8(cls, v):
        arr = np.asarray(v)
        return arr.astype(np.uint8)

    # fluence carries continuous weights and must not be rounded or wrapped
    @field_validator("fluence", mode="before")
    @classmethod
    def _to_float(cls, v):
        arr = np.asarray(v)
        return arr.astype(float)

    # ensure shapes_weight always 1D float
    @field_validator("shapes_weight", mode="before")
    @classmethod
    def _to_float1d(cls, v):
        arr = np.asarray(v)
        return arr.astype(float).reshape(-1)

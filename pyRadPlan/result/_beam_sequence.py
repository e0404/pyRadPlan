"""Sequenced beam object."""

from typing import Any, Union
from numpy.typing import NDArray
import logging
import numpy as np

from pydantic import (
    Field,
    model_validator,
    field_validator,
    ValidationInfo,
)
from pyRadPlan.core import PyRadPlanBaseModel

logger = logging.getLogger(__name__)


class BeamSequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    num_of_shapes: int
    shapes: NDArray
    shapes_weight: NDArray
    bixel_ix: list[int]
    fluence: NDArray
    sum_shapes: Union[NDArray, None] = Field(default=None, alias="sum")

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
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)

        return data

    @field_validator("shapes", "fluence", mode="before")
    @classmethod
    def _to_uint8(cls, v):
        arr = np.asarray(v)
        return arr.astype(np.uint8)

    # ensure shapes_weight always 1D float
    @field_validator("shapes_weight", mode="before")
    @classmethod
    def _to_float1d(cls, v):
        arr = np.asarray(v)
        return arr.astype(float).reshape(-1)

    # ensure sum_shapes, if present, always float
    @field_validator("sum_shapes", mode="before")
    @classmethod
    def _sum_to_float(cls, v):
        if v is None:
            return None
        arr = np.asarray(v)
        return arr.astype(float)

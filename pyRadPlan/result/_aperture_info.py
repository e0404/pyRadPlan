"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import Any, Union
from numpy.typing import NDArray
import logging

from pydantic import (
    Field,
    model_validator,
    ValidationInfo,
)
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.machines import MLC

logger = logging.getLogger(__name__)


class ApertureInfo(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    beam: Union[list, dict[str, Any]] = Field(default_factory=dict)
    bixel_width: float = Field(default=0.0, gt=0.0)
    num_of_mlc_leaf_pairs: int = Field(default=0, gt=0, alias="num_of_MLC_leaf_pairs")
    total_num_of_bixels: int = Field(default=0, gt=0)
    total_num_of_shapes: int = Field(default=0, gt=0)
    total_num_of_leaf_pairs: int = Field(default=0, gt=0)
    aperture_vector: Union[NDArray, None] = Field(default=None)
    mapping_mx: Union[NDArray, None] = Field(default=None)
    lim_mx: Union[NDArray, None] = Field(default=None)

    mlc_sequence: Union[list[MLC], None] = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # Early return for distribution input (assuming distribution is defined/imported elsewhere)

        return data

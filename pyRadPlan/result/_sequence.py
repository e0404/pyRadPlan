"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import Any
import logging
import numpy as np

from pydantic import (
    Field,
    model_validator,
    ValidationInfo,
)
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._aperture_info import ApertureInfo

logger = logging.getLogger(__name__)


class Sequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    fluence: np.ndarray = Field(default=None)
    aperture_info: ApertureInfo = Field(default=None)
    # sequencing: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # Early return for distribution input (assuming distribution is defined/imported elsewhere)

        return data

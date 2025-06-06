"""
Module for handling the result of calc_dose_forward and calc_dose_influence.

Defines a result pydantic model and factories.
"""

from typing import Any
import logging
import numpy as np
from pydantic.alias_generators import to_snake
from pydantic import (
    Field,
    model_validator,
    ValidationInfo,
)
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._aperture_info import ApertureInfo
from pyRadPlan.result._beam_sequence import BeamSequence

logger = logging.getLogger(__name__)


class Sequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    fluence: np.ndarray = Field(default=None, alias="w")
    aperture_info: ApertureInfo = Field(default=None)
    beam: list[BeamSequence] = Field(default=None)

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad."""
        if data == {}:
            return data

        if "beam" in data:
            # subtract 1 from bixelIx to convert from 1-based to 0-based indexing
            if "bixelIx" in data["beam"]:
                # handle case where bixelIx is a list of arrays
                data["beam"]["bixelIx"] = [
                    np.asarray(arr, dtype=int) - 1 for arr in data["beam"]["bixelIx"]
                ]

            beam_list = []
            first_key = next(iter(data["beam"]))
            n = len(data["beam"][first_key])

            for beam_ix in range(n):
                entry = {to_snake(k): data["beam"][k][beam_ix] for k in data["beam"]}
                beam_list.append(entry)

            data["beam"] = beam_list

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # Early return for distribution input (assuming distribution is defined/imported elsewhere)
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)
        return data

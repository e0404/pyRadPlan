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
    computed_field,
)
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._beam_sequence import BeamSequence

logger = logging.getLogger(__name__)


class Sequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam."""

    fluence: np.ndarray | None = Field(default=None, alias="w")
    # aperture_info: ApertureInfo | None = Field(default=None)
    beam: list[BeamSequence] | None = Field(default=None)
    bixel_width: float | None = Field(default=None)
    lim_mx: np.ndarray | None = Field(default=None)
    mapping_mx: np.ndarray | None = Field(default=None)
    aperture_vector: list[float] | None = Field(default=None)
    num_of_mlc_leaf_pairs: int | None = Field(default=80, alias="numOfMLCLeafPairs")
    total_num_of_bixels: int | None = Field(default=None)

    @computed_field
    @property
    def total_num_of_leaf_pairs(self) -> int:
        """Total number of MLC leaf pairs across all beams."""
        if self.beam is None:
            return 0
        return sum(beam.num_of_shapes * beam.num_of_active_leaf_pairs for beam in self.beam)

    @computed_field
    @property
    def total_num_of_shapes(self) -> int:
        """Total number of shapes across all beams."""
        if self.beam is None:
            return 0
        return sum(beam.num_of_shapes for beam in self.beam)

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad."""
        if data == {}:
            return data

        if "beam" in data:
            # convert bixelix from 1-based to 0-based indexing
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
        if "aperture_info" in data:
            matrad_keys = [
                "totalNumOfBixels",
                "bixelWidth",
                "numOfMLCLeafpairs",
                "limMx",
                "mappingMx",
                "apertureVector",
            ]
            aperture_info = data.pop("aperture_info")

            for key in list(aperture_info.keys()):
                if key == "beam":
                    beam_aperture_list = []
                    first_key = next(iter(aperture_info[key]))
                    m = len(aperture_info[key][first_key])
                    if m != n:
                        logger.warning(
                            f"Number of beams in apertureInfo ({m}) does not match number of beams in data ({n})."
                        )
                    for beam_ix in range(m):
                        beam_entry = {
                            to_snake(k): aperture_info[key][k][beam_ix] for k in aperture_info[key]
                        }
                        beam_aperture_list.append(beam_entry)
                elif key in matrad_keys:
                    data[key] = aperture_info[key]
                else:
                    aperture_info.pop(key)
            for beam_ix, beam_data in enumerate(beam_aperture_list):
                data["beam"][beam_ix].update(beam_data)
        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # check matRad data format
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)
        return data

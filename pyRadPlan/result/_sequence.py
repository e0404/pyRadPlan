"""Sequencing information of a plan result."""

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

#: matRad apertureInfo entries that are stored flat on the Sequence itself.
_MATRAD_APERTURE_KEYS = (
    "totalNumOfBixels",
    "bixelWidth",
    "numOfMLCLeafPairs",
    "limMx",
    "mappingMx",
    "apertureVector",
)


class Sequence(PyRadPlanBaseModel):
    """A class representing the sequencing information of a plan."""

    fluence: np.ndarray | None = Field(default=None, alias="w")
    beam: list[BeamSequence] | None = Field(default=None)
    sequencer: str | None = Field(default=None)
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
        return sum(
            beam.num_of_shapes * beam.num_of_active_leaf_pairs
            for beam in self.beam
            if beam.num_of_active_leaf_pairs is not None
        )

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
        # Guard against non-dict inputs (e.g. an empty matRad struct read as an
        # ndarray by newer readers) before the truthiness test, since ``ndarray == {}``
        # raises an ambiguous-truth ValueError.
        if not isinstance(data, dict) or not data:
            return data

        # Copy so that the caller's dictionary survives validation unchanged.
        data = dict(data)
        beam_list = []

        if "beam" in data:
            beams = dict(data["beam"])
            # convert bixelix from 1-based to 0-based indexing
            if "bixelIx" in beams:
                # handle case where bixelIx is a list of arrays
                beams["bixelIx"] = [np.asarray(arr, dtype=int) - 1 for arr in beams["bixelIx"]]

            first_key = next(iter(beams))
            for beam_ix in range(len(beams[first_key])):
                beam_list.append({to_snake(k): beams[k][beam_ix] for k in beams})

            data["beam"] = beam_list

        if "aperture_info" in data:
            aperture_info = data.pop("aperture_info")
            beam_aperture_list = []

            for key in list(aperture_info.keys()):
                if key == "beam":
                    beam_apertures = aperture_info[key]
                    first_key = next(iter(beam_apertures))
                    for beam_ix in range(len(beam_apertures[first_key])):
                        beam_aperture_list.append(
                            {to_snake(k): beam_apertures[k][beam_ix] for k in beam_apertures}
                        )
                elif key in _MATRAD_APERTURE_KEYS:
                    data[key] = aperture_info[key]

            if beam_aperture_list and len(beam_aperture_list) != len(beam_list):
                logger.warning(
                    "Number of beams in apertureInfo (%d) does not match number of beams "
                    "in the sequencing data (%d). Aperture information is dropped.",
                    len(beam_aperture_list),
                    len(beam_list),
                )
            else:
                for beam_ix, beam_data in enumerate(beam_aperture_list):
                    beam_list[beam_ix].update(beam_data)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        # check matRad data format
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)
        return data

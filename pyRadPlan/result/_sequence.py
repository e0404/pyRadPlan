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


def _beam_structs(beams: Any) -> list[dict]:
    """Split a matRad struct array of beams into one dictionary per beam.

    A struct array is read as one list per field. A single beam is read as a plain
    struct whose fields hold the values themselves.
    """
    if isinstance(beams, (list, tuple)):
        return [dict(beam) for beam in beams]
    values = list(beams.values())
    if not values:
        return []
    if all(isinstance(v, list) for v in values) and len({len(v) for v in values}) == 1:
        return [{key: beams[key][ix] for key in beams} for ix in range(len(values[0]))]
    return [dict(beams)]


class Sequence(PyRadPlanBaseModel):
    """A class representing the sequencing information of a plan."""

    fluence: np.ndarray | None = Field(default=None, alias="w")
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
        if data == {}:
            return data

        # Copy so that the caller's dictionary survives validation unchanged.
        data = dict(data)
        beam_list = []

        if "beam" in data:
            for beam in _beam_structs(data["beam"]):
                if "bixelIx" in beam:
                    # matRad counts the bixels from 1
                    beam["bixelIx"] = np.asarray(beam["bixelIx"], dtype=int).reshape(-1) - 1
                beam_list.append({to_snake(k): v for k, v in beam.items()})

            data["beam"] = beam_list

        if "aperture_info" in data:
            aperture_info = data.pop("aperture_info")
            beam_aperture_list = []

            for key in list(aperture_info.keys()):
                if key == "beam":
                    beam_aperture_list = [
                        {to_snake(k): v for k, v in beam.items()}
                        for beam in _beam_structs(aperture_info[key])
                    ]
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

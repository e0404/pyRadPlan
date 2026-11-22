"""Delivery sequence of a plan result."""

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from pydantic import Field, ValidationInfo, computed_field, field_validator, model_validator

from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.result._beam_sequence import BeamSequence

if TYPE_CHECKING:
    from pyRadPlan.result._aperture_info import ApertureInfo
    from pyRadPlan.stf import SteeringInformation


class Sequence(PyRadPlanBaseModel):
    """What the machine does to deliver a plan.

    Attributes
    ----------
    beams : list[BeamSequence]
        The deliverable beams, each a static field or an arc made of segments.
    fluence : np.ndarray, optional
        Beamlet weights the sequence delivers, in the order of the beamlets of the steering
        information.
    sequencer : str, optional
        Name of the sequencer that produced the sequence.
    num_of_levels : int, optional
        Number of intensity levels the fluence was stratified into.
    weight_to_mu : float, optional
        Monitor units per unit of segment weight.
    """

    beams: list[BeamSequence] = Field(default_factory=list)
    fluence: Optional[np.ndarray] = Field(default=None, alias="w")
    sequencer: Optional[str] = None
    num_of_levels: Optional[int] = None
    weight_to_mu: Optional[float] = Field(default=None, gt=0.0)

    @model_validator(mode="before")
    @classmethod
    def _from_matrad(cls, data: Any, info: ValidationInfo) -> Any:
        if info.context and info.context.get("from_matRad") and isinstance(data, dict):
            from pyRadPlan.result._matrad_sequence import sequence_from_matrad  # noqa: PLC0415

            return sequence_from_matrad(data)
        return data

    @field_validator("fluence", mode="before")
    @classmethod
    def _fluence_vector(cls, v: Any) -> Any:
        return None if v is None else np.asarray(v, dtype=float).reshape(-1)

    @computed_field
    @property
    def total_num_of_segments(self) -> int:
        """Number of segments of all beams."""
        return sum(beam.num_of_segments for beam in self.beams)

    def aperture_info(self, stf: "SteeringInformation") -> "ApertureInfo":
        """Relate the segments to the beamlets of *stf* (see :class:`ApertureInfo`)."""
        from pyRadPlan.result._aperture_info import ApertureInfo  # noqa: PLC0415

        return ApertureInfo.from_sequence(self, stf)

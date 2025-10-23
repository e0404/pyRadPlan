from typing import Any


# import warnings
import numpy as np
from pydantic import (
    field_validator,
)
from pyRadPlan.machines.particles.kernel._base import ParticlePencilBeamKernel


class IonPencilBeamKernel(ParticlePencilBeamKernel):
    """Data Model for a single Ion Pencil Beam Kernel."""

    peak_pos: np.float64

    @field_validator("peak_pos", mode="before")
    @classmethod
    def validate_peak_pos_cast(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64."""
        try:
            return np.float64(v)
        except ValueError:
            return v

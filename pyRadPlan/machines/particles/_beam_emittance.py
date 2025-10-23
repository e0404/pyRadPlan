from typing import Final

from pydantic import (
    Field,
)
from pyRadPlan.core import PyRadPlanBaseModel


class ChargedBeamEmittance(PyRadPlanBaseModel):
    """Emittance data for charged particle accelerator."""

    type: Final[str] = "bigaussian"

    sigma_x: float = Field(alias="sigmaX", ge=0.0)
    sigma_y: float = Field(alias="sigmaY", ge=0.0)
    div_x: float = Field(default=1e-12, alias="divX")
    div_y: float = Field(default=1e-12, alias="divY")
    corr_x: float = Field(default=1e-12, alias="corrX", ge=-1.0, le=1.0)
    corr_y: float = Field(default=1e-12, alias="corrY", ge=-1.0, le=1.0)

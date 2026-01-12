# import warnings
import numpy as np
from typing import Any
from pydantic import (
    Field,
    field_validator,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel


class LateralCutOff(PyRadPlanBaseModel):
    """Lateral cut-off radius and compensation for ion pencil beam kernels."""

    comp_fac: np.float64 = Field(default=np.float64(1.0))
    cut_off: NDArray[Shape["1-*"], np.float64] = Field(
        default=np.array([np.inf, np.inf], dtype=np.float64)
    )
    depths: NDArray[Shape["1-*"], np.float64] = Field(
        default=np.array([0.0, np.inf], dtype=np.float64)
    )

    @field_validator("comp_fac", mode="before")
    @classmethod
    def validate_comp_fac(cls, v: Any) -> Any:
        """Validate the lateral cut-off compensation factor."""
        try:
            return np.float64(v)
        except ValueError as exc:
            raise exc

    @field_validator("cut_off", mode="before")
    @classmethod
    def validate_arrays(cls, v: Any) -> Any:
        """Validate the lateral cut-off radial distance and depths."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

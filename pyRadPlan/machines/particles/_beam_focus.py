from typing import Any, Optional

# import warnings
import numpy as np
from pydantic import (
    field_validator,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel
from ._beam_emittance import ChargedBeamEmittance


class ChargedBeamFocus(PyRadPlanBaseModel):
    """Focus data for charged particle pencil beam kernels."""

    dist: NDArray[Shape["1-*"], np.float64]
    sigma: NDArray[Shape["1-*"], np.float64]
    fwhm_iso: Optional[np.float64] = None

    # emittance parameterization
    emittance: Optional[ChargedBeamEmittance] = None

    @field_validator("dist", "sigma", mode="before")
    @classmethod
    def validate_arrays(cls, v: Any) -> Any:
        """Validate the focus distance and sigma arrays."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    @property
    def has_emittance(self) -> bool:
        """Check if emittance parameters are available."""
        return self.sigma_x is not None and self.sigma_y is not None

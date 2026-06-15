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
    fwhm_iso: Optional[float] = None

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

    @classmethod
    def from_dict(cls, data: dict) -> "list[ChargedBeamFocus] | ChargedBeamFocus":
        emittance = data.get("emittance", None)

        if emittance is not None and len(data["dist"].shape) > 1:
            # Return a list of ChargedBeamFocus, one per energy slice
            focuses = []
            fwhm_iso = data.get("SisFWHMAtIso", None)
            for i in range(data["dist"].shape[0]):
                single_emittance = ChargedBeamEmittance(
                    type=data["emittance"]["type"][i],
                    sigma_x=data["emittance"]["sigmaX"][i],
                    sigma_y=data["emittance"]["sigmaY"][i],
                    div_x=data["emittance"]["divX"][i],
                    div_y=data["emittance"]["divY"][i],
                    corr_x=data["emittance"]["corrX"][i],
                    corr_y=data["emittance"]["corrY"][i],
                )
                focuses.append(
                    cls(
                        dist=data["dist"][i, :],
                        sigma=data["sigma"][i, :],
                        fwhm_iso=fwhm_iso[i] if fwhm_iso is not None else None,
                        emittance=single_emittance,
                    )
                )
            return focuses

        elif emittance is not None:
            return cls(
                dist=data["dist"],
                sigma=data["sigma"],
                fwhm_iso=data.get("SisFWHMAtIso", None),
                emittance=ChargedBeamEmittance(**data["emittance"]),
            )
        else:
            return cls(
                dist=data["dist"],
                sigma=data["sigma"],
                fwhm_iso=data.get("SisFWHMAtIso", None),
                emittance=None,
            )

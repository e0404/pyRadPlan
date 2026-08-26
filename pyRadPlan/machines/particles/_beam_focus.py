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
        return self.emittance is not None

    @classmethod
    def from_dict(cls, data: dict) -> "list[ChargedBeamFocus] | ChargedBeamFocus":
        """
        Create a focus from a matRad ``initFocus`` entry.

        Parameters
        ----------
        data : dict
            Entry with ``dist``, ``sigma``, optionally ``SisFWHMAtIso`` and ``emittance``
            (matRad field names). If ``dist`` is 2-D, each row describes one focus with its
            own emittance row and a list of foci is returned.
        """
        emittance = data.get("emittance", None)
        dist = np.asarray(data["dist"], dtype=np.float64)

        if emittance is not None and dist.ndim > 1:
            # Return a list of ChargedBeamFocus, one per row
            focuses = []
            fwhm_iso = data.get("SisFWHMAtIso", None)
            if fwhm_iso is not None:
                fwhm_iso = np.broadcast_to(np.asarray(fwhm_iso, dtype=float), (dist.shape[0],))
            sigma = np.asarray(data["sigma"], dtype=np.float64)
            for i in range(dist.shape[0]):
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
                        dist=dist[i, :],
                        sigma=sigma[i, :],
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

from typing import Any, Optional, Union


# import warnings
import numpy as np
from pydantic import (
    Field,
    field_validator,
    ValidationInfo,
    AliasChoices,
    computed_field,
    model_validator,
)
from numpydantic import NDArray, Shape
from pyRadPlan.core import PyRadPlanBaseModel
from .._beam_cutoff import LateralCutOff


class ParticlePencilBeamKernel(PyRadPlanBaseModel):
    """Data Model for a single Charged Pencil Beam Kernel."""

    energy: np.float64
    range: Optional[np.float64] = None
    offset: np.float64 = Field(default=np.float64(0.0))
    depths: NDArray[Shape["1-*"], np.float64]

    # all kernels
    idd: NDArray[Shape["1-*"], np.float64] = Field(
        validation_alias=AliasChoices("Z", "z"), serialization_alias="Z"
    )
    sigma: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_1: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_2: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_x: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_y: Optional[NDArray[Shape["1-*"], np.float64]] = None
    weight: Optional[NDArray[Shape["1-*"], np.float64]] = None
    sigma_multi: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    weight_multi: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    let: Optional[NDArray[Shape["1-*"], np.float64]] = Field(alias="LET", default=None)
    alpha_x: Optional[NDArray[Shape["1-*"], np.float64]] = None
    beta_x: Optional[NDArray[Shape["1-*"], np.float64]] = None
    alpha: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    beta: Optional[NDArray[Shape["1-*,1-*"], np.float64]] = None
    lateral_cut_off: Optional[LateralCutOff] = None

    @model_validator(mode="before")
    @classmethod
    def validate_sigma_xy(cls, data: Any) -> Any:
        # simga_x and sigma_y are passed together as sigmaXY in .mat
        if "sigma_x" in data and "sigma_y" in data:
            return data
        if "sigmaXY" in data and data["sigmaXY"] is not None:
            try:
                data["sigma_x"] = data["sigmaXY"][:, 0]
                data["sigma_y"] = data["sigmaXY"][:, 1]
                data.pop("sigmaXY")
            except Exception as exc:
                raise ValueError("sigmaXY could not be parsed correctly.") from exc
        return data

    @field_validator("energy", "range", "offset", mode="before")
    @classmethod
    def validate_possible_cast(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64."""
        try:
            return np.float64(v)
        except ValueError:
            return v

    @field_validator(
        "depths",
        "idd",
        "sigma",
        "sigma_1",
        "sigma_2",
        "sigma_x",
        "sigma_y",
        "weight",
        "sigma_multi",
        "weight_multi",
        "let",
        "alpha",
        "beta",
        "alpha_x",
        "beta_x",
        mode="before",
    )
    @classmethod
    def validate_possible_cast_array(cls, v: Any) -> Any:
        """Validate if the input can be cast to a float64 array."""
        try:
            v = np.array(v, dtype=np.float64)
        except ValueError as exc:
            raise exc

        return v

    @field_validator(
        "idd", "sigma", "sigma_1", "sigma_2", "sigma_x", "sigma_y", "let", mode="after"
    )
    @classmethod
    def validate_kernel_lengths(
        cls, v: Union[np.ndarray, None], info: ValidationInfo
    ) -> Union[np.ndarray, None]:
        """Validate the length of the kernel data."""
        if v is None:
            return v

        if v.shape != info.data["depths"].shape:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @field_validator("sigma_multi", "weight_multi", mode="after")
    @classmethod
    def validate_multi_gaussian_kernel(
        cls, v: Union[np.ndarray, None], info: ValidationInfo
    ) -> Union[np.ndarray, None]:
        """Validate the length of the multi-gaussian kernel data."""
        if v is None:
            return v

        if v.shape[0] != info.data["depths"].shape[0]:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @field_validator("alpha", "beta", mode="after")
    @classmethod
    def validate_alpha_beta(
        cls, v: Union[NDArray[Shape["1-*"], np.float64], None], info: ValidationInfo
    ) -> Union[NDArray[Shape["1-*"], np.float64], None]:
        """Validate the length of the alpha-beta kernel data."""
        if v is None:
            return v

        # Check if kernel is transposed but oterhwise of correct shape
        if (
            v.shape[0] == info.data["depths"].shape[0]
            and v.shape[1] == info.data["alpha_x"].size
            and v.shape[1] == info.data["beta_x"].size
        ):
            v = np.ascontiguousarray(v.T)

        if v.shape[0] != info.data["alpha_x"].size and v.shape[0] != info.data["beta_x"].size:
            raise ValueError(
                "Kernel data length does not match the number of reference photon alpha / beta "
                "values."
            )

        if v.shape[1] != info.data["depths"].size:
            raise ValueError("Kernel data length does not match the depth data length.")

        return v

    @computed_field(return_type=NDArray[Shape["1-*"], np.float64], alias="alphaBetaRatio")
    @property
    def alpha_beta_ratio(self) -> Union[NDArray[Shape["1-*"], np.float64], None]:
        """Photon alpha-beta ratio for each alpha / beta kernel."""
        if self.alpha is None or self.beta is None:
            return None
        return self.alpha_x / self.beta_x

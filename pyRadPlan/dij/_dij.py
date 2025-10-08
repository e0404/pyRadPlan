"""Contains the dij class as a (collection of) influence matrices."""

from typing import Any, Union, Annotated, Optional, cast
from pydantic import (
    Field,
    field_validator,
    ValidationInfo,
    computed_field,
    field_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
)

import numpy as np
from numpydantic import NDArray
import SimpleITK as sitk
import scipy.sparse as sp

from pyRadPlan.core import Grid
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.util import swap_orientation_sparse_matrix


class Dij(PyRadPlanBaseModel):
    """
    Collection of Dose (or other quantity) Influence Matrices.

    Attributes
    ----------
    resolution : dict[str, Any]
        Voxel resolution in each dimension ('x', 'y', 'z').
    physical_dose : scipy.sparse.sparray
        Physical dose matrix.
    total_num_of_bixels : int
        Total number of bixels in the matrix.
    num_of_voxels : int
        Total number of voxels in the matrix.
    """

    dose_grid: Annotated[Grid, Field(default=None)]
    ct_grid: Annotated[Grid, Field(default=None)]

    physical_dose: Annotated[NDArray, Field(default=None)]
    let_dose: Annotated[NDArray, Field(default=None, alias="mLETDose")]
    alpha_dose: Annotated[NDArray, Field(default=None)]
    sqrt_beta_dose: Annotated[NDArray, Field(default=None)]

    num_of_beams: Annotated[int, Field(default=None)]

    bixel_num: Annotated[NDArray, Field(default=None)]
    ray_num: Annotated[NDArray, Field(default=None)]
    beam_num: Annotated[NDArray, Field(default=None)]

    rad_depth_cubes: Optional[list[NDArray]] = Field(default=None)

    @computed_field
    @property
    def total_num_of_bixels(self) -> int:
        """Number of bixels / beamlets in the dose influence matrix."""
        return int(self.bixel_num.size)

    @computed_field
    @property
    def num_of_voxels(self) -> int:
        """Number of voxels in the dose influence matrix."""
        return self.physical_dose.flat[0].shape[0]

    @computed_field
    @property
    def quantities(self) -> list[str]:
        """Name of available uantities matrices."""
        potential_quantities = ["physical_dose", "let_dose", "alpha_dose", "sqrt_beta_dose"]
        return [q for q in potential_quantities if getattr(self, q) is not None]

    @field_validator("physical_dose", "let_dose", "alpha_dose", "sqrt_beta_dose", mode="before")
    @classmethod
    def validate_matrices(cls, v: Any, info: ValidationInfo) -> np.ndarray:
        """
        Validate the physical dose matrix.

        Raises
        ------
            ValueError: if physical dose is not a 2D numpy array.
        """

        if v is None:
            return v

        if isinstance(v, (sp.spmatrix, sp.sparray)) or (
            isinstance(v, np.ndarray) and v.dtype != np.dtype(object)
        ):
            tmp = np.empty((1,), dtype=object)
            tmp[0] = v
            v = tmp

        if isinstance(v, list):
            v = np.asarray(v, dtype=object)

        for i in range(v.size):
            if v.flat[i] is not None:
                mat = v.flat[i]
                if not isinstance(mat, (sp.spmatrix, sp.sparray, np.ndarray)) or not np.issubdtype(
                    mat.dtype, np.number
                ):
                    raise ValueError(f"{info.field_name} must be a numeric array.")
                if not mat.ndim == 2:
                    raise ValueError(f"{info.field_name} must be a 2D array.")
                if not mat.shape == mat.shape:
                    raise ValueError(f"{info.field_name} must have consistent number of voxels.")

                if mat.shape[0] != info.data["dose_grid"].num_voxels:
                    raise ValueError(f"{info.field_name} shape inconsistent with ct grid")

        if info.context and "from_matRad" in info.context and info.context["from_matRad"]:
            if v is not None:
                for i in range(v.size):
                    shape = (
                        int(info.data["dose_grid"].dimensions[2]),
                        int(info.data["dose_grid"].dimensions[0]),
                        int(info.data["dose_grid"].dimensions[1]),
                    )
                    v.flat[i] = swap_orientation_sparse_matrix(
                        v.flat[i],
                        shape,
                        (1, 2),  # (65, 100, 100) example
                    )
                    if v.flat[i] is not None and not isinstance(v.flat[i], sp.csc_matrix):
                        v.flat[i] = sp.csc_matrix(v.flat[i])
            else:
                v = np.array([0])
            return [[v]]  # TODO

        return v

    @field_validator("dose_grid", "ct_grid", mode="before")
    @classmethod
    def validate_grid(cls, grid: Union[Grid, dict], info: ValidationInfo) -> Union[Grid, dict]:
        """
        Validate grid dictionaries.

        Raises
        ------
            ValueError:
        """
        # Check if it is a dictionary and then try to create a Grid object
        if isinstance(grid, dict):
            if info.context and "from_matRad" in info.context and info.context["from_matRad"]:
                grid["dimensions"] = np.array(
                    [grid["dimensions"][1], grid["dimensions"][0], grid["dimensions"][2]]
                )
                # TODO: might swap offset and resolution
                grid = Grid.model_validate(grid)
            else:
                grid = Grid.model_validate(grid)
        return grid

    @field_validator("beam_num", mode="before")
    @classmethod
    def validate_unique_indices_in_beam_num(
        cls, v: np.ndarray, info: ValidationInfo
    ) -> np.ndarray:
        """
        Validate the number of unique indices in beam_num.

        Raises
        ------
            ValueError: Number of unique indices does not match number of beams.
        """
        num_of_beams = info.data["num_of_beams"]
        if len(np.unique(v)) != num_of_beams:
            raise ValueError(
                "Number of unique indices in beam_num does not match number of beams."
            )
        return v

    @field_validator("beam_num", "ray_num", "bixel_num", mode="before")
    @classmethod
    def validate_numbering_arrays(cls, v: Any, info: ValidationInfo) -> np.ndarray:
        """
        Validate the numbering arrays.

        Raises
        ------
            ValueError: inconsistent numbering arrays.
        """
        if not isinstance(v, np.ndarray) and isinstance(v, int):
            v = np.array([v])
        # Check if the numbering arrays have the correct shape
        if info.data.get("physical_dose") is not None:
            dij_matrices = cast(np.ndarray, info.data["physical_dose"])
            for i in range(dij_matrices.size):
                if dij_matrices.flat[i] is not None:
                    mat = cast(Union[sp.spmatrix, sp.sparray, np.ndarray], dij_matrices.flat[i])

                    bix_num = mat.shape[1]

                    if v.ndim != 1:
                        raise ValueError("Numbering arrays must be 1-dimensional")

                    if v.size != bix_num:
                        raise ValueError(
                            "Numbering arrays shape inconsistent with number of bixels"
                        )

        if info.context and "from_matRad" in info.context and info.context["from_matRad"]:
            v -= 1
        return v

    # Serialization
    @field_serializer("dose_grid", "ct_grid", mode="wrap")
    def grid_serializer(
        self, value: Grid, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> dict:
        context = info.context
        if context and context.get("matRad") == "mat-file":
            return value.to_matrad(context=context["matRad"])
        return handler(value, info)

    @field_serializer("physical_dose", "let_dose", "alpha_dose", "sqrt_beta_dose")
    def physical_dose_serializer(self, value: np.ndarray, info: SerializationInfo) -> np.ndarray:
        context = info.context
        if context and context.get("matRad") == "mat-file" and value is not None:
            for i in range(value.size):
                shape = (
                    int(self.dose_grid.dimensions[2]),
                    int(self.dose_grid.dimensions[0]),
                    int(self.dose_grid.dimensions[1]),
                )
                value.flat[i] = swap_orientation_sparse_matrix(
                    value.flat[i],
                    shape,
                    (1, 2),  # (65, 100, 100) example
                )
                if value.flat[i] is not None and not isinstance(value.flat[i], sp.csc_matrix):
                    value.flat[i] = sp.csc_matrix(value.flat[i])
        # return 0 if value is None. savemat() cant handle 'None'
        elif context and context.get("matRad") == "mat-file" and value is None:
            value = np.array([0])
        return value

    @field_serializer("rad_depth_cubes")
    def rad_depth_cubes_serializer(self, value: np.ndarray, info: SerializationInfo) -> np.ndarray:
        context = info.context
        if context and context.get("matRad") == "mat-file" and value is not None:
            # TODO: it might be necessary to rotate the cube!
            return value
        # return 0 if value is None. savemat() cant handle 'None'
        elif context and context.get("matRad") == "mat-file" and value is None:
            value = np.array([0])
        return value

    @field_serializer("bixel_num", "ray_num", "beam_num")
    def numbering_arrays_serializer(
        self, value: np.ndarray, info: SerializationInfo
    ) -> np.ndarray:
        context = info.context
        if context and context.get("matRad") == "mat-file":
            return value.reshape(-1, 1)
        return value

    def to_matrad(self, context: str = "mat-file") -> Any:
        """Convert the Dij to matRad-compatible dictionary."""

        dij_dict = super().to_matrad(context=context)

        return dij_dict

    def get_result_arrays_from_intensity(
        self, intensity: np.ndarray, scenario_index: int = 0
    ) -> dict[str, np.ndarray]:
        """
        Compute result arrays from an intensity vector.

        Parameters
        ----------
        intensity : np.ndarray
            The intensity to apply to the dose influence matrix.
        scenario_index : int
            The scenario index to apply the intensity to.

        Returns
        -------
        dict[str,sitk.Image]
            A dictionary containing the quantity images for each scenario.
        """

        out = {}

        # TODO: implement quantity system to select the corresponding quantities automatically
        if self.physical_dose is not None:
            out["physical_dose"] = self.physical_dose.flat[scenario_index] @ intensity

        if self.let_dose is not None:
            if self.physical_dose is None:
                raise ValueError("Physical dose must be calculated for dose-weighted let")

            indices = out["physical_dose"] > 0.05 * np.max(out["physical_dose"])

            let_dose = self.let_dose.flat[scenario_index] @ intensity
            out["let"] = np.zeros_like(let_dose)
            out["let"][indices] = let_dose[indices] / out["physical_dose"][indices]

        if self.alpha_dose is not None and self.sqrt_beta_dose is not None:
            out["effect"] = (
                self.alpha_dose.flat[scenario_index] @ intensity
                + (self.sqrt_beta_dose.flat[scenario_index] @ intensity) ** 2
            )

        return out

    def compute_result_dose_grid(
        self, intensities: np.ndarray, scenario_index: int = 0
    ) -> dict[str, sitk.Image]:
        """
        Compute results on the dose grid from intensity vector.

        Parameters
        ----------
        intensity : np.ndarray
            The intensity to apply to the dose influence matrix.
        scenario_index : int
            The scenario index to apply the intensity to.

        Returns
        -------
        dict[str,sitk.Image]
            A dictionary containing the quantity images for each scenario.
        """

        out = self.get_result_arrays_from_intensity(intensities, scenario_index=scenario_index)
        # Create a sitk image for each scenario

        for key, value in out.items():
            # Create a sitk image for each scenario
            out[key] = sitk.GetImageFromArray(value.reshape(self.dose_grid.dimensions[::-1]))
            out[key].SetOrigin(self.dose_grid.origin)
            out[key].SetSpacing(self.dose_grid.resolution_vector)
            out[key].SetDirection(self.dose_grid.direction.ravel())

        return out

    def compute_result_ct_grid(
        self, intensities: np.ndarray, scenario_index: int = 0
    ) -> dict[str, sitk.Image]:
        """
        Compute results on the CT grid from intensity vector.

        Parameters
        ----------
        intensity : np.ndarray
            The intensity to apply to the dose influence matrix.
        scenario_index : int
            The scenario index to apply the intensity to.

        Returns
        -------
        dict[str,sitk.Image]
            A dictionary containing the quantity images for each scenario.
        """

        out = self.compute_result_dose_grid(intensities, scenario_index=scenario_index)
        # Create a sitk image for each scenario

        for key, value in out.items():
            # Create a sitk image for each scenario
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputDirection(self.ct_grid.direction.ravel())
            resampler.SetOutputOrigin(self.ct_grid.origin)
            resampler.SetOutputSpacing(self.ct_grid.resolution_vector)
            resampler.SetSize(self.ct_grid.dimensions)
            out[key] = resampler.Execute(value)

        return out


def create_dij(data: Union[dict[str, Any], Dij, None] = None, **kwargs) -> Dij:
    """
    Create a Dij object from raw data or keyword arguments.

    Parameters
    ----------
    data : Union[dict[str, Any], Dij, None]
        Dictionary containing the data to create the Dij object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Dij
        A Dij object.
    """

    if data:
        # If data is already a Dij object, return it directly
        if isinstance(data, Dij):
            return data

        if "beamNum" in data and np.min(data["beamNum"]) != 0:
            # add context when from matRad
            context = {"from_matRad": True}
        else:
            context = {"from_matRad": False}
        return Dij.model_validate(data, context=context)

    return Dij(**kwargs)


def validate_dij(dij: Union[dict[str, Any], Dij, None] = None, **kwargs) -> Dij:
    """
    Validate and creates a Dij object.

    Synonym to create_dij but should be used in validation context.

    Parameters
    ----------
    dij : Union[dict[str, Any], Dij, None], optional
        Dictionary containing the data to create the Dij object, by default None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Dij
        A validated Dij object.

    """
    return create_dij(dij, **kwargs)

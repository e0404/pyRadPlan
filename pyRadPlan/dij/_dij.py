"""Contains the dij class as a (collection of) influence matrices."""

from typing import Any, Union, Annotated, Optional, cast, ClassVar
from typing_extensions import Self
import logging

from pydantic import (
    Field,
    field_validator,
    ValidationInfo,
    computed_field,
    field_serializer,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_validator,
)

from numpydantic import NDArray, Shape
from ..core.xp_utils.typing import Array, ArrayNamespace

import array_api_compat

import numpy as np
import SimpleITK as sitk
import scipy.sparse as sp

from pyRadPlan.core import Grid
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.bio_models import BiologicalModel, ConstantRBEModel, create_bio_model
from pyRadPlan.util import swap_orientation_sparse_matrix

from ..core.xp_utils import to_namespace
from ..core.xp_utils.helpers import _rebuild_scipy_csc_in_namespace

InfluenceMatrixArray = Union[Array, sp.spmatrix, sp.sparray]
InfluenceMatrixContainer = NDArray[Shape["*, ..."], object]


logger = logging.getLogger(__name__)


def _check_influence_matrix(mat: Any, info: ValidationInfo):
    """Validate/coerce the input as influence matrix."""

    if not (
        isinstance(mat, (sp.spmatrix, sp.sparray, np.ndarray))
        or not np.issubdtype(mat.dtype, np.number)
    ) and not array_api_compat.is_array_api_obj(mat):
        raise ValueError(f"{info.field_name} must be a numeric array.")
    if not mat.ndim == 2:
        raise ValueError(f"{info.field_name} must be a 2D array.")
    if not mat.shape == mat.shape:
        raise ValueError(f"{info.field_name} must have consistent number of voxels.")

    if mat.shape[0] != info.data["dose_grid"].num_voxels:
        raise ValueError(f"{info.field_name} shape inconsistent with ct grid")


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

    physical_dose: Annotated[InfluenceMatrixContainer, Field(default=None)]
    physical_dose_var: Annotated[Optional[InfluenceMatrixContainer], Field(default=None)]
    let_dose: Annotated[Optional[InfluenceMatrixContainer], Field(default=None, alias="mLETDose")]
    alpha_dose: Annotated[Optional[InfluenceMatrixContainer], Field(default=None)]
    sqrt_beta_dose: Annotated[Optional[InfluenceMatrixContainer], Field(default=None)]

    num_of_beams: Annotated[int, Field(default=None)]

    bixel_num: Annotated[NDArray, Field(default=None)]
    ray_num: Annotated[NDArray, Field(default=None)]
    beam_num: Annotated[NDArray, Field(default=None)]

    alphax: Annotated[Optional[Array], Field(default=None)]
    betax: Annotated[Optional[Array], Field(default=None)]

    rad_depth_cubes: Optional[list[Array]] = Field(default=None)

    bio_model: Optional[Any] = Field(
        default=None,
        description="Biological model the alpha/beta influence matrices or the constant RBE "
        "stem from: None, a {'model': name, **parameters} dict or a BiologicalModel.",
    )

    @model_validator(mode="before")
    @classmethod
    def _legacy_constant_rbe(cls, data: Any) -> Any:
        """Fold a legacy scalar ``rbe`` / ``RBE`` (matRad) into a constant-RBE model."""
        if not isinstance(data, dict) or not ({"rbe", "RBE"} & set(data)):
            return data
        data = dict(data)
        values = [
            np.asarray(data.pop(key), dtype=float).ravel() for key in ("rbe", "RBE") if key in data
        ]
        # matRad writes an empty / zero placeholder when no constant RBE applies
        rbe = next((float(v[0]) for v in values if v.size == 1 and v[0] > 0), None)
        if rbe is not None and data.get("bio_model") is None and data.get("bioModel") is None:
            data["bio_model"] = {"model": "constant_rbe", "rbe": rbe}
        return data

    @field_validator("bio_model", mode="before")
    @classmethod
    def _validate_bio_model(cls, v: Any) -> Optional[BiologicalModel]:
        if v is None:
            return None
        return create_bio_model(v)

    @field_serializer("bio_model")
    def _serialize_bio_model(self, value: Any, info: SerializationInfo) -> Any:
        if not isinstance(value, BiologicalModel):
            return value
        if (info.context or {}).get("matRad"):
            return None  # matRad dijs only carry the constant RBE, see to_matrad
        return value.to_dict()

    @property
    def rbe(self) -> Optional[float]:
        """Constant RBE of the biological model, if it is a constant-RBE model."""
        if isinstance(self.bio_model, ConstantRBEModel):
            return self.bio_model.rbe
        return None

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
        potential_quantities = [
            "physical_dose",
            "physical_dose_var",
            "let_dose",
            "alpha_dose",
            "sqrt_beta_dose",
        ]
        return [q for q in potential_quantities if getattr(self, q) is not None]

    @field_validator(
        "physical_dose",
        "physical_dose_var",
        "let_dose",
        "alpha_dose",
        "sqrt_beta_dose",
        mode="wrap",
    )
    @classmethod
    def validate_influenc_matrix_conatiner(
        cls, v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
    ) -> InfluenceMatrixContainer:
        """
        Validate the physical dose matrix.

        Raises
        ------
            ValueError: if physical dose is not a 2D numpy array.
        """

        if v is None:
            return v

        if not (isinstance(v, np.ndarray) and v.dtype == np.dtype(object)) and (
            isinstance(v, (sp.spmatrix, sp.sparray, np.ndarray))
            or array_api_compat.is_array_api_obj(v)
        ):
            # is a numeric matrix, not a container
            # make it a container of one element
            v = np.array([v], dtype=object)
        elif isinstance(v, list):
            v = np.asarray(v, dtype=object)

        # Starting here, it should be a NDArray of objects
        v: InfluenceMatrixContainer = handler(v, info)  # run any other validators

        [_check_influence_matrix(v.flat[i], info) for i in range(v.size) if v.flat[i] is not None]

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

                    if array_api_compat.size(v) != bix_num:
                        raise ValueError(
                            "Numbering arrays shape inconsistent with number of bixels"
                        )

        if info.context and "from_matRad" in info.context and info.context["from_matRad"]:
            v -= 1
        return v

    @field_validator(
        "alphax",
        "betax",
        mode="before",
    )
    @classmethod
    def validate_voxel_arrays(cls, v: Any, info: ValidationInfo) -> np.ndarray:
        """
        Validate the voxel arrays.

        Raises
        ------
            ValueError: inconsistent voxel arrays.
        """
        if v is None:
            return v
        if not hasattr(v, "ndim"):
            v = np.asarray(v)
        # Voxel arrays carry one column per CT scenario; accept plain 1-D input
        # (e.g. matRad-imported alphaX/betaX) as a single scenario. Keep the array
        # namespace, since this also runs on assignment in to_namespace().
        if v.ndim == 1:
            v = array_api_compat.array_namespace(v).reshape(v, (-1, 1))
        if v.ndim != 2:
            raise ValueError("Voxel arrays must have shape (num_voxels, num_ct_scenarios)")
        # Check if the voxel arrays have the correct shape
        if info.data.get("physical_dose") is not None:
            dij_matrices = cast(np.ndarray, info.data["physical_dose"])
            for i in range(dij_matrices.size):
                if dij_matrices.flat[i] is not None:
                    mat = cast(Union[sp.spmatrix, sp.sparray, np.ndarray], dij_matrices.flat[i])

                    vox_num = mat.shape[0]

                    scen_num = dij_matrices.size

                    if v.shape[0] != vox_num:
                        raise ValueError("Voxel arrays shape inconsistent with number of voxels")
                    if v.shape[1] != scen_num:
                        raise ValueError(
                            "Voxel arrays shape inconsistent with number of scenarios"
                        )

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

    @field_serializer(
        "physical_dose",
        "physical_dose_var",
        "let_dose",
        "alpha_dose",
        "sqrt_beta_dose",
    )
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
        # return 0 if value is None. savemat() can't handle 'None'
        elif context and context.get("matRad") == "mat-file" and value is None:
            value = np.array([0])
        return value

    @field_serializer("rad_depth_cubes")
    def rad_depth_cubes_serializer(self, value: np.ndarray, info: SerializationInfo) -> np.ndarray:
        context = info.context
        if context and context.get("matRad") == "mat-file" and value is not None:
            # TODO: it might be necessary to rotate the cube!
            return value
        # return 0 if value is None. savemat() can't handle 'None'
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
        dij_dict.pop("bioModel", None)
        if self.rbe is not None:
            dij_dict["RBE"] = float(self.rbe)

        # Replace None values with np.array([0]) for savemat compatibility
        for key, value in dij_dict.items():
            if value is None:
                dij_dict[key] = np.array([0])

        return dij_dict

    #: Result quantities that are ratios / averages and do not scale with fractions.
    _INTENSIVE_RESULTS: ClassVar[frozenset[str]] = frozenset(
        {"let", "rbe", "alpha", "beta", "let_beam", "rbe_beam", "alpha_beam", "beta_beam"}
    )
    _SQRT_EXTENSIVE_RESULTS: ClassVar[frozenset[str]] = frozenset(
        {"sqrt_beta_dose", "sqrt_beta_dose_beam"}
    )

    @classmethod
    def _scale_result_to_fractions(cls, out: dict[str, Any], num_of_fractions: int) -> dict:
        """Scale per-fraction result arrays to ``num_of_fractions`` identical fractions."""
        if num_of_fractions == 1:
            return out
        for key, value in out.items():
            if key in cls._INTENSIVE_RESULTS:
                continue
            if key in cls._SQRT_EXTENSIVE_RESULTS:
                factor = num_of_fractions**0.5
            elif key.startswith("physical_dose_var"):
                factor = num_of_fractions**2
            else:
                factor = num_of_fractions
            if isinstance(value, list):
                out[key] = [factor * v for v in value]
            else:
                out[key] = factor * value
        return out

    def get_result_arrays_from_intensity(
        self, intensity: np.ndarray, scenario_index: int = 0, num_of_fractions: int = 1
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
        xp = array_api_compat.array_namespace(intensity)
        beam_num = xp.asarray(self.beam_num, device=array_api_compat.device(intensity))
        zero_intensity = xp.zeros_like(intensity)
        beam_intensities = [
            xp.where(beam_num == i, intensity, zero_intensity) for i in range(self.num_of_beams)
        ]

        # TODO: implement quantity system to select the corresponding quantities automatically
        if self.physical_dose is not None:
            dose_mat = self.physical_dose.flat[scenario_index]
            out["physical_dose"] = dose_mat @ intensity

            # Mask the fluence instead of selecting matrix columns. Integer-array
            # indexing is not part of the Python Array API and fails for strict
            # backends.
            out["physical_dose_beam"] = [
                dose_mat @ beam_intensity for beam_intensity in beam_intensities
            ]

        if self.physical_dose_var is not None:
            out["physical_dose_var"] = self.physical_dose_var.flat[scenario_index] @ intensity

        if self.let_dose is not None:
            if self.physical_dose is None:
                raise ValueError("Physical dose must be calculated for dose-weighted let")

            indices = out["physical_dose"] > 0.05 * xp.max(out["physical_dose"])

            let_mat = self.let_dose.flat[scenario_index]
            let_dose = let_mat @ intensity
            safe_dose = xp.where(indices, out["physical_dose"], xp.ones_like(let_dose))
            out["let"] = xp.where(indices, let_dose / safe_dose, xp.zeros_like(let_dose))

            let_dose_beams = [let_mat @ beam_intensity for beam_intensity in beam_intensities]
            out["let_beam"] = []
            for i, let_dose_beam in enumerate(let_dose_beams):
                phys_dose_beam = out["physical_dose_beam"][i]
                max_phys = xp.max(phys_dose_beam)
                indices_beam = (max_phys > 0) & (phys_dose_beam > 0.05 * max_phys)
                safe_dose_beam = xp.where(
                    indices_beam, phys_dose_beam, xp.ones_like(phys_dose_beam)
                )
                let_beam = xp.where(
                    indices_beam,
                    let_dose_beam / safe_dose_beam,
                    xp.zeros_like(let_dose_beam),
                )
                out["let_beam"].append(let_beam)

        # Lazy import: the quantities module depends on the dij, so importing at
        # module level would create a circular import.
        from pyRadPlan.quantities._rbe_x_dose import RBExDose, lq_inverse_dose  # noqa: PLC0415

        has_lq = self.alpha_dose is not None and self.sqrt_beta_dose is not None
        rbe_path = RBExDose.choose_path(self, has_effect=has_lq, strict=False)

        if has_lq:
            alphax = self.alphax[:, scenario_index]
            betax = self.betax[:, scenario_index]
            alpha_mat = self.alpha_dose.flat[scenario_index]
            sqrt_beta_mat = self.sqrt_beta_dose.flat[scenario_index]
            out["effect"] = alpha_mat @ intensity + (sqrt_beta_mat @ intensity) ** 2
            out["alpha_dose"] = alpha_mat @ intensity
            out["sqrt_beta_dose"] = sqrt_beta_mat @ intensity

            valid = out["physical_dose"] > 0
            phys = out["physical_dose"]
            safe_phys = xp.where(valid, phys, xp.ones_like(phys))
            out["alpha"] = xp.where(valid, out["alpha_dose"] / safe_phys, xp.zeros_like(phys))
            out["beta"] = xp.where(
                valid,
                (out["sqrt_beta_dose"] / safe_phys) ** 2,
                xp.zeros_like(phys),
            )

            out["effect_beam"] = []
            out["alpha_beam"] = []
            out["beta_beam"] = []
            out["alpha_dose_beam"] = []
            out["sqrt_beta_dose_beam"] = []
            for beam_intensity, phys_beam in zip(
                beam_intensities, out["physical_dose_beam"], strict=True
            ):
                alpha_dose_beam = alpha_mat @ beam_intensity
                sqrt_beta_dose_beam = sqrt_beta_mat @ beam_intensity
                mask_beam = phys_beam > 0
                denom_beam = xp.where(mask_beam, phys_beam, xp.ones_like(phys_beam))
                out["effect_beam"].append(alpha_dose_beam + sqrt_beta_dose_beam**2)
                out["alpha_dose_beam"].append(alpha_dose_beam)
                out["sqrt_beta_dose_beam"].append(sqrt_beta_dose_beam)
                out["alpha_beam"].append(
                    xp.where(
                        mask_beam,
                        alpha_dose_beam / denom_beam,
                        xp.zeros_like(phys_beam),
                    )
                )
                out["beta_beam"].append(
                    xp.where(
                        mask_beam,
                        (sqrt_beta_dose_beam / denom_beam) ** 2,
                        xp.zeros_like(phys_beam),
                    )
                )

        if rbe_path == "effect":
            phys = out["physical_dose"]
            valid = phys > 0
            out["rbe_x_dose"] = lq_inverse_dose(out["effect"], alphax, betax)
            safe_phys = xp.where(valid, phys, xp.ones_like(phys))
            out["rbe"] = xp.where(
                valid,
                out["rbe_x_dose"] / safe_phys,
                xp.zeros_like(phys),
            )
            out["rbe_x_dose_beam"] = [
                lq_inverse_dose(effect_beam, alphax, betax) for effect_beam in out["effect_beam"]
            ]
            out["rbe_beam"] = [
                xp.where(
                    phys_beam > 0,
                    rbe_x_beam / xp.where(phys_beam > 0, phys_beam, xp.ones_like(phys_beam)),
                    xp.zeros_like(phys_beam),
                )
                for rbe_x_beam, phys_beam in zip(out["rbe_x_dose_beam"], out["physical_dose_beam"])
            ]
        elif rbe_path == "constant":
            out["rbe_x_dose"] = self.rbe * out["physical_dose"]
            out["rbe_x_dose_beam"] = [
                self.rbe * phys_beam for phys_beam in out["physical_dose_beam"]
            ]

        return self._scale_result_to_fractions(out, num_of_fractions)

    def compute_result_dose_grid(
        self, intensities: np.ndarray, scenario_index: int = 0, num_of_fractions: int = 1
    ) -> dict[str, sitk.Image]:
        """
        Compute results on the dose grid from intensity vector.

        Parameters
        ----------
        intensity : np.ndarray
            The intensity to apply to the dose influence matrix.
        scenario_index : int
            The scenario index to apply the intensity to.
        num_of_fractions : int
            Report doses for this many identical fractions (1 = per-fraction dose of the
            influence matrix). See ``Plan.dose_convention`` / ``Plan.result_dose_factor``.

        Returns
        -------
        dict[str,sitk.Image]
            A dictionary containing the quantity images for each scenario.
        """

        out = self.get_result_arrays_from_intensity(
            intensities, scenario_index=scenario_index, num_of_fractions=num_of_fractions
        )
        # Create a sitk image for each scenario

        for key, value in out.items():
            # Create a sitk image for each scenario
            if isinstance(value, list):
                #  handle every single beam information
                for i in range(len(value)):
                    value[i] = sitk.GetImageFromArray(
                        value[i].reshape(self.dose_grid.dimensions[::-1])
                    )
                    value[i].SetOrigin(self.dose_grid.origin)
                    value[i].SetSpacing(self.dose_grid.resolution_vector)
                    value[i].SetDirection(self.dose_grid.direction.ravel())
            else:
                # handling collective of all beams
                out[key] = sitk.GetImageFromArray(value.reshape(self.dose_grid.dimensions[::-1]))
                out[key].SetOrigin(self.dose_grid.origin)
                out[key].SetSpacing(self.dose_grid.resolution_vector)
                out[key].SetDirection(self.dose_grid.direction.ravel())

        return out

    def compute_result_ct_grid(
        self, intensities: np.ndarray, scenario_index: int = 0, num_of_fractions: int = 1
    ) -> dict[str, sitk.Image]:
        """
        Compute results on the CT grid from intensity vector.

        Parameters
        ----------
        intensity : np.ndarray
            The intensity to apply to the dose influence matrix.
        scenario_index : int
            The scenario index to apply the intensity to.
        num_of_fractions : int
            Report doses for this many identical fractions (1 = per-fraction dose of the
            influence matrix). See ``Plan.dose_convention`` / ``Plan.result_dose_factor``.

        Returns
        -------
        dict[str,sitk.Image]
            A dictionary containing the quantity images for each scenario.
        """

        out = self.compute_result_dose_grid(
            intensities, scenario_index=scenario_index, num_of_fractions=num_of_fractions
        )
        # Create a sitk image for each scenario

        for key, value in out.items():
            # Create a sitk image for each scenario
            resampler = sitk.ResampleImageFilter()
            resampler.SetInterpolator(sitk.sitkLinear)
            resampler.SetOutputDirection(self.ct_grid.direction.ravel())
            resampler.SetOutputOrigin(self.ct_grid.origin)
            resampler.SetOutputSpacing(self.ct_grid.resolution_vector)
            resampler.SetSize(self.ct_grid.dimensions)

            if isinstance(value, list):
                #  handle every single beam information
                for i in range(len(value)):
                    out[key][i] = resampler.Execute(value[i])
            else:
                # handle the collective of all beams
                out[key] = resampler.Execute(value)
        return out

    def to_namespace(
        self,
        xp_new: Union[ArrayNamespace, str],
        *,
        keep_sparse_compat: bool = True,
        device: Optional[str] = None,
    ) -> Self:
        """
        Convert all influence matrices in the Dij to a different array namespace.

        Parameters
        ----------
        xp_new : ArrayNamespace
            The target array namespace.
        keep_sparse_compat : bool
            Whether to keep sparse matrix compatibility when converting to a new namespace.
            If False, sparse matrices will be converted to arrays of the namespace, even if the
            sparse format is compatible with the target namespace. For example, converting from
            scipy.sparse to numpy will result in a dense numpy array instead of a sparse matrix.
            Default is True.
        device : Optional[str]
            Target device for the arrays (e.g. "cuda:0", "cpu").
        """

        # record memory addresses here, before model_copy would break the sharing.
        _shared_idx_cache: dict[int, tuple | None] = {}
        if keep_sparse_compat:
            for q in self.quantities:
                q_container = getattr(self, q)
                if q_container is None:
                    continue
                for i in range(q_container.size):
                    mat = q_container.flat[i]
                    if mat is not None and isinstance(mat, (sp.csc_matrix, sp.csc_array)):
                        _shared_idx_cache.setdefault(mat.indices.ctypes.data, None)

            n_unique = len(_shared_idx_cache)
            n_total = sum(
                sum(
                    1
                    for idx in range(getattr(self, q).size)
                    if getattr(self, q).flat[idx] is not None
                    and isinstance(getattr(self, q).flat[idx], (sp.csc_matrix, sp.csc_array))
                )
                for q in self.quantities
                if getattr(self, q) is not None
            )
            if n_total > n_unique:
                ns_name = xp_new if isinstance(xp_new, str) else xp_new.__name__
                logger.debug(
                    "to_namespace: %d CSC matrices share %d unique row-index array(s) — "
                    "converting index arrays only once to namespace '%s'.",
                    n_total,
                    n_unique,
                    ns_name,
                )

        # shallow copy and then convert the matrices in-place to avoid unnecessary copying of large arrays.
        dij_copy = self.model_copy(deep=False)
        for _q in self.quantities:
            _src = getattr(self, _q)
            if _src is not None:
                _fresh = np.empty_like(_src)  # same shape, dtype=object
                _fresh.flat[:] = None
                object.__setattr__(dij_copy, _q, _fresh)

        for q in self.quantities:
            q_container: InfluenceMatrixContainer = getattr(self, q)
            if q_container is not None:
                for i in range(q_container.size):
                    mat = q_container.flat[i]
                    if mat is None:
                        continue

                    # scipy CSC matrices whose index arrays may be shared
                    if keep_sparse_compat and isinstance(mat, (sp.csc_matrix, sp.csc_array)):
                        addr = mat.indices.ctypes.data
                        if addr in _shared_idx_cache:
                            if _shared_idx_cache[addr] is None:
                                # First encounter for this index array: convert once
                                conv_indices = to_namespace(xp_new, mat.indices, device=device)
                                conv_indptr = to_namespace(xp_new, mat.indptr, device=device)
                                _shared_idx_cache[addr] = (conv_indices, conv_indptr)
                            conv_indices, conv_indptr = _shared_idx_cache[addr]
                            conv_data = to_namespace(xp_new, mat.data, device=device)
                            getattr(dij_copy, q).flat[i] = _rebuild_scipy_csc_in_namespace(
                                xp_new, conv_data, conv_indices, conv_indptr, mat.shape
                            )
                            continue

                    getattr(dij_copy, q).flat[i] = to_namespace(
                        xp_new, mat, keep_sparse_compat=keep_sparse_compat, device=device
                    )

        if self.alphax is not None:
            dij_copy.alphax = to_namespace(xp_new, self.alphax, device=device)
        if self.betax is not None:
            dij_copy.betax = to_namespace(xp_new, self.betax, device=device)
        name = xp_new.__name__ if not isinstance(xp_new, str) else xp_new

        logger.info(f"Converted Dij to namespace '{name}'")

        return dij_copy


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

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
from pyRadPlan.biology.models import BiologicalModel, ConstantRBEModel, create_bio_model
from pyRadPlan.scenarios import ScenarioModel, ScenarioSet
from pyRadPlan.util import swap_orientation_sparse_matrix

from ..core.xp_utils import to_namespace
from ..core.xp_utils.helpers import _rebuild_scipy_csc_in_namespace

InfluenceMatrixArray = Union[Array, sp.spmatrix, sp.sparray]
InfluenceMatrixContainer = NDArray[Shape["*, ..."], object]

_QUANTITY_FIELDS: tuple[str, ...] = (
    "physical_dose",
    "physical_dose_var",
    "let_dose",
    "alpha_dose",
    "sqrt_beta_dose",
)
_VOXEL_FIELDS: tuple[str, ...] = ("alphax", "betax", "tissue_class_ix")


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


def _as_scenario_set(scenarios: Any) -> Optional[ScenarioSet]:
    """Coerce a scenario model, set or serialized set to a :class:`ScenarioSet`."""
    if scenarios is None or isinstance(scenarios, ScenarioSet):
        return scenarios
    if isinstance(scenarios, ScenarioModel):
        return scenarios.scenarios()
    if isinstance(scenarios, dict):
        return ScenarioSet.model_validate(scenarios)
    raise ValueError(f"Cannot interpret {type(scenarios).__name__} as a scenario set")


def _is_empty_cell(entry: Any) -> bool:
    """Whether a matRad cell entry is empty (``[]``, a scalar placeholder or ``None``)."""
    if entry is None:
        return True
    if isinstance(entry, (sp.spmatrix, sp.sparray)):
        return False
    shape = getattr(entry, "shape", None)
    if shape is None:
        return np.ndim(entry) != 2
    return len(shape) != 2 or int(np.prod(shape)) == 0


def _is_matrad_placeholder(value: Any) -> bool:
    """Whether ``value`` is the numeric scalar matRad export writes in place of ``None``."""
    if isinstance(value, (int, float, np.number)):
        return True
    return isinstance(value, np.ndarray) and value.dtype != np.dtype(object) and value.size <= 1


def _nested_to_object_array(value: Any) -> np.ndarray:
    """Turn (nested) lists of cell entries, e.g. from pymatreader, into an object array."""
    if isinstance(value, np.ndarray) and value.dtype == np.dtype(object):
        return value
    if not isinstance(value, (list, tuple)):
        container = np.empty((1,), dtype=object)
        container[0] = value
        return container
    shape = []
    level = value
    while isinstance(level, (list, tuple)):
        shape.append(len(level))
        if len(level) == 0:
            break
        level = level[0]
    cells = np.empty(tuple(shape), dtype=object)
    for index in np.ndindex(cells.shape):
        entry = value
        for k in index:
            entry = entry[k]
        cells[index] = entry
    return cells


def _matrad_cells_to_flat(value: Any, scenarios: Optional[ScenarioSet]) -> np.ndarray:
    """
    Flatten a matRad ``(ct, shift, range)`` cell array into a flat scenario container.

    Without ``scenarios``, the non-empty cells are taken in MATLAB (column-major) order. With
    a set, every computed scenario takes the cell at its position in the set's matRad layout,
    so the cell array may be squeezed (as pymatreader loads it).
    """
    cells = _nested_to_object_array(value)
    if scenarios is None:
        entries = [c for c in cells.flatten(order="F") if not _is_empty_cell(c)]
        flat = np.empty((len(entries),), dtype=object)
        for i, entry in enumerate(entries):
            flat[i] = entry
        return flat

    shape, position = scenarios.matrad_layout()
    if cells.shape != shape:
        squeezed = tuple(d for d in shape if d != 1)
        if tuple(d for d in cells.shape if d != 1) != squeezed:
            raise ValueError(
                f"matRad cell array of shape {cells.shape} does not match the scenario layout "
                f"{shape}"
            )
        cells = cells.reshape(shape)
    flat = np.empty((len(scenarios),), dtype=object)
    used = 0
    for i in range(len(scenarios)):
        if scenarios.bases[i] is not None:
            continue
        entry = cells[tuple(position[i])]
        if _is_empty_cell(entry):
            raise ValueError(f"matRad cell of scenario {i} at {tuple(position[i])} is empty")
        flat[i] = entry
        used += 1
    num_filled = sum(not _is_empty_cell(c) for c in cells.flat)
    if num_filled != used:
        raise ValueError(
            f"matRad cell array holds {num_filled} matrices, but the scenario set computes {used}"
        )
    return flat


def _flat_to_matrad_cells(container: np.ndarray, scenarios: ScenarioSet) -> np.ndarray:
    """Arrange a flat container on matRad's ``(ct, shift, range)`` cells, empty off-mask."""
    shape, position = scenarios.matrad_layout()
    cells = np.empty(shape, dtype=object)
    for i, entry in enumerate(container.flat):
        cells[tuple(position[i])] = entry
    for index in np.ndindex(shape):
        if cells[index] is None:
            cells[index] = np.zeros((0, 0))
    return cells


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
    scenarios : ScenarioSet, optional
        Scenario set the influence matrices were computed for. Scenario ``i`` of the set lives
        in ``container.flat[i]`` of every quantity container, and the per-voxel parameter arrays
        (``alphax``, ``betax``, ``tissue_class_ix``) carry one column per voxel parameter set
        of the set. ``None`` keeps the implicit layout: one slot per container entry and one
        column per slot (or a single shared column).
    """

    dose_grid: Annotated[Grid, Field(default=None)]
    ct_grid: Annotated[Grid, Field(default=None)]

    scenarios: Optional[ScenarioSet] = Field(
        default=None,
        description="Scenario set addressing the flat container slots (None: implicit).",
    )

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
    tissue_class_ix: Annotated[Optional[Array], Field(default=None)]

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

    @model_validator(mode="before")
    @classmethod
    def _drop_matrad_placeholders(cls, data: Any, info: ValidationInfo) -> Any:
        """Drop the scalar placeholders matRad export writes for absent optional fields."""
        if not isinstance(data, dict) or not (info.context or {}).get("from_matRad"):
            return data
        optional = (*_QUANTITY_FIELDS[1:], *_VOXEL_FIELDS, "rad_depth_cubes")
        keys = {name for f in optional for name in (f, cls.model_fields[f].alias) if name}
        return {k: v for k, v in data.items() if not (k in keys and _is_matrad_placeholder(v))}

    @field_validator("bio_model", mode="before")
    @classmethod
    def _validate_bio_model(cls, v: Any) -> Optional[BiologicalModel]:
        if v is None:
            return None
        return create_bio_model(v)

    @field_validator("scenarios", mode="before")
    @classmethod
    def _validate_scenarios(cls, v: Any) -> Optional[ScenarioSet]:
        return _as_scenario_set(v)

    @field_serializer("scenarios")
    def _serialize_scenarios(self, value: Optional[ScenarioSet], info: SerializationInfo) -> Any:
        if value is None or (info.context or {}).get("matRad"):
            return None
        return value.model_dump(mode=info.mode)

    @model_validator(mode="after")
    def _check_scenarios(self, info: ValidationInfo) -> Self:
        """Attach a scenario set from the validation context and check the containers."""
        context = info.context or {}
        if self.scenarios is None and context.get("scenario_model") is not None:
            object.__setattr__(self, "scenarios", _as_scenario_set(context["scenario_model"]))
        if self.scenarios is not None:
            self._check_scenario_consistency(self.scenarios)
        return self

    def _check_scenario_consistency(self, scenarios: ScenarioSet) -> None:
        """Check container sizes, derived slots and voxel parameter columns against a set."""
        n = len(scenarios)
        for q in _QUANTITY_FIELDS:
            container = getattr(self, q)
            if container is None:
                continue
            if container.size != n:
                raise ValueError(
                    f"{q} holds {container.size} scenario slot(s), the scenario set has {n}"
                )
            if all(entry is None for entry in container.flat):
                # Placeholder of an external calculation: nothing to check per slot
                continue
            for i, basis in enumerate(scenarios.bases):
                entry = container.flat[i]
                if basis is None:
                    if entry is None:
                        raise ValueError(f"{q} has no matrix for computed scenario {i}")
                elif basis.is_alias:
                    ref = container.flat[next(iter(basis.coefficients))]
                    if entry is None:
                        container.flat[i] = ref
                    elif entry is not ref:
                        raise ValueError(
                            f"{q} slot {i} must share the matrix of the scenario its alias "
                            "basis references (same object) or be empty"
                        )
                elif entry is not None:
                    raise ValueError(
                        f"{q} slot {i} belongs to a derived scenario and must be empty"
                    )
        num_cols = scenarios.num_voxel_param_sets
        for field in _VOXEL_FIELDS:
            arr = getattr(self, field)
            if arr is not None and arr.shape[1] != num_cols:
                raise ValueError(
                    f"{field} has {arr.shape[1]} column(s), but the scenario set has "
                    f"{num_cols} voxel parameter set(s)"
                )

    @computed_field
    @property
    def num_of_scenarios(self) -> int:
        """Number of scenario slots (the set's length, else the container size)."""
        if self.scenarios is not None:
            return len(self.scenarios)
        for q in _QUANTITY_FIELDS:
            container = getattr(self, q)
            if container is not None:
                return int(container.size)
        return 0

    def voxel_param_column(self, scenario_index: int) -> int:
        """
        Column of the per-voxel parameter arrays used by a scenario.

        With a scenario set, this is :meth:`ScenarioSet.voxel_param_set`. Without, scenario
        ``i`` uses column ``i``, clipped to the last column so that a single column is shared
        by all scenarios.
        """
        if self.scenarios is not None:
            return self.scenarios.voxel_param_set(int(scenario_index))
        for field in _VOXEL_FIELDS:
            arr = getattr(self, field)
            if arr is not None:
                return min(int(scenario_index), int(arr.shape[1]) - 1)
        return 0

    def attach_scenarios(self, scenarios: Union[ScenarioSet, ScenarioModel]) -> Self:
        """
        Attach a scenario set (or a model's set) to this Dij after construction.

        The containers must hold one slot per scenario and the per-voxel parameter arrays one
        column per voxel parameter set. Returns ``self``.
        """
        scenarios = _as_scenario_set(scenarios)
        if scenarios is not None:
            self._check_scenario_consistency(scenarios)
        self.scenarios = scenarios
        return self

    def extend_scenarios(self, other: "Dij") -> "Dij":
        """
        Concatenate the scenarios of two Dijs computed for the same plan geometry.

        Both Dijs need a scenario set over the same dimensions, identical grids, bixel
        bookkeeping and quantities. The containers of ``other`` are appended (its scenario
        indices shift by ``len(self.scenarios)``), the sets are merged with
        :meth:`ScenarioSet.extend` and the per-voxel parameter columns are rebuilt for the
        voxel parameter sets of the merged set. Matrices are shared, not copied.

        Parameters
        ----------
        other : Dij
            Dij whose scenarios are appended.

        Returns
        -------
        Dij
            New Dij holding the scenarios of both.
        """
        if self.scenarios is None or other.scenarios is None:
            raise ValueError("extend_scenarios needs a scenario set on both Dijs")
        if self.dose_grid != other.dose_grid or self.ct_grid != other.ct_grid:
            raise ValueError("Dijs to extend must have identical dose and CT grids")
        if self.num_of_beams != other.num_of_beams or any(
            not np.array_equal(np.asarray(getattr(self, f)), np.asarray(getattr(other, f)))
            for f in ("bixel_num", "ray_num", "beam_num")
        ):
            raise ValueError("Dijs to extend must have identical bixel bookkeeping")
        if set(self.quantities) != set(other.quantities):
            raise ValueError(
                f"Dijs to extend must carry the same quantities, got {self.quantities} and "
                f"{other.quantities}"
            )
        for field in _VOXEL_FIELDS:
            if (getattr(self, field) is None) != (getattr(other, field) is None):
                raise ValueError(f"{field} is present on only one of the Dijs to extend")

        merged = self.scenarios.extend(other.scenarios)
        n_self = len(self.scenarios)

        data: dict[str, Any] = {
            "ct_grid": self.ct_grid,
            "dose_grid": self.dose_grid,
            "scenarios": merged,
            "num_of_beams": self.num_of_beams,
            "bixel_num": np.array(self.bixel_num, copy=True),
            "ray_num": np.array(self.ray_num, copy=True),
            "beam_num": np.array(self.beam_num, copy=True),
            "bio_model": self.bio_model,
        }
        for q in self.quantities:
            container = np.empty((len(merged),), dtype=object)
            for i, entry in enumerate(getattr(self, q).flat):
                container[i] = entry
            for i, entry in enumerate(getattr(other, q).flat):
                container[n_self + i] = entry
            data[q] = container

        for field in _VOXEL_FIELDS:
            if getattr(self, field) is not None:
                data[field] = self._merged_voxel_columns(other, field, merged, n_self)

        return Dij.model_validate(data)

    def _merged_voxel_columns(
        self, other: "Dij", field: str, merged: ScenarioSet, n_self: int
    ) -> Array:
        """Per-voxel parameter columns of ``field`` for the voxel parameter sets of ``merged``."""
        own, theirs = getattr(self, field), getattr(other, field)
        xp = array_api_compat.array_namespace(own, theirs)
        columns = []
        for _, indices in merged.voxel_param_sets():
            sources = [
                own[:, self.voxel_param_column(i)]
                if i < n_self
                else theirs[:, other.voxel_param_column(i - n_self)]
                for i in indices
            ]
            if any(not bool(xp.all(src == sources[0])) for src in sources[1:]):
                raise ValueError(
                    f"Scenarios {indices} share their voxel parameter realization, but their "
                    f"{field} columns differ"
                )
            columns.append(sources[0])
        return xp.stack(columns, axis=1)

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

        from_matrad = bool(info.context and info.context.get("from_matRad"))
        if from_matrad and (
            isinstance(v, (list, tuple))
            or (isinstance(v, np.ndarray) and v.dtype == np.dtype(object))
        ):
            v = _matrad_cells_to_flat(
                v, _as_scenario_set((info.context or {}).get("scenario_model"))
            )

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
        v: InfluenceMatrixContainer = handler(v)  # run any other validators

        [_check_influence_matrix(v.flat[i], info) for i in range(v.size) if v.flat[i] is not None]

        if from_matrad:
            if v is not None:
                for i in range(v.size):
                    if v.flat[i] is None:
                        continue
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
        "tissue_class_ix",
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
        # Accept plain 1-D input (e.g. matRad-imported alphaX/betaX) as a single column. Keep
        # the array namespace, since this also runs on assignment in to_namespace().
        if v.ndim == 1:
            v = array_api_compat.array_namespace(v).reshape(v, (-1, 1))
        if v.ndim != 2:
            raise ValueError("Voxel arrays must have shape (num_voxels, num_voxel_param_sets)")
        dij_matrices = info.data.get("physical_dose")
        if dij_matrices is not None:
            dij_matrices = cast(np.ndarray, dij_matrices)
            mats = [m for m in dij_matrices.flat if m is not None]
            if mats and v.shape[0] != mats[0].shape[0]:
                raise ValueError("Voxel arrays shape inconsistent with number of voxels")
            # With a scenario set, the column count is checked against its voxel parameter
            # sets in the model validator.
            if info.data.get("scenarios") is None and v.shape[1] not in (1, dij_matrices.size):
                raise ValueError(
                    "Voxel arrays need one column or one column per scenario slot "
                    f"({dij_matrices.size}), got {v.shape[1]}: inconsistent with number of "
                    "scenarios"
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
        return handler(value)

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
            if self.scenarios is not None:
                self.scenarios.require_matrad_compatible(allow_alias=True)
            shape = (
                int(self.dose_grid.dimensions[2]),
                int(self.dose_grid.dimensions[0]),
                int(self.dose_grid.dimensions[1]),
            )
            swapped: dict[int, Any] = {}
            for i in range(value.size):
                mat = value.flat[i]
                if mat is None:
                    continue
                if id(mat) not in swapped:
                    new = swap_orientation_sparse_matrix(mat, shape, (1, 2))  # (65, 100, 100)
                    if not isinstance(new, sp.csc_matrix):
                        new = sp.csc_matrix(new)
                    swapped[id(mat)] = new
                value.flat[i] = swapped[id(mat)]
            if self.scenarios is not None:
                value = _flat_to_matrad_cells(value, self.scenarios)
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
        dij_dict.pop("scenarios", None)
        if self.rbe is not None:
            dij_dict["RBE"] = float(self.rbe)

        # Replace None values with np.array([0]) for savemat compatibility
        for key, value in dij_dict.items():
            if value is None:
                dij_dict[key] = np.array([0])

        return dij_dict

    #: Fraction exponents (see ``RTQuantity.fraction_exponent``) of result arrays that are not
    #: quantities; all other results scale like the quantity they are named after.
    _RESULT_FRACTION_EXPONENTS: ClassVar[dict[str, float]] = {
        "rbe": 0.0,
        "alpha": 0.0,
        "beta": 0.0,
        "physical_dose_var": 2.0,
    }

    @classmethod
    def _scale_result_to_fractions(cls, out: dict[str, Any], num_of_fractions: int) -> dict:
        """Scale per-fraction result arrays to ``num_of_fractions`` identical fractions."""
        if num_of_fractions == 1:
            return out
        # Lazy import: the quantities module depends on the dij.
        from pyRadPlan.quantities import get_quantity  # noqa: PLC0415

        for key, value in out.items():
            name = key.removesuffix("_beam")
            if name in cls._RESULT_FRACTION_EXPONENTS:
                factor = num_of_fractions ** cls._RESULT_FRACTION_EXPONENTS[name]
            else:
                factor = get_quantity(name).total_course_factor(num_of_fractions)
            if factor == 1:
                continue
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

        if self.scenarios is not None:
            basis = self.scenarios[scenario_index].basis
            if basis is not None and not basis.is_alias:
                raise ValueError(
                    f"Scenario {scenario_index} is derived from a linear combination and has no "
                    "influence matrices; evaluate it through the quantities instead"
                )

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
            column = self.voxel_param_column(scenario_index)
            alphax = self.alphax[:, column]
            betax = self.betax[:, column]
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

        # Scenarios sharing a matrix (alias bases) keep sharing the converted one.
        converted: dict[int, Any] = {}
        for q in self.quantities:
            q_container: InfluenceMatrixContainer = getattr(self, q)
            if q_container is not None:
                for i in range(q_container.size):
                    mat = q_container.flat[i]
                    if mat is None:
                        continue
                    if id(mat) in converted:
                        getattr(dij_copy, q).flat[i] = converted[id(mat)]
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
                            converted[id(mat)] = getattr(dij_copy, q).flat[i]
                            continue

                    getattr(dij_copy, q).flat[i] = to_namespace(
                        xp_new, mat, keep_sparse_compat=keep_sparse_compat, device=device
                    )
                    converted[id(mat)] = getattr(dij_copy, q).flat[i]

        if self.alphax is not None:
            dij_copy.alphax = to_namespace(xp_new, self.alphax, device=device)
        if self.betax is not None:
            dij_copy.betax = to_namespace(xp_new, self.betax, device=device)
        if self.tissue_class_ix is not None:
            dij_copy.tissue_class_ix = to_namespace(xp_new, self.tissue_class_ix, device=device)
        name = xp_new.__name__ if not isinstance(xp_new, str) else xp_new

        logger.info(f"Converted Dij to namespace '{name}'")

        return dij_copy


def create_dij(
    data: Union[dict[str, Any], Dij, None] = None,
    *,
    scenario_model: Union[ScenarioModel, ScenarioSet, None] = None,
    **kwargs,
) -> Dij:
    """
    Create a Dij object from raw data or keyword arguments.

    Parameters
    ----------
    data : Union[dict[str, Any], Dij, None]
        Dictionary containing the data to create the Dij object.
    scenario_model : ScenarioModel or ScenarioSet, optional
        Scenario model / set the data was computed for. Its set is attached to the Dij and, for
        matRad data, used to place the ``(ct, shift, range)`` cells on the flat containers.
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
            if scenario_model is not None and data.scenarios is None:
                data.attach_scenarios(scenario_model)
            return data

        if "beamNum" in data and np.min(data["beamNum"]) != 0:
            # add context when from matRad
            context = {"from_matRad": True}
        else:
            context = {"from_matRad": False}
        if scenario_model is not None:
            context["scenario_model"] = scenario_model
        return Dij.model_validate(data, context=context)

    if scenario_model is not None:
        return Dij.model_validate(kwargs, context={"scenario_model": scenario_model})
    return Dij(**kwargs)


def validate_dij(
    dij: Union[dict[str, Any], Dij, None] = None,
    *,
    scenario_model: Union[ScenarioModel, ScenarioSet, None] = None,
    **kwargs,
) -> Dij:
    """
    Validate and creates a Dij object.

    Synonym to create_dij but should be used in validation context.

    Parameters
    ----------
    dij : Union[dict[str, Any], Dij, None], optional
        Dictionary containing the data to create the Dij object, by default None.
    scenario_model : ScenarioModel or ScenarioSet, optional
        See :func:`create_dij`.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    Dij
        A validated Dij object.

    """
    return create_dij(dij, scenario_model=scenario_model, **kwargs)

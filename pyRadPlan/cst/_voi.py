from abc import ABC
from typing import Any, Optional, Union
from typing_extensions import Annotated, Self
import warnings
from pydantic import (
    Field,
    field_validator,
    model_validator,
    computed_field,
    StringConstraints,
)

import numpy as np
import SimpleITK as sitk
import matplotlib.colors as mcolors

from pyRadPlan.core import PyRadPlanBaseModel, np2sitk
from pyRadPlan.ct import CT
from pyRadPlan.core import Grid

# Default overlap priorities
DEFAULT_OVERLAPS = {"TARGET": 0, "OAR": 5, "HELPER": 10, "EXTERNAL": 15}

# Preferred colors per VOI type (RGB 0..255), tried in order before HSV fallback.
DEFAULT_VOI_COLORS: dict[str, list[tuple[int, int, int]]] = {
    "TARGET": [
        (255, 80, 80),
        (200, 0, 0),
        (255, 120, 120),
        (180, 0, 80),
        (255, 50, 150),
    ],
    "OAR": [
        (80, 140, 255),
        (0, 80, 200),
        (120, 180, 255),
        (0, 200, 200),
        (0, 150, 180),
    ],
    "EXTERNAL": [
        (100, 220, 100),
        (0, 180, 0),
        (150, 255, 150),
    ],
    "HELPER": [
        (255, 200, 60),
        (255, 140, 0),
        (230, 180, 50),
    ],
}


class VOI(PyRadPlanBaseModel, ABC):
    """
    Represents a Volume of Interest (VOI).

    Parameters
    ----------
    name : str
        The name of the VOI.
    grid : Grid, optional
        The grid for the VOI. Defaults to a new Grid instance.
    mask : np.ndarray or sitk.Image
        Boolean mask (using 0,1) for referencing of voxels
        (Multiple allocations possible for robust scenarios)
    alpha_x : float, optional
        The alpha_x value. Defaults to 0.1.
    beta_x : float, optional
        The beta_x value. Defaults to 0.05.
    overlap_priority : int
        The overlap priority of the VOI. Lowest number is overlapping higher numbers.
    """

    name: str
    grid: Grid
    mask: sitk.Image
    alpha_x: float = Field(default=0.1)
    beta_x: float = Field(default=0.05)
    voi_type: Annotated[str, StringConstraints(strip_whitespace=True, to_upper=True)]

    overlap_priority: int = Field(
        alias="Priority", default_factory=lambda data: DEFAULT_OVERLAPS[data["voi_type"]]
    )

    visible: bool = Field(default=True, description="Flag to set visibility in GUI applications")
    visible_color: Union[tuple[int, int, int], None] = Field(
        default=None, description="RGB color for visualization in GUI applications"
    )
    default_color: tuple[int, int, int] = Field(
        default_factory=lambda data: DEFAULT_VOI_COLORS[data["voi_type"]][0],
        description="Default RGB color bound to the VOI type",
    )

    # Annotating list[Objective] would create a circular import, so entries are converted to
    # Objective instances lazily in the validator below.
    objectives: list[Any] = Field(default=[], description="List of objective function definitions")

    @field_validator("objectives", mode="before")
    @classmethod
    def validate_objectives(cls, v: Any) -> Any:
        """
        Convert objective definitions (names or dictionaries) into Objective instances.

        Entries that are not recognizable objective definitions (e.g. empty placeholders from
        matRad imports) are passed through unchanged.

        Parameters
        ----------
        v : Any
            The objectives value to be validated.

        Returns
        -------
        list
            The objectives with recognizable definitions converted to Objective instances.
        """

        # deferred import to avoid circular import issues
        from pyRadPlan.optimization.objectives import get_objective  # noqa: PLC0415

        if not isinstance(v, list):
            v = [v]

        return [
            get_objective(entry)
            if isinstance(entry, dict) and ("name" in entry or "className" in entry)
            else entry
            for entry in v
        ]

    @model_validator(mode="before")
    @classmethod
    def validate_inputs(cls, v: Any) -> Any:
        """
        Validate the inputs and handle CT/Grid conversion.

        Parameters
        ----------
        v : Any
            The input data dictionary to be validated.

        Returns
        -------
        Any
            The validated input data with proper Grid/CT handling.
        """
        if not isinstance(v, dict):
            return v

        # Check the grid sources Grid/CT/SimpleITK
        grid_sources = cls._detect_grid_sources(v)

        # Check for corresponding grid sources and handle them
        # We copy and delete so that pydantic will work as intended.
        # It works without copy/del but might be troubling when editing the cst in the future!

        if grid_sources["count"] > 1:
            return cls._handle_multiple_grid_sources(v, grid_sources)
        elif grid_sources["has_ct"]:
            return cls._convert_ct_to_grid(v, grid_sources["ct_key"])
        elif grid_sources["has_sitk_image"]:
            return cls._convert_sitk_to_grid(v, grid_sources["sitk_image_key"])

        # If no grid sources are provided, create Grid from mask dimensions
        if "mask" in v and "grid" not in v:
            v_copy = v.copy()
            v_copy["grid"] = cls._create_grid_from_mask(v["mask"])
            return v_copy

        # Only Grid / no modification needed:
        return v

    @classmethod
    def _detect_grid_sources(cls, v: dict) -> dict:
        """Detect available grid sources in the input data."""
        sources = {
            "has_ct": False,
            "has_grid": False,
            "has_sitk_image": False,
            "ct_key": None,
            "sitk_image_key": None,
            "count": 0,
        }

        for key, value in v.items():
            if isinstance(value, CT):
                sources["has_ct"] = True
                sources["ct_key"] = key
                sources["count"] += 1
            elif isinstance(value, Grid):
                sources["has_grid"] = True
                sources["count"] += 1
            # make sure we handle sitk too without using "mask" by accident
            elif isinstance(value, sitk.Image) and key in ["grid", "image", "ct_image"]:
                sources["has_sitk_image"] = True
                sources["sitk_image_key"] = key
                sources["count"] += 1

        return sources

    @classmethod
    def _handle_multiple_grid_sources(cls, v: dict, sources: dict) -> dict:
        """Handle cases where multiple grid sources are provided."""
        v_copy = v.copy()

        if sources["has_grid"]:
            # Keep Grid, remove others
            if sources["has_ct"]:
                del v_copy[sources["ct_key"]]
            if sources["has_sitk_image"]:
                del v_copy[sources["sitk_image_key"]]
            warnings.warn(
                "Multiple grid sources provided (Grid, CT, and/or SimpleITK image). Using Grid and ignoring others...",
                UserWarning,
            )
        elif sources["has_ct"]:
            # Keep CT (convert to Grid), remove SimpleITK image
            if sources["has_sitk_image"]:
                del v_copy[sources["sitk_image_key"]]
            v_copy = cls._convert_ct_to_grid(v_copy, sources["ct_key"])
            warnings.warn(
                "Both CT and SimpleITK image provided. Using CT and ignoring SimpleITK image.",
                UserWarning,
            )

        return v_copy

    @classmethod
    def _convert_ct_to_grid(cls, v: dict, ct_key: str) -> dict:
        """Convert CT to Grid."""
        v_copy = v.copy()
        v_copy["grid"] = Grid.from_sitk_image(v_copy[ct_key].cube_hu)

        if ct_key != "grid":
            del v_copy[ct_key]
        return v_copy

    @classmethod
    def _convert_sitk_to_grid(cls, v: dict, sitk_key: str) -> dict:
        """Convert SimpleITK image to Grid."""
        v_copy = v.copy()
        v_copy["grid"] = Grid.from_sitk_image(v_copy[sitk_key])

        if sitk_key != "grid":
            del v_copy[sitk_key]
        return v_copy

    @classmethod
    def _create_grid_from_mask(cls, mask) -> Grid:
        """Create a default Grid from mask dimensions."""

        # Handle different mask types
        if hasattr(mask, "shape"):  # numpy
            return Grid.from_numpy_array(mask)
        elif hasattr(mask, "GetSize"):  # SimpleITK
            return Grid.from_sitk_image(mask)
        else:
            # Fallback for unknown mask types
            raise ValueError(
                f"Unsupported mask type: {type(mask)}. Expected numpy array or SimpleITK image."
            )

    @field_validator("grid", mode="before")
    @classmethod
    def validate_grid(cls, v: Any) -> Any:
        """
        Validate the grid type.

        Parameters
        ----------
        v : Any
            The grid value to be validated.

        Returns
        -------
        Grid
        """
        if isinstance(v, Grid):
            return v
        if isinstance(v, dict):
            return Grid.model_validate(v)
        raise ValueError("grid must be an instance of Grid")

    @field_validator("mask", mode="before")
    @classmethod
    def validate_mask_type(cls, v: Any) -> Any:
        """
        Validate the mask type.

        Parameters
        ----------
        v : Any
            The mask value to be validated.

        Returns
        -------
        sitk.Image
            The validated mask.

        Raises
        ------
        ValueError
            If the mask type is not supported.
        """
        if isinstance(v, np.ndarray):
            if v.dtype in ["bool", "int"]:
                v = v.astype("uint8")
            if v.dtype != "uint8":
                raise ValueError(
                    f"{v.dtype} is not supported for index mask. Please use uint8 or boolean mask."
                )

            if v.ndim == 3:
                return sitk.GetImageFromArray(v, False)

            if v.ndim == 4:
                mask = []
                for i in range(v.shape[0]):
                    mask.append(sitk.GetImageFromArray(v[i], False))
                v = sitk.JoinSeries(mask)
                return v

            raise ValueError("Dimensionality not supported!")

        if isinstance(v, sitk.Image):
            if sitk.GetArrayViewFromImage(v).dtype != "uint8":
                raise ValueError(
                    f"""{sitk.GetArrayViewFromImage(v).dtype} is not supported
                      for index mask. Please use uint8."""
                )
            return v

        raise ValueError("mask must be either passed as numpy array or SimpleITK image")

    @field_validator("visible_color", mode="before")
    @classmethod
    def validate_visible_color(cls, v: Any) -> Any:
        """
        Validate the visible color.

        Parameters
        ----------
        v : Any
            The visible color value to be validated.

        Returns
        -------
        tuple[int, int, int]
            The validated visible color.
        """

        if isinstance(v, str):
            # convert color to rgb tuple
            rgb = mcolors.to_rgb(v)
            return tuple(int(round(c * 255)) for c in rgb)

        # Accept array-like inputs, handle scaling and conversion
        if isinstance(v, (tuple, list, np.ndarray)):
            arr = np.asarray(v)
            if arr.size == 3 and np.issubdtype(arr.dtype, np.number):
                if np.issubdtype(arr.dtype, np.floating):
                    arr = np.round(arr * 255)
                return tuple(arr.astype(int).tolist())

        return v

    @model_validator(mode="after")
    def validate_mask(self):
        """
        Check if the given indices are valid for the CT image.

        Raises
        ------
        ValueError
            If the mask is not a sitk.Image.
        ValueError
            If the dimensions of the mask do not match the CT image.
        """
        if not isinstance(self.mask, sitk.Image):
            raise ValueError("Sanity check failed - mask is not a SimpleITK image")

        # check dimensions of sitk image
        dims = self.mask.GetSize()
        if dims != self.grid.dimensions:
            raise ValueError(
                f"Mask provided with dimensions {dims}, "
                f"but grid has dimensions {self.grid.dimensions}"
            )

        self.mask.SetOrigin(self.grid.origin)
        self.mask.SetSpacing(self.grid.resolution_vector)
        self.mask.SetDirection(self.grid.direction_vector)

        return self

    @computed_field
    @property
    def indices(self) -> np.ndarray:
        """
        Return linear voxel indices into the full mask cube (sitk / Fortran order).

        The indices reference the full cube as stored, regardless of dimensionality.

        - For a 3D mask of sitk size ``(X, Y, Z)`` the linear index is
          ``idx = z + Z * y + Z * Y * x`` (Z varies fastest, X slowest).
        - For a 4D mask of sitk size ``(X, Y, Z, T)`` the linear index is
          ``idx = t + T * z + T * Z * y + T * Z * Y * x`` (T varies fastest).
          Indices from different scenarios are therefore *interleaved* in the
          returned array. Use :meth:`scenario_indices` to obtain indices into a
          single 3D scenario sub-cube.

        Returns
        -------
        np.ndarray
            1D array of linear voxel indices into the full mask cube.
        """
        return np2sitk.sitk_mask_to_linear_indices(self.mask, order="sitk")

    @computed_field
    @property
    def indices_numpy(self) -> np.ndarray:
        """
        Return linear voxel indices into the full mask cube (numpy / C order).

        The indices reference the full cube as stored, regardless of dimensionality.

        - For a 3D mask of numpy shape ``(Z, Y, X)`` the linear index is
          ``idx = x + X * y + X * Y * z`` (X varies fastest, Z slowest).
        - For a 4D mask of numpy shape ``(T, Z, Y, X)`` the linear index is
          ``idx = x + X * y + X * Y * z + X * Y * Z * t`` (X varies fastest, T
          slowest). With this convention each scenario occupies a contiguous
          block of size ``X * Y * Z`` within the returned array. Use
          :meth:`scenario_indices` to obtain indices into a single 3D
          scenario sub-cube instead.

        Returns
        -------
        np.ndarray
            1D array of linear voxel indices into the full mask cube.
        """
        return np2sitk.sitk_mask_to_linear_indices(self.mask, order="numpy")

    @computed_field
    @property
    def _numpy_mask(self) -> np.ndarray:
        """
        Returns the mask as a numpy array.

        Returns
        -------
        np.ndarray
            The mask as a numpy array.
        """
        return sitk.GetArrayViewFromImage(self.mask)

    @computed_field
    @property
    def num_of_scenarios(self) -> int:
        """
        Return the number of scenarios stored in the mask.

        A 3D mask always has a single implicit scenario. A 4D mask has one
        scenario per slice along the time axis.

        Returns
        -------
        int
            The number of scenarios (``1`` for a 3D mask, ``size_t`` for 4D).
        """

        if self.mask.GetDimension() == 4:
            return self.mask.GetSize()[3]

        return 1

    def _nominal_mask_3d(self) -> sitk.Image:
        """Return the 3D mask of the nominal (first) scenario."""
        if self.mask.GetDimension() == 4:
            return self.mask[:, :, :, 0]
        return self.mask

    def _label_shape_statistics(self) -> Optional[sitk.LabelShapeStatisticsImageFilter]:
        """Run a shape statistics filter on the nominal mask (None if the mask is empty)."""
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(self._nominal_mask_3d() != 0)
        if not stats.HasLabel(1):
            return None
        return stats

    @computed_field
    @property
    def center_of_mass(self) -> Optional[tuple[float, float, float]]:
        """
        Return the center of mass in world (x, y, z) coordinates (nominal scenario).

        Coordinates are physical LPS coordinates in the grid's units (typically mm).

        Returns
        -------
        tuple[float, float, float] or None
            The center of mass, or None for an empty mask.
        """
        stats = self._label_shape_statistics()
        if stats is None:
            return None
        return tuple(float(c) for c in stats.GetCentroid(1))

    @computed_field
    @property
    def principal_axes(self) -> Optional[tuple[tuple[float, float, float], ...]]:
        """
        Return the principal axes of the VOI (nominal scenario).

        Unit vectors in world (x, y, z) coordinates, ordered by descending spatial
        extent: the first axis points along the VOI's largest elongation. The sign
        of each axis is arbitrary.

        Returns
        -------
        tuple of tuple[float, float, float], or None
            Three principal axis unit vectors, or None for an empty mask.
        """
        stats = self._label_shape_statistics()
        if stats is None:
            return None
        # ITK orders principal moments ascending; reverse to descending extent.
        axes = np.asarray(stats.GetPrincipalAxes(1)).reshape(3, 3)[::-1]
        return tuple(tuple(float(c) for c in axis) for axis in axes)

    @computed_field
    @property
    def shape_parameters(self) -> Optional[dict[str, Any]]:
        """
        Return scalar shape descriptors of the VOI (nominal scenario).

        All lengths are in the grid's units (typically mm):

        - ``volume``: volume of the VOI (units cubed).
        - ``bounding_box_size``: extent of the axis-aligned bounding box (x, y, z).
        - ``equivalent_ellipsoid_diameters``: diameters of the volume-equivalent
          ellipsoid, ordered like :attr:`principal_axes` (largest first).
        - ``elongation``, ``flatness``: ITK shape ratios (>= 1); larger values
          mean a more elongated / flatter shape.

        Returns
        -------
        dict or None
            The shape descriptors, or None for an empty mask.
        """
        stats = self._label_shape_statistics()
        if stats is None:
            return None
        spacing = self._nominal_mask_3d().GetSpacing()
        bbox = stats.GetBoundingBox(1)
        return {
            "volume": float(stats.GetPhysicalSize(1)),
            "bounding_box_size": tuple(
                float(n * s) for n, s in zip(bbox[3:], spacing, strict=True)
            ),
            "equivalent_ellipsoid_diameters": tuple(
                float(d) for d in reversed(stats.GetEquivalentEllipsoidDiameter(1))
            ),
            "elongation": float(stats.GetElongation(1)),
            "flatness": float(stats.GetFlatness(1)),
        }

    def get_indices(self, order: str = "sitk") -> np.ndarray:
        """
        Return linear voxel indices into the full mask cube.

        Convenience wrapper around :attr:`indices` and :attr:`indices_numpy`.
        See those properties for the precise meaning of the linear index for
        3D and 4D masks.

        Parameters
        ----------
        order : str, optional
            Indexing order, ``"sitk"`` (Fortran) or ``"numpy"`` (C). Defaults
            to ``"sitk"``.

        Returns
        -------
        np.ndarray
            1D array of linear voxel indices into the full mask cube.
        """
        if order == "numpy":
            return self.indices_numpy
        if order == "sitk":
            return self.indices
        raise ValueError(f"Unknown order: {order}")

    def scenario_indices(
        self,
        scenario: Optional[int] = None,
        order: str = "numpy",
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Return linear voxel indices restricted to a single 3D scenario.

        Unlike :attr:`indices` / :attr:`indices_numpy`, which return indices
        into the *full* mask cube (including the time axis for 4D masks), the
        indices returned here are always relative to a single 3D scenario
        sub-cube of size ``X * Y * Z``.

        Parameters
        ----------
        scenario : int, optional
            Scenario index in ``[0, num_of_scenarios)``. If ``None`` (default),
            a list with one index array per scenario is returned. The list has
            length ``1`` for a 3D mask and ``num_of_scenarios`` for a 4D mask.
            If an integer is given, only the index array for that scenario is
            returned. For a 3D mask only ``scenario == 0`` is valid.
        order : str, optional
            Indexing order, ``"numpy"`` (C-order) or ``"sitk"`` (Fortran
            order). Defaults to ``"numpy"``.

        Returns
        -------
        np.ndarray
            The 3D linear indices for the requested scenario, if ``scenario``
            is an integer.
        list[np.ndarray]
            One index array per scenario, if ``scenario`` is ``None``.

        Raises
        ------
        ValueError
            If ``order`` is not ``"numpy"`` or ``"sitk"``, if ``scenario`` is
            out of range, or if the underlying mask has invalid dimensionality.
        """
        if order == "numpy":
            _order = "C"
        elif order == "sitk":
            _order = "F"
        else:
            raise ValueError(f"Unknown order: {order}")

        arr = sitk.GetArrayViewFromImage(self.mask)
        if arr.ndim == 3:
            scenario_arrays = [arr]
        elif arr.ndim == 4:
            scenario_arrays = [arr[i] for i in range(arr.shape[0])]
        else:
            raise ValueError("Sanity check failed - mask has invalid dimensions")

        def _flatten(scen_arr: np.ndarray) -> np.ndarray:
            if _order == "C":
                return np.flatnonzero(scen_arr)
            return np.nonzero(scen_arr.ravel(order="F"))[0]

        if scenario is None:
            return [_flatten(s) for s in scenario_arrays]

        if not isinstance(scenario, (int, np.integer)) or isinstance(scenario, bool):
            raise ValueError(f"Scenario index must be an integer, got {type(scenario).__name__}")
        if scenario < 0 or scenario >= len(scenario_arrays):
            raise ValueError(
                f"Scenario index {scenario} is out of range [0, {len(scenario_arrays) - 1}]"
            )

        return _flatten(scenario_arrays[scenario])

    def mask_image(self, ct: CT, order_type="numpy") -> Union[sitk.Image, np.ndarray]:
        """
        Return the masked CT image, either as a numpy array or a SimpleITK image.

        Parameters
        ----------
        ct : CT
            The CT image to be masked.
        order_type : str, optional
            The order type. Defaults to "numpy".

        Returns
        -------
        sitk.Image or np.ndarray
            The masked CT image.
        """

        if order_type not in ["numpy", "sitk"]:
            raise ValueError(f"Invalid order type requested: {order_type}")

        if len(self.mask.GetSize()) == 3:
            masked_ct = sitk.Mask(ct.cube_hu, self.mask)
        elif len(self.mask.GetSize()) == 4:
            masked_ct = [
                sitk.Mask(ct.cube_hu[:, :, :, i], self.mask[:, :, :, i])
                for i in range(self.mask.GetSize()[-1])
            ]
            masked_ct = sitk.JoinSeries(masked_ct)
        else:
            raise ValueError("Sanity check failed - mask has invalid dimensions")

        if order_type == "numpy":
            return sitk.GetArrayFromImage(masked_ct)
        if order_type == "sitk":
            return masked_ct

        raise ValueError(f"Sanity check failed -- Invalid order type requested: {order_type}")

    def scenario_ct_data(
        self,
        ct: CT,
        scenario: Optional[int] = None,
    ) -> Union[np.ndarray, list[np.ndarray]]:
        """
        Return per-scenario CT values extracted by the mask.

        Parameters
        ----------
        ct : CT
            The CT image to extract values from.
        scenario : int, optional
            Scenario index in ``[0, num_of_scenarios)``. If ``None`` (default),
            a list with one array per scenario is returned. The list has
            length ``1`` for a 3D mask and ``num_of_scenarios`` for a 4D mask.
            If an integer is given, only the values for that scenario are
            returned. For a 3D mask only ``scenario == 0`` is valid.

        Returns
        -------
        np.ndarray
            The masked CT values for the requested scenario, if ``scenario``
            is an integer.
        list[np.ndarray]
            One masked CT array per scenario, if ``scenario`` is ``None``.

        Raises
        ------
        ValueError
            If ``scenario`` is out of range or the mask has invalid
            dimensionality.
        """

        mask_np = sitk.GetArrayFromImage(self.mask).astype("bool")
        ct_np = sitk.GetArrayFromImage(ct.cube_hu)

        if mask_np.ndim == 3:
            scenario_data = [ct_np[mask_np]]
        elif mask_np.ndim == 4:
            scenario_data = [ct_np[i][mask_np[i]] for i in range(mask_np.shape[0])]
        else:
            raise ValueError("Sanity Check failed -- Unsupported dimensionality of stored mask")

        if scenario is None:
            return scenario_data

        if not isinstance(scenario, (int, np.integer)) or isinstance(scenario, bool):
            raise ValueError(f"Scenario index must be an integer, got {type(scenario).__name__}")
        if scenario < 0 or scenario >= len(scenario_data):
            raise ValueError(
                f"Scenario index {scenario} is out of range [0, {len(scenario_data) - 1}]"
            )

        return scenario_data[scenario]

    def to_matrad(self, context: str = "mat-file") -> Any:
        """
        Create an object that can be interpreted by matRad in the given context.

        Returns
        -------
        Any
            VOI as list to write cell arrays.
        """

        if context != "mat-file":
            raise ValueError(f"Context {context} not supported")

        voi_list = [0]  # We store an ID which will be changed by cst if exported from there
        voi_list.append(self.name)
        voi_list.append(self.voi_type)
        if self.num_of_scenarios == 1:
            index_lists = np.ndarray(shape=(1,), dtype=object)
            mask_array = sitk.GetArrayFromImage(self.mask)
            mask_array = np.swapaxes(mask_array, 1, 2)
            indices = np.argwhere(mask_array.ravel(order="C") > 0) + 1
            index_lists[0] = np.array(indices, dtype=float)

        else:
            index_lists = self.scenario_indices(order="numpy")
            for i, index_list in enumerate(index_lists):
                index_lists[i] = index_list.astype(float)

        voi_list.append(index_lists)

        property_dict = {
            "alphaX": self.alpha_x,
            "betaX": self.beta_x,
            "Priority": self.overlap_priority,
        }
        voi_list.append(property_dict)

        # Will not be populated in here but in cst if exported from there
        objective_dict = {}
        voi_list.append([objective_dict])

        return voi_list

    def _resample_on_new_ct(self, new_ct: CT) -> Self:
        """
        Resample on new CT image.

        Parameters
        ----------
        new_ct : CT
            The new CT image to resample the VOI on.

        Returns
        -------
        Self
            The resampled VOI.
        """

        if not isinstance(new_ct, CT):
            raise ValueError("new_ct must be a CT object")

        if self.mask.GetDimension() == 3:
            new_mask = sitk.Resample(
                self.mask, new_ct.cube_hu, sitk.Transform(), sitk.sitkNearestNeighbor, 0
            )
        elif self.mask.GetDimension() == 4:
            new_mask = []
            for i in range(self.mask.GetSize()[-1]):
                new_mask.append(
                    sitk.Resample(
                        self.mask[:, :, :, i],
                        new_ct.cube_hu,
                        sitk.Transform(),
                        sitk.sitkNearestNeighbor,
                        0,
                    )
                )
            new_mask = sitk.JoinSeries(new_mask)
        else:
            raise ValueError("Sanity check failed -- mask has invalid dimensions")

        resampled_voi = self.model_copy(
            update={"mask": new_mask, "grid": Grid.from_sitk_image(new_ct.cube_hu)}
        )

        if len(resampled_voi.indices) == 0:
            warnings.warn("Resampling created an empty structure")

        return resampled_voi


class OAR(VOI):
    """
    Represents an organ at risk (OAR).

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'OAR'.
    """

    voi_type: str = "OAR"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for an OAR.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "OAR".
        """

        if v != "OAR":
            raise ValueError('VOI type for OAR must be "OAR"')
        return v


class Target(VOI):
    """
    Represents a target VOI.

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'TARGET'.
    """

    voi_type: str = "TARGET"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for a Target.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "OAR".
        """

        if v != "TARGET":
            raise ValueError('VOI type for a Target must be "TARGET"')
        return v


class HelperVOI(VOI):
    """
    Represents a helper VOI.

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'HELPER'.
    """

    voi_type: str = "HELPER"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for a HelperVOI.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "HELPER".
        """
        if v != "HELPER":
            raise ValueError('VOI type for a HelperVOI must be "HELPER"')
        return v


class ExternalVOI(VOI):
    """
    Represents an external contour limiting voxels to be considered for planning (EXTERNAL).

    Attributes
    ----------
    Inherits all attributes from Plan.

    Methods
    -------
    voi_type : str
        Returns the voi_type as 'EXTERNAL'.
    """

    voi_type: str = "EXTERNAL"

    @field_validator("voi_type", mode="after")
    @classmethod
    def validate_voi_type(cls, v: str) -> str:
        """
        Validate the voi type for an EXTERNAL contour.

        Parameters
        ----------
        v : str
            The voi type to be validated.

        Returns
        -------
        str
            The validated voi type.

        Raises
        ------
        ValueError
            If the voi type is not "EXTERNAL".
        """

        if v != "EXTERNAL":
            raise ValueError('VOI type for EXTERNAL must be "EXTERNAL"')
        return v


__VOITYPES__ = {"OAR": OAR, "TARGET": Target, "HELPER": HelperVOI, "EXTERNAL": ExternalVOI}


def create_voi(data: Union[dict[str, Any], VOI, None] = None, **kwargs) -> VOI:
    """
    Create a VOI object.

    Parameters
    ----------
    data : Union[dict[str, Any], VOI, None]
        Dictionary containing the data to create the VOI object.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    VOI
        A VOI object.
    """

    if data:
        # If data is already a VOI object, return it directly
        if isinstance(data, VOI):
            return data

        # obtain voi type if we have a dict including camelCase check
        voi_type = data.get("voi_type", data.get("voiType", None))

        if voi_type in __VOITYPES__:
            return __VOITYPES__[voi_type].model_validate(data)

        raise ValueError(f"Invalid VOI type: {voi_type}")

    voi_type = kwargs.get("voi_type", "")

    if voi_type in __VOITYPES__:
        return __VOITYPES__[voi_type](**kwargs)

    raise ValueError(f"Invalid VOI type: {voi_type}")


def validate_voi(data: Union[dict[str, Any], VOI, None] = None, **kwargs) -> VOI:
    """
    Validate and create a VOI object.

    Synonym to create_voi but should be used in validation context.

    Parameters
    ----------
    voi : Union[dict[str, Any], VOI, None], optional
        Dictionary containing the data to create the VOI object, by default None.
    **kwargs
        Arbitrary keyword arguments.

    Returns
    -------
    VOI
        A validated VOI object.
    """
    return create_voi(data, **kwargs)

"""Sequenced beam object."""

from collections.abc import Mapping
from typing import Any, Optional
from numpy.typing import NDArray
import logging
import numpy as np

from pydantic import (
    Field,
    model_validator,
    field_validator,
    field_serializer,
    SerializationInfo,
    ValidationInfo,
    computed_field,
    SerializeAsAny,
)
from pyRadPlan.core import PyRadPlanBaseModel
from pyRadPlan.machines import MLC, BeamLimitingDevice, Jaw

logger = logging.getLogger(__name__)


def _as_list(devices: Any) -> list:
    if isinstance(devices, np.ndarray):
        return devices.tolist()
    return [devices] if isinstance(devices, (Mapping, BeamLimitingDevice)) else list(devices)


def _concrete_bld(device: Any) -> Any:
    """Validate a dumped beam limiting device as an MLC or a jaw, told apart by its fields."""
    if not isinstance(device, Mapping):
        return device
    if "leaf_positions" in device or "leafPositions" in device:
        return MLC.model_validate(device)
    if "positions" in device:
        return Jaw.model_validate(device)
    return device


class BeamSequence(PyRadPlanBaseModel):
    """A class representing the aperture information of a beam.

    Following matRad's ``resultGUI.sequencing.beam`` and ``apertureInfo.beam``,
    merged into a single object on import. ``cal_fac``, ``mlc_sequence`` and
    ``extra_blds`` have no matRad counterpart and are pyRadPlan additions.

    All positions are in mm in the beam's eye view frame of the ray matrix (LPS +x and +z at
    gantry and couch angle 0, see :class:`~pyRadPlan.machines.BeamLimitingDevice`).

    Parameters
    ----------
    shapes : Optional[NDArray]
        A 3D array of the binary shapes in the beam, ``(dim_stack, dim_travel, num_shapes)``.
    shapes_weight : NDArray
        Weight of each shape in the beam, ``(num_shapes,)``, in physical fluence units (not
        stratified).
    bixel_ix : Optional[list[int]]
        Global bixel indices of this beam's rays (columns in dij).
    stratified_fluence : Optional[NDArray]
        The stratified fluence map (matRad's ``D_0``), shaped ``(dim_stack, dim_travel)`` on the
        leaf-row/column grid. Integer-valued in *level* units (physical fluence in
        ``Sequence.fluence``).
    cal_fac : Optional[float]
        Calibration factor the fluence was stratified against (the beam's peak fluence),
        so one intensity level of ``stratified_fluence`` is worth
        ``cal_fac / num_of_levels``.
    leaf_pair_pos : Optional[list[float]]
        A list of positions for each leaf pair in the beam (leaf centers, not boundaries).
    is_active_leaf_pair : Optional[list[bool]]
        A list indicating whether each leaf pair is active (True) or inactive (False).
    bixel_ind_map : Optional[NDArray]
        The same indices as ``bixel_ix`` (native ray resolution) as 2D array spanning the
        beam's bounding box, with ``NaN`` wherever the box holds no ray.
    pos_of_corner_bixel : Optional[list[float]]
        BEV position of the grid's corner bixel: a 3-vector carrying ``min_travel`` at
        the travel axis and ``min_stack`` at the stack axis, 0 elsewhere. The origin
        for turning grid indices back into BEV coordinates.
    mlc_window : Optional[list[float]]
        Field window ``[travel_min, travel_max, stack_min, stack_max]``. The travel axis spans the ray
        extent plus half a bixel either side; the stack axis spans the union of that
        and the leaf rows in use, so the jaws never clip a leaf.
    lim_l : Optional[list[float]]
        Left leaf travel limit (DAO bound, not a position) for each leaf pair.

    lim_r : Optional[list[float]]
        Right leaf travel limit (DAO bound, not a position) for each leaf pair.
    central_leaf_pair : Optional[float]
        1-based index of the leaf pair on isocenter. On a physical leaf grid with an
        even leaf count a leaf boundary sits on the isocenter (``N/2 + 0.5``), for an
        odd leaf count a leaf center (``(N + 1) / 2``). Sequencing directly on the ray
        grid keeps matRad's ``ceil(N/2)``, with a leaf centre on isocenter.
    left_leaf_pos : Optional[list[NDArray]]
        A list of 1D arrays containing the left leaf positions of the active leaf pairs
        for each shape in the beam. Mirrors ``mlc_sequence[m].leaf_positions[:, 0]``.
    right_leaf_pos : Optional[list[NDArray]]
        A list of 1D arrays containing the right leaf positions of the active leaf pairs
        for each shape in the beam. Mirrors ``mlc_sequence[m].leaf_positions[:, 1]``.
    vector_offset : Optional[list[int]]
        A list containing the vector offset for each shape in the beam indicating the
        starting index of the shape in the flat ``Sequence.aperture_vector``.
    num_of_active_leaf_pairs : Optional[int]
        The number of active leaf pairs (also closed pairs) within the field.
    mlc_sequence : Optional[list[MLC]]
        One ``MLC`` object per shape, carrying the full geometry (leaf width, boundaries,
        leaf positions, orientation).
    extra_blds : Optional[list[list[BeamLimitingDevice]]]
        Beam limiting devices active for each shape beyond the MLCs, currently the
        static X/Y jaw pair bounding ``mlc_window``.


    Attributes
    ----------
    sum_shapes : NDArray
        Unweighted sum of ``shapes`` over the shape axis: how many shapes each
        bixel is open in.
    num_of_shapes : int
        Number of delivered shapes.
    shape_map : Optional[list[NDArray]]
        ``shapes`` sliced per shape, in matRad's per-shape layout.
    weight : list[float]
        ``shapes_weight`` in matRad's per-shape layout.
    """

    shapes: Optional[NDArray] = Field(default=None)
    shapes_weight: NDArray
    bixel_ix: Optional[list[int]] = Field(default=None)
    stratified_fluence: Optional[NDArray] = Field(default=None, alias="fluence")
    cal_fac: Optional[float] = Field(default=None)
    leaf_pair_pos: Optional[list[float]] = Field(default=None)
    is_active_leaf_pair: Optional[list[bool]] = Field(default=None)
    bixel_ind_map: Optional[NDArray] = Field(default=None)
    pos_of_corner_bixel: Optional[list[float]] = Field(default=None)
    mlc_window: Optional[list[float]] = Field(alias="MLCWindow", default=None)
    lim_l: Optional[list[float]] = Field(default=None)
    lim_r: Optional[list[float]] = Field(default=None)
    central_leaf_pair: Optional[float] = Field(default=None)
    left_leaf_pos: Optional[list[NDArray]] = Field(default=None)
    right_leaf_pos: Optional[list[NDArray]] = Field(default=None)
    vector_offset: Optional[list[int]] = Field(default=None)
    num_of_active_leaf_pairs: Optional[int] = Field(default=None)
    mlc_sequence: Optional[list[MLC]] = Field(default=None)
    extra_blds: Optional[list[list[SerializeAsAny[BeamLimitingDevice]]]] = Field(default=None)

    @computed_field
    @property
    def sum_shapes(self) -> NDArray:
        """Sum of all shapes in the beam along the last dimension."""
        if self.shapes is None:
            return np.array([])
        if self.shapes.ndim == 3:
            return np.sum(self.shapes, axis=-1)  # Sum over the last dimension (e.g. n=1 => 3x3x1)
        elif self.shapes.ndim == 2:
            return self.shapes
        else:
            logger.warning("Shapes have an unsupported number of dimensions.")
            return np.array([])  # Return an empty array for unsupported cases

    @computed_field
    @property
    def shape_map(self) -> Optional[list[NDArray]]:
        """``shapes`` sliced per shape, as matRad's ``shape(j).shapeMap``."""
        if self.shapes is None:
            return None
        if self.shapes.ndim == 3:
            return [self.shapes[:, :, m] for m in range(self.shapes.shape[2])]
        return [self.shapes]

    @computed_field
    @property
    def weight(self) -> list[float]:
        """``shapes_weight`` in matRad's per-shape ``shape(j).weight`` layout."""
        return [float(w) for w in self.shapes_weight]

    @computed_field
    @property
    def num_of_shapes(self) -> int:
        """Total number of shapes in the beam."""
        if self.shapes is not None:
            return self.shapes.shape[2] if self.shapes.ndim == 3 else 1
        return len(self.shapes_weight)

    @field_validator("bixel_ind_map", mode="after")
    @classmethod
    def validate_bixel_ind_map(cls, v: Any, info: ValidationInfo) -> np.ndarray:
        """Rotate 2D index map and convert from 1-based to 0-based indexing."""

        if info.context and info.context.get("from_matRad"):
            arr = np.asarray(v, dtype=np.float64)
            mask = ~np.isnan(arr)
            arr[mask] -= 1  # Convert to 0-based indexing
            return arr.T

        return v

    @field_serializer("bixel_ix")
    def _bixel_ix_to_matrad(self, v: Optional[list[int]], info: SerializationInfo) -> Any:
        """Count the bixels from 1 for matRad."""
        if v is not None and info.context and info.context.get("matRad") == "mat-file":
            return np.asarray(v, dtype=float) + 1
        return v

    @field_serializer("bixel_ind_map")
    def _bixel_ind_map_to_matrad(self, v: Optional[np.ndarray], info: SerializationInfo) -> Any:
        """Undo the rotation and the 0-based indexing of the import for matRad."""
        if v is not None and info.context and info.context.get("matRad") == "mat-file":
            arr = np.array(v, dtype=np.float64)
            arr[~np.isnan(arr)] += 1
            return arr.T
        return v

    @classmethod
    def _process_matrad_data(cls, data: Any, info: ValidationInfo) -> dict:
        """Handle data coming from matRad."""
        # Guard against non-dict inputs (e.g. an empty matRad struct read as an
        # ndarray by newer readers) before the truthiness test, since ``ndarray == {}``
        # raises an ambiguous-truth ValueError.
        if not isinstance(data, dict) or not data:
            return data

        # add dimension because if num_of_shapes = 1, matLab saves only has 2d Shapes
        # TODO: In future we should add up all shapes along axis=0 (currently axis=2)
        if "shapes" in data:
            if data["shapes"].ndim == 2:
                shapes = data["shapes"][:, :, np.newaxis]
                data["shapes"] = shapes
            # matRad's shapeMap weighted for Xia and binary for Siochi and Engel
            data["shapes"] = (np.asarray(data["shapes"]) > 0).astype(np.uint8)
        if "shapes_weight" in data:
            shapes_weight = np.array(data["shapes_weight"])
            data["shapes_weight"] = shapes_weight

        if "shape" in data:
            for key in data["shape"].keys():
                if data["num_of_shapes"] != 1:
                    data.update({key: data["shape"][key]})
                else:
                    data.update({key: [data["shape"][key]]})
            data.pop("shape", None)

        return data

    @model_validator(mode="before")
    @classmethod
    def validate_distribution(cls, data: Any, info: ValidationInfo) -> Any:
        if info.context and info.context.get("from_matRad"):
            data = cls._process_matrad_data(data, info)

        return data

    # The devices are stored by their base class: their own fields are serialized
    # (SerializeAsAny) and a dumped device is validated as the class its fields describe.
    @field_validator("extra_blds", mode="before")
    @classmethod
    def _concrete_blds(cls, v):
        if v is None:
            return None
        if isinstance(v, np.ndarray):
            v = v.tolist()
        # A single shape is read from a .mat file without its list of shapes
        if len(v) > 0 and all(isinstance(d, (Mapping, BeamLimitingDevice)) for d in v):
            v = [v]
        return [[_concrete_bld(device) for device in _as_list(devices)] for devices in v]

    # shapes are binary aperture masks, so uint8 is lossless and compact
    @field_validator("shapes", mode="before")
    @classmethod
    def _to_uint8(cls, v):
        if v is None:
            return None
        arr = np.asarray(v)
        return arr.astype(np.uint8)

    # the level map must stay float: uint8 like shapes would wrap above 255
    @field_validator("stratified_fluence", mode="before")
    @classmethod
    def _to_float(cls, v):
        if v is None:
            return None
        arr = np.asarray(v)
        return arr.astype(float)

    # ensure shapes_weight always 1D float
    @field_validator("shapes_weight", mode="before")
    @classmethod
    def _to_float1d(cls, v):
        arr = np.asarray(v)
        return arr.astype(float).reshape(-1)

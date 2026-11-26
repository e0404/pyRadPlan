"""AI-driven pencil-beam dose engine, backed by the ``pyRadPlan.ai.modelhub`` model contract."""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, List, Optional, Tuple

import SimpleITK as sitk
import numpy as np

import array_api_compat
from ...core.xp_utils.typing import Array

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.scenarios import Scenario
from pyRadPlan.core import Grid, PyRadPlanBaseModel
from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.core.xp_utils.compat import array_meshgrid
from pyRadPlan.core.xp_utils import (
    ArrayBackendGridInterpolator,
    device_string,
    to_namespace,
    to_numpy,
)
from pyRadPlan.geometry import get_beam_rotation_matrix
from pyRadPlan.ai.modelhub import (
    ModelTask,
    hub_config_dir,
    list_hub_models,
    list_local_models,
    load_model_with_dir,
    local_model_dir,
    model_facts,
    resolve_model_dir,
    task_from_dir,
    task_from_name,
)
from pydantic import ConfigDict, Field, ValidationError, field_validator

from ._base import ct_phase_images
from ._base_pencilbeam import PencilBeamEngineAbstract
from pyRadPlan._settings import get_settings

logger = logging.getLogger(__name__)


def _interpolate_or_fill(
    interpolator: ArrayBackendGridInterpolator,
    axes: tuple[Array, ...],
    xi: Array,
    fill_value: float,
) -> Array:
    """Interpolate, with ``fill_value`` for queries outside the grid.

    ``ArrayBackendGridInterpolator`` clips such queries to the grid instead. A query is
    inside if it lies within the first and last point of every (ascending) axis, as for
    SciPy's ``RegularGridInterpolator(..., bounds_error=False)``.

    Parameters
    ----------
    interpolator : ArrayBackendGridInterpolator
        Interpolator built on ``axes``.
    axes : tuple[Array, ...]
        The interpolator's grid axes.
    xi : Array
        Queries of shape ``(N, len(axes))``.
    fill_value : float
        Value for queries outside the grid.

    Returns
    -------
    Array
        Interpolated values of shape ``(N, *value_shape)``.
    """
    xp = array_api_compat.array_namespace(xi)
    inside = None
    for i, axis in enumerate(axes):
        axis_inside = (xi[:, i] >= axis[0]) & (xi[:, i] <= axis[-1])
        inside = axis_inside if inside is None else inside & axis_inside
    values = interpolator(xi)
    inside = xp.reshape(inside, inside.shape + (1,) * (values.ndim - 1))
    return xp.where(inside, values, fill_value)


def _rotation_onto(a: Array, b: Array) -> Array:
    """Return the rotation matrix ``R`` with ``R @ a`` along ``b`` (Rodrigues' formula).

    Uses the form ``I + K + K @ K / (1 + a.b)`` for unit vectors, with ``K`` the
    cross-product matrix of ``a x b``. It is well conditioned for all but
    antiparallel ``a`` and ``b``, and exact for ``a == b``. The engine only rotates a
    central ray onto a beamlet direction, which are never antiparallel.

    Parameters
    ----------
    a, b : Array
        Nonzero 3-vectors; they need not be normalized.

    Returns
    -------
    Array
        The ``(3, 3)`` rotation matrix, in the dtype of ``a`` and ``b``.
    """
    xp = array_api_compat.array_namespace(a, b)
    a = a / xp.sqrt(xp.sum(a * a))
    b = b / xp.sqrt(xp.sum(b * b))
    v0 = a[1] * b[2] - a[2] * b[1]
    v1 = a[2] * b[0] - a[0] * b[2]
    v2 = a[0] * b[1] - a[1] * b[0]
    zero = xp.zeros_like(v0)
    k = xp.stack(
        [
            xp.stack([zero, -v2, v1]),
            xp.stack([v2, zero, -v0]),
            xp.stack([-v1, v0, zero]),
        ]
    )
    eye = xp.eye(3, dtype=k.dtype, device=array_api_compat.device(k))
    return eye + k + (k @ k) / (1 + xp.sum(a * b))


def _host_vector(value: Any) -> np.ndarray:
    """Return a point of the steering information as a float64 NumPy vector of length 3."""
    if array_api_compat.is_array_api_obj(value):
        value = to_numpy(value)
    return np.reshape(np.asarray(value, dtype=np.float64), (3,))


class GridParametersConfig(PyRadPlanBaseModel):
    """Sampling-grid parameters a dose-calc model was trained on."""

    model_config = ConfigDict(frozen=True)

    spacing: float = Field(..., gt=0, description="Grid spacing [mm]")
    lateral_range: float = Field(..., gt=0, description="Total width of the lateral grid [mm]")
    depth_offset_start: float = Field(..., description="Depth offset start [mm]")
    depth_range: float = Field(..., gt=0, description="Total depth range [mm]")


class DoseCalcModelConfig(PyRadPlanBaseModel):
    """AIBeamletEngine-specific model metadata, read from ``metadata.dose_calc``.

    The generic :func:`pyRadPlan.ai.modelhub.load_model` loader only cares about
    ``model_name``/``model_params``/``model_preprocessing``/``preprocessor_name``; this
    covers the extra fields the AI beamlet engine needs (sampling grid, quantities
    produced, training assumptions) which live in the free-form ``metadata`` section
    of a model's ``model_config.json``.
    """

    model_config = ConfigDict(frozen=True)

    grid_parameters: GridParametersConfig
    quantities: List[str] = Field(
        default_factory=lambda: ["physical_dose"],
        description="Quantities the model predicts, 'physical_dose' first",
    )
    radiation_mode: Optional[str] = None
    energy_range_mev: Optional[Tuple[float, float]] = None
    trained_machine: Optional[str] = None

    @field_validator("quantities")
    @classmethod
    def _physical_dose_first(cls, quantities: List[str]) -> List[str]:
        if not quantities or quantities[0] != "physical_dose":
            raise ValueError(f"must start with 'physical_dose', got {quantities}")
        if len(set(quantities)) != len(quantities):
            raise ValueError(f"must not repeat a quantity, got {quantities}")
        return quantities


class AIBeamletEngine(PencilBeamEngineAbstract):
    """AI-driven pencil-beam dose engine.

    What it does
    - Builds a beamlet-aligned regular sampling grid (depth, lateral, lateral) for each ray.
    - Samples CT on that grid and feeds it, along with the bixel energy, into an AI model
        to predict a local dose cuboid. This is done as a batch per ray.
    - Maps world dose-grid voxel centers into the beamlet-local frame and interpolates the
        predicted cuboid at those points.
    - Emits sparse bixel contributions (linear indices + `physical_dose`) that the base
        pencil-beam engine accumulates into the dose influence matrix.

    How it works
    - Coordinate frames: world → BEV → beamlet-local via rotation/translation from the beam.
    - Grid construction: uniform spacing in depth and lateral with configurable ranges; axis
        order is (depth, lateral, lateral) to match model conventions.
    - Model inference: model + preprocessor loaded with `pyRadPlan.ai.modelhub` from the
        '<org>/<model>' id in `prop_dose_calc["model"]`; backend-agnostic arrays using the
        Array API (`self.xp`). Loading executes the code shipped with the model: a model
        folder placed under ``settings.ai.modelhub_local_models_dir`` by the user runs
        as-is, a downloaded one only once trusted, with
        ``pyRadPlan.ai.modelhub.trust_model("<org>/<model>")`` or
        ``settings.ai.modelhub_trust_remote_code = True``
        (``PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1``).
    - Interpolation: 3D linear interpolation over (depth, lateral, lateral) axes to evaluate
        predicted dose at world dose-grid locations.
    - Scenarios: setup shifts are supported, each computed as its own geometry. Range
        errors and several CT phases are refused (see ``supported_uncertainties``).

    Notes
    -----
    - Inference is delegated to the model's own preprocessor: ``preprocess`` builds the
        network input from the CT cuboid and the beamlet energies, and
        ``predict(model, model_input)`` returns a dict of predicted cuboid batches keyed
        by quantity name, one per quantity in ``metadata.dose_calc.quantities``.
    - Positions are float32 on every backend; each ray's transforms are composed in
        float64 on the host.
    """

    short_name = "AIBeamlet"
    name = "AI Beamlet Engine"
    possible_radiation_modes = ["protons", "VHEE"]

    _dij_guarantee_canonical: ClassVar[bool] = True
    _dij_guarantee_nonzero: ClassVar[bool] = True
    # A setup shift only moves the beamlets. The model has no input for a range error, and
    # only one CT phase is sampled per calculation.
    supported_uncertainties: ClassVar[tuple[str, ...]] = ("setup",)

    model: Annotated[
        Optional[str],
        Field(description="AI model as a '<org>/<model>' id, e.g. 'DKFZ-RadOpt/my-model'"),
    ] = None
    max_batch_size: Annotated[
        Optional[int],
        Field(
            ge=1,
            description="Maximum number of beamlets per model call; unset predicts a whole "
            "ray at once",
        ),
    ] = 32
    ensemble_size: Annotated[
        Optional[int],
        Field(
            ge=1,
            description="Ensemble members per prediction, for models that sample an ensemble; "
            "unset keeps the model's own setting",
        ),
    ] = None
    dose_threshold: Annotated[
        float,
        Field(ge=0.0, description="Predicted dose at or below which a voxel is not stored"),
    ] = 1e-7
    assumed_sparsity = 1e-2

    @classmethod
    def config_choices(cls, radiation_mode: Optional[str] = None) -> dict[str, List[str]]:
        """Suggested values per parameter, for the GUI's engine-configuration form.

        Lists the dose-calculation models available locally and, unless offline,
        those offered by the configured hub organization, leaving out the ones
        this engine could not run: anything that does not declare itself a
        dose-calculation model, models trained for another *radiation_mode* and
        models whose framework is not installed. For a model
        that is not on disk, only its ``model_config.json`` is fetched, never its
        weights. The list is a suggestion: any other id can still be typed.
        """
        cfg = get_settings().ai
        names = set(list_local_models()) | set(
            list_hub_models(cfg.modelhub_hf_org, cfg.modelhub_offline)
        )

        models = []
        for name in sorted(names):
            # Only full '<org>/<model>' ids are usable; a bare folder name is not.
            if "/" not in name:
                continue
            # What is not on disk is judged by its config alone, which is fetched
            # without the weights.
            model_dir = local_model_dir(name) or hub_config_dir(name, cfg.modelhub_offline)
            if model_dir is None:
                continue

            config, modality, missing = model_facts(model_dir)
            # A model must show that it does dose calculation: a declared task, the
            # name, or the engine's own metadata section. The task of a downloaded
            # config cannot come from its folder, which is a revision hash.
            is_dose_calc = ModelTask.DOSE_CALC in (
                task_from_dir(model_dir),
                task_from_name(name),
            ) or "dose_calc" in (config.get("metadata") or {})
            if not is_dose_calc:
                continue
            if missing or (radiation_mode and modality and modality != radiation_mode):
                continue
            models.append(name)
        return {"model": models}

    @staticmethod
    def is_available(pln, machine) -> Tuple[bool, str]:
        """Whether the engine can run for *pln*, and why not.

        Never raises and never contacts the network: only a model already on disk
        (or in the HuggingFace cache) is inspected. The engine is offered unless the
        plan names such a model and it cannot be used -- trained for another radiation
        mode, or the packages it declares under ``metadata.requirements`` are not
        installed -- or names a model that is not an ``'<org>/<model>'`` id. Without a
        model, or with one that is not on disk, the engine is available; the model is
        then resolved, and downloaded if needed, only when a calculation runs.

        Parameters
        ----------
        pln : Plan, dict or str
            The plan (or plan dict) to check. A bare radiation mode is accepted too.
        machine : Any
            Unused; part of the engine interface.

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` if available, otherwise ``(False, reason)``.
        """
        # pln may be a Plan, a plan dict, or just a radiation mode
        if pln is None or isinstance(pln, str):
            radiation_mode, prop_dose_calc = pln, None
        elif isinstance(pln, dict):
            radiation_mode = pln.get("radiation_mode")
            prop_dose_calc = pln.get("prop_dose_calc")
        else:
            radiation_mode = getattr(pln, "radiation_mode", None)
            prop_dose_calc = getattr(pln, "prop_dose_calc", None)

        if isinstance(prop_dose_calc, dict):
            model = prop_dose_calc.get("model")
        else:  # an already constructed engine
            model = getattr(prop_dose_calc, "model", None)

        if not model:  # an empty name counts as "none given"
            return True, ""
        if "/" not in model:
            return False, f"Model '{model}' is not a '<org>/<model>'."

        model_dir = local_model_dir(model)
        if model_dir is None:
            return True, ""

        _, modality, missing = model_facts(model_dir)
        if radiation_mode and modality and modality != radiation_mode:
            return False, (
                f"Model '{model}' was trained for '{modality}', "
                f"but the plan uses '{radiation_mode}'."
            )
        if missing:
            return False, (
                f"Model '{model}' requires {', '.join(missing)}, which is not installed."
            )
        return True, ""

    def _compute_bixel(self, ray: dict, k: int) -> dict:
        """Read back a precomputed bixel dose contribution.

        Parameters
        ----------
        ray : dict
            Ray dictionary with ``_precomputed_bixels`` populated by ``_init_ray``.
        k : int
            Index of the beamlet within the ray.

        Returns
        -------
        dict
            The beamlet's steering information (its ``weight`` scales a forward
            calculation) with ``ix`` and one entry per quantity the model predicts.
        """
        # A copy, so the beam's beamlets do not keep the predicted arrays alive.
        bixel = ray["beamlets"][k].copy()
        bixel.update(ray["_precomputed_bixels"][k])
        return bixel

    def _predict(self, ct_cuboid: Array, energies: Array) -> dict[str, Array]:
        """Predict the dose cuboids of a batch of beamlets sharing one CT cuboid.

        The model's preprocessor runs the inference: ``preprocess`` builds the network
        input and ``predict(model, model_input)`` returns ``{quantity: array}``.

        Parameters
        ----------
        ct_cuboid : Array
            CT cuboid sampled on the beamlet-local grid, shared by the whole batch.
        energies : Array
            Per-bixel energies (batch dimension).

        Returns
        -------
        dict[str, Array]
            One ``(batch, depth, lateral, lateral)`` array in ``self.xp`` per declared
            quantity; anything else the model returns is dropped.

        Raises
        ------
        ValueError
            If the prediction is not a dict holding every declared quantity in the
            shape of the sampling grid.
        """
        xp = self.xp
        model_input = self.preprocessor.preprocess({"ct_cuboid": ct_cuboid, "energy": energies})
        result = self.preprocessor.predict(self.network, model_input)

        quantities = self.model_cfg.quantities
        if not isinstance(result, Mapping):
            raise ValueError(
                f"The preprocessor of model '{self.model}' must return a dict "
                f"{{quantity: array}} from predict(), not {type(result).__name__}."
            )
        missing = [q for q in quantities if q not in result]
        if missing:
            raise ValueError(
                f"Model '{self.model}' declares the quantities {quantities}, but its "
                f"prediction lacks {missing} (it holds {list(result)})."
            )

        expected = (energies.shape[0], *ct_cuboid.shape)
        predicted = {q: to_namespace(xp, result[q], device=self.device) for q in quantities}
        for q_name, value in predicted.items():
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Model '{self.model}' predicted '{q_name}' with shape "
                    f"{tuple(value.shape)}, but the batch of {expected[0]} beamlets on its "
                    f"sampling grid (depth, lateral, lateral) needs {expected}."
                )
        return predicted

    def _construct_grid(self):
        """Build the beamlet-local sampling grid the model was trained on.

        Sets ``self.grid``, the sample centers as ``(x, y, z) = (lateral, depth, lateral)``
        rows ordered like a ``(depth, lateral, lateral)`` cuboid, ``self.corners_local``,
        the cuboid's corners half a sample beyond the outermost centers, and the axes in
        ``self.grid_information``.
        """
        xp = self.xp
        grid_params = self.model_cfg.grid_parameters
        spacing = grid_params.spacing
        lateral_half_width = grid_params.lateral_range / 2
        depth_start = grid_params.depth_offset_start

        lateral_range = xp.arange(
            -lateral_half_width + spacing / 2,
            lateral_half_width + spacing / 2,
            spacing,
            dtype=xp.float32,
            device=self.device,
        )
        depth_range = xp.arange(
            depth_start,
            depth_start + grid_params.depth_range,
            spacing,
            dtype=xp.float32,
            device=self.device,
        )

        depth_vals, lateral_x_vals, lateral_z_vals = array_meshgrid(
            depth_range, lateral_range, lateral_range, indexing="ij"
        )
        self.grid = xp.stack(
            (
                xp.reshape(lateral_x_vals, (-1,)),
                xp.reshape(depth_vals, (-1,)),
                xp.reshape(lateral_z_vals, (-1,)),
            ),
            axis=-1,
        )

        # The corners bound the dose-grid voxels worth transforming into the local frame.
        lx_min, lx_max = lateral_range[0] - spacing / 2, lateral_range[-1] + spacing / 2
        ly_min, ly_max = depth_range[0] - spacing / 2, depth_range[-1] + spacing / 2
        lz_min, lz_max = lx_min, lx_max

        self.corners_local = xp.asarray(
            [
                [lx_min, ly_min, lz_min],
                [lx_min, ly_min, lz_max],
                [lx_min, ly_max, lz_min],
                [lx_min, ly_max, lz_max],
                [lx_max, ly_min, lz_min],
                [lx_max, ly_min, lz_max],
                [lx_max, ly_max, lz_min],
                [lx_max, ly_max, lz_max],
            ],
            dtype=xp.float32,
            device=self.device,
        )

        self.grid_information = {
            "lateral_range": lateral_range,
            "depth_range": depth_range,
            "spacing": spacing,
        }

    def _extract_ct_cuboid(self, ray: dict[str, Any]) -> None:
        """Sample the CT on this ray's beamlet-local grid into ``ray["ct_cuboid"]``.

        Parameters
        ----------
        ray : dict[str, Any]
            The ray, as prepared by ``_init_ray``.
        """
        self._rotate_grid(ray)
        ray["ct_cuboid"] = self._extract_sitk_grid_and_interpolate(self.grid_information)

    def _rotate_grid(self, ray: dict) -> None:
        """Place the sampling grid on a ray and find the dose-grid voxels it covers.

        The local frame has its depth axis along the ray, through the source. Updates
        ``self.grid_information`` with ``grid_rotated``, the sampling grid in world
        coordinates, ``ix_mask``, the linear dose-grid indices of the voxels within the
        cuboid's bounding box, and ``local_points_masked``, their local coordinates.

        Parameters
        ----------
        ray : dict
            The ray, with ``source_point_bev``, ``target_point_bev``, ``iso_center`` and
            the beam's world-to-BEV ``rotation_mat``, all float64 NumPy arrays.
        """
        source = ray["source_point_bev"]
        # Row vectors times this map the beamlet's direction onto the central ray (BEV y).
        derived_rot_mat = _rotation_onto(-source, ray["target_point_bev"] - source)

        # Both rotations and the source offset compose into one affine map, so no point is
        # shifted by the source position (10 m away) and back, which would cost float32
        # precision: local = (world - iso) @ world_to_local + offset
        world_to_local = ray["rotation_mat"] @ derived_rot_mat
        offset = source - source @ derived_rot_mat

        xp = self.xp
        iso_center = xp.asarray(ray["iso_center"], dtype=xp.float32, device=self.device)
        offset = xp.asarray(offset, dtype=xp.float32, device=self.device)
        world_to_local = xp.asarray(world_to_local, dtype=xp.float32, device=self.device)
        local_to_world = world_to_local.T

        grid_rotated = (self.grid - offset) @ local_to_world + iso_center
        corners_world = (self.corners_local - offset) @ local_to_world + iso_center

        min_world = xp.min(corners_world, axis=0)
        max_world = xp.max(corners_world, axis=0)
        bbox_mask = (
            (self.coords_all[:, 0] >= min_world[0])
            & (self.coords_all[:, 0] <= max_world[0])
            & (self.coords_all[:, 1] >= min_world[1])
            & (self.coords_all[:, 1] <= max_world[1])
            & (self.coords_all[:, 2] >= min_world[2])
            & (self.coords_all[:, 2] <= max_world[2])
        )
        candidate_indices = xp.nonzero(bbox_mask)[0]
        candidate_points = xp.take(self.coords_all, candidate_indices, axis=0)

        self.grid_information.update(
            {
                "grid_rotated": grid_rotated,
                "local_points_masked": (candidate_points - iso_center) @ world_to_local + offset,
                "ix_mask": candidate_indices,
            }
        )

    def _extract_sitk_grid_and_interpolate(self, grid_information: dict) -> Array:
        """Interpolate the CT at the world positions of the sampling grid.

        Parameters
        ----------
        grid_information : dict
            ``self.grid_information``, with ``grid_rotated``, ``lateral_range`` and
            ``depth_range``.

        Returns
        -------
        Array
            The CT in HU, shaped ``(depth, lateral, lateral)``; air (-1000 HU) outside
            the CT.
        """
        xp = self.xp
        lateral_range = grid_information["lateral_range"]
        depth_range = grid_information["depth_range"]

        values = _interpolate_or_fill(
            self.ct_interpolator,
            self.ct_grid_dimensions,
            grid_information["grid_rotated"],
            fill_value=-1000.0,
        )
        return xp.reshape(
            values, (depth_range.shape[0], lateral_range.shape[0], lateral_range.shape[0])
        )

    def _init_beam(
        self, _dij: dict, _ct: CT, _cst: StructureSet, stf: SteeringInformation, i: int
    ) -> dict:
        """Prepare the geometry of one beam.

        Parameters
        ----------
        _dij : dict
            The dose influence matrix dictionary; unused.
        _ct : CT
            The CT; unused.
        _cst : StructureSet
            The structure set; unused.
        stf : SteeringInformation
            The steering information.
        i : int
            Index of the beam.

        Returns
        -------
        dict
            ``beam`` (the beam as a dict, its points float64 NumPy vectors),
            ``beam_index`` and ``rot_mat_system_T``, the world-to-BEV rotation.
        """
        beam = stf.beams[i].model_dump()

        # The geometry stays in float64 on the host, where each ray's transform is composed.
        beam["iso_center"] = _host_vector(beam["iso_center"])
        beam["source_point_bev"] = _host_vector(beam["source_point_bev"])
        for ray in beam["rays"]:
            if ray["target_point_bev"] is None:
                ray["target_point_bev"] = (
                    2 * _host_vector(ray["ray_pos_bev"]) - beam["source_point_bev"]
                )
            else:
                ray["target_point_bev"] = _host_vector(ray["target_point_bev"])

        rot_mat = get_beam_rotation_matrix(beam["gantry_angle"], beam["couch_angle"])
        return {
            "beam": beam,
            "beam_index": i,
            "rot_mat_system_T": np.asarray(to_numpy(rot_mat), dtype=np.float64),
        }

    def _init_ray(self, beam_info: dict[str, Any], j: int) -> dict[str, Any]:
        """Predict the dose of every beamlet of a ray.

        Samples the CT on the ray's beamlet-local grid, runs the model on all its
        beamlets (in batches of at most ``max_batch_size``) and interpolates the
        predicted cuboids at the dose-grid voxels they cover.

        Parameters
        ----------
        beam_info : dict[str, Any]
            The beam, as prepared by ``_init_beam``.
        j : int
            Index of the ray.

        Returns
        -------
        dict[str, Any]
            The ray with ``num_of_bixels``, ``ct_cuboid`` and ``_precomputed_bixels``,
            the sparse contribution of each beamlet, read by ``_compute_bixel``.
        """
        ray = beam_info["beam"]["rays"][j].copy()
        ray["num_of_bixels"] = len(ray["beamlets"])
        if ray["num_of_bixels"] == 0:
            ray["_precomputed_bixels"] = []
            return ray

        ray["iso_center"] = beam_info["beam"]["iso_center"]
        ray["source_point_bev"] = beam_info["beam"]["source_point_bev"]
        ray["rotation_mat"] = beam_info["rot_mat_system_T"]
        self._extract_ct_cuboid(ray)

        xp = self.xp
        local_points_masked = self.grid_information["local_points_masked"]
        ix_mask = self.grid_information["ix_mask"]
        depth = self.grid_information["depth_range"]
        lateral = self.grid_information["lateral_range"]
        pts_eval = xp.stack(
            (local_points_masked[:, 1], local_points_masked[:, 0], local_points_masked[:, 2]),
            axis=1,
        )
        vdose_sub = self._vdose_grid_mask[ix_mask]

        num_bixels = ray["num_of_bixels"]
        energies_arr = xp.asarray([b["energy"] for b in ray["beamlets"]], device=self.device)
        quantities = self.model_cfg.quantities

        # Model inference — split into mini-batches when max_batch_size is set
        if self.max_batch_size is not None and num_bixels > self.max_batch_size:
            chunks_by_qty = {q: [] for q in quantities}
            for start in range(0, num_bixels, self.max_batch_size):
                chunk_energies = energies_arr[start : start + self.max_batch_size]
                chunk_res = self._predict(ray["ct_cuboid"], chunk_energies)
                for q_name in quantities:
                    chunks_by_qty[q_name].append(chunk_res[q_name])
            cuboid_by_qty = {q: xp.concat(chunks, axis=0) for q, chunks in chunks_by_qty.items()}
        else:
            cuboid_by_qty = self._predict(ray["ct_cuboid"], energies_arr)

        # Interpolate each quantity's cuboid batch to world dose-grid points, with the
        # batch as trailing value axis: (N, B) values per quantity
        axes = (depth, lateral, lateral)
        vals_by_qty = {}
        for q_name, cuboid_batch in cuboid_by_qty.items():
            interp = ArrayBackendGridInterpolator(
                axes, xp.permute_dims(cuboid_batch, (1, 2, 3, 0))
            )
            vals_by_qty[q_name] = xp.astype(
                _interpolate_or_fill(interp, axes, pts_eval, fill_value=0.0), xp.float32
            )

        # Build per-bixel precomputed entries; non-zero mask is driven by physical_dose
        dose_vals = vals_by_qty["physical_dose"]
        precomputed = []
        for i in range(num_bixels):
            vals_dose = xp.where(vdose_sub, dose_vals[:, i], 0.0)
            nz_mask = vals_dose > self.dose_threshold
            entry = {"ix": ix_mask[nz_mask], "physical_dose": vals_dose[nz_mask]}
            for q_name in quantities[1:]:
                entry[q_name] = xp.where(vdose_sub, vals_by_qty[q_name][:, i], 0.0)[nz_mask]
            precomputed.append(entry)

        ray["_precomputed_bixels"] = precomputed
        return ray

    def _apply_scenario_to_ray(self, ray: dict, scenario: Scenario) -> dict:
        """Return the ray for a specific scenario without modifying depths.

        Parameters
        ----------
        ray : dict
            Base ray dictionary.
        scenario : Scenario
            The scenario; unused, since a setup shift -- the only supported
            uncertainty -- is already applied to the geometry the ray was built from.

        Returns
        -------
        dict
            A shallow copy of the input ray.
        """
        scen_ray = ray.copy()

        return scen_ray

    def _allocate_quantity_container(self, dij: dict[str, Any]):
        """Allocate dij containers for the quantities the model predicts beyond physical_dose."""
        extra = self.model_cfg.quantities[1:]
        if extra:
            dij = self._allocate_quantity_matrices(dij, extra)
            logger.info("Allocated containers for extra quantities: %s", extra)
        return dij

    def _read_model_config(self, model_dir: Path) -> DoseCalcModelConfig:
        """Judge a model by its ``model_config.json`` alone, before any of its code runs.

        Parameters
        ----------
        model_dir : Path
            The resolved model directory.

        Returns
        -------
        DoseCalcModelConfig
            The validated ``metadata.dose_calc`` section.

        Raises
        ------
        ValueError
            If the section is missing or invalid, or the model was trained for another
            radiation mode than the steering information's.
        ImportError
            If packages the model declares under ``metadata.requirements`` are missing.
        """
        raw_config, modality, missing = model_facts(model_dir)

        dose_calc = (raw_config.get("metadata") or {}).get("dose_calc")
        if not isinstance(dose_calc, dict):
            raise ValueError(
                f"Model '{self.model}' in '{model_dir}' has no 'metadata.dose_calc' section "
                "in its model_config.json, so it is no dose-calculation model."
            )
        try:
            model_cfg = DoseCalcModelConfig.model_validate(dose_calc)
        except ValidationError as exc:
            raise ValueError(
                f"Model '{self.model}' has an invalid 'metadata.dose_calc' section: {exc}"
            ) from exc

        radiation_mode = self._machine.radiation_mode
        if modality and modality != radiation_mode:
            raise ValueError(
                f"Model '{self.model}' was trained for radiation mode "
                f"'{modality}' but the plan uses '{radiation_mode}'."
            )
        if missing:
            raise ImportError(
                f"Model '{self.model}' requires {', '.join(missing)}, which is not installed."
            )
        return model_cfg

    def _check_training_data_compatibility(self, stf: SteeringInformation) -> None:
        """Warn when the machine or the energies differ from the model's training data."""
        config = self.model_cfg

        trained_machine = config.trained_machine
        if trained_machine and self._machine.name != trained_machine:
            logger.warning(
                "Machine '%s' differs from the machine used during model training ('%s'). "
                "Dose predictions may be less accurate.",
                self._machine.name,
                trained_machine,
            )

        energy_range = config.energy_range_mev
        if energy_range:
            all_energies = {e for beam in stf.beams for e in beam.energy_layers.keys()}
            out_of_range = sorted(
                e for e in all_energies if e < energy_range[0] or e > energy_range[1]
            )
            if out_of_range:
                logger.warning(
                    "STF contains energies outside the model's training range "
                    "[%g, %g] MeV: %s. Dose predictions may be less accurate.",
                    energy_range[0],
                    energy_range[1],
                    out_of_range,
                )

    def _init_dose_calc(self, ct: CT, cst: StructureSet, stf: SteeringInformation):
        """Prepare the CT interpolator, load the model and build its sampling grid.

        The model is placed on the device of the engine's arrays. The world
        coordinates of all dose-grid voxels are kept for ``_rotate_grid``.

        Parameters
        ----------
        ct : CT
            The CT.
        cst : StructureSet
            The structure set.
        stf : SteeringInformation
            The steering information.

        Returns
        -------
        dict
            The dose influence matrix scaffolding as produced by the base initializer.
        """
        dij = super()._init_dose_calc(ct, cst, stf)
        xp = self.xp

        # ct_phase is not a supported uncertainty, so every computed scenario uses this phase
        self.ct_cube_hu = ct_phase_images(ct.cube_hu)[self._reference_ct_phase]
        self._vdose_grid_mask = xp.asarray(self._vdose_grid_mask, device=self.device)

        # Positions and the CT are float32 throughout, on every backend (also JAX without
        # x64): the per-ray transforms keep float32 positions accurate to about 10 nm.
        sitk_grid = Grid.from_sitk_image(self.ct_cube_hu)
        self.ct_grid_dimensions = tuple(
            xp.asarray(axis, dtype=xp.float32, device=self.device)
            for axis in (sitk_grid.x, sitk_grid.y, sitk_grid.z)
        )
        ct_values = xp.asarray(
            np.transpose(sitk.GetArrayFromImage(self.ct_cube_hu), (2, 1, 0)),
            dtype=xp.float32,
            device=self.device,
        )
        self.ct_interpolator = ArrayBackendGridInterpolator(self.ct_grid_dimensions, ct_values)

        if not self.model:
            raise ValueError(
                "AIBeamletEngine requires a model, e.g. "
                "pln.prop_dose_calc = {'engine': 'AIBeamlet', 'model': '<name>'}."
            )

        if "/" not in self.model:
            raise ValueError(f"Model '{self.model}' is not a '<org>/<model>'.")

        # Resolving (and downloading) runs none of the model's code, so everything its
        # model_config.json tells is checked before the model is loaded.
        try:
            model_dir = resolve_model_dir(repo_id=self.model)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"AI model '{self.model}' is neither on disk nor could it be downloaded: {exc}"
            ) from exc
        self.model_cfg = self._read_model_config(model_dir)
        self._check_training_data_compatibility(stf)

        # Loading the resolved folder offline resolves nothing again. The modelhub still
        # decides from the folder's provenance whether its code may run: downloaded code
        # only once trusted.
        self.network, self.preprocessor, _ = load_model_with_dir(
            repo_id=self.model,
            local_dir=model_dir,
            offline=True,
            device=device_string(self.device),
        )

        if self.ensemble_size is not None:
            self.preprocessor.config["ensemble_size"] = self.ensemble_size

        dij = self._allocate_quantity_container(dij)

        all_idx = np.arange(self.dose_grid.num_voxels, dtype=int)
        self.coords_all = xp.asarray(
            linear_indices_to_grid_coordinates(all_idx, self.dose_grid, index_type="numpy"),
            dtype=xp.float32,
            device=self.device,
        )
        self._construct_grid()

        return dij

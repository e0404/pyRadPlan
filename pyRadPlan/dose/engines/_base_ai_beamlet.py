"""AI-driven pencil-beam dose engine, backed by the ``pyRadPlan.ai.modelhub`` model contract."""

import logging
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
    task_from_dir,
    task_from_name,
)
from pydantic import ConfigDict, Field

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


def _host_vector(value: Any) -> np.ndarray:
    """Return a point of the steering information as a float64 NumPy vector of length 3."""
    if array_api_compat.is_array_api_obj(value):
        value = to_numpy(value)
    return np.reshape(np.asarray(value, dtype=np.float64), (3,))


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
    - Model inference: model + preprocessor loaded via `pyRadPlan.ai.modelhub.load_model` from the
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
    - Input assembly and output scaling are delegated to the model's own preprocessor
        (`preprocess`/`postprocess`); multi-output models (e.g. dose+physical_dose_var,
        dose+LET) return a dict from `postprocess` keyed by quantity name.
    - Interop with SimpleITK/NumPy occurs where required; otherwise operations follow
        the configured array backend (`self.xp`).
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

    def __init__(self, pln=None):
        """Initialize the AI engine.

        Parameters
        ----------
        pln : Any, optional
            Plan-like object providing engine configuration, by default ``None``. If
            provided, engine properties are derived from it and the model named in
            `prop_dose_calc["model"]` is loaded (via `pyRadPlan.ai.modelhub.load_model`) at
            dose-calc init time.

        Returns
        -------
        None
        """
        self._model_modality = None
        super().__init__(pln)
        self.pln = pln

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
            Bixel dictionary with ``ix``, ``physical_dose``, and any second quantity.
        """
        bixel = self._init_bixel(ray, k)
        bixel.update(ray["_precomputed_bixels"][k])
        return bixel

    def _predict(self, ct_cuboid: Array, energies: Array) -> dict[str, Array]:
        """Run one model prediction and return per-quantity arrays in ``self.xp``.

        Delegates the inference procedure (forward pass(es), output scaling) to the
        model's own preprocessor's ``predict``, which always returns a
        ``{quantity_name: array}`` dict; this method only bridges array backends,
        which is generic across all models.

        Parameters
        ----------
        ct_cuboid : Array
            CT cuboid sampled on the beamlet-local grid, shared by the whole batch.
        energies : Array
            Per-bixel energies (batch dimension).

        Returns
        -------
        dict[str, Array]
            Quantity name -> predicted cuboid batch, in ``self.xp``.
        """
        xp = self.xp
        model_input = self.preprocessor.preprocess({"ct_cuboid": ct_cuboid, "energy": energies})
        result = self.preprocessor.predict(self.network, model_input)

        return {q_name: to_namespace(xp, value) for q_name, value in result.items()}

    def _init_bixel(self, curr_ray: dict, k: int) -> dict:
        """Initialize bixel bookkeeping and energy indexing.

        Parameters
        ----------
        curr_ray : dict
            Ray dictionary with entries such as ``beamlets``, energy, MU constraints, etc.
        k : int
            Index of the bixel inside ``curr_ray['beamlets']``.

        Returns
        -------
        dict
            The bixel dictionary augmented with indices (beam/ray/bixel), MU limits,
            particles-per-MU, and the machine energy index used for lookups.
        """
        # A copy, so the predicted arrays are not kept alive in the beam for the whole beam.
        bixel = curr_ray["beamlets"][k].copy()
        bixel["beam_index"] = curr_ray["beam_index"]
        bixel["ray_index"] = curr_ray["ray_index"]
        bixel["bixel_index"] = k

        bixel["min_mu"] = 0
        if "min_mu" in curr_ray:
            bixel["min_mu"] = curr_ray["min_mu"][k]

        bixel["max_mu"] = float("inf")
        if "max_mu" in curr_ray:
            bixel["max_mu"] = curr_ray["max_mu"][k]

        bixel["num_particles_per_mu"] = 1e6
        if "num_particles_per_mu" in curr_ray:
            bixel["num_particles_per_mu"] = curr_ray["num_particles_per_mu"][k]

        # Find energy index in base data
        energy = curr_ray["beamlets"][k]["energy"]
        energy_ix = self._machine.get_energy_index(energy, 4)

        if energy_ix.size > 1:
            raise ValueError("Multiple energies found in base data for one bixel!")

        xp = self.xp
        energy_ix = xp.asarray(energy_ix, dtype=xp.int64, device=self.device)

        bixel["energy_ix"] = energy_ix

        return bixel

    def _construct_grid(self):
        xp = self.xp
        # Load grid parameters from model config
        grid_params = self.model_cfg.grid_parameters
        spacing = grid_params.spacing
        lateral_range = grid_params.lateral_range
        lateral_half_width = lateral_range / 2
        depth_offset_start = grid_params.depth_offset_start
        depth_range = grid_params.depth_range

        # coordinates in bev
        y_start = depth_offset_start
        y_end = depth_offset_start + depth_range

        lateral_range = xp.arange(
            -lateral_half_width + spacing / 2,
            lateral_half_width + spacing / 2,
            spacing,
            dtype=xp.float32,
            device=self.device,
        )
        depth_range = xp.arange(y_start, y_end, spacing, dtype=xp.float32, device=self.device)

        depth_vals, lateral_x_vals, lateral_z_vals = array_meshgrid(
            depth_range, lateral_range, lateral_range, indexing="ij"
        )

        # Assign world coordinates: x=lateral_x, y=depth, z=lateral_z
        x = lateral_x_vals
        y = depth_vals
        z = lateral_z_vals

        self.grid = xp.stack(
            (xp.reshape(x, (-1,)), xp.reshape(y, (-1,)), xp.reshape(z, (-1,))),
            axis=-1,
        )

        # We reduce computation by first computing a bounding box of the beamlet in world coordinate,
        # then filtering world points before the full transformation.

        # Define 8 corners of the cuboid in local beamlet frame
        lx_min, lx_max = lateral_range[0] - spacing / 2, lateral_range[-1] + spacing / 2
        ly_min, ly_max = depth_range[0] - spacing / 2, depth_range[-1] + spacing / 2
        lz_min, lz_max = lateral_range[0] - spacing / 2, lateral_range[-1] + spacing / 2

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
            dtype=lateral_range.dtype,
            device=self.device,
        )

        self.grid_information = {
            "lateral_range": lateral_range,
            "depth_range": depth_range,
            "spacing": spacing,
        }

    def _extract_ct_cuboid(self, ray: dict[str, Any], beam_info) -> None:
        """Sample CT on the beamlet-local grid constructed for this ray.

        Method
        - Constructs a regular local grid (depth, lateral, lateral) aligned with the current
          beamlet geometry and transforms it to world coordinates.
        - Interpolates the CT on this transformed grid.
        - Stores the resulting cuboid into ``ray['ct_cuboid']`` for downstream model use.

        Parameters
        ----------
        ray : dict[str, Any]
            Ray dictionary to be augmented with the sampled CT cuboid.
        beam_info : dict
            Per-beam precomputed geometry and ray-tracing information used to build the grid.

        Returns
        -------
        None
            The function mutates ``ray`` in place by inserting ``ray['ct_cuboid']`` and
            updates ``self.grid_information`` with the constructed grids and transforms.
        """

        self._rotate_grid(ray, beam_info)

        interpolated_values_ct = self._extract_sitk_grid_and_interpolate(self.grid_information)
        ray["ct_cuboid"] = interpolated_values_ct

    def _rotate_grid(self, ray: dict, beam_info: dict) -> dict:
        """Construct a beamlet-local regular sampling grid and transforms.

        Parameters
        ----------
        ray : dict
            Current ray dictionary containing rotation matrices, BEV positions and isocenter.
        beam_info : dict
            Beam-level geometry and traced coordinate arrays (e.g., BEV coordinates).

        Returns
        -------
        dict
            A dictionary with
            - `lateral_range` (Array[float]): lateral coordinates used for X/Z axes,
            - `depth_range` (Array[float]): longitudinal coordinates (Y axis),
            - `spacing` (float): grid spacing used for all axes,
            - `grid` (Array[N,3]): beamlet-local grid points, (x=lateral, y=depth, z=lateral),
            - `grid_rotated` (Array[N,3]): corresponding world coordinates,
            - `R_world_to_BEV`, `R_BEV_to_beamlet` (Array[3,3]): transforms between frames,
            - `iso_center`, `source_point_bev` (Array[3]): translation references.
        """
        source = ray["source_point_bev"]
        a = -source / np.linalg.norm(source)
        b = ray["target_point_bev"] - source
        b = b / np.linalg.norm(b)

        if not np.all(a == b):
            cross_ab = np.cross(a, b)
            ssc_matrix = np.array(
                [
                    [0.0, -cross_ab[2], cross_ab[1]],
                    [cross_ab[2], 0.0, -cross_ab[0]],
                    [-cross_ab[1], cross_ab[0], 0.0],
                ]
            )
            derived_rot_mat = (
                np.eye(3)
                + ssc_matrix
                + ssc_matrix @ ssc_matrix * (1 - a @ b) / (np.linalg.norm(cross_ab) ** 2)
            )
        else:
            derived_rot_mat = np.eye(3)

        # Both rotations and the source offset compose into one affine map, so no point is
        # shifted by the source position (10 m away) and back, which would cost float32
        # precision: local = (world - iso) @ world_to_local + offset
        world_to_local = ray["rotation_mat"] @ derived_rot_mat
        offset = source - source @ derived_rot_mat

        xp = self.xp
        dtype = self.coords_all.dtype
        iso_center = xp.asarray(ray["iso_center"], dtype=dtype, device=self.device)
        offset = xp.asarray(offset, dtype=dtype, device=self.device)
        world_to_local = xp.asarray(world_to_local, dtype=dtype, device=self.device)
        local_to_world = world_to_local.T

        grid_rotated = (self.grid - offset) @ local_to_world + iso_center
        corners_world = (self.corners_local - offset) @ local_to_world + iso_center

        min_world = xp.min(corners_world, axis=0)
        max_world = xp.max(corners_world, axis=0)

        # Filter world coordinates
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

        # Use candidate points for precise transformation
        local_points = (candidate_points - iso_center) @ world_to_local + offset

        # Store transforms and masked local points in grid information for downstream use
        self.grid_information.update(
            {
                "grid_rotated": grid_rotated,
                "iso_center": ray["iso_center"],
                "source_point_bev": ray["source_point_bev"],
                "local_points_masked": local_points,
                "ix_mask": candidate_indices,
            }
        )

    def _extract_sitk_grid_and_interpolate(self, grid_information: dict):
        """Interpolate a volume (CT) on a provided world-space sampling grid.

        Parameters
        ----------
        grid_information : dict
            Dictionary returned by ``_construct_grid`` containing `grid_rotated`,
            `lateral_range` and `depth_range`.
        ct_interpolation : bool, optional
            If ``True``, assigns HU air value (-1000) to out-of-volume samples; otherwise
            zeros are kept. Defaults to ``False``.

        Returns
        -------
        Array
            Interpolated values reshaped to (depth, lateral, lateral), matching the
            regular local grid axis order for downstream model consumption.
        """
        xp = array_api_compat.array_namespace(grid_information["grid_rotated"])

        lateral_range = grid_information["lateral_range"]
        depth_range = grid_information["depth_range"]
        grid_rotated = grid_information["grid_rotated"]

        values = _interpolate_or_fill(
            self.ct_interpolator, self.ct_grid_dimensions, grid_rotated, fill_value=-1000.0
        )
        # Meshgrid now built with shape (depth, lateral, lateral); direct reshape yields desired ordering
        values_reshaped = xp.reshape(
            values, (depth_range.shape[0], lateral_range.shape[0], lateral_range.shape[0])
        )

        # debug: visualize ct cuboid (network input)
        # import matplotlib.pyplot as plt
        # z_slice = int(np.round(len(grid_information["lateral_range"]) / 2))
        # plt.imshow(values_reshaped[:, :, z_slice], cmap="gray")

        return xp.asarray(values_reshaped, device=self.device)

    def _init_beam(
        self, _dij: dict, ct: CT, _cst: StructureSet, stf: SteeringInformation, i
    ) -> dict:
        """
        Initialize the beam for pencil beam dose calculation using AI models.

        Parameters
        ----------
        dij : dict
            The dose influence matrix dictionary.
        ct : CT
            The CT object.
        _cst : StructureSet
            The structure set object. Unused here
        stf : SteeringInformation
            The steering information object.
        i : int
            Index of the beam.

        Returns
        -------
        dict
            Beam Information dictionary
        """
        # TODO: here .model_dump() is used to get beam as dict. Its possible to change...
        # ... the ray_tracing to handle this as the model
        beam_info = {"beam": stf.beams[i].model_dump(), "beam_index": i}

        # The geometry stays in float64 on the host, where each ray's transform is composed.
        for key in ["iso_center", "source_point_bev", "source_point"]:
            beam_info["beam"][key] = _host_vector(beam_info["beam"][key])
        for ray in beam_info["beam"]["rays"]:
            for key in ["ray_pos", "ray_pos_bev", "target_point", "target_point_bev"]:
                if ray[key] is not None:
                    ray[key] = _host_vector(ray[key])

        rot_mat = get_beam_rotation_matrix(
            beam_info["beam"]["gantry_angle"], beam_info["beam"]["couch_angle"]
        )
        beam_info["rot_mat_system_T"] = np.asarray(to_numpy(rot_mat), dtype=np.float64)

        # Check existence of target_points
        if any(r["target_point"] is None for r in beam_info["beam"]["rays"]):
            logger.debug("Missing target_point in rays. Calculating target points...")
            for ray in beam_info["beam"]["rays"]:
                ray["target_point"] = 2 * ray["ray_pos"] - beam_info["beam"]["source_point"]
                ray["target_point_bev"] = (
                    2 * ray["ray_pos_bev"] - beam_info["beam"]["source_point_bev"]
                )

        logger.info("Done.")

        return beam_info

    def _init_ray(self, beam_info: dict[str], j: int) -> dict[str]:
        """Initialize per-ray geometry and CT sampling for the AI engine.

        Performs forward pass for all bixel of current ray and stores the outputs

        Parameters
        ----------
        beam_info : dict[str]
            Beam information dictionary produced in beam initialization, containing
            rotation matrices, BEV coordinates, and ray metadata.
        j : int
            Index of the ray to initialize.

        Returns
        -------
        dict[str]
            Ray dictionary augmented with indices, geometry, an attached
            `ct_cuboid` sampled on the local grid for later model inference
            and saves the precomputed batch for this ray.
        """
        ray = beam_info["beam"]["rays"][j].copy()
        ray["beam_index"] = beam_info["beam_index"]
        ray["ray_index"] = j
        ray["iso_center"] = beam_info["beam"]["iso_center"]

        if "num_of_bixels_per_ray" in beam_info["beam"]:
            ray["num_of_bixels"] = int(beam_info["beam"]["num_of_bixels_per_ray"][j])
        else:
            # Fallback: use the actual number of beamlets on this ray
            # This is needed for machines like "Focused" that don't precompute counts.
            ray["num_of_bixels"] = len(ray.get("beamlets", [])) or 0

        ray["source_point_bev"] = beam_info["beam"]["source_point_bev"]
        ray["sad"] = beam_info["beam"]["sad"]
        ray["bixel_width"] = beam_info["beam"]["bixel_width"]

        # self._get_ray_geometry_from_beam(ray, beam_info)

        ray["rotation_mat"] = beam_info["rot_mat_system_T"]
        ray["radiation_mode"] = beam_info["beam"]["radiation_mode"]
        ray["source_point"] = beam_info["beam"]["source_point"]
        ray["machine"] = beam_info["beam"]["machine"]
        self._extract_ct_cuboid(ray, beam_info)

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

        energies = [b.get("energy") for b in ray.get("beamlets", [])]
        B = len(energies)
        assert B == ray["num_of_bixels"], "Inconsistent bixel count in ray!"
        assert B > 0, "No bixels found in ray!"

        energies_arr = xp.asarray(energies, device=self.device)
        quantities = self.model_cfg.quantities

        # Model inference — split into mini-batches when max_batch_size is set
        if self.max_batch_size is not None and B > self.max_batch_size:
            chunks_by_qty = {q: [] for q in quantities}
            for start in range(0, B, self.max_batch_size):
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
        for i in range(B):
            vals_dose = xp.where(vdose_sub, dose_vals[:, i], 0.0)
            nz_mask = vals_dose > self.dose_threshold
            entry = {
                "ix": xp.asarray(ix_mask[nz_mask], dtype=xp.int64, device=self.device),
                "physical_dose": xp.asarray(
                    vals_dose[nz_mask], dtype=xp.float32, device=self.device
                ),
            }
            for q_name in quantities[1:]:
                vals_q = xp.where(vdose_sub, vals_by_qty[q_name][:, i], 0.0)
                entry[q_name] = xp.asarray(vals_q[nz_mask], dtype=xp.float32, device=self.device)
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
        """Allocate dij containers for all quantities declared by the model beyond physical_dose."""
        extra = [q for q in self.model_cfg.quantities if q != "physical_dose"]
        if extra:
            dij = self._allocate_quantity_matrices(dij, extra)
            logger.info("Allocated containers for extra quantities: %s", extra)
        return dij

    def _check_training_data_compatibility(self, stf: SteeringInformation) -> None:
        config = self.model_cfg
        if config is None:
            return

        def _get(key, default=None):
            if isinstance(config, dict):
                return config.get(key, default)
            return getattr(config, key, default)

        pln_radiation_mode = getattr(self.pln, "radiation_mode", None)
        radiation_mode = self._model_modality
        if pln_radiation_mode and radiation_mode and radiation_mode != pln_radiation_mode:
            raise ValueError(
                f"Model '{self.model}' was trained for radiation mode "
                f"'{radiation_mode}' but the plan uses '{pln_radiation_mode}'."
            )

        trained_machine = _get("trained_machine")
        if trained_machine and self._machine.name != trained_machine:
            logger.warning(
                "Machine '%s' differs from the machine used during model training ('%s'). "
                "Dose predictions may be less accurate.",
                self._machine.name,
                trained_machine,
            )

        energy_range = _get("energy_range_mev")
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
        """Initialize CT grid context and the AI model for dose calculation.

        Method
        - Calls the base initializer to set up common state and the dose grid.
        - Extracts SimpleITK image arrays and builds a compatible Grid for interpolation.
        - Loads the configured AI model and wires the current array backend to it.
        - Precomputes world coordinates for all dose-grid voxels (as linear indices) used
          later by interpolation in `_compute_bixel`.

        Parameters
        ----------
        ct : CT
            The CT object (SimpleITK-backed) providing voxel data and geometry.
        cst : StructureSet
            Structure set; unused here but required by the interface.
        stf : SteeringInformation
            Steering information including beams and rays for the plan.

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

        self.sitk_xp = xp.asarray(sitk.GetArrayFromImage(self.ct_cube_hu), device=self.device)
        sitk_grid = Grid.from_sitk_image(self.ct_cube_hu)
        self.ct_grid_dimensions = (
            xp.asarray(sitk_grid.x, device=self.device),
            xp.asarray(sitk_grid.y, device=self.device),
            xp.asarray(sitk_grid.z, device=self.device),
        )

        ct_values = xp.permute_dims(self.sitk_xp, (2, 1, 0))
        if not xp.isdtype(ct_values.dtype, "real floating"):
            # integer HU (e.g. int16) cannot be mixed with float weights on every backend
            ct_values = xp.astype(ct_values, xp.float32)
        self.ct_interpolator = ArrayBackendGridInterpolator(self.ct_grid_dimensions, ct_values)

        if not self.model:
            raise ValueError(
                "AIBeamletEngine requires a model, e.g. "
                "pln.prop_dose_calc = {'engine': 'AIBeamlet', 'model': '<name>'}."
            )

        if "/" not in self.model:
            raise ValueError(f"Model '{self.model}' is not a '<org>/<model>'.")

        # The modelhub refuses downloaded code nobody trusted before importing any of it, also
        # for a folder it downloaded, so the engine must not load a resolved folder directly.
        try:
            self.network, self.preprocessor, model_dir = load_model_with_dir(
                repo_id=self.model, device=device_string(self.device)
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"AI model '{self.model}' is neither on disk nor could it be downloaded: {exc}"
            ) from exc
        raw_config, self._model_modality, _ = model_facts(model_dir)
        self.model_cfg = DoseCalcModelConfig.model_validate(
            raw_config.get("metadata", {}).get("dose_calc", {})
        )

        if self.ensemble_size is not None:
            self.preprocessor.config["ensemble_size"] = self.ensemble_size

        self._check_training_data_compatibility(stf)

        dij = self._allocate_quantity_container(dij)

        # leave numpy here for now, since linear_indices_to_grid_coordinates moves to numpy anyway
        all_idx = np.arange(self.dose_grid.num_voxels, dtype=int)
        self.coords_all = xp.asarray(
            linear_indices_to_grid_coordinates(all_idx, self.dose_grid, index_type="numpy"),
            device=self.device,
        )
        self._construct_grid()

        return dij


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
    quantities: List[str] = Field(default_factory=lambda: ["physical_dose"])
    radiation_mode: Optional[str] = None
    energy_range_mev: Optional[Tuple[float, float]] = None
    trained_machine: Optional[str] = None

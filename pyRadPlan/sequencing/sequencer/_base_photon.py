"""Base class for photon sequencing."""

import logging
import math
import numpy as np
from abc import abstractmethod
from typing import Optional, Union

from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.result import PlanResult, ApertureInfo, validate_result
from pyRadPlan.stf import SteeringInformation, Beam
from pyRadPlan.machines import MLC, Jaw, load_from_name, validate_machine
from ._base import Sequencer

logger = logging.getLogger(__name__)


class PhotonSequencer(Sequencer):
    SEQUENCED_SUFFIX = "_sequenced"

    def __init__(self, pln: Union[Plan, dict] = None):
        self.num_of_mlc_leaf_pairs = 80
        self.leaf_width: Optional[float] = None
        self.mlc_orientation = "X"
        super().__init__(pln)

        if pln is not None:
            self._load_mlc_from_machine(pln)

    def _load_mlc_from_machine(self, pln: Union[Plan, dict]) -> None:
        """Load leaf width / leaf count / orientation from the plan's machine, if it defines one.

        Falls back to the legacy defaults (``leaf_width=None``, i.e. sequence
        directly on the ray grid, ``num_of_mlc_leaf_pairs=80``, and
        ``mlc_orientation="X"``) whenever the machine can't be resolved or
        doesn't define an MLC.
        """
        try:
            _pln = validate_pln(pln)
            machine = _pln.machine
            if isinstance(machine, str):
                machine = load_from_name(_pln.radiation_mode, machine)
            else:
                machine = validate_machine(machine)
        except Exception:
            logger.debug(
                "Could not resolve machine for MLC restrictions; using defaults.", exc_info=True
            )
            return

        mlc = getattr(machine, "mlc", None)
        if mlc is not None:
            self.leaf_width = mlc.leaf_width
            self.num_of_mlc_leaf_pairs = mlc.number_of_elements
            self.mlc_orientation = mlc.device_orientation

    @abstractmethod
    def sequence(self, result: PlanResult, stf: SteeringInformation, dij) -> PlanResult:
        """
        Execute the sequencing algorithm.

        Parameters
        ----------
        result : object
            The result object or dictionary to store sequencing results.
        stf : object
            The steering information structure.
        dij (Optional): object
            The dij matrix or related input.

        Returns
        -------
        object
            The updated PlanResult after sequencing.
        """
        pass

    def _leaf_pitch(self, bixel_width: float) -> Optional[float]:
        """Return the leaf pitch to sequence shapes on, or None to use the ray grid as-is.

        None is returned both when no machine MLC is configured and when the
        configured leaf width matches the ray/bixel width so callers skip
        resampling entirely and sequence directly on the ray grid.
        """
        if self.leaf_width is None:
            return None
        if self.leaf_width < bixel_width and not math.isclose(self.leaf_width, bixel_width):
            raise ValueError(
                f"MLC leaf_width ({self.leaf_width} mm) is narrower than the ray/bixel "
                f"width ({bixel_width} mm). The sequencer cannot resolve leaves finer "
                "than the dose-calculation grid."
            )
        if math.isclose(self.leaf_width, bixel_width):
            return None
        return self.leaf_width

    def _central_leaf_pair(self, bixel_width: float) -> float:
        """1-based index of the leaf pair on isocenter (see ``MLC.is_active_leaf_pair``).

        On a physical leaf grid (defined in machine MLC) the device is symmetric about
        isocenter, so an even leaf count puts a leaf boundary, an odd count puts a leaf
        center on isocenter.
        Sequencing directly on the ray grid (no machine MLC or leaf width equal to bixel
        width) keeps matRad's ``ceil(N/2)``, with a leaf centre on isocenter.
        """
        if self._leaf_pitch(bixel_width) is None:
            return int(math.ceil(self.num_of_mlc_leaf_pairs / 2.0))
        return (self.num_of_mlc_leaf_pairs + 1) / 2.0

    @staticmethod
    def _leaf_grid_centers(
        field_min: float, field_max: float, leaf_width: float, num_of_leaf_pairs: int
    ) -> np.ndarray:
        """
        Physical MLC leaf row centers (the "MLC grid") covering ``(field_min, field_max)``.

        The device is symmetric about isocenter: with an even number of leaf pairs
        a leaf boundary sits on isocenter (centers at ``(j + 0.5) * leaf_width``),
        with an odd number a leaf center does (centers at ``j * leaf_width``).
        Every leaf row overlapping the open interval is returned, so no part of
        the field is left uncovered.

        Parameters
        ----------
        field_min, field_max : float
            Physical extent (mm) of the beam's native bixel field (outer bixel
            edges) along the leaf-stack direction.
        leaf_width : float
            Physical leaf pitch (mm).
        num_of_leaf_pairs : int
            Number of leaf pairs of the device, which fixes the lattice offset.

        Returns
        -------
        np.ndarray
            Ascending physical leaf-row centers (mm), length dim_stack_leaf.
        """
        offset = 0.5 if num_of_leaf_pairs % 2 == 0 else 0.0
        # Row j spans [(j + offset - 0.5) w, (j + offset + 0.5) w]; the tolerance
        # drops rows that only touch the field at an edge.
        j_start = math.floor(field_min / leaf_width - offset - 0.5 + 1e-9) + 1
        j_end = math.ceil(field_max / leaf_width - offset + 0.5 - 1e-9) - 1
        return (np.arange(j_start, j_end + 1, dtype=float) + offset) * leaf_width

    @staticmethod
    def _interp_rows(
        matrix: np.ndarray, src_row_pos: np.ndarray, dst_row_pos: np.ndarray
    ) -> np.ndarray:
        """
        Linearly interpolate a matrix row-wise from one leaf-stack grid to another.

        Used both to resample the native ray/bixel grid down onto the
        physical MLC leaf grid, and to project a sequenced quantity back up
        from the leaf grid onto the native ray grid — the same operation in
        either direction, just with ``src_row_pos``/``dst_row_pos`` swapped.
        Query points outside ``src_row_pos`` are clamped to the nearest edge
        value (``numpy.interp``'s default), which only matters at a field's
        edge, where the coarser grid can't resolve anything finer anyway.

        Parameters
        ----------
        matrix : np.ndarray
            ``(len(src_row_pos), dim_travel)`` matrix sampled at ``src_row_pos``.
        src_row_pos : np.ndarray
            Ascending physical stack-axis coordinate of each row of ``matrix``.
        dst_row_pos : np.ndarray
            Ascending physical stack-axis coordinate to interpolate onto.

        Returns
        -------
        np.ndarray
            ``(len(dst_row_pos), dim_travel)`` interpolated matrix.
        """
        dim_travel = matrix.shape[1]
        resampled = np.empty((len(dst_row_pos), dim_travel))
        for col in range(dim_travel):
            resampled[:, col] = np.interp(dst_row_pos, src_row_pos, matrix[:, col])
        return resampled

    @staticmethod
    def _project_rows_to_positions(
        matrix: np.ndarray,
        leaf_row_pos: np.ndarray,
        stack_ray_pos: np.ndarray,
        travel_ind_pos: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a ``(dim_stack_leaf, dim_travel)`` matrix at each ray.

        Each ray is evaluated at its own (stack position, travel column).
        """
        result = np.empty(len(stack_ray_pos))
        for col in np.unique(travel_ind_pos):
            rays_in_col = travel_ind_pos == col
            result[rays_in_col] = np.interp(
                stack_ray_pos[rays_in_col], leaf_row_pos, matrix[:, col]
            )
        return result

    def _project_leaf_grid_to_rays(self, matrix: np.ndarray, tmp_fluence_dict: dict) -> np.ndarray:
        """Project a quantity computed back onto each native ray.

        Plain native-resolution flat gather when no MLC grid was configured
        (``leaf_row_pos`` is None), else an interpolated projection from the
        leaf grid.
        """
        leaf_row_pos = tmp_fluence_dict.get("leaf_row_pos")
        if leaf_row_pos is None:
            return matrix.flatten(order="C")[tmp_fluence_dict["indices"]]
        return self._project_rows_to_positions(
            matrix,
            leaf_row_pos,
            tmp_fluence_dict["stack_ray_pos"],
            tmp_fluence_dict["travel_ind_pos"],
        )

    def _inititialize_beam_fluence_matrix(
        self, beam: Beam, w_of_curr_beams: np.ndarray = None
    ) -> dict:
        """
        Initialize the fluence/ray-map matrices for a given beam.

        Parameters
        ----------
        beam : object
            The current beam object.
        w_of_curr_beams : numpy.ndarray
            The fluence weights for the current beam.

        Returns
        -------
        dict
            A dictionary containing the initialized fluence matrix and related parameters.
        """
        # Get ray positions.
        travel_axis, stack_axis = (0, 2) if self.mlc_orientation == "X" else (2, 0)
        travel_ray_pos = np.array([ray.ray_pos_bev[travel_axis] for ray in beam.rays])
        stack_ray_pos = np.array([ray.ray_pos_bev[stack_axis] for ray in beam.rays])

        min_travel, max_travel = travel_ray_pos.min(), travel_ray_pos.max()
        min_stack, max_stack = stack_ray_pos.min(), stack_ray_pos.max()

        bixel_width = beam.bixel_width
        dim_of_fluence_mx_travel = int((max_travel - min_travel) / bixel_width + 1)
        dim_of_fluence_mx_stack = int((max_stack - min_stack) / bixel_width + 1)

        travel_ind_pos = ((travel_ray_pos - min_travel) / bixel_width).astype(int)
        stack_ind_pos = ((stack_ray_pos - min_stack) / bixel_width).astype(int)
        indices = stack_ind_pos * dim_of_fluence_mx_travel + travel_ind_pos

        # Bixel indices identify physical dij columns 1:1 and must stay at native
        # ray resolution even when the ray/presence map below gets resampled
        # onto a coarser physical leaf grid (several native rows can then
        # share one leaf row).
        native_indices = indices
        native_dim_stack = dim_of_fluence_mx_stack

        # Physical MLC leaf pitch to resample onto, if the machine defines one
        # that differs from the ray/bixel grid; None means sequence directly
        # on the ray grid.
        leaf_width = self._leaf_pitch(bixel_width)
        native_row_z = min_stack + np.arange(dim_of_fluence_mx_stack) * bixel_width

        # Ray/presence map: always built, since lim_l/lim_r (and thus the
        # aperture info) need it regardless of whether weights are given.
        ray_map = np.zeros((dim_of_fluence_mx_stack, dim_of_fluence_mx_travel))
        ray_map.flat[indices] = 1

        dim_stack = dim_of_fluence_mx_stack
        row_pos = native_row_z
        leaf_row_pos = None
        if leaf_width is not None:
            leaf_row_pos = self._leaf_grid_centers(
                min_stack - bixel_width / 2.0,
                max_stack + bixel_width / 2.0,
                leaf_width,
                self.num_of_mlc_leaf_pairs,
            )
            device_half = self.num_of_mlc_leaf_pairs * leaf_width / 2.0
            rows_min = leaf_row_pos.min() - leaf_width / 2.0
            rows_max = leaf_row_pos.max() + leaf_width / 2.0
            if rows_min < -device_half - 1e-9 or rows_max > device_half + 1e-9:
                raise ValueError(
                    f"Beam field (leaf rows z={rows_min:.1f}..{rows_max:.1f} mm) "
                    "does not fit within the machine's MLC coverage "
                    f"(z={-device_half:.1f}..{device_half:.1f} mm for "
                    f"{self.num_of_mlc_leaf_pairs} leaf pairs at {leaf_width:.1f} mm leaf width, "
                    "centered on isocenter). "
                    "The field does not fit within the physical collimator."
                )
            ray_map = self._interp_rows(ray_map, native_row_z, leaf_row_pos)
            # > 0.0 to ensure the leaf travel limits cover every bixel the fluence can open
            ray_map = (ray_map > 0.0).astype(float)
            dim_stack = ray_map.shape[0]
            min_stack, max_stack = leaf_row_pos.min(), leaf_row_pos.max()
            row_pos = leaf_row_pos

        init_param = {
            "ray_map": ray_map,
            "min_travel": min_travel,
            "max_travel": max_travel,
            "min_stack": min_stack,
            "max_stack": max_stack,
            "bixel_width": bixel_width,
            "row_pos": row_pos,
            "native_indices": native_indices,
            "native_dim_stack": native_dim_stack,
            "indices": indices,
            "dim_travel": dim_of_fluence_mx_travel,
            "dim_stack": dim_stack,
            "travel_axis": travel_axis,
            "stack_axis": stack_axis,
        }

        if w_of_curr_beams is not None:
            fluence_mx = np.zeros((dim_of_fluence_mx_stack, dim_of_fluence_mx_travel))
            for xp, zp, weight in zip(travel_ind_pos, stack_ind_pos, w_of_curr_beams):
                fluence_mx[zp, xp] = weight

            if leaf_row_pos is not None:
                fluence_mx = self._interp_rows(fluence_mx, native_row_z, leaf_row_pos)

            cal_fac = np.max(fluence_mx)
            if cal_fac > 0:
                d_0 = np.round(fluence_mx / cal_fac * self.num_of_levels).astype(int)
            else:
                # A beam the optimization zeroed out has no peak to stratify against.
                d_0 = np.zeros_like(fluence_mx, dtype=int)
            d_k = d_0.copy()

            # d_0 never changes during shape decomposition, so project it onto
            # the native rays once, here, instead of the sequencing algorithms
            # each re-interpolating it themselves at the end of sequence().
            if leaf_row_pos is not None:
                d_0_native = self._project_rows_to_positions(
                    d_0, leaf_row_pos, stack_ray_pos, travel_ind_pos
                )
            else:
                d_0_native = d_0.flatten(order="C")[indices]

            init_param.update(
                {
                    "fluence_mx": fluence_mx,
                    "cal_fac": cal_fac,
                    "d_0": d_0,
                    "d_k": d_k,
                    "d_0_native": d_0_native,
                    "leaf_row_pos": leaf_row_pos,
                    "stack_ray_pos": stack_ray_pos,
                    "travel_ind_pos": travel_ind_pos,
                }
            )

        return init_param

    def _compute_leaf_limits(self, init_param: dict) -> tuple:
        """
        Compute the limits of the leaves based on the ray map.

        Parameters
        ----------
        init_param : dict
            A dictionary containing initialization parameters including the ray map,

        Returns
        -------
        tuple
            A tuple containing the left and right limits of the leaves for each layer.
        """
        ray_map = init_param["ray_map"]
        dim_stack = init_param["dim_stack"]
        min_travel = init_param["min_travel"]
        bw = init_param["bixel_width"]

        lim_l = np.full(dim_stack, np.nan)
        lim_r = np.full(dim_stack, np.nan)
        for n in range(dim_stack):
            cols = np.where(ray_map[n, :] != 0)[0]
            if cols.size > 0:
                lim_l[n] = (cols[0]) * bw + min_travel - 0.5 * bw
                lim_r[n] = (cols[-1]) * bw + min_travel + 0.5 * bw
            else:
                lim_l[n] = min_travel
                lim_r[n] = min_travel

        return lim_l, lim_r

    @staticmethod
    def _build_jaw_pair(mlc_window: list[float], travel_axis: int) -> list[Jaw]:
        """Build the static X/Y jaw pair bounding the beam's overall field window.

        Jaws don't move between segments in step-and-shoot delivery, so the
        same pair is reused for every shape of the beam.

        Parameters
        ----------
        mlc_window : list[float]
            ``[travel_min, travel_max, stack_min, stack_max]`` (the beam's
            ``MLCWindow``), on the leaf travel/stack axes rather than fixed BEV axes.
        travel_axis : int
            BEV index (0 or 2) of the leaf travel direction; identifies
            which of the two window axes is the BEV "X" axis vs. "Y" axis.
        """
        travel_min, travel_max, stack_min, stack_max = mlc_window
        travel_bounds = (travel_min, travel_max)
        stack_bounds = (stack_min, stack_max)
        bev_x_bounds = travel_bounds if travel_axis == 0 else stack_bounds
        bev_y_bounds = stack_bounds if travel_axis == 0 else travel_bounds

        def _jaw(orientation: str, bounds: tuple[float, float]) -> Jaw:
            lo, hi = bounds
            field_width = 2 * max(abs(bound) for bound in mlc_window)
            return Jaw(device_orientation=orientation, positions=[lo, hi], field_width=field_width)

        return [_jaw("X", bev_x_bounds), _jaw("Y", bev_y_bounds)]

    def _compute_beam_aperture_info(
        self,
        stf: SteeringInformation,
        sequencing,
        central_leaf_pair,
        bixel_ind_offset,
        vector_offset,
    ) -> list:
        """
        Compute the beam aperture information.

        Parameters
        ----------
        stf : SteeringInformation
            The steering information containing the beams.
        central_leaf_pair : int
            The index of the central leaf pair.
        bixel_ind_offset : int
            The offset for bixel indices.

        Returns
        -------
        list
            A list containing the beam aperture information.
        """
        info_beam = []
        bixel_width = stf.beams[0].bixel_width
        for i, beam in enumerate(stf.beams):
            init = sequencing["beam"][i]["_fluence_dict"]

            # Initialize limits and dimensions
            lim_l, lim_r = self._compute_leaf_limits(init)

            # Create bixelIndMap at native ray resolution: each physical bixel/dij
            # column needs its own unique index even when shapes below are
            # resampled onto a coarser physical leaf grid.
            bixel_ind_map = np.full((init["native_dim_stack"], init["dim_travel"]), np.nan)
            num_rays = len(beam.rays)
            bixel_ind_map.flat[init["native_indices"]] = np.arange(0, num_rays) + bixel_ind_offset
            bixel_ind_offset += num_rays

            beam_aperture = {}
            # TODO: This should in general be correct. If not, change .T
            beam_aperture["bixel_ind_map"] = bixel_ind_map.T
            # min_travel/min_stack are on the leaf axes, not fixed BEV axes,
            # so place them at their true BEV index.
            pos_of_corner_bixel = [0.0, 0.0, 0.0]
            pos_of_corner_bixel[init["travel_axis"]] = init["min_travel"]
            pos_of_corner_bixel[init["stack_axis"]] = init["min_stack"]
            beam_aperture["pos_of_corner_bixel"] = pos_of_corner_bixel

            # Process shapes for the current beam
            # initialize per‐shape lists in beam_aperture
            shapes_arr = sequencing["beam"][i]["shapes"]
            weights_arr = sequencing["beam"][i]["shapes_weight"]
            num_shapes = sequencing["beam"][i]["num_of_shapes"]
            beam_aperture["weight"] = list(weights_arr)
            beam_aperture["vector_offset"] = [
                vector_offset + m * init["dim_stack"] for m in range(num_shapes)
            ]

            # advance the global counter once
            vector_offset += num_shapes * init["dim_stack"]

            # Effective row pitch shapes were computed on: the machine's physical
            # leaf width when configured and applicable, else the ray/bixel width.
            leaf_pitch = self._leaf_pitch(bixel_width) or bixel_width

            # compute all mlc in one go
            mlc_sequence = []
            for m in range(num_shapes):
                shape_map = shapes_arr[:, :, m]
                mlc = MLC(
                    device_type=f"MLC{self.mlc_orientation}",
                    number_of_elements=self.num_of_mlc_leaf_pairs,
                )
                mlc.calculate_from_shape(
                    shape_map,
                    init["min_travel"],
                    init["row_pos"],
                    bixel_width,
                    lim_l,
                    lim_r,
                    leaf_width=leaf_pitch,
                )
                mlc_sequence.append(mlc)
            beam_aperture["mlc_sequence"] = mlc_sequence

            # Derive flattened per-beam fields from the MLC objects (single source of
            # truth = MLC methods). These stay on the beam sequence for matRad interop.
            beam_aperture["left_leaf_pos"] = [mlc.leaf_positions[:, 0] for mlc in mlc_sequence]
            beam_aperture["right_leaf_pos"] = [mlc.leaf_positions[:, 1] for mlc in mlc_sequence]
            # leaf pair positions (matRad's leafPairPos) holds row centres, the MLC
            # stores boundaries (row's lower edges)
            beam_aperture["leaf_pair_pos"] = (
                mlc_sequence[0].leaf_position_boundaries + mlc_sequence[0].leaf_width / 2.0
            )
            beam_aperture["is_active_leaf_pair"] = mlc_sequence[0].is_active_leaf_pair(
                central_leaf_pair
            )
            beam_aperture["num_of_active_leaf_pairs"] = mlc_sequence[0].num_of_active_leaf_pairs
            beam_aperture["central_leaf_pair"] = central_leaf_pair
            beam_aperture["lim_l"] = lim_l
            beam_aperture["lim_r"] = lim_r

            # Define MLC window using physical coordinates. The stack direction takes
            # whichever reaches further, the ray extent or the leaf rows actually used
            stack_min = init["min_stack"] - 0.5 * init["bixel_width"]
            stack_max = init["max_stack"] + 0.5 * init["bixel_width"]
            leaf_row_pos = init.get("leaf_row_pos")
            if leaf_row_pos is not None and len(leaf_row_pos) > 0:
                pitch = (
                    float(np.diff(leaf_row_pos).min())
                    if len(leaf_row_pos) > 1
                    else self.leaf_width
                )
                stack_min = min(stack_min, float(leaf_row_pos[0]) - 0.5 * pitch)
                stack_max = max(stack_max, float(leaf_row_pos[-1]) + 0.5 * pitch)

            beam_aperture["MLCWindow"] = [
                init["min_travel"] - 0.5 * init["bixel_width"],
                init["max_travel"] + 0.5 * init["bixel_width"],
                stack_min,
                stack_max,
            ]

            # Static jaw pair bounding the field
            jaw_pair = self._build_jaw_pair(beam_aperture["MLCWindow"], init["travel_axis"])
            beam_aperture["extra_blds"] = [list(jaw_pair) for _ in range(num_shapes)]

            info_beam.append(beam_aperture)
        return info_beam

    def _dao_aperture_info_to_vec(self, aperture_info: dict) -> tuple:
        """
        Convert aperture information to a vector and mapping matrix.

        Parameters
        ----------
        aperture_info : dict
            A dictionary containing the aperture information of the beams.

        Returns
        -------
        tuple
            A tuple containing the aperture vector, mapping matrix, and limits matrix.
        """
        total_shapes = aperture_info["total_num_of_shapes"]
        total_leaf_pairs = aperture_info["total_num_of_leaf_pairs"]
        vec_length = total_shapes + 2 * total_leaf_pairs

        # 1. initialize output vector
        aperture_vec = np.full(vec_length, np.nan, dtype=float)

        # 2. aperture weights
        offset = 0
        for beam in aperture_info["beam"]:
            for weight in beam["weight"]:
                aperture_vec[offset] = weight
                offset += 1

        # 3. left & right leaf positions
        offset = total_shapes
        for beam in aperture_info["beam"]:
            mlc_sequence = beam["mlc_sequence"]
            for mlc in mlc_sequence:
                n_active = mlc.num_of_active_leaf_pairs
                left_pos = mlc.leaf_positions[:, 0]
                right_pos = mlc.leaf_positions[:, 1]
                aperture_vec[offset : offset + n_active] = left_pos
                aperture_vec[offset + total_leaf_pairs : offset + total_leaf_pairs + n_active] = (
                    right_pos
                )
                offset += n_active

        # 4. mapping and limits
        mapping_mx = np.full((vec_length, 4), np.nan)
        lim_mx = np.full((vec_length, 2), np.nan)

        # weights bounds = [0, inf]
        lim_mx[:total_shapes, 0] = 0
        lim_mx[:total_shapes, 1] = np.inf

        # mapping for weights (store beam index)
        counter = 0
        for i, beam in enumerate(aperture_info["beam"], start=1):
            for _ in range(len(beam["mlc_sequence"])):
                mapping_mx[counter, 0] = i
                counter += 1

        # mapping and bounds for leaves
        shape_offset = 0
        counter = total_shapes
        for i, beam in enumerate(aperture_info["beam"], start=1):
            mlc_sequence = beam["mlc_sequence"]
            lim_l = beam["lim_l"]
            lim_r = beam["lim_r"]
            for j, mlc in enumerate(mlc_sequence):
                n_active = mlc.num_of_active_leaf_pairs
                for k in range(n_active):
                    mapping_mx[counter, 0] = i
                    mapping_mx[counter, 1] = j + 1 + shape_offset
                    mapping_mx[counter, 2] = j + 1
                    mapping_mx[counter, 3] = k + 1
                    lim_mx[counter, 0] = lim_l[k]
                    lim_mx[counter, 1] = lim_r[k]
                    counter += 1
            shape_offset += len(mlc_sequence)

        # replicate tail if any unused slots remain
        if counter < vec_length:
            mapping_mx[counter:] = mapping_mx[total_shapes:counter]
            lim_mx[counter:] = lim_mx[total_shapes:counter]

        return aperture_vec, mapping_mx, lim_mx

    def _build_beam_sequencing(
        self,
        shapes_list: list,
        shapes_weight_list: list,
        tmp_fluence_dict: dict,
        offset: int,
        num_of_rays_per_beam: int,
    ) -> dict:
        """
        Build the sequencing structure for a single beam.

        Parameters
        ----------
        shapes_list : list
            List of shape matrices.
        shapes_weight_list : list
            List of shape weights.
        tmp_fluence_dict : dict
            Dictionary containing fluence matrix information.
        offset : int
            Current bixel offset.
        num_of_rays_per_beam : int
            Number of rays in the current beam.

        Returns
        -------
        dict
            Beam sequencing dictionary.
        """
        # >0 ensures shapes ares binary (Xia's shapes scaled by level in matRad)
        if shapes_list:
            shapes_arr = (np.stack(shapes_list, axis=2) > 0).astype(np.uint8)
            shapes_w_arr = (
                np.array(shapes_weight_list) / self.num_of_levels * tmp_fluence_dict["cal_fac"]
            )
        else:
            # A beam with no fluence decomposes into no shapes at all. Emit one fully
            # closed shape of zero weight instead.
            shapes_arr = np.zeros(
                (tmp_fluence_dict["dim_stack"], tmp_fluence_dict["dim_travel"], 1), dtype=np.uint8
            )
            shapes_w_arr = np.zeros(1)
        num_shapes = shapes_arr.shape[2]

        # Beam sequencing structure
        beam_sequencing = {
            "num_of_shapes": num_shapes,
            "shapes": shapes_arr,
            "shapes_weight": shapes_w_arr,
            "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
            "stratified_fluence": tmp_fluence_dict["d_0"],
            "cal_fac": tmp_fluence_dict["cal_fac"],
            "_fluence_dict": tmp_fluence_dict,
        }

        # Compute the summed weighted shape for the beam
        beam_sum = np.zeros((tmp_fluence_dict["dim_stack"], tmp_fluence_dict["dim_travel"]))
        for j in range(num_shapes):
            beam_sum += beam_sequencing["shapes"][:, :, j] * beam_sequencing["shapes_weight"][j]
        beam_sequencing["sum_shapes"] = beam_sum

        return beam_sequencing

    def sequencing_to_aperture_info(self, sequencing, stf: SteeringInformation) -> ApertureInfo:
        """
        Convert the sequencing result to an ApertureInfo object.

        Parameters
        ----------
        sequencing : dict
            The sequencing result containing per-beam details.
        stf : object
            The steering information.

        Returns
        -------
        ApertureInfo : object
            A data structure representing the aperture information of the beams.
        """
        aperture_info = {}

        # MLC parameters
        bixel_width = stf.beams[0].bixel_width  # [mm]
        central_leaf_pair = self._central_leaf_pair(bixel_width)

        bixel_ind_offset = 0
        total_num_of_bixels = sum(beam.total_number_of_bixels for beam in stf.beams)
        total_num_of_shapes = sum(beam_seq["num_of_shapes"] for beam_seq in sequencing["beam"])
        vector_offset = total_num_of_shapes + 1
        aperture_info["beam"] = self._compute_beam_aperture_info(
            stf, sequencing, central_leaf_pair, bixel_ind_offset, vector_offset
        )

        # Global aperture information
        aperture_info["bixel_width"] = bixel_width
        aperture_info["num_of_MLC_leaf_pairs"] = self.num_of_mlc_leaf_pairs
        aperture_info["total_num_of_bixels"] = total_num_of_bixels
        aperture_info["total_num_of_shapes"] = total_num_of_shapes
        total_leaf_pairs = sum(
            sum(mlc.num_of_active_leaf_pairs for mlc in b["mlc_sequence"])
            for b in aperture_info["beam"]
        )
        aperture_info["total_num_of_leaf_pairs"] = total_leaf_pairs

        # Optimization vectors
        ap_vec, map_mx, lim_mx = self._dao_aperture_info_to_vec(aperture_info)
        aperture_info["aperture_vector"] = ap_vec
        aperture_info["mapping_mx"] = map_mx
        aperture_info["lim_mx"] = lim_mx

        return aperture_info

    def _result_with_sequenced_quantities(
        self, result_data: dict, fluence: np.ndarray, dij: Optional[Dij]
    ) -> PlanResult:
        """Validate *result_data*, adding quantities recomputed from *fluence*.

        Parameters
        ----------
        result_data : dict
            The dumped result to validate, already carrying the ``sequence``.
        fluence : np.ndarray
            The sequenced (deliverable) fluence.
        dij : Dij, optional
            Dose influence matrix.

        Returns
        -------
        PlanResult
            The validated result.
        """
        if dij is None:
            return validate_result(result_data)

        sequenced = dij.compute_result_ct_grid(fluence).model_dump()
        for field in ("quantities", "beam_quantities"):
            delivered = {
                f"{name}{self.SEQUENCED_SUFFIX}": value
                for name, value in (sequenced.get(field) or {}).items()
            }
            if delivered:
                result_data[field] = {**(result_data.get(field) or {}), **delivered}

        return validate_result(result_data)

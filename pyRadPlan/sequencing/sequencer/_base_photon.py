"""Base class for photon sequencing."""

import logging
import math
import numpy as np
from abc import abstractmethod
from typing import Union

from pyRadPlan.plan import Plan
from pyRadPlan.result import PlanResult, ApertureInfo
from pyRadPlan.stf import SteeringInformation, Beam
from pyRadPlan.machines import MLC
from ._base import Sequencer

logger = logging.getLogger(__name__)


class PhotonSequencer(Sequencer):
    def __init__(self, pln: Union[Plan, dict] = None):
        self.num_of_mlc_leaf_pairs = 80
        self.num_of_levels = 5
        super().__init__(pln)

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

    def _inititialize_beam_fluence_matrix(
        self, beam: Beam, w_of_curr_beams: np.ndarray = None
    ) -> dict:
        """
        Initialize the fluence matrix for a given beam.

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
        # Get ray positions
        x_ray_pos = np.array([ray.ray_pos_bev[0] for ray in beam.rays])
        z_ray_pos = np.array([ray.ray_pos_bev[2] for ray in beam.rays])

        min_x, max_x = x_ray_pos.min(), x_ray_pos.max()
        min_z, max_z = z_ray_pos.min(), z_ray_pos.max()

        bixel_width = beam.bixel_width
        dim_of_fluence_mx_x = int((max_x - min_x) / bixel_width + 1)
        dim_of_fluence_mx_z = int((max_z - min_z) / bixel_width + 1)

        # Create fluence matrix
        fluence_mx = np.zeros((dim_of_fluence_mx_z, dim_of_fluence_mx_x))
        x_ind_pos = ((x_ray_pos - min_x) / bixel_width).astype(int)
        z_ind_pos = ((z_ray_pos - min_z) / bixel_width).astype(int)

        # Compute indices:
        indices = z_ind_pos * dim_of_fluence_mx_x + x_ind_pos
        # Create fluence_mx if w is given and called in sequener
        if w_of_curr_beams is not None:
            for xp, zp, weight in zip(x_ind_pos, z_ind_pos, w_of_curr_beams):
                fluence_mx[zp, xp] = weight
            cal_fac = np.max(fluence_mx)
            d_0 = np.round(fluence_mx / cal_fac * self.num_of_levels).astype(int)
            d_k = d_0.copy()

            init_param = {
                "fluence_mx": fluence_mx,
                "cal_fac": cal_fac,
                "d_0": d_0,
                "d_k": d_k,
            }
        # else return ray_map
        else:
            # Build ray map on the fluence grid
            ray_map = np.zeros((dim_of_fluence_mx_z, dim_of_fluence_mx_x))
            # fill
            ray_map.flat[indices] = 1
            init_param = {
                "ray_map": ray_map,
                "min_x": min_x,
                "max_x": max_x,
                "min_z": min_z,
                "max_z": max_z,
                "bixel_width": bixel_width,
                "x_ray_pos": x_ray_pos,
                "z_ray_pos": z_ray_pos,
            }
        init_param.update(
            {
                "indices": indices,
                "dim_x": dim_of_fluence_mx_x,
                "dim_z": dim_of_fluence_mx_z,
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
        dim_z = init_param["dim_z"]
        min_x = init_param["min_x"]
        bw = init_param["bixel_width"]

        lim_l = np.full(dim_z, np.nan)
        lim_r = np.full(dim_z, np.nan)
        for n in range(dim_z):
            cols = np.where(ray_map[n, :] != 0)[0]
            if cols.size > 0:
                lim_l[n] = (cols[0]) * bw + min_x - 0.5 * bw
                lim_r[n] = (cols[-1]) * bw + min_x + 0.5 * bw
            else:
                lim_l[n] = min_x
                lim_r[n] = min_x

        return lim_l, lim_r

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
            # Get x and z positions from each ray of current beam.
            init = self._inititialize_beam_fluence_matrix(beam)

            # Initialize limits and dimensions
            lim_l, lim_r = self._compute_leaf_limits(init)

            # Create bixelIndMap
            bixel_ind_map = np.full((init["dim_z"], init["dim_x"]), np.nan)
            num_rays = len(beam.rays)
            bixel_ind_map.flat[init["indices"]] = np.arange(0, num_rays) + bixel_ind_offset
            bixel_ind_offset += num_rays

            beam_aperture = {}
            # TODO: This should in general be correct. If not, change .T
            beam_aperture["bixel_ind_map"] = bixel_ind_map.T
            beam_aperture["pos_of_corner_bixel"] = [init["min_x"], 0, init["min_z"]]

            # Process shapes for the current beam
            # initialize per‐shape lists in beam_aperture
            shapes_arr = sequencing["beam"][i]["shapes"]
            weights_arr = sequencing["beam"][i]["shapes_weight"]
            num_shapes = sequencing["beam"][i]["num_of_shapes"]
            beam_aperture["weight"] = list(weights_arr)
            beam_aperture["shape_map"] = [shapes_arr[:, :, m] for m in range(num_shapes)]
            beam_aperture["vector_offset"] = [
                vector_offset + m * init["dim_z"] for m in range(num_shapes)
            ]

            # advance the global counter once
            vector_offset += num_shapes * init["dim_z"]

            # compute all mlc in one go
            mlc_sequence = []
            for m in range(num_shapes):
                shape_map = shapes_arr[:, :, m]
                # TODO: differentiate between MLCX & MLCY
                mlc = MLC(
                    device_type="MLCX",
                    number_of_elements=self.num_of_mlc_leaf_pairs,
                )
                mlc.calculate_from_shape(
                    shape_map,
                    init["min_x"],
                    init["z_ray_pos"],
                    bixel_width,
                    lim_l,
                    lim_r,
                )
                mlc_sequence.append(mlc)
            beam_aperture["mlc_sequence"] = mlc_sequence

            # Derive flattened per-beam fields from the MLC objects (single source of
            # truth = MLC methods). These stay on the beam sequence for matRad interop.
            beam_aperture["left_leaf_pos"] = [mlc.leaf_positions[:, 0] for mlc in mlc_sequence]
            beam_aperture["right_leaf_pos"] = [mlc.leaf_positions[:, 1] for mlc in mlc_sequence]
            beam_aperture["leaf_pair_pos"] = mlc_sequence[0].leaf_position_boundaries
            beam_aperture["is_active_leaf_pair"] = mlc_sequence[0].is_active_leaf_pair(
                central_leaf_pair, bixel_width
            )
            beam_aperture["num_of_active_leaf_pairs"] = mlc_sequence[0].num_of_active_leaf_pairs
            beam_aperture["central_leaf_pair"] = central_leaf_pair
            beam_aperture["lim_l"] = lim_l
            beam_aperture["lim_r"] = lim_r

            # Define MLC window for plotting using physical coordinates
            beam_aperture["MLCWindow"] = [
                init["min_x"] - 0.5 * init["bixel_width"],
                init["max_x"] + 0.5 * init["bixel_width"],
                init["min_z"] - 0.5 * init["bixel_width"],
                init["max_z"] + 0.5 * init["bixel_width"],
            ]

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
        shapes_arr = np.stack(shapes_list, axis=2)  # (dim_z, dim_x, num_shapes)
        shapes_w_arr = (
            np.array(shapes_weight_list) / self.num_of_levels * tmp_fluence_dict["cal_fac"]
        )
        num_shapes = shapes_arr.shape[2]

        # Beam sequencing structure
        beam_sequencing = {
            "num_of_shapes": num_shapes,
            "shapes": shapes_arr,
            "shapes_weight": shapes_w_arr,
            "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
            "fluence": tmp_fluence_dict["d_0"],
        }

        # Compute the summed weighted shape for the beam
        beam_sum = np.zeros((tmp_fluence_dict["dim_z"], tmp_fluence_dict["dim_x"]))
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
        central_leaf_pair = int(math.ceil(self.num_of_mlc_leaf_pairs / 2.0))

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

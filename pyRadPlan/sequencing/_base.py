"""Base class for sequencing."""

import logging

from abc import ABC, abstractmethod
from typing import ClassVar, Union
import warnings
import numpy as np
import math

from pyRadPlan.result import PlanResult, ApertureInfo
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.stf import Beam

logger = logging.getLogger(__name__)


class Sequencer(ABC):
    name: ClassVar[str]
    short_name: ClassVar[str]

    def __init__(self, pln: Union[Plan, dict] = None):
        self.resultGUI: PlanResult
        self.aperture_info = None
        self.num_of_levels = 5
        if pln is not None:
            self.assign_properties_from_pln(pln, True)

    @abstractmethod
    def sequence(self, result: PlanResult, stf: SteeringInformation, dij) -> PlanResult:
        """
        Execute the sequencing algorithm.

        Parameters
        ----------
        resultGUI : object
            The result object or dictionary to store sequencing results.
        stf : object
            The steering information structure.
        dij (Optional): object
            The dij matrix or related input.

        Returns
        -------
        object
            The updated resultGUI after sequencing.
        """
        pass

    def _inititialize_beam_fluence_matrix(self, beam: Beam, w_of_curr_beams):
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
        numpy.ndarray
            The initialized fluence matrix for the beam.
        """
        # Get ray positions
        x_positions = np.array([ray.ray_pos_bev[0] for ray in beam.rays])
        z_positions = np.array([ray.ray_pos_bev[2] for ray in beam.rays])

        min_x, max_x = x_positions.min(), x_positions.max()
        min_z, max_z = z_positions.min(), z_positions.max()

        bixel_width = beam.bixel_width
        dim_of_fluence_mx_x = int((max_x - min_x) / bixel_width + 1)
        dim_of_fluence_mx_z = int((max_z - min_z) / bixel_width + 1)

        # Create fluence matrix
        fluence_mx = np.zeros((dim_of_fluence_mx_z, dim_of_fluence_mx_x))
        x_pos = ((x_positions - min_x) / bixel_width).astype(int)  # adjust for 0-indexing
        z_pos = ((z_positions - min_z) / bixel_width).astype(int)

        for xp, zp, weight in zip(x_pos, z_pos, w_of_curr_beams):
            fluence_mx[zp, xp] = weight

        cal_fac = np.max(fluence_mx)
        d_0 = np.round(fluence_mx / cal_fac * self.num_of_levels).astype(int)
        d_k = d_0.copy()

        init_param = {
            "fluence_mx": fluence_mx,
            "cal_fac": cal_fac,
            "d_0": d_0,
            "d_k": d_k,
            "dim_of_fluence_mx_x": dim_of_fluence_mx_x,
            "dim_of_fluence_mx_z": dim_of_fluence_mx_z,
            "x_pos": x_pos,
            "z_pos": z_pos,
        }
        return init_param

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
        num_of_MLC_leaf_pairs = 80
        central_leaf_pair = int(math.ceil(num_of_MLC_leaf_pairs / 2.0))

        bixel_ind_offset = 0
        total_num_of_bixels = sum(beam.total_number_of_bixels for beam in stf.beams)
        total_num_of_shapes = sum(beam_seq["num_of_shapes"] for beam_seq in sequencing["beam"])
        vector_offset = total_num_of_shapes + 1

        aperture_info["beam"] = []

        for i, beam in enumerate(stf.beams):
            # Get x and z positions from each ray of current beam.
            rays = beam.rays  # each ray is expected to have attribute ray_pos_bev: [x, y, z]
            x_positions = np.array([ray.ray_pos_bev[0] for ray in rays])
            z_positions = np.array([ray.ray_pos_bev[2] for ray in rays])
            min_x, max_x = x_positions.min(), x_positions.max()
            min_z, max_z = z_positions.min(), z_positions.max()

            # Compute grid dimensions
            dim_x = int((max_x - min_x) / bixel_width + 1)
            dim_z = int((max_z - min_z) / bixel_width + 1)

            # Build ray map on the fluence grid
            ray_map = np.zeros((dim_z, dim_x))
            # Adjust positions to grid indices (0-indexed)
            ix = ((x_positions - min_x) / bixel_width).astype(int)
            iz = ((z_positions - min_z) / bixel_width).astype(int)
            indices = iz + ix * dim_z
            # Fill ray map
            ray_map.flat[indices] = 1

            # Create bixelIndMap
            bixel_ind_map = np.full((dim_z, dim_x), np.nan)
            num_rays = len(rays)
            bixel_ind_map.flat[indices] = np.arange(1, num_rays + 1) + bixel_ind_offset
            bixel_ind_offset += num_rays

            pos_of_corner_bixel = [min_x, 0, min_z]

            # Compute leaf limits
            lim_l = np.full(dim_z, np.nan)
            lim_r = np.full(dim_z, np.nan)
            for l in range(dim_z):
                cols = np.where(ray_map[l, :] != 0)[0]
                if cols.size > 0:
                    lim_l[l] = (cols[0]) * bixel_width + min_x - 0.5 * bixel_width
                    lim_r[l] = (cols[-1]) * bixel_width + min_x + 0.5 * bixel_width
                else:
                    lim_l[l] = min_x
                    lim_r[l] = min_x

            beam_aperture = {}
            beam_aperture["num_of_shapes"] = sequencing["beam"][i]["num_of_shapes"]
            beam_aperture["bixel_ind_map"] = bixel_ind_map
            beam_aperture["pos_of_corner_bixel"] = pos_of_corner_bixel

            # Process shapes for the current beam
            shapes_list = []
            for m in range(sequencing["beam"][i]["num_of_shapes"]):
                # Each shape is a 2D numpy array (dim_z x dim_x)
                shape_map = sequencing["beam"][i]["shapes"][:, :, m]
                left_leaf_pos = np.empty(dim_z)
                right_leaf_pos = np.empty(dim_z)
                for l in range(dim_z):
                    open_cols = np.where(shape_map[l, :] > 0)[0]
                    if open_cols.size == 0:
                        # If no bixel is open, use the midpoint of limits
                        left_leaf_pos[l] = (lim_l[l] + lim_r[l]) / 2
                        right_leaf_pos[l] = (lim_l[l] + lim_r[l]) / 2
                    else:
                        left_leaf_pos[l] = (open_cols[0]) * bixel_width + min_x - 0.5 * bixel_width
                        right_leaf_pos[l] = (
                            (open_cols[-1]) * bixel_width + min_x + 0.5 * bixel_width
                        )
                shape_info = {
                    "left_leaf_pos": left_leaf_pos,
                    "right_leaf_pos": right_leaf_pos,
                    "weight": sequencing["beam"][i]["shapes_weight"][m],
                    "shape_map": shape_map,
                    "vector_offset": vector_offset,
                }
                vector_offset += dim_z
                shapes_list.append(shape_info)
            beam_aperture["shapes"] = shapes_list

            # Leaf pair positions and active leaf pairs
            leaf_pair_pos = np.unique(z_positions)
            top_leaf_pair_pos = max_z
            bottom_leaf_pair_pos = min_z
            top_leaf_pair = central_leaf_pair - top_leaf_pair_pos / bixel_width
            bottom_leaf_pair = central_leaf_pair - bottom_leaf_pair_pos / bixel_width
            is_active_leaf_pair = np.zeros(num_of_MLC_leaf_pairs)
            # Activate indices between top and bottom leaf pairs (casting to int)
            start_ix = int(math.floor(top_leaf_pair))
            end_ix = int(math.ceil(bottom_leaf_pair))
            is_active_leaf_pair[start_ix : end_ix + 1] = 1

            beam_aperture["num_of_active_leaf_pairs"] = dim_z
            beam_aperture["leaf_pair_pos"] = leaf_pair_pos
            beam_aperture["is_active_leaf_pair"] = is_active_leaf_pair
            beam_aperture["central_leaf_pair"] = central_leaf_pair
            beam_aperture["lim_l"] = lim_l
            beam_aperture["lim_r"] = lim_r

            # Define MLC window for plotting using physical coordinates
            beam_aperture["MLCWindow"] = [
                min_x - 0.5 * bixel_width,
                max_x + 0.5 * bixel_width,
                min_z - 0.5 * bixel_width,
                max_z + 0.5 * bixel_width,
            ]

            aperture_info["beam"].append(beam_aperture)

        # Global aperture information
        aperture_info["bixel_width"] = bixel_width
        aperture_info["num_of_MLC_leaf_pairs"] = num_of_MLC_leaf_pairs
        aperture_info["total_num_of_bixels"] = total_num_of_bixels
        aperture_info["total_num_of_shapes"] = total_num_of_shapes
        total_leaf_pairs = sum(
            b["num_of_active_leaf_pairs"] * b["num_of_shapes"] for b in aperture_info["beam"]
        )
        aperture_info["total_num_of_leaf_pairs"] = total_leaf_pairs

        # Optimization vectors (placeholders – implement as needed)
        aperture_info["aperture_vector"] = None
        aperture_info["mapping_mx"] = None
        aperture_info["lim_mx"] = None

        return aperture_info

    def assign_properties_from_pln(self, pln: Plan, warn_when_property_changed: bool = False):
        """
        Assign properties from a Plan object to the Sequencer.

        This includes the Scenario Model and the Biological Model, and any
        other properties that
        can be stored in the prop_seq dictionary within the Plan object.
        This function will check if a property exists for the dose engine and,
        if yes, set it.

        Parameters
        ----------
        pln : Plan
            The Plan object to assign properties from.
        warn_when_property_changed : bool
            Whether to warn when properties are changed.
        """

        pln = validate_pln(pln)

        # Set Scenario Model
        if hasattr(pln, "mult_scen"):
            self.mult_scen = pln.mult_scen

        # Assign Biologival Model
        if hasattr(pln, "bio_param"):
            self.bio_param = pln.bio_param  # TODO: No bio_param yet

        if not isinstance(warn_when_property_changed, bool):
            warn_when_property_changed = False

        # Overwrite default properties within the engine
        # with the ones given in the prop_seq dict
        if pln.prop_seq is not None:
            prop_dict = pln.prop_seq
            if (
                "sequencer" in prop_dict
                and prop_dict["sequencer"]
                and prop_dict["sequencer"] != self.short_name
            ):
                raise ValueError(
                    f"Inconsistent sequencer given! pln asks for '{prop_dict['sequencer']}', "
                    f"but you are using '{self.short_name}'!"
                )
            prop_dict.pop("sequencer", None)
        else:
            prop_dict = {}

        fields = prop_dict.keys()

        # Set up warning message
        if warn_when_property_changed:
            warning_msg = "Property in Dose Engine overwritten from pln.prop_seq"
        else:
            warning_msg = None

        for field in fields:
            if not hasattr(self, field):
                warnings.warn('Property "{}" not found in Sequencer!'.format(field))
            elif warn_when_property_changed and warning_msg:
                logger.warning(warning_msg + f": {field}")

            setattr(self, field, prop_dict[field])

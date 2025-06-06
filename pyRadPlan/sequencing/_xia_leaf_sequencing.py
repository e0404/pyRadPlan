"""Xia Leaf Sequencer class."""

import logging

from typing import Union
import numpy as np
import math

from pyRadPlan.result import PlanResult, validate_result
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from ._base_photon import PhotonSequencer

logger = logging.getLogger(__name__)


class XiaLeafSequencer(PhotonSequencer):
    name = "xia_leaf_sequencing"
    short_name = "xia"

    def __init__(self, pln: Union[Plan, dict] = None):
        """Initialize the Sequencer class."""

        self.mode = "reducing_levels"  # 'sliding_window' or 'reducing_levels'
        super().__init__(pln)

    def sequence(self, result: PlanResult, stf: SteeringInformation, dij: Dij = None):
        """
        Execute the sequencing algorithm.

        Parameters
        ----------
        result : object
            The result object or dictionary to store sequencing results.
        stf : object
            The steering information structure.
        dij : object
            The dij matrix or related input.

        Returns
        -------
        object
            The updated result after sequencing.
        """

        num_of_beams = len(stf.beams)
        offset = 0
        sequencing = {}
        sequencing["fluence"] = np.zeros_like(result.fluence)
        sequencing["beam"] = []

        for i in range(num_of_beams):
            beam = stf.beams[i]
            num_of_rays_per_beam = len(beam.rays)
            w_of_curr_beams = result.fluence[offset : offset + num_of_rays_per_beam]

            tmp_fluence_dict = self._inititialize_beam_fluence_matrix(
                stf.beams[i], w_of_curr_beams
            )

            d_k = tmp_fluence_dict["d_k"]
            d_0 = tmp_fluence_dict["d_0"]
            l_k = int(np.max(d_k))
            l_0 = l_k

            shapes_list = []
            shapes_weight_list = []

            k = 0

            while l_k > 0:
                k += 1
                m = max(math.floor(math.log2(l_k)), 1)
                d_k_value = math.floor(2 ** (m - 1))

                # Calculate opening matrix
                opening_mx = d_k >= d_k_value

                # Apply reducing levels technique
                if self.mode == "reducing_levels":
                    for row in range(tmp_fluence_dict["dim_of_fluence_mx_z"]):
                        # mimic [maxVal, maxIx] = max(openingMx(j,:) .* D_k(j,:)) - matrad code
                        row_vals = opening_mx[row, :] * d_k[row, :]
                        if row_vals.max() > 0:
                            max_ix = int(np.argmax(row_vals))

                            # find first zero *after* max_ix
                            zeros_after = np.where(~opening_mx[row, max_ix + 1 :])[0]
                            if zeros_after.size > 0:
                                # must add +1 because we sliced from max_ix+1
                                close_ix = max_ix + 1 + int(zeros_after[0])
                                opening_mx[row, close_ix:] = False

                            # find last zero *before* max_ix
                            zeros_before = np.where(~opening_mx[row, :max_ix])[0]
                            if zeros_before.size > 0:
                                open_ix = int(zeros_before[-1])
                                opening_mx[row, : open_ix + 1] = False
                elif self.mode == "sliding_window":
                    for row in range(tmp_fluence_dict["dim_of_fluence_mx_z"]):
                        row_open = np.where(opening_mx[row, :])[0]
                        if row_open.size > 0:
                            open_ix = row_open[0]
                            zeros_after = np.where(~opening_mx[row, open_ix + 1 :])[0]
                            if zeros_after.size > 0:
                                close_ix = open_ix + 1 + zeros_after[0]
                                opening_mx[row, close_ix:] = False

                shape_k = opening_mx.astype(int) * d_k_value

                shapes_list.append(shape_k)
                shapes_weight_list.append(d_k_value)

                d_k = d_k - shape_k
                l_k = int(np.max(d_k))

            shapes_arr = np.stack(shapes_list, axis=2)  # (Z, X, num_shapes)
            shapes_w_arr = (
                np.array(shapes_weight_list) / self.num_of_levels * tmp_fluence_dict["cal_fac"]
            )
            num_shapes = shapes_arr.shape[2]

            beam_sequencing = {
                "num_of_shapes": num_shapes,
                "shapes": shapes_arr,
                "shapes_weight": shapes_w_arr,
                "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
                "fluence": d_0,
            }
            sequencing["beam"].append(beam_sequencing)

            # Map fluence back to weights (using linear indices)
            linear_indices = (
                tmp_fluence_dict["z_pos"]
                + tmp_fluence_dict["x_pos"] * tmp_fluence_dict["dim_of_fluence_mx_z"]
            )
            flat_d0 = d_0.flatten(order="F")
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = (
                flat_d0[linear_indices] / self.num_of_levels * tmp_fluence_dict["cal_fac"]
            )

            offset += num_of_rays_per_beam

        # finalize sequencing result
        result_tmp = {}
        result_tmp["fluence"] = result.fluence
        result_tmp["sequence"] = sequencing
        result_tmp["sequence"]["aperture_info"] = self.sequencing_to_aperture_info(sequencing, stf)
        result_tmp["sequence"]["fluence"] = sequencing["fluence"]

        # Interpolate dose (placeholder implementation)
        cubes = dij.compute_result_ct_grid(sequencing["fluence"])

        # if hasattr(result, 'w_dao'):
        #     pass

        return validate_result({**cubes, **result_tmp})

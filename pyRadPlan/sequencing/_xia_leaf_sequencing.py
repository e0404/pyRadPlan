"""Xia Leaf Sequencer class."""

import logging

from typing import Union
import numpy as np
import math

from pyRadPlan.result import PlanResult, validate_result
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from ._base import Sequencer

logger = logging.getLogger(__name__)


class XiaLeafSequencer(Sequencer):
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

            shapes = np.full(
                (
                    tmp_fluence_dict["dim_of_fluence_mx_z"],
                    tmp_fluence_dict["dim_of_fluence_mx_x"],
                    10000,
                ),
                np.nan,
            )
            shapes_weight = np.zeros(10000)
            k = 0

            while l_k > 0:
                k += 1
                m = math.floor(math.log2(l_k)) if l_k > 0 else 0
                m = max(m, 1)
                d_k_value = math.floor(2 ** (m - 1))

                # Calculate opening matrix
                opening_mx = d_k >= d_k_value

                # Apply reducing levels technique
                if self.mode == "recucing_levels":
                    for row in range(tmp_fluence_dict["dim_of_fluence_mx_z"]):
                        row_vals = opening_mx[row, :] * d_k[row, :]
                        if np.any(row_vals):
                            max_ix = np.argmax(row_vals)
                            false_after = np.where(~opening_mx[row, max_ix + 1 :])[0]
                            if false_after.size > 0:
                                close_ix = max_ix + false_after[0]
                                opening_mx[row, close_ix:] = False
                                false_before = np.where(~opening_mx[row, :max_ix])[0]
                                if false_before.size > 0:
                                    open_ix = false_before[-1]
                                    opening_mx[row, : open_ix + 1] = False

                elif self.mode == "sliding_window":  # sliding window technique (if needed)
                    for row in range(tmp_fluence_dict["dim_of_fluence_mx_z"]):
                        row_open = np.where(opening_mx[row, :])[0]
                        if row_open.size > 0:
                            open_ix = row_open[0]
                            zeros_after = np.where(~opening_mx[row, open_ix + 1 :])[0]
                            if zeros_after.size > 0:
                                close_ix = open_ix + zeros_after[0]
                                opening_mx[row, close_ix:] = False

                shape_k = opening_mx.astype(int) * d_k_value

                shapes[:, :, k - 1] = shape_k
                shapes_weight[k - 1] = d_k_value
                d_k = d_k - shape_k
                l_k = int(np.max(d_k))

            beam_sequencing = {
                "num_of_shapes": k,
                "shapes": shapes[:, :, :k],
                "shapes_weight": shapes_weight[:k]
                / self.num_of_levels
                * tmp_fluence_dict["cal_fac"],
                "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
                "fluence": d_0,
            }
            sequencing["beam"].append(beam_sequencing)

            # Map fluence back to weights (using linear indices)
            linear_indices = (
                tmp_fluence_dict["z_pos"]
                + tmp_fluence_dict["x_pos"] * tmp_fluence_dict["dim_of_fluence_mx_z"]
            )
            flat_d0 = d_0.flatten()
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = (
                flat_d0[linear_indices] / self.num_of_levels * tmp_fluence_dict["cal_fac"]
            )

            offset += num_of_rays_per_beam

        # finalize sequencing result
        result_tmp = {}
        result_tmp["fluence"] = result.fluence
        result_tmp["fluence_sequenced"] = sequencing["fluence"]
        result_tmp["sequencing"] = sequencing
        result_tmp["aperture_info"] = self.sequencing_to_aperture_info(sequencing, stf)

        # Interpolate dose (placeholder implementation)
        cubes = dij.compute_result_ct_grid(sequencing["fluence"])

        # if hasattr(result, 'w_dao'):
        #     pass

        return validate_result({**cubes, **result_tmp})

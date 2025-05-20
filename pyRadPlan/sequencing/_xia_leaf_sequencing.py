"""Base class for sequencing."""

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
        resultGUI : object
            The result object or dictionary to store sequencing results.
        stf : object
            The steering information structure.
        dij : object
            The dij matrix or related input.
        numOfLevels : int
            The number of intensity levels used for stratification.

        Returns
        -------
        object
            The updated resultGUI after sequencing.
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
            l_k = int(np.max(d_k))
            l_0 = l_k

            shapes = np.full((dim_of_fluence_mx_z, dim_of_fluence_mx_x, 10000), np.nan)
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
                    for row in range(dim_of_fluence_mx_z):
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
                    for row in range(dim_of_fluence_mx_z):
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
                    "shapes_weight": shapes_weight[:k] / self.num_of_levels * cal_fac,
                    "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
                    "fluence": d_0,
                }
                sequencing["beam"].append(beam_sequencing)

                # Map fluence back to weights (using linear indices)
                linear_indices = z_pos + x_pos * dim_of_fluence_mx_z
                flat_d0 = d_0.flatten()
                sequencing["fluence"][offset : offset + num_of_rays_per_beam] = (
                    flat_d0[linear_indices] / self.num_of_levels * cal_fac
                )

                offset += num_of_rays_per_beam

        # Finalize result_gui
        result.fluence = sequencing["w"]
        result.fluence_sequenced = sequencing["w"]
        result.sequencing = sequencing
        result.aperture_info = self.sequencing_to_aperture_info(sequencing, stf)

        # Interpolate dose (placeholder implementation)
        dose_sequenced_dose_grid = (dij["physicalDose"][0] * sequencing["w"]).reshape(
            dij["doseGrid"]["dimensions"]
        )
        # result_gui['physicalDose'] = self.interp3(dij['doseGrid']['x'],
        #                       np.array(dij['doseGrid']['y']).T,
        #                       dij['doseGrid']['z'],
        #                       dose_sequenced_dose_grid,
        #                       dij['ctGrid']['x'],
        #                       np.array(dij['ctGrid']['y']).T,
        #                       dij['ctGrid']['z'])
        # if 'w_dao' in result_gui:
        #     del result_gui['w_dao']

        return validate_result(result)

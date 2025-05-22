"""Siochi Leaf Sequencer class."""

import logging

from typing import Union
import numpy as np

from pyRadPlan.result import PlanResult, validate_result
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from ._base import Sequencer

logger = logging.getLogger(__name__)


class SiochiLeafSequencer(Sequencer):
    name = "siochi_leaf_sequencing"
    short_name = "siochi"

    def __init__(self, pln: Union[Plan, dict] = None):
        """Initialize the Sequencer class."""

        # self.mode = "reducing_levels"  # 'sliding_window' or 'reducing_levels'
        super().__init__(pln)

    def sequence(self, result: PlanResult, stf: SteeringInformation, dij: Dij = None):
        """
        Execute the Siochi leaf sequencing algorithm (without visualization).

        Parameters
        ----------
        result : PlanResult
            The result object that contains fluence.
        stf : SteeringInformation
            The steering information containing the beams.
        dij : Dij, optional
            The dij structure with dose grid information.

        Returns
        -------
        PlanResult
            The updated result after sequencing.
        """
        num_of_beams = len(stf.beams)
        offset = 0
        sequencing = {"fluence": np.zeros_like(result.fluence), "beam": []}

        # We'll also store the combined weight vector in sequencing["w"]
        sequencing["fluence"] = np.zeros_like(result.fluence)

        # --- Helper functions defined locally ---

        def _decompose_port(dk, dim_z, dim_x):
            """
            Decompose the port from the stratified intensity matrix.

            Parameters
            ----------
                dk : numpy.ndarray
                    The stratified intensity matrix.
                dim_z : int
                    The number of rows in the matrix.
                dim_x : int
                    The number of columns in the matrix.

            Returns
            -------
                tops : numpy.ndarray
                    The top boundaries of the slabs.
                bases : numpy.ndarray
                    The bottom boundaries of the slabs.
            """
            tops = np.zeros((dim_z, dim_x))
            bases = np.zeros((dim_z, dim_x))
            # Loop over columns (x-direction)
            for i in range(dim_x):
                # maxTop = -1
                for j in range(dim_z):
                    if i == 0:
                        bases[j, i] = 1
                        tops[j, i] = bases[j, i] + dk[j, i] - 1
                    elif dk[j, i] >= dk[j, i - 1]:
                        bases[j, i] = bases[j, i - 1]
                        tops[j, i] = bases[j, i] + dk[j, i] - 1
                    elif dk[j, i] == 0:
                        bases[j, i] = tops[j, i - 1] + 1
                        tops[j, i] = bases[j, i] - 1
                    else:
                        tops[j, i] = tops[j, i - 1]
                        bases[j, i] = tops[j, i] - dk[j, i] + 1
                # Note: We ignore corrections for collision / tongue-and-groove errors
            return tops, bases

        def _different_slap(tops, bases, level):
            """Return True if the slab at the current level is different from the one below."""
            if level == 1:
                return True
            shape_lvl = ((bases <= level) & (level <= tops)).astype(int)
            shape_lvl_prev = ((bases <= (level - 1)) & ((level - 1) <= tops)).astype(int)
            return not np.array_equal(shape_lvl, shape_lvl_prev)

        def _convert_to_segments(shapes, shapes_w, k, tops, bases):
            """Convert tops and bases to shape matrices."""
            levels = int(np.max(tops))
            for level in range(1, levels + 1):
                if level == 1 or _different_slap(tops, bases, level):
                    k += 1
                    shape_k = ((bases <= level) & (level <= tops)).astype(int)
                    shapes[:, :, k - 1] = shape_k
                shapes_w[k - 1] += 1
            return shapes, shapes_w, k

        # --- End of helper functions ---

        for i in range(num_of_beams):
            beam = stf.beams[i]
            num_of_rays_per_beam = len(beam.rays)
            w_of_curr_beam = result.fluence[offset : offset + num_of_rays_per_beam]

            tmp_fluence_dict = self._inititialize_beam_fluence_matrix(beam, w_of_curr_beam)
            dk = tmp_fluence_dict["d_k"]
            d0 = tmp_fluence_dict["d_0"]
            dim_z = tmp_fluence_dict["dim_of_fluence_mx_z"]
            dim_x = tmp_fluence_dict["dim_of_fluence_mx_x"]

            # Compute index mapping for fluence matrix
            # MATLAB: ind_in_fluence_mx = zPos + (xPos-1)*dim_z
            # Here, xPos and zPos are 0-indexed integers stored in tmp_fluence_dict.
            ind_in_fluence_mx = tmp_fluence_dict["z_pos"] + tmp_fluence_dict["x_pos"] * dim_z

            # Decomposition: find non_zero indices to compute boundaries
            # non_zero = np.where(dk != 0)
            # if non_zero[0].size > 0:
            #     dkMinZ = int(non_zero[0].min())
            #     dkMaxZ = int(non_zero[0].max())
            #     dkMinX = int(non_zero[1].min())
            #     dkMaxX = int(non_zero[1].max())
            # else:
            #     dkMinZ = dkMaxZ = dkMinX = dkMaxX = 0

            # Decompose port (rod-pushing)
            tops, bases = _decompose_port(dk, dim_z, dim_x)

            # Set up containers for segments
            shapes = np.full((dim_z, dim_x, 10000), np.nan)
            shapes_w = np.zeros(10000)
            k = 0

            # Convert ports to shape segments
            shapes, shapes_w, k = _convert_to_segments(shapes, shapes_w, k, tops, bases)

            # Build beam sequencing structure
            beam_sequencing = {
                "num_of_shapes": k,
                "shapes": shapes[:, :, :k],
                "shapes_weight": shapes_w[:k] / self.num_of_levels * tmp_fluence_dict["cal_fac"],
                "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
                "fluence": d0,
            }

            # Compute the summed weighted shape for the beam.
            beam_sum = np.zeros((dim_z, dim_x))
            for j in range(k):
                beam_sum += (
                    beam_sequencing["shapes"][:, :, j] * beam_sequencing["shapes_weight"][j]
                )
            # Map summed shapes back into a fluence vector using the computed indices.
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = beam_sum.flat[
                ind_in_fluence_mx
            ]

            sequencing["beam"].append(beam_sequencing)
            offset += num_of_rays_per_beam

        # Finalize sequencing result
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

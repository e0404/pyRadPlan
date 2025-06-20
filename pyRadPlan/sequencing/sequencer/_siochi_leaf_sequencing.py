"""Class for Siochi-Leaf-Sequencing."""

import logging

from typing import Union
import numpy as np

from pyRadPlan.result import PlanResult, validate_result
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan
from ._base_photon import PhotonSequencer

logger = logging.getLogger(__name__)


class SiochiLeafSequencer(PhotonSequencer):
    name = "siochi_leaf_sequencing"
    short_name = "siochi"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Union[Plan, dict] = None):
        """Initialize the Sequencer class."""

        super().__init__(pln)

    def _tongue_and_groove_correction(
        self, tops: np.ndarray, bases: np.ndarray, dk: np.ndarray, indexing: tuple
    ):
        """Apply the downward + upward TnG correction loop for column i."""
        # reducing arguments in function:
        i, max_row, min_z, max_z = indexing

        while True:
            changed = False
            # TODO: This implementation reduces branching but is less readable. Might change it in future again
            # downward pass (j = max_row-1 … min_z)
            for j in range(max_row - 1, min_z - 1, -1):
                # shorter rod above longer rod
                check = dk[j, i] < dk[j + 1, i]
                if check and tops[j, i] > tops[j + 1, i]:
                    tops[j + 1, i] = tops[j, i]
                    bases[j + 1, i] = tops[j + 1, i] - dk[j + 1, i] + 1
                    changed = True
                elif check and bases[j, i] < bases[j + 1, i]:
                    bases[j, i] = bases[j + 1, i]
                    tops[j, i] = bases[j, i] + dk[j, i] - 1
                    changed = True
                # current >= next, fix opposite peeking
                elif not check and tops[j, i] < tops[j + 1, i]:
                    tops[j, i] = tops[j + 1, i]
                    bases[j, i] = tops[j, i] - dk[j, i] + 1
                    changed = True
                elif not check and bases[j, i] > bases[j + 1, i]:
                    bases[j + 1, i] = bases[j, i]
                    tops[j + 1, i] = bases[j + 1, i] + dk[j + 1, i] - 1
                    changed = True

            # upward pass (j = max_row+1 … max_z)
            for j in range(max_row + 1, max_z + 1):
                # shorter rod below longer rod
                check = dk[j, i] < dk[j - 1, i]
                if check and tops[j, i] > tops[j - 1, i]:
                    tops[j - 1, i] = tops[j, i]
                    bases[j - 1, i] = tops[j - 1, i] - dk[j - 1, i] + 1
                    changed = True
                elif check and bases[j, i] < bases[j - 1, i]:
                    bases[j, i] = bases[j - 1, i]
                    tops[j, i] = bases[j, i] + dk[j, i] - 1
                    changed = True
                # current >= above, fix opposite peeking
                elif not check and tops[j, i] < tops[j - 1, i]:
                    tops[j, i] = tops[j - 1, i]
                    bases[j, i] = tops[j, i] - dk[j, i] + 1
                    changed = True
                elif not check and bases[j, i] > bases[j - 1, i]:
                    bases[j - 1, i] = bases[j, i]
                    tops[j - 1, i] = bases[j - 1, i] + dk[j - 1, i] - 1
                    changed = True

            if not changed:
                break

        return tops, bases

    def _decompose_port(self, dk, dim_z, dim_x):
        """Decompose the port from the stratified intensity matrix, including collision & tongue-and-groove fixes."""
        tops = np.zeros((dim_z, dim_x), dtype=int)
        bases = np.zeros((dim_z, dim_x), dtype=int)

        # replicate MATLAB's minZ/maxZ/minX/maxX over full range
        min_z, max_z = 0, dim_z - 1
        min_x, max_x = 0, dim_x - 1

        for i in range(min_x, max_x + 1):
            # first pass: set bases/tops and track the highest top
            max_top = -1
            max_row = min_z
            for j in range(min_z, max_z + 1):
                if i == min_x:
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

                # update maxTop, maxRow
                if tops[j, i] > max_top:
                    max_top = tops[j, i]
                    max_row = j

            # tongue-and-groove correction loop
            tops, bases = self._tongue_and_groove_correction(
                tops, bases, dk, [i, max_row, min_z, max_z]
            )

        return tops, bases

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

        sequencing["fluence"] = np.zeros_like(result.fluence)

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
            ind_in_fluence_mx = tmp_fluence_dict["z_pos"] + tmp_fluence_dict["x_pos"] * dim_z

            # Decompose port (rod-pushing)
            tops, bases = self._decompose_port(dk, dim_z, dim_x)

            # Build segments dynamically (no preallocation)
            shapes_list = []
            shapes_weight_list = []
            max_level = int(np.max(tops))
            for level in range(1, max_level + 1):
                shape_k = ((bases <= level) & (level <= tops)).astype(int)
                # decide if this slab is new or extends the last one
                if level == 1 or not np.array_equal(
                    ((bases <= level - 1) & ((level - 1) <= tops)).astype(int),
                    shape_k,
                ):
                    shapes_list.append(shape_k)
                    shapes_weight_list.append(1)
                else:
                    shapes_weight_list[-1] += 1

            # stack and weight
            shapes_arr = np.stack(shapes_list, axis=2)  # (dim_z, dim_x, num_shapes)
            shapes_w_arr = (
                np.array(shapes_weight_list) / self.num_of_levels * tmp_fluence_dict["cal_fac"]
            )
            num_shapes = shapes_arr.shape[2]

            # Build beam sequencing structure
            beam_sequencing = {
                "num_of_shapes": num_shapes,
                "shapes": shapes_arr,
                "shapes_weight": shapes_w_arr,
                "bixel_ix": np.arange(offset, offset + num_of_rays_per_beam),
                "fluence": d0,
            }

            # Compute the summed weighted shape for the beam.
            beam_sum = np.zeros((dim_z, dim_x))
            for j in range(num_shapes):
                beam_sum += (
                    beam_sequencing["shapes"][:, :, j] * beam_sequencing["shapes_weight"][j]
                )
            beam_sequencing["sum_shapes"] = beam_sum
            # Map summed shapes back into a fluence vector using the computed indices.
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = beam_sum.ravel(
                order="F"
            )[ind_in_fluence_mx]

            sequencing["beam"].append(beam_sequencing)
            offset += num_of_rays_per_beam

        # Finalize sequencing result
        result_tmp = result.model_dump()
        # we do this so that "beam" in aperture_info does not overwrite the beam in sequencing
        result_tmp["sequence"] = sequencing
        ap_info = self.sequencing_to_aperture_info(sequencing, stf)
        # merge per‐beam aperture info into each beam entry
        for idx, beam_info in enumerate(ap_info.get("beam", [])):
            result_tmp["sequence"]["beam"][idx].update(beam_info)
        # merge any other (global) aperture_info key
        for key, val in ap_info.items():
            if key != "beam":
                result_tmp["sequence"][key] = val

        # Interpolate dose.
        if dij is not None:
            cubes = dij.compute_result_ct_grid(sequencing["fluence"])

        # if hasattr(result, 'w_dao'):
        #     pass
        # TODO: Currently overwriting quantities in main result. In the future might need to update sequence
        return validate_result({**cubes, **result_tmp})

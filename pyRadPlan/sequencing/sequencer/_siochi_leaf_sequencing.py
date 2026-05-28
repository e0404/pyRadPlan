"""Multi-leaf collimator leaf sequencing using the Siochi algorithm.

Implements the Siochi (1999) leaf sequencing optimization algorithm that minimizes
delivery time for static intensity-modulated radiation therapy (IMRT) fields.
The algorithm uses an "intensity solid paradigm" with rod-pushing techniques
and includes corrections for leaf collision constraints and tongue-and-groove effects.

References
----------
.. [1] Siochi, R. A. C. (1999). Minimizing static intensity modulation delivery
       time using an intensity solid paradigm. International Journal of Radiation
       Oncology*Biology*Physics, 43(3), 671-680.
       https://doi.org/10.1016/S0360-3016(98)00430-1
.. [2] matRad development team. matRad - An open source multi-modality
       radiation treatment planning system. https://github.com/e0404/matRad

Notes
-----
The algorithm creates segment sets through intensity map operations subject to
physical constraints, evaluates delivery times, and selects the optimal configuration.
Includes tongue-and-groove underdosage removal and minimal leakage.
"""

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
    ) -> tuple:
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

    def _decompose_port(self, dk: np.ndarray, dim_z: np.ndarray, dim_x: np.ndarray) -> tuple:
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

    def sequence(
        self, result: PlanResult, stf: SteeringInformation, dij: Dij = None
    ) -> PlanResult:
        """
        Execute the Siochi leaf sequencing algorithm.

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
            # d0 = tmp_fluence_dict["d_0"]
            dim_z = tmp_fluence_dict["dim_z"]
            dim_x = tmp_fluence_dict["dim_x"]

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

            # Build beam sequencing structure
            beam_sequencing = self._build_beam_sequencing(
                shapes_list, shapes_weight_list, tmp_fluence_dict, offset, num_of_rays_per_beam
            )

            # Map summed shapes back into a fluence vector using the computed indices.
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = beam_sequencing[
                "sum_shapes"
            ].ravel(order="C")[tmp_fluence_dict["indices"]]

            sequencing["beam"].append(beam_sequencing)
            offset += num_of_rays_per_beam

        # Finalize sequencing result
        result_tmp = result.model_dump()
        # Pop old quantities because they are no longer correct after sequencing.
        # TODO: might change this in the future. Keeping old quantities and passing new ones
        # to result.sequence
        result_tmp.pop("quantities", None)
        result_tmp.pop("beam_quantities", None)
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
        cubes = {}
        if dij is not None:
            cubes = dij.compute_result_ct_grid(sequencing["fluence"]).model_dump()

        # if hasattr(result, 'w_dao'):
        #     pass
        # TODO: Currently overwriting quantities in main result. In the future might need to update sequence
        return validate_result({**cubes, **result_tmp})

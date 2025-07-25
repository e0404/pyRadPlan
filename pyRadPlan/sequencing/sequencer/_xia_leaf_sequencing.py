"""Multi-leaf collimator leaf sequencing using the Xia algorithm.

Implements the Xia & Verhey (1998) leaf sequencing algorithm for intensity-modulated
radiation therapy (IMRT) using the "stop and shoot" method with static multileaf
collimator (MLC) segments. The algorithm minimizes the number of required segments
by using a power-of-2 intensity sequence and optimizing leaf positions to irradiate
the largest possible area in each segment.

References
----------
.. [1] Xia, P., & Verhey, L. J. (1998). Multileaf collimator leaf sequencing
       algorithm for intensity modulated beams with multiple static segments.
       Medical Physics, 25(8), 1424-1434. https://doi.org/10.1118/1.598315
.. [2] matRad development team. matRad - An open source multi-modality
       radiation treatment planning system. https://github.com/e0404/matRad

Notes
-----
Supports both "reducing levels" and "sliding window" techniques for MLC
systems with different motion constraints.
"""

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
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Union[Plan, dict] = None):
        """Initialize the Sequencer class."""

        self.mode = "reducing_levels"  # 'sliding_window' or 'reducing_levels'
        super().__init__(pln)

    def _apply_reducing_levels(
        self, opening_mx: np.ndarray, d_k: np.ndarray, dim_of_fluence_mx_z: np.ndarray
    ) -> np.ndarray:
        """
        Apply the reducing levels technique.

        Parameters
        ----------
        opening_mx : np.ndarray
            The opening matrix.
        d_k : np.ndarray
            The current fluence values for each bixel.
        dim_of_fluence_mx_z : np.ndarray
            The dimensions of the fluence matrix in the z-direction.

        Returns
        -------
        np.ndarray
            The modified opening matrix.
        """
        for row in range(dim_of_fluence_mx_z):
            row_vals = opening_mx[row, :] * d_k[row, :]
            if row_vals.max() > 0:
                max_ix = int(np.argmax(row_vals))

                # Find first zero *after* max_ix
                zeros_after = np.where(~opening_mx[row, max_ix + 1 :])[0]
                if zeros_after.size > 0:
                    close_ix = max_ix + 1 + int(zeros_after[0])
                    opening_mx[row, close_ix:] = False

                # Find last zero *before* max_ix
                zeros_before = np.where(~opening_mx[row, :max_ix])[0]
                if zeros_before.size > 0:
                    open_ix = int(zeros_before[-1])
                    opening_mx[row, : open_ix + 1] = False
        return opening_mx

    def _apply_sliding_window(
        self, opening_mx: np.ndarray, dim_of_fluence_mx_z: np.ndarray
    ) -> np.ndarray:
        """
        Apply the sliding window technique.

        Parameters
        ----------
        opening_mx : np.ndarray
            The opening matrix.
        dim_of_fluence_mx_z : int
            The number of rows in the fluence matrix.

        Returns
        -------
        np.ndarray
            The modified opening matrix.
        """
        for row in range(dim_of_fluence_mx_z):
            row_open = np.where(opening_mx[row, :])[0]
            if row_open.size > 0:
                open_ix = row_open[0]

                # Find first zero *after* open_ix
                zeros_after = np.where(~opening_mx[row, open_ix + 1 :])[0]
                if zeros_after.size > 0:
                    close_ix = open_ix + 1 + zeros_after[0]
                    opening_mx[row, close_ix:] = False
        return opening_mx

    def sequence(
        self, result: PlanResult, stf: SteeringInformation, dij: Dij = None
    ) -> PlanResult:
        """
        Execute the xia sequencing algorithm.

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
            The updated result object after sequencing.
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
            # l_0 = l_k

            shapes_list = []
            shapes_weight_list = []

            k = 0

            while l_k > 0:
                k += 1
                m = max(math.floor(math.log2(l_k)), 1)
                d_k_value = math.floor(2 ** (m - 1))

                # Calculate opening matrix
                opening_mx = d_k >= d_k_value

                # Apply the appropriate technique
                if self.mode == "reducing_levels":
                    opening_mx = self._apply_reducing_levels(
                        opening_mx, d_k, tmp_fluence_dict["dim_z"]
                    )
                elif self.mode == "sliding_window":
                    opening_mx = self._apply_sliding_window(opening_mx, tmp_fluence_dict["dim_z"])

                shape_k = opening_mx.astype(int) * d_k_value

                shapes_list.append(shape_k)
                shapes_weight_list.append(d_k_value)

                d_k = d_k - shape_k
                l_k = int(np.max(d_k))

            # Build beam sequencing structure
            beam_sequencing = self._build_beam_sequencing(
                shapes_list, shapes_weight_list, tmp_fluence_dict, offset, num_of_rays_per_beam
            )

            # Map fluence back to weights (using linear indices)
            flat_d0 = d_0.flatten(order="C")
            sequencing["fluence"][offset : offset + num_of_rays_per_beam] = (
                flat_d0[tmp_fluence_dict["indices"]]
                / self.num_of_levels
                * tmp_fluence_dict["cal_fac"]
            )

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
        if dij is not None:
            cubes = dij.compute_result_ct_grid(sequencing["fluence"])

        # if hasattr(result, 'w_dao'):
        #     pass
        # TODO: Currently overwriting quantities in main result. In the future might need to update sequence
        return validate_result({**cubes, **result_tmp})

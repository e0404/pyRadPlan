"""Multi-leaf collimator leaf sequencing using the Engel algorithm.

Multileaf collimator leaf sequencing algorithm
for intensity modulated beams with multiple static segments according
to Engel et al. 2005 Discrete Applied Mathematics.

References
----------
.. [1] Engel, K. (2005). A new algorithm for optimal multileaf collimator
       field segmentation. Discrete Applied Mathematics, 152(1-3), 35-51.
       https://doi.org/10.1016/j.dam.2004.10.007
.. [2] matRad development team. matRad - An open source multi-modality
       radiation treatment planning system. https://github.com/e0404/matRad

Notes
-----
The algorithm processes each beam individually, calculating intervals
and their potentials to determine optimal leaf positions for each segment.
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


class EngelLeafSequencer(PhotonSequencer):
    name = "engel_leaf_sequencing"
    short_name = "engel"
    possible_radiation_modes = ["photons"]

    def __init__(self, pln: Union[Plan, dict] = None):
        """Initialize the Sequencer class."""
        super().__init__(pln)

    def _calculate_essential_intervals(
        self, diff_mat_row: np.ndarray, g_j: float, d_k_row: np.ndarray
    ) -> dict:
        """
        Calculate essential intervals for a given row according to Engel algorithm.

        Parameters
        ----------
        diff_mat_row : np.ndarray
            The difference matrix row.
        g_j : float
            The row complexity gap.
        d_k_row : np.ndarray
            The current fluence values for the row.

        Returns
        -------
        dict
            Dictionary containing interval data (left, right, v, w, u).
        """
        data = {"left": [0], "right": [0], "v": [g_j], "w": [np.inf], "u": [g_j]}

        # Find positive and negative elements in diff_mat_row
        pos_indices = np.where(diff_mat_row > 0)[0]
        neg_indices = np.where(diff_mat_row < 0)[0]

        # Loop over positive elements (possible left interval limits)
        for pos_idx in pos_indices:
            # Loop over negative elements (possible right interval limits)
            for neg_idx in neg_indices:
                # Take only intervals I=[l,r] with l<=r
                if pos_idx <= neg_idx - 1:
                    left = pos_idx + 1  # TODO: one can change the indexing. Using nan instead of 0
                    right = neg_idx  # -1

                    # Calculate v according to Lemma 8
                    if g_j <= abs(diff_mat_row[pos_idx] + diff_mat_row[neg_idx]):
                        v = min(diff_mat_row[pos_idx], -diff_mat_row[neg_idx]) + g_j
                    else:
                        v = (diff_mat_row[pos_idx] - diff_mat_row[neg_idx] + g_j) / 2

                    # Calculate w and u according to equality (11) and (12)
                    w = np.min(d_k_row[pos_idx:neg_idx])
                    u = min(v, w)

                    data["left"].append(left)
                    data["right"].append(right)
                    data["v"].append(v)
                    data["w"].append(w)
                    data["u"].append(u)

        return data

    def _calculate_interval_potential(
        self, data: dict, d_k: float, diff_mat_row: np.ndarray, d_k_row: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the potential of intervals according to Engel algorithm.

        Parameters
        ----------
        data : dict
            Interval data dictionary.
        d_k : float
            Current intensity level.
        diff_mat_row : np.ndarray
            The difference matrix row.
        d_k_row : np.ndarray
            The current fluence values for the row.

        Returns
        -------
        np.ndarray
            Array of potentials for each interval.
        """
        num_intervals = len(data["left"])
        potentials = np.full(num_intervals, -np.inf)

        # Find candidates (intervals with u >= d_k)
        candidates = [i for i in range(num_intervals) if data["u"][i] >= d_k]

        # If no candidates found, return all -inf potentials
        if not candidates:
            return potentials

        for idx in candidates:
            if idx == 0 and data["left"][idx] == 0:
                potentials[idx] = 0
            else:
                left = data["left"][idx]
                right = data["right"][idx]

                # Calculate p1 according to equality (17)
                if abs(d_k - diff_mat_row[left]) < 1e-10 and abs(d_k - d_k_row[left]) > 1e-10:
                    p1 = 1
                else:
                    p1 = 0

                # Calculate p2 according to equality (18)
                if right + 1 < len(diff_mat_row):
                    if (
                        abs(d_k + diff_mat_row[right + 1]) < 1e-10
                        and abs(d_k - d_k_row[right]) > 1e-10
                    ):
                        p2 = 1
                    else:
                        p2 = 0
                else:
                    p2 = 0

                # Calculate p3 according to equality (19)
                p3 = np.sum(np.abs(d_k_row[left : right + 1] - d_k) < 1e-10)

                potentials[idx] = p1 + p2 + p3

        return potentials

    def _select_best_interval(self, data: dict, potentials: np.ndarray) -> tuple:
        """
        Select the interval with maximum potential and maximum length if tie.

        Parameters
        ----------
        data : dict
            Interval data dictionary.
        potentials : np.ndarray
            Array of potentials for each interval.

        Returns
        -------
        tuple
            (left_limit, right_limit) of the selected interval.
        """
        # Check if all potentials are -inf (no valid intervals)
        if np.all(potentials == -np.inf):
            return 0, 0  # Return empty interval

        # Find intervals with maximum potential
        max_potential = np.max(potentials)
        max_pot_indices = np.where(potentials == max_potential)[0]

        if len(max_pot_indices) > 1:
            # Calculate lengths for tied intervals
            lengths = []
            for idx in max_pot_indices:
                if idx == 0 and data["left"][idx] == 0:
                    lengths.append(0)
                else:
                    lengths.append(data["right"][idx] - data["left"][idx] + 1)

            # Select interval with maximum length
            max_length_idx = np.argmax(lengths)
            selected_idx = max_pot_indices[max_length_idx]
        else:
            selected_idx = max_pot_indices[0]

        return data["left"][selected_idx], data["right"][selected_idx]

    def _process_beam_shapes(self, d_k: np.ndarray) -> tuple:
        """
        Process shapes for a single beam using Engel algorithm.

        Parameters
        ----------
        d_k : np.ndarray
            The current fluence matrix.

        Returns
        -------
        tuple
            (shapes_list, shapes_weight_list) containing the generated shapes and weights.
        """
        shapes_list = []
        shapes_weight_list = []
        k = 0
        max_iterations = 1000  # Safety limit to prevent infinite loops

        while np.max(d_k) > 0 and k < max_iterations:
            k += 1

            # Calculate difference matrix by adding zeros at boundaries
            zeros_left = np.zeros((d_k.shape[0], 1))
            zeros_right = np.zeros((d_k.shape[0], 1))
            extended_d_k = np.hstack([zeros_left, d_k, zeros_right])
            diff_mat = np.diff(extended_d_k, axis=1)

            # Calculate complexities
            c = np.sum(np.maximum(0, diff_mat), axis=1)  # TNMU-row-complexity
            com = np.max(c)  # TNMU complexity
            g = com - c  # row complexity gap

            # Initialize segment
            segment = np.zeros_like(d_k)

            # Calculate essential intervals for all rows first (like MATLAB)
            data_all_rows = []
            u = np.zeros(d_k.shape[0])

            for j in range(d_k.shape[0]):
                # Calculate essential intervals for row j
                data = self._calculate_essential_intervals(diff_mat[j, :], g[j], d_k[j, :])
                data_all_rows.append(data)
                if data["u"]:  # Check if u is not empty
                    u[j] = np.max(data["u"])
                else:
                    u[j] = 0

            # Calculate d_k from theorem 9
            d_k_value = np.min(u)

            # If d_k_value is 0 or negative, we can't make progress
            # Canceling instead of inf looping
            if d_k_value <= 0:
                logger.warning(
                    f"Cannot make progress in Engel sequencing at iteration {k}, d_k_value={d_k_value}"
                )
                break

            # Loop over all rows to determine segments using pre-calculated data
            segment_has_content = False
            for j in range(d_k.shape[0]):
                # Use pre-calculated essential intervals for row j
                data = data_all_rows[j]

                # Calculate "potential"
                potentials = self._calculate_interval_potential(
                    data, d_k_value, diff_mat[j, :], d_k[j, :]
                )

                # Select best interval (indexing is important here)
                left_limit, right_limit = self._select_best_interval(data, potentials)

                # Create segment for the selected interval
                if left_limit != 0 and right_limit >= left_limit:
                    # TODO: Here is "0" used as nan just like in matRad
                    # Logic can be reimplemented using nan
                    # If so, change in
                    segment[j, left_limit - 1 : right_limit] = 1
                    segment_has_content = True

            # TODO: This here is a recommended failsafe and might not be needed
            # If no segment was created, try to force progress by setting d_k_value to 1
            # and creating a minimal segment
            if not segment_has_content:
                logger.warning(f"No segment created at iteration {k}, forcing minimal progress")
                # Find the position with maximum value and create a minimal segment
                max_pos = np.unravel_index(np.argmax(d_k), d_k.shape)
                if d_k[max_pos] > 0:
                    segment[max_pos] = 1
                    d_k_value = 1
                    segment_has_content = True
                else:
                    # Break of no pos values are left
                    break

            shape_k = segment
            shapes_list.append(shape_k)
            shapes_weight_list.append(d_k_value)

            # Update D_k by subtracting the current shape (multiplied by the weight)
            d_k = d_k - d_k_value * shape_k

            # TODO: Another failsafe to ensure no negative values. Sould work without
            d_k = np.maximum(0, d_k)

        if k >= max_iterations:
            logger.warning(f"Engel sequencing reached maximum iterations ({max_iterations})")

        return shapes_list, shapes_weight_list

    def sequence(
        self, result: PlanResult, stf: SteeringInformation, dij: Dij = None
    ) -> PlanResult:
        """
        Execute the Engel leaf sequencing algorithm.

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

            # Process shapes for the current beam
            shapes_list, shapes_weight_list = self._process_beam_shapes(d_k.copy())

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
        # Pop old quantities because they are no longer correct after sequencing
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

        # Interpolate dose
        if dij is not None:
            cubes = dij.compute_result_ct_grid(sequencing["fluence"])

        # TODO: Currently overwriting quantities in main result. In the future might need to update sequence
        return validate_result({**cubes, **result_tmp})

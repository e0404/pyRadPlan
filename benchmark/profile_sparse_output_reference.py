"""Frozen CPU collector from dd670e87, for storage performance comparisons."""

from pyRadPlan.stf import SteeringInformation
from typing import Any

import logging

import numpy as np

from pyRadPlan.core.xp_utils import to_numpy

logger = logging.getLogger(__name__)


def _allocate_quantity_matrices(self, dij: dict[str, Any], names: list[str]):
    device = "cpu"
    xp = np  # self.xp
    # initialize shared structure container on first call
    if "shared_structure" not in dij:
        dij["shared_structure"] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

    # Loop over all requested quantities
    for q_name in names:
        # Create dij list for each quantity
        dij[q_name] = np.empty(self.mult_scen.scen_mask.shape, dtype=object)

        # Loop over all scenarios and preallocate quantity containers
        # TODO: write test for this
        for i in range(self.mult_scen.scen_mask.size):
            # Only if there is a scenario we will allocate
            if self.mult_scen.scen_mask.flat[i]:
                if self._calc_dose_direct:
                    dij[q_name].flat[i] = xp.zeros(
                        (self.dose_grid.num_voxels, dij["num_of_beams"]),
                        dtype=xp.float32,
                        device=device,
                    )
                else:
                    # We allocate raw csc sparse matrix structures using
                    # a rough estimat ebased on number of voxels and
                    # beamlets

                    if dij["shared_structure"].flat[i] is None:
                        est_nnz = np.fix(
                            self.assumed_sparsity
                            * self.dose_grid.num_voxels
                            * self._num_of_columns_dij
                        ).astype(np.int64)
                        dij["shared_structure"].flat[i] = {
                            "indices": xp.empty((est_nnz,), dtype=xp.int64, device=device),
                            "indptr": xp.zeros(
                                (self._num_of_columns_dij + 1,), dtype=xp.int64, device=device
                            ),
                            "nnz": 0,
                        }
                    else:
                        # If shared structure already exists, we use its size
                        est_nnz = dij["shared_structure"].flat[i]["indices"].size

                    dij[q_name].flat[i] = {
                        "data": xp.empty((est_nnz,), dtype=xp.float32, device=device),
                    }

        self._computed_quantities.append(q_name)

    self._effective_lateral_cutoff = self.geometric_lateral_cutoff

    return dij


def _fill_dij(
    self,
    bixel: dict,
    dij: dict,
    _stf: SteeringInformation,
    scen_idx: int,
    curr_beam_idx: int,
    curr_ray_idx: int,
    curr_bixel_idx: int,
    bixel_counter: int,
):
    # Debugging timing

    bixel["physical_dose"] = to_numpy(bixel["physical_dose"])

    ix_size = 0
    if bixel and "ix" in bixel:
        ix_val = bixel["ix"]
        ix_size = ix_val.shape[0]

    sub_scen_idx = tuple(np.unravel_index(scen_idx, self.mult_scen.scen_mask.shape))

    if not self._calc_dose_direct:
        shared_struct = dij["shared_structure"][sub_scen_idx]
        # Advance column pointer even when ix_size == 0 to keep CSC valid
        start = shared_struct["indptr"][bixel_counter]
        end = start + ix_size
        shared_struct["indptr"][bixel_counter + 1] = end

        if ix_size > 0:
            need_nnz = shared_struct["nnz"] + ix_size
            if shared_struct["indices"].size < need_nnz:
                logger.debug("Resizing data and indices arrays for all quantities...")
                # heuristic growth factor
                grow = max(ix_size, (self._num_of_columns_dij - bixel_counter) * (ix_size + 1))
                new_size = shared_struct["indices"].size + grow
                shared_struct["indices"].resize((new_size,), refcheck=False)

                # Trigger resize for all quantities associated with this scenario
                for q in self._computed_quantities:
                    dij[q][sub_scen_idx]["data"].resize((new_size,), refcheck=False)
            bixel["ix"] = to_numpy(bixel["ix"])
            shared_struct["indices"][start:end] = bixel["ix"]
            shared_struct["nnz"] += ix_size

    for q_name in self._computed_quantities:
        if self._calc_dose_direct:
            if ix_size > 0:
                bixel["ix"] = to_numpy(bixel["ix"])
                bixel[q_name] = to_numpy(bixel[q_name])
                dij[q_name][sub_scen_idx][bixel["ix"], curr_beam_idx] += (
                    bixel["weight"] * bixel[q_name]
                )
        else:
            data_dict = dij[q_name][sub_scen_idx]

            if ix_size > 0:
                start = dij["shared_structure"][sub_scen_idx]["indptr"][bixel_counter]
                end = start + ix_size
                # Fill values into the allocated slice
                bixel[q_name] = to_numpy(bixel[q_name])
                data_dict["data"][start:end] = bixel[q_name]

    # Bookkeeping of bixel numbers
    # remember beam and bixel number
    if self._calc_dose_direct:
        dij["beam_num"][curr_beam_idx] = curr_beam_idx
        dij["ray_num"][curr_beam_idx] = curr_beam_idx
        dij["bixel_num"][curr_beam_idx] = curr_beam_idx
    else:
        dij["beam_num"][bixel_counter] = curr_beam_idx
        dij["ray_num"][bixel_counter] = curr_ray_idx
        dij["bixel_num"][bixel_counter] = curr_bixel_idx

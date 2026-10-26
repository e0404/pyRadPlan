"""Frozen voxel preparation from 4e747717 for exact and timing comparisons."""

import numpy as np
import SimpleITK as sitk
from pyRadPlan.core import Grid
from pyRadPlan.core.np2sitk import linear_indices_to_grid_coordinates
from pyRadPlan.core.resample import resample_numpy_array
from pyRadPlan.ct import resample_ct


def legacy_voxel_selection(self, ct, cst):
    """Execute the original index unions, resampling, coordinates and masks."""
    if self.voxel_sub_ix is None:
        self.voxel_sub_ix = np.unique(
            np.concatenate([cst.vois[c].indices_numpy for c in range(0, len(cst.vois))])
        )

    tmp_vct_grid_scen = [np.asarray(self.voxel_sub_ix)] * ct.num_of_ct_scen

    self._vct_grid = np.unique(np.concatenate(tmp_vct_grid_scen))

    # Find subindexes for the individual scenarios. This helps with subselection later.
    self._vct_grid_scen_ix = [np.isin(self._vct_grid, c) for c in tmp_vct_grid_scen]

    # Initialize tmp_vdose_grid_scen
    tmp_vdose_grid_scen = [None] * ct.num_of_ct_scen

    if self._ct_grid == self.dose_grid:
        tmp_vdose_grid_scen = tmp_vct_grid_scen
        resampled_ct = ct
    else:
        resampled_ct = resample_ct(
            ct=ct,
            interpolator=sitk.sitkNearestNeighbor,
            target_grid=self.dose_grid,
        )
        self.dose_grid = Grid.from_sitk_image(resampled_ct.cube_hu)

        for s in range(ct.num_of_ct_scen):
            # Receive linear indices and grid locations from the dose grid
            tmp_cube = np.zeros(ct.cube_dim[::-1], dtype=np.float32)
            tmp_cube.ravel()[tmp_vct_grid_scen[s]] = 1

            interpolated_tmp_cube = resample_numpy_array(
                tmp_cube,
                reference_image=ct.cube_hu,
                interpolator=sitk.sitkNearestNeighbor,
                target_image=resampled_ct.cube_hu,
            )

            tmp_vdose_grid_scen[s] = np.where(
                np.abs(np.round(interpolated_tmp_cube)).ravel().astype(bool)
            )

    # Get unique dose grid
    self._vdose_grid = np.unique(np.concatenate(tmp_vdose_grid_scen, axis=0))

    # Find subindexes for the dose grid scenarios
    self._vdose_grid_scen_ix = [np.isin(self._vdose_grid, c) for c in tmp_vdose_grid_scen]

    # Convert CT subscripts to world coordinates.
    self._vox_world_coords = linear_indices_to_grid_coordinates(
        self._vct_grid, self._ct_grid, index_type="numpy"
    )

    self._vox_world_coords_dose_grid = linear_indices_to_grid_coordinates(
        self._vdose_grid, self.dose_grid, index_type="numpy"
    )

    # Create helper masks
    self._vdose_grid_mask = np.zeros(self.dose_grid.num_voxels, dtype=bool)
    self._vdose_grid_mask[self._vdose_grid] = True

    self._vct_grid_mask = np.zeros(np.prod(ct.cube_dim), dtype=bool)
    self._vct_grid_mask[self._vct_grid] = True

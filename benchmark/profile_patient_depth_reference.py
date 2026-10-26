"""Host tissue/depth preparation from 164473bb, adapted to the new private hook."""

import SimpleITK as sitk

from pyRadPlan.core.resample import resample_image
from pyRadPlan.core.xp_utils import to_numpy


def legacy_tissue_data(self, resample_grid=None):
    """Resolve overlaps twice and resample all three property volumes."""
    return (
        *self.get_reference_lq_params(resample_grid=resample_grid),
        self.get_tissue_class_indices(resample_grid=resample_grid),
        self.get_tissue_classes(),
    )


def legacy_depth_selection(self, arrays):
    """Reproduce the image round trip and full-volume SimpleITK resampling."""
    selected = []
    for array, reference in zip(arrays, self._raytracer.cubes):
        image = sitk.GetImageFromArray(to_numpy(array))
        image.CopyInformation(reference)
        if not self.trace_on_dose_grid:
            image = resample_image(image, interpolator=sitk.sitkLinear, target_grid=self.dose_grid)
        if self.keep_rad_depth_cubes:
            self._rad_depth_cubes.append(image)
        values = self.xp.asarray(
            sitk.GetArrayViewFromImage(image),
            dtype=self._vox_world_coords.dtype,
            device=self.device,
        )
        selected.append(self.xp.take(self.xp.reshape(values, (-1,)), self._vdose_grid, axis=0))
    return selected

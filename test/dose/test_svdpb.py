import SimpleITK as sitk
import numpy as np

from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.visualization._plot_slice import plot_slice


# TODO: PhotonEngine does not yet have sub-sampling.
# this here only tests if it runs without error.
def test_photons(test_data_photons):
    pln, ct, cst, stf, dij, result = test_data_photons

    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])
    result_matRad = np.transpose(result["physicalDose"], (2, 0, 1))

    assert np.allclose(result_py, result_matRad, atol=1e-2)

    # Keeping this for debugging:
    # plot_slice(
    #     image_volume=ct,
    #     cst=cst,
    #     overlay=result_matRad_rot,#-result_matRad_rot,
    #     view_slice=5,
    #     plane="axial",
    #     overlay_unit="Gy",
    #     plt_show = True,
    #     use_global_max = False,
    # )

import SimpleITK as sitk
import numpy as np

from pyRadPlan import PhotonPlan, generate_stf, load_tg119
from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.machines import create_bld
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


def test_photons_field_based():
    """Exercise the per-ray kernel interpolation path (_init_ray) on the configured backend.

    Field-based dose calculation is the one configuration that reaches ``_init_ray``'s
    custom fluence branch, which builds its own kernel interpolators via FFT convolution.
    The default configuration skips it entirely, so it needs its own coverage.
    """
    ct, cst = load_tg119()

    leaf_width = 5.0
    positions = [[-20.0, 20.0]] * 4
    boundaries = np.arange(-2 * leaf_width, 2 * leaf_width, leaf_width)

    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": boundaries,
            "leaf_positions": positions,
            "leaf_width": leaf_width,
            "leaf_leakage": 0.1,
        }
    )

    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "generator": "photonSingleBixel",
        "field_based": True,
        "blds": [mlc],
        "resolution": 2.0,
        "energy": 6.0,
    }
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    stf = generate_stf(ct, cst, pln)
    dose = sitk.GetArrayFromImage(calc_dose_forward(ct, cst, stf, pln)["physical_dose"])

    assert np.all(np.isfinite(dose))
    assert np.all(dose >= 0.0)
    assert dose.max() > 0.0

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

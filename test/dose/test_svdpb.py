import pytest
import SimpleITK as sitk
import numpy as np

from pyRadPlan import PhotonPlan, generate_stf, calc_dose_forward, load_tg119
from pyRadPlan.machines import MLC, Jaw
from pyRadPlan.result._beam_sequence import BeamSequence
from pyRadPlan.visualization._plot_slice import plot_slice


# TODO: PhotonEngine does not yet have sub-sampling.
# this here only tests if it runs without error.
def test_photons(test_data_photons):
    pln, ct, cst, stf, dij, result = test_data_photons

    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py.physical_dose.values)
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


def _mlc(opening: float) -> MLC:
    boundaries = np.array([-20.0, -10.0, 0.0, 10.0, 20.0])
    positions = np.tile([-opening, opening], (len(boundaries), 1))
    return MLC(
        device_type="MLCX",
        leaf_position_boundaries=boundaries,
        leaf_positions=positions,
        leaf_width=10.0,
    )


def _forward_dose(ct, cst, mlc_sequence, weights, extra_blds=None):
    pln = PhotonPlan(machine="Generic")
    beam_sequence = BeamSequence(
        mlc_sequence=mlc_sequence, shapes_weight=weights, extra_blds=extra_blds
    )
    pln.prop_stf = {
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "generator": "photonSingleBixel",
        "beam_sequences": [beam_sequence],
        "resolution": 1.0,
        "energy": 6.0,
    }
    stf = generate_stf(ct, cst, pln)
    result = calc_dose_forward(ct, cst, stf, pln)
    return sitk.GetArrayFromImage(result.physical_dose.values), stf


def test_grouped_shapes_dose_equals_sum_of_individual_shape_doses():
    ct, cst = load_tg119()

    mlc_a = _mlc(opening=15.0)
    mlc_b = _mlc(opening=8.0)
    weight_a, weight_b = 1.0, 0.6

    combined_dose, stf_combined = _forward_dose(ct, cst, [mlc_a, mlc_b], [weight_a, weight_b])
    assert len(stf_combined.beams[0].rays) == 1  # grouped into one ray

    dose_a, stf_a = _forward_dose(ct, cst, [mlc_a], [weight_a])
    dose_b, stf_b = _forward_dose(ct, cst, [mlc_b], [weight_b])
    assert len(stf_a.beams[0].rays) == 1
    assert len(stf_b.beams[0].rays) == 1

    np.testing.assert_allclose(combined_dose, dose_a + dose_b, rtol=1e-4, atol=1e-6)

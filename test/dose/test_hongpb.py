import SimpleITK as sitk
import numpy as np

from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import (
    ParticleHongPencilBeamEngine,
    DoseEngineBase,
)


def test_ParticleHongPencilBeamEngine(test_data_protons):
    engine = ParticleHongPencilBeamEngine(test_data_protons[0])
    assert engine
    assert engine.name != None
    assert isinstance(engine, ParticleHongPencilBeamEngine)
    assert isinstance(engine, DoseEngineBase)


def test_protons_cd_forward(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)

    # plot_slice(
    #     ct=ct,
    #     cst=cst,
    #     overlay=result_py-result_matRad_rot,
    #     view_slice=5,
    #     plane="axial",
    #     overlay_unit="Gy",
    #     plt_show = True,
    #     use_global_max = False,
    # )


def test_helium_cd_forward(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_carbon_cd_forward(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon

    result_py = calc_dose_forward(ct, cst, stf, pln)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_influence(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_helium_cd_influence(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_carbon_cd_influence(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)

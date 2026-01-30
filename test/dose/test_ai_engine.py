import SimpleITK as sitk
import numpy as np
import pytest

from pyRadPlan import xp_utils, plot_slice
from pyRadPlan.dose import calc_dose_forward, calc_dose_influence
from pyRadPlan.dose.engines import AIEngine, DoseEngineBase
from pyRadPlan.stf import validate_stf

# Set backends
xp_utils.PREFER_GPU = False
xp_utils.PREFERRED_CPU_ARRAY_BACKEND = "numpy"
xp_utils.PREFERRED_GPU_ARRAY_BACKEND = "torch"

torch = pytest.importorskip("torch", reason="torch is required for AI engine tests.")


def test_AIEngine_init(test_data_protons):
    pln = test_data_protons[0]
    # Modify plan to use AI engine
    pln.prop_dose_calc["engine"] = "AI"
    pln.prop_dose_calc["model"] = "ConvLSTM"

    engine = AIEngine(pln)
    assert engine
    assert engine.name == "AI"
    assert isinstance(engine, AIEngine)
    assert isinstance(engine, DoseEngineBase)


def test_protons_cd_forward(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    pln.prop_dose_calc["engine"] = "AI"
    pln.prop_dose_calc["model"] = "ConvLSTM"

    # Ensure dose grid matches reference data to avoid shape mismatch
    # The reference result is computed on dij.dose_grid
    if hasattr(dij, "dose_grid") and dij.dose_grid is not None:
        pln.prop_dose_calc["dose_grid"] = dij.dose_grid

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)

    assert result_py.shape == result_matRad_rot.shape
    assert np.all(np.isfinite(result_py))
    # assert np.allclose(result_py, result_matRad_rot, atol=1e-3)

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


def test_protons_cd_influence(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    pln.prop_dose_calc["engine"] = "AI"
    pln.prop_dose_calc["model"] = "ConvLSTM"

    if hasattr(dij, "dose_grid") and dij.dose_grid is not None:
        pln.prop_dose_calc["dose_grid"] = dij.dose_grid

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert physical_dose_py_dense.shape == physical_dose_mat_dense.shape
    assert np.all(np.isfinite(physical_dose_py_dense))
    # assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)

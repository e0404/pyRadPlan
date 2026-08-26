import pytest
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

    result_matRad_rot = np.transpose(result["physicalDose"], (2, 0, 1))
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)

    # plot_slice(
    #     image_volume=ct,
    #     cst=cst,
    #     overlay=result_py-result_matRad_rot,
    #     view_slice=5,
    #     plane="axial",
    #     overlay_unit="Gy",
    #     plt_show = True,
    #     use_global_max = False,
    # )


def test_protons_cd_forward_bio_model_none(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "none"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_forward_bio_model_WED(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "WED"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_forward_bio_model_MCN(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "MCN"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_forward_bio_model_CAR(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "CAR"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_forward_bio_model_LSM(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "LSM"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_forward_bio_model_const(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "constant_rbe"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_helium_cd_forward(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.transpose(result["physicalDose"], (2, 0, 1))
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_helium_cd_forward_bio_model_none(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium
    pln.bio_model = "none"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_helium_cd_forward_bio_model_LSM(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium
    pln.bio_model = "LSM"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_carbon_cd_forward(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon

    result_py = calc_dose_forward(ct, cst, stf, pln)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.transpose(result["physicalDose"], (2, 0, 1))
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_carbon_cd_forward_bio_model_none(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon
    pln.bio_model = "none"

    result_py = calc_dose_forward(ct, cst, stf, pln, weights=None)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.swapaxes(result["physicalDose"], 0, 1)
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_oxygen_cd_forward(test_data_oxygen):
    pln, ct, cst, stf, dij, result = test_data_oxygen

    result_py = calc_dose_forward(ct, cst, stf, pln)
    result_py = sitk.GetArrayFromImage(result_py["physical_dose"])

    result_matRad_rot = np.transpose(result["physicalDose"], (2, 0, 1))
    # only comparing to 1e-4 since matRad rounds to 4 digits
    assert np.allclose(result_py, result_matRad_rot, atol=1e-4)


def test_protons_cd_influence(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_protons_cd_influence_bio_model_none(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "none"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_protons_cd_influence_bio_model_WED(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "WED"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_protons_cd_influence_bio_model_MCN(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "MCN"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_protons_cd_influence_bio_model_CAR(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "CAR"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_protons_cd_influence_bio_model_LSM(test_data_protons):
    pln, ct, cst, stf, dij, result = test_data_protons
    pln.bio_model = "LSM"

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


def test_helium_cd_influence_bio_model_none(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium
    pln.bio_model = "none"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_helium_cd_influence_bio_model_LSM(test_data_helium):
    pln, ct, cst, stf, dij, result = test_data_helium
    pln.bio_model = "LSM"

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


def test_carbon_cd_influence_bio_model_none(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon
    pln.bio_model = "none"

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-6)


def test_oxygen_cd_influence(test_data_oxygen):
    pln, ct, cst, stf, dij, result = test_data_oxygen

    dij_py = calc_dose_influence(ct, cst, stf, pln)

    physical_dose_py_dense = dij_py.physical_dose.flat[0].toarray()
    physical_dose_mat_dense = dij.physical_dose.flat[0].toarray()

    assert np.allclose(physical_dose_py_dense, physical_dose_mat_dense, atol=1e-4)


def test_carbon_cd_influence_bio_model_unavailable(test_data_carbon):
    """LET-based models must fail loudly on a machine without LET kernels."""
    pln, ct, cst, stf, dij, result = test_data_carbon
    pln.bio_model = "LSM"

    with pytest.raises(ValueError, match="not available"):
        calc_dose_influence(ct, cst, stf, pln)


def test_carbon_cd_influence_bio_model_unknown(test_data_carbon):
    pln, ct, cst, stf, dij, result = test_data_carbon

    # the plan validates the model name on assignment already
    with pytest.raises(ValueError, match="Unknown biological model"):
        pln.bio_model = "does_not_exist"


def test_carbon_multi_gaussian_reduces_to_single(test_data_carbon, monkeypatch):
    """A multi-Gaussian kernel with all extra weights zero equals the single-Gaussian model."""
    from pathlib import Path

    import pyRadPlan
    from pyRadPlan.dose.engines._base import DoseEngineBase
    from pyRadPlan.machines import load_machine_from_mat, validate_machine

    pln, ct, cst, stf, dij, result = test_data_carbon
    machine_file = Path(pyRadPlan.__file__).parent / "data" / "machines" / "carbon_Generic.mat"
    machine = validate_machine(load_machine_from_mat(machine_file))
    for kernel in machine.pb_kernels.values():
        kernel.sigma_multi = np.stack([kernel.sigma, 2 * kernel.sigma, 3 * kernel.sigma])
        kernel.weight_multi = np.zeros((2, kernel.depths.shape[0]))
    assert machine.has_multi_gaussian_kernel
    monkeypatch.setattr(DoseEngineBase, "load_machine", staticmethod(lambda *_: machine))

    pln.prop_dose_calc["lateral_model"] = "single"
    dose_single = calc_dose_influence(ct, cst, stf, pln).physical_dose.flat[0].toarray()
    pln.prop_dose_calc["lateral_model"] = "multi"
    dose_multi = calc_dose_influence(ct, cst, stf, pln).physical_dose.flat[0].toarray()
    assert dose_multi.max() > 0
    assert np.allclose(dose_multi, dose_single, rtol=1e-6, atol=1e-8)


def test_oxygen_multi_gaussian_conserves_bixel_dose(test_data_oxygen):
    """On a grid fine enough to resolve the kernels, every bixel deposits the same total dose
    with the multi- and the single-Gaussian lateral model (both are normalised)."""
    pln, ct, cst, stf, dij, result = test_data_oxygen
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 2.0, "y": 2.0, "z": 2.0}}
    totals = {}
    for lateral_model in ("single", "multi"):
        pln.prop_dose_calc["lateral_model"] = lateral_model
        dose = calc_dose_influence(ct, cst, stf, pln).physical_dose.flat[0]
        assert dose.min() >= 0
        totals[lateral_model] = np.asarray(dose.sum(axis=0)).ravel()
    assert np.all(totals["multi"] > 0)
    assert np.allclose(totals["multi"], totals["single"], rtol=1e-2)

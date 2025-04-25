import pytest

from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf
from pyRadPlan.dij import validate_dij
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst


@pytest.fixture
def test_data_photons(test_data_photons_raw):
    tmp = test_data_photons_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])

    result = tmp["resultGUI"]

    return pln, ct, cst, stf, dij_matRad, result


@pytest.fixture
def test_data_protons(test_data_protons_raw):
    tmp = test_data_protons_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])

    result = tmp["resultGUI"]

    return pln, ct, cst, stf, dij_matRad, result


@pytest.fixture
def test_data_helium(test_data_helium_raw):
    tmp = test_data_helium_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])

    result = tmp["resultGUI"]

    return pln, ct, cst, stf, dij_matRad, result


@pytest.fixture
def test_data_carbon(test_data_carbon_raw):
    tmp = test_data_carbon_raw
    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])

    result = tmp["resultGUI"]

    return pln, ct, cst, stf, dij_matRad, result

from typing import Dict
import pytest

from pyRadPlan.plan import IonPlan, PhotonPlan
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.dij import Dij
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet


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


def test_validate_protons(test_data_protons):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p = test_data_protons
    assert isinstance(pln_p, IonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, Dict)


def test_validate_helium(test_data_helium):
    pln_h, ct_h, cst_h, stf_h, dij_h, result_h = test_data_helium
    assert isinstance(pln_h, IonPlan)
    assert isinstance(ct_h, CT)
    assert isinstance(cst_h, StructureSet)
    assert isinstance(stf_h, SteeringInformation)
    assert isinstance(dij_h, Dij)
    assert isinstance(result_h, Dict)


def test_validate_carbon(test_data_carbon):
    pln_c, ct_c, cst_c, stf_c, dij_c, result_c = test_data_carbon
    assert isinstance(pln_c, IonPlan)
    assert isinstance(ct_c, CT)
    assert isinstance(cst_c, StructureSet)
    assert isinstance(stf_c, SteeringInformation)
    assert isinstance(dij_c, Dij)
    assert isinstance(result_c, Dict)


def test_validate_photons(test_data_photons):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p = test_data_photons
    assert isinstance(pln_p, PhotonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, Dict)

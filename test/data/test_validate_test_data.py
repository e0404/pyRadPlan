from typing import Dict
import pytest

from pyRadPlan.plan import IonPlan, PhotonPlan, validate_pln
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.result import PlanResult, validate_result


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
    result = validate_result(tmp["resultGUI"])

    return pln, ct, cst, stf, dij_matRad, result


@pytest.fixture
def test_data_engel_seq(test_data_engel_seq_raw):
    tmp = test_data_engel_seq_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    result = validate_result(tmp["resultGUI"])
    result_before_seq = validate_result(tmp["resultGUI_before_seq"])

    return pln, ct, cst, stf, dij_matRad, result, result_before_seq


@pytest.fixture
def test_data_siochi_seq(test_data_siochi_seq_raw):
    tmp = test_data_siochi_seq_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    result = validate_result(tmp["resultGUI"])
    result_before_seq = validate_result(tmp["resultGUI_before_seq"])

    return pln, ct, cst, stf, dij_matRad, result, result_before_seq


@pytest.fixture
def test_data_xia_seq(test_data_xia_seq_raw):
    tmp = test_data_xia_seq_raw

    pln = validate_pln(tmp["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij_matRad = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    result = validate_result(tmp["resultGUI"])
    result_before_seq = validate_result(tmp["resultGUI_before_seq"])

    return pln, ct, cst, stf, dij_matRad, result, result_before_seq


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
    result = validate_result(tmp["resultGUI"])

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
    result = validate_result(tmp["resultGUI"])

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
    result = validate_result(tmp["resultGUI"])

    return pln, ct, cst, stf, dij_matRad, result


def test_validate_protons(test_data_protons):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p = test_data_protons
    assert isinstance(pln_p, IonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, PlanResult)


def test_validate_helium(test_data_helium):
    pln_h, ct_h, cst_h, stf_h, dij_h, result_h = test_data_helium
    assert isinstance(pln_h, IonPlan)
    assert isinstance(ct_h, CT)
    assert isinstance(cst_h, StructureSet)
    assert isinstance(stf_h, SteeringInformation)
    assert isinstance(dij_h, Dij)
    assert isinstance(result_h, PlanResult)


def test_validate_carbon(test_data_carbon):
    pln_c, ct_c, cst_c, stf_c, dij_c, result_c = test_data_carbon
    assert isinstance(pln_c, IonPlan)
    assert isinstance(ct_c, CT)
    assert isinstance(cst_c, StructureSet)
    assert isinstance(stf_c, SteeringInformation)
    assert isinstance(dij_c, Dij)
    assert isinstance(result_c, PlanResult)


def test_validate_photons(test_data_photons):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p = test_data_photons
    assert isinstance(pln_p, PhotonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, PlanResult)


def test_validate_xia_seq(test_data_xia_seq):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p, result_before_seq_p = test_data_xia_seq
    assert isinstance(pln_p, PhotonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, PlanResult)
    assert isinstance(result_before_seq_p, PlanResult)


def test_validate_siochi_seq(test_data_siochi_seq):
    pln_p, ct_p, cst_p, stf_p, dij_p, result_p, result_before_seq_p = test_data_siochi_seq
    assert isinstance(pln_p, PhotonPlan)
    assert isinstance(ct_p, CT)
    assert isinstance(cst_p, StructureSet)
    assert isinstance(stf_p, SteeringInformation)
    assert isinstance(dij_p, Dij)
    assert isinstance(result_p, PlanResult)
    assert isinstance(result_before_seq_p, PlanResult)

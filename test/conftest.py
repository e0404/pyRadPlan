import pytest
import pymatreader
import sys
import os
import SimpleITK as sitk

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+
from pyRadPlan.dose.engines._fredmc import read_sparse_dij_bin_v21


@pytest.fixture
def tg119_raw():
    phantom_data_str = resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")
    phantom_data = pymatreader.read_mat(phantom_data_str)
    return phantom_data["ct"], phantom_data["cst"]


@pytest.fixture
def test_data_photons_raw():
    path = "test/data/photons_testData.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_protons_raw():
    path = "test/data/protons_testData.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_helium_raw():
    path = "test/data/helium_testData.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_carbon_raw():
    path = "test/data/carbon_testData.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_stf_one_beam_raw():
    path = "test/data/stf/stf_test_data_one_beam.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_stf_n_beams_raw():
    path = "test/data/stf/stf_test_data_n_beams.mat"
    tmp = pymatreader.read_mat(path)
    return tmp


@pytest.fixture
def test_data_fred_inp():
    path = "test/data/mc_fred/inp/"
    with open(path + "fred.inp", "r") as file:
        inp = file.read()
    with open(path + "plan/plan.inp", "r") as file:
        plan = file.read()
    with open(path + "plan/planDelivery.inp", "r") as file:
        planDelivery = file.read()
    with open(path + "regions/regions.inp", "r") as file:
        regions = file.read()
    with open(path + "regions/hLut.inp", "r") as file:
        hLut = file.read()
    with open(path + "regions/CTpatient.mhd", "r") as file:
        ct_patient = file.read()
    with open(path + "regions/CTpatient.raw", "rb") as file:
        ct_patient_raw = file.read()
    return {
        "inp": inp,
        "plan": plan,
        "planDelivery": planDelivery,
        "regions": regions,
        "hLut": hLut,
        "CTpatient.mhd": ct_patient,
        "CTpatient.raw": ct_patient_raw,
    }


@pytest.fixture
def test_data_fred_out():
    path = "test/data/mc_fred/out/"
    phantom_dose = sitk.GetArrayFromImage(sitk.ReadImage(path + "score/Phantom.Dose.mhd"))
    phantom_dij = read_sparse_dij_bin_v21(path + "scoreij/Phantom.Dose.bin")

    return phantom_dose, phantom_dij

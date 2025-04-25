import pytest
import pymatreader
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+


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

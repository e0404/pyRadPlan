import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst


# Shared QApplication fixture for all GUI tests
@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.fixture
def test_data_photons(test_data_photons_raw):
    tmp = test_data_photons_raw

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    result = tmp["resultGUI"]

    return ct, cst, result


@pytest.fixture
def test_data_protons(test_data_protons_raw):
    tmp = test_data_protons_raw

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    result = tmp["resultGUI"]

    return ct, cst, result


@pytest.fixture
def test_data_helium(test_data_helium_raw):
    tmp = test_data_helium_raw

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    result = tmp["resultGUI"]

    return ct, cst, result


@pytest.fixture
def test_data_carbon(test_data_carbon_raw):
    tmp = test_data_carbon_raw

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    result = tmp["resultGUI"]

    return ct, cst, result

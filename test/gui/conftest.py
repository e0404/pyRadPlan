import gc

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


# Free each test's widgets while the QApplication is still alive. Left to interpreter
# shutdown, the PySide6/pyqtgraph wrappers are collected after Qt is torn down, which
# segfaults on some Python versions. Never force a C++ delete here (deleteLater): pyqtgraph
# items still referenced from Python would then be freed twice.
@pytest.fixture(autouse=True)
def _collect_widgets():
    yield
    gc.collect()


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


@pytest.fixture
def test_data_oxygen(test_data_oxygen_raw):
    tmp = test_data_oxygen_raw

    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    result = tmp["resultGUI"]

    return ct, cst, result


@pytest.fixture
def dark_palette():
    """A dark palette with distinct Window/Base/WindowText colors, returned with them."""
    from PySide6.QtGui import QColor, QPalette

    colors = {"window": "#202020", "base": "#101010", "text": "#f0f0f0"}
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(colors["window"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(colors["base"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    return palette, colors

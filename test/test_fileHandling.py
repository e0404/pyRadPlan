import pytest
import os
import sys

if sys.version_info < (3, 10):
    import importlib_resources as resources  # Backport for older versions
else:
    from importlib import resources  # Standard from Python 3.9+

from pyRadPlan.io._matlab_file_handler import MatlabFileHandler
import pyRadPlan.io.matfile as matfile
from pyRadPlan.io import available_phantoms, load_patient, load_tg119, resolve_phantom
from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet


# Paths
@pytest.fixture
def tg119_path():
    return resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")


def test_file_handler(tmp_path, tg119_path):
    """Test loading method."""

    tg119_load = matfile.load(tg119_path)

    tg119 = {
        "ct": tg119_load["ct"],
        "cst": tg119_load["cst"],
    }

    file_handler = MatlabFileHandler(tmp_path)
    file_handler.save(**tg119)

    assert os.path.exists(tmp_path.joinpath("ct.mat"))
    assert os.path.exists(tmp_path.joinpath("cst.mat"))

    ct = file_handler.load("ct")
    assert isinstance(ct, dict)
    cst = file_handler.load("cst")
    assert isinstance(cst, dict)

    file_handler.delete("ct", "cst")
    assert not os.path.exists(tmp_path.joinpath("ct.mat"))
    assert not os.path.exists(tmp_path.joinpath("cst.mat"))


def test_load_patient_from_file(tg119_path):
    ct, cst = load_patient(tg119_path)

    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)


def test_load_tg119(tg119_path):
    ct, cst = load_patient(tg119_path)
    ct2, cst2 = load_tg119()

    assert isinstance(ct2, CT)
    assert isinstance(cst2, StructureSet)
    assert ct == ct2
    assert cst == cst2


def test_load_file_extradata(tg119_path):
    extra_data = {}
    extra_plan_data = {}
    ct, cst = load_patient(tg119_path, extra_data=extra_data, extra_plan_data=extra_plan_data)

    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)
    assert isinstance(extra_data, dict)
    assert isinstance(extra_plan_data, dict)


def test_load_invalid_file():
    with pytest.raises(FileNotFoundError):
        _, _ = load_patient("unsupported_file.txt")


def test_available_phantoms(tg119_path):
    phantoms = available_phantoms()

    assert "TG119" in phantoms
    assert str(phantoms["TG119"]) == str(tg119_path)


def test_resolve_phantom_case_and_extension_insensitive(tg119_path):
    for name in ("TG119", "tg119", "TG119.mat", "tg119.MAT"):
        assert str(resolve_phantom(name)) == str(tg119_path)


def test_resolve_phantom_unknown_lists_available():
    with pytest.raises(ValueError, match="TG119"):
        resolve_phantom("does_not_exist")

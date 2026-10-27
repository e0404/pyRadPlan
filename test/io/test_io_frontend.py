"""Tests for the top-level frontend: format detection and save_data routing."""

import os

import pytest

from pyRadPlan.io import load_data, save_data, load_tg119, phantom_path, available_phantoms
from pyRadPlan.io._factory import detect_format, DEFAULT_SAVE_FORMAT
from pyRadPlan.io._load_save import _normalize_format


def test_detect_format_by_extension(tmp_path):
    f = tmp_path / "x.mat"
    f.write_bytes(b"")
    assert detect_format(f) == "mat"


def test_detect_format_dicom_dir(dicom_dir):
    assert detect_format(dicom_dir) == "dcm"


def test_detect_format_unsupported(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"")
    with pytest.raises(ValueError):
        detect_format(f)


@pytest.mark.parametrize(
    "value,expected",
    [("mat", "mat"), (".mat", "mat"), ("matlab", "mat"), ("dcm", "dcm"), ("dicom", "dcm")],
)
def test_normalize_format(value, expected):
    assert _normalize_format(value) == expected


def test_normalize_format_invalid():
    with pytest.raises(ValueError):
        _normalize_format("xyz")


def test_save_data_extensionless_filename_uses_default(tmp_path):
    ct, cst = load_tg119()
    target = tmp_path / "patient"  # no extension
    written = save_data(ct=ct, file_name=str(target))
    assert DEFAULT_SAVE_FORMAT == "mat"
    assert written == str(target) + ".mat"
    assert os.path.exists(written)


def test_save_data_explicit_format_overrides(tmp_path):
    ct, _ = load_tg119()
    target = tmp_path / "patient.dat"
    written = save_data(ct=ct, file_name=str(target), format="mat")
    # Extension already present -> kept; format forces the .mat backend/content.
    assert os.path.exists(written)


def test_save_data_no_filename_writes_per_object(tmp_path, monkeypatch):
    ct, cst = load_tg119()
    monkeypatch.chdir(tmp_path)
    written = save_data(ct=ct, cst=cst)
    assert sorted(written) == sorted(["ct.mat", "cst.mat"])
    assert (tmp_path / "ct.mat").exists()
    assert (tmp_path / "cst.mat").exists()


def test_save_data_with_dict(tmp_path):
    ct, cst = load_tg119()
    out = tmp_path / "p.mat"
    written = save_data({"ct": ct, "cst": cst}, file_name=str(out))
    assert os.path.exists(written)
    data = load_data(out)
    assert "ct" in data and "cst" in data


def test_save_data_nothing_raises():
    with pytest.raises(ValueError):
        save_data()


@pytest.mark.parametrize("name", ["TG119", "tg119", "TG119.mat", "tg119.MAT"])
def test_phantom_path_resolves_by_name(name):
    """Bundled phantoms resolve case-insensitively, with or without the extension."""
    assert str(phantom_path(name)).lower().endswith("tg119.mat")


def test_phantom_path_unknown_lists_available():
    with pytest.raises(FileNotFoundError, match="TG119"):
        phantom_path("no_such_phantom")


def test_available_phantoms_lists_tg119():
    assert "TG119" in available_phantoms()


@pytest.fixture
def phantom_folders(tmp_path, monkeypatch):
    """An isolated user phantom folder and one shared base data root holding phantoms."""
    from pyRadPlan._settings import get_settings

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data"))
    user = tmp_path / "data" / "phantoms"
    shared = tmp_path / "course" / "phantoms"
    user.mkdir(parents=True)
    shared.mkdir(parents=True)
    settings = get_settings()
    previous = settings.base_data_path
    settings.base_data_path = str(tmp_path / "course")
    yield user, shared
    settings.base_data_path = previous


def test_phantoms_are_found_in_the_user_and_shared_folders(phantom_folders):
    user, shared = phantom_folders
    ct, cst = load_tg119()
    (user / "MyCase.mat").write_bytes(phantom_path("TG119").read_bytes())
    save_data(file_name=shared / "Course Case.npz", ct=ct, cst=cst)
    # the shipped phantom wins over files of the same name, whatever their format
    save_data(file_name=shared / "tg119.npz", ct=ct, cst=cst)
    (shared / "notes.txt").write_text("not a phantom", encoding="utf-8")

    assert available_phantoms() == ["TG119", "MyCase", "Course Case"]
    assert phantom_path("mycase") == user / "MyCase.mat"
    assert phantom_path("Course Case.npz") == shared / "Course Case.npz"
    assert str(phantom_path("TG119")).lower().endswith("tg119.mat")
    data = load_data(phantom_path("Course Case"))
    assert data["ct"] is not None and data["cst"] is not None

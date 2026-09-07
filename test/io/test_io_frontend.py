"""Tests for the top-level frontend: format detection and save_data routing."""

import os

import pytest

from pyRadPlan.io import load_data, save_data, load_tg119
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

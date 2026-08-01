"""Tests for the import/export factory (registration and lookup)."""

import pytest

from pyRadPlan.io import get_importer, get_exporter, get_available_formats
from pyRadPlan.io.matlab import MatlabImporter, MatlabExporter
from pyRadPlan.io.dicom import DicomImporter, DicomExporter
from pyRadPlan.io.base import BaseImporter, BaseExporter
from pyRadPlan.io._factory import (
    register_importer,
    register_exporter,
    is_container_format,
    default_extension,
    format_for_extension,
    format_for_path,
)


def test_builtin_formats_are_registered():
    assert get_available_formats() == {"mat", "dcm", "npz", "nifti", "nrrd", "meta", "pickle"}
    assert get_importer("mat") is MatlabImporter
    assert get_exporter("mat") is MatlabExporter
    assert get_importer("dcm") is DicomImporter
    assert get_exporter("dcm") is DicomExporter


def test_get_unknown_format_raises():
    with pytest.raises(ValueError):
        get_importer("nope")
    with pytest.raises(ValueError):
        get_exporter("nope")


def test_format_metadata():
    assert is_container_format("mat") is True
    assert is_container_format("dcm") is False
    assert is_container_format("npz") is True
    assert is_container_format("nifti") is False
    assert default_extension("mat") == ".mat"
    assert default_extension("npz") == ".npz"
    assert default_extension("nifti") == ".nii.gz"
    assert format_for_extension(".MAT") == "mat"
    assert format_for_extension(".dcm") == "dcm"
    assert format_for_extension(".npz") == "npz"
    assert format_for_extension(".unknown") is None


def test_format_for_path_compound_extensions():
    # Suffix-based matching handles compound extensions like .nii.gz.
    assert format_for_path("patient/ct.nii.gz") == "nifti"
    assert format_for_path("ct.nii") == "nifti"
    assert format_for_path("seg.nrrd") == "nrrd"
    assert format_for_path("vol.mha") == "meta"
    assert format_for_path("vol.mhd") == "meta"
    assert format_for_path("x.mat") == "mat"
    assert format_for_path("x.pkl") == "pickle"
    assert format_for_path("x.unknown") is None


def test_register_rejects_wrong_base():
    class NotAnImporter:
        format = "x"
        extensions = (".x",)

    with pytest.raises(ValueError):
        register_importer(NotAnImporter)
    with pytest.raises(ValueError):
        register_exporter(NotAnImporter)


def test_register_requires_format_and_extensions():
    class NoFormat(BaseImporter):
        extensions = (".x",)

        def load_ct(self): ...

        def load_cst(self, ct=None): ...

    with pytest.raises(ValueError):
        register_importer(NoFormat)

    class NoExtensions(BaseExporter):
        format = "x"

        def save(self, **kwargs): ...

    with pytest.raises(ValueError):
        register_exporter(NoExtensions)


def test_duplicate_registration_warns():
    with pytest.warns(UserWarning, match="already registered"):
        register_importer(MatlabImporter)

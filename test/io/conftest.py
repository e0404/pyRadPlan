"""Shared fixtures for the io test suite."""

import sys
from pathlib import Path

import pytest

if sys.version_info < (3, 10):
    import importlib_resources as resources
else:
    from importlib import resources


@pytest.fixture
def dicom_dir() -> Path:
    """Directory holding the bundled DICOM test data (CT series, RTSTRUCT, RTDOSE)."""
    return Path(__file__).parent.parent / "data" / "dicom"


@pytest.fixture
def dicom_reference_mat(dicom_dir) -> Path:
    """matRad reference (.mat) for the same patient as the DICOM test data."""
    return dicom_dir / "dicom_testData.mat"


@pytest.fixture
def nifti_dir() -> Path:
    """Directory with NIfTI test data (ct + cst) for the same patient as the DICOM data."""
    return Path(__file__).parent.parent / "data" / "nifti"


@pytest.fixture
def nrrd_dir() -> Path:
    """Directory with NRRD test data (ct + cst, with embedded metadata)."""
    return Path(__file__).parent.parent / "data" / "nrrd"


@pytest.fixture
def meta_dir() -> Path:
    """Directory with MetaImage test data (ct + cst, with embedded metadata)."""
    return Path(__file__).parent.parent / "data" / "meta_image"


@pytest.fixture
def tg119_path():
    """Path to the bundled TG119 phantom .mat file."""
    return resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")

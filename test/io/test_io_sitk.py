"""Tests for the SimpleITK-based backends (NIfTI, NRRD, MetaImage)."""

import numpy as np
import SimpleITK as sitk
import pytest

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_voi, validate_cst
from pyRadPlan.io import load_data, load_patient, save_data, NiftiHandler
from pyRadPlan.io._factory import detect_format, is_container_format

FORMATS = ["nifti", "nrrd", "meta"]


# --------------------------------------------------------------------------
# Helpers / synthetic data
# --------------------------------------------------------------------------


def _make_ct() -> CT:
    arr = np.arange(10 * 12 * 8, dtype=np.float32).reshape(10, 12, 8)  # (z, y, x)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing((2.0, 3.0, 4.0))
    img.SetOrigin((-5.0, -10.0, -2.0))
    return validate_ct(cube_hu=img)


def _make_cst(ct: CT) -> StructureSet:
    shape = sitk.GetArrayFromImage(ct.cube_hu).shape
    m_target = np.zeros(shape, dtype=np.uint8)
    m_target[1:3, 1:4, 1:3] = 1
    m_core = np.zeros(shape, dtype=np.uint8)
    m_core[5:7, 6:9, 4:6] = 1  # disjoint from target -> lossless label-map round-trip

    target = validate_voi(
        name="Target", voi_type="TARGET", mask=m_target, ct_image=ct, alpha_x=0.2, beta_x=0.03
    )
    core = validate_voi(name="Core", voi_type="OAR", mask=m_core, ct_image=ct)
    return validate_cst([target, core], ct)


def _dummy_dose(ct: CT) -> sitk.Image:
    arr = np.random.default_rng(0).random(
        sitk.GetArrayFromImage(ct.cube_hu).shape, dtype=np.float32
    )
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(ct.cube_hu)
    return dose


def _ct_close(a: CT, b: CT) -> bool:
    return (
        a.cube_hu.GetSize() == b.cube_hu.GetSize()
        and np.allclose(a.cube_hu.GetSpacing(), b.cube_hu.GetSpacing())
        and np.allclose(a.cube_hu.GetOrigin(), b.cube_hu.GetOrigin())
        and np.allclose(a.cube_hu.GetDirection(), b.cube_hu.GetDirection())
        and np.allclose(sitk.GetArrayFromImage(a.cube_hu), sitk.GetArrayFromImage(b.cube_hu))
    )


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


# --------------------------------------------------------------------------
# Import of the bundled test data (same patient as the DICOM/mat reference)
# --------------------------------------------------------------------------


def test_import_nrrd_data(nrrd_dir):
    data = load_data(nrrd_dir)
    ct, cst = data["ct"], data["cst"]

    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert np.allclose(ct.cube_hu.GetSpacing(), (10.0, 15.0, 5.0))
    assert np.allclose(ct.cube_hu.GetOrigin(), (-75.0, -150.0, -25.0))

    # NRRD carries embedded pyradplan/Slicer metadata, so names/types survive.
    by_name = {v.name: v for v in cst.vois}
    assert set(by_name) == {"Target", "Body", "Core"}
    assert by_name["Target"].voi_type == "TARGET"

    # Label map -> disjoint, non-empty VOIs.
    sets = list(_voxel_sets(cst).values())
    assert all(len(s) > 0 for s in sets)
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            assert sets[i].isdisjoint(sets[j])


def test_import_meta_data(meta_dir):
    data = load_data(meta_dir)
    ct, cst = data["ct"], data["cst"]

    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert np.allclose(ct.cube_hu.GetOrigin(), (-75.0, -150.0, -25.0))

    # MetaImage carries embedded metadata (no sidecar), so names/types survive.
    by_name = {v.name: v for v in cst.vois}
    assert set(by_name) == {"Target", "Body", "Core"}
    assert by_name["Target"].voi_type == "TARGET"


def test_import_nifti_data(nifti_dir):
    data = load_data(nifti_dir)
    ct, cst = data["ct"], data["cst"]
    assert ct.cube_hu.GetSize() == (15, 20, 10)
    # NIfTI cannot store arbitrary metadata and there is no sidecar -> 3 VOIs, generic names.
    assert len(cst.vois) == 3


def test_real_data_ct_identical_across_formats(nifti_dir, nrrd_dir, meta_dir):
    arrays = [
        sitk.GetArrayFromImage(load_data(d)["ct"].cube_hu) for d in (nifti_dir, nrrd_dir, meta_dir)
    ]
    assert np.array_equal(arrays[0], arrays[1])
    assert np.array_equal(arrays[0], arrays[2])


def test_load_patient_single_nifti_file(nifti_dir):
    ct, cst = load_patient(nifti_dir / "ct.nii.gz")
    assert isinstance(ct, CT)
    assert ct.cube_hu.GetSize() == (15, 20, 10)
    assert cst is None  # a lone image file is just a CT


# --------------------------------------------------------------------------
# Round-trip (export + import), parametrized over all three formats
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_ct(fmt, tmp_path):
    ct = _make_ct()
    out = tmp_path / f"patient_{fmt}"
    save_data(ct=ct, format=fmt, file_name=str(out))

    assert detect_format(out) == fmt
    assert is_container_format(fmt) is False
    assert _ct_close(ct, load_data(out)["ct"])


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_dose(fmt, tmp_path):
    ct = _make_ct()
    dose = _dummy_dose(ct)
    out = tmp_path / f"withdose_{fmt}"
    save_data(ct=ct, dose=dose, format=fmt, file_name=str(out))

    data = load_data(out)
    assert isinstance(data["dose"], sitk.Image)
    assert np.allclose(
        sitk.GetArrayFromImage(data["dose"]), sitk.GetArrayFromImage(dose), atol=1e-5
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_roundtrip_cst(fmt, tmp_path):
    ct = _make_ct()
    cst = _make_cst(ct)
    out = tmp_path / f"cst_{fmt}"
    save_data(ct=ct, cst=cst, format=fmt, file_name=str(out))

    cst2 = load_data(out)["cst"]
    assert isinstance(cst2, StructureSet)
    # Disjoint VOIs -> exact label-map round-trip; sidecar preserves metadata (all formats).
    assert _voxel_sets(cst) == _voxel_sets(cst2)
    by_name = {v.name: v for v in cst2.vois}
    assert set(by_name) == {"Target", "Core"}
    assert by_name["Target"].voi_type == "TARGET"
    assert by_name["Target"].alpha_x == pytest.approx(0.2)
    assert by_name["Target"].beta_x == pytest.approx(0.03)


def test_single_file_ct_roundtrip(tmp_path):
    ct = _make_ct()
    out = tmp_path / "ct.nii.gz"
    written = save_data(ct=ct, file_name=str(out))
    assert str(written).endswith("ct.nii.gz")
    assert detect_format(out) == "nifti"
    ct2, _ = load_patient(out)
    assert _ct_close(ct, ct2)


def test_single_file_target_rejects_cst(tmp_path):
    ct = _make_ct()
    cst = _make_cst(ct)
    with pytest.raises(ValueError):
        NiftiHandler(tmp_path / "seg.nii.gz").save(cst=cst)


def test_single_file_target_rejects_ct_and_dose(tmp_path):
    # A single image file can hold only one image; refuse rather than drop the dose.
    ct = _make_ct()
    dose = sitk.Cast(ct.cube_hu, sitk.sitkFloat32)
    with pytest.raises(ValueError):
        NiftiHandler(tmp_path / "image.nii.gz").save(ct=ct, dose=dose)

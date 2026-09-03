"""Tests for the DICOM import/export backend.

The bundled DICOM data (CT series, RTSTRUCT, RTDOSE) describes the same patient
as ``dicom_testData.mat`` (a matRad export), which lets us cross-validate the
DICOM importer against a known-good reference, and verify import/export
round-trips.
"""

import os
import shutil
from unittest import mock

import numpy as np
import pydicom
import SimpleITK as sitk
import pytest

from pyRadPlan.core import ProgressReport, observe_reports
from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.io import load_patient, load_data
from pyRadPlan.io.matlab import _matfile as matfile
from pyRadPlan.io.dicom import DicomImporter, DicomExporter, DicomHandler
from pyRadPlan.io.dicom._import_dose import import_dose
from pyRadPlan.io.dicom._helpers import determine_structure_type, generate_colors


def _voxel_sets(cst: StructureSet) -> dict:
    return {v.name.lower(): set(v.indices_numpy.tolist()) for v in cst.vois}


def _dice(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("PTV", "TARGET"),
        ("Target", "TARGET"),
        ("CTV_boost", "TARGET"),
        ("Body", "EXTERNAL"),
        ("External", "EXTERNAL"),
        ("Core", "OAR"),
        ("Lung_L", "OAR"),
    ],
)
def test_determine_structure_type(name, expected):
    assert determine_structure_type(name) == expected


def test_generate_colors():
    colors = generate_colors(5)
    assert len(colors) == 5
    assert all(len(c) == 3 and all(0 <= v <= 255 for v in c) for c in colors)


# --------------------------------------------------------------------------
# Import (cross-validated against the matRad reference)
# --------------------------------------------------------------------------


def test_import_ct_matches_reference(dicom_dir, dicom_reference_mat):
    ref_ct = validate_ct(matfile.load(dicom_reference_mat)["ct"])
    ct = DicomImporter(dicom_dir).load_ct()

    assert isinstance(ct, CT)
    assert ct.cube_hu.GetSize() == ref_ct.cube_hu.GetSize()
    assert np.allclose(ct.cube_hu.GetSpacing(), ref_ct.cube_hu.GetSpacing())
    assert np.allclose(ct.cube_hu.GetOrigin(), ref_ct.cube_hu.GetOrigin())
    assert np.allclose(
        sitk.GetArrayFromImage(ct.cube_hu),
        sitk.GetArrayFromImage(ref_ct.cube_hu),
        atol=1e-4,
    )


def test_import_cst_matches_reference(dicom_dir, dicom_reference_mat):
    m = matfile.load(dicom_reference_mat)
    ref_ct = validate_ct(m["ct"])
    ref_cst = validate_cst(m["cst"], ref_ct)

    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    assert isinstance(cst, StructureSet)

    sets = _voxel_sets(cst)
    ref_sets = _voxel_sets(ref_cst)
    assert set(sets) == set(ref_sets)
    for name, voxels in ref_sets.items():
        # The RTSTRUCT contour->mask conversion must reproduce the reference voxels exactly.
        assert voxels == sets[name], f"VOI '{name}' voxels differ from reference"


def test_import_dose_matches_reference(dicom_dir, dicom_reference_mat):
    ref_dose = np.asarray(matfile.load(dicom_reference_mat)["resultGUI"]["physicalDose"])
    ref_zyx = np.transpose(ref_dose, (2, 0, 1))  # matRad (y,x,z) -> sitk (z,y,x)

    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))
    assert isinstance(dose, sitk.Image)
    arr = sitk.GetArrayFromImage(dose)
    assert arr.shape == ref_zyx.shape
    assert np.allclose(arr, ref_zyx, atol=1e-3)


def test_list_ct_series(dicom_dir):
    series = DicomImporter(dicom_dir).list_ct_series()
    assert len(series) >= 1
    entry = series[0]
    assert entry["num_slices"] == len(entry["files"])
    assert entry["series_uid"]


def test_list_structure_sets(dicom_dir):
    sets = DicomImporter(dicom_dir).list_structure_sets()
    assert len(sets) >= 1
    # The RTSTRUCT enumerates its ROI names.
    rtstructs = [s for s in sets if s["modality"] == "RTSTRUCT"]
    assert rtstructs
    names = {n for s in rtstructs for n in s["structure_names"]}
    assert {"Target", "Body", "Core"} <= names


def test_list_doses_and_selective_load(dicom_dir):
    importer = DicomImporter(dicom_dir)
    doses = importer.list_doses()
    # matRad exports per-beam / LET distributions too, so there are several.
    assert len(doses) > 1
    assert all("summation" in d and "description" in d for d in doses)

    # Explicitly loading a chosen file returns exactly that distribution.
    physical = next(d for d in doses if d["path"].endswith("RTDose_6_physicalDose.dcm"))
    dose = importer.load_dose(dose_file=physical["path"])
    assert isinstance(dose, sitk.Image)


def test_load_patient_from_dicom_folder(dicom_dir):
    ct, cst = load_patient(dicom_dir)
    assert isinstance(ct, CT)
    assert isinstance(cst, StructureSet)
    assert {v.name for v in cst.vois} == {"Target", "Body", "Core"}


def test_load_data_from_dicom_folder(dicom_dir, dicom_reference_mat):
    data = load_data(dicom_dir)
    assert isinstance(data["ct"], CT)
    assert isinstance(data["cst"], StructureSet)
    assert isinstance(data["dose"], sitk.Image)

    # The folder holds per-beam doses and LET cubes (all modality RTDOSE);
    # load_data must select the plan-level physical dose, not an arbitrary one.
    ref_dose = np.asarray(matfile.load(dicom_reference_mat)["resultGUI"]["physicalDose"])
    ref_zyx = np.transpose(ref_dose, (2, 0, 1))
    arr = sitk.GetArrayFromImage(data["dose"])
    assert arr.shape == ref_zyx.shape
    assert np.allclose(arr, ref_zyx, atol=1e-3)


def test_dicom_import_reports_progress(dicom_dir):
    """A folder import reports nested, determinate progress for each step."""
    reports = []

    with observe_reports(lambda r: reports.append(r)):
        load_data(dicom_dir)

    stacks = [
        tuple((lvl.name, lvl.total) for lvl in r.levels)
        for r in reports
        if isinstance(r, ProgressReport) and r.levels
    ]
    assert stacks

    # Every step nests under the overall "Importing" level.
    nested = {s for s in stacks if len(s) > 1}
    assert {
        (("Importing", 3), ("Scanning files", 22)),
        (("Importing", 3), ("Reading CT slices", 10)),
        (("Importing", 3), ("Structure", 3)),
        (("Importing", 3), ("Reading dose cube", 1)),
    } <= nested

    # Every level is determinate, so a consumer can drive a real progress bar.
    assert all(lvl.total for r in reports if isinstance(r, ProgressReport) for lvl in r.levels)

    # The outer level runs to completion.
    outer = [
        r.levels[0]
        for r in reports
        if isinstance(r, ProgressReport) and r.levels and r.levels[0].name == "Importing"
    ]
    assert outer[-1].current == outer[-1].total == 3


def test_dicom_importer_caches_headers(dicom_dir):
    """The folder is scanned once; the listings reuse the cached headers."""
    importer = DicomImporter(dicom_dir)
    assert not importer._headers  # nothing read before the first classification

    series = importer.list_ct_series()
    scanned = dict(importer._headers)
    assert scanned

    with mock.patch("pydicom.dcmread", side_effect=AssertionError("re-read the headers")):
        assert importer.list_ct_series() == series
        importer.list_structure_sets()
        importer.list_doses()


# --------------------------------------------------------------------------
# Export / round-trip
# --------------------------------------------------------------------------


def test_dicom_roundtrip_ct(dicom_dir, tmp_path):
    ct = DicomImporter(dicom_dir).load_ct()
    DicomExporter(tmp_path).save(ct=ct)

    ct2 = DicomImporter(tmp_path).load_ct()
    assert ct2.cube_hu.GetSize() == ct.cube_hu.GetSize()
    assert np.allclose(ct2.cube_hu.GetSpacing(), ct.cube_hu.GetSpacing())
    assert np.allclose(ct2.cube_hu.GetOrigin(), ct.cube_hu.GetOrigin())
    assert np.allclose(
        sitk.GetArrayFromImage(ct2.cube_hu), sitk.GetArrayFromImage(ct.cube_hu), atol=1e-3
    )


def test_dicom_roundtrip_dose(dicom_dir, tmp_path):
    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))
    DicomExporter(tmp_path).save(ct=DicomImporter(dicom_dir).load_ct(), dose=dose)

    dose2 = DicomImporter(tmp_path).load_dose()
    a = sitk.GetArrayFromImage(dose)
    b = sitk.GetArrayFromImage(dose2)
    assert b.shape == a.shape
    # uint32 quantization -> values reproduced to high precision.
    assert np.allclose(a, b, atol=1e-4 * max(a.max(), 1.0))


def test_dicom_roundtrip_structures(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path).save(ct=ct, cst=cst)

    cst2 = DicomImporter(tmp_path).load_cst()
    sets = _voxel_sets(cst)
    sets2 = _voxel_sets(cst2)
    assert set(sets) == set(sets2)
    for name, voxels in sets.items():
        # Contour extraction then re-rasterization is near-lossless (boundary voxels).
        assert _dice(voxels, sets2[name]) > 0.95, f"VOI '{name}' degraded in roundtrip"


def test_dicom_rtstruct_references_ct(dicom_dir, tmp_path):
    # The RTSTRUCT must reference the exported CT series and slices so external
    # viewers can associate structures with the CT (not just via frame of reference).
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path).save(ct=ct, cst=cst)

    rtstruct = pydicom.dcmread(tmp_path / "RTSTRUCT.dcm")
    ref_series = (
        rtstruct.ReferencedFrameOfReferenceSequence[0]
        .RTReferencedStudySequence[0]
        .RTReferencedSeriesSequence[0]
    )
    ct_slices = [pydicom.dcmread(f, stop_before_pixels=True) for f in tmp_path.glob("CT_*.dcm")]
    ct_series_uids = {s.SeriesInstanceUID for s in ct_slices}
    ct_sop_uids = {s.SOPInstanceUID for s in ct_slices}

    assert ref_series.SeriesInstanceUID in ct_series_uids
    referenced = {img.ReferencedSOPInstanceUID for img in ref_series.ContourImageSequence}
    assert referenced == ct_sop_uids


def test_dicom_seg_is_binary_packed(dicom_dir, tmp_path):
    # SegmentationType BINARY requires 1-bit packed pixel data; verify the file is
    # conformant and that pydicom still unpacks it to the expected frame shape.
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomExporter(tmp_path, structure_format="seg").save(ct=ct, cst=cst)

    seg = pydicom.dcmread(tmp_path / "SEG.dcm")
    assert seg.BitsAllocated == 1
    assert seg.pixel_array.shape == (seg.NumberOfFrames, seg.Rows, seg.Columns)
    assert set(np.unique(seg.pixel_array)) <= {0, 1}


def test_dicom_roundtrip_structures_as_seg(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    # SEG stores the mask directly, so the roundtrip is exact (unlike RTSTRUCT contours).
    DicomExporter(tmp_path, structure_format="seg").save(ct=ct, cst=cst)
    assert (tmp_path / "SEG.dcm").exists()

    cst2 = DicomImporter(tmp_path).load_cst()
    sets = _voxel_sets(cst)
    sets2 = _voxel_sets(cst2)
    assert set(sets) == set(sets2)
    for name, voxels in sets.items():
        assert voxels == sets2[name], f"VOI '{name}' changed in SEG roundtrip"


def test_dicom_exporter_rejects_bad_structure_format(tmp_path):
    with pytest.raises(ValueError):
        DicomExporter(tmp_path, structure_format="bogus")


def test_dicom_full_roundtrip_via_handler(dicom_dir, tmp_path):
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)
    dose = import_dose(str(dicom_dir / "RTDose_6_physicalDose.dcm"))

    handler = DicomHandler(tmp_path)
    handler.save(ct=ct, cst=cst, dose=dose)

    data = DicomImporter(tmp_path).load_data()
    assert isinstance(data["ct"], CT)
    assert isinstance(data["cst"], StructureSet)
    assert isinstance(data["dose"], sitk.Image)


def test_dicom_handler_honors_structure_format(dicom_dir, tmp_path):
    # The handler must forward structure_format to its exporter half (not silently
    # fall back to the rtstruct default from the cooperative __init__ chain).
    importer = DicomImporter(dicom_dir)
    ct = importer.load_ct()
    cst = importer.load_cst(ct)

    DicomHandler(tmp_path, structure_format="seg").save(ct=ct, cst=cst)
    assert (tmp_path / "SEG.dcm").exists()


def test_dicom_exporter_requires_data(tmp_path):
    with pytest.raises(ValueError):
        DicomExporter(tmp_path).save()


# --------------------------------------------------------------------------
# Multiple structure sets in one folder: warn and use the first (no merging)
# --------------------------------------------------------------------------


def test_load_cst_multiple_structure_sets_uses_first(dicom_dir, tmp_path):
    # Copy the CT series and the RTSTRUCT, then add a second RTSTRUCT (fresh UIDs).
    rtstruct = None
    for f in os.listdir(dicom_dir):
        if f.startswith("ct_slice") or f == "RTstruct.dcm":
            shutil.copy(dicom_dir / f, tmp_path / f)
            if f == "RTstruct.dcm":
                rtstruct = tmp_path / f

    ds = pydicom.dcmread(rtstruct)
    ds.SOPInstanceUID = pydicom.uid.generate_uid()
    ds.SeriesInstanceUID = pydicom.uid.generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.save_as(tmp_path / "RTstruct2.dcm", enforce_file_format=True)

    importer = DicomImporter(tmp_path)
    ct = importer.load_ct()
    with pytest.warns(UserWarning, match="Multiple structure sets"):
        cst = importer.load_cst(ct)

    # One structure set is loaded (3 VOIs), not both merged (6).
    assert [v.name for v in cst.vois] == ["Target", "Body", "Core"]

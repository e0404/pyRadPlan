"""Tests for DICOM export metadata handling (DicomInfo, context resolution, IOD conformance)."""

import logging
import os
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from pyRadPlan.core import Grid
from pyRadPlan.ct import CT, DicomInfo, resample_ct, validate_ct
from pyRadPlan.io import load_data, load_patient, save_data
from pyRadPlan.io._helpers import roi_interpreted_type
from pyRadPlan.io.dicom import DicomExporter, DicomImporter, find_dicom_files
from pyRadPlan.io.dicom._export_common import resolve_export_context
from pyRadPlan.io.dicom._import_dose import import_dose


@pytest.fixture(scope="module")
def dicom_patient():
    return load_patient(Path(__file__).parent.parent / "data" / "dicom")


@pytest.fixture
def tg119_ct(tg119_path):
    ct, _ = load_patient(tg119_path)
    return ct


@pytest.fixture
def plain_ct():
    img = sitk.GetImageFromArray(np.zeros((4, 8, 8), dtype=np.float32))
    img.SetSpacing((2.0, 2.0, 3.0))
    return validate_ct(cube_hu=img)


def _export(tmp_path, ct=None, cst=None, dose=None, **kwargs):
    out = tmp_path / "out"
    DicomExporter(out, **kwargs).save(ct=ct, cst=cst, dose=dose)
    return out


def _dose_like(ct):
    return sitk.Cast(ct.cube_hu * 0 + 1.0, sitk.sitkFloat64)


def _read(path):
    return pydicom.dcmread(path)


# --------------------------------------------------------------------------
# DicomInfo
# --------------------------------------------------------------------------


def test_dicom_info_from_dataset():
    ds = pydicom.Dataset()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "P1"
    ds.KVP = "120"
    ds.PatientPosition = "HFS"
    ds.ImageType = ["ORIGINAL", "PRIMARY"]
    ds.PixelSpacing = [1.0, 1.0]
    ds.add_new(0x00091001, "LO", "secret")
    info = DicomInfo.model_validate(ds)
    assert info.patient_name == "Doe^Jane"
    assert info.patient_id == "P1"
    assert info.kvp == 120.0
    assert info.additional["ImageType"] == ["ORIGINAL", "PRIMARY"]
    assert "PixelSpacing" not in info.additional
    assert not any("0009" in k or "Private" in k for k in info.additional)
    assert DicomInfo.model_validate_json(info.model_dump_json()) == info


@pytest.mark.parametrize(
    "data",
    [
        {"patient_id": "P1", "kvp": 120},
        {"patientId": "P1", "kvp": 120},
        {"PatientID": "P1", "KVP": 120},
    ],
)
def test_dicom_info_key_styles(data):
    info = DicomInfo.model_validate(data)
    assert info.patient_id == "P1"
    assert info.kvp == 120.0
    assert DicomInfo.model_validate(info) is info


def test_dicom_info_json_round_trip():
    info = DicomInfo(patient_id="P1", sop_instance_uids=["1.2", "1.3"], additional={"A": [1, 2]})
    assert DicomInfo.model_validate_json(info.model_dump_json()) == info
    assert DicomInfo.model_validate(info.model_dump(by_alias=True)) == info


def test_dicom_info_from_matrad(tg119_ct):
    info = tg119_ct.dicom_info
    assert info.patient_id == "000-007"
    assert info.patient_name == "Eclipse^IMRT^Rad"
    assert info.patient_position == "HFS"
    assert info.kvp == 120.0
    assert info.manufacturer == "GE MEDICAL SYSTEMS"
    assert info.study_instance_uid
    assert info.frame_of_reference_uid
    assert info.series_instance_uid
    assert info.position_reference_indicator is None
    banned = {"Filename", "FileSize", "Width", "Height", "PixelSpacing", "RescaleSlope"}
    assert not banned & set(info.additional)
    assert not any(k.startswith(("Private_", "MediaStorage")) for k in info.additional)
    assert DicomInfo.model_validate_json(info.model_dump_json()) == info


def test_dicom_info_from_matrad_helper_prefers_meta():
    info = DicomInfo.from_matrad({"PatientID": "A", "Manufacturer": "M"}, {"PatientID": "B"})
    assert info.patient_id == "B"
    assert info.manufacturer == "M"


# --------------------------------------------------------------------------
# CT integration
# --------------------------------------------------------------------------


def test_ct_to_matrad_dicom_info(tg119_ct, plain_ct):
    mat = tg119_ct.to_matrad()
    assert mat["dicomInfo"]["PatientID"] == "000-007"
    assert "dicomMeta" not in mat
    assert "dicomInfo" not in plain_ct.to_matrad()


def test_ct_mat_round_trip_keeps_dicom_info(tg119_ct, tmp_path):
    target = tmp_path / "ct.mat"
    save_data(file_name=target, ct=tg119_ct)
    loaded, _ = load_patient(target)
    assert loaded.dicom_info is not None
    assert loaded.dicom_info.patient_id == tg119_ct.dicom_info.patient_id
    assert loaded.dicom_info.study_instance_uid == tg119_ct.dicom_info.study_instance_uid
    assert loaded.dicom_info.kvp == tg119_ct.dicom_info.kvp


def test_ct_resample_keeps_dicom_info(tg119_ct):
    grid = Grid.from_sitk_image(tg119_ct.cube_hu)
    assert tg119_ct.resample_to_grid(grid).dicom_info == tg119_ct.dicom_info
    resampled = resample_ct(tg119_ct, target_grid_spacing=(5.0, 5.0, 5.0))
    assert resampled.dicom_info == tg119_ct.dicom_info


def test_ct_equality_with_dicom_info(tg119_ct):
    other = tg119_ct.model_copy(deep=True)
    assert other == tg119_ct
    other.dicom_info = DicomInfo(patient_id="other")
    assert other != tg119_ct


# --------------------------------------------------------------------------
# DICOM import / export round trip
# --------------------------------------------------------------------------


def test_dicom_import_fills_dicom_info(dicom_patient):
    ct, _ = dicom_patient
    info = ct.dicom_info
    assert info is not None
    assert info.patient_id
    assert info.study_instance_uid
    assert info.frame_of_reference_uid
    assert len(info.sop_instance_uids) == ct.size[2]


def test_round_trip_keeps_identity_new_series(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    out = _export(tmp_path, ct, cst, _dose_like(ct))

    src, new = ct.dicom_info, _read(out / "CT_0001.dcm")
    assert new.PatientID == src.patient_id
    assert str(new.PatientName) == (src.patient_name or "pyRadPlan^Export")
    assert new.StudyInstanceUID == src.study_instance_uid
    assert new.FrameOfReferenceUID == src.frame_of_reference_uid
    assert new.SeriesInstanceUID != src.series_instance_uid
    assert new.SOPInstanceUID not in src.sop_instance_uids

    reimported = DicomImporter(out).load_ct()
    assert reimported.dicom_info.patient_id == src.patient_id
    assert reimported.dicom_info.study_instance_uid == src.study_instance_uid
    assert reimported.dicom_info.frame_of_reference_uid == src.frame_of_reference_uid
    assert reimported.dicom_info.series_instance_uid != src.series_instance_uid

    for name in ("RTSTRUCT.dcm", "RTDOSE.dcm"):
        ds = _read(out / name)
        assert ds.StudyInstanceUID == new.StudyInstanceUID
        assert ds.FrameOfReferenceUID == new.FrameOfReferenceUID
        assert ds.SeriesInstanceUID != new.SeriesInstanceUID
    ref = _read(out / "RTSTRUCT.dcm").ReferencedFrameOfReferenceSequence[0]
    assert ref.RTReferencedStudySequence[0].ReferencedSOPInstanceUID == src.study_instance_uid


def test_fallback_defaults_without_dicom_info(plain_ct, tmp_path):
    ds = _read(_export(tmp_path, plain_ct) / "CT_0001.dcm")
    assert str(ds.PatientName) == "pyRadPlan^Export"
    assert ds.PatientID == "pyRadPlan"
    assert ds.PatientPosition == "HFS"
    assert ds.StudyID == "1"
    assert not ds.PositionReferenceIndicator
    assert not ds.StudyDate
    assert ds.SeriesDate == ds.InstanceCreationDate == ds.ContentDate
    assert ds.SeriesTime == ds.InstanceCreationTime == ds.ContentTime


def test_metadata_overrides_ct_info(dicom_patient, tmp_path):
    ct, _ = dicom_patient
    out = _export(tmp_path, ct, metadata={"PatientID": "OVERRIDE", "patient_position": "FFS"})
    ds = _read(out / "CT_0001.dcm")
    assert ds.PatientID == "OVERRIDE"
    assert ds.PatientPosition == "FFS"
    assert str(ds.PatientName) == (ct.dicom_info.patient_name or "pyRadPlan^Export")
    assert ds.StudyInstanceUID == ct.dicom_info.study_instance_uid


def test_metadata_accepts_dicom_info(plain_ct, tmp_path):
    out = _export(tmp_path, plain_ct, metadata=DicomInfo(patient_id="X", kvp=100))
    ds = _read(out / "CT_0001.dcm")
    assert ds.PatientID == "X"
    assert float(ds.KVP) == 100.0


def test_anonymize(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    ct = ct.model_copy(deep=True)
    ct.dicom_info = DicomInfo.model_validate(
        {**ct.dicom_info.model_dump(), "patient_birth_date": "19700101", "patient_sex": "F"}
    )
    out = _export(tmp_path, ct, cst, _dose_like(ct), anonymize=True)
    src = ct.dicom_info
    uids = set()
    for name in ("CT_0001.dcm", "RTSTRUCT.dcm", "RTDOSE.dcm"):
        ds = _read(out / name)
        assert str(ds.PatientName) == "Anonymous^Patient"
        assert ds.PatientID == "ANON"
        assert not ds.PatientBirthDate
        assert not ds.PatientSex
        assert not ds.AccessionNumber
        assert ds.StudyInstanceUID != src.study_instance_uid
        assert ds.FrameOfReferenceUID != src.frame_of_reference_uid
        uids.add((ds.StudyInstanceUID, ds.FrameOfReferenceUID))
    assert len(uids) == 1
    ref = _read(out / "RTSTRUCT.dcm").ReferencedFrameOfReferenceSequence[0]
    assert ref.RTReferencedStudySequence[0].ReferencedSOPInstanceUID == next(iter(uids))[0]
    reimported = DicomImporter(out).load_ct()
    assert reimported.size == ct.size
    assert np.allclose(reimported.origin, ct.origin)


def test_mismatch_warning(dicom_patient, tg119_ct, caplog):
    _, cst = dicom_patient
    with caplog.at_level(logging.WARNING):
        ctx = resolve_export_context(tg119_ct, cst)
    assert "differ" in caplog.text
    assert ctx.patient_id == tg119_ct.dicom_info.patient_id


def test_save_data_exporter_options(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    target = tmp_path / "viasave"
    save_data(
        file_name=target,
        format="dcm",
        ct=ct,
        cst=cst,
        exporter_options={"structure_format": "seg", "anonymize": True},
    )
    assert (target / "SEG.dcm").exists()
    assert not (target / "RTSTRUCT.dcm").exists()
    assert _read(target / "CT_0001.dcm").PatientID == "ANON"


# --------------------------------------------------------------------------
# Overwrite policy
# --------------------------------------------------------------------------


def _dicom_clutter(directory):
    directory.mkdir(parents=True, exist_ok=True)
    ds = pydicom.dcmread(Path(__file__).parent.parent / "data" / "dicom" / "RTDose_1_LET.dcm")
    ds.save_as(directory / "named_without_extension")
    (directory / "upper.DCM").write_bytes(b"not really dicom")
    (directory / "note.txt").write_text("keep")
    (directory / "sub").mkdir()
    (directory / "sub" / "nested.dcm").write_bytes(b"keep")


def test_find_dicom_files(tmp_path):
    assert find_dicom_files(tmp_path / "missing") == []
    _dicom_clutter(tmp_path)
    found = sorted(Path(p).name for p in find_dicom_files(tmp_path))
    assert found == ["named_without_extension", "upper.DCM"]


def test_export_into_non_empty_directory_raises_without_writing(plain_ct, tmp_path):
    _dicom_clutter(tmp_path)
    before = sorted(p.name for p in tmp_path.iterdir())
    with pytest.raises(FileExistsError, match="2 DICOM file.*overwrite=True"):
        DicomExporter(tmp_path).save(ct=plain_ct)
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_overwrite_deletes_only_dicom_files(plain_ct, tmp_path):
    _dicom_clutter(tmp_path)
    DicomExporter(tmp_path, overwrite=True).save(ct=plain_ct)
    names = {p.name for p in tmp_path.iterdir()}
    assert names == {"CT_0001.dcm", "CT_0002.dcm", "CT_0003.dcm", "CT_0004.dcm", "note.txt", "sub"}
    assert (tmp_path / "sub" / "nested.dcm").exists()


def test_invalid_input_does_not_delete_with_overwrite(plain_ct, tmp_path):
    _dicom_clutter(tmp_path)
    with pytest.raises(ValueError):
        DicomExporter(tmp_path, overwrite=True).save()
    assert (tmp_path / "upper.DCM").exists()


def test_overwrite_reachable_via_save_data(plain_ct, tmp_path):
    save_data(file_name=tmp_path, format="dcm", ct=plain_ct)
    with pytest.raises(FileExistsError):
        save_data(file_name=tmp_path, format="dcm", ct=plain_ct)
    save_data(file_name=tmp_path, format="dcm", ct=plain_ct, exporter_options={"overwrite": True})


def test_save_data_dicom_requires_file_name(plain_ct, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="file_name"):
        save_data(format="dcm", ct=plain_ct)
    assert list(tmp_path.iterdir()) == []


def test_overwrite_leaves_no_stale_slices(dicom_patient, tmp_path):
    ct, _ = dicom_patient
    out = tmp_path / "out"
    DicomExporter(out).save(ct=ct)
    n = len(list(out.glob("CT_*.dcm")))
    assert n == ct.size[2]
    fewer = validate_ct(cube_hu=ct.cube_hu[:, :, : n - 3])
    DicomExporter(out, overwrite=True).save(ct=fewer)
    assert len(list(out.glob("CT_*.dcm"))) == n - 3


# --------------------------------------------------------------------------
# RT ROI Interpreted Type
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "voi_type", "expected"),
    [
        ("Body", "EXTERNAL", "EXTERNAL"),
        ("GTV", "TARGET", "GTV"),
        ("gtv_tumor", "TARGET", "GTV"),
        ("CTV1", "TARGET", "CTV"),
        ("ITV_lung", "TARGET", "CTV"),
        ("PTV_70", "TARGET", "PTV"),
        ("ptv-boost", "TARGET", "PTV"),
        ("Boost", "TARGET", "PTV"),
        ("Target", "TARGET", "PTV"),
        ("Pitvitary", "TARGET", "PTV"),
        ("Lung", "OAR", "ORGAN"),
        ("Avoid_Heart", "OAR", "AVOIDANCE"),
        ("PTV_ring", "HELPER", "CONTROL"),
    ],
)
def test_roi_interpreted_type_mapping(name, voi_type, expected):
    assert roi_interpreted_type(name, voi_type) == expected


def _two_voi_cst(dicom_patient):
    _, cst = dicom_patient
    return cst


def test_roi_interpreted_types_override(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    first = cst.vois[0].name
    out = _export(tmp_path, ct, cst, roi_interpreted_types={first: "avoidance"})
    obs = _read(out / "RTSTRUCT.dcm").RTROIObservationsSequence
    assert obs[0].RTROIInterpretedType == "AVOIDANCE"
    assert {o.RTROIInterpretedType for o in obs[1:]} != {"AVOIDANCE"}


def test_roi_interpreted_types_invalid_override(tmp_path):
    with pytest.raises(ValueError, match="BOGUS"):
        DicomExporter(tmp_path / "x", roi_interpreted_types={"Lung": "BOGUS"})


def test_round_trip_keeps_voi_type(dicom_patient, tmp_path):
    from pyRadPlan.cst import StructureSet, create_voi

    ct, _ = dicom_patient
    vois = []
    for i, (name, voi_type) in enumerate(
        [("Body", "EXTERNAL"), ("Odd", "TARGET"), ("Target_x", "OAR"), ("Ring", "HELPER")]
    ):
        arr = np.zeros(ct.cube_hu.GetSize()[::-1], dtype=np.uint8)
        arr[2:8, 2 + 4 * i : 6 + 4 * i, 2:12] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(ct.cube_hu)
        vois.append(create_voi(name=name, voi_type=voi_type, mask=mask, ct_image=ct))
    cst = StructureSet(vois=vois, ct_image=ct)
    out = _export(tmp_path, ct, cst)
    imported = DicomImporter(out).load_cst(ct)
    assert {v.name: v.voi_type for v in imported.vois} == {
        "Body": "EXTERNAL",
        "Odd": "TARGET",
        "Target_x": "OAR",
        "Ring": "HELPER",
    }


@pytest.mark.parametrize("legacy_type", ["ORGAN", "AVOIDANCE", "OAR"])
def test_body_marked_as_organ_imports_as_external(dicom_patient, tmp_path, legacy_type):
    from pyRadPlan.cst import StructureSet, create_voi

    ct, _ = dicom_patient
    vois = []
    for i, name in enumerate(["Body", "Lung"]):
        arr = np.zeros(ct.cube_hu.GetSize()[::-1], dtype=np.uint8)
        arr[2:8, 2 + 4 * i : 6 + 4 * i, 2:12] = 1
        mask = sitk.GetImageFromArray(arr)
        mask.CopyInformation(ct.cube_hu)
        vois.append(create_voi(name=name, voi_type="OAR", mask=mask, ct_image=ct))
    cst = StructureSet(vois=vois, ct_image=ct)
    out = _export(
        tmp_path, ct, cst, roi_interpreted_types={"Body": legacy_type, "Lung": legacy_type}
    )
    imported = DicomImporter(out).load_cst(ct)
    assert {v.name: v.voi_type for v in imported.vois} == {"Body": "EXTERNAL", "Lung": "OAR"}


def test_bundled_rtstruct_voi_types(dicom_dir):
    cst = DicomImporter(dicom_dir).load_cst()
    assert {v.name: v.voi_type for v in cst.vois} == {
        "Body": "EXTERNAL",
        "Core": "OAR",
        "Target": "TARGET",
    }


# --------------------------------------------------------------------------
# Required attributes per modality
# --------------------------------------------------------------------------

_COMMON = [
    "SpecificCharacterSet",
    "InstanceCreationDate",
    "InstanceCreationTime",
    "SOPClassUID",
    "SOPInstanceUID",
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientSex",
    "StudyInstanceUID",
    "StudyDate",
    "StudyTime",
    "ReferringPhysicianName",
    "StudyID",
    "AccessionNumber",
    "Modality",
    "SeriesInstanceUID",
    "SeriesNumber",
    "SeriesDate",
    "SeriesTime",
    "FrameOfReferenceUID",
    "PositionReferenceIndicator",
    "Manufacturer",
    "SoftwareVersions",
]
_REQUIRED = {
    "CT_0001.dcm": [
        "PatientPosition",
        "ImageType",
        "KVP",
        "AcquisitionNumber",
        "InstanceNumber",
        "ImagePositionPatient",
        "ImageOrientationPatient",
        "PixelSpacing",
        "SliceThickness",
        "RescaleIntercept",
        "RescaleSlope",
        "Rows",
        "Columns",
        "BitsAllocated",
        "PixelRepresentation",
    ],
    "RTSTRUCT.dcm": [
        "StructureSetLabel",
        "StructureSetDate",
        "StructureSetTime",
        "StructureSetROISequence",
        "ROIContourSequence",
        "RTROIObservationsSequence",
        "ReferencedFrameOfReferenceSequence",
        "OperatorsName",
    ],
    "RTDOSE.dcm": [
        "OperatorsName",
        "DoseUnits",
        "DoseType",
        "DoseSummationType",
        "DoseGridScaling",
        "GridFrameOffsetVector",
        "NumberOfFrames",
        "ImagePositionPatient",
    ],
    "SEG.dcm": [
        "DeviceSerialNumber",
        "ManufacturerModelName",
        "ContentQualification",
        "ContentLabel",
        "ContentDate",
        "ContentTime",
        "SegmentationType",
        "SegmentSequence",
        "DimensionOrganizationSequence",
        "DimensionIndexSequence",
        "SharedFunctionalGroupsSequence",
        "PerFrameFunctionalGroupsSequence",
    ],
}


@pytest.fixture(scope="module")
def exports(dicom_patient, tmp_path_factory):
    ct, cst = dicom_patient
    base = tmp_path_factory.mktemp("exports")
    out = {}
    for fmt in ("rtstruct", "seg"):
        target = base / fmt
        DicomExporter(target, structure_format=fmt).save(ct=ct, cst=cst, dose=_dose_like(ct))
        out[fmt] = target
    return out


@pytest.mark.parametrize("name", list(_REQUIRED))
def test_required_tags_present(exports, name):
    ds = _read(exports["seg" if name == "SEG.dcm" else "rtstruct"] / name)
    missing = [kw for kw in _COMMON + _REQUIRED[name] if kw not in ds]
    assert not missing
    assert ds.SpecificCharacterSet == "ISO_IR 192"
    assert ds.Manufacturer == "pyRadPlan"
    assert "research use only" in ds.SeriesDescription


# --------------------------------------------------------------------------
# dicom-validator
# --------------------------------------------------------------------------

_FUNCTIONAL_GROUPS = (0x52009229, 0x52009230)


@pytest.fixture(scope="module")
def dicom_standard():
    pytest.importorskip("dicom_validator")
    from dicom_validator.spec_reader.edition_reader import EditionReader  # noqa: PLC0415

    cache = Path(
        os.environ.get(
            "PYRADPLAN_DICOM_STANDARD_DIR", Path.home() / ".cache" / "pyradplan-dicom-standard"
        )
    )
    try:
        cache.mkdir(parents=True, exist_ok=True)
        reader = EditionReader(str(cache))
        edition = reader.get_edition("local") or reader.get_edition("current")
        if edition is None or reader.get_edition_path(edition) is None:
            raise RuntimeError("DICOM standard unavailable")
        return reader.load_dicom_info(edition)
    except Exception as exc:  # noqa: BLE001 - any failure (e.g. offline) must skip
        pytest.skip(f"DICOM standard not available: {exc}")


def _validation_errors(info, path, plan_reference_expected=True):
    from dicom_validator.validator.dicom_file_validator import (  # noqa: PLC0415
        DicomFileValidator,
    )
    from dicom_validator.validator.error_handler import (  # noqa: PLC0415
        NullValidationResultHandler,
    )
    from dicom_validator.validator.validation_result import ErrorCode  # noqa: PLC0415

    validator = DicomFileValidator(info, error_handler=NullValidationResultHandler())
    errors = []
    for result in validator.validate(path).values():
        for module, tag_errors in result.module_errors.items():
            for tag, err in tag_errors.items():
                if not err.is_error():
                    continue
                # RTDOSE needs a referenced RTPLAN, which is not exported yet.
                if plan_reference_expected and tag.tag == 0x300C0002:
                    continue
                # The functional group macros of the SEG IOD are missing from the validator's
                # parsed specification, so valid groups are reported as unexpected.
                in_group = tag.parents and tag.parents[0] in _FUNCTIONAL_GROUPS
                if in_group and err.code == ErrorCode.TagUnexpected:
                    continue
                errors.append((os.path.basename(path), module, str(tag), err.code.name))
    return errors


@pytest.mark.parametrize(
    "fmt,name",
    [
        ("rtstruct", "CT_0001.dcm"),
        ("rtstruct", "CT_0010.dcm"),
        ("rtstruct", "RTSTRUCT.dcm"),
        ("rtstruct", "RTDOSE.dcm"),
        ("seg", "SEG.dcm"),
    ],
)
def test_exports_validate(dicom_standard, exports, fmt, name):
    assert _validation_errors(dicom_standard, exports[fmt] / name) == []


# --------------------------------------------------------------------------
# RTDOSE from results
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def photon_data():
    import pymatreader

    from pyRadPlan.cst import validate_cst
    from pyRadPlan.plan import validate_pln
    from pyRadPlan.stf import validate_stf

    raw = pymatreader.read_mat(str(Path(__file__).parent.parent / "data" / "photons_testData.mat"))
    pln = validate_pln(raw["pln"])
    pln.prop_dose_calc["dosimetric_lateral_cutoff"] = 0.995
    pln.prop_dose_calc["lateral_model"] = "single"
    ct = validate_ct(raw["ct"])
    cst = validate_cst(raw["cst"], ct=ct)
    stf = validate_stf(raw["stf"])
    return pln, ct, cst, stf


@pytest.fixture(scope="module")
def photon_result(photon_data):
    from pyRadPlan.dose import calc_dose_forward

    pln, ct, cst, stf = photon_data
    w = np.ones(stf.total_number_of_bixels)
    return calc_dose_forward(ct, cst, stf, pln, w)


def _fake_result():
    rng = np.random.default_rng(0)

    def img(scale=1.0):
        image = sitk.GetImageFromArray((rng.random((3, 4, 5)) * scale).astype(np.float32))
        image.SetSpacing((2.0, 2.0, 3.0))
        image.SetOrigin((-10.0, -8.0, 4.0))
        return image

    return {
        "physical_dose": img(),
        "physical_dose_beam": [img(), img()],
        "rbe_x_dose": img(1.1),
        "rbe_x_dose_beam": [img(1.1), img(1.1)],
        "let": img(5),
        "effect": img(),
        "alpha_dose": img(),
        "w": np.ones(3),
    }


class _Pln:
    def __init__(self, dose_convention):
        self.dose_convention = dose_convention


def test_result_export_dose_types_and_skips(plain_ct, tmp_path):
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="alpha_dose, effect, let") as rec:
        DicomExporter(out).save(ct=plain_ct, result=_fake_result())
    assert len([w for w in rec if "skips non-dose" in str(w.message)]) == 1
    names = sorted(p.name for p in out.glob("RTDOSE*.dcm"))
    assert names == ["RTDOSE_physical_dose.dcm", "RTDOSE_rbe_x_dose.dcm"]
    phys, rbe = (_read(out / n) for n in names)
    assert (phys.DoseType, rbe.DoseType) == ("PHYSICAL", "EFFECTIVE")
    assert phys.DoseSummationType == rbe.DoseSummationType == "PLAN"
    assert "physical_dose" in phys.DoseComment and "rbe_x_dose" in rbe.DoseComment
    assert phys.SeriesDescription == phys.DoseComment
    assert phys.SeriesInstanceUID != rbe.SeriesInstanceUID
    assert "ReferencedRTPlanSequence" not in phys


def test_result_export_descriptors_are_not_let(plain_ct, tmp_path):
    from pyRadPlan.io.dicom._import_dose import _is_let_descriptor

    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        DicomExporter(out, per_beam_doses=True).save(ct=plain_ct, result=_fake_result())
    for path in out.glob("RTDOSE*.dcm"):
        assert not _is_let_descriptor(_read(path).DoseComment)


def test_result_export_values_round_trip(tmp_path):
    result = {"physical_dose": _fake_result()["physical_dose"]}
    out = tmp_path / "out"
    DicomExporter(out).save(result=result)
    back = import_dose(out / "RTDOSE_physical_dose.dcm")
    ref = sitk.GetArrayFromImage(result["physical_dose"])
    np.testing.assert_allclose(sitk.GetArrayFromImage(back), ref, rtol=1e-6, atol=1e-6)


def test_result_without_dose_raises(tmp_path):
    with pytest.warns(UserWarning, match="skips non-dose"):
        with pytest.raises(ValueError, match="no dose quantity"):
            DicomExporter(tmp_path / "out").save(result={"let": _fake_result()["let"]})


def test_result_scenarios_warn_and_export_nominal(plain_ct, tmp_path):
    result = _fake_result()
    result["physical_dose_scenario"] = [result["physical_dose"], result["physical_dose"]]
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="nominal scenario only"):
        DicomExporter(out).save(ct=plain_ct, result=result)
    assert sorted(p.name for p in out.glob("RTDOSE*.dcm")) == [
        "RTDOSE_physical_dose.dcm",
        "RTDOSE_rbe_x_dose.dcm",
    ]


def test_single_entry_scenario_list_does_not_warn(plain_ct, tmp_path, recwarn):
    result = {"physical_dose": _fake_result()["physical_dose"]}
    result["physical_dose_scenario"] = [result["physical_dose"]]
    DicomExporter(tmp_path / "out").save(ct=plain_ct, result=result)
    assert not [w for w in recwarn if "scenario" in str(w.message)]


def test_per_beam_doses(plain_ct, tmp_path):
    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        DicomExporter(out, per_beam_doses=True).save(ct=plain_ct, result=_fake_result())
    for q, dose_type in (("physical_dose", "PHYSICAL"), ("rbe_x_dose", "EFFECTIVE")):
        for n in (1, 2):
            ds = _read(out / f"RTDOSE_{q}_beam{n}.dcm")
            assert ds.DoseSummationType == "BEAM" and ds.DoseType == dose_type
            assert f"beam {n}" in ds.DoseComment
    assert not (out / "RTDOSE_physical_dose_beam3.dcm").exists()

    off = tmp_path / "off"
    with pytest.warns(UserWarning):
        DicomExporter(off).save(ct=plain_ct, result=_fake_result())
    assert not list(off.glob("*beam*"))


def test_per_beam_without_beam_doses_warns(tmp_path):
    with pytest.warns(UserWarning, match="no per-beam doses"):
        DicomExporter(tmp_path / "out", per_beam_doses=True).save(
            result={"physical_dose": _fake_result()["physical_dose"]}
        )


@pytest.mark.parametrize(
    "convention,expected", [("per_fraction", "FRACTION"), ("total", "PLAN"), (None, "PLAN")]
)
def test_summation_follows_dose_convention(tmp_path, convention, expected):
    dose = _fake_result()["physical_dose"]
    pln = None if convention is None else _Pln(convention)
    out = tmp_path / "out"
    DicomExporter(out).save(result={"physical_dose": dose}, pln=pln)
    assert _read(out / "RTDOSE_physical_dose.dcm").DoseSummationType == expected
    DicomExporter(tmp_path / "out2").save(dose=dose, pln=pln)
    assert _read(tmp_path / "out2" / "RTDOSE.dcm").DoseSummationType == expected


def test_dose_image_path_unchanged(plain_ct, tmp_path):
    out = _export(tmp_path, ct=plain_ct, dose=_dose_like(plain_ct))
    ds = _read(out / "RTDOSE.dcm")
    assert ds.DoseType == "PHYSICAL" and ds.DoseSummationType == "PLAN"
    assert "DoseComment" not in ds and "ReferencedRTPlanSequence" not in ds
    assert [float(o) for o in ds.GridFrameOffsetVector] == [0.0, 3.0, 6.0, 9.0]


def test_save_data_result_and_pln(photon_data, photon_result, tmp_path, caplog):
    pln, ct, _cst, _stf = photon_data
    out = tmp_path / "dcm"
    with caplog.at_level(logging.WARNING):
        save_data(
            file_name=out,
            format="dcm",
            ct=ct,
            result=photon_result,
            pln=pln,
            exporter_options={"per_beam_doses": True},
        )
    assert "ignores unsupported" not in caplog.text
    expected = "FRACTION" if pln.dose_convention == "per_fraction" else "PLAN"
    ds = _read(out / "RTDOSE_physical_dose.dcm")
    assert ds.DoseSummationType == expected
    assert ds.FrameOfReferenceUID == _read(next(out.glob("CT_*.dcm"))).FrameOfReferenceUID
    n_beams = len(photon_result["physical_dose_beam"])
    assert len(list(out.glob("RTDOSE_physical_dose_beam*.dcm"))) == n_beams
    back = DicomImporter(out).load_dose()
    ref = sitk.GetArrayFromImage(photon_result["physical_dose"])
    np.testing.assert_allclose(sitk.GetArrayFromImage(back), ref, atol=ref.max() * 1e-6)


def test_other_exporters_unaffected_by_result_and_pln(photon_data, tmp_path):
    pln, _ct, _cst, _stf = photon_data
    path = save_data(file_name=tmp_path / "res.mat", result={"physical_dose": 1.0}, pln=pln)
    assert Path(path).exists()


# -- plan reference ---------------------------------------------------------


def _plan_ref(**kwargs):
    from pyRadPlan.io.dicom._export_common import PlanReference

    return PlanReference(sop_instance_uid=pydicom.uid.generate_uid(), **kwargs)


def _export_with_plan_ref(tmp_path, plan_ref, pln=None):
    out = tmp_path / "out"
    exporter = DicomExporter(out, per_beam_doses=True)
    exporter._plan_ref = plan_ref
    with pytest.warns(UserWarning):
        exporter.save(result=_fake_result(), pln=pln)
    return out


@pytest.mark.parametrize(
    "convention,name,summation",
    [
        ("total", "RTDOSE_physical_dose.dcm", "PLAN"),
        ("per_fraction", "RTDOSE_rbe_x_dose.dcm", "FRACTION"),
        ("total", "RTDOSE_physical_dose_beam2.dcm", "BEAM"),
    ],
)
def test_plan_reference_validates(dicom_standard, tmp_path, convention, name, summation):
    ref = _plan_ref(fraction_group_number=1, beam_numbers=[10, 20])
    out = _export_with_plan_ref(tmp_path, ref, _Pln(convention))
    ds = _read(out / name)
    assert ds.DoseSummationType == summation
    item = ds.ReferencedRTPlanSequence[0]
    assert item.ReferencedSOPClassUID == "1.2.840.10008.5.1.4.1.1.481.8"
    assert item.ReferencedSOPInstanceUID == ref.sop_instance_uid
    if summation == "PLAN":
        assert "ReferencedFractionGroupSequence" not in item
    else:
        group = item.ReferencedFractionGroupSequence[0]
        assert group.ReferencedFractionGroupNumber == 1
        if summation == "BEAM":
            assert group.ReferencedBeamSequence[0].ReferencedBeamNumber == 20
    assert _validation_errors(dicom_standard, out / name, plan_reference_expected=False) == []


def test_missing_plan_reference_is_the_only_error(dicom_standard, plain_ct, tmp_path):
    out = tmp_path / "out"
    with pytest.warns(UserWarning):
        DicomExporter(out, per_beam_doses=True).save(
            ct=plain_ct, result=_fake_result(), pln=_Pln("per_fraction")
        )
    for path in sorted(out.glob("RTDOSE*.dcm")):
        assert _validation_errors(dicom_standard, path) == []
        errors = _validation_errors(dicom_standard, path, plan_reference_expected=False)
        assert errors and {e[2] for e in errors} == {"(300C,0002)"}


def test_plan_reference_photon_class(tmp_path):
    from pyRadPlan.io.dicom._export_common import RT_PLAN_STORAGE

    out = _export_with_plan_ref(tmp_path, _plan_ref(sop_class_uid=RT_PLAN_STORAGE))
    ds = _read(out / "RTDOSE_physical_dose.dcm")
    assert ds.ReferencedRTPlanSequence[0].ReferencedSOPClassUID == RT_PLAN_STORAGE
    beam = _read(out / "RTDOSE_physical_dose_beam1.dcm")
    group = beam.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0]
    assert group.ReferencedBeamSequence[0].ReferencedBeamNumber == 1


# -- non-axial grids --------------------------------------------------------


def _oriented_dose(direction, origin=(5.0, -3.0, 12.0)):
    rng = np.random.default_rng(1)
    image = sitk.GetImageFromArray(rng.random((4, 5, 6)) + 0.1)
    image.SetSpacing((1.5, 2.0, 2.5))
    image.SetDirection(direction)
    image.SetOrigin(origin)
    return image


def _rotation_z(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return (c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0)


def _rotation_x(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return (1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)


_FLIP_Z = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
_DIRECTIONS = {"flip_z": _FLIP_Z, "oblique_z": _rotation_z(20), "oblique_x": _rotation_x(15)}


def _voxel_position(ds, i, j, k):
    ipp = np.array(ds.ImagePositionPatient, dtype=float)
    iop = np.array(ds.ImageOrientationPatient, dtype=float)
    row_dir, col_dir = iop[:3], iop[3:]  # along columns (x) and along rows (y)
    normal = np.cross(row_dir, col_dir)
    dy, dx = (float(v) for v in ds.PixelSpacing)
    offset = float(ds.GridFrameOffsetVector[k])
    return ipp + row_dir * dx * i + col_dir * dy * j + normal * offset


@pytest.mark.parametrize("direction", _DIRECTIONS.values(), ids=_DIRECTIONS.keys())
def test_non_axial_dose_geometry(tmp_path, direction):
    dose = _oriented_dose(direction)
    out = tmp_path / "out"
    DicomExporter(out).save(dose=dose)
    ds = _read(out / "RTDOSE.dcm")
    nx, ny, nz = dose.GetSize()
    assert len(ds.GridFrameOffsetVector) == nz
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                np.testing.assert_allclose(
                    _voxel_position(ds, i, j, k),
                    dose.TransformIndexToPhysicalPoint((i, j, k)),
                    atol=1e-6,
                )


def test_flipped_dose_reimport(tmp_path):
    dose = _oriented_dose(_FLIP_Z)
    out = tmp_path / "out"
    DicomExporter(out).save(dose=dose)
    back = import_dose(out / "RTDOSE.dcm")
    assert back.GetSize() == dose.GetSize()
    np.testing.assert_allclose(
        back.TransformIndexToPhysicalPoint((2, 1, 3)),
        dose.TransformIndexToPhysicalPoint((2, 1, 3)),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(back), sitk.GetArrayFromImage(dose), rtol=1e-6
    )


@pytest.mark.parametrize("name", ["flip_z", "oblique_x"])
def test_non_axial_dose_validates(dicom_standard, tmp_path, name):
    out = tmp_path / "out"
    DicomExporter(out).save(dose=_oriented_dose(_DIRECTIONS[name]))
    assert _validation_errors(dicom_standard, out / "RTDOSE.dcm") == []


# --------------------------------------------------------------------------
# DVH sequence of the RTDOSE
# --------------------------------------------------------------------------


def _gradient_dose(ct):
    array = np.zeros(sitk.GetArrayFromImage(ct.cube_hu).shape)
    array += np.linspace(0.0, 60.0, array.shape[-1])
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(ct.cube_hu)
    return image


@pytest.fixture(scope="module")
def dvh_export(dicom_patient, tmp_path_factory):
    ct, cst = dicom_patient
    dose = _gradient_dose(ct)
    target = tmp_path_factory.mktemp("dvh_export")
    DicomExporter(target, dvh=True).save(ct=ct, cst=cst, dose=dose)
    return target, cst, dose


def test_dvh_sequence_structure(dvh_export):
    out, cst, _ = dvh_export
    dose = _read(out / "RTDOSE.dcm")
    struct = _read(out / "RTSTRUCT.dcm")

    reference = dose.ReferencedStructureSetSequence[0]
    assert reference.ReferencedSOPClassUID == struct.SOPClassUID
    assert reference.ReferencedSOPInstanceUID == struct.SOPInstanceUID

    assert len(dose.DVHSequence) == len(cst.vois)
    roi_numbers = {r.ROIName: r.ROINumber for r in struct.StructureSetROISequence}
    for voi, item in zip(cst.vois, dose.DVHSequence):
        assert item.DVHType == "CUMULATIVE"
        assert item.DoseUnits == "GY"
        assert item.DoseType == dose.DoseType
        assert item.DVHDoseScaling == 1
        assert item.DVHVolumeUnits == "CM3"
        assert len(item.DVHData) == 2 * item.DVHNumberOfBins
        roi = item.DVHReferencedROISequence[0]
        assert roi.ReferencedROINumber == roi_numbers[voi.name]
        assert roi.DVHROIContributionType == "INCLUDED"
        assert item.DVHMinimumDose <= item.DVHMeanDose <= item.DVHMaximumDose


def test_dvh_sequence_matches_dvh_compute(dvh_export):
    from pyRadPlan.analysis import DVH

    out, cst, dose = dvh_export
    ds = _read(out / "RTDOSE.dcm")
    stored = sitk.GetImageFromArray(ds.pixel_array.astype(np.float64) * float(ds.DoseGridScaling))
    stored.CopyInformation(dose)
    spacing = stored.GetSpacing()
    voxel_volume = spacing[0] * spacing[1] * spacing[2] / 1000.0

    for voi, item in zip(cst.vois, ds.DVHSequence):
        data = np.asarray(item.DVHData, dtype=float).reshape(-1, 2)
        width = data[0, 0]
        assert np.allclose(data[:, 0], width)
        edges = np.arange(item.DVHNumberOfBins + 1) * width
        reference = DVH.compute(
            stored,
            mask=voi.mask,
            num_points=item.DVHNumberOfBins,
            max_value=edges[-1],
        )
        expected = reference.cum_volume / 100.0 * (_voxels(voi) * voxel_volume)
        assert np.allclose(data[:, 1], expected, rtol=1e-6, atol=1e-6 * expected[0])
        assert data[0, 1] == pytest.approx(_voxels(voi) * voxel_volume, rel=1e-6)

        selected = sitk.GetArrayFromImage(stored)[sitk.GetArrayFromImage(voi.mask).astype(bool)]
        assert item.DVHMeanDose == pytest.approx(selected.mean(), rel=1e-6)
        assert item.DVHMaximumDose == pytest.approx(selected.max(), rel=1e-6)
        assert item.DVHMinimumDose == pytest.approx(selected.min(), rel=1e-6, abs=1e-9)


def _voxels(voi):
    return int(sitk.GetArrayFromImage(voi.mask).astype(bool).sum())


def test_dvh_off_by_default(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    out = _export(tmp_path, ct=ct, cst=cst, dose=_dose_like(ct))
    ds = _read(out / "RTDOSE.dcm")
    assert "DVHSequence" not in ds and "ReferencedStructureSetSequence" not in ds


def test_dvh_resamples_masks_to_dose_grid(dicom_patient, tmp_path):
    ct, cst = dicom_patient
    coarse = sitk.Resample(
        _gradient_dose(ct),
        [max(s // 2, 1) for s in ct.cube_hu.GetSize()],
        sitk.Transform(),
        sitk.sitkLinear,
        ct.cube_hu.GetOrigin(),
        [2 * s for s in ct.cube_hu.GetSpacing()],
        ct.cube_hu.GetDirection(),
    )
    DicomExporter(tmp_path / "out", dvh=True).save(ct=ct, cst=cst, dose=coarse)
    ds = _read(tmp_path / "out" / "RTDOSE.dcm")
    assert len(ds.DVHSequence) >= 1
    voxel_volume = np.prod(coarse.GetSpacing()) / 1000.0
    for item in ds.DVHSequence:
        total = float(item.DVHData[1])
        assert total > 0
        assert total / voxel_volume == pytest.approx(round(total / voxel_volume), abs=1e-3)


@pytest.mark.parametrize("kwargs", [{"structure_format": "seg"}, {"no_cst": True}])
def test_dvh_without_rtstruct_warns_and_skips(dicom_patient, tmp_path, kwargs):
    ct, cst = dicom_patient
    no_cst = kwargs.pop("no_cst", False)
    with pytest.warns(UserWarning, match="DVH export needs a structure set"):
        DicomExporter(tmp_path / "out", dvh=True, **kwargs).save(
            ct=ct, cst=None if no_cst else cst, dose=_dose_like(ct)
        )
    ds = _read(tmp_path / "out" / "RTDOSE.dcm")
    assert "DVHSequence" not in ds


def test_dvh_in_result_doses(plain_ct, tmp_path):
    mask = np.zeros((4, 8, 8), dtype=np.uint8)
    mask[1:3, 2:6, 2:6] = 1
    mask = sitk.GetImageFromArray(mask)
    mask.CopyInformation(plain_ct.cube_hu)
    from pyRadPlan.cst import create_cst

    cst = create_cst({"vois": [{"voi_type": "TARGET", "name": "PTV", "mask": mask}]}, ct=plain_ct)
    DicomExporter(tmp_path / "out", dvh=True).save(
        ct=plain_ct, cst=cst, result=_fake_result_for(plain_ct)
    )
    for name in ("RTDOSE_physical_dose.dcm", "RTDOSE_rbe_x_dose.dcm"):
        ds = _read(tmp_path / "out" / name)
        assert len(ds.DVHSequence) == 1
        assert ds.DVHSequence[0].DoseType == ds.DoseType


def _fake_result_for(ct):
    rng = np.random.default_rng(1)

    def img(scale):
        image = sitk.GetImageFromArray((rng.random((4, 8, 8)) * scale).astype(np.float32))
        image.CopyInformation(ct.cube_hu)
        return image

    return {"physical_dose": img(2.0), "rbe_x_dose": img(2.2)}


def test_dvh_rtdose_validates(dicom_standard, dvh_export):
    out, _, _ = dvh_export
    assert _validation_errors(dicom_standard, out / "RTDOSE.dcm") == []


# --------------------------------------------------------------------------
# 4D CT (one series per phase) and overlapping segments
# --------------------------------------------------------------------------

NUM_PHASES = 3


def _ct_4d_phases():
    rng = np.random.default_rng(7)
    phases = []
    for _ in range(NUM_PHASES):
        image = sitk.GetImageFromArray(rng.integers(-1000, 1000, (10, 20, 20)).astype(np.float32))
        phases.append(image)
    cube = sitk.JoinSeries(phases)
    cube.SetSpacing((2.0, 2.5, 3.0, 1.0))
    cube.SetOrigin((-10.0, -12.0, -5.0, 0.0))
    return validate_ct(cube_hu=cube)


def _cst_for(ct, moving=False):
    from pyRadPlan.cst import validate_cst, validate_voi

    boxes = {
        "Body": ((0, 10), (2, 18), (2, 18), "EXTERNAL"),
        "PTV": ((3, 8), (6, 14), (6, 14), "TARGET"),
        "CTV": ((4, 7), (8, 12), (8, 12), "TARGET"),
    }
    vois = []
    for name, (z, y, x, voi_type) in boxes.items():
        mask = np.zeros((10, 20, 20), np.uint8)
        mask[z[0] : z[1], y[0] : y[1], x[0] : x[1]] = 1
        if ct.num_of_ct_scen > 1:
            mask = np.stack([np.roll(mask, p if moving else 0, axis=2) for p in range(NUM_PHASES)])
        vois.append(validate_voi(name=name, voi_type=voi_type, mask=mask, ct_image=ct))
    return validate_cst(vois, ct)


@pytest.fixture(scope="module")
def ct_4d_export(tmp_path_factory):
    ct = _ct_4d_phases()
    out = tmp_path_factory.mktemp("ct4d")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        DicomExporter(out).save(ct=ct)
    return ct, out


def test_4d_ct_writes_one_series_per_phase(ct_4d_export):
    _, out = ct_4d_export
    files = sorted(out.glob("*.dcm"))
    assert len(files) == NUM_PHASES * 10

    by_series = {}
    for path in files:
        ds = _read(path)
        by_series.setdefault(ds.SeriesInstanceUID, []).append(ds)
    assert len(by_series) == NUM_PHASES

    datasets = [ds for series in by_series.values() for ds in series]
    assert len({ds.StudyInstanceUID for ds in datasets}) == 1
    assert len({ds.FrameOfReferenceUID for ds in datasets}) == 1

    phases = []
    for series in by_series.values():
        phase = series[0].TemporalPositionIdentifier
        phases.append(phase)
        assert {ds.TemporalPositionIdentifier for ds in series} == {phase}
        assert {ds.NumberOfTemporalPositions for ds in series} == {NUM_PHASES}
        assert {ds.SeriesDescription for ds in series} == {
            f"pyRadPlan CT phase {phase}/{NUM_PHASES} - research use only"
        }
        assert {ds.SeriesNumber for ds in series} == {phase}
        assert sorted(ds.InstanceNumber for ds in series) == list(range(1, 11))
        ordered = sorted(series, key=lambda d: d.InstanceNumber)
        z = [float(ds.ImagePositionPatient[2]) for ds in ordered]
        assert z == pytest.approx([-5.0 + 3.0 * k for k in range(10)])
    assert sorted(phases) == [1, 2, 3]


def test_4d_ct_imports_as_4d_ct(ct_4d_export):
    ct, out = ct_4d_export
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ct2 = DicomImporter(out).load_ct()
    assert ct2.num_of_ct_scen == NUM_PHASES
    cube, cube2 = ct.cube_hu, ct2.cube_hu
    assert cube2.GetSize() == cube.GetSize()
    assert np.allclose(cube2.GetSpacing(), cube.GetSpacing())
    assert np.allclose(cube2.GetOrigin(), cube.GetOrigin())
    assert np.allclose(cube2.GetDirection(), cube.GetDirection())
    assert np.array_equal(sitk.GetArrayFromImage(cube2), sitk.GetArrayFromImage(cube))
    assert ct2.dicom_info.study_instance_uid == _read(next(out.glob("*.dcm"))).StudyInstanceUID


def test_4d_ct_load_data_and_load_patient(ct_4d_export):
    _, out = ct_4d_export
    assert load_data(out)["ct"].num_of_ct_scen == NUM_PHASES
    with pytest.warns(UserWarning, match="cst"):
        ct, _ = load_patient(out)
    assert ct.num_of_ct_scen == NUM_PHASES


def test_4d_ct_without_temporal_positions_imports_largest_series(tmp_path):
    DicomExporter(tmp_path / "out").save(ct=_ct_4d_phases())
    for path in (tmp_path / "out").glob("*.dcm"):
        ds = _read(path)
        del ds.TemporalPositionIdentifier
        ds.save_as(path)
    with pytest.warns(UserWarning, match="Multiple CT series"):
        ct2 = DicomImporter(tmp_path / "out").load_ct()
    assert ct2.num_of_ct_scen == 1


def test_4d_ct_with_differing_geometry_imports_largest_series(tmp_path):
    DicomExporter(tmp_path / "out").save(ct=_ct_4d_phases())
    for path in (tmp_path / "out").glob("CT_p03_*.dcm"):
        ds = _read(path)
        ds.ImagePositionPatient = [ds.ImagePositionPatient[0] + 5.0, *ds.ImagePositionPatient[1:]]
        ds.save_as(path)
    with pytest.warns(UserWarning, match="differ in geometry"):
        ct2 = DicomImporter(tmp_path / "out").load_ct()
    assert ct2.num_of_ct_scen == 1


@pytest.mark.parametrize("fmt", ["rtstruct", "seg"])
def test_4d_cst_exports_first_phase_referencing_first_series(fmt, tmp_path):
    ct = _ct_4d_phases()
    cst = _cst_for(ct, moving=True)
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="first of 3 phases"):
        DicomExporter(out, structure_format=fmt).save(ct=ct, cst=cst)

    first_series = _read(out / "CT_p01_0001.dcm").SeriesInstanceUID
    if fmt == "rtstruct":
        ds = _read(out / "RTSTRUCT.dcm")
        ref = ds.ReferencedFrameOfReferenceSequence[0].RTReferencedStudySequence[0]
        assert ref.RTReferencedSeriesSequence[0].SeriesInstanceUID == first_series
    else:
        ds = _read(out / "SEG.dcm")
        assert ds.ReferencedSeriesSequence[0].SeriesInstanceUID == first_series

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        imported = DicomImporter(out).load_data()
    assert imported["ct"].num_of_ct_scen == NUM_PHASES
    cst2 = imported["cst"]
    assert cst2.num_of_scenarios == NUM_PHASES
    by_name = {v.name: v for v in cst2.vois}
    for voi in cst.vois:
        phases = by_name[voi.name].scenario_indices(order="numpy")
        assert all(np.array_equal(phases[0], p) for p in phases)
        assert np.array_equal(phases[0], voi.scenario_indices(0, order="numpy"))


def test_seg_export_keeps_overlapping_segments(tmp_path):
    ct = validate_ct(cube_hu=_ct_4d_phases().cube_hu[:, :, :, 0])
    cst = _cst_for(ct)
    arrays = {v.name: sitk.GetArrayFromImage(v.mask) for v in cst.vois}
    assert (arrays["Body"] & arrays["PTV"] & arrays["CTV"]).any()

    DicomExporter(tmp_path / "out", structure_format="seg").save(ct=ct, cst=cst)
    seg = _read(tmp_path / "out" / "SEG.dcm")
    frames = seg.pixel_array
    numbers = [
        f.SegmentIdentificationSequence[0].ReferencedSegmentNumber
        for f in seg.PerFrameFunctionalGroupsSequence
    ]
    assert [s.SegmentLabel for s in seg.SegmentSequence] == ["Body", "PTV", "CTV"]
    for number, name in enumerate(arrays, start=1):
        slices = [frames[i] for i, n in enumerate(numbers) if n == number]
        assert len(slices) == int(arrays[name].any(axis=(1, 2)).sum())
        assert sum(int(s.sum()) for s in slices) == int(arrays[name].sum())

    imported = DicomImporter(tmp_path / "out").load_cst(ct)
    for voi in imported.vois:
        assert np.array_equal(sitk.GetArrayFromImage(voi.mask), arrays[voi.name])


def test_4d_ct_files_validate(dicom_standard, ct_4d_export):
    _, out = ct_4d_export
    # Temporal Position Identifier and Number of Temporal Positions are not attributes of the
    # CT IOD, but vendors use them to mark the phases of a 4D CT; nothing else may be reported.
    expected = {
        ("General", "(0020,0100)", "TagUnexpected"),
        ("General", "(0020,0105)", "TagUnexpected"),
    }
    for name in ("CT_p01_0001.dcm", "CT_p02_0005.dcm", "CT_p03_0010.dcm"):
        errors = _validation_errors(dicom_standard, out / name)
        assert {e[1:] for e in errors} == expected


@pytest.mark.parametrize("fmt, name", [("rtstruct", "RTSTRUCT.dcm"), ("seg", "SEG.dcm")])
def test_4d_export_with_structures_validates(dicom_standard, fmt, name, tmp_path):
    ct = _ct_4d_phases()
    out = tmp_path / "out"
    with pytest.warns(UserWarning, match="first of 3 phases"):
        DicomExporter(out, structure_format=fmt).save(ct=ct, cst=_cst_for(ct))
    assert _validation_errors(dicom_standard, out / name) == []


def test_nan_dose_voxels_are_written_as_zero_with_a_warning(tmp_path, plain_ct):
    arr = np.ones((4, 8, 8))
    arr[1, 2, 3] = np.nan
    arr[2, 2, 2] = 3.0
    dose = sitk.GetImageFromArray(arr)
    dose.CopyInformation(plain_ct.cube_hu)

    with pytest.warns(UserWarning, match="non-finite"):
        out = _export(tmp_path, ct=plain_ct, dose=dose)

    ds = _read(out / "RTDOSE.dcm")
    pixels = ds.pixel_array.astype(float) * float(ds.DoseGridScaling)
    assert pixels[1, 2, 3] == 0.0
    assert pixels.max() == pytest.approx(3.0, rel=1e-6)
    assert pixels[0, 0, 0] == pytest.approx(1.0, rel=1e-6)


def test_empty_roi_has_no_empty_contour_sequence(tmp_path, plain_ct):
    mask = sitk.GetImageFromArray(np.zeros((4, 8, 8), np.uint8))
    mask.CopyInformation(plain_ct.cube_hu)
    from pyRadPlan.cst import validate_cst

    cst = validate_cst(
        [{"name": "empty", "voi_type": "OAR", "mask": mask, "ct_image": plain_ct}], plain_ct
    )
    out = _export(tmp_path, ct=plain_ct, cst=cst)
    ds = _read(out / "RTSTRUCT.dcm")
    assert "ContourSequence" not in ds.ROIContourSequence[0]

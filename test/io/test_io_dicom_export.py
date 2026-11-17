"""Tests for DICOM export metadata handling (DicomInfo, context resolution, IOD conformance)."""

import logging
import os
from pathlib import Path

import numpy as np
import pydicom
import pytest
import SimpleITK as sitk

from pyRadPlan.core import Grid
from pyRadPlan.ct import CT, DicomInfo, resample_ct, validate_ct
from pyRadPlan.io import load_patient, save_data
from pyRadPlan.io.dicom import DicomExporter, DicomImporter
from pyRadPlan.io.dicom._export_common import resolve_export_context


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


def test_4d_ct_warns(plain_ct, tmp_path):
    ct4d = CT(cube_hu=sitk.JoinSeries([plain_ct.cube_hu, plain_ct.cube_hu]))
    with pytest.warns(UserWarning, match="first phase"):
        _export(tmp_path, ct4d)


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


def _validation_errors(info, path):
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
                # RTDOSE of summation type PLAN needs an RTPLAN, which is not exported yet.
                if tag.tag == 0x300C0002:
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

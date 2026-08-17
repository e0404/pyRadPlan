"""Shared helpers and constants for DICOM export."""

from dataclasses import dataclass, field

from pydicom.dataset import FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

# SOP Class UIDs.
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
RT_DOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
RT_STRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
SEG_STORAGE = "1.2.840.10008.5.1.4.1.1.66.4"
#: Referenced from an RTSTRUCT's RTReferencedStudySequence to point back at the study.
DETACHED_STUDY_MANAGEMENT = "1.2.840.10008.3.1.2.3.1"


@dataclass
class UIDContext:
    """Shared identifiers so exported objects reference one another correctly."""

    study_uid: str = field(default_factory=generate_uid)
    frame_uid: str = field(default_factory=generate_uid)
    patient_name: str = "pyRadPlan^Export"
    patient_id: str = "pyRadPlan"


def make_file_meta(sop_class_uid: str, sop_instance_uid: str) -> FileMetaDataset:
    """Create a minimal, valid file meta dataset."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class_uid
    fm.MediaStorageSOPInstanceUID = sop_instance_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()
    return fm


def direction_to_orientation(direction: tuple) -> list:
    """Convert a SimpleITK 3x3 direction (row-major) to DICOM ImageOrientationPatient."""
    d = direction
    # Columns of the direction matrix are the x- and y-axis directions.
    return [d[0], d[3], d[6], d[1], d[4], d[7]]


def populate_common(  # noqa: PLR0913 - DICOM datasets need several identifiers
    ds, ctx: UIDContext, sop_class_uid: str, series_uid: str, modality: str, series_number: int
):
    """Populate the file meta and common patient/study/series tags of a dataset."""
    sop_instance_uid = generate_uid()
    ds.file_meta = make_file_meta(sop_class_uid, sop_instance_uid)
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid
    ds.StudyInstanceUID = ctx.study_uid
    ds.SeriesInstanceUID = series_uid
    ds.FrameOfReferenceUID = ctx.frame_uid
    ds.Modality = modality
    ds.PatientName = ctx.patient_name
    ds.PatientID = ctx.patient_id
    ds.PatientBirthDate = ""
    ds.PatientSex = ""
    ds.StudyDate = ""
    ds.StudyTime = ""
    ds.StudyID = "1"
    ds.AccessionNumber = ""
    ds.SeriesNumber = series_number
    return sop_instance_uid

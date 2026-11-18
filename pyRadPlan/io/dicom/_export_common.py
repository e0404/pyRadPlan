"""Shared helpers and constants for DICOM export."""

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import SimpleITK as sitk
from pydicom.dataset import FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.valuerep import format_number_as_ds

import pyRadPlan
from pyRadPlan.cst import StructureSet
from pyRadPlan.ct import CT, DicomInfo

logger = logging.getLogger(__name__)

# SOP Class UIDs.
CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
RT_DOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
RT_STRUCT_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
RT_PLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"
RT_ION_PLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.8"
SEG_STORAGE = "1.2.840.10008.5.1.4.1.1.66.4"
#: Referenced from an RTSTRUCT's RTReferencedStudySequence to point back at the study.
DETACHED_STUDY_MANAGEMENT = "1.2.840.10008.3.1.2.3.1"


@dataclass
class ExportContext:
    """Resolved identifiers and metadata shared by all objects of one export.

    Study and frame-of-reference UIDs link the exported objects (and, unless
    anonymized, the source images); series and SOP instance UIDs are always new.
    """

    study_uid: str = field(default_factory=generate_uid)
    frame_uid: str = field(default_factory=generate_uid)
    patient_name: str = "pyRadPlan^Export"
    patient_id: str = "pyRadPlan"
    patient_birth_date: str = ""
    patient_sex: str = ""
    patient_position: str = "HFS"
    study_id: str = "1"
    study_date: str = ""
    study_time: str = ""
    study_description: str = ""
    accession_number: str = ""
    referring_physician_name: str = ""
    position_reference_indicator: str = ""
    kvp: str = ""
    export_date: str = ""
    export_time: str = ""

    def __post_init__(self):
        if not self.export_date:
            now = datetime.now()
            self.export_date = now.strftime("%Y%m%d")
            self.export_time = now.strftime("%H%M%S")


@dataclass
class PlanReference:
    """Reference to an exported RT (Ion) Plan, for the RTDOSE Referenced RT Plan Sequence.

    Parameters
    ----------
    sop_instance_uid : str
        SOP Instance UID of the referenced plan.
    fraction_group_number : int, default 1
        Fraction group of the plan the dose refers to (FRACTION and BEAM doses).
    beam_numbers : list of int, optional
        DICOM Beam Number of each beam, in STF beam order. Defaults to ``1..N``.
    sop_class_uid : str
        SOP Class UID of the plan; RT Ion Plan Storage by default, use
        :data:`RT_PLAN_STORAGE` for photon plans.
    skipped_beams : list of int, optional
        0-based STF indices of beams that are not part of the exported plan (no spot with
        positive weight); their per-beam doses are not exported.
    """

    sop_instance_uid: str
    fraction_group_number: int = 1
    beam_numbers: Optional[list[int]] = None
    sop_class_uid: str = RT_ION_PLAN_STORAGE
    skipped_beams: list[int] = field(default_factory=list)

    def beam_number(self, beam_index: int) -> int:
        """DICOM Beam Number of the beam with the given 0-based STF index."""
        if self.beam_numbers is None:
            return beam_index + 1
        return int(self.beam_numbers[beam_index])


#: ExportContext attribute -> DicomInfo field.
_CONTEXT_FIELDS = {
    "study_uid": "study_instance_uid",
    "frame_uid": "frame_of_reference_uid",
    **{
        n: n
        for n in (
            "patient_name",
            "patient_id",
            "patient_birth_date",
            "patient_sex",
            "patient_position",
            "study_id",
            "study_date",
            "study_time",
            "study_description",
            "accession_number",
            "referring_physician_name",
            "position_reference_indicator",
            "kvp",
        )
    },
}


def _warn_on_mismatch(ct: Optional[CT], cst: Optional[StructureSet]) -> None:
    if ct is None or cst is None:
        return
    a, b = ct.dicom_info, cst.ct_image.dicom_info
    if a is None or b is None:
        return
    for name in ("patient_id", "study_instance_uid", "frame_of_reference_uid"):
        if getattr(a, name) != getattr(b, name):
            logger.warning(
                "The CT and the structure set's reference CT differ in %s; using the CT's.",
                name,
            )


def resolve_export_context(
    ct: Optional[CT] = None,
    cst: Optional[StructureSet] = None,
    metadata: Any = None,
    anonymize: bool = False,
) -> ExportContext:
    """
    Resolve the export context.

    Precedence: explicit ``metadata`` over ``ct.dicom_info`` (or the structure
    set's reference CT when ``ct`` is None) over generated defaults.

    Parameters
    ----------
    ct : CT, optional
        The CT to export.
    cst : StructureSet, optional
        The structure set to export.
    metadata : DicomInfo or dict or pydicom.Dataset, optional
        Explicit metadata; attributes set here override the CT's.
    anonymize : bool, default False
        Replace identifying attributes and generate new study and frame of
        reference UIDs (consistent over the whole export).

    Returns
    -------
    ExportContext
        The resolved context.
    """
    _warn_on_mismatch(ct, cst)
    source = ct if ct is not None else (cst.ct_image if cst is not None else None)

    infos = []
    if source is not None and source.dicom_info is not None:
        infos.append(source.dicom_info)
    if metadata is not None:
        infos.append(DicomInfo.model_validate(metadata))

    kwargs: dict = {}
    for info in infos:
        for ctx_name, info_name in _CONTEXT_FIELDS.items():
            value = getattr(info, info_name)
            if value is not None:
                kwargs[ctx_name] = f"{value:g}" if ctx_name == "kvp" else str(value)
    ctx = ExportContext(**kwargs)

    if anonymize:
        ctx.patient_name = "Anonymous^Patient"
        ctx.patient_id = "ANON"
        ctx.patient_birth_date = ""
        ctx.patient_sex = ""
        ctx.accession_number = ""
        ctx.referring_physician_name = ""
        ctx.study_description = ""
        ctx.study_date = ""
        ctx.study_time = ""
        ctx.study_uid = generate_uid()
        ctx.frame_uid = generate_uid()
    return ctx


def make_file_meta(sop_class_uid: str, sop_instance_uid: str) -> FileMetaDataset:
    """Create a minimal, valid file meta dataset."""
    fm = FileMetaDataset()
    fm.MediaStorageSOPClassUID = sop_class_uid
    fm.MediaStorageSOPInstanceUID = sop_instance_uid
    fm.TransferSyntaxUID = ExplicitVRLittleEndian
    fm.ImplementationClassUID = generate_uid()
    return fm


def first_phase_image(cst: StructureSet) -> sitk.Image:
    """Return the 3D reference image of a structure set (first phase of a 4D CT)."""
    image = cst.ct_image.cube_hu
    return image[:, :, :, 0] if image.GetDimension() == 4 else image


def warn_if_multiphase(cst: StructureSet, kind: str) -> None:
    """Warn that ``kind`` (RTSTRUCT or SEG) holds only the first phase of a 4D structure set."""
    phases = max((v.mask.GetSize()[3] for v in cst.vois if v.mask.GetDimension() == 4), default=1)
    if phases > 1:
        warnings.warn(
            f"The {kind} export holds the structures of the first of {phases} phases only; "
            "the further phases are not exported.",
            UserWarning,
            stacklevel=4,
        )


def ds_list(values) -> list:
    """Quantize numbers to Decimal Strings (16 characters at most)."""
    return [float(format_number_as_ds(float(v))) for v in values]


def direction_to_orientation(direction: tuple) -> list:
    """Convert a SimpleITK 3x3 direction (row-major) to DICOM ImageOrientationPatient."""
    d = direction
    # Columns of the direction matrix are the x- and y-axis directions.
    return [d[0], d[3], d[6], d[1], d[4], d[7]]


def populate_common(  # noqa: PLR0913 - DICOM datasets need several identifiers
    ds,
    ctx: ExportContext,
    sop_class_uid: str,
    series_uid: str,
    modality: str,
    series_number: int,
    series_description: str,
):
    """Populate file meta and the patient, study, series, frame of reference and equipment tags."""
    sop_instance_uid = generate_uid()
    ds.file_meta = make_file_meta(sop_class_uid, sop_instance_uid)
    ds.SpecificCharacterSet = "ISO_IR 192"
    ds.InstanceCreationDate = ctx.export_date
    ds.InstanceCreationTime = ctx.export_time
    ds.SOPClassUID = sop_class_uid
    ds.SOPInstanceUID = sop_instance_uid

    ds.PatientName = ctx.patient_name
    ds.PatientID = ctx.patient_id
    ds.PatientBirthDate = ctx.patient_birth_date
    ds.PatientSex = ctx.patient_sex

    ds.StudyInstanceUID = ctx.study_uid
    ds.StudyDate = ctx.study_date
    ds.StudyTime = ctx.study_time
    ds.ReferringPhysicianName = ctx.referring_physician_name
    ds.StudyID = ctx.study_id
    ds.AccessionNumber = ctx.accession_number
    if ctx.study_description:
        ds.StudyDescription = ctx.study_description

    ds.Modality = modality
    ds.SeriesInstanceUID = series_uid
    ds.SeriesNumber = series_number
    ds.SeriesDate = ctx.export_date
    ds.SeriesTime = ctx.export_time
    ds.SeriesDescription = series_description

    ds.FrameOfReferenceUID = ctx.frame_uid
    ds.PositionReferenceIndicator = ctx.position_reference_indicator

    ds.Manufacturer = "pyRadPlan"
    ds.ManufacturerModelName = "pyRadPlan"
    ds.DeviceSerialNumber = "0"
    ds.SoftwareVersions = pyRadPlan.__version__
    return sop_instance_uid

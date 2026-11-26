"""DICOM provenance metadata carried alongside a CT image."""

from typing import Any, Optional

import numpy as np
from pydicom.dataset import Dataset
from pydantic import Field, model_validator

from pyRadPlan.core import PyRadPlanBaseModel

#: DICOM keyword -> field name for all explicitly modelled attributes.
_KEYWORD_TO_FIELD = {
    "PatientName": "patient_name",
    "PatientID": "patient_id",
    "PatientBirthDate": "patient_birth_date",
    "PatientSex": "patient_sex",
    "StudyInstanceUID": "study_instance_uid",
    "StudyID": "study_id",
    "StudyDate": "study_date",
    "StudyTime": "study_time",
    "StudyDescription": "study_description",
    "AccessionNumber": "accession_number",
    "ReferringPhysicianName": "referring_physician_name",
    "FrameOfReferenceUID": "frame_of_reference_uid",
    "PositionReferenceIndicator": "position_reference_indicator",
    "PatientPosition": "patient_position",
    "Manufacturer": "manufacturer",
    "ManufacturerModelName": "manufacturer_model_name",
    "ConvolutionKernel": "convolution_kernel",
    "KVP": "kvp",
    "SeriesInstanceUID": "series_instance_uid",
    "SeriesDescription": "series_description",
}

_STRING_FIELDS = {f for f in _KEYWORD_TO_FIELD.values() if f != "kvp"}

#: Keys not stored in ``additional`` (file info, geometry defined by the image, private data).
_EXCLUDED_KEYWORDS = {
    "Filename",
    "FileModDate",
    "FileSize",
    "Format",
    "FormatVersion",
    "Width",
    "Height",
    "BitDepth",
    "ColorType",
    "PixelSpacing",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "SlicePositions",
    "SliceThickness",
    "Rows",
    "Columns",
    "RescaleSlope",
    "RescaleIntercept",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "SOPInstanceUID",
    "SOPClassUID",
    "PixelData",
    "TransferSyntaxUID",
    "ImplementationClassUID",
    "ImplementationVersionName",
}
_EXCLUDED_PREFIXES = ("Private_", "MediaStorage", "FileMetaInformation")

_PERSON_NAME_PARTS = ("FamilyName", "GivenName", "MiddleName", "NamePrefix", "NameSuffix")


def _is_excluded(keyword: str) -> bool:
    return keyword in _EXCLUDED_KEYWORDS or keyword.startswith(_EXCLUDED_PREFIXES)


def _plain(value: Any) -> Any:
    """Convert pydicom/numpy/matRad values to plain JSON-able Python types (or None)."""
    result: Any
    if value is None or isinstance(value, (bytes, bytearray)):
        result = None
    elif isinstance(value, np.ndarray):
        result = None if value.size == 0 else _plain(value.tolist())
    elif isinstance(value, np.generic):
        result = value.item()
    elif isinstance(value, dict):
        if "FamilyName" in value:
            parts = [_plain(value.get(p)) or "" for p in _PERSON_NAME_PARTS]
            while parts and not parts[-1]:
                parts.pop()
            result = "^".join(str(p) for p in parts) or None
        else:
            result = {str(k): _plain(v) for k, v in value.items() if _plain(v) is not None}
    elif isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        result = [_plain(v) for v in value]
    elif isinstance(value, bool):
        result = value
    elif isinstance(value, int):
        result = int(value)
    elif isinstance(value, float):
        result = float(value)
    elif isinstance(value, str):
        result = str(value) or None
    elif type(value).__name__ in ("Sequence", "Dataset"):
        result = None
    else:
        result = str(value) or None
    return result


def _as_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return "\\".join(str(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


class DicomInfo(PyRadPlanBaseModel):
    """
    DICOM provenance metadata of a CT image.

    One model replaces matRad's ``dicomInfo`` and ``dicomMeta`` structs. It
    accepts snake_case, camelCase and DICOM keyword (PascalCase) keys, pydicom
    datasets and matRad dictionaries.

    Attributes
    ----------
    patient_name, patient_id, patient_birth_date, patient_sex : str, optional
        Patient module attributes.
    study_instance_uid, study_id, study_date, study_time, study_description : str, optional
        General study module attributes.
    accession_number, referring_physician_name : str, optional
        General study module attributes.
    frame_of_reference_uid, position_reference_indicator : str, optional
        Frame of reference module attributes.
    patient_position : str, optional
        Patient position, e.g. ``"HFS"``.
    manufacturer, manufacturer_model_name, convolution_kernel : str, optional
        Source equipment attributes (kept for provenance only).
    kvp : float, optional
        Peak kilovoltage of the source acquisition.
    series_instance_uid, series_description : str, optional
        Source CT series attributes.
    sop_instance_uids : list[str], optional
        Per-slice SOP instance UIDs of the source series (provenance only).
    additional : dict
        Remaining public DICOM attributes of the source, keyed by DICOM keyword.
    """

    patient_name: Optional[str] = None
    patient_id: Optional[str] = None
    patient_birth_date: Optional[str] = None
    patient_sex: Optional[str] = None

    study_instance_uid: Optional[str] = None
    study_id: Optional[str] = None
    study_date: Optional[str] = None
    study_time: Optional[str] = None
    study_description: Optional[str] = None
    accession_number: Optional[str] = None
    referring_physician_name: Optional[str] = None

    frame_of_reference_uid: Optional[str] = None
    position_reference_indicator: Optional[str] = None

    patient_position: Optional[str] = None

    manufacturer: Optional[str] = None
    manufacturer_model_name: Optional[str] = None
    convolution_kernel: Optional[str] = None
    kvp: Optional[float] = None

    series_instance_uid: Optional[str] = None
    series_description: Optional[str] = None
    sop_instance_uids: Optional[list[str]] = None

    additional: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any) -> Any:
        """Accept DicomInfo, dicts (any key style), pydicom datasets and matRad structs."""
        if data is None or isinstance(data, cls):
            return data
        if isinstance(data, Dataset):
            data = cls._dataset_to_dict(data)
        if not isinstance(data, dict):
            return data

        keys = set(data)
        if keys and keys <= {"dicomInfo", "dicomMeta"}:
            merged: dict = {}
            for part in ("dicomInfo", "dicomMeta"):
                merged.update(cls._flatten(data.get(part)))
            return cls._normalize_dict(merged)
        return cls._normalize_dict(cls._flatten(data))

    @classmethod
    def _flatten(cls, data: Any) -> dict:
        if data is None:
            return {}
        if isinstance(data, cls):
            return data.model_dump()
        if not isinstance(data, dict):
            if isinstance(data, Dataset):
                return cls._dataset_to_dict(data)
            raise TypeError(f"Cannot interpret {type(data)} as DICOM metadata.")
        return dict(data)

    @staticmethod
    def _dataset_to_dict(ds: Any) -> dict:
        out = {}
        for elem in ds:
            if elem.tag.is_private or not elem.keyword or elem.VR == "SQ":
                continue
            out[elem.keyword] = elem.value
        return out

    @classmethod
    def _normalize_dict(cls, data: dict) -> dict:
        fields = cls.model_fields
        alias_to_field = {f.alias: n for n, f in fields.items() if f.alias}
        result: dict = {}
        additional: dict = {}
        for raw_key, value in data.items():
            key = str(raw_key)
            if key == "additional":
                if isinstance(value, dict):
                    additional.update(value)
                continue
            if key in fields:
                name = key
            elif key in alias_to_field:
                name = alias_to_field[key]
            elif key in _KEYWORD_TO_FIELD:
                name = _KEYWORD_TO_FIELD[key]
            elif key == "SOPInstanceUIDs":
                name = "sop_instance_uids"
            else:
                if not _is_excluded(key):
                    additional[key] = value
                continue
            result[name] = value

        for name, raw in list(result.items()):
            result[name] = cls._coerce(name, _plain(raw))

        result["additional"] = {
            k: v for k, v in ((k, _plain(v)) for k, v in additional.items()) if v is not None
        }
        return result

    @staticmethod
    def _coerce(name: str, value: Any) -> Any:
        if name in _STRING_FIELDS:
            return _as_string(value)
        if name == "sop_instance_uids" and isinstance(value, str):
            return [value]
        if name == "kvp" and isinstance(value, list):
            return value[0] if value else None
        return value

    @classmethod
    def from_matrad(cls, dicom_info: Any = None, dicom_meta: Any = None) -> "DicomInfo":
        """
        Build a DicomInfo from matRad's ``dicomInfo`` and ``dicomMeta`` structs.

        Parameters
        ----------
        dicom_info, dicom_meta : dict, optional
            The matRad structs. Values of ``dicom_meta`` take precedence.

        Returns
        -------
        DicomInfo
            The merged metadata.
        """
        return cls.model_validate({"dicomInfo": dicom_info, "dicomMeta": dicom_meta})

    def to_dicom_dict(self) -> dict:
        """
        Flatten to a dictionary keyed by DICOM keyword.

        Returns
        -------
        dict
            All set attributes, e.g. for a matRad ``dicomInfo`` struct.
        """
        out: dict[str, Any] = {}
        for keyword, name in _KEYWORD_TO_FIELD.items():
            value = getattr(self, name)
            if value is not None:
                out[keyword] = value
        if self.sop_instance_uids:
            out["SOPInstanceUIDs"] = list(self.sop_instance_uids)
        out.update(self.additional)
        return out

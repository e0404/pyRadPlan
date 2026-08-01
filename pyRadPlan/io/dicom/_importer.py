"""DICOM importer: scans a folder and dispatches per modality."""

import os
import logging
import warnings
from typing import ClassVar, Optional

import pydicom
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import validate_cst, StructureSet

from ..base import BaseImporter
from ._import_ct import import_ct
from ._import_cst import import_rtstruct, import_seg
from ._import_dose import import_dose, _dose_descriptor

logger = logging.getLogger(__name__)

_MODALITIES = ("CT", "RTSTRUCT", "SEG", "RTDOSE")


class DicomImporter(BaseImporter):
    """Importer for a folder (or single file) of DICOM data."""

    format: ClassVar[str] = "dcm"
    name: ClassVar[str] = "DICOM Importer"
    extensions: ClassVar[tuple[str, ...]] = (".dcm",)

    def __init__(self, path):
        super().__init__(path)
        self._classified = None

    @classmethod
    def handles_directory(cls, path) -> bool:
        """Return True if the directory appears to contain DICOM files."""
        try:
            entries = os.listdir(path)
        except OSError:
            return False
        if any(entry.lower().endswith(".dcm") for entry in entries):
            return True
        # Fall back to probing files with pydicom.
        for entry in entries:
            full = os.path.join(path, entry)
            if not os.path.isfile(full):
                continue
            try:
                pydicom.dcmread(full, stop_before_pixels=True, specific_tags=[])
                return True
            except Exception:  # noqa: BLE001 - non-DICOM files are simply skipped
                continue
        return False

    def _classify(self) -> tuple[str, dict[str, list[str]]]:
        """Return (directory, {modality: [files]}) for the source, cached."""
        if self._classified is not None:
            return self._classified

        path = self._require_path()
        if os.path.isdir(path):
            directory = path
            files = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            directory = os.path.dirname(path) or "."
            files = [path]

        groups: dict[str, list[str]] = {modality: [] for modality in _MODALITIES}
        for f in files:
            if not os.path.isfile(f):
                continue
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 - non-DICOM files are simply ignored
                continue
            modality = getattr(ds, "Modality", None)
            if modality in groups:
                groups[modality].append(f)

        self._classified = (directory, groups)
        return self._classified

    def list_ct_series(self) -> list[dict]:
        """List the CT series in the source, one entry per ``SeriesInstanceUID``.

        Each entry has ``series_uid``, ``description``, ``num_slices`` and the
        member ``files``. Used by the import dialog to let the user pick a series.
        """
        _, groups = self._classify()
        series: dict[str, dict] = {}
        for f in groups["CT"]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 - unreadable files are skipped
                continue
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            entry = series.setdefault(
                uid,
                {
                    "series_uid": uid,
                    "description": str(getattr(ds, "SeriesDescription", "") or ""),
                    "files": [],
                },
            )
            entry["files"].append(f)
        result = list(series.values())
        for entry in result:
            entry["num_slices"] = len(entry["files"])
        result.sort(key=lambda e: (e["description"], e["series_uid"]))
        return result

    def list_structure_sets(self) -> list[dict]:
        """List the structure sets (RTSTRUCT/SEG), one entry per file."""
        _, groups = self._classify()
        result = []
        for f in groups["RTSTRUCT"] + groups["SEG"]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 - unreadable files are skipped
                continue
            modality = getattr(ds, "Modality", "")
            if modality == "RTSTRUCT" and hasattr(ds, "StructureSetROISequence"):
                names = [str(getattr(roi, "ROIName", "")) for roi in ds.StructureSetROISequence]
            elif modality == "SEG" and hasattr(ds, "SegmentSequence"):
                names = [str(getattr(seg, "SegmentLabel", "")) for seg in ds.SegmentSequence]
            else:
                names = []
            label = str(
                getattr(ds, "StructureSetLabel", None)
                or getattr(ds, "SeriesDescription", None)
                or os.path.basename(f)
            )
            result.append(
                {"path": f, "modality": modality, "label": label, "structure_names": names}
            )
        return result

    def list_doses(self) -> list[dict]:
        """List the RTDOSE distributions, one entry per file."""
        _, groups = self._classify()
        result = []
        for f in groups["RTDOSE"]:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 - unreadable files are skipped
                continue
            result.append(
                {
                    "path": f,
                    "summation": str(getattr(ds, "DoseSummationType", "") or ""),
                    "description": _dose_descriptor(ds),
                }
            )
        return result

    def load_ct(self, series_uid: Optional[str] = None) -> CT:
        directory, groups = self._classify()
        if not groups["CT"]:
            raise ValueError(f"No CT series found in {self.path}.")

        series = self.list_ct_series()
        if series_uid is not None:
            match = next((s for s in series if s["series_uid"] == series_uid), None)
            if match is None:
                raise ValueError(f"CT series {series_uid!r} not found in {self.path}.")
            files = match["files"]
        elif len(series) > 1:
            match = max(series, key=lambda s: s["num_slices"])
            warnings.warn(
                f"Multiple CT series found in {self.path}; using "
                f"'{match['description']}' ({match['num_slices']} slices).",
                stacklevel=2,
            )
            files = match["files"]
        else:
            files = groups["CT"]
        return import_ct(directory, files)

    def load_cst(
        self, ct: Optional[CT] = None, struct_file: Optional[str] = None
    ) -> Optional[StructureSet]:
        _, groups = self._classify()
        struct_files = groups["RTSTRUCT"] + groups["SEG"]
        if not struct_files:
            return None

        if ct is None:
            ct = self.load_ct()

        if struct_file is None:
            if len(struct_files) > 1:
                warnings.warn(
                    f"Multiple structure sets found in {self.path}; using the first "
                    f"({os.path.basename(struct_files[0])}).",
                    stacklevel=2,
                )
            struct_file = struct_files[0]
        elif struct_file not in struct_files:
            raise ValueError(f"Structure set {struct_file!r} not found in {self.path}.")

        f = struct_file
        if f in groups["RTSTRUCT"]:
            vois = import_rtstruct(pydicom.dcmread(f), ct)
        else:
            ct_datasets = [pydicom.dcmread(c, stop_before_pixels=True) for c in groups["CT"]]
            vois = import_seg(pydicom.dcmread(f), ct_datasets, ct)

        if not vois:
            return None
        return validate_cst(vois, ct)

    def load_dose(self, dose_file: Optional[str] = None) -> Optional[sitk.Image]:
        _, groups = self._classify()
        if not groups["RTDOSE"]:
            return None
        if dose_file is not None:
            if dose_file not in groups["RTDOSE"]:
                raise ValueError(f"RTDOSE {dose_file!r} not found in {self.path}.")
            return import_dose([dose_file])
        return import_dose(groups["RTDOSE"])

"""DICOM importer: scans a folder and dispatches per modality."""

import os
import logging
import warnings
from typing import ClassVar, Optional

import numpy as np
import pydicom
import SimpleITK as sitk

from pyRadPlan.core.resample import resample_image
from pyRadPlan.ct import CT
from pyRadPlan.cst import validate_cst, StructureSet

from ..base import BaseImporter
from ._import_ct import import_ct
from ._import_cst import import_rtstruct, import_seg
from ._import_dose import import_dose, _dose_descriptor

logger = logging.getLogger(__name__)

_MODALITIES = ("CT", "RTSTRUCT", "SEG", "RTDOSE")


def _resample_dose_to_ct(dose: sitk.Image, ct: CT) -> sitk.Image:
    """Return *dose* on the grid of *ct*, resampling only when they differ.

    Dose outside the RTDOSE cube is zero rather than nearest-neighbour
    extrapolated: the cube covers the irradiated region, and smearing its edge
    values across the rest of the patient would invent dose that was never
    computed.
    """
    reference = ct.cube_hu
    if (
        dose.GetSize() == reference.GetSize()
        and np.allclose(dose.GetSpacing(), reference.GetSpacing())
        and np.allclose(dose.GetOrigin(), reference.GetOrigin())
        and np.allclose(dose.GetDirection(), reference.GetDirection())
    ):
        return dose

    logger.info(
        "Resampling dose from %s at %s mm onto the CT grid %s at %s mm.",
        dose.GetSize(),
        tuple(round(s, 3) for s in dose.GetSpacing()),
        reference.GetSize(),
        tuple(round(s, 3) for s in reference.GetSpacing()),
    )
    return resample_image(
        dose, interpolator=sitk.sitkLinear, target_image=reference, extrapolate=0
    )


class DicomImporter(BaseImporter):
    """Importer for a folder (or single file) of DICOM data."""

    format: ClassVar[str] = "dcm"
    name: ClassVar[str] = "DICOM Importer"
    extensions: ClassVar[tuple[str, ...]] = (".dcm",)

    def __init__(self, path):
        super().__init__(path)
        self._classified = None
        # Pixel-free headers of every DICOM file found, keyed by path. Filled by
        # :meth:`_classify` so the listing methods do not re-read them.
        self._headers: dict[str, pydicom.Dataset] = {}
        # Last CT loaded, reused by :meth:`load_dose` as the grid to resample onto.
        self._ct: Optional[CT] = None

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
        """Return (directory, {modality: [files]}) for the source, cached.

        Reading the header of every file in the folder is the slow part of an
        import, so it is reported as its own progress level and the headers are
        kept in :attr:`_headers` for the listing and load methods to reuse.
        """
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
        logger.info("Scanning %d file(s) in %s…", len(files), directory)
        for f in self.track(files, name="Scanning files", unit="file"):
            if not os.path.isfile(f):
                continue
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True)
            except Exception:  # noqa: BLE001 - non-DICOM files are simply ignored
                continue
            modality = getattr(ds, "Modality", None)
            if modality in groups:
                groups[modality].append(f)
                self._headers[f] = ds

        logger.info(
            "Found %s.",
            ", ".join(f"{len(paths)}x {modality}" for modality, paths in groups.items() if paths)
            or "no CT, RTSTRUCT, SEG or RTDOSE data",
        )

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
            ds = self._headers[f]
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
            ds = self._headers[f]
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
            ds = self._headers[f]
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
        self._ct = import_ct(directory, files, reporter=self)
        return self._ct

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
        logger.info("Importing structure set %s.", os.path.basename(f))
        if f in groups["RTSTRUCT"]:
            vois = import_rtstruct(pydicom.dcmread(f), ct, reporter=self)
        else:
            ct_datasets = [self._headers[c] for c in groups["CT"]]
            vois = import_seg(pydicom.dcmread(f), ct_datasets, ct, reporter=self)

        if not vois:
            return None
        logger.info("Assembling structure set from %d VOI(s).", len(vois))
        return validate_cst(vois, ct)

    def load_dose(
        self, dose_file: Optional[str] = None, ct: Optional[CT] = None
    ) -> Optional[sitk.Image]:
        """Load a dose distribution, resampled onto the CT grid.

        RTDOSE is almost always stored on its own grid -- typically coarser than
        the CT and covering only the irradiated region -- so the cube is
        resampled onto the CT grid before it is returned. Everything downstream
        (the viewer's overlays, dose/structure comparisons) indexes quantities
        with CT voxel indices, and matRad's DICOM import interpolates onto the CT
        grid for the same reason.

        Parameters
        ----------
        dose_file : str, optional
            An explicit RTDOSE path. By default the plan-level physical dose is
            selected from the distributions found in the source.
        ct : CT, optional
            The CT defining the target grid. Defaults to the one this importer
            loaded most recently, and failing that it loads one.

        Returns
        -------
        sitk.Image or None
            The dose in Gy on the CT grid, or ``None`` if the source has no RTDOSE.
        """
        _, groups = self._classify()
        if not groups["RTDOSE"]:
            return None
        if dose_file is not None:
            if dose_file not in groups["RTDOSE"]:
                raise ValueError(f"RTDOSE {dose_file!r} not found in {self.path}.")
            dose = import_dose([dose_file], reporter=self)
        else:
            dose = import_dose(groups["RTDOSE"], reporter=self)

        if ct is None:
            ct = self._ct if self._ct is not None else self.load_ct()
        return _resample_dose_to_ct(dose, ct)

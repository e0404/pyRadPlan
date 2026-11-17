"""DICOM exporter: writes CT, RTSTRUCT and RTDOSE into a folder."""

import os
import logging
from typing import Any, ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter
from ._export_common import resolve_export_context
from ._export_ct import export_ct
from ._export_cst import export_cst, validate_roi_interpreted_types
from ._export_dose import export_dose
from ._export_seg import export_seg
from ._helpers import find_dicom_files

logger = logging.getLogger(__name__)


class DicomExporter(BaseExporter):
    """Exporter writing pyRadPlan objects as DICOM into a directory.

    Parameters
    ----------
    path : str or os.PathLike
        Output directory.
    structure_format : str, optional
        How to export a StructureSet: ``"rtstruct"`` (default) or ``"seg"``.
    metadata : DicomInfo or dict or pydicom.Dataset, optional
        Explicit DICOM metadata. Attributes set here override those of the
        CT's ``dicom_info``, which in turn override generated defaults.
    anonymize : bool, default False
        Replace patient identifiers and generate new study and frame of
        reference UIDs, consistently over all exported objects.
    overwrite : bool, default False
        If the output directory already contains DICOM files (top level only; ``.dcm``
        files and files recognized by :func:`pydicom.misc.is_dicom`), saving raises
        :class:`FileExistsError`. With ``overwrite=True`` exactly these files are
        deleted before writing; other files and subdirectories are left alone.
    roi_interpreted_types : dict, optional
        Maps ROI names to a Defined Term of RT ROI Interpreted Type (3006,00A4) for
        RTSTRUCT export, overriding the type derived from the VOI type and name.
        Unknown terms raise :class:`ValueError`.
    """

    format: ClassVar[str] = "dcm"
    name: ClassVar[str] = "DICOM Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".dcm",)
    container: ClassVar[bool] = False

    #: Default used when the exporter is reached via the combined handler (whose
    #: cooperative __init__ chain does not pass through DicomExporter.__init__).
    structure_format: str = "rtstruct"
    metadata: Any = None
    anonymize: bool = False
    overwrite: bool = False
    roi_interpreted_types: Optional[dict[str, str]] = None

    def __init__(  # noqa: PLR0913 - exporter options
        self,
        path,
        structure_format: str = "rtstruct",
        metadata: Any = None,
        anonymize=False,
        overwrite: bool = False,
        roi_interpreted_types: Optional[dict[str, str]] = None,
    ):
        super().__init__(path)
        self.overwrite = bool(overwrite)
        self.roi_interpreted_types = validate_roi_interpreted_types(roi_interpreted_types)
        self.metadata = metadata
        self.anonymize = bool(anonymize)
        if structure_format not in ("rtstruct", "seg"):
            raise ValueError("structure_format must be 'rtstruct' or 'seg'.")
        self.structure_format = structure_format

    def save(
        self,
        *,
        ct: Optional[CT] = None,
        cst: Optional[StructureSet] = None,
        dose: Optional[sitk.Image] = None,
        **extra,
    ) -> None:
        if ct is None and cst is None and dose is None:
            raise ValueError("Nothing to export: provide at least one of ct, cst, dose.")

        if extra:
            logger.warning("DICOM exporter ignores unsupported objects: %s", list(extra))

        directory = self.path
        ctx = resolve_export_context(ct, cst, self.metadata, self.anonymize)

        existing = find_dicom_files(directory)
        if existing and not self.overwrite:
            raise FileExistsError(
                f"Directory {str(directory)!r} already contains {len(existing)} DICOM file(s); "
                "pass overwrite=True to replace them."
            )
        os.makedirs(directory, exist_ok=True)
        for path in existing:
            os.remove(path)

        # A CT (or the cst's reference CT) anchors the frame of reference.
        if ct is None and cst is not None:
            ct = cst.ct_image

        ct_info = {}
        if ct is not None:
            ct_info = export_ct(ct, directory, ctx)

        if cst is not None:
            if self.structure_format == "seg":
                export_seg(cst, directory, ctx, ct_info)
            else:
                export_cst(cst, directory, ctx, ct_info, self.roi_interpreted_types)

        if dose is not None:
            export_dose(dose, directory, ctx)

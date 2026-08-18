"""DICOM exporter: writes CT, RTSTRUCT and RTDOSE into a folder."""

import os
import logging
from typing import ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter
from ._export_common import UIDContext
from ._export_ct import export_ct
from ._export_cst import export_cst
from ._export_dose import export_dose
from ._export_seg import export_seg

logger = logging.getLogger(__name__)


class DicomExporter(BaseExporter):
    """Exporter writing pyRadPlan objects as DICOM into a directory.

    Parameters
    ----------
    path : str or os.PathLike
        Output directory.
    structure_format : str, optional
        How to export a StructureSet: ``"rtstruct"`` (default) or ``"seg"``.
    """

    format: ClassVar[str] = "dcm"
    name: ClassVar[str] = "DICOM Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".dcm",)
    container: ClassVar[bool] = False

    #: Default used when the exporter is reached via the combined handler (whose
    #: cooperative __init__ chain does not pass through DicomExporter.__init__).
    structure_format: str = "rtstruct"

    def __init__(self, path, structure_format: str = "rtstruct"):
        super().__init__(path)
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
        os.makedirs(directory, exist_ok=True)

        ctx = UIDContext()

        # A CT (or the cst's reference CT) anchors the frame of reference.
        if ct is None and cst is not None:
            ct = cst.ct_image

        ct_info = {}
        if ct is not None:
            ct_info = export_ct(ct, directory, ctx)

        if cst is not None:
            if self.structure_format == "seg":
                export_seg(cst, directory, ctx)
            else:
                export_cst(cst, directory, ctx, ct_info)

        if dose is not None:
            export_dose(dose, directory, ctx)

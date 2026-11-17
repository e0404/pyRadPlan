"""DICOM exporter: writes CT, RTSTRUCT and RTDOSE into a folder."""

import os
import logging
from typing import Any, ClassVar, Optional

import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet

from ..base import BaseExporter
from ._export_common import PlanReference, resolve_export_context
from ._export_ct import export_ct
from ._export_cst import export_cst, validate_roi_interpreted_types
from ._export_dose import export_dose, export_result
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
    per_beam_doses : bool, default False
        With a ``result`` that holds per-beam doses (``physical_dose_beam``, ...), also
        write one RTDOSE per beam as ``RTDOSE_<quantity>_beam<N>.dcm`` (Dose Summation
        Type ``BEAM``, 1-based beam numbers in STF beam order).

    Notes
    -----
    ``save(result=...)`` exports the dose quantities of a result dictionary: physical dose
    (``physical_dose``, Dose Type ``PHYSICAL``) and RBE-weighted dose (``rbe_x_dose``, Dose
    Type ``EFFECTIVE``) as ``RTDOSE_<quantity>.dcm``. All other quantities are skipped with a
    warning, and for multi-scenario results only the nominal (unsuffixed) scenario is written.
    With ``pln`` the Dose Summation Type follows ``pln.dose_convention`` (``"per_fraction"``
    gives ``FRACTION``, ``"total"`` gives ``PLAN``); without it, ``PLAN`` is written. The
    ``dose=`` image is written as ``RTDOSE.dcm`` (Dose Type ``PHYSICAL``). ``stf`` is
    accepted for the RT plan export and currently unused.
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
    per_beam_doses: bool = False
    #: Plan referenced by the exported RTDOSE files; set by the RT plan export.
    _plan_ref: Optional[PlanReference] = None

    def __init__(  # noqa: PLR0913 - exporter options
        self,
        path,
        structure_format: str = "rtstruct",
        metadata: Any = None,
        anonymize=False,
        overwrite: bool = False,
        roi_interpreted_types: Optional[dict[str, str]] = None,
        per_beam_doses: bool = False,
    ):
        super().__init__(path)
        self.per_beam_doses = bool(per_beam_doses)
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
        result: Optional[dict] = None,
        pln: Optional[Any] = None,
        stf: Optional[Any] = None,
        **extra,
    ) -> None:
        if ct is None and cst is None and dose is None and result is None:
            raise ValueError("Nothing to export: provide at least one of ct, cst, dose, result.")

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

        summation = "PLAN"
        if pln is not None and getattr(pln, "dose_convention", "total") == "per_fraction":
            summation = "FRACTION"

        if dose is not None:
            export_dose(dose, directory, ctx, summation_type=summation, plan_ref=self._plan_ref)

        if result is not None:
            written = export_result(
                result,
                directory,
                ctx,
                summation_type=summation,
                per_beam=self.per_beam_doses,
                plan_ref=self._plan_ref,
            )
            if not written and ct is None and cst is None and dose is None:
                raise ValueError("The result holds no dose quantity to export.")

"""DICOM exporter: writes CT, RTSTRUCT and RTDOSE into a folder."""

import os
import tempfile
import logging
import warnings
from typing import Any, ClassVar, Optional

import pydicom
import SimpleITK as sitk

from pyRadPlan.ct import CT
from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult, validate_result

from ..base import BaseExporter
from ._export_common import RT_STRUCT_STORAGE, PlanReference, resolve_export_context
from ._export_ct import export_ct
from ._export_cst import export_cst, validate_roi_interpreted_types
from ._export_dose import export_dose, export_result
from ._export_dvh import DvhExport
from ._export_photon_plan import export_photon_plan, is_photon_stf
from ._export_plan import export_ion_plan, is_ion_stf
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
        With a ``result`` that holds per-beam doses (``result.beam.physical_dose``, ...), also
        write one RTDOSE per beam as ``RTDOSE_<quantity>_beam<N>.dcm`` (Dose Summation
        Type ``BEAM``, 1-based beam numbers in STF beam order).
    export_plan : bool, default True
        With an ``stf`` (and spot weights for ions), also write the RT (Ion) Plan
        (``RTPLAN.dcm``) before the doses, which then reference it.
    dvh : bool, default False
        Add the RT DVH module to the RTDOSE files: one cumulative DVH per exported ROI
        (1000 bins from zero to the maximum dose, volumes in cm3), computed on the dose grid
        with the masks of the structures. Needs a ``cst`` exported as RTSTRUCT in the same
        call, which the DVHs reference; otherwise a warning is issued and no DVHs are written.

    Notes
    -----
    ``save(result=...)`` exports the dose quantities of a :class:`~pyRadPlan.result.PlanResult`
    (a mapping of quantities is validated into one): physical dose (``physical_dose``, Dose
    Type ``PHYSICAL``) and RBE-weighted dose (``rbe_x_dose``, Dose Type ``EFFECTIVE``) as
    ``RTDOSE_<quantity>.dcm``. All other quantities are skipped with a warning, and for
    multi-scenario results only the nominal scenario (``result.quantities``) is written.
    With ``pln`` the Dose Summation Type follows ``pln.dose_convention`` (``"per_fraction"``
    gives ``FRACTION``, ``"total"`` gives ``PLAN``); without it, ``PLAN`` is written. The
    ``dose=`` image is written as ``RTDOSE.dcm`` (Dose Type ``PHYSICAL``).

    With photon steering information, ``save`` writes an RT Plan ``RTPLAN.dcm`` (beams as
    above, weights optional) with jaws and multileaf collimators of field-based beams, or an
    open field around the bixels of bixel-based beams, which only holds the beam geometry and
    warns that it is not deliverable. There is no meterset, as photon weights have no
    calibration to monitor units. See :mod:`pyRadPlan.io.dicom._export_photon_plan`.

    With an ion ``stf`` and spot weights (``weights=`` or ``result.fluence``), ``save`` writes
    an RT Ion Plan ``RTPLAN.dcm`` (plan intent RESEARCH, unapproved) with one beam per STF
    beam (Beam Number = STF index + 1), using the isocenter of each beam and
    ``pln.num_of_fractions`` planned fractions (one, with a warning, without ``pln``). Weights
    are per fraction and written as numbers of particles, ``w * 1e6`` (Primary Dosimeter Unit
    ``NP``). The RTDOSE files of the same export reference the plan, and the plan references
    the RTSTRUCT written in the same call. Only Head First Supine is supported
    (:class:`NotImplementedError` otherwise). See :mod:`pyRadPlan.io.dicom._export_plan` for
    the geometry conventions.
    """

    format: ClassVar[str] = "dcm"
    name: ClassVar[str] = "DICOM Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".dcm",)
    container: ClassVar[bool] = False
    supports: ClassVar[frozenset[str]] = frozenset(
        {"ct", "cst", "dose", "result", "pln", "stf", "weights"}
    )

    #: Default used when the exporter is reached via the combined handler (whose
    #: cooperative __init__ chain does not pass through DicomExporter.__init__).
    structure_format: str = "rtstruct"
    metadata: Any = None
    anonymize: bool = False
    overwrite: bool = False
    roi_interpreted_types: Optional[dict[str, str]] = None
    per_beam_doses: bool = False
    export_plan: bool = True
    dvh: bool = False
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
        export_plan: bool = True,
        dvh: bool = False,
    ):
        super().__init__(path)
        self.export_plan = bool(export_plan)
        self.dvh = bool(dvh)
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
        result: Optional[PlanResult] = None,
        pln: Optional[Any] = None,
        stf: Optional[Any] = None,
        weights: Optional[Any] = None,
        **extra,
    ) -> None:
        plan = self.export_plan and stf is not None
        if ct is None and cst is None and dose is None and result is None and not plan:
            raise ValueError(
                "Nothing to export: provide at least one of ct, cst, dose, result, stf."
            )
        if result is not None:
            result = validate_result(result)

        preset = self._plan_ref
        try:
            self._save(ct, cst, dose, result, pln, stf, weights)
        finally:
            self._plan_ref = preset

    def _save(  # noqa: PLR0913 - mirrors save
        self, ct, cst, dose, result, pln, stf, weights
    ) -> None:
        directory = self.path
        ctx = resolve_export_context(ct, cst, self.metadata, self.anonymize)

        existing = find_dicom_files(directory)
        if existing and not self.overwrite:
            raise FileExistsError(
                f"Directory {str(directory)!r} already contains {len(existing)} DICOM file(s); "
                "pass overwrite=True to replace them."
            )
        writes_plan = self._validate_plan(stf, pln, weights, result, ctx)
        if ct is None and cst is None and dose is None and result is None and not writes_plan:
            raise ValueError(
                "Nothing to export: the steering information gives no RT plan "
                "(an ion plan needs spot weights)."
            )
        os.makedirs(directory, exist_ok=True)
        for path in existing:
            os.remove(path)

        # A CT (or the cst's reference CT) anchors the frame of reference.
        if ct is None and cst is not None:
            ct = cst.ct_image

        ct_info = {}
        struct_ref = None
        if ct is not None:
            ct_info = export_ct(ct, directory, ctx)

        if cst is not None:
            if self.structure_format == "seg":
                export_seg(cst, directory, ctx, ct_info)
            else:
                struct_path = export_cst(cst, directory, ctx, ct_info, self.roi_interpreted_types)
                struct_ref = (
                    RT_STRUCT_STORAGE,
                    str(
                        pydicom.dcmread(
                            struct_path, specific_tags=["SOPInstanceUID"]
                        ).SOPInstanceUID
                    ),
                )

        dvh_export = None
        if self.dvh:
            if cst is not None and struct_ref is not None:
                dvh_export = DvhExport(cst, struct_ref)
            else:
                warnings.warn(
                    "DVH export needs a structure set exported as RTSTRUCT in the same call; "
                    "skipping the DVHs.",
                    stacklevel=4,
                )

        summation = "PLAN"
        if pln is not None and getattr(pln, "dose_convention", "total") == "per_fraction":
            summation = "FRACTION"

        if self.export_plan and stf is not None:
            self._export_plan(stf, pln, weights, result, directory, ctx, struct_ref)

        if dose is not None:
            export_dose(
                dose,
                directory,
                ctx,
                summation_type=summation,
                plan_ref=self._plan_ref,
                dvh=dvh_export,
            )

        if result is not None:
            written = export_result(
                result,
                directory,
                ctx,
                summation_type=summation,
                per_beam=self.per_beam_doses,
                plan_ref=self._plan_ref,
                dvh=dvh_export,
                skipped_beams=self._plan_ref.skipped_beams if self._plan_ref else (),
            )
            if not written and ct is None and cst is None and dose is None:
                raise ValueError("The result holds no dose quantity to export.")

    def _validate_plan(self, stf, pln, weights, result, ctx) -> bool:
        """Run the plan export into a scratch directory so that errors surface before any write.

        Returns whether the export writes an RT plan.
        """
        if not (self.export_plan and stf is not None):
            return False
        preset = self._plan_ref
        self._plan_ref = None
        try:
            with tempfile.TemporaryDirectory() as scratch, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._export_plan(stf, pln, weights, result, scratch, ctx, None)
            return self._plan_ref is not None
        finally:
            self._plan_ref = preset

    def _export_plan(self, stf, pln, weights, result, directory, ctx, struct_ref) -> None:
        from pyRadPlan.stf import validate_stf  # noqa: PLC0415

        stf = validate_stf(stf)
        if weights is None and result is not None:
            weights = result.fluence
        if is_photon_stf(stf):
            # photon weights are optional: they only select beams and bixels
            self._plan_ref = export_photon_plan(
                stf, pln, weights, directory, ctx, structure_set_ref=struct_ref
            )
            return
        if not is_ion_stf(stf):
            warnings.warn(
                "The DICOM RT plan export supports ion and photon steering information only; "
                "skipping the plan export.",
                stacklevel=4,
            )
            return
        if weights is None:
            warnings.warn(
                "No spot weights (weights= or result.fluence) for the steering information; "
                "skipping the plan export.",
                stacklevel=4,
            )
            return
        self._plan_ref = export_ion_plan(
            stf, pln, weights, directory, ctx, structure_set_ref=struct_ref
        )

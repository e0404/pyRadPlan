"""NIfTI import/export backend (``.nii`` / ``.nii.gz``)."""

from typing import ClassVar

from ..base import SitkImporterBase, SitkExporterBase
from ..._factory import register_importer, register_exporter


class NiftiImporter(SitkImporterBase):
    """Importer for NIfTI images."""

    format: ClassVar[str] = "nifti"
    name: ClassVar[str] = "NIfTI Importer"
    extensions: ClassVar[tuple[str, ...]] = (".nii.gz", ".nii")


class NiftiExporter(SitkExporterBase):
    """Exporter for NIfTI images (NIfTI cannot store arbitrary header metadata)."""

    format: ClassVar[str] = "nifti"
    name: ClassVar[str] = "NIfTI Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".nii.gz", ".nii")
    stamp_interop_metadata: ClassVar[bool] = False


class NiftiHandler(NiftiImporter, NiftiExporter):
    """Combined NIfTI import/export handler."""

    name = "NIfTI Handler"


register_importer(NiftiImporter)
register_exporter(NiftiExporter)

__all__ = ["NiftiImporter", "NiftiExporter", "NiftiHandler"]

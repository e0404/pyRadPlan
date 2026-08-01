"""NRRD import/export backend (``.nrrd``)."""

from typing import ClassVar

from ..base import SitkImporterBase, SitkExporterBase
from ..._factory import register_importer, register_exporter


class NrrdImporter(SitkImporterBase):
    """Importer for NRRD images."""

    format: ClassVar[str] = "nrrd"
    name: ClassVar[str] = "NRRD Importer"
    extensions: ClassVar[tuple[str, ...]] = (".nrrd",)


class NrrdExporter(SitkExporterBase):
    """Exporter for NRRD images (stamps pyradplan/3D-Slicer metadata)."""

    format: ClassVar[str] = "nrrd"
    name: ClassVar[str] = "NRRD Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".nrrd",)
    stamp_interop_metadata: ClassVar[bool] = True


class NrrdHandler(NrrdImporter, NrrdExporter):
    """Combined NRRD import/export handler."""

    name = "NRRD Handler"


register_importer(NrrdImporter)
register_exporter(NrrdExporter)

__all__ = ["NrrdImporter", "NrrdExporter", "NrrdHandler"]

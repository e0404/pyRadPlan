"""MetaImage import/export backend (``.mha`` / ``.mhd``)."""

from typing import ClassVar

from ..base import SitkImporterBase, SitkExporterBase
from ..._factory import register_importer, register_exporter


class MetaImageImporter(SitkImporterBase):
    """Importer for MetaImage files."""

    format: ClassVar[str] = "meta"
    name: ClassVar[str] = "MetaImage Importer"
    extensions: ClassVar[tuple[str, ...]] = (".mha", ".mhd")


class MetaImageExporter(SitkExporterBase):
    """Exporter for MetaImage files (stamps pyradplan/3D-Slicer metadata)."""

    format: ClassVar[str] = "meta"
    name: ClassVar[str] = "MetaImage Exporter"
    extensions: ClassVar[tuple[str, ...]] = (".mha", ".mhd")
    stamp_interop_metadata: ClassVar[bool] = True


class MetaImageHandler(MetaImageImporter, MetaImageExporter):
    """Combined MetaImage import/export handler."""

    name = "MetaImage Handler"


register_importer(MetaImageImporter)
register_exporter(MetaImageExporter)

__all__ = ["MetaImageImporter", "MetaImageExporter", "MetaImageHandler"]

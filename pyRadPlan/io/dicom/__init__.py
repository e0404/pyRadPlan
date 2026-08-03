"""DICOM import/export backend (CT, RTSTRUCT, SEG, RTDOSE)."""

from .._factory import register_importer, register_exporter
from ._importer import DicomImporter
from ._exporter import DicomExporter
from ._handler import DicomHandler

register_importer(DicomImporter)
register_exporter(DicomExporter)

__all__ = ["DicomImporter", "DicomExporter", "DicomHandler"]

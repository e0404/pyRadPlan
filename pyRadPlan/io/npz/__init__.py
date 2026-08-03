"""NumPy ``.npz`` import/export backend (CT, StructureSet, dose)."""

from .._factory import register_importer, register_exporter
from ._importer import NpzImporter
from ._exporter import NpzExporter
from ._handler import NpzHandler

register_importer(NpzImporter)
register_exporter(NpzExporter)

__all__ = ["NpzImporter", "NpzExporter", "NpzHandler"]

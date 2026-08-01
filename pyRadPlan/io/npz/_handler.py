"""Combined import/export handler for the NumPy ``.npz`` backend."""

from ._importer import NpzImporter
from ._exporter import NpzExporter


class NpzHandler(NpzImporter, NpzExporter):
    """Low-level handler bundling :class:`NpzImporter` and :class:`NpzExporter`."""

    name = "NumPy Handler"

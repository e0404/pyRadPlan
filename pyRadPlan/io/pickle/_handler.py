"""Combined import/export handler for the pickle backend."""

from ._importer import PickleImporter
from ._exporter import PickleExporter


class PickleHandler(PickleImporter, PickleExporter):
    """Low-level handler bundling :class:`PickleImporter` and :class:`PickleExporter`."""

    name = "Pickle Handler"

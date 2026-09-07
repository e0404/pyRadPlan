"""Combined import/export handler for MATLAB ``.mat`` files."""

from ._importer import MatlabImporter
from ._exporter import MatlabExporter


class MatlabHandler(MatlabImporter, MatlabExporter):
    """Low-level handler bundling :class:`MatlabImporter` and :class:`MatlabExporter`."""

    name = "MATLAB Handler"

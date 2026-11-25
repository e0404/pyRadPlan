"""Combined import/export handler for the HDF5 backend."""

from ._importer import Hdf5Importer
from ._exporter import Hdf5Exporter


class Hdf5Handler(Hdf5Importer, Hdf5Exporter):
    """Low-level handler bundling :class:`Hdf5Importer` and :class:`Hdf5Exporter`."""

    name = "HDF5 Handler"

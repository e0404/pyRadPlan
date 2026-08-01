"""Pickle import/export backend.

Stores a dict of pyRadPlan objects (ct, cst, dose, and any extras) in a single
``.pkl`` file at full fidelity. Fast, but format-fragile across versions.

.. warning::
    Unpickling executes arbitrary code. Only load ``.pkl`` files from trusted sources.
"""

from .._factory import register_importer, register_exporter
from ._importer import PickleImporter
from ._exporter import PickleExporter
from ._handler import PickleHandler

register_importer(PickleImporter)
register_exporter(PickleExporter)

__all__ = ["PickleImporter", "PickleExporter", "PickleHandler"]

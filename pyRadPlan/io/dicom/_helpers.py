"""Helper utilities for DICOM structure import.

The structure-naming heuristics are format-neutral and shared with the binary
importer; they live in :mod:`pyRadPlan.io._helpers` and are re-exported here so
existing ``from ._helpers import ...`` imports keep working.
"""

from .._helpers import determine_structure_type, generate_colors

__all__ = ["determine_structure_type", "generate_colors"]

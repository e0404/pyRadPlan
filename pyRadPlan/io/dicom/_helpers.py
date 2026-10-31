"""Helper utilities for DICOM import.

The structure-naming heuristics are format-neutral and shared with the binary
importer; they live in :mod:`pyRadPlan.io._helpers` and are re-exported here so
existing ``from ._helpers import ...`` imports keep working.
"""

import warnings

import numpy as np

from .._helpers import determine_structure_type, generate_colors

__all__ = ["determine_structure_type", "generate_colors", "warn_if_not_equally_spaced"]


def warn_if_not_equally_spaced(positions, what: str, stacklevel: int = 3) -> None:
    """Warn when slice or frame positions (mm) are not equally spaced.

    SimpleITK and GDCM place every slice of a series (every frame of a multi-frame
    RTDOSE) on a uniform grid, so a missing slice or a change of slice thickness
    displaces the data in z without an error.
    """
    positions = np.sort(np.asarray(positions, dtype=float))
    if positions.size < 3:
        return
    spacing = (positions[-1] - positions[0]) / (positions.size - 1)
    displacement = np.abs(positions - (positions[0] + spacing * np.arange(positions.size)))
    steps = np.diff(positions)
    if np.max(np.abs(steps - spacing)) > 0.01 * abs(spacing):
        warnings.warn(
            f"{what} are not equally spaced (steps from {steps.min():g} to {steps.max():g} mm). "
            f"They are read onto a uniform grid of {spacing:g} mm, which displaces them by up "
            f"to {displacement.max():.3g} mm.",
            stacklevel=stacklevel,
        )

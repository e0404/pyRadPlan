"""Errors, constants and helpers shared by the parts of the planning state."""

from typing import Any

import numpy as np
import SimpleITK as sitk

_NO_DOSE = "No dose available yet. Call optimize first."
# Evaluated like a VOI but never part of the structure set: the patient outside all targets
NORMAL_TISSUE = "NormalTissue"


class PlanningError(Exception):
    """Errors meant to be shown to the model so it can correct itself."""


class UnknownVoiError(PlanningError):
    def __init__(self, name: str, valid: list[str]):
        super().__init__(f"Unknown VOI '{name}'. Valid VOIs: {valid}.")


def sitk_array(img: Any) -> np.ndarray:
    """Return a dose image (sitk) or array as numpy array (z, y, x)."""
    if isinstance(img, sitk.Image):
        return sitk.GetArrayFromImage(img)
    return np.asarray(img)

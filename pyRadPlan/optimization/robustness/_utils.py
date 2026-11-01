"""Helpers shared by the robustness operators."""

import numpy as np
import array_api_compat

from ...core import xp_utils
from ...core.xp_utils.typing import Array


def as_backend(array: np.ndarray, like: Array) -> Array:
    """Convert a host (numpy) array to the namespace, device and dtype of ``like``."""
    xp = array_api_compat.array_namespace(like)
    return xp.asarray(
        xp_utils.from_numpy(xp, np.asarray(array), device=array_api_compat.device(like)),
        dtype=like.dtype,
    )

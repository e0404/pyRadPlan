"""Helpers shared by the robustness operators."""

import numpy as np
import array_api_compat

from ...core import xp_utils
from ...core.xp_utils.typing import Array


def scenario_value(stacked, s: int) -> Array:
    """Values of scenario ``s``: a row of an ``(S, n)`` stack or an entry of a per-scenario list."""
    return stacked[s] if isinstance(stacked, (list, tuple)) else stacked[s, ...]


def stack_like(stacked, per_scenario: list) -> Array:
    """Arrange per-scenario arrays like ``stacked``: an ``(S, n)`` stack or a list."""
    if isinstance(stacked, (list, tuple)):
        return list(per_scenario)
    return array_api_compat.array_namespace(per_scenario[0]).stack(per_scenario)


def as_backend(array: np.ndarray, like: Array) -> Array:
    """Convert a host (numpy) array to the namespace, device and dtype of ``like``."""
    xp = array_api_compat.array_namespace(like)
    return xp.asarray(
        xp_utils.from_numpy(xp, np.asarray(array), device=array_api_compat.device(like)),
        dtype=like.dtype,
    )

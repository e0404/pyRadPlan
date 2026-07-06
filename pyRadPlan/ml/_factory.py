"""Listing of locally available pyRadPlan ML models.

:func:`list_local_models` reports the model repositories present under
:attr:`MlSettings.local_models_dir` (a folder holding the full model contract,
see :data:`pyRadPlan.ml._hub.REQUIRED_FILES`). It touches no network. A model's
:class:`Modality` is derived from its repository-name prefix (``dosecalc-*`` /
``outcome-*``).
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from ._hub import is_valid_model_dir
from ._settings import get_ml_settings

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported ML model modalities."""

    DOSE_CALC = "dose_calc"
    OUTCOME = "outcome"


def modality_from_name(name: str) -> Optional[Modality]:
    """Derive a :class:`Modality` from a repository/model name prefix."""
    low = name.lower()
    if low.startswith("dosecalc"):
        return Modality.DOSE_CALC
    if low.startswith("outcome"):
        return Modality.OUTCOME
    return None


def list_local_models(modality: Optional[Modality] = None) -> list[str]:
    """List names of models available locally, optionally filtered by modality.

    Scans :attr:`MlSettings.local_models_dir` for subdirectories that hold the
    full model contract. No network access is performed.

    Parameters
    ----------
    modality : Modality, optional
        If given, only return models of this modality.

    Returns
    -------
    list[str]
        Sorted model repository names available on disk.
    """
    cfg = get_ml_settings()
    base = cfg.local_models_dir
    names: list[str] = []
    if base is not None and Path(base).is_dir():
        for repo_dir in sorted(p for p in Path(base).iterdir() if p.is_dir()):
            if is_valid_model_dir(repo_dir):
                names.append(repo_dir.name)
    if modality is not None:
        names = [n for n in names if modality_from_name(n) == modality]
    return names

"""Listing of locally available pyRadPlan AI models.

:func:`list_local_models` reports the model repositories present on disk, both
under :attr:`AiModelSettings.local_models_dir` and in the HuggingFace cache (where
downloaded models land by default). A directory counts as a model when it holds
the full model contract (see :data:`pyRadPlan.ai_models._hub.REQUIRED_FILES`). It
touches no network. A model's :class:`Modality` is derived from its
repository-name prefix (``dosecalc-*`` / ``outcome-*``).
"""

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from ._hub import is_valid_model_dir
from ._settings import get_ai_model_settings

logger = logging.getLogger(__name__)


class Modality(str, Enum):
    """Supported AI model modalities."""

    DOSE_CALC = "dose_calc"
    OUTCOME = "outcome"


def modality_from_name(name: str) -> Optional[Modality]:
    """Derive a :class:`Modality` from a repository/model name prefix."""
    low = name.lower()
    if low.startswith("dosecalc") or low.startswith("pyradplan-dosecalc"):
        return Modality.DOSE_CALC
    if low.startswith("outcome") or low.startswith("pyradplan-outcome"):
        return Modality.OUTCOME
    return None


def list_local_models(modality: Optional[Modality] = None) -> list[str]:
    """List names of models available locally, optionally filtered by modality.

    Scans both :attr:`AiModelSettings.local_models_dir` and the HuggingFace cache for
    directories that hold the full model contract. No network access is
    performed.

    Parameters
    ----------
    modality : Modality, optional
        If given, only return models of this modality.

    Returns
    -------
    list[str]
        Sorted model repository names available on disk.
    """
    cfg = get_ai_model_settings()
    names: set[str] = set()

    base = cfg.local_models_dir
    if base is not None and Path(base).is_dir():
        for repo_dir in Path(base).iterdir():
            if repo_dir.is_dir() and is_valid_model_dir(repo_dir):
                names.add(repo_dir.name)

    names.update(_scan_hf_cache(cfg))

    result = sorted(names)
    if modality is not None:
        result = [n for n in result if modality_from_name(n) == modality]
    return result


def _scan_hf_cache(cfg) -> set[str]:
    """Model repository names present in the HuggingFace cache (no network)."""
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        info = scan_cache_dir(cache_dir=str(cfg.cache_dir) if cfg.cache_dir else None)
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("Could not scan HuggingFace cache: %s", exc)
        return set()

    names: set[str] = set()
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        if any(is_valid_model_dir(Path(rev.snapshot_path)) for rev in repo.revisions):
            names.add(repo.repo_id.split("/")[-1])
    return names

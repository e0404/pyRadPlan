"""Listing of locally available pyRadPlan AI models.

:func:`list_local_models` reports the model repositories present on disk, both
under :attr:`AiSettings.modelhub_local_models_dir` and in the HuggingFace cache (where
downloaded models land by default). A directory counts as a model when it holds
the full model contract (see :data:`pyRadPlan.ai.modelhub._resolve.REQUIRED_FILES`). It
touches no network.

Models are reported by their full ``"<org>/<repo>"`` id, so a private fork is
never confused with the upstream repository of the same name. A model's
:class:`ModelTask` -- what the model does, as opposed to the radiation modality
it was trained for -- is read from the ``metadata`` section of its
``model_config.json``, falling back to the repository-name prefix
(``dosecalc-*`` / ``outcome-*``) for repositories that do not declare one.
"""

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pyRadPlan._settings import get_settings

from ._resolve import is_valid_model_dir

logger = logging.getLogger(__name__)


class ModelTask(str, Enum):
    """What a model does.

    Not to be confused with the radiation modality (``radiation_mode`` on a
    :class:`~pyRadPlan.Plan`), which a model repository records separately under
    ``metadata.training.modality``.
    """

    DOSE_CALC = "dose_calc"
    OUTCOME = "outcome"


def _normalize_task(value: object) -> Optional[ModelTask]:
    """Map a declared task string onto a :class:`ModelTask`."""
    if isinstance(value, ModelTask):
        return value
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("dose_calc", "dosecalc", "dose_calculation"):
        return ModelTask.DOSE_CALC
    if key == "outcome":
        return ModelTask.OUTCOME
    return None


def task_from_name(name: str) -> Optional[ModelTask]:
    """Derive a :class:`ModelTask` from a repository/model name prefix.

    Parameters
    ----------
    name : str
        A model name or full ``"<org>/<repo>"`` id; only the repository part is
        inspected.

    Returns
    -------
    ModelTask or None
        The task, or ``None`` if the name carries no known prefix.
    """
    low = name.rsplit("/", 1)[-1].lower()
    if low.startswith("dosecalc") or low.startswith("pyradplan-dosecalc"):
        return ModelTask.DOSE_CALC
    if low.startswith("outcome") or low.startswith("pyradplan-outcome"):
        return ModelTask.OUTCOME
    return None


def task_from_dir(path: Path) -> Optional[ModelTask]:
    """Determine the task of a model directory.

    Reads ``metadata.task`` from the directory's ``model_config.json`` and falls
    back to :func:`task_from_name` on the directory name when the config
    declares nothing usable.

    Parameters
    ----------
    path : Path
        A model directory.

    Returns
    -------
    ModelTask or None
        The task, or ``None`` if it could be determined from neither.
    """
    config_file = Path(path) / "model_config.json"
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
        declared = config.get("metadata", {}).get("task")
    except (json.JSONDecodeError, OSError, AttributeError) as exc:
        logger.debug("Could not read task from %s: %s", config_file, exc)
        declared = None

    return _normalize_task(declared) or task_from_name(Path(path).name)


def list_local_models(task: Optional[ModelTask] = None) -> list[str]:
    """List models available locally, optionally filtered by task.

    Scans both :attr:`AiSettings.modelhub_local_models_dir` and the HuggingFace cache for
    directories that hold the full model contract. No network access is
    performed.

    Models are reported as ``"<org>/<repo>"``, which is what
    :func:`~pyRadPlan.ai.modelhub.load_model` accepts, so the returned names can
    be fed straight back in. A model placed by hand directly under
    ``local_models_dir`` (rather than in an ``<org>/`` subdirectory) has no
    organization to report and is listed under its bare folder name.

    Parameters
    ----------
    task : ModelTask, optional
        If given, only return models for this task.

    Returns
    -------
    list[str]
        Sorted model ids available on disk.
    """
    cfg = get_settings().ai
    found: dict[str, Optional[ModelTask]] = {}

    base = cfg.modelhub_local_models_dir
    if base is not None and Path(base).is_dir():
        found.update(_scan_local_dir(Path(base)))

    for name, model_task in _scan_hf_cache(cfg).items():
        found.setdefault(name, model_task)

    names = sorted(found)
    if task is not None:
        names = [n for n in names if found[n] == task]
    return names


def _scan_local_dir(base: Path) -> dict[str, Optional[ModelTask]]:
    """Model ids under a local models base directory.

    Expects the ``<base>/<org>/<repo>`` layout that downloads use, but also
    reports a model folder placed directly under ``base`` (under its bare name,
    since no organization is known).
    """
    found: dict[str, Optional[ModelTask]] = {}
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if is_valid_model_dir(entry):
            found[entry.name] = task_from_dir(entry)
            continue
        for repo_dir in sorted(entry.iterdir()):
            if repo_dir.is_dir() and is_valid_model_dir(repo_dir):
                found[f"{entry.name}/{repo_dir.name}"] = task_from_dir(repo_dir)
    return found


def _scan_hf_cache(cfg) -> dict[str, Optional[ModelTask]]:
    """Model repositories present in the HuggingFace cache (no network)."""
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        cache_dir = str(cfg.modelhub_cache_dir) if cfg.modelhub_cache_dir else None
        info = scan_cache_dir(cache_dir=cache_dir)
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("Could not scan HuggingFace cache: %s", exc)
        return {}

    found: dict[str, Optional[ModelTask]] = {}
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        for rev in repo.revisions:
            snapshot = Path(rev.snapshot_path)
            if is_valid_model_dir(snapshot):
                found.setdefault(repo.repo_id, task_from_dir(snapshot))
                break
    return found

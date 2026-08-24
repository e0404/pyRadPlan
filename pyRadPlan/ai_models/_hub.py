"""Thin wrappers around :mod:`huggingface_hub` for resolving model directories.

These helpers centralize three concerns the rest of the package relies on:

* **HuggingFace-first resolution** — when a ``repo_id`` is given and we're not
  offline, HuggingFace is checked first so a newer version there is always
  picked up, even when a local copy already exists under ``local_dir`` (e.g.
  ``pyRadPlan/data/ai_models``). A valid local copy is only used as a
  *fallback* when HuggingFace can't be reached or the repo doesn't exist there
  — which is what lets a purely local, never-published model keep working.
* **Version dedup** — when downloading into a ``local_dir``, a small
  ``.pyradplan_model.json`` records the resolved ``repo_id``/``revision`` so a
  pinned revision that's already present locally skips the network entirely.
* **Offline operation** — with :class:`AiModelSettings.offline` set (or
  ``offline=True``), no network is attempted at all; a valid local copy is
  used directly, otherwise the HuggingFace cache is checked
  (``local_files_only=True``).
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from pyRadPlan._settings import get_settings

logger = logging.getLogger(__name__)

#: Files a valid model directory must contain.
REQUIRED_FILES = (
    "model.py",
    "preprocessor.py",
    "weights.safetensors",
    "model_config.json",
)

#: Name of the version-metadata file written into a managed ``local_dir``.
METADATA_FILENAME = ".pyradplan_model.json"

_INSTALL_HINT = (
    "huggingface_hub is required to download models. "
    "Install the optional dependencies with: pip install pyRadPlan[ai]"
)


def _import_snapshot_download():
    """Import :func:`huggingface_hub.snapshot_download` lazily."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(_INSTALL_HINT) from exc
    return snapshot_download


def is_valid_model_dir(path: Union[str, Path]) -> bool:
    """Return whether ``path`` contains all required model files.

    Parameters
    ----------
    path : str or Path
        Candidate model directory.

    Returns
    -------
    bool
        ``True`` if every file in :data:`REQUIRED_FILES` is present.
    """
    path = Path(path)
    return path.is_dir() and all((path / name).is_file() for name in REQUIRED_FILES)


def _read_metadata(path: Path) -> dict:
    """Read the version metadata from a model directory (empty dict if absent)."""
    meta_file = path / METADATA_FILENAME
    if not meta_file.is_file():
        return {}
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read model metadata at %s: %s", meta_file, exc)
        return {}


def _write_metadata(path: Path, repo_id: str, revision: Optional[str]) -> None:
    """Write version metadata into a model directory."""
    meta = {
        "repo_id": repo_id,
        "revision": revision,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        (path / METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.warning("Could not write model metadata at %s: %s", path, exc)


def _metadata_matches(path: Path, repo_id: str, revision: Optional[str]) -> bool:
    """Return whether on-disk metadata matches the requested version.

    A match (and therefore a skipped download) only happens when an explicit
    ``revision`` is requested; with ``revision=None`` we cannot prove the local
    copy is up to date, so we defer to :func:`snapshot_download` (which performs
    its own ETag/commit-based caching).
    """
    if revision is None:
        return False
    meta = _read_metadata(path)
    return meta.get("repo_id") == repo_id and meta.get("revision") == revision


def _download(
    snapshot_download,
    repo_id: str,
    revision: Optional[str],
    cache_dir: Optional[Path],
    local_dir: Optional[Path],
    offline: bool,
    local_fallback: Optional[Path],
) -> Path:
    """Resolve a model from HuggingFace, checking it before any local fallback.

    When online, HuggingFace is always queried first — so a newer version there
    is picked up even if ``local_fallback`` (an already-valid local model
    directory) exists. ``local_fallback`` is only used if that check fails
    (network issue, repo not found, ...), which lets a purely local model that
    was never pushed to HuggingFace keep working. If neither succeeds, the
    HuggingFace cache is tried as a last resort (``local_files_only=True``).
    """
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_dir": str(local_dir) if local_dir else None,
    }

    def _snapshot(local_files_only: bool) -> Path:
        path = Path(snapshot_download(local_files_only=local_files_only, **kwargs))
        if local_dir is not None:
            _write_metadata(path, repo_id, revision)
        return path

    if offline:
        if local_fallback is not None:
            return local_fallback
        try:
            return _snapshot(local_files_only=True)
        except Exception as exc:
            raise FileNotFoundError(
                f"Model '{repo_id}' is not available locally and offline mode is enabled."
            ) from exc

    try:
        return _snapshot(local_files_only=False)
    except Exception as exc:
        if local_fallback is not None:
            logger.info(
                "Could not reach HuggingFace for '%s' (%s); using local copy at %s.",
                repo_id,
                exc,
                local_fallback,
            )
            return local_fallback
        logger.warning(
            "Could not download model '%s' (%s). Falling back to cached files.",
            repo_id,
            exc,
        )
        try:
            return _snapshot(local_files_only=True)
        except Exception as exc2:
            raise FileNotFoundError(
                f"Could not download model '{repo_id}' and no cached copy was found."
            ) from exc2


def resolve_model_dir(
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a model directory, preferring HuggingFace over any local copy.

    When ``repo_id`` is given and we're not offline, HuggingFace is always checked
    first — even if a valid model already sits at ``local_dir`` — so a newer
    version there is picked up automatically. The local copy is only used as a
    fallback when that check fails (offline, network issue, or the repo doesn't
    exist on HuggingFace at all), which is what lets a custom model that was
    never pushed to HuggingFace keep working from ``local_dir`` alone.

    The one exception is a *pinned* ``revision`` that's already present locally
    (recorded via the version-dedup metadata): since that exact revision can't
    change, it's used directly without touching the network.

    Parameters
    ----------
    repo_id : str, optional
        HuggingFace repository id (``"<org>/<repo>"``). Required unless a valid
        ``local_dir`` is given.
    revision : str, optional
        Git revision (tag, branch or commit) to download. Used for version
        dedup against ``local_dir`` metadata.
    local_dir : str or Path, optional
        A directory to use as a fallback (if it already holds the model) or to
        download into. Enables offline use and version tracking. When omitted
        and :attr:`AiModelSettings.local_models_dir` (``PYRADPLAN_HUGGINGFACE_PATH``)
        is set, it defaults to ``"<local_models_dir>/<repo>"``.
    offline : bool, optional
        Force offline mode. Defaults to :attr:`AiModelSettings.offline`.
    cache_dir : str or Path, optional
        Override the HuggingFace cache directory. Defaults to
        :attr:`AiModelSettings.cache_dir`.

    Returns
    -------
    Path
        The resolved model directory.

    Raises
    ------
    ValueError
        If neither a usable ``local_dir`` nor a ``repo_id`` is provided.
    FileNotFoundError
        If the model can be neither found locally nor downloaded.
    """
    cfg = get_settings().ai_models
    if offline is None:
        offline = cfg.offline
    if cache_dir is None:
        cache_dir = cfg.cache_dir
    cache_dir = Path(cache_dir) if cache_dir else None

    if local_dir is None and cfg.local_models_dir is not None and repo_id is not None:
        local_dir = cfg.local_models_dir / repo_id.split("/")[-1]
        logger.info("Using configured local models base, local_dir=%s.", local_dir)

    target = Path(local_dir) if local_dir is not None else None
    local_valid = target is not None and is_valid_model_dir(target)

    if repo_id is None:
        if local_valid:
            logger.info("Using existing local model at %s.", target)
            return target
        if target is not None:
            raise FileNotFoundError(
                f"Local model directory '{target}' is missing required files "
                f"{REQUIRED_FILES} and no repo_id was given to download it."
            )
        raise ValueError(
            "Provide either a valid 'local_dir' containing the model files or a 'repo_id'."
        )

    if local_valid and revision is not None and _metadata_matches(target, repo_id, revision):
        logger.info("Using existing local model at %s (pinned revision already present).", target)
        return target

    snapshot_download = _import_snapshot_download()

    logger.info("Resolving model '%s' (revision=%s, offline=%s).", repo_id, revision, offline)
    return _download(
        snapshot_download,
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_dir=target,
        offline=offline,
        local_fallback=target if local_valid else None,
    )

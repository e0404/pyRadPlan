"""Thin wrappers around :mod:`huggingface_hub` for resolving model directories.

These helpers centralize three concerns the rest of the package relies on:

* **Local directories** — a model may simply live on disk (e.g. shipped under
  ``pyRadPlan/data/ai_models``); no network is touched.
* **Version dedup** — when downloading into a ``local_dir``, a small
  ``.pyradplan_model.json`` records the resolved ``repo_id``/``revision`` so a
  pinned version is not downloaded twice. Unpinned requests are handed to
  :func:`~huggingface_hub.snapshot_download`, whose own ETag/commit check keeps
  the copy current.
* **Offline operation** — downloads transparently fall back to cached/local
  files when the network is unavailable or :attr:`AiSettings.modelhub_offline` is set.
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
    "huggingface_hub is required to download models. It ships with pyRadPlan; "
    "restore it with: pip install huggingface_hub"
)


def _import_snapshot_download():
    """Import :func:`huggingface_hub.snapshot_download` lazily."""
    try:
        from huggingface_hub import snapshot_download  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised via error path
        raise ImportError(_INSTALL_HINT) from exc
    return snapshot_download


def repo_subpath(repo_id: str) -> Path:
    """Relative on-disk path for a repository id: ``"<org>/<repo>"``.

    Keeping the organization in the path is what stops a private fork and its
    upstream namesake from sharing (and clobbering) one directory.

    Parameters
    ----------
    repo_id : str
        HuggingFace repository id.

    Returns
    -------
    Path
        The relative path, one directory per ``/``-separated segment.

    Raises
    ------
    ValueError
        If a segment is empty or would escape the base directory.
    """
    parts = repo_id.split("/")
    if not all(part and part not in (".", "..") for part in parts):
        raise ValueError(f"Invalid repository id '{repo_id}'.")
    return Path(*parts)


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


def _download(  # noqa: PLR0913 - a thin passthrough to snapshot_download's own signature
    snapshot_download,
    repo_id: str,
    revision: Optional[str],
    cache_dir: Optional[Path],
    local_dir: Optional[Path],
    offline: bool,
) -> Path:
    """Run ``snapshot_download`` with an offline fallback."""
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_dir": str(local_dir) if local_dir else None,
    }
    try:
        return Path(snapshot_download(local_files_only=offline, **kwargs))
    except Exception as exc:
        if offline:
            raise FileNotFoundError(
                f"Model '{repo_id}' is not available locally and offline mode is enabled."
            ) from exc
        logger.warning(
            "Could not download model '%s' (%s). Falling back to cached files.",
            repo_id,
            exc,
        )
        try:
            return Path(snapshot_download(local_files_only=True, **kwargs))
        except Exception as exc2:
            raise FileNotFoundError(
                f"Could not download model '{repo_id}' ({exc}) and no cached copy was found "
                f"({exc2})."
            ) from exc2


def resolve_model_dir(
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Resolve a model directory, downloading from HuggingFace if necessary.

    Parameters
    ----------
    repo_id : str, optional
        HuggingFace repository id (``"<org>/<repo>"``). Required unless a valid
        ``local_dir`` is given.
    revision : str, optional
        Git revision (tag, branch or commit) to download. Pinning one enables
        version dedup against ``local_dir`` metadata, so a matching local copy
        is used without touching the network. Without it the hub is asked
        whether the local copy is current.
    local_dir : str or Path, optional
        A directory to use directly (if it already holds the model) or to
        download into. Enables offline use and version tracking. When omitted
        and :attr:`AiSettings.modelhub_local_models_dir`
        (``PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR``) is set, it defaults to
        ``"<local_models_dir>/<org>/<repo>"``.
    offline : bool, optional
        Force offline mode. Defaults to :attr:`AiSettings.modelhub_offline`.
    cache_dir : str or Path, optional
        Override the HuggingFace cache directory. Defaults to
        :attr:`AiSettings.modelhub_cache_dir`.

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
    cfg = get_settings().ai
    if offline is None:
        offline = cfg.modelhub_offline
    if cache_dir is None:
        cache_dir = cfg.modelhub_cache_dir
    cache_dir = Path(cache_dir) if cache_dir else None

    if local_dir is None and cfg.modelhub_local_models_dir is not None and repo_id is not None:
        local_dir = Path(cfg.modelhub_local_models_dir) / repo_subpath(repo_id)
        logger.info("Using configured local models base, local_dir=%s.", local_dir)

    target: Optional[Path] = None
    if local_dir is not None:
        local_dir = Path(local_dir)
        target = local_dir

        # A present local copy is used as-is when there is nothing to compare it
        # against (no repo_id), when the network is off limits, or when it
        # already carries the pinned revision. An unpinned request falls through
        # to snapshot_download, which does its own freshness check.
        if is_valid_model_dir(target) and (
            repo_id is None or offline or _metadata_matches(target, repo_id, revision)
        ):
            logger.info("Using existing local model at %s.", target)
            return target

        if repo_id is None:
            raise FileNotFoundError(
                f"Local model directory '{target}' is missing required files "
                f"{REQUIRED_FILES} and no repo_id was given to download it."
            )

    if repo_id is None:
        raise ValueError(
            "Provide either a valid 'local_dir' containing the model files or a 'repo_id'."
        )

    snapshot_download = _import_snapshot_download()

    logger.info("Resolving model '%s' (revision=%s, offline=%s).", repo_id, revision, offline)
    try:
        snapshot = _download(
            snapshot_download,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=target,
            offline=offline,
        )
    except FileNotFoundError:
        # A copy placed by hand (rather than by huggingface_hub) carries no
        # download metadata, so local_files_only cannot see it. Prefer a stale
        # local model over failing outright.
        if target is not None and is_valid_model_dir(target):
            logger.warning(
                "Could not reach the hub for '%s'; using the local copy at %s.", repo_id, target
            )
            return target
        raise

    if target is not None:
        _write_metadata(snapshot, repo_id, revision)

    return snapshot

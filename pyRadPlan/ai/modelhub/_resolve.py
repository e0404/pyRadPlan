"""Thin wrappers around :mod:`huggingface_hub` for resolving model directories.

These helpers centralize four concerns the rest of the package relies on:

* **Local directories** — a model may simply live on disk (e.g. shipped under
  ``pyRadPlan/data/ai_models``); no network is touched.
* **Provenance** — a folder pyRadPlan downloads into carries a
  ``.pyradplan_model.json`` marker, written *before* the download starts, so a
  folder only counts as placed by the user when pyRadPlan did not fill it (no
  marker, not inside the HuggingFace cache). Such a folder is used as-is and
  never refreshed from the Hub; pyRadPlan downloads only into an empty folder
  or one that already carries the marker.
* **Version dedup** — the marker records the resolved ``repo_id``/``revision``
  so a pinned version is not downloaded twice. Unpinned requests are handed to
  :func:`~huggingface_hub.snapshot_download`, whose own ETag/commit check keeps
  the copy current.
* **Offline operation** — downloads transparently fall back to cached/local
  files when the network is unavailable or :attr:`AiSettings.modelhub_offline` is set.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional, Union

from pyRadPlan._settings import get_settings

logger = logging.getLogger(__name__)

#: Files a valid model directory must contain.
REQUIRED_FILES = (
    "model.py",
    "preprocessor.py",
    "weights.safetensors",
    "model_config.json",
)

#: Name of the marker written into a ``local_dir`` pyRadPlan downloads into.
METADATA_FILENAME = ".pyradplan_model.json"

_DOWNLOADING = "downloading"
_COMPLETE = "complete"

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
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        logger.warning("Could not read model metadata at %s: %s", meta_file, exc)
        return {}
    return meta if isinstance(meta, dict) else {}


def _write_metadata(path: Path, repo_id: str, revision: Optional[str], state: str) -> None:
    """Write the download marker into a model directory.

    Raises
    ------
    OSError
        If the marker cannot be written. A download must not start without it,
        or the folder it fills would afterwards pass as placed by the user.
    """
    meta = {
        "repo_id": repo_id,
        "revision": revision,
        "state": state,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    path.mkdir(parents=True, exist_ok=True)
    (path / METADATA_FILENAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _metadata_matches(path: Path, repo_id: str, revision: Optional[str]) -> bool:
    """Return whether on-disk metadata matches the requested version.

    A match (and therefore a skipped download) only happens when an explicit
    ``revision`` is requested and its download completed; with ``revision=None``
    we cannot prove the local copy is up to date, so we defer to
    :func:`snapshot_download` (which performs its own ETag/commit-based caching).
    A marker without ``state`` predates download tracking and was only ever
    written after a completed download.
    """

    if revision is None:
        return False
    meta = _read_metadata(path)
    return (
        meta.get("repo_id") == repo_id
        and meta.get("revision") == revision
        and meta.get("state", _COMPLETE) == _COMPLETE
    )


def _hf_cache_roots() -> list[Path]:
    """List the HuggingFace cache directories pyRadPlan may resolve models from."""
    roots = []
    cache_dir = get_settings().ai.modelhub_cache_dir
    if cache_dir is not None:
        roots.append(Path(cache_dir))
    try:
        from huggingface_hub import constants  # noqa: PLC0415

        roots.append(Path(constants.HF_HUB_CACHE))
    except Exception as exc:  # noqa: BLE001 - no hub client means no default cache
        logger.debug("Could not determine the default HuggingFace cache: %s", exc)
    return roots


def _hf_cache_repo_id(path: Path) -> Optional[str]:
    """Return the repository id of the HuggingFace cache snapshot ``path`` lies in, if any.

    Recognised by the cache layout ``models--<org>--<repo>/snapshots/<revision>``
    after resolving symlinks, so a cache at any location counts.
    """
    resolved = path.resolve()
    for folder in (resolved, *resolved.parents):
        repo_folder = folder.parent.parent
        if folder.parent.name == "snapshots" and repo_folder.name.startswith("models--"):
            org, _, repo = repo_folder.name[len("models--") :].partition("--")
            return f"{org}/{repo}" if org and repo else None
    return None


def _in_hf_cache(path: Path) -> bool:
    """Return whether ``path`` lies inside a HuggingFace cache (symlinks resolved)."""
    if _hf_cache_repo_id(path) is not None:
        return True
    resolved = path.resolve()
    return any(resolved.is_relative_to(root.resolve()) for root in _hf_cache_roots())


FolderKind = Literal["missing", "empty", "marked", "cache", "user", "foreign"]


def _classify(path: Path) -> FolderKind:
    """Classify a model folder by where its content came from.

    ``"marked"`` and ``"cache"`` folders were filled by a download, ``"user"``
    folders hold a complete model pyRadPlan did not download, and ``"foreign"``
    ones hold something else, which pyRadPlan neither loads nor downloads over.
    """
    if not path.exists():
        return "missing"
    if _in_hf_cache(path):
        return "cache"
    if not path.is_dir():
        return "foreign"
    # exists() rather than is_file(): whatever occupies the name still marks a download
    if (path / METADATA_FILENAME).exists():
        return "marked"
    if not any(path.iterdir()):
        return "empty"
    if is_valid_model_dir(path):
        return "user"
    return "foreign"


@dataclass(frozen=True)
class ResolvedModel:
    """A resolved model folder and where its content came from.

    Attributes
    ----------
    path : Path
        The model directory.
    downloaded : bool
        Whether a download filled the folder (pyRadPlan's, or the HuggingFace
        cache), as opposed to a folder the user placed.
    repo_id : str or None
        The repository the content belongs to: the requested id, else the one the
        download marker or the cache layout records.
    """

    path: Path
    downloaded: bool
    repo_id: Optional[str]


def _foreign_folder_error(path: Path) -> FileNotFoundError:
    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    return FileNotFoundError(
        f"'{path}' is neither a complete model (missing {missing}) nor a download managed "
        f"by pyRadPlan (no {METADATA_FILENAME}), so pyRadPlan neither loads it nor "
        "downloads into it. Complete the model files, or empty the folder to have the "
        "model downloaded into it."
    )


def _download(  # noqa: PLR0913 - a thin passthrough to snapshot_download's own signature
    snapshot_download,
    repo_id: str,
    revision: Optional[str],
    cache_dir: Optional[Path],
    local_dir: Optional[Path],
    offline: bool,
) -> tuple[Path, bool]:
    """Run ``snapshot_download`` with an offline fallback.

    Returns the snapshot path and whether it was fetched from the Hub (``False``
    for cached or local files).
    """
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir) if cache_dir else None,
        "local_dir": str(local_dir) if local_dir else None,
    }
    try:
        return Path(snapshot_download(local_files_only=offline, **kwargs)), not offline
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
            return Path(snapshot_download(local_files_only=True, **kwargs)), False
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

    Never imports or executes the model's code.

    Parameters
    ----------
    repo_id : str, optional
        HuggingFace repository id (``"<org>/<repo>"``). Required unless a valid
        ``local_dir`` is given.
    revision : str, optional
        Git revision (tag, branch or commit) to download. Pinning one enables
        version dedup against ``local_dir`` metadata, so a matching local copy
        is used without touching the network. Without it the hub is asked
        whether a downloaded local copy is current.
    local_dir : str or Path, optional
        A directory to use directly (if it already holds the model) or to
        download into. Enables offline use and version tracking. When omitted
        and :attr:`AiSettings.modelhub_local_models_dir`
        (``PYRADPLAN_AI_MODELHUB_LOCAL_MODELS_DIR``) is set, it defaults to
        ``"<local_models_dir>/<org>/<repo>"``. A complete model pyRadPlan did not
        download (no ``.pyradplan_model.json`` marker, not in the HuggingFace
        cache) is used as-is and never refreshed from the Hub. pyRadPlan
        downloads only into a missing or empty folder or one carrying the marker,
        which it writes before the download starts.
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
        If the model can be neither found locally nor downloaded, or ``local_dir``
        holds something that is neither a complete model nor a managed download.
    """
    return _resolve_model(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        offline=offline,
        cache_dir=cache_dir,
    ).path


def _resolve_model(  # noqa: C901, PLR0912 - one decision per kind of folder
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    cache_dir: Optional[Union[str, Path]] = None,
) -> ResolvedModel:
    """Resolve a model directory like :func:`resolve_model_dir`, with its provenance."""
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
    kind: Optional[FolderKind] = None
    if local_dir is not None:
        target = Path(local_dir)
        kind = _classify(target)

        if kind == "user":
            logger.info("Using the model placed at %s as-is.", target)
            return ResolvedModel(target, downloaded=False, repo_id=repo_id)
        if kind == "foreign":
            raise _foreign_folder_error(target)
        if kind == "cache":
            if not is_valid_model_dir(target):
                raise FileNotFoundError(
                    f"'{target}' in the HuggingFace cache is missing required files "
                    f"{REQUIRED_FILES}."
                )
            return ResolvedModel(
                target, downloaded=True, repo_id=repo_id or _hf_cache_repo_id(target)
            )

        # A downloaded copy is used as-is when there is nothing to compare it
        # against (no repo_id), when the network is off limits, or when it
        # already carries the pinned revision. An unpinned request falls through
        # to snapshot_download, which does its own freshness check.
        if (
            kind == "marked"
            and is_valid_model_dir(target)
            and (repo_id is None or offline or _metadata_matches(target, repo_id, revision))
        ):
            logger.info("Using existing local model at %s.", target)
            return ResolvedModel(
                target, downloaded=True, repo_id=repo_id or _read_metadata(target).get("repo_id")
            )

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

    # Marked before the first byte arrives, so an interrupted download can never
    # pass as a folder the user placed. Offline, snapshot_download writes nothing.
    if target is not None and not offline:
        _write_metadata(target, repo_id, revision, _DOWNLOADING)

    logger.info("Resolving model '%s' (revision=%s, offline=%s).", repo_id, revision, offline)
    try:
        snapshot, fresh = _download(
            snapshot_download,
            repo_id=repo_id,
            revision=revision,
            cache_dir=cache_dir,
            local_dir=target,
            offline=offline,
        )
    except FileNotFoundError:
        if kind == "marked" and is_valid_model_dir(target):
            logger.warning(
                "Could not reach the hub for '%s'; using the local copy at %s.", repo_id, target
            )
            return ResolvedModel(target, downloaded=True, repo_id=repo_id)
        _discard_empty_marker(target, kind)
        raise

    if not is_valid_model_dir(snapshot):
        _discard_empty_marker(target, kind)
        raise FileNotFoundError(
            f"The copy of model '{repo_id}' at '{snapshot}' is missing required files "
            f"{REQUIRED_FILES}; it was not (fully) downloaded."
        )

    if target is not None and fresh:
        _write_metadata(target, repo_id, revision, _COMPLETE)

    return ResolvedModel(snapshot, downloaded=True, repo_id=repo_id)


def _discard_empty_marker(target: Optional[Path], kind: Optional[FolderKind]) -> None:
    """Remove the marker of a download into a fresh folder that fetched nothing."""
    if target is None or kind not in ("missing", "empty"):
        return
    try:
        if [entry.name for entry in target.iterdir()] == [METADATA_FILENAME]:
            (target / METADATA_FILENAME).unlink()
            if kind == "missing":
                target.rmdir()
    except OSError as exc:  # pragma: no cover - filesystem dependent
        logger.debug("Could not clean up %s: %s", target, exc)

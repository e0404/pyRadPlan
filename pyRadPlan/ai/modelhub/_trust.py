"""Trust records for the code of downloaded models.

Loading a model executes the Python shipped with it. A folder the user placed is
under the user's control and runs as-is; code pyRadPlan downloaded runs only if
the user trusts it, either globally (:attr:`AiSettings.modelhub_trust_remote_code`)
or for this very code: :func:`trust_model` records a hash of the model's code in
``<data_dir>/ai_trusted_models.json`` (see :func:`pyRadPlan.core.get_data_dir`),
keyed by repository id. When the code on disk changes, for instance because an
unpinned load pulled an update from the Hub, the hash no longer matches and the
model is refused until it is trusted again.
"""

import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Optional, Union

from pyRadPlan._settings import get_settings

from ._resolve import (
    METADATA_FILENAME,
    REQUIRED_FILES,
    ResolvedModel,
    _resolve_model,
    repo_subpath,
)

logger = logging.getLogger(__name__)

#: Name of the trust-record file in the pyRadPlan data directory.
TRUST_FILENAME = "ai_trusted_models.json"

_SETTING_HINT = (
    "pyRadPlan.settings.ai.modelhub_trust_remote_code = True "
    "(environment variable PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1)"
)


class UntrustedModelError(ValueError):
    """Raised when loading a model would execute downloaded code nobody trusted."""


class TrustedModel(NamedTuple):
    """A model whose code was trusted.

    Attributes
    ----------
    model_dir : Path
        The model directory the hash was taken of.
    code_hash : str
        The SHA-256 hash of the model's code.
    """

    model_dir: Path
    code_hash: str


def _hashed_files(model_dir: Path) -> list[tuple[str, Path]]:
    """List the files of a model folder that the code hash covers, sorted by name.

    Everything Python could import or a model could execute is covered: ``model.py``,
    ``preprocessor.py`` and any sibling module or package they import (the loader puts
    the folder on the import path), ``__pycache__`` (an unchecked ``.pyc`` runs instead
    of its source), native extensions, pickles and ``model_config.json``, which picks
    the class to run. ``*.safetensors`` weights are left out: the format holds tensors
    only and cannot carry code. So are pyRadPlan's own download marker and the
    ``.cache`` folder ``huggingface_hub`` keeps in a download folder.
    """
    root = model_dir.resolve()
    files = []
    visited = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)
        rel_dir = Path(dirpath).relative_to(root)
        if rel_dir == Path("."):
            dirnames[:] = [d for d in dirnames if d != ".cache"]
        for name in filenames:
            rel = (rel_dir / name).as_posix()
            if rel == METADATA_FILENAME or name.endswith(".safetensors"):
                continue
            files.append((rel, Path(dirpath) / name))
    return sorted(files)


def code_hash(model_dir: Union[str, Path]) -> str:
    """Hash the code of a model folder.

    Parameters
    ----------
    model_dir : str or Path
        A model directory.

    Returns
    -------
    str
        The hex SHA-256 over the names and contents of every file but the
        ``*.safetensors`` weights (which hold tensors only and cannot carry code),
        pyRadPlan's download marker and the ``huggingface_hub`` ``.cache`` folder.
    """
    digest = hashlib.sha256()
    for rel, path in _hashed_files(Path(model_dir)):
        file_digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                file_digest.update(chunk)
        digest.update(rel.encode("utf-8") + b"\0" + file_digest.digest())
    return digest.hexdigest()


def _trust_file() -> Path:
    from pyRadPlan.core import get_data_dir  # noqa: PLC0415 - pyRadPlan.core is heavy

    return get_data_dir() / TRUST_FILENAME


def _read_records() -> dict:
    """Read the trust records, treating a missing or unreadable file as empty."""
    path = _trust_file()
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring the unreadable model trust records in %s: %s", path, exc)
        return {}
    if not isinstance(records, dict):
        logger.warning("Ignoring the malformed model trust records in %s.", path)
        return {}
    return {
        repo_id: entry
        for repo_id, entry in records.items()
        if isinstance(entry, dict) and isinstance(entry.get("hashes"), dict)
    }


def _write_records(records: dict) -> None:
    """Replace the trust-record file atomically."""
    path = _trust_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{TRUST_FILENAME}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(records, handle, indent=2, sort_keys=True)
        # On Windows the replace fails while another process has the file open.
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _record(repo_id: str, digest: str, model_dir: Path) -> None:
    records = _read_records()
    hashes = records.setdefault(repo_id, {"hashes": {}})["hashes"]
    if digest in hashes:
        return
    hashes[digest] = {
        "trusted_at": datetime.now(timezone.utc).isoformat(),
        "model_dir": str(model_dir),
    }
    _write_records(records)
    logger.info("Recorded trust in the code of '%s' (%s).", repo_id, digest[:12])


def _repo_id(name: str) -> str:
    """Turn a bare model name into ``"<hf_org>/<name>"``; full ids are kept."""
    return name if "/" in name else f"{get_settings().ai.modelhub_hf_org}/{name}"


def _trust_call(repo_id: str, model_dir: Path) -> str:
    """Spell out the ``trust_model`` call that trusts the copy in ``model_dir``."""
    base = get_settings().ai.modelhub_local_models_dir
    if base is None or model_dir == Path(base) / repo_subpath(repo_id):
        return f'pyRadPlan.ai.modelhub.trust_model("{repo_id}")'
    return f'pyRadPlan.ai.modelhub.trust_model("{repo_id}", local_dir=r"{model_dir}")'


def _check_trust(resolved: ResolvedModel, trust_remote_code: bool) -> None:
    """Refuse to run a downloaded model's code unless it is trusted.

    Never imports anything from the model folder. With ``trust_remote_code`` the
    code is trusted as it is and recorded, so it keeps working after the opt-in is
    switched off, until it changes.

    Raises
    ------
    UntrustedModelError
        If the code is neither trusted globally nor recorded as trusted.
    """
    if not resolved.downloaded:
        return
    repo_id = resolved.repo_id
    digest = code_hash(resolved.path)
    if trust_remote_code:
        if repo_id is not None:
            try:
                _record(repo_id, digest, resolved.path)
            except OSError as exc:
                logger.warning("Could not record trust in the code of '%s': %s", repo_id, exc)
        return

    entry = _read_records().get(repo_id) if repo_id is not None else None
    if entry is not None and digest in entry["hashes"]:
        return

    if repo_id is None:
        raise UntrustedModelError(
            f"The model in '{resolved.path}' was downloaded, and loading it executes the "
            "code shipped with it. Load it by its '<org>/<model>' id after calling "
            'pyRadPlan.ai.modelhub.trust_model("<org>/<model>", local_dir=...), or '
            f"trust all downloaded models with {_SETTING_HINT}."
        )
    trust_call = _trust_call(repo_id, resolved.path)
    if entry is not None:
        raise UntrustedModelError(
            f"The code of model '{repo_id}' in '{resolved.path}' changed since you trusted "
            "it (e.g. an update pulled from the Hub). Inspect the code, then call "
            f"{trust_call} again, or trust all downloaded models with {_SETTING_HINT}."
        )
    raise UntrustedModelError(
        f"Model '{repo_id}' was downloaded from the Hub, and loading it executes the code "
        f"shipped with it ({', '.join(REQUIRED_FILES[:2])}). Inspect the code in "
        f"'{resolved.path}' and, if you trust it, call {trust_call}, or trust all "
        f"downloaded models with {_SETTING_HINT}."
    )


def trust_model(
    name: str,
    *,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
) -> TrustedModel:
    """Trust the code of a model as it is now, without running it.

    Resolves the model like :func:`load_model` does -- downloading it if needed --
    but imports nothing. A hash of its code -- every file but the ``*.safetensors``
    weights, which cannot carry code -- is recorded in
    ``<data_dir>/ai_trusted_models.json``, after which :func:`load_model` and the
    ``"AIBeamlet"`` dose engine run this model without
    :attr:`AiSettings.modelhub_trust_remote_code`. Once its code changes, e.g. when an
    unpinned load pulls an update, it is refused until it is trusted again.

    Inspect the code in the returned directory before relying on it. A model folder
    you placed yourself needs no trust; nothing is recorded for it.

    Parameters
    ----------
    name : str
        Model name; a bare name is resolved to ``"<hf_org>/<name>"`` like in
        :func:`load_model`.
    revision : str, optional
        Git revision (tag/branch/commit) to trust.
    local_dir : str or Path, optional
        Model directory, as in :func:`load_model`.
    offline : bool, optional
        Force offline mode. Defaults to :attr:`AiSettings.modelhub_offline`.

    Returns
    -------
    TrustedModel
        The model directory and the hash of the code that is now trusted.

    Raises
    ------
    FileNotFoundError
        If the model can be neither found locally nor downloaded.
    """
    repo_id = _repo_id(name)
    resolved = _resolve_model(
        repo_id=repo_id, revision=revision, local_dir=local_dir, offline=offline
    )
    digest = code_hash(resolved.path)
    if resolved.downloaded:
        _record(repo_id, digest, resolved.path)
    else:
        logger.info("'%s' is a folder you placed; it needs no trust record.", resolved.path)
    return TrustedModel(resolved.path, digest)


def untrust_model(name: str) -> bool:
    """Forget every trust record of a model.

    Parameters
    ----------
    name : str
        Model name or ``"<org>/<model>"`` id, as passed to :func:`trust_model`.

    Returns
    -------
    bool
        Whether there was a record to remove.
    """
    repo_id = _repo_id(name)
    records = _read_records()
    if records.pop(repo_id, None) is None:
        return False
    _write_records(records)
    return True


def is_trusted(
    name: str,
    *,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
) -> bool:
    """Return whether the copy of a model on disk may run without the global opt-in.

    True for a model folder you placed yourself and for a downloaded model whose
    code matches a :func:`trust_model` record. Never touches the network.

    Parameters
    ----------
    name : str
        Model name or ``"<org>/<model>"`` id.
    revision : str, optional
        Git revision, as in :func:`load_model`.
    local_dir : str or Path, optional
        Model directory, as in :func:`load_model`.

    Returns
    -------
    bool
        ``False`` also when the model is not available locally.
    """
    try:
        resolved = _resolve_model(
            repo_id=_repo_id(name), revision=revision, local_dir=local_dir, offline=True
        )
        _check_trust(resolved, trust_remote_code=False)
    except (FileNotFoundError, UntrustedModelError) as exc:
        logger.debug("Model '%s' is not trusted: %s", name, exc)
        return False
    return True

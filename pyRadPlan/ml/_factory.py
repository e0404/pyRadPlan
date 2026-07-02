"""Dynamic discovery of pyRadPlan ML models.

Models are **not** hardcoded. :func:`list_models` discovers them at call time
from two kinds of source:

* **Local** — the configured :attr:`MlSettings.local_models_dir` and the
  HuggingFace cache. Local models are reported with *all* versions available on
  disk (git tags for a cloned ``local_models_dir`` repo, cache refs for the HF
  cache) and work fully offline.
* **Remote** — the HuggingFace Hub under :attr:`MlSettings.hf_org`, reporting
  only the *most recent* version. Skipped when offline or unreachable.

A model's :class:`Modality` is derived from its repository-name prefix
(``dosecalc-*`` / ``outcome-*``). Each model lives in its own repository whose
root holds the full model contract (see
:data:`pyRadPlan.ml._hub.REQUIRED_FILES`); it is listed under the repository name.
"""

import logging
import re
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

from ._hub import REQUIRED_FILES, is_valid_model_dir, _read_metadata
from ._settings import MlSettings, get_ml_settings

logger = logging.getLogger(__name__)

#: Seconds a remote (Hub) discovery result is reused before re-querying.
REMOTE_CACHE_TTL = 300.0

#: Cache of remote discovery results, keyed by org: ``{org: (monotonic_time, result)}``.
_REMOTE_CACHE: dict[str, tuple[float, dict]] = {}


class Modality(str, Enum):
    """Supported ML model modalities."""

    DOSE_CALC = "dose_calc"
    OUTCOME = "outcome"


class ModelSpec(BaseModel):
    """Coordinates of a discovered model.

    Attributes
    ----------
    name : str
        Friendly, unique key used to look the model up (the repository name).
    repo : str
        Repository name (without the organization/namespace prefix).
    modality : Optional[Modality]
        Modality derived from the repository name, or ``None`` if it does not
        match a known prefix.
    local_versions : tuple[str, ...]
        Versions available locally (git tags / cache refs / ``"local"``). Empty
        when the model is not present on disk.
    remote_version : Optional[str]
        The most recent version available on the Hub, or ``None`` when the model
        was not found remotely (offline, unreachable, or not published).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    repo: str
    modality: Optional[Modality] = None
    local_versions: tuple[str, ...] = ()
    remote_version: Optional[str] = None

    @property
    def is_local(self) -> bool:
        """Whether the model is available on disk."""
        return bool(self.local_versions)

    @property
    def is_remote(self) -> bool:
        """Whether the model was found on the Hub."""
        return self.remote_version is not None


def modality_from_name(name: str) -> Optional[Modality]:
    """Derive a :class:`Modality` from a repository/model name prefix."""
    low = name.lower()
    if low.startswith("dosecalc"):
        return Modality.DOSE_CALC
    if low.startswith("outcome"):
        return Modality.OUTCOME
    return None


def resolve_repo_id(spec: ModelSpec, org: str) -> str:
    """Build the full repository id ``"<org>/<repo>"`` for a spec."""
    return f"{org}/{spec.repo}"


# ---------------------------------------------------------------------------
# Naming / version helpers
# ---------------------------------------------------------------------------


def _version_key(version: str):
    """Sort key giving natural/semver-ish ordering (``v1.0.2`` > ``v1.0.10``-safe)."""
    core = version[1:] if version[:1] in "vV" else version
    key = []
    for part in re.split(r"[.\-+_]", core):
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return key


def _git_tags(repo_dir: Path) -> list[str]:
    """Return the git tags of ``repo_dir`` (empty if not a git repo / git absent)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "tag", "--list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    return [t for t in proc.stdout.split() if t]


def _local_versions(repo_dir: Path) -> list[str]:
    """Best-effort version labels for a locally stored repo directory."""
    if (repo_dir / ".git").exists():
        tags = _git_tags(repo_dir)
        if tags:
            return sorted(tags, key=_version_key)
    meta = _read_metadata(repo_dir)
    if meta.get("revision"):
        return [meta["revision"]]
    return ["local"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _discover_local(cfg: MlSettings) -> dict[str, dict]:
    """Discover models on disk (local_models_dir + HuggingFace cache)."""
    found: dict[str, dict] = {}

    base = cfg.local_models_dir
    if base is not None and Path(base).is_dir():
        for repo_dir in sorted(p for p in Path(base).iterdir() if p.is_dir()):
            if not is_valid_model_dir(repo_dir):
                continue
            repo = repo_dir.name
            found[repo] = {"repo": repo, "versions": _local_versions(repo_dir)}

    _scan_hf_cache(found)
    return found


def _scan_hf_cache(found: dict[str, dict]) -> None:
    """Add HuggingFace-cache models (and their refs) to ``found``."""
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        info = scan_cache_dir()
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("Could not scan HuggingFace cache: %s", exc)
        return

    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        repo_name = repo.repo_id.split("/")[-1]
        for rev in repo.revisions:
            if not is_valid_model_dir(Path(rev.snapshot_path)):
                continue
            labels = sorted(rev.refs) or [rev.commit_hash[:8]]
            entry = found.setdefault(repo_name, {"repo": repo_name, "versions": []})
            for label in labels:
                if label not in entry["versions"]:
                    entry["versions"].append(label)


def _has_model_contract(files: set[str]) -> bool:
    """Whether ``files`` (a repo's root file list) holds the full model contract."""
    return all(req in files for req in REQUIRED_FILES)


def _latest_remote_version(api, repo_id: str) -> str:
    """Most recent tag for ``repo_id``, falling back to ``"main"``."""
    try:
        tags = [t.name for t in api.list_repo_refs(repo_id).tags]
    except Exception as exc:  # pragma: no cover - network dependent
        logger.debug("Could not list refs for %s: %s", repo_id, exc)
        tags = []
    return max(tags, key=_version_key) if tags else "main"


def _discover_remote(org: str) -> dict[str, dict]:
    """Discover models published on the Hub under ``org``.

    Uses a single ``list_models(expand=["siblings"])`` call to obtain each repo's
    file list (avoiding a per-repo tree request); only repos that actually hold a
    model incur a follow-up ``list_repo_refs`` call for their latest tag.
    """
    from huggingface_hub import HfApi  # noqa: PLC0415

    api = HfApi()
    found: dict[str, dict] = {}
    for model in api.list_models(author=org, expand=["siblings"]):
        repo = model.id.split("/")[-1]
        files = {s.rfilename for s in (model.siblings or [])}
        if not _has_model_contract(files):
            continue
        found[repo] = {"repo": repo, "version": _latest_remote_version(api, model.id)}
    return found


def _discover_remote_cached(org: str, refresh: bool) -> dict[str, dict]:
    """Remote discovery with a short TTL cache (see :data:`REMOTE_CACHE_TTL`)."""
    now = time.monotonic()
    if not refresh:
        cached = _REMOTE_CACHE.get(org)
        if cached is not None and now - cached[0] < REMOTE_CACHE_TTL:
            return cached[1]
    result = _discover_remote(org)
    _REMOTE_CACHE[org] = (now, result)
    return result


def clear_model_cache() -> None:
    """Clear the cached remote discovery results, forcing a fresh Hub query."""
    _REMOTE_CACHE.clear()


def _blank(name: str, repo: str) -> dict:
    return {
        "name": name,
        "repo": repo,
        "modality": modality_from_name(repo),
        "local_versions": (),
        "remote_version": None,
    }


def discover_models(*, offline: Optional[bool] = None, refresh: bool = False) -> list[ModelSpec]:
    """Discover all model variants from local storage and the Hub.

    Parameters
    ----------
    offline : bool, optional
        Skip the remote Hub query. Defaults to :attr:`MlSettings.offline`.
    refresh : bool, optional
        Bypass the remote cache and re-query the Hub. Defaults to ``False``;
        remote results are otherwise reused for :data:`REMOTE_CACHE_TTL` seconds.

    Returns
    -------
    list[ModelSpec]
        One spec per discovered model, sorted by name. A model present both
        locally and remotely appears once, carrying both ``local_versions`` and
        ``remote_version``.
    """
    cfg = get_ml_settings()
    if offline is None:
        offline = cfg.offline

    entries: dict[str, dict] = {}

    for name, info in _discover_local(cfg).items():
        entry = entries.setdefault(name, _blank(name, info["repo"]))
        entry["local_versions"] = tuple(info["versions"])

    if not offline:
        try:
            remote = _discover_remote_cached(cfg.hf_org, refresh)
        except ImportError:
            logger.debug("huggingface_hub not installed; skipping remote discovery.")
            remote = {}
        except Exception as exc:
            logger.warning("Remote model discovery failed: %s", exc)
            remote = {}
        for name, info in remote.items():
            entry = entries.setdefault(name, _blank(name, info["repo"]))
            entry["remote_version"] = info["version"]

    return [ModelSpec(**e) for e in sorted(entries.values(), key=lambda d: d["name"])]


def list_models(
    modality: Optional[Modality] = None, *, offline: Optional[bool] = None, refresh: bool = False
) -> list[ModelSpec]:
    """List discovered model variants, optionally filtered by modality.

    Parameters
    ----------
    modality : Modality, optional
        If given, only return models of this modality.
    offline : bool, optional
        Skip the remote Hub query. Defaults to :attr:`MlSettings.offline`.
    refresh : bool, optional
        Bypass the remote cache and re-query the Hub.

    Returns
    -------
    list[ModelSpec]
        The matching specifications.
    """
    specs = discover_models(offline=offline, refresh=refresh)
    if modality is not None:
        specs = [s for s in specs if s.modality == modality]
    return specs


def get_model_spec(
    name: str, *, offline: Optional[bool] = None, refresh: bool = False
) -> ModelSpec:
    """Look up a discovered model variant by its friendly name.

    Parameters
    ----------
    name : str
        The friendly name of the model.
    offline : bool, optional
        Skip the remote Hub query. Defaults to :attr:`MlSettings.offline`.
    refresh : bool, optional
        Bypass the remote cache and re-query the Hub.

    Returns
    -------
    ModelSpec
        The matching specification.

    Raises
    ------
    KeyError
        If no model with that name is discovered.
    """
    specs = discover_models(offline=offline, refresh=refresh)
    for spec in specs:
        if spec.name == name:
            return spec
    available = ", ".join(sorted(s.name for s in specs)) or "<none>"
    raise KeyError(f"Unknown model '{name}'. Discovered models: {available}.")

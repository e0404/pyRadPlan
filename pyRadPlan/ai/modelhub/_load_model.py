"""Generic loader for pyRadPlan AI models.

Instantiation is driven entirely by ``model_config.json``:
"""

import hashlib
import importlib.util
import inspect
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Any, Optional, Union

from pyRadPlan._settings import get_settings

from ._resolve import resolve_model_dir
from ._preprocessor import BasePreprocessor

logger = logging.getLogger(__name__)

_SAFETENSORS_HINT = (
    "safetensors (and the torch build it loads into) is required to load model weights. "
    "safetensors ships with pyRadPlan; install a torch build matching your platform "
    "(see https://pytorch.org/get-started/locally/)."
)

# Modules a model folder may expose, importable by name during loading.
_MODEL_SUBMODULES = ("model", "preprocessor")

#: Namespace the code of every loaded model repository is registered under.
_REPO_PACKAGE = "pyRadPlan._ai_model_repos"

#: Modules already loaded, keyed by resolved model directory.
_loaded_repos: dict[Path, tuple[types.ModuleType, types.ModuleType]] = {}


def _repo_package_name(model_dir: Path) -> str:
    """Derive a unique, importable package name for a model directory."""
    slug = re.sub(r"\W+", "_", model_dir.name).strip("_") or "model_repo"
    digest = hashlib.sha256(str(model_dir).encode("utf-8")).hexdigest()[:8]
    return f"{_REPO_PACKAGE}.{slug}_{digest}"


def _repo_package(model_dir: Path) -> types.ModuleType:
    """Return (creating if needed) the package module a model repo lives under."""
    if _REPO_PACKAGE not in sys.modules:
        namespace = types.ModuleType(_REPO_PACKAGE)
        namespace.__doc__ = "Code of model repositories loaded by pyRadPlan.ai.modelhub."
        namespace.__path__ = []
        sys.modules[_REPO_PACKAGE] = namespace

    name = _repo_package_name(model_dir)
    pkg = sys.modules.get(name)
    if pkg is None:
        pkg = types.ModuleType(name)
        # Setting __path__ makes the repo's own files importable as submodules,
        # so `from .preprocessor import X` works inside model.py.
        pkg.__path__ = [str(model_dir)]
        sys.modules[name] = pkg
    return pkg


def _exec_module(module_name: str, path: Path) -> types.ModuleType:
    """Import ``path`` as a module registered under ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path.name} from {path.parent}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_model_modules(model_dir: Path) -> tuple[types.ModuleType, types.ModuleType]:
    """Import a model folder's ``model.py`` and ``preprocessor.py``.

    Each folder is registered under its own package name derived from its path
    (``pyRadPlan._ai_model_repos.<folder>_<hash>``), so two model repositories
    can be loaded side by side and the classes they define keep a resolvable
    ``__module__`` -- which pickling, ``torch.save`` and ``multiprocessing``
    workers rely on. The modules therefore stay in :data:`sys.modules`, and the
    result is cached so a repeated load of the same folder reuses them.

    The folder is additionally placed on ``sys.path`` for the duration of the
    import so a repository importing a sibling file by bare name still works;
    that entry, and any top-level module such an import leaks, are removed
    afterwards.
    """
    model_dir = model_dir.resolve()
    cached = _loaded_repos.get(model_dir)
    if cached is not None:
        return cached

    pkg = _repo_package(model_dir)

    path_str = str(model_dir)
    path_inserted = path_str not in sys.path
    if path_inserted:
        sys.path.insert(0, path_str)

    # Save and clear any pre-existing modules a bare sibling import would collide with.
    saved = {name: sys.modules.pop(name, None) for name in _MODEL_SUBMODULES}

    # Avoid writing __pycache__ into the (possibly shipped/read-only) model dir.
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True

    try:
        modules = tuple(
            _exec_module(f"{pkg.__name__}.{name}", model_dir / f"{name}.py")
            for name in _MODEL_SUBMODULES
        )
        for name, module in zip(_MODEL_SUBMODULES, modules):
            setattr(pkg, name, module)
    finally:
        sys.dont_write_bytecode = dont_write_bytecode
        if path_inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:  # pragma: no cover - defensive
                pass
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod

    _loaded_repos[model_dir] = modules
    return modules


def _find_preprocessor_class(module: types.ModuleType, name: Optional[str]) -> type:
    """Find the preprocessor class in ``preprocessor.py``.

    Uses ``name`` if given, otherwise the single :class:`BasePreprocessor`
    subclass defined in the module.
    """
    if name is not None:
        cls = getattr(module, name, None)
        if cls is None:
            raise ImportError(
                f"preprocessor.py defines no '{name}' "
                "(named by 'preprocessor_name' in model_config.json)."
            )
        if not (isinstance(cls, type) and issubclass(cls, BasePreprocessor)):
            raise TypeError(f"'{name}' in preprocessor.py is not a BasePreprocessor subclass.")
        return cls

    candidates = [
        obj
        for _, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, BasePreprocessor)
        and obj is not BasePreprocessor
        and obj.__module__ == module.__name__
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ImportError(
            "preprocessor.py must define a BasePreprocessor subclass "
            "(or set 'preprocessor_name' in model_config.json)."
        )
    raise ImportError(
        "preprocessor.py defines multiple BasePreprocessor subclasses "
        f"({[c.__name__ for c in candidates]}); set 'preprocessor_name' in model_config.json."
    )


def _read_config(model_dir: Path) -> dict:
    """Read and minimally validate a model folder's ``model_config.json``."""
    config_path = model_dir / "model_config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"'{config_path}' is not valid JSON: {exc}") from exc
    if not isinstance(config, dict):
        raise TypeError(f"'{config_path}' must contain a JSON object.")
    if "model_name" not in config:
        raise KeyError(
            f"'{config_path}' is missing the required 'model_name' key "
            "(the class in model.py to instantiate)."
        )
    return config


def _load_weights(model: Any, weights_path: Path, device: str) -> None:
    """Load safetensors weights into ``model`` in place."""
    try:
        from safetensors.torch import load_model as st_load_model  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_SAFETENSORS_HINT) from exc
    st_load_model(model, str(weights_path), device=device)


def load_model(  # noqa: PLR0913 - all but `name` are keyword-only options
    name: Optional[str] = None,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[Union[str, Path]] = None,
    offline: Optional[bool] = None,
    device: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> tuple[Any, BasePreprocessor]:
    """Load a model and its preprocessor.

    Parameters
    ----------
    name : str, optional
        Model name. A bare name is resolved to ``"<hf_org>/<name>"`` using
        :attr:`AiSettings.modelhub_hf_org`; a name that already contains a ``/`` is
        used as the repository id as-is.
    repo_id : str, optional
        Explicit HuggingFace repository id, overriding the registry lookup.
    revision : str, optional
        Git revision (tag/branch/commit) to load. Pinning one makes a matching
        local copy usable without contacting the hub.
    local_dir : str or Path, optional
        Load directly from (or download into) this directory. Required for
        offline use without a prior download.
    offline : bool, optional
        Force offline mode. Defaults to :attr:`AiSettings.modelhub_offline`.
    device : str, optional
        Device to place the model on. Defaults to :attr:`AiSettings.modelhub_device`.
    trust_remote_code : bool, optional
        Allow executing the Python shipped with the model. Defaults to
        :attr:`AiSettings.modelhub_trust_remote_code` (``False``), except for a
        purely local load -- ``local_dir`` given without ``name``/``repo_id`` --
        where the code is already under the caller's control and no opt-in is
        needed. Loading fails if this resolves to ``False``.

    Returns
    -------
    tuple[Any, BasePreprocessor]
        The loaded model (in eval mode) and its preprocessor.

    Raises
    ------
    ValueError
        If ``trust_remote_code`` is disabled, or neither ``name``, ``repo_id``
        nor a usable ``local_dir`` is provided.
    """
    cfg = get_settings().ai
    if device is None:
        device = cfg.modelhub_device

    if name is not None and repo_id is None:
        # A name that already carries an organization is a full id; this is what
        # list_local_models() returns, so its output feeds straight back in.
        repo_id = name if "/" in name else f"{cfg.modelhub_hf_org}/{name}"

    from_hub = repo_id is not None
    if trust_remote_code is None:
        trust_remote_code = cfg.modelhub_trust_remote_code or not from_hub

    # Checked before resolving: there is no reason to download code we refuse to run.
    if not trust_remote_code:
        source = f"'{repo_id}'" if from_hub else f"'{local_dir}'"
        raise ValueError(
            f"Loading the model in {source} executes code shipped with the model "
            "(model.py, preprocessor.py). Pass trust_remote_code=True "
            "(or set PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1) only for sources you trust."
        )

    model_dir = resolve_model_dir(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        offline=offline,
    )

    config = _read_config(model_dir)
    model_module, preprocessor_module = _load_model_modules(model_dir)

    model_name = config["model_name"]
    model_cls = getattr(model_module, model_name, None)
    if model_cls is None:
        raise AttributeError(
            f"model.py in '{model_dir}' defines no '{model_name}' "
            "(named by 'model_name' in model_config.json)."
        )
    model = model_cls(**config.get("model_params", {}))

    _load_weights(model, model_dir / "weights.safetensors", "cpu")
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    preprocessor_cls = _find_preprocessor_class(
        preprocessor_module, config.get("preprocessor_name")
    )
    preprocessor = preprocessor_cls(config.get("model_preprocessing", {}))

    logger.info("Loaded model from %s onto device '%s'.", model_dir, device)
    return model, preprocessor

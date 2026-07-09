"""Generic loader for pyRadPlan ML models.

Instantiation is driven entirely by ``model_config.json``:
"""

import importlib.util
import inspect
import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

from ._hub import resolve_model_dir
from ._preprocessor import BasePreprocessor
from ._settings import get_ml_settings

logger = logging.getLogger(__name__)

_SAFETENSORS_HINT = (
    "safetensors is required to load model weights. "
    "Install the optional dependencies with: pip install pyRadPlan[ml]"
)

# Modules a model folder may expose, importable by name during loading.
_MODEL_SUBMODULES = ("model", "preprocessor")


def _exec_module(module_name: str, path: Path):
    """Import ``path`` as a module registered under ``module_name``."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path.name} from {path.parent}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_model_modules(model_dir: Path):
    """Import a model folder's ``model.py`` and ``preprocessor.py``.

    The model directory is placed on ``sys.path`` for the duration of the import
    so the modules can import each other by name. ``sys.path`` and
    ``sys.modules`` are restored afterwards so different model folders do not
    shadow each other.
    """
    path_str = str(model_dir)
    path_inserted = path_str not in sys.path
    if path_inserted:
        sys.path.insert(0, path_str)

    # Save and clear any pre-existing modules that would collide.
    saved = {name: sys.modules.pop(name, None) for name in _MODEL_SUBMODULES}

    # Avoid writing __pycache__ into the (possibly shipped/read-only) model dir.
    dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True

    try:
        model_module = _exec_module("model", model_dir / "model.py")
        preprocessor_module = _exec_module("preprocessor", model_dir / "preprocessor.py")
        return model_module, preprocessor_module
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


def _find_preprocessor_class(module, name: Optional[str]):
    """Find the preprocessor class in ``preprocessor.py``.

    Uses ``name`` if given, otherwise the single :class:`BasePreprocessor`
    subclass defined in the module.
    """
    if name is not None:
        return getattr(module, name)

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


def _load_weights(model: Any, weights_path: Path, device: str) -> None:
    """Load safetensors weights into ``model`` in place."""
    try:
        from safetensors.torch import load_model as st_load_model  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(_SAFETENSORS_HINT) from exc
    st_load_model(model, str(weights_path), device=device)


def load_model(
    name: Optional[str] = None,
    *,
    repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_dir: Optional[str] = None,
    offline: Optional[bool] = None,
    device: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
) -> tuple[Any, BasePreprocessor]:
    """Load a model and its preprocessor.

    Parameters
    ----------
    name : str, optional
        Friendly model name, resolved to the repository id
        ``"<hf_org>/<name>"`` using :attr:`MlSettings.hf_org`.
    repo_id : str, optional
        Explicit HuggingFace repository id, overriding the registry lookup.
    revision : str, optional
        Git revision (tag/branch/commit) to load.
    local_dir : str, optional
        Load directly from (or download into) this directory. Required for
        offline use without a prior download.
    offline : bool, optional
        Force offline mode. Defaults to :attr:`MlSettings.offline`.
    device : str, optional
        Device to place the model on. Defaults to :attr:`MlSettings.device`.
    trust_remote_code : bool, optional
        Allow executing the Python shipped with the model. Defaults to
        :attr:`MlSettings.trust_remote_code`. Loading fails if this is
        ``False``.

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
    cfg = get_ml_settings()
    if device is None:
        device = cfg.device
    if trust_remote_code is None:
        trust_remote_code = cfg.trust_remote_code

    if name is not None and repo_id is None:
        repo_id = f"{cfg.hf_org}/{name}"

    model_dir = resolve_model_dir(
        repo_id=repo_id,
        revision=revision,
        local_dir=local_dir,
        offline=offline,
    )

    if not trust_remote_code:
        raise ValueError(
            f"Loading the model in '{model_dir}' executes code shipped with the model "
            "(model.py, preprocessor.py). Pass trust_remote_code=True "
            "(or set PYRADPLAN_ML_TRUST_REMOTE_CODE=1) only for sources you trust."
        )

    config = json.loads((model_dir / "model_config.json").read_text(encoding="utf-8"))

    model_module, preprocessor_module = _load_model_modules(model_dir)

    model_cls = getattr(model_module, config["model_name"])
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

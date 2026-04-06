"""Simple model importer for PyTorch models.

Loads models from a named folder containing:
 - One JSON config file (*.json)
 - One Python model definition file (*.py)
 - One weights file (*.pth or *.pt)

Usage:
    model, config = load_model("densenet", category="electrons")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import importlib.util as importlib_util

try:  # pragma: no cover
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

from pyRadPlan.core.xp_utils import pytorch_gpu_available

__all__ = ["load_model", "discover_models", "list_categories"]

# Category aliases mapping keywords to subdirectory paths
CATEGORY_ALIASES: Dict[str, str] = {
    # Particle types (dose calculation)
    "electrons": "dose_calc/electrons",
    "photons": "dose_calc/photons",
    "protons": "dose_calc/protons",
    "carbon": "dose_calc/carbon",
    "vhee": "dose_calc/vhee",
    # Outcome models
    "ntcp": "outcome/ntcp",
    "tcp": "outcome/tcp",
}

_DEFAULT_BASE_DIR = Path(__file__).parent


def load_model(
    name: str,
    category: str,
    base_dir: Optional[Path | str] = None,
    device: Optional[str] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load a PyTorch model + config by name and category keyword.

    Parameters
    ----------
    name : str
        Name of the model folder (e.g., 'densenet').
    category : str
        Category keyword as defined in CATEGORY_ALIASES
        (e.g., 'electrons', 'ntcp').
    base_dir : Path | str, optional
        Base directory containing model folders. Defaults to this
        module's parent directory.
    device : str, optional
        Device to load the model on. Defaults to 'cpu'.

    Returns
    -------
    model : torch.nn.Module
        Initialized model with loaded weights in eval mode.
    config : dict
        The configuration dictionary from the JSON file.
    """
    if torch is None:
        raise RuntimeError("PyTorch required for model loading.")

    device = device or ("cuda" if pytorch_gpu_available() else "cpu")
    base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR

    # Resolve category to subdirectory path
    if category not in CATEGORY_ALIASES:
        raise ValueError(
            f"Unknown category '{category}'. Available: {sorted(CATEGORY_ALIASES.keys())}"
        )

    model_dir = base_dir / CATEGORY_ALIASES[category] / name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find the three required files
    config_path = _find_single(model_dir, "*.json", "config")
    model_file = _find_single(model_dir, "*.py", "model definition")
    weights_path = _find_weights(model_dir)

    # Load config
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    class_name = config.get("class_name") or config.get("model_name")
    if not class_name:
        raise KeyError("Config must contain 'class_name' or 'model_name'.")

    init_params = config.get("init_params", {})

    # Import model class from .py file
    spec = importlib_util.spec_from_file_location(f"model_{name}", str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {model_file}")
    module = importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    model_cls = getattr(module, class_name, None)
    if model_cls is None:
        available = [n for n in dir(module) if not n.startswith("_")]
        raise ImportError(
            f"Class '{class_name}' not found in {model_file}. Available: {available}"
        )

    # Instantiate and load weights
    model = model_cls(**init_params)

    ckpt = torch.load(str(weights_path), map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in ckpt:
                ckpt = ckpt[key]
                break
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    return model, config


def discover_models(
    category: Optional[str] = None,
    base_dir: Optional[Path | str] = None,
) -> Dict[str, List[str]]:
    """List available models, optionally filtered by category.

    Returns
    -------
    dict
        Mapping of category keyword -> list of model folder names.
    """
    base_dir = Path(base_dir) if base_dir else _DEFAULT_BASE_DIR
    result: Dict[str, List[str]] = {}

    categories = [category] if category else list(CATEGORY_ALIASES.keys())

    for cat in categories:
        if cat not in CATEGORY_ALIASES:
            continue
        cat_dir = base_dir / CATEGORY_ALIASES[cat]
        if not cat_dir.is_dir():
            continue
        models = [
            d.name
            for d in sorted(cat_dir.iterdir())
            if d.is_dir() and not d.name.startswith(("_", "."))
        ]
        if models:
            result[cat] = models

    return result


def list_categories(base_dir: Optional[Path | str] = None) -> List[str]:
    """List categories that have at least one model."""
    return sorted(discover_models(base_dir=base_dir).keys())


# --- Helpers ---


def _find_single(folder: Path, pattern: str, label: str) -> Path:
    """Find exactly one file matching a glob pattern in a folder."""
    files = sorted(folder.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {label} file ({pattern}) in {folder}")
    if len(files) == 1:
        return files[0]
    # Prefer file named after folder (e.g., densenet/densenet.py)
    for f in files:
        if f.stem == folder.name:
            return f
    return files[0]


def _find_weights(folder: Path) -> Path:
    """Find a single weights file (.pth or .pt)."""
    files = sorted(folder.glob("*.pth")) + sorted(folder.glob("*.pt"))
    if not files:
        raise FileNotFoundError(f"No weights file (.pth/.pt) in {folder}")
    if len(files) == 1:
        return files[0]
    for f in files:
        if f.stem == folder.name:
            return f
    return files[0]

"""Provide information about the available compute backends."""

import importlib
import sys
import warnings
from types import ModuleType

import array_api_compat

from ..._settings import get_settings

try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
from typing import Optional, Union

try:
    import torch
except ImportError:
    torch = None

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import jax
except ImportError:
    jax = None

from .helpers import (
    DLPACK_CPU,
    DLPACK_CUDA,
    get_device_info,
    is_on_gpu,
    get_current_stream,
    create_stream,
    synchronize,
    free_gpu_memory,
    record_event,
    elapsed_time,
    to_numpy,
    from_numpy,
    to_namespace,
)

from .typing import Array, ArrayNamespace


# Check if GPU is available
def cupy_available() -> bool:
    """Check if CuPy is available and a compatible GPU is present."""
    return cp is not None and cp.cuda.is_available()


def pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return torch is not None


def pytorch_gpu_available() -> bool:
    """Check if PyTorch is available and a compatible GPU is present."""
    return torch is not None and torch.cuda.is_available()


def jax_available() -> bool:
    """Check if JAX is available."""
    return jax is not None


# TODO: Check if this here is working as intended.
def jax_gpu_available() -> bool:
    """Check if JAX is available and a compatible GPU is present."""
    if jax is None:
        return False
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except RuntimeError:
        return False


# Currently not needed. Maybe in future?
def numba_cuda_available() -> bool:
    """Check if Numba CUDA is available."""
    try:
        return numba_cuda.is_available()
    except (ImportError, AttributeError):
        return False


def _get_preferred_gpu_backend() -> Optional[str]:
    """Traverse recommended backends list."""
    wish_list = [
        ("cupy", cupy_available),
        ("torch", pytorch_gpu_available),
        ("jax", jax_gpu_available),
    ]
    for backend, is_available_fn in wish_list:
        if is_available_fn():
            return backend
    return None


def choose_array_api_namespace(namespace: Optional[str] = None) -> ArrayNamespace:
    """
    Get the name of the preferred Array API conform computational namespace / backend.

    The preferred backend is configured via :data:`pyRadPlan.settings`
    (``xp.prefer_gpu``, ``xp.preferred_cpu_array_backend``,
    ``xp.preferred_gpu_array_backend``).

    Parameters
    ----------
    namespace : Optional[str], optional
        The name of the desired backend. If None, the preferred backend is used.
    """
    if namespace is None:
        xp_settings = get_settings().xp
        gpu_backend = xp_settings.preferred_gpu_array_backend or _get_preferred_gpu_backend()
        if xp_settings.prefer_gpu and gpu_backend is not None:
            namespace = gpu_backend
        else:
            namespace = xp_settings.preferred_cpu_array_backend
    else:
        namespace = namespace.lower()

    try:
        return importlib.import_module(f"array_api_compat.{namespace}")
    except ModuleNotFoundError:
        return importlib.import_module(namespace)


def choose_device(xp: Optional[ArrayNamespace] = None, gpu_index: int = 0) -> Union[str, None]:
    """
    Choose the default device for the given array namespace.

    Parameters
    ----------
    xp : Optional[ArrayNamespace], optional
        The array namespace. If None, the preferred backend is used.
    gpu_index : int, optional
        GPU device index to use when a GPU backend is selected. Default is 0.
        Pass ``1``, ``2``, etc. for multi-GPU setups.
    """
    if xp is None:
        xp = choose_array_api_namespace()

    if get_settings().xp.prefer_gpu:
        if array_api_compat.is_torch_namespace(xp) and pytorch_gpu_available():
            return f"cuda:{gpu_index}"
        if array_api_compat.is_cupy_namespace(xp) and cupy_available():
            return str(gpu_index)

    # We need to handle array_api_strict separately, as it does not provide device info!
    if array_api_compat.is_array_api_strict_namespace(xp):
        return None

    return "cpu"


_DEPRECATED_SETTINGS_ALIASES = {
    "PREFER_GPU": "prefer_gpu",
    "PREFERRED_CPU_ARRAY_BACKEND": "preferred_cpu_array_backend",
    "PREFERRED_GPU_ARRAY_BACKEND": "preferred_gpu_array_backend",
}


def _warn_deprecated_alias(name: str, field: str) -> None:
    warnings.warn(
        f"'xp_utils.{name}' is deprecated; use 'pyRadPlan.settings.xp.{field}' instead.",
        DeprecationWarning,
        stacklevel=3,
    )


class _SettingsAliasModule(ModuleType):
    """Redirect the legacy module-level backend constants to pyRadPlan.settings.xp."""

    def __getattr__(self, name: str):
        field = _DEPRECATED_SETTINGS_ALIASES.get(name)
        if field is None:
            raise AttributeError(f"module {self.__name__!r} has no attribute {name!r}")
        _warn_deprecated_alias(name, field)
        value = getattr(get_settings().xp, field)
        if name == "PREFERRED_GPU_ARRAY_BACKEND" and value is None:
            value = _get_preferred_gpu_backend()
        return value

    def __setattr__(self, name: str, value) -> None:
        field = _DEPRECATED_SETTINGS_ALIASES.get(name)
        if field is None:
            super().__setattr__(name, value)
            return
        _warn_deprecated_alias(name, field)
        setattr(get_settings().xp, field, value)


# Assignments like ``xp_utils.PREFER_GPU = False`` must keep steering the global
# settings, which requires intercepting attribute writes on the module itself.
sys.modules[__name__].__class__ = _SettingsAliasModule


__all__ = [
    "DLPACK_CPU",
    "DLPACK_CUDA",
    "choose_array_api_namespace",
    "choose_device",
    "get_device_info",
    "is_on_gpu",
    "cp",
    "torch",
    "jax",
    "cupy_available",
    "pytorch_available",
    "pytorch_gpu_available",
    "jax_available",
    "jax_gpu_available",
    "get_current_stream",
    "create_stream",
    "record_event",
    "synchronize",
    "free_gpu_memory",
    "elapsed_time",
    "to_numpy",
    "from_numpy",
    "to_namespace",
    "Array",
    "ArrayNamespace",
]

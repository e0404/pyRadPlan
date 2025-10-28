"""Provide information about the available compute backends."""

import importlib

from typing import Optional

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import torch
except ImportError:
    torch = None

from .helpers import (
    get_current_stream,
    create_stream,
    synchronize,
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


PREFERRED_CPU_ARRAY_BACKEND: str = "numpy"
PREFERRED_GPU_ARRAY_BACKEND: str = "cupy" if cupy_available() else None
PREFER_GPU = True


def choose_array_api_namespace(namespace: Optional[str] = None) -> ArrayNamespace:
    """
    Get the name of the preferred Array API conform computational namespace / backend.

    Parameters
    ----------
    namespace : Optional[str], optional
        The name of the desired backend. If None, the preferred backend is used.
    """
    if namespace is None:
        if PREFER_GPU and PREFERRED_GPU_ARRAY_BACKEND is not None:
            namespace = PREFERRED_GPU_ARRAY_BACKEND
        else:
            namespace = PREFERRED_CPU_ARRAY_BACKEND
    else:
        namespace = namespace.lower()

    try:
        return importlib.import_module(f"array_api_compat.{namespace}")
    except ModuleNotFoundError:
        return importlib.import_module(namespace)


__all__ = [
    "cp",
    "torch",
    "cupy_available",
    "torch_gpu_available",
    "get_current_stream",
    "create_stream",
    "record_event",
    "synchronize",
    "elapsed_time",
    "to_numpy",
    "from_numpy",
    "to_namespace",
    "Array",
    "ArrayNamespace",
]

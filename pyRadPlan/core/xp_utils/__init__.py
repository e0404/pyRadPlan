"""Provide information about the available compute backends."""

import importlib
import array_api_compat

try:
    from numba import cuda as numba_cuda
except ImportError:
    numba_cuda = None
from typing import Optional, Union

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


# Currently not needed. Maybe in future?
def numba_cuda_available() -> bool:
    """Check if Numba CUDA is available."""
    try:
        return numba_cuda.is_available()
    except (ImportError, AttributeError):
        return False


PREFERRED_CPU_ARRAY_BACKEND: str = "numpy"
PREFERRED_GPU_ARRAY_BACKEND: str = (
    "cupy" if cupy_available() else ("torch" if pytorch_gpu_available() else None)
)
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


def choose_device(xp: Optional[ArrayNamespace] = None) -> Union[str, int, None]:
    """
    Choose the default device for the given array namespace.

    Parameters
    ----------
    xp : Optional[ArrayNamespace], optional
        The array namespace. If None, the preferred backend is used.
    """
    if xp is None:
        xp = choose_array_api_namespace()

    if PREFER_GPU:
        if array_api_compat.is_torch_namespace(xp) and pytorch_gpu_available():
            return "cuda"
        if array_api_compat.is_cupy_namespace(xp) and cupy_available():
            return "0"

    # We need to handle apray_api_strict separately, as it does not provide device info!
    if array_api_compat.is_array_api_strict_namespace(xp):
        return None

    return "cpu"


__all__ = [
    "cp",
    "torch",
    "cupy_available",
    "torch_gpu_available",
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

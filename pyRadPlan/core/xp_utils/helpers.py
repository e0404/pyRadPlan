"""Helper functions for array namespace operations."""

from __future__ import annotations
from typing import ContextManager, Optional, Union, Any
from contextlib import nullcontext
import importlib

import logging
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    CupySpmatrix = csp.spmatrix
except ImportError:
    cp = None
    CupySpmatrix = Any
try:
    import torch

except ImportError:
    torch = None

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse
except ImportError:
    jax = None
    jnp = None
    jsparse = None

try:
    import array_api_strict
except ImportError:
    array_api_strict = None

import array_api_compat

import numpy as np

from ..._settings import get_settings

import scipy.sparse as scp

from timeit import default_timer as timer
from datetime import timedelta

from numpy.typing import NDArray
from .typing import Array, ArrayNamespace

# from array_api._2024_12 import ArrayNamespace
# ArrayNamespace: type = TypeVar(ArrayNamespace)
# Array: type = TypeVar(Array)

logger = logging.getLogger(__name__)

# DLPack device type constants (from DLPack spec)
DLPACK_CPU = 1
DLPACK_CUDA = 2
DLPACK_CUDA_HOST = 3
DLPACK_ROCM = 8

# Device types considered "GPU"
# Intel GPUs are not supported. Feel free to contribute and reach out to us.
_GPU_DEVICE_TYPES = {DLPACK_CUDA, DLPACK_ROCM}


def get_device_info(arr: Any) -> tuple[int, int]:
    """
    Return the DLPack device type and device id for an array.

    Uses ``__dlpack_device__()`` as the primary method, with fallbacks

    Returns
    -------
    tuple[int, int]
        ``(device_type, device_id)`` following the DLPack spec.
        Device type constants: DLPACK_CPU=1, DLPACK_CUDA=2, etc.
    """
    # Primary: DLPack protocol
    if hasattr(arr, "__dlpack_device__"):
        try:
            device = arr.__dlpack_device__()
            return (int(device[0]), int(device[1]))
        except Exception:
            logger.debug(
                "__dlpack_device__() failed for %s; falling back to library checks.",
                type(arr).__name__,
                exc_info=True,
            )

    # Fallback: library-specific checks
    # TODO: Not sure that these fallbacks are needed. Can't hurt right?
    if array_api_compat.is_numpy_array(arr):
        return (DLPACK_CPU, 0)

    if array_api_compat.is_cupy_array(arr):
        device_id = arr.device.id if hasattr(arr, "device") else 0
        return (DLPACK_CUDA, device_id)

    if array_api_compat.is_torch_array(arr):
        if arr.is_cuda:
            device_id = arr.device.index if arr.device.index is not None else 0
            return (DLPACK_CUDA, device_id)
        return (DLPACK_CPU, 0)

    # Unknown -- assume CPU, but say so: silently reporting a device array as CPU makes
    # is_on_gpu() wrong and lets to_namespace() skip a requested device transfer.
    warnings.warn(
        f"Cannot determine the device of a '{type(arr).__name__}' object; assuming CPU. "
        "If this array lives on a GPU, device placement will be incorrect.",
        UserWarning,
        stacklevel=2,
    )
    return (DLPACK_CPU, 0)


def is_on_gpu(arr: Any) -> bool:
    """Return True if the array resides on a GPU device."""
    device_type, _ = get_device_info(arr)
    return int(device_type) in _GPU_DEVICE_TYPES


def _is_torch_sparse_tensor(arr: Any) -> bool:
    if torch is None or not isinstance(arr, torch.Tensor):
        return False
    return arr.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


def _parse_device_to_dlpack(device: Any) -> tuple[int, int] | None:
    """Parse a device specification to a DLPack device tuple.

    None means that no explicit target device was requested.
    """
    if device is None:
        return None

    # Already normalized DLPack-style tuple
    if isinstance(device, tuple) and len(device) == 2:
        return (int(device[0]), int(device[1]))

    # String-like specifications
    if isinstance(device, str):
        device_str = device.lower()

        if device_str == "cpu":
            return (DLPACK_CPU, 0)

        if device_str in ("gpu", "cuda"):
            return (DLPACK_CUDA, 0)

        if device_str.startswith("cuda:"):
            return (DLPACK_CUDA, int(device_str.split(":", 1)[1]))

        if device_str.startswith("gpu:"):
            return (DLPACK_CUDA, int(device_str.split(":", 1)[1]))

        if device_str.startswith("gpu") and device_str[3:].isdigit():
            return (DLPACK_CUDA, int(device_str[3:]))

    # torch.device
    if torch is not None and isinstance(device, torch.device):
        if device.type == "cpu":
            return (DLPACK_CPU, 0)
        if device.type == "cuda":
            return (DLPACK_CUDA, 0 if device.index is None else int(device.index))

    # JAX Device
    # Usually has .platform and .id, e.g. platform="cpu"/"gpu"/"cuda"
    platform = getattr(device, "platform", None)
    if platform is not None:
        platform = str(platform).lower()

        # .id is JAX's *global* device index, while dlpack_to_backend_device indexes
        # jax.devices(platform), which is per-platform. Resolve the position in that
        # list so the two agree on hosts with more than one platform.
        def _jax_platform_index(plat: str) -> int:
            if jax is not None:
                try:
                    return jax.devices(plat).index(device)
                except (ValueError, RuntimeError):
                    pass
            return int(getattr(device, "id", 0))

        if platform == "cpu":
            return (DLPACK_CPU, _jax_platform_index("cpu"))
        if platform in ("gpu", "cuda"):
            return (DLPACK_CUDA, _jax_platform_index("gpu"))

    # CuPy Device
    # cupy.cuda.Device has .id
    if cp is not None:
        cupy_cuda = getattr(cp, "cuda", None)
        cupy_device_cls = getattr(cupy_cuda, "Device", None)
        if cupy_device_cls is not None and isinstance(device, cupy_device_cls):
            return (DLPACK_CUDA, int(device.id))

    # array-api-strict Device. Matched by type rather than by its repr: a substring
    # test would also swallow unrelated devices whose repr merely mentions "cpu".
    if array_api_strict is not None and isinstance(device, array_api_strict.Device):
        if device == array_api_strict.__array_namespace_info__().default_device():
            return (DLPACK_CPU, 0)
        raise ValueError(
            f"array-api-strict device {device!r} has no DLPack equivalent; "
            "only its default (CPU) device is supported."
        )

    raise ValueError(
        f"Invalid device specification {device!r}. "
        "Supported values are None, 'cpu', 'gpu', 'cuda', 'gpu:N', 'cuda:N', "
        "a DLPack (type, id) tuple, or backend device objects."
    )


def dlpack_to_backend_device(xp: ArrayNamespace, device: tuple[int, int] | None):
    """Return backend-specific device object for a given DLPack device tuple."""
    if device is None:
        return None

    device_type = int(device[0])
    device_id = int(device[1])

    if array_api_compat.is_cupy_namespace(xp) and cp is not None:
        if device_type == DLPACK_CUDA:
            return cp.cuda.Device(device_id)
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")

    if array_api_compat.is_torch_namespace(xp) and torch is not None:
        if device_type == DLPACK_CUDA:
            return torch.device("cuda", device_id)
        if device_type == DLPACK_CPU:
            return torch.device("cpu")

    if array_api_compat.is_jax_namespace(xp) and jax is not None:
        platform = (
            "gpu" if device_type == DLPACK_CUDA else "cpu" if device_type == DLPACK_CPU else None
        )
        if platform is not None:
            devices = jax.devices(platform)
            if device_id >= len(devices):
                raise ValueError(
                    f"JAX {platform} device index {device_id} is out of range; "
                    f"{len(devices)} {platform} device(s) available."
                )
            return devices[device_id]

    if array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_array_api_strict_namespace(
        xp
    ):
        if device_type != DLPACK_CPU:
            raise ValueError("NumPy and array-api-strict do not support GPU devices.")
        return None

    raise ValueError(
        f"Cannot convert DLPack device {device} "
        f"to backend-specific device for namespace '{xp.__name__}'."
    )


def _namespace_gpu_available(xp: ArrayNamespace) -> bool:
    """Return True if the namespace can place arrays on a CUDA device right now."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp is not None and cp.cuda.is_available()
    if array_api_compat.is_torch_namespace(xp):
        return torch is not None and torch.cuda.is_available()
    if array_api_compat.is_jax_namespace(xp):
        if jax is None:
            return False
        try:
            return len(jax.devices("gpu")) > 0
        except RuntimeError:
            # CPU-only JAX raises for unknown platforms instead of returning []
            return False
    return False


def _default_dlpack_device_for_namespace(xp: ArrayNamespace) -> tuple[int, int] | None:
    """Return the default target DLPack device for a namespace.

    Honors ``settings.xp.prefer_gpu``: a GPU is only chosen when it is preferred *and*
    available. CuPy is GPU-only and therefore always maps to a CUDA device.
    """
    if array_api_compat.is_cupy_namespace(xp):
        return (DLPACK_CUDA, 0)

    if array_api_compat.is_torch_namespace(xp) or array_api_compat.is_jax_namespace(xp):
        if get_settings().xp.prefer_gpu and _namespace_gpu_available(xp):
            return (DLPACK_CUDA, 0)
        return (DLPACK_CPU, 0)

    if array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_array_api_strict_namespace(
        xp
    ):
        return (DLPACK_CPU, 0)

    raise ValueError(f"Cannot determine default DLPack device for namespace '{xp.__name__}'.")


def _source_dlpack_device(arr: Any) -> tuple[int, int] | None:
    """Return the DLPack device of an array, or None if it cannot be determined."""
    if not is_sparse_array(arr):
        if hasattr(arr, "__dlpack_device__") or array_api_compat.is_array_api_obj(arr):
            return get_device_info(arr)
        return None

    if isinstance(arr, (scp.spmatrix, scp.sparray)):
        return (DLPACK_CPU, 0)
    if cp is not None and csp.issparse(arr):
        return (DLPACK_CUDA, int(arr.data.device.id))
    if _is_torch_sparse_tensor(arr):
        return get_device_info(arr)
    if jsparse is not None and isinstance(arr, jsparse.JAXSparse):
        return get_device_info(arr.data)
    return None


def _resolve_target_device(xp_new: ArrayNamespace, arr: Any) -> tuple[int, int]:
    """Pick the target device when the caller did not request one.

    The source array's device is kept whenever the target namespace supports it, so a
    conversion never silently moves data between host and device. Otherwise the
    namespace default (see :func:`_default_dlpack_device_for_namespace`) is used.
    """
    source = _source_dlpack_device(arr)
    if source is not None:
        device_type = int(source[0])
        if device_type == DLPACK_CPU and not array_api_compat.is_cupy_namespace(xp_new):
            return (DLPACK_CPU, 0)
        if device_type == DLPACK_CUDA and _namespace_gpu_available(xp_new):
            return (DLPACK_CUDA, int(source[1]))
    return _default_dlpack_device_for_namespace(xp_new)


def get_current_stream(xp: ArrayNamespace) -> ContextManager:
    """Get the current stream based on the array namespace."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.get_current_stream()
    if array_api_compat.is_torch_namespace(xp):
        if torch.cuda.is_available():
            return torch.cuda.stream(torch.cuda.current_stream())
        return torch.cpu.stream(torch.cpu.current_stream())
    else:
        return nullcontext()


def create_stream(xp: ArrayNamespace) -> ContextManager:
    """Create a context manager for the appropriate stream based on the array namespace."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.Stream(non_blocking=True)
    if array_api_compat.is_torch_namespace(xp):
        if torch.cuda.is_available():
            s = torch.cuda.Stream()
            return torch.cuda.stream(s)
        s = torch.cpu.Stream()
        return torch.cpu.stream(s)
    else:
        return nullcontext()


def synchronize(xp: ArrayNamespace, stream: Optional[ContextManager] = None) -> None:
    """Synchronize the device if using CuPy."""

    if stream is not None and not isinstance(stream, nullcontext):
        # torch.cuda.stream() returns a StreamContext — delegate to the inner Stream
        sync = getattr(stream, "synchronize", None) or getattr(
            getattr(stream, "stream", None), "synchronize", None
        )
        if sync is None:
            warnings.warn("The provided stream does not support synchronization.")
        else:
            sync()
        return

    # Synchronize device
    if array_api_compat.is_cupy_namespace(xp):
        cp.cuda.runtime.deviceSynchronize()
    elif array_api_compat.is_torch_namespace(xp):
        try:
            torch.accelerator.synchronize()
        except Exception:
            torch.cpu.synchronize()
    elif array_api_compat.is_numpy_namespace(xp):
        pass
    else:
        warnings.warn(
            "Synchronization helper for namespace '{}' is not implemented.".format(xp.__name__)
        )


def free_gpu_memory(xp: ArrayNamespace) -> None:
    """
    Free unused GPU memory for the given array namespace.

    This function releases cached GPU memory back to the system. For CuPy,
    it frees all blocks in the default memory pool. For PyTorch, it empties
    the CUDA cache.

    Parameters
    ----------
    xp : ArrayNamespace
        The array namespace (e.g., cupy, torch, numpy).

    Notes
    -----
    This is useful to manage GPU memory usage in long-running applications.
    Will ignore cpu arrays.
    """
    if array_api_compat.is_cupy_namespace(xp) and cp is not None:
        cp.get_default_memory_pool().free_all_blocks()
    elif array_api_compat.is_torch_namespace(xp) and torch is not None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_array_api_strict_namespace(
        xp
    ):
        # Skip for numpy
        return
    else:
        warnings.warn(
            "Free GPU memory helper for namespace '{}' is not implemented or supported.".format(
                xp.__name__
            )
        )


def record_event(xp: ArrayNamespace, stream: Optional[ContextManager] = None) -> Optional[object]:
    """Record an event in the current stream if using CuPy or PyTorch."""
    if not isinstance(stream, nullcontext):
        if array_api_compat.is_cupy_namespace(xp):
            event = xp.cuda.Event()
            event.record(stream)
            return event
        if array_api_compat.is_torch_namespace(xp):
            try:
                # Unwrap StreamContext to get the underlying Stream
                stream_obj = getattr(stream, "stream", stream)

                if torch.cuda.is_available() and stream_obj is not None:
                    event = torch.cuda.Event(enable_timing=True)
                    event.record(stream_obj)
                    return event

                return timer()
            except Exception:
                pass

        if stream is not None:
            warnings.warn(
                "The provided stream does not support event recording"
                "or the helper function is not implemented for namespace '{}'.".format(xp.__name__)
            )

    return timer()


def elapsed_time(xp: ArrayNamespace, start, end) -> float:
    """Calculate the elapsed time between two events or timestamps."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.get_elapsed_time(start, end) / 1000.0  # Convert ms to s
    if array_api_compat.is_torch_namespace(xp):
        if isinstance(end, torch.Event):
            td = timedelta(milliseconds=end.elapsed_time(start)).total_seconds()
        else:
            td = timedelta(seconds=(end - start)).total_seconds()
        return td
    else:
        # Assuming start and end are timestamps from timer()
        return (timedelta(seconds=(end - start))).total_seconds()


def to_numpy(arr: Array, detach: bool = True, dtype: np.dtype | type | None = None) -> NDArray:
    """Convert an array to a NumPy array."""
    if array_api_compat.is_torch_array(arr):
        if detach and arr.requires_grad:
            arr = arr.detach()
        out = arr.cpu().numpy()
    elif array_api_compat.is_cupy_array(arr):
        out = cp.asnumpy(arr)
    else:
        out = np.asarray(arr)

    if dtype is not None:
        out = out.astype(dtype)

    return out


def from_numpy(xp: ArrayNamespace, arr: np.ndarray, *, device: Any = None) -> Array:
    """Convert a NumPy array to the specified array namespace.

    Parameters
    ----------
    xp : ArrayNamespace
        The target array namespace.
    arr : np.ndarray
        The NumPy array to convert.
    device : Optional[str | tuple[int, int] | None] = None
        Target device. Accepts ``"cpu"``, ``"gpu"``, ``"gpu:N"``, ``"cuda:N"`` or dlpack tuple (normalized automatically).
        Default ``None`` keeps the array on the CPU (NumPy/torch/jax) or uses the current CuPy device.
    """
    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp)

    device_type = int(device[0])
    device_id = int(device[1])

    if array_api_compat.is_cupy_namespace(xp) and cp is not None:
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )
        with cp.cuda.Device(device_id):
            return cp.asarray(arr)

    if array_api_compat.is_torch_namespace(xp) and torch is not None:
        t = torch.from_numpy(arr)
        if device_type == DLPACK_CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            return t.to(device=dlpack_to_backend_device(xp, device))
        elif device_type == DLPACK_CPU:
            return t
        else:
            raise ValueError(
                f"PyTorch only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )

    if array_api_compat.is_jax_namespace(xp) and jax is not None:
        jax_arr = jnp.asarray(arr)
        jax_device = dlpack_to_backend_device(xp, device)
        return jax.device_put(jax_arr, jax_device)

    if array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_array_api_strict_namespace(
        xp
    ):
        if device_type != DLPACK_CPU:
            raise ValueError("NumPy and array-api-strict do not support GPU devices.")
        return xp.asarray(arr)

    raise TypeError(
        "Conversion helper from NumPy not implemented for namespace '{}'.".format(xp.__name__)
    )


def to_namespace(
    xp_new: Union[ArrayNamespace, str],
    arr: Array,
    *,
    copy: Optional[bool] = None,
    keep_sparse_compat: bool = True,
    device: Any = None,
) -> Array:
    """
    Convert an array to the specified array namespace.

    Parameters
    ----------
    xp_new : Union[ArrayNamespace,str]
        The target array namespace or its name as a string.
    arr : Array
        The Array to convert.
    copy : Optional[bool], optional
        Whether to force a copy during conversion, by default None.
        If None, the default behavior of the target namespace is used.
    keep_sparse_compat : bool, optional
        Whether to keep sparse array compatible, by default True.
        For example, when converting to numpy, scipy sparse matrices will not be converted to dense
        arrays because they are compatible with numpy.
    device : Optional[str, tuple[int, int]], optional
        The target device for the array. Can be specified as a string, or even better as a DLPack device tuple (type, id). Supported string values are "cpu", "gpu", "gpu:N", "cuda:N".
        If None, the source array's device is kept when the target namespace supports it;
        otherwise the namespace default is used (a GPU only if ``settings.xp.prefer_gpu``).

    Returns
    -------
    Array
        The converted array.
    """
    # We need these checks, to ensure that the error makes sense.
    # Otherwise this function will complain about a sparse array error :)
    if isinstance(arr, (int, float, bool, complex)):
        raise TypeError(
            "Conversion of scalar values to array namespaces is not supported. "
            "Create an array first or use it as is in calculations as int/float/bool/complex do not need conversion."
        )
    if isinstance(arr, (list, tuple)):
        raise TypeError(
            "Conversion of lists or tuples to array namespaces is not supported. "
            "Create an array first!"
        )

    if isinstance(xp_new, str):
        try:
            xp_new = importlib.import_module(xp_new, "array_api_compat")
        except ModuleNotFoundError:
            # Try to import the module directly
            xp_new = importlib.import_module(xp_new)

    # Convert explicit device input to DLPack tuple. If no device was requested, keep the
    # source device where the target namespace supports it, else use the namespace default.
    device = _parse_device_to_dlpack(device)
    if device is None:
        device = _resolve_target_device(xp_new, arr)

    if is_sparse_array(arr):
        return _convert_sparse_for_namespace(
            xp_new, arr, keep_sparse_compat=keep_sparse_compat, device=device
        )

    # Optimization: same namespace and same DLPack device.
    xp_old = array_api_compat.array_namespace(arr)
    if xp_new == xp_old:
        current_device_type, current_device_id = get_device_info(arr)
        current_device = (int(current_device_type), int(current_device_id))

        if current_device == device and not copy:
            return arr

    # --- Target: NumPy ---
    if array_api_compat.is_numpy_namespace(xp_new):
        device_type = int(device[0])

        # NumPy only supports CPU arrays.
        if device_type != DLPACK_CPU:
            raise ValueError("NumPy does not support GPU.")

        np_arr = to_numpy(arr)
        if copy is True:
            return xp_new.array(np_arr, copy=True)
        return xp_new.asarray(np_arr)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        device_type = int(device[0])
        device_id = int(device[1])

        # CuPy only supports CUDA/GPU arrays.
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )

        # Select the requested CUDA device.
        with cp.cuda.Device(device_id):
            # CuPy < 14.0 workaround for CPU->GPU via DLPack.
            if cp.__version__ < "14.0" and not is_on_gpu(arr):
                return cp.asarray(to_numpy(arr))

            try:
                return xp_new.from_dlpack(arr, copy=copy)
            except Exception:
                return cp.asarray(to_numpy(arr))

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        device_type = int(device[0])
        device_id = int(device[1])

        # Convert to Torch, preserving device if possible.
        try:
            if hasattr(xp_new, "from_dlpack"):
                new_arr = xp_new.from_dlpack(arr)
            else:
                new_arr = torch.utils.dlpack.from_dlpack(arr)
        except Exception:
            new_arr = torch.from_numpy(to_numpy(arr))

        # Move to requested device if needed.
        torch_device = dlpack_to_backend_device(xp_new, device)

        if device_type == DLPACK_CUDA and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but not available for PyTorch.")

        if new_arr.device != torch_device:
            new_arr = new_arr.to(device=torch_device)

        # Force copy if requested.
        if copy:
            new_arr = new_arr.clone()

        return new_arr

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None:
        # Convert to JAX, preserving device if possible.
        try:
            if hasattr(xp_new, "from_dlpack"):
                new_arr = xp_new.from_dlpack(arr)
            else:
                new_arr = jax.dlpack.from_dlpack(arr)
        except Exception:
            new_arr = jnp.asarray(to_numpy(arr))

        # Move to requested JAX device.
        jax_device = dlpack_to_backend_device(xp_new, device)
        new_arr = jax.device_put(new_arr, jax_device)

        # Force copy if requested.
        if copy:
            new_arr = new_arr.copy()

        return new_arr

    # --- Generic Fallback ---
    return xp_new.from_dlpack(arr, copy=copy)


def _convert_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: Array,
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    """Convert a sparse matrix to be compatible with a new array namespace."""

    if isinstance(sparray, (scp.spmatrix, scp.sparray)):
        return _convert_scipy_sparse_for_namespace(
            xp_new, sparray, keep_sparse_compat, device=device
        )

    if cp is not None and isinstance(sparray, CupySpmatrix):
        return _convert_cupy_sparse_for_namespace(
            xp_new, sparray, keep_sparse_compat, device=device
        )

    if _is_torch_sparse_tensor(sparray):
        return _convert_torch_sparse_for_namespace(
            xp_new, sparray, keep_sparse_compat, device=device
        )

    if jsparse is not None and isinstance(
        sparray, (jsparse.BCOO, jsparse.COO, jsparse.CSR, jsparse.CSC)
    ):
        return _convert_jax_sparse_for_namespace(
            xp_new, sparray, keep_sparse_compat, device=device
        )

    raise TypeError("Sparse conversion not implemented for type '{}'.".format(type(sparray)))


def _convert_scipy_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: Union[scp.spmatrix, scp.sparray],
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    """Convert a scipy sparse matrix to be compatible with a new array namespace."""
    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp_new)

    device_type = int(device[0])
    device_id = int(device[1])

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.toarray(), device=device)

    fmt = sparray.format

    # --- Target: NumPy / array-api-strict ---
    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        # SciPy sparse is already NumPy-compatible, but only on CPU.
        if device_type != DLPACK_CPU:
            raise ValueError("NumPy does not support GPU sparse arrays.")
        return sparray

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        if fmt in ("csr", "csc"):
            f_create = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
            compressed_indices = torch.from_numpy(sparray.indptr.astype(np.int64, copy=False))
            plain_indices = torch.from_numpy(sparray.indices.astype(np.int64, copy=False))
            values = torch.from_numpy(sparray.data)
            sparray = f_create(compressed_indices, plain_indices, values, size=sparray.shape)

        else:
            if fmt != "coo":
                sparray = sparray.tocoo()
            values = torch.from_numpy(sparray.data)
            indices = torch.from_numpy(np.vstack((sparray.row, sparray.col)).astype(np.int64))
            sparray = torch.sparse_coo_tensor(indices, values, size=sparray.shape)

        # Move sparse tensor to requested CUDA device.
        if device_type == DLPACK_CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            sparray = sparray.to(device=torch.device("cuda", device_id))

        # Keep sparse tensor on CPU.
        elif device_type == DLPACK_CPU:
            pass
        else:
            raise ValueError(f"Unsupported DLPack device for PyTorch sparse tensor: {device}")

        return sparray

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )

        with cp.cuda.Device(device_id):
            try:
                f_create = getattr(csp, fmt + "_matrix")
            except AttributeError:
                sparray = sparray.tocoo()
                logger.warning(
                    f"Conversion of sparse matrix with format '{fmt}' to cupy sparse matrix "
                    "is not directly supported. Converting to 'coo' format first."
                )
                f_create = csp.coo_matrix
            return f_create(sparray)

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None and jsparse is not None:
        if device_type not in (DLPACK_CPU, DLPACK_CUDA):
            raise ValueError(
                f"JAX only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )
        jax_device = dlpack_to_backend_device(xp_new, device)
        try:
            if fmt == "csr":
                jax_sparse = jsparse.CSR(
                    (
                        jnp.asarray(sparray.data),
                        jnp.asarray(sparray.indices, dtype=jnp.int32),
                        jnp.asarray(sparray.indptr, dtype=jnp.int32),
                    ),
                    shape=sparray.shape,
                )
            elif fmt == "csc":
                jax_sparse = jsparse.CSC(
                    (
                        jnp.asarray(sparray.data),
                        jnp.asarray(sparray.indices, dtype=jnp.int32),
                        jnp.asarray(sparray.indptr, dtype=jnp.int32),
                    ),
                    shape=sparray.shape,
                )
            else:
                sparray = sparray.tocoo()
                jax_sparse = jsparse.COO(
                    (
                        jnp.asarray(sparray.data),
                        jnp.asarray(sparray.row, dtype=jnp.int32),
                        jnp.asarray(sparray.col, dtype=jnp.int32),
                    ),
                    shape=sparray.shape,
                )
        except Exception as exc:
            raise TypeError(
                "Could not convert scipy sparse matrix to a JAX sparse array. "
                f"Input sparse format was '{fmt}'."
            ) from exc
        return jax.device_put(jax_sparse, jax_device)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _convert_cupy_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: CupySpmatrix,
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp_new)

    device_type = int(device[0])
    device_id = int(device[1])

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.toarray(), device=device)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )
        # No format conversion needed, but the array may still live on another GPU
        if sparray.data.device.id != device_id:
            with cp.cuda.Device(device_id):
                sparray = sparray.copy()
        return sparray

    # --- Target: Numpy / Scipy / array-api-strict ---
    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        if device_type == DLPACK_CUDA:
            warnings.warn(
                "Converting CuPy sparse to SciPy sparse moves the array to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
        return sparray.get()

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        fmt = sparray.getformat()

        if fmt in ("csr", "csc"):
            f_create = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
            indptr = torch.utils.dlpack.from_dlpack(sparray.indptr.astype(np.int64, copy=False))
            indices = torch.utils.dlpack.from_dlpack(sparray.indices.astype(np.int64, copy=False))
            values = torch.utils.dlpack.from_dlpack(sparray.data)
            sparray = f_create(indptr, indices, values, size=sparray.shape)
        else:
            if fmt != "coo":
                sparray = sparray.tocoo()
            row = sparray.row.astype(cp.int64, copy=False)
            col = sparray.col.astype(cp.int64, copy=False)
            indices_cp = cp.stack((row, col), axis=0)

            indices = torch.utils.dlpack.from_dlpack(indices_cp)
            values = torch.utils.dlpack.from_dlpack(sparray.data)
            sparray = torch.sparse_coo_tensor(indices, values, size=sparray.shape)

        # Move to requested DLPack device if needed.
        if device_type == DLPACK_CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            sparray = sparray.to(device=dlpack_to_backend_device(xp_new, device))
        elif device_type == DLPACK_CPU:
            sparray = sparray.to(device=dlpack_to_backend_device(xp_new, device))
        else:
            raise ValueError(f"Unsupported DLPack device for PyTorch sparse tensor: {device}")

        return sparray

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None and jsparse is not None:
        if device_type not in (DLPACK_CPU, DLPACK_CUDA):
            raise ValueError(
                f"JAX only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )
        jax_device = dlpack_to_backend_device(xp_new, device)
        fmt = sparray.getformat()

        try:
            if fmt in ("csr", "csc"):
                sparse_type = jsparse.CSR if fmt == "csr" else jsparse.CSC
                data = jax.dlpack.from_dlpack(sparray.data)
                indices = jax.dlpack.from_dlpack(sparray.indices.astype(cp.int32, copy=False))
                indptr = jax.dlpack.from_dlpack(sparray.indptr.astype(cp.int32, copy=False))
                jax_sparse = sparse_type((data, indices, indptr), shape=sparray.shape)
            else:
                if fmt != "coo":
                    sparray = sparray.tocoo()
                data = jax.dlpack.from_dlpack(sparray.data)
                row = jax.dlpack.from_dlpack(sparray.row.astype(cp.int32, copy=False))
                col = jax.dlpack.from_dlpack(sparray.col.astype(cp.int32, copy=False))
                jax_sparse = jsparse.COO((data, row, col), shape=sparray.shape)
        except Exception as exc:
            raise TypeError(
                "Could not convert CuPy sparse matrix to a JAX sparse array. "
                f"Input sparse format was '{fmt}'."
            ) from exc

        return jax.device_put(jax_sparse, jax_device)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _convert_torch_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: torch.Tensor,
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    """Convert a PyTorch sparse tensor to be compatible with a new array namespace."""

    if not _is_torch_sparse_tensor(sparray):
        raise ValueError("Expected a PyTorch sparse tensor, got a dense tensor.")

    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp_new)

    device_type = int(device[0])
    device_id = int(device[1])

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.to_dense(), device=device)

    fmt = sparray.layout

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        if device_type == DLPACK_CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            return sparray.to(device=dlpack_to_backend_device(xp_new, device))
        elif device_type == DLPACK_CPU:
            return sparray.to(device=dlpack_to_backend_device(xp_new, device))
        raise ValueError(f"Unsupported DLPack device for PyTorch sparse tensor: {device}")

    # --- Target: NumPy / Scipy / array-api-strict ---
    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        if device_type != DLPACK_CPU:
            warnings.warn(
                "Converting PyTorch sparse to SciPy sparse moves the array to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Consider different sparse formats.
        sparray_cpu = sparray.cpu()

        if fmt == torch.sparse_csr:
            crow_indices = sparray_cpu.crow_indices().numpy()
            col_indices = sparray_cpu.col_indices().numpy()
            values = sparray_cpu.values().numpy()
            return scp.csr_array(
                (values, col_indices, crow_indices), shape=sparray_cpu.shape, copy=False
            )

        elif fmt == torch.sparse_csc:
            ccol_indices = sparray_cpu.ccol_indices().numpy()
            row_indices = sparray_cpu.row_indices().numpy()
            values = sparray_cpu.values().numpy()
            return scp.csc_array(
                (values, row_indices, ccol_indices), shape=sparray_cpu.shape, copy=False
            )

        else:
            if fmt != torch.sparse_coo:
                try:
                    sparray_cpu = sparray_cpu.to_sparse_coo()
                except RuntimeError as exc:
                    raise TypeError(
                        f"Unsupported PyTorch sparse layout for COO conversion: {fmt}"
                    ) from exc
            sparray_cpu = sparray_cpu.coalesce()
            values = sparray_cpu.values().numpy()
            indices = sparray_cpu.indices().numpy()
            return scp.coo_array(
                (values, (indices[0], indices[1])), shape=tuple(sparray_cpu.shape), copy=False
            )

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )

        # Depending if torch on GPU or CPU, proceed differently to avoid unnecessary GPU->CPU->GPU transfers.
        with cp.cuda.Device(device_id):
            if sparray.is_cuda:
                if sparray.device.index != device_id:
                    sparray = sparray.to(device=torch.device("cuda", device_id))

                if fmt == torch.sparse_csr:
                    crow_indices = cp.from_dlpack(sparray.crow_indices(), copy=False)
                    col_indices = cp.from_dlpack(sparray.col_indices(), copy=False)
                    values = cp.from_dlpack(sparray.values(), copy=False)
                    return csp.csr_matrix(
                        (values, col_indices, crow_indices), shape=tuple(sparray.shape), copy=False
                    )

                if fmt == torch.sparse_csc:
                    ccol_indices = cp.from_dlpack(sparray.ccol_indices(), copy=False)
                    row_indices = cp.from_dlpack(sparray.row_indices(), copy=False)
                    values = cp.from_dlpack(sparray.values(), copy=False)
                    return csp.csc_matrix(
                        (values, row_indices, ccol_indices), shape=tuple(sparray.shape), copy=False
                    )

                if fmt != torch.sparse_coo:
                    try:
                        sparray = sparray.to_sparse_coo()
                    except RuntimeError as exc:
                        raise TypeError(
                            f"Unsupported PyTorch sparse layout for COO conversion: {fmt}"
                        ) from exc

                sparray = sparray.coalesce()
                indices = cp.from_dlpack(sparray.indices(), copy=False)
                values = cp.from_dlpack(sparray.values(), copy=False)
                return csp.coo_matrix(
                    (values, (indices[0], indices[1])), shape=tuple(sparray.shape), copy=False
                )

            # CPU path: Torch CPU sparse -> NumPy/SciPy -> CuPy sparse.
            sp_cpu = sparray.detach().cpu()

            if fmt == torch.sparse_csr:
                crow_indices = sp_cpu.crow_indices().numpy()
                col_indices = sp_cpu.col_indices().numpy()
                values = sp_cpu.values().numpy()
                return csp.csr_matrix(
                    (values, col_indices, crow_indices), shape=tuple(sp_cpu.shape), copy=False
                )

            if fmt == torch.sparse_csc:
                ccol_indices = sp_cpu.ccol_indices().numpy()
                row_indices = sp_cpu.row_indices().numpy()
                values = sp_cpu.values().numpy()
                return csp.csc_matrix(
                    (values, row_indices, ccol_indices), shape=tuple(sp_cpu.shape), copy=False
                )

            if fmt != torch.sparse_coo:
                try:
                    sp_cpu = sp_cpu.to_sparse_coo()
                except RuntimeError as exc:
                    raise TypeError(
                        f"Unsupported PyTorch sparse layout for COO conversion: {fmt}"
                    ) from exc

            sp_cpu = sp_cpu.coalesce()
            indices = sp_cpu.indices().numpy()
            values = sp_cpu.values().numpy()
            return csp.coo_matrix(
                (values, (indices[0], indices[1])), shape=tuple(sp_cpu.shape), copy=False
            )

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None and jsparse is not None:
        if device_type not in (DLPACK_CPU, DLPACK_CUDA):
            raise ValueError(
                f"JAX only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )
        jax_device = dlpack_to_backend_device(xp_new, device)
        try:
            if fmt == torch.sparse_csr:
                values = jax.dlpack.from_dlpack(sparray.values().detach())
                indices = jax.dlpack.from_dlpack(
                    sparray.col_indices().to(dtype=torch.int32).detach()
                )
                indptr = jax.dlpack.from_dlpack(
                    sparray.crow_indices().to(dtype=torch.int32).detach()
                )
                jax_sparse = jsparse.CSR((values, indices, indptr), shape=tuple(sparray.shape))
            elif fmt == torch.sparse_csc:
                values = jax.dlpack.from_dlpack(sparray.values().detach())
                indices = jax.dlpack.from_dlpack(
                    sparray.row_indices().to(dtype=torch.int32).detach()
                )
                indptr = jax.dlpack.from_dlpack(
                    sparray.ccol_indices().to(dtype=torch.int32).detach()
                )
                jax_sparse = jsparse.CSC((values, indices, indptr), shape=tuple(sparray.shape))
            else:
                if fmt != torch.sparse_coo:
                    sparray = sparray.to_sparse_coo()
                sparray = sparray.coalesce()
                indices_torch = sparray.indices().to(dtype=torch.int32).detach()
                values = jax.dlpack.from_dlpack(sparray.values().detach())
                indices = jax.dlpack.from_dlpack(indices_torch)
                jax_sparse = jsparse.COO(
                    (values, indices[0], indices[1]), shape=tuple(sparray.shape)
                )

        except Exception as exc:
            raise TypeError(
                "Could not convert PyTorch sparse tensor to a JAX sparse array. "
                f"Input sparse layout was '{fmt}'."
            ) from exc

        return jax.device_put(jax_sparse, jax_device)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _convert_jax_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: Any,
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    """Convert a JAX sparse array to be compatible with a new array namespace."""

    if jsparse is None:
        raise ValueError("JAX sparse support is not available.")

    supported_types = (jsparse.BCOO, jsparse.COO, jsparse.CSR, jsparse.CSC)
    if not isinstance(sparray, supported_types):
        raise ValueError("Expected a JAX BCOO, COO, CSR, or CSC sparse array.")

    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp_new)

    device_type = int(device[0])
    device_id = int(device[1])

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.todense(), device=device)

    shape = tuple(sparray.shape)

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None:
        if device_type not in (DLPACK_CPU, DLPACK_CUDA):
            raise ValueError(
                f"JAX only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )
        return jax.device_put(sparray, dlpack_to_backend_device(xp_new, device))

    if len(shape) != 2:
        raise ValueError("Only 2D JAX sparse arrays can be converted to sparse matrix backends.")

    if isinstance(sparray, jsparse.CSR):
        fmt = "csr"
    elif isinstance(sparray, jsparse.CSC):
        fmt = "csc"
    else:
        fmt = "coo"

    values = sparray.data

    # --- Target: NumPy / SciPy / array-api-strict ---
    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        if device_type != DLPACK_CPU:
            warnings.warn(
                "Converting a JAX sparse array to SciPy moves the array to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )

        values_np = np.asarray(jax.device_get(values))

        if fmt in ("csr", "csc"):
            indices_np = np.asarray(jax.device_get(sparray.indices))
            indptr_np = np.asarray(jax.device_get(sparray.indptr))
            sparse_type = scp.csr_array if fmt == "csr" else scp.csc_array
            return sparse_type((values_np, indices_np, indptr_np), shape=shape, copy=False)

        if isinstance(sparray, jsparse.BCOO):
            indices_np = np.asarray(jax.device_get(sparray.indices))
            row_np, col_np = indices_np[:, 0], indices_np[:, 1]
        else:
            row_np = np.asarray(jax.device_get(sparray.row))
            col_np = np.asarray(jax.device_get(sparray.col))
        return scp.coo_array((values_np, (row_np, col_np)), shape=shape, copy=False)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device_type == DLPACK_CPU:
            raise ValueError("CuPy does not support CPU.")
        if device_type != DLPACK_CUDA:
            raise ValueError(
                f"CuPy only supports CUDA devices, got DLPack device type {device_type}."
            )

        with cp.cuda.Device(device_id):
            values_cp = cp.from_dlpack(values)

            if fmt in ("csr", "csc"):
                indices_cp = cp.from_dlpack(sparray.indices)
                indptr_cp = cp.from_dlpack(sparray.indptr)
                sparse_type = csp.csr_matrix if fmt == "csr" else csp.csc_matrix
                return sparse_type((values_cp, indices_cp, indptr_cp), shape=shape, copy=False)

            if isinstance(sparray, jsparse.BCOO):
                indices_cp = cp.from_dlpack(sparray.indices)
                row_cp, col_cp = indices_cp[:, 0], indices_cp[:, 1]
            else:
                row_cp = cp.from_dlpack(sparray.row)
                col_cp = cp.from_dlpack(sparray.col)
            return csp.coo_matrix((values_cp, (row_cp, col_cp)), shape=shape, copy=False)

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        if device_type not in (DLPACK_CPU, DLPACK_CUDA):
            raise ValueError(
                f"PyTorch only supports CPU and CUDA devices, got DLPack device type {device_type}."
            )

        target_torch_device = dlpack_to_backend_device(xp_new, device)

        values_torch = torch.utils.dlpack.from_dlpack(values)

        if fmt in ("csr", "csc"):
            indices_torch = torch.utils.dlpack.from_dlpack(sparray.indices).to(dtype=torch.int64)
            indptr_torch = torch.utils.dlpack.from_dlpack(sparray.indptr).to(dtype=torch.int64)
            sparse_type = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
            torch_sparse = sparse_type(indptr_torch, indices_torch, values_torch, size=shape)
        else:
            if isinstance(sparray, jsparse.BCOO):
                indices_torch = (
                    torch.utils.dlpack.from_dlpack(sparray.indices).to(dtype=torch.int64).T
                )
            else:
                row_torch = torch.utils.dlpack.from_dlpack(sparray.row)
                col_torch = torch.utils.dlpack.from_dlpack(sparray.col)
                indices_torch = torch.stack((row_torch, col_torch)).to(dtype=torch.int64)
            torch_sparse = torch.sparse_coo_tensor(indices_torch, values_torch, size=shape)

        return torch_sparse.to(device=target_torch_device)

    raise TypeError(
        "Conversion of JAX sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def is_sparse_array(arr: Any) -> bool:
    """
    Check if the array is a sparse array.

    Parameters
    ----------
    arr : Any
        The array to check.

    Returns
    -------
    bool
        True if the array is sparse, False otherwise.

    Raises
    ------
    TypeError
        If the type of arr is not supported.
    """

    if isinstance(arr, (scp.spmatrix, scp.sparray)):
        return True

    if cp is not None and csp.issparse(arr):
        return True

    if torch is not None and isinstance(arr, torch.Tensor):
        return _is_torch_sparse_tensor(arr)

    if jsparse is not None and isinstance(arr, jsparse.JAXSparse):
        return True

    if array_api_compat.is_array_api_obj(arr):
        if array_api_compat.is_cupy_array(arr):
            return csp.issparse(arr)

        if array_api_compat.is_pydata_sparse_array(arr):
            return arr.issparse()

        return False

    raise TypeError("Sparse check helper not implemented for type '{}'.".format(type(arr)))


def _rebuild_scipy_csc_in_namespace(
    xp_new: ArrayNamespace,
    data: Array,
    indices: Array,
    indptr: Array,
    shape: tuple,
) -> Union[scp.spmatrix, scp.sparray]:
    """Reconstruct a CSC sparse matrix in *xp_new* from pre-converted component arrays.

    This is a low-level helper used by :meth:`Dij.to_namespace` to avoid uploading
    shared row-index arrays more than once when multiple dose quantities share the
    same CSC index storage (as assembled by the pencil-beam engine).

    Parameters
    ----------
    xp_new :
        Target array namespace (numpy, cupy, torch, …).
    data :
        Non-zero values already in *xp_new*.
    indices :
        CSC row-index array already in *xp_new*.
    indptr :
        CSC column-pointer array already in *xp_new*.
    shape :
        Matrix shape ``(nrows, ncols)``.

    Returns
    -------
        Sparse matrix compatible with *xp_new*.

    Raises
    ------
    TypeError
        If rebuilding is not yet supported for *xp_new*.
    """
    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        return scp.csc_array((data, indices, indptr), shape=shape, copy=False)

    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        return csp.csc_matrix((data, indices, indptr), shape=shape)

    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        return torch.sparse_csc_tensor(indptr, indices, data, size=shape)

    raise TypeError(
        f"Rebuilding a CSC sparse matrix in namespace '{xp_new.__name__}' is not yet supported."
    )

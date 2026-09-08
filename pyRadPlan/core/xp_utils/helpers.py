"""Helper functions for array namespace operations."""

from __future__ import annotations
from typing import ContextManager, Optional, Union, Any
from contextlib import nullcontext
import importlib

import logging
import re
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


def _linked_openblas_version() -> Optional[tuple[int, int, int]]:
    """Version of the OpenBLAS NumPy is linked against, or None (e.g. MKL builds)."""
    try:
        blas_config = np.__config__.CONFIG["Build Dependencies"]["blas"]
        openblas_info = blas_config.get("openblas configuration", "")
    except (AttributeError, KeyError, TypeError):
        return None

    version_match = re.search(r"OpenBLAS\s+(\d+)\.(\d+)\.(\d+)", str(openblas_info))
    if version_match is None:
        return None
    return tuple(int(g) for g in version_match.groups())


def openblas_has_gemm_race() -> bool:
    """
    Whether NumPy links an OpenBLAS affected by the multithreaded-GEMM race.

    The scipy-openblas 0.3.27 bundled with the NumPy 2.0 wheels can silently return
    corrupted, run-to-run varying elements from multithreaded float32 matmuls with
    tall-skinny operands (observed for ``(N, 3) @ (3, 3)`` coordinate rotations).
    The upstream thread-race fixes landed in OpenBLAS 0.3.28; the 0.3.31 bundled with
    current NumPy wheels is verified clean.
    """
    version = _linked_openblas_version()
    return version is not None and version < (0, 3, 28)


def warn_on_unreliable_openblas() -> None:
    """Warn when NumPy is linked against an OpenBLAS with the multithreaded-GEMM race."""
    if openblas_has_gemm_race():
        warnings.warn(
            f"NumPy {np.__version__} is linked against OpenBLAS "
            f"{'.'.join(str(v) for v in _linked_openblas_version())}, whose "
            "multithreaded GEMM can silently return corrupted, nondeterministic results "
            "for some operand shapes. Consider upgrading NumPy (newer wheels bundle a "
            "fixed OpenBLAS) or setting OPENBLAS_NUM_THREADS=1.",
            RuntimeWarning,
            stacklevel=2,
        )


def is_host_device(device: Any) -> bool:
    """
    Check whether a device specification refers to a host (CPU) device.

    Accepts anything :func:`_parse_device_to_dlpack` understands (backend device
    objects, strings, DLPack tuples). ``None`` counts as host, since namespaces
    without a device concept (NumPy, array-api-strict) use it.
    """
    dlpack_device = _parse_device_to_dlpack(device)
    return dlpack_device is None or dlpack_device[0] == DLPACK_CPU


def device_cache_key(arr: Any) -> Any:
    """
    Hashable identity of the device an array lives on, usable as a dict key.

    Backend device objects cannot simply be used as dictionary keys: ``cupy.cuda.Device``
    is not hashable. The DLPack ``(device_type, device_id)`` tuple is always hashable but
    is not always distinguishing either -- every ``array_api_strict`` logical device
    reports ``(1, 0)``. This therefore prefers the backend's own device object, which
    identifies the device exactly, and falls back to the DLPack tuple only for the
    backends whose device objects are unhashable (where the tuple does distinguish the
    devices, since it carries the GPU index).

    Parameters
    ----------
    arr : Array
        Array whose device should be identified.

    Returns
    -------
    Hashable
        A value that compares equal for two arrays on the same device and differs for
        arrays on different devices of the same backend.
    """
    device = array_api_compat.device(arr)
    try:
        hash(device)
    except TypeError:
        return get_device_info(arr)
    return device


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


def stream_wait_event(
    xp: ArrayNamespace, stream: Optional[ContextManager], event: Optional[object]
) -> None:
    """
    Make ``stream`` wait device-side on ``event`` without blocking the host.

    Unlike :func:`synchronize`, this only enqueues an ordering constraint: work
    submitted to ``stream`` afterwards runs once ``event`` has completed, while
    the host keeps enqueueing. A no-op for host timestamps returned by
    :func:`record_event` on backends without stream support.
    """
    if event is None or stream is None or isinstance(stream, nullcontext):
        return

    if array_api_compat.is_cupy_namespace(xp) and isinstance(event, cp.cuda.Event):
        stream_obj = getattr(stream, "stream", stream)
        stream_obj.wait_event(event)
        return

    if (
        array_api_compat.is_torch_namespace(xp)
        and torch.cuda.is_available()
        and isinstance(event, torch.cuda.Event)
    ):
        # torch.cuda.stream() returns a StreamContext — delegate to the inner Stream
        stream_obj = getattr(stream, "stream", stream)
        if stream_obj is not None:
            stream_obj.wait_event(event)
        return

    # timer()-based pseudo events (numpy/jax/torch-cpu) need no device-side ordering


def elapsed_time(xp: ArrayNamespace, start, end) -> float:
    """Calculate the elapsed time between two events or timestamps."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.get_elapsed_time(start, end) / 1000.0  # Convert ms to s
    if array_api_compat.is_torch_namespace(xp):
        if isinstance(end, torch.Event):
            td = timedelta(milliseconds=start.elapsed_time(end)).total_seconds()
        else:
            td = timedelta(seconds=(end - start)).total_seconds()
        return td
    else:
        # Assuming start and end are timestamps from timer()
        return (timedelta(seconds=(end - start))).total_seconds()


def scatter(arr: Array, key: Any, values: Any) -> Array:
    """
    Backend-agnostic ``arr[key] = values``.

    Parameters
    ----------
    arr : Array
        The array to write into.
    key : Any
        Index expression (integer array, boolean mask, slice, ...).
    values : Any
        Values to assign at ``key``.

    Returns
    -------
    Array
        The updated array. The update can be out-of-place (JAX arrays are immutable,
        and array-api-strict forbids integer-array assignment), so callers must use
        the returned array; other backends assign in place and return ``arr`` itself.
    """
    if array_api_compat.is_jax_array(arr):
        return arr.at[key].set(values)
    try:
        arr[key] = values
        return arr
    except IndexError:
        xp = array_api_compat.array_namespace(arr)
        np_arr = to_numpy(arr).copy()
        np_arr[to_numpy(key) if array_api_compat.is_array_api_obj(key) else key] = (
            to_numpy(values) if array_api_compat.is_array_api_obj(values) else values
        )
        return xp.asarray(np_arr)


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


def _ensure_torch_compatible_strides(arr: Any) -> Any:
    """Materialize a NumPy view that has negative strides.

    PyTorch supports neither ``torch.from_numpy`` nor a DLPack import of a negatively strided
    array; the DLPack path aborts the process instead of raising on some builds. Reversed views
    (``arr[::-1]``, ``np.flip``, ``np.rot90``) therefore have to be copied before conversion.
    """
    if isinstance(arr, np.ndarray) and arr.strides and min(arr.strides) < 0:
        return np.ascontiguousarray(arr)
    return arr


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
        t = torch.from_numpy(_ensure_torch_compatible_strides(arr))
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

        arr = _ensure_torch_compatible_strides(arr)

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


def _resolve_sparse_device(xp_new: ArrayNamespace, device: Any) -> tuple[int, int]:
    """Resolve a device specification into the DLPack ``(device_type, device_id)`` to convert to.

    Falls back to the namespace's default device when no device is requested.
    """
    device = _parse_device_to_dlpack(device)

    if device is None:
        device = _default_dlpack_device_for_namespace(xp_new)

    return int(device[0]), int(device[1])


def _require_cuda_device(device_type: int, backend: str = "CuPy") -> None:
    """Reject DLPack device types other than CUDA for backends that only run on CUDA."""
    if device_type == DLPACK_CPU:
        raise ValueError(f"{backend} does not support CPU.")
    if device_type != DLPACK_CUDA:
        raise ValueError(
            f"{backend} only supports CUDA devices, got DLPack device type {device_type}."
        )


def _require_cpu_or_cuda_device(device_type: int, backend: str) -> None:
    """Reject DLPack device types other than CPU and CUDA."""
    if device_type not in (DLPACK_CPU, DLPACK_CUDA):
        raise ValueError(
            f"{backend} only supports CPU and CUDA devices, got DLPack device type {device_type}."
        )


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
    device = _resolve_sparse_device(xp_new, device)
    device_type, device_id = device

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

        # Move sparse tensor to requested CUDA device; a CPU target needs no move.
        _require_cpu_or_cuda_device(device_type, "PyTorch")
        if device_type == DLPACK_CUDA:
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            sparray = sparray.to(device=torch.device("cuda", device_id))

        return sparray

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        _require_cuda_device(device_type)

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
        _require_cpu_or_cuda_device(device_type, "JAX")
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
    """Convert a CuPy sparse matrix to be compatible with a new array namespace."""
    device = _resolve_sparse_device(xp_new, device)
    device_type, device_id = device

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.toarray(), device=device)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        _require_cuda_device(device_type)
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
        _require_cpu_or_cuda_device(device_type, "PyTorch")
        if device_type == DLPACK_CUDA and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but not available for PyTorch.")
        return _cupy_sparse_to_torch(sparray, xp_new, device)

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None and jsparse is not None:
        _require_cpu_or_cuda_device(device_type, "JAX")
        return _cupy_sparse_to_jax(sparray, xp_new, device)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _cupy_sparse_to_torch(
    sparray: CupySpmatrix, xp_new: ArrayNamespace, device: tuple[int, int]
) -> torch.Tensor:
    """Convert a CuPy sparse matrix to a PyTorch sparse tensor on the requested device."""
    fmt = sparray.getformat()

    if fmt in ("csr", "csc"):
        f_create = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
        indptr = torch.utils.dlpack.from_dlpack(sparray.indptr.astype(np.int64, copy=False))
        indices = torch.utils.dlpack.from_dlpack(sparray.indices.astype(np.int64, copy=False))
        values = torch.utils.dlpack.from_dlpack(sparray.data)
        torch_sparse = f_create(indptr, indices, values, size=sparray.shape)
    else:
        if fmt != "coo":
            sparray = sparray.tocoo()
        row = sparray.row.astype(cp.int64, copy=False)
        col = sparray.col.astype(cp.int64, copy=False)
        indices = torch.utils.dlpack.from_dlpack(cp.stack((row, col), axis=0))
        values = torch.utils.dlpack.from_dlpack(sparray.data)
        torch_sparse = torch.sparse_coo_tensor(indices, values, size=sparray.shape)

    return torch_sparse.to(device=dlpack_to_backend_device(xp_new, device))


def _cupy_sparse_to_jax(
    sparray: CupySpmatrix, xp_new: ArrayNamespace, device: tuple[int, int]
) -> Any:
    """Convert a CuPy sparse matrix to a JAX sparse array on the requested device."""
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

    return jax.device_put(jax_sparse, dlpack_to_backend_device(xp_new, device))


def _convert_torch_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: torch.Tensor,
    keep_sparse_compat: bool,
    device: Any = None,
) -> Array:
    """Convert a PyTorch sparse tensor to be compatible with a new array namespace."""

    if not _is_torch_sparse_tensor(sparray):
        raise ValueError("Expected a PyTorch sparse tensor, got a dense tensor.")

    device = _resolve_sparse_device(xp_new, device)
    device_type, device_id = device

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.to_dense(), device=device)

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        _require_cpu_or_cuda_device(device_type, "PyTorch")
        if device_type == DLPACK_CUDA and not torch.cuda.is_available():
            raise RuntimeError("GPU requested but not available for PyTorch.")
        return sparray.to(device=dlpack_to_backend_device(xp_new, device))

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
        return _torch_sparse_to_scipy(sparray)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        _require_cuda_device(device_type)
        return _torch_sparse_to_cupy(sparray, device_id)

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None and jsparse is not None:
        _require_cpu_or_cuda_device(device_type, "JAX")
        return _torch_sparse_to_jax(sparray, xp_new, device)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _torch_sparse_components(sparray: torch.Tensor) -> tuple[Any, tuple[torch.Tensor, ...]]:
    """Split a PyTorch sparse tensor into its layout and its constituent tensors.

    Layouts other than CSR and CSC are coalesced into COO first, so the returned tensors are
    ``(values, col_indices, crow_indices)`` for CSR, ``(values, row_indices, ccol_indices)`` for
    CSC and ``(values, indices)`` for COO -- the ``(data, indices, indptr)`` order that the
    SciPy, CuPy and JAX sparse constructors expect.
    """
    fmt = sparray.layout

    if fmt == torch.sparse_csr:
        return fmt, (sparray.values(), sparray.col_indices(), sparray.crow_indices())

    if fmt == torch.sparse_csc:
        return fmt, (sparray.values(), sparray.row_indices(), sparray.ccol_indices())

    if fmt != torch.sparse_coo:
        try:
            sparray = sparray.to_sparse_coo()
        except RuntimeError as exc:
            raise TypeError(
                f"Unsupported PyTorch sparse layout for COO conversion: {fmt}"
            ) from exc

    sparray = sparray.coalesce()
    return torch.sparse_coo, (sparray.values(), sparray.indices())


def _torch_sparse_to_scipy(sparray: torch.Tensor) -> Union[scp.spmatrix, scp.sparray]:
    """Convert a PyTorch sparse tensor to the matching SciPy sparse array on the host."""
    shape = tuple(sparray.shape)
    fmt, parts = _torch_sparse_components(sparray.cpu())
    values = parts[0].numpy()

    if fmt == torch.sparse_csr:
        return scp.csr_array((values, parts[1].numpy(), parts[2].numpy()), shape=shape, copy=False)

    if fmt == torch.sparse_csc:
        return scp.csc_array((values, parts[1].numpy(), parts[2].numpy()), shape=shape, copy=False)

    indices = parts[1].numpy()
    return scp.coo_array((values, (indices[0], indices[1])), shape=shape, copy=False)


def _torch_sparse_to_cupy(sparray: torch.Tensor, device_id: int) -> CupySpmatrix:
    """Convert a PyTorch sparse tensor to a CuPy sparse matrix on the given CUDA device."""
    shape = tuple(sparray.shape)

    with cp.cuda.Device(device_id):
        # A tensor already on a GPU is handed over via DLPack; routing it through the host
        # would cost an unnecessary GPU->CPU->GPU transfer. A CPU tensor is uploaded via
        # NumPy: the cupyx constructors reject host arrays.
        if sparray.is_cuda:
            if sparray.device.index != device_id:
                sparray = sparray.to(device=torch.device("cuda", device_id))
            fmt, parts = _torch_sparse_components(sparray)
            arrays = tuple(cp.from_dlpack(part) for part in parts)
        else:
            fmt, parts = _torch_sparse_components(sparray.detach().cpu())
            arrays = tuple(cp.asarray(part.numpy()) for part in parts)

        if fmt == torch.sparse_csr:
            return csp.csr_matrix(arrays, shape=shape, copy=False)

        if fmt == torch.sparse_csc:
            return csp.csc_matrix(arrays, shape=shape, copy=False)

        values, indices = arrays
        return csp.coo_matrix((values, (indices[0], indices[1])), shape=shape, copy=False)


def _torch_sparse_to_jax(
    sparray: torch.Tensor, xp_new: ArrayNamespace, device: tuple[int, int]
) -> Any:
    """Convert a PyTorch sparse tensor to a JAX sparse array on the requested device."""
    layout = sparray.layout

    try:
        shape = tuple(sparray.shape)
        fmt, parts = _torch_sparse_components(sparray)
        values = jax.dlpack.from_dlpack(parts[0].detach())
        indices = jax.dlpack.from_dlpack(parts[1].to(dtype=torch.int32).detach())

        if fmt in (torch.sparse_csr, torch.sparse_csc):
            indptr = jax.dlpack.from_dlpack(parts[2].to(dtype=torch.int32).detach())
            sparse_type = jsparse.CSR if fmt == torch.sparse_csr else jsparse.CSC
            jax_sparse = sparse_type((values, indices, indptr), shape=shape)
        else:
            jax_sparse = jsparse.COO((values, indices[0], indices[1]), shape=shape)
    except Exception as exc:
        raise TypeError(
            "Could not convert PyTorch sparse tensor to a JAX sparse array. "
            f"Input sparse layout was '{layout}'."
        ) from exc

    return jax.device_put(jax_sparse, dlpack_to_backend_device(xp_new, device))


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

    device = _resolve_sparse_device(xp_new, device)
    device_type, device_id = device

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.todense(), device=device)

    # --- Target: JAX ---
    if array_api_compat.is_jax_namespace(xp_new) and jax is not None:
        _require_cpu_or_cuda_device(device_type, "JAX")
        return jax.device_put(sparray, dlpack_to_backend_device(xp_new, device))

    if len(sparray.shape) != 2:
        raise ValueError("Only 2D JAX sparse arrays can be converted to sparse matrix backends.")

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
        return _jax_sparse_to_scipy(sparray)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        _require_cuda_device(device_type)
        return _jax_sparse_to_cupy(sparray, device_id)

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        _require_cpu_or_cuda_device(device_type, "PyTorch")
        return _jax_sparse_to_torch(sparray, xp_new, device)

    raise TypeError(
        "Conversion of JAX sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _jax_sparse_format(sparray: Any) -> str:
    """Return the sparse format ('csr', 'csc' or 'coo') of a JAX sparse array."""
    if isinstance(sparray, jsparse.CSR):
        return "csr"
    if isinstance(sparray, jsparse.CSC):
        return "csc"
    return "coo"


def _jax_sparse_coo_indices(sparray: Any) -> tuple[Any, Any]:
    """Return the row and column index arrays of a JAX COO or BCOO sparse array."""
    if isinstance(sparray, jsparse.BCOO):
        return sparray.indices[:, 0], sparray.indices[:, 1]
    return sparray.row, sparray.col


def _jax_sparse_to_scipy(sparray: Any) -> Union[scp.spmatrix, scp.sparray]:
    """Convert a JAX sparse array to the matching SciPy sparse array on the host."""
    shape = tuple(sparray.shape)
    fmt = _jax_sparse_format(sparray)
    values = np.asarray(jax.device_get(sparray.data))

    if fmt in ("csr", "csc"):
        indices = np.asarray(jax.device_get(sparray.indices))
        indptr = np.asarray(jax.device_get(sparray.indptr))
        sparse_type = scp.csr_array if fmt == "csr" else scp.csc_array
        return sparse_type((values, indices, indptr), shape=shape, copy=False)

    row, col = _jax_sparse_coo_indices(sparray)
    row = np.asarray(jax.device_get(row))
    col = np.asarray(jax.device_get(col))
    return scp.coo_array((values, (row, col)), shape=shape, copy=False)


def _jax_sparse_to_cupy(sparray: Any, device_id: int) -> CupySpmatrix:
    """Convert a JAX sparse array to a CuPy sparse matrix on the given CUDA device."""
    shape = tuple(sparray.shape)
    fmt = _jax_sparse_format(sparray)

    with cp.cuda.Device(device_id):
        values = cp.from_dlpack(sparray.data)

        if fmt in ("csr", "csc"):
            indices = cp.from_dlpack(sparray.indices)
            indptr = cp.from_dlpack(sparray.indptr)
            sparse_type = csp.csr_matrix if fmt == "csr" else csp.csc_matrix
            return sparse_type((values, indices, indptr), shape=shape, copy=False)

        row, col = _jax_sparse_coo_indices(sparray)
        return csp.coo_matrix(
            (values, (cp.from_dlpack(row), cp.from_dlpack(col))), shape=shape, copy=False
        )


def _jax_sparse_to_torch(
    sparray: Any, xp_new: ArrayNamespace, device: tuple[int, int]
) -> torch.Tensor:
    """Convert a JAX sparse array to a PyTorch sparse tensor on the requested device."""
    shape = tuple(sparray.shape)
    fmt = _jax_sparse_format(sparray)
    values = torch.utils.dlpack.from_dlpack(sparray.data)

    if fmt in ("csr", "csc"):
        indices = torch.utils.dlpack.from_dlpack(sparray.indices).to(dtype=torch.int64)
        indptr = torch.utils.dlpack.from_dlpack(sparray.indptr).to(dtype=torch.int64)
        sparse_type = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
        torch_sparse = sparse_type(indptr, indices, values, size=shape)
    else:
        row, col = _jax_sparse_coo_indices(sparray)
        indices = torch.stack(
            (
                torch.utils.dlpack.from_dlpack(row),
                torch.utils.dlpack.from_dlpack(col),
            )
        ).to(dtype=torch.int64)
        torch_sparse = torch.sparse_coo_tensor(indices, values, size=shape)

    return torch_sparse.to(device=dlpack_to_backend_device(xp_new, device))


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

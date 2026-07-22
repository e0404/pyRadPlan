"""Helper functions for array namespace operations."""

from typing import ContextManager, Optional, Union, Any, cast
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

import array_api_compat

import numpy as np

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


try:
    # NumPy >= 2.1 supports device/copy kwargs on from_dlpack
    np.from_dlpack(np.empty(0), device="cpu")
    _NP_DLPACK_HAS_DEVICE = True
except TypeError:
    _NP_DLPACK_HAS_DEVICE = False


def get_device_info(arr: Any) -> tuple:
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
            return arr.__dlpack_device__()
        except Exception:
            pass

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

    # Unknown -- assume CPU
    return (DLPACK_CPU, 0)


def is_on_gpu(arr: Any) -> bool:
    """Return True if the array resides on a GPU device."""
    device_type, _ = get_device_info(arr)
    return device_type in _GPU_DEVICE_TYPES


def _parse_device(device: Optional[str]) -> Optional[str]:
    """Normalize a device string to the internal convention."""
    if device is None:
        return None
    device = device.lower()
    if device.startswith("cuda"):
        device = "gpu" + device[4:]  # preserves ':N' suffix
    return device


def _device_index(device: Optional[str]) -> int:
    """Extract the GPU device index from a normalized device string."""
    if device is None or ":" not in device:
        return 0
    return int(device.split(":")[1])


def _device_index_from_array(arr: Any) -> int:
    """Return the GPU device index of an existing array via DLPack."""
    _, device_id = get_device_info(arr)
    return device_id


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
        try:
            stream.synchronize()
        except AttributeError:
            # torch.cuda.stream() returns a StreamContext — delegate to the inner Stream
            try:
                stream.stream.synchronize()
            except AttributeError:
                warnings.warn("The provided stream does not support synchronization.")
        finally:
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


def to_numpy(arr: Array, detach: bool = True) -> NDArray:
    """Convert an array to a NumPy array."""
    if array_api_compat.is_torch_array(arr):
        if detach and arr.requires_grad:
            arr = arr.detach()
        return arr.cpu().numpy()
    if array_api_compat.is_cupy_array(arr):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def from_numpy(xp: ArrayNamespace, arr: np.ndarray, *, device: Optional[str] = None) -> Array:
    """Convert a NumPy array to the specified array namespace.

    Parameters
    ----------
    xp : ArrayNamespace
        The target array namespace.
    arr : np.ndarray
        The NumPy array to convert.
    device : Optional[str], optional
        Target device. Accepts ``"cpu"``, ``"gpu"``, ``"gpu:N"``,
        or ``"cuda:N"`` (normalized automatically). Default ``None``
        uses the namespace default (CPU for NumPy/torch, current CuPy device).
    """
    device = _parse_device(device)

    if array_api_compat.is_cupy_namespace(xp):
        if device == "cpu":
            raise ValueError("CuPy does not support CPU.")
        gpu_idx = _device_index(device)
        with cp.cuda.Device(gpu_idx):
            return cp.asarray(arr)
    elif array_api_compat.is_torch_namespace(xp):
        t = torch.from_numpy(arr)
        if device is not None and device.startswith("gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            gpu_idx = _device_index(device)
            t = t.to(device=torch.device("cuda", gpu_idx))
        return t
    elif array_api_compat.is_numpy_namespace(xp) or array_api_compat.is_array_api_strict_namespace(
        xp
    ):
        return xp.asarray(arr)
    else:
        raise TypeError(
            "Conversion helper from NumPy not implemented for namespace '{}'.".format(xp.__name__)
        )


def to_namespace(
    xp_new: Union[ArrayNamespace, str],
    arr: Array,
    *,
    copy: Optional[bool] = None,
    keep_sparse_compat: bool = True,
    device: Optional[str] = None,
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
    device : Optional[str], optional
        The target device for the array. Can be "cpu" or "gpu".
        If None, the device is inferred from the source array or the target namespace default.

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

    # Normalize device — preserves ':N' device index
    device = _parse_device(device)

    if is_sparse_array(arr):
        return _convert_sparse_for_namespace(
            xp_new, arr, keep_sparse_compat=keep_sparse_compat, device=device
        )

    # Optimization: Same namespace and compatible device
    xp_old = array_api_compat.array_namespace(arr)
    if xp_new == xp_old:
        # Check if device movement is needed
        if device is None:
            if copy:
                # If copy is requested, we can't just return arr.
                # But we can let the specific handlers below handle it, or do it here.
                # For simplicity, let's fall through if copy is True.
                pass
            else:
                return arr
        else:
            # Device is specified. Check if we are already on that exact device.
            if is_on_gpu(arr):
                current_token = f"gpu:{_device_index_from_array(arr)}"
            else:
                current_token = "cpu"

            # Normalize target: "gpu" without index matches "gpu:0"
            target_token = (
                device if ":" in device else (f"{device}:0" if device == "gpu" else device)
            )

            if current_token == target_token:
                if not copy:
                    return arr

    # --- Target: NumPy ---
    if array_api_compat.is_numpy_namespace(xp_new):
        if device is not None and device.startswith("gpu"):
            raise ValueError("NumPy does not support GPU.")

        np_arr = to_numpy(arr)
        if copy is True:
            return xp_new.array(np_arr, copy=True)
        return xp_new.asarray(np_arr)

    # --- Target: CuPy ---
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device == "cpu":
            raise ValueError("CuPy does not support CPU.")

        gpu_idx = _device_index(device)
        with cp.cuda.Device(gpu_idx):
            # CuPy < 14.0 workaround for CPU->GPU via DLPack
            if cp.__version__ < "14.0" and not is_on_gpu(arr):
                return cp.asarray(to_numpy(arr))

            try:
                return xp_new.from_dlpack(arr, copy=copy)
            except Exception:
                return cp.asarray(to_numpy(arr))

    # --- Target: PyTorch ---
    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        # 1. Convert to Torch (preserving device if possible)
        try:
            if hasattr(xp_new, "from_dlpack"):
                new_arr = xp_new.from_dlpack(arr)
            else:
                new_arr = torch.utils.dlpack.from_dlpack(arr)
        except Exception:
            new_arr = torch.from_numpy(to_numpy(arr))

        # 2. Handle Device
        if device is not None and device.startswith("gpu"):
            if not new_arr.is_cuda:
                if not torch.cuda.is_available():
                    raise RuntimeError("GPU requested but not available for PyTorch.")
                gpu_idx = _device_index(device)
                new_arr = new_arr.to(device=torch.device("cuda", gpu_idx))
        elif device == "cpu":
            if new_arr.is_cuda:
                new_arr = new_arr.cpu()

        # 3. Handle Copy
        if copy:
            new_arr = new_arr.clone()

        return new_arr

    # --- Generic Fallback ---
    return xp_new.from_dlpack(arr, copy=copy)


def _convert_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: Array,
    keep_sparse_compat: bool,
    device: Optional[str] = None,
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


def _convert_scipy_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: Union[scp.spmatrix, scp.sparray],
    keep_sparse_compat: bool,
    device: Optional[str] = None,
) -> Array:
    """Convert a scipy sparse matrix to be compatible with a new array namespace."""

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.toarray(), device=device)

    fmt = sparray.format
    # For now, we always go the coo route

    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        if device == "gpu":
            raise ValueError("NumPy does not support GPU.")
        return sparray  # No conversion needed, scipy sparse is compatible with numpy

    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        if fmt in ("csr", "csc"):
            f_create = torch.sparse_csr_tensor if fmt == "csr" else torch.sparse_csc_tensor
            crow_indices = torch.from_numpy(sparray.indptr.astype(np.int64, copy=False))
            col_indices = torch.from_numpy(sparray.indices.astype(np.int64, copy=False))
            values = torch.from_numpy(sparray.data)
            sparray = f_create(crow_indices, col_indices, values, size=sparray.shape)
        else:
            if fmt != "coo":
                sparray = sparray.tocoo()
            values = torch.from_numpy(sparray.data)
            indices = torch.from_numpy(np.vstack((sparray.row, sparray.col)).astype(np.int64))
            sparray = torch.sparse_coo_tensor(indices, values, size=sparray.shape)

        if device is not None and device.startswith("gpu"):
            if not torch.cuda.is_available():
                raise RuntimeError("GPU requested but not available for PyTorch.")
            gpu_idx = _device_index(device)
            sparray = sparray.to(device=torch.device("cuda", gpu_idx))

        return sparray

    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        if device == "cpu":
            raise ValueError("CuPy does not support CPU.")
        try:
            # Access the copy constructor from the fmt string
            f_create = getattr(csp, fmt + "_matrix")
        except AttributeError:
            sparray = sparray.tocoo()
            logger.warning(
                f"Conversion of sparse matrix with format '{fmt}' to cupy sparse matrix "
                "is not directly supported. Converting to 'coo' format first."
            )
            f_create = csp.coo_matrix
        return f_create(sparray)

    raise TypeError(
        "Conversion of sparse matrix to namespace '{}' is not yet supported.".format(
            xp_new.__name__
        )
    )


def _convert_cupy_sparse_for_namespace(
    xp_new: ArrayNamespace,
    sparray: CupySpmatrix,
    keep_sparse_compat: bool,
    device: Optional[str] = None,
) -> Array:
    if array_api_compat.is_cupy_namespace(xp_new):
        if device == "cpu":
            raise ValueError("CuPy does not support CPU.")
        return sparray  # No conversion needed

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.toarray(), device=device)

    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        if device == "gpu":
            raise ValueError("NumPy does not support GPU.")
        return sparray.get()

    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        fmt = sparray.getformat()
        if fmt == "coo":
            sparray = cast(csp.coo_matrix, sparray)
            row = sparray.row.astype(cp.int64, copy=False)
            col = sparray.col.astype(cp.int64, copy=False)
            data = sparray.data

            idx_cp = cp.stack([row, col], axis=0)

            idx_t = torch.utils.dlpack.from_dlpack(idx_cp.toDlpack())
            data_t = torch.utils.dlpack.from_dlpack(cp.asarray(data).toDlpack())

            # Device is inferred from the DLPack
            t_sp = torch.sparse_coo_tensor(idx_t, data_t, size=sparray.shape)

        elif fmt in ("csr", "csc"):
            if fmt == "csr":
                sparray = cast(csp.csr_matrix, sparray)
                f_create = torch.sparse_csr_tensor
            else:
                sparray = cast(csp.csc_matrix, sparray)
                f_create = torch.sparse_csc_tensor

            indptr = sparray.indptr.astype(cp.int64, copy=False)
            indices = sparray.indices.astype(cp.int64, copy=False)
            data = sparray.data

            indptr_t = torch.utils.dlpack.from_dlpack(indptr.toDlpack())
            indices_t = torch.utils.dlpack.from_dlpack(indices.toDlpack())
            data_t = torch.utils.dlpack.from_dlpack(data.toDlpack())

            # Device is inferred from the DLPack
            t_sp = f_create(indptr_t, indices_t, data_t, size=sparray.shape)
        else:
            raise TypeError(
                "Conversion of sparse matrix to format '{}' is not yet supported in PyTorch.".format(
                    fmt
                )
            )

        if device == "cpu":
            return t_sp.cpu()
        if device is not None and device.startswith("gpu"):
            gpu_idx = _device_index(device)
            return t_sp.to(device=torch.device("cuda", gpu_idx))

        return t_sp


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

    if array_api_compat.is_array_api_obj(arr):
        if array_api_compat.is_torch_array(arr):
            return arr.is_sparse
        if array_api_compat.is_cupy_array(arr):
            return csp.issparse(arr)
        if array_api_compat.is_pydata_sparse_array(arr):
            return arr.issparse()

        return False

    if isinstance(arr, (scp.spmatrix, scp.sparray)):
        return True

    if cp is not None and csp.issparse(arr):
        return True

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

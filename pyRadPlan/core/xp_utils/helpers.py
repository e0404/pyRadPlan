"""Helper functions for array namespace operations."""

from typing import ContextManager, Optional, Union, Any, cast
from contextlib import nullcontext
import importlib

import logging
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    CupySpmatrix = type(csp.spmatrix)
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


def get_current_stream(xp: ArrayNamespace) -> ContextManager:
    """Get the current stream based on the array namespace."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.get_current_stream()
    if array_api_compat.is_torch_namespace(xp):
        return torch.cpu.stream(torch.cpu.current_stream())
    else:
        return nullcontext()


def create_stream(xp: ArrayNamespace) -> ContextManager:
    """Create a context manager for the appropriate stream based on the array namespace."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.cuda.Stream(non_blocking=True)
    if array_api_compat.is_torch_namespace(xp):
        return torch.cpu.stream(torch.cpu.Stream())
    else:
        return nullcontext()


def synchronize(xp: ArrayNamespace, stream: Optional[ContextManager] = None) -> None:
    """Synchronize the device if using CuPy."""

    if stream is not None and not isinstance(stream, nullcontext):
        try:
            stream.synchronize()
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


def record_event(xp: ArrayNamespace, stream: Optional[ContextManager] = None) -> Optional[object]:
    """Record an event in the current stream if using CuPy or PyTorch."""
    if not isinstance(stream, nullcontext):
        if array_api_compat.is_cupy_namespace(xp):
            event = xp.cuda.Event()
            event.record(stream)
            return event
        if array_api_compat.is_torch_namespace(xp):
            try:
                if isinstance(stream, ContextManager):
                    event = stream.stream.record_event()
                elif stream is not None:
                    event = stream.record_event()
                else:
                    event = None

                if event is None:
                    event = timer()

                return event
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
    if array_api_compat.is_cupy_array(arr):
        return cp.asnumpy(arr)
    elif array_api_compat.is_torch_array(arr):
        if detach and arr.requires_grad:
            arr = arr.detach()
        return arr.cpu().numpy()
    elif array_api_compat.is_numpy_array(arr):
        return arr
    else:
        try:
            xnp: ArrayNamespace = array_api_compat.numpy
            return to_namespace(xnp, arr)
        except Exception as e:
            raise TypeError(
                "Conversion helper to NumPy not implemented for type '{}'.".format(type(arr))
            ) from e


def from_numpy(xp: ArrayNamespace, arr: np.ndarray) -> Array:
    """Convert a NumPy array to the specified array namespace."""
    if array_api_compat.is_cupy_namespace(xp):
        return cp.asarray(arr)
    elif array_api_compat.is_torch_namespace(xp):
        return torch.from_numpy(arr)
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
) -> Array:
    """
    Convert an array to the specified array namespace.

    Parameters
    ----------
    xp_new : Union[ArrayNamespace,str]
        The target array namespace or its name as a string.
    arr : Array
        The array to convert.
    copy : Optional[bool], optional
        Whether to force a copy during conversion, by default None.
        If None, the default behavior of the target namespace is used.
    keep_sparse_compat : bool, optional
        Whether to keep sparse array compatible, by default True.
        For example, when converting to numpy, scipy sparse matrices will not be converted to dense
        arrays because they are compatible with numpy.

    Returns
    -------
    Array
        The converted array.
    """

    if isinstance(xp_new, str):
        try:
            xp_new = importlib.import_module(xp_new, "array_api_compat")
        except ModuleNotFoundError:
            # Try to import the module directly
            xp_new = importlib.import_module(xp_new)

    if is_sparse_array(arr):
        return _convert_sparse_for_namespace(xp_new, arr, keep_sparse_compat=keep_sparse_compat)

    # Compatibility wrapper for cupy to facilitate copy to GPU
    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        # Check if cupy version is < 14.0
        if cp.__version__ < "14.0" and arr.__dlpack_device__()[0] != 2:
            return cp.asarray(to_numpy(arr))  # Direct conversion for older versions

    if array_api_compat.is_numpy_namespace(xp_new):
        if xp_new.__version__ < "2.1":
            return xp_new.from_dlpack(arr)

    xp_old = array_api_compat.array_namespace(arr)

    if xp_new == xp_old:
        return arr  # No conversion needed

    return xp_new.from_dlpack(arr, copy=copy)


def _convert_sparse_for_namespace(
    xp_new: ArrayNamespace, sparray: Array, keep_sparse_compat: bool
) -> Array:
    """Convert a sparse matrix to be compatible with a new array namespace."""

    if isinstance(sparray, (scp.spmatrix, scp.sparray)):
        return _convert_scipy_sparse_for_namespace(xp_new, sparray, keep_sparse_compat)

    if cp is not None and isinstance(sparray, CupySpmatrix):
        return _convert_cupy_sparse_for_namespace(xp_new, sparray, keep_sparse_compat)


def _convert_scipy_sparse_for_namespace(
    xp_new: ArrayNamespace, sparray: Union[scp.spmatrix, scp.sparray], keep_sparse_compat: bool
) -> Array:
    """Convert a scipy sparse matrix to be compatible with a new array namespace."""

    if not keep_sparse_compat:
        return from_numpy(xp_new, sparray.todense())

    fmt = sparray.format
    # For now, we always go the coo route

    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
        return sparray  # No conversion needed, scipy sparse is compatible with numpy

    if array_api_compat.is_torch_namespace(xp_new) and torch is not None:
        if fmt != "coo":
            sparray = sparray.tocoo()
        values = torch.from_numpy(sparray.data)
        indices = torch.from_numpy(np.vstack((sparray.row, sparray.col)).astype(np.int64))
        shape = sparray.shape
        sparray = torch.sparse_coo_tensor(indices, values, size=shape)

        format_func = getattr(torch.Tensor, "to_sparse_" + fmt, None)
        if format_func is not None:
            if fmt != "coo":
                sparray = format_func(sparray)
        else:
            warnings.warn(
                f"Conversion of sparse matrix to format '{fmt}' "
                "is not directly supported in PyTorch. Keeping 'coo' format."
            )
        return sparray

    if array_api_compat.is_cupy_namespace(xp_new) and cp is not None:
        try:
            # Access the copy constructur from the fmt string
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
    xp_new: ArrayNamespace, sparray: CupySpmatrix, keep_sparse_compat: bool
) -> Array:
    if array_api_compat.is_cupy_namespace(xp_new):
        return sparray  # No conversion needed

    if not keep_sparse_compat:
        return to_namespace(xp_new, sparray.todense())

    if array_api_compat.is_numpy_namespace(
        xp_new
    ) or array_api_compat.is_array_api_strict_namespace(xp_new):
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

            return torch.sparse_coo_tensor(idx_t, data_t, size=sparray.shape, device="cuda")

        if fmt in ("csr", "csc"):
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

            return f_create(indptr_t, indices_t, data_t, size=sparray.shape, device="cuda")

        raise TypeError(
            "Conversion of sparse matrix to format '{}' is not yet supported in PyTorch.".format(
                fmt
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

    if array_api_compat.is_array_api_obj(arr):
        if array_api_compat.is_torch_array(arr):
            return arr.is_sparse()
        if array_api_compat.is_cupy_array(arr):
            return csp.issparse(arr)
        if array_api_compat.is_pydata_sparse_array(arr):
            return arr.issparse()

        return False

    if isinstance(arr, (scp.spmatrix, scp.sparray)):
        return True

    raise TypeError("Sparse check helper not implemented for type '{}'.".format(type(arr)))

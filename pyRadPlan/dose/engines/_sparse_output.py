"""Index sizing for the shared pencil-beam sparse output collector."""

import numpy as np


_INT32_MAX = np.iinfo(np.int32).max


def sparse_index_dtype(num_rows: int, num_columns: int, nnz: int):
    """Choose a common CSC/CSR index dtype for dimensions and stored-entry count.

    Counts are Python integers; the total dense matrix size is irrelevant.
    The collector must recheck the entry count before writing each column pointer.
    """
    if min(num_rows, num_columns, nnz) < 0:
        raise ValueError("Sparse dimensions and entry count must be nonnegative.")
    return np.int32 if max(num_rows, num_columns, nnz) <= _INT32_MAX else np.int64


def ensure_index_capacity(structure: dict, end: int) -> None:
    """Promote shared row indices and column pointers before an append overflows.

    All quantities access these arrays through the shared structure, so replacing
    them together preserves sharing. This collector owns NumPy buffers; the dose
    computation backend and its voxel-selection index dtype are unaffected.
    """
    if structure["indptr"].dtype == np.int32 and end > _INT32_MAX:
        indices = structure["indices"].astype(np.int64)
        indptr = structure["indptr"].astype(np.int64)
        structure["indices"], structure["indptr"] = indices, indptr

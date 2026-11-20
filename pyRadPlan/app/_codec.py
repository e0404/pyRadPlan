"""JSON + binary buffers codec of the application protocol.

Arrays never travel inside JSON: :func:`encode` replaces every :class:`numpy.ndarray`
by ``{"$array": i, "dtype": ..., "shape": [...]}`` and appends its C-contiguous,
little-endian bytes to a buffer list; :func:`decode` reverses this.
"""

from __future__ import annotations

import math
import os
import sys
from enum import Enum
from typing import Any, Sequence

import numpy as np

ARRAY_KEY = "$array"


def _array_to_buffer(array: np.ndarray) -> memoryview:
    big_endian = array.dtype.byteorder == ">" or (
        array.dtype.byteorder == "=" and sys.byteorder == "big"
    )
    if big_endian:
        array = array.astype(array.dtype.newbyteorder("<"))
    return memoryview(np.ascontiguousarray(array).reshape(-1)).cast("B")


def _encode_float(value: float) -> Any:
    return value if math.isfinite(value) else None


def encode(obj: Any) -> tuple[Any, list[memoryview]]:
    """Split *obj* into a JSON-able structure and a list of binary buffers.

    Numpy arrays become ``{"$array": i, "dtype", "shape"}`` descriptors indexing the
    returned buffers (zero-copy ``memoryview`` objects where the array is C-contiguous
    and little-endian). Numpy scalars become Python numbers, tuples and sets lists,
    non-finite floats ``None`` (JSON has no NaN), paths and enums strings. Any other
    object raises :class:`TypeError`, so a non-serializable value is caught in Python
    rather than in the transport.

    Returns
    -------
    tuple
        ``(json_obj, buffers)``.
    """
    buffers: list[memoryview] = []

    def walk(value: Any) -> Any:
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return _encode_float(value)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [walk(v) for v in value.tolist()]
            if value.dtype.kind not in "biuf":
                raise TypeError(f"Cannot encode arrays of dtype {value.dtype}")
            buffers.append(_array_to_buffer(value))
            return {
                ARRAY_KEY: len(buffers) - 1,
                "dtype": np.dtype(value.dtype).newbyteorder("=").name,
                "shape": [int(n) for n in value.shape],
            }
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return _encode_float(float(value))
        if isinstance(value, dict):
            return {str(k): walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set, frozenset)):
            return [walk(v) for v in value]
        if isinstance(value, Enum):
            return walk(value.value)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        raise TypeError(f"Cannot encode object of type {type(value).__name__}")

    return walk(obj), buffers


def decode(obj: Any, buffers: Sequence[Any] = ()) -> Any:
    """Reverse :func:`encode`: replace array descriptors by arrays read from *buffers*.

    Buffers can be anything supporting the buffer protocol (``bytes``, ``bytearray``,
    ``memoryview``, numpy arrays; under Pyodide a converted ``ArrayBuffer``). The returned
    arrays are read-only views on the buffers, copy them before writing.
    """

    def walk(value: Any) -> Any:
        if isinstance(value, dict):
            if ARRAY_KEY in value:
                index = value[ARRAY_KEY]
                if not isinstance(index, int) or not 0 <= index < len(buffers):
                    raise ValueError(f"Array descriptor references missing buffer {index!r}")
                dtype = np.dtype(value.get("dtype", "uint8")).newbyteorder("<")
                array = np.frombuffer(buffers[index], dtype=dtype)
                shape = value.get("shape")
                if shape is not None:
                    array = array.reshape([int(n) for n in shape])
                return array
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(obj)

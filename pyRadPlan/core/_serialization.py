"""Lossless serialization of array-valued data to JSON and other storage backends.

Arrays, sparse matrices and SimpleITK images are replaced by tagged dictionaries (the *array
leaves* of a data tree). The tree walkers :func:`encode_tree` and :func:`decode_tree` are free
of any storage format: the leaf encoder and decoder are pluggable, so a different backend
(e.g. HDF5 datasets) can reuse them. :func:`json_leaf_encoder` and :func:`json_leaf_decoder`
write the leaves as plain JSON.

JSON leaf formats (every leaf is a dictionary with the key ``"__array__"``)::

    {"__array__": "ndarray", "dtype": "float64", "shape": [2, 3], "data": [... flat C order]}
    {"__array__": "ndarray", "dtype": "object", "shape": [2], "data": [<encoded leaf or None>, ...]}
    {"__array__": "sparse", "format": "csc", "container": "matrix" | "array", "dtype": ...,
     "shape": [rows, cols], "data": [...], "indices": [...], "indptr": [...]}
    {"__array__": "sitk.Image", "dtype": ..., "pixel_type": "32-bit float", "shape": [...],
     "data": [...], "origin": [...], "spacing": [...], "direction": [...], "components": 3}
    {"__array__": "float", "value": "nan" | "inf" | "-inf"}

The ``shape`` of an image is the shape of ``sitk.GetArrayFromImage`` (z, y, x[, components]),
``components`` is only present for vector pixel types. Complex data is stored as interleaved
real and imaginary parts.

Non-finite floating point numbers are written as the strings ``"NaN"``, ``"Infinity"`` and
``"-Infinity"`` inside the ``data`` of float arrays and as a ``"float"`` leaf elsewhere. The
output therefore stays standard JSON (no ``NaN`` constants), unlike pydantic's
``ser_json_inf_nan="constants"``, and unlike the default ``"null"`` mode it does not lose the
value.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import array_api_compat
import numpy as np
import scipy.sparse as sp
import SimpleITK as sitk

ARRAY_TAG = "__array__"

_NONFINITE_TO_TOKEN = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}
_TOKEN_TO_FLOAT = {
    "NaN": math.nan,
    "Infinity": math.inf,
    "-Infinity": -math.inf,
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}

LeafEncoder = Callable[[Any], Any]
LeafDecoder = Callable[[dict], Any]


def _nonfinite_name(value: float) -> str:
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def is_array_leaf(obj: Any) -> bool:
    """Whether ``obj`` is a leaf of a data tree: an array, sparse matrix, image or non-finite float.

    Parameters
    ----------
    obj : Any
        Object to test.

    Returns
    -------
    bool
        True for arrays of any array API backend, scipy sparse matrices and arrays,
        ``SimpleITK.Image`` and non-finite Python floats.
    """
    if isinstance(obj, float):
        return not math.isfinite(obj)
    if isinstance(obj, (np.generic, str, bytes, int, bool, type(None))):
        return False
    if isinstance(obj, (sitk.Image, np.ndarray)) or sp.issparse(obj):
        return True
    return array_api_compat.is_array_api_obj(obj)


def encode_tree(obj: Any, leaf_encoder: LeafEncoder | None = None) -> Any:
    """Replace the array leaves of a tree of dicts, lists and tuples.

    Parameters
    ----------
    obj : Any
        Tree of dictionaries, lists and tuples. Numpy scalars become Python scalars, tuples
        become lists. Everything that is not an array leaf is kept as it is.
    leaf_encoder : callable, optional
        Called with every array leaf (see :func:`is_array_leaf`), returns its replacement.
        Defaults to :func:`json_leaf_encoder`.

    Returns
    -------
    Any
        A new tree. Containers are copied, array leaves are not modified.
    """
    if leaf_encoder is None:
        leaf_encoder = json_leaf_encoder
    return _encode(obj, leaf_encoder)


def _encode(obj: Any, leaf_encoder: LeafEncoder) -> Any:
    if isinstance(obj, np.generic):
        obj = obj.item()
    if is_array_leaf(obj):
        return leaf_encoder(obj)
    if isinstance(obj, dict):
        return {key: _encode(value, leaf_encoder) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode(value, leaf_encoder) for value in obj]
    return obj


def decode_tree(obj: Any, leaf_decoder: LeafDecoder | None = None) -> Any:
    """Replace the tagged leaves of a tree of dicts, lists and tuples by arrays.

    Parameters
    ----------
    obj : Any
        Tree of dictionaries, lists and tuples.
    leaf_decoder : callable, optional
        Called with every dictionary that has the key ``"__array__"``, returns the decoded
        object. Defaults to :func:`json_leaf_decoder`.

    Returns
    -------
    Any
        A new tree. Containers without tagged leaves are returned unchanged and not copied.
    """
    if leaf_decoder is None:
        leaf_decoder = json_leaf_decoder
    return _decode(obj, leaf_decoder)


def _decode(obj: Any, leaf_decoder: LeafDecoder) -> Any:
    if isinstance(obj, dict):
        if ARRAY_TAG in obj:
            return leaf_decoder(obj)
        decoded = {key: _decode(value, leaf_decoder) for key, value in obj.items()}
        if all(decoded[key] is obj[key] for key in obj):
            return obj
        return decoded
    if isinstance(obj, (list, tuple)):
        decoded = [_decode(value, leaf_decoder) for value in obj]
        if all(new is old for new, old in zip(decoded, obj)):
            return obj
        return type(obj)(decoded) if isinstance(obj, tuple) else decoded
    return obj


def _float_data(values: list, flat: np.ndarray) -> list:
    """Replace non-finite entries of a flat list by their string tokens."""
    if flat.dtype.kind not in "fc" or flat.size == 0 or np.isfinite(flat).all():
        return values
    return [
        _NONFINITE_TO_TOKEN[_nonfinite_name(v)]
        if isinstance(v, float) and not math.isfinite(v)
        else v
        for v in values
    ]


def _flat_data(arr: np.ndarray) -> list:
    flat = arr.reshape(-1)
    if flat.dtype.kind == "c":
        flat = np.ascontiguousarray(flat).view(flat.real.dtype)
    return _float_data(flat.tolist(), flat)


def _unflat_data(data: list, dtype: np.dtype) -> np.ndarray:
    if dtype.kind == "c":
        real = np.asarray(_floats_from_tokens(data), dtype=np.dtype(f"f{dtype.itemsize // 2}"))
        return real.view(dtype)
    if dtype.kind == "f":
        return np.asarray(_floats_from_tokens(data), dtype=dtype)
    return np.asarray(data, dtype=dtype)


def _floats_from_tokens(data: list) -> list:
    return [_TOKEN_TO_FLOAT[v] if isinstance(v, str) else v for v in data]


def _to_numpy(arr: Any) -> np.ndarray:
    from .xp_utils import to_numpy

    return to_numpy(arr)


def _encode_ndarray(arr: Any) -> dict:
    arr = _to_numpy(arr)
    if arr.dtype.names is not None:
        raise TypeError("Structured arrays cannot be serialized")
    if arr.dtype.kind == "O":
        data = [_encode(item, json_leaf_encoder) for item in arr.reshape(-1)]
    else:
        data = _flat_data(arr)
    return {
        ARRAY_TAG: "ndarray",
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": data,
    }


def _encode_sparse(mat: Any) -> dict:
    csc = mat.tocsc()
    csc.sum_duplicates()
    return {
        ARRAY_TAG: "sparse",
        "format": "csc",
        "container": "matrix" if isinstance(mat, sp.spmatrix) else "array",
        "dtype": str(csc.dtype),
        "shape": [int(s) for s in csc.shape],
        "data": _flat_data(np.asarray(csc.data)),
        "indices": csc.indices.tolist(),
        "indptr": csc.indptr.tolist(),
    }


def _encode_image(image: sitk.Image) -> dict:
    arr = sitk.GetArrayViewFromImage(image)
    components = image.GetNumberOfComponentsPerPixel()
    out = {
        ARRAY_TAG: "sitk.Image",
        "dtype": str(arr.dtype),
        "pixel_type": image.GetPixelIDTypeAsString(),
        "shape": list(arr.shape),
        "data": _flat_data(np.ascontiguousarray(arr)),
        "origin": list(image.GetOrigin()),
        "spacing": list(image.GetSpacing()),
        "direction": list(image.GetDirection()),
    }
    if arr.ndim > image.GetDimension():
        out["components"] = components
    return out


def json_leaf_encoder(obj: Any) -> dict:
    """Encode an array leaf as a tagged, JSON compatible dictionary.

    Parameters
    ----------
    obj : Any
        Array of any array API backend (converted to numpy), scipy sparse matrix or array,
        ``SimpleITK.Image`` or non-finite float. Object arrays are encoded element by element.

    Returns
    -------
    dict
        See the module documentation for the formats.

    Raises
    ------
    TypeError
        For structured arrays.
    """
    if isinstance(obj, float):
        return {ARRAY_TAG: "float", "value": _nonfinite_name(obj)}
    if isinstance(obj, sitk.Image):
        return _encode_image(obj)
    if sp.issparse(obj):
        return _encode_sparse(obj)
    return _encode_ndarray(obj)


def _decode_ndarray(leaf: dict) -> np.ndarray:
    shape = tuple(leaf["shape"])
    dtype = np.dtype(leaf["dtype"])
    if dtype.kind == "O":
        out = np.empty(len(leaf["data"]), dtype=object)
        for i, item in enumerate(leaf["data"]):
            out[i] = _decode(item, json_leaf_decoder)
        return out.reshape(shape)
    return _unflat_data(leaf["data"], dtype).reshape(shape)


def _decode_sparse(leaf: dict) -> Any:
    if leaf.get("format", "csc") != "csc":
        raise ValueError(f"Unsupported sparse format {leaf['format']!r}")
    dtype = np.dtype(leaf["dtype"])
    cls = sp.csc_array if leaf.get("container") == "array" else sp.csc_matrix
    return cls(
        (
            _unflat_data(leaf["data"], dtype),
            np.asarray(leaf["indices"], dtype=np.int64),
            np.asarray(leaf["indptr"], dtype=np.int64),
        ),
        shape=tuple(leaf["shape"]),
    )


def _decode_image(leaf: dict) -> sitk.Image:
    arr = _unflat_data(leaf["data"], np.dtype(leaf["dtype"])).reshape(tuple(leaf["shape"]))
    components = leaf.get("components", 1)
    image = sitk.GetImageFromArray(arr, isVector=components > 1)
    pixel_type = leaf.get("pixel_type")
    if pixel_type is not None and image.GetPixelIDTypeAsString() != pixel_type:
        image = sitk.Cast(image, _pixel_id(pixel_type))
    image.SetOrigin(leaf["origin"])
    image.SetSpacing(leaf["spacing"])
    image.SetDirection(leaf["direction"])
    return image


def _pixel_id(pixel_type: str) -> int:
    for name in dir(sitk):
        if name.startswith("sitk"):
            value = getattr(sitk, name)
            if isinstance(value, int) and sitk.GetPixelIDValueAsString(value) == pixel_type:
                return value
    raise ValueError(f"Unknown pixel type {pixel_type!r}")


def json_leaf_decoder(leaf: dict) -> Any:
    """Decode a tagged dictionary written by :func:`json_leaf_encoder`.

    Parameters
    ----------
    leaf : dict
        Dictionary with the key ``"__array__"``.

    Returns
    -------
    Any
        A numpy array, a ``scipy.sparse`` CSC matrix (array if the source was one), a
        ``SimpleITK.Image`` or a float. Arrays are always decoded to numpy.

    Raises
    ------
    ValueError
        For unknown tags.
    """
    tag = leaf[ARRAY_TAG]
    if tag == "ndarray":
        return _decode_ndarray(leaf)
    if tag == "sparse":
        return _decode_sparse(leaf)
    if tag == "sitk.Image":
        return _decode_image(leaf)
    if tag == "float":
        return _TOKEN_TO_FLOAT[leaf["value"]]
    raise ValueError(f"Unknown array tag {tag!r}")


def to_json_tree(data: Any) -> Any:
    """Convert a python-mode dump into a tree of JSON types with tagged array leaves.

    Parameters
    ----------
    data : Any
        Output of ``model_dump()``.

    Returns
    -------
    Any
        Tree of dicts, lists, strings, numbers, booleans and None. Enums, paths, dates and
        similar objects are converted like pydantic does in JSON mode.
    """
    from pydantic_core import to_jsonable_python

    return to_jsonable_python(encode_tree(data), inf_nan_mode="constants")

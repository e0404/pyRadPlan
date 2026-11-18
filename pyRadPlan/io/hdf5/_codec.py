"""HDF5 leaf encoder and decoder plugging into :func:`pyRadPlan.core.encode_tree`."""

from __future__ import annotations

import json
import math
from typing import Any

import h5py
import numpy as np
import scipy.sparse as sp
import SimpleITK as sitk
from pydantic_core import to_jsonable_python

from pyRadPlan.core import ARRAY_TAG, decode_tree, encode_tree, json_leaf_decoder
from pyRadPlan.core._serialization import _pixel_id
from pyRadPlan.core.xp_utils import to_numpy

FORMAT_NAME = "workspace"
FORMAT_VERSION = 1
COMPRESSION = "gzip"
COMPRESSION_OPTS = 1
COMPRESSION_THRESHOLD = 64 * 1024

_JSON_SCALARS = (str, int, float, bool, type(None))


def check_json_tree(obj: Any, path: str = "tree") -> None:
    """Raise ``TypeError`` if an encoded tree holds anything that is not plain JSON data."""
    if isinstance(obj, np.generic):
        return
    if isinstance(obj, _JSON_SCALARS):
        if isinstance(obj, float) and not math.isfinite(obj):
            raise TypeError(f"{path}: non-finite float")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(key, str):
                raise TypeError(f"{path}: dictionary key {key!r} is not a string")
            check_json_tree(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            check_json_tree(value, f"{path}[{i}]")
    else:
        raise TypeError(f"{path}: objects of type {type(obj).__name__} cannot be stored")


class ArrayWriter:
    """Leaf encoder writing array leaves as datasets below ``group/arrays``.

    Parameters
    ----------
    group : h5py.Group
        Group of the object. Datasets are created as ``arrays/<n>`` relative to it.
    """

    def __init__(self, group: h5py.Group):
        self.arrays = group.create_group("arrays")
        self._count = 0

    def encode(self, tree: Any) -> Any:
        """Replace the array leaves of ``tree`` by tags, writing their data."""
        return encode_tree(tree, self)

    def _next(self) -> str:
        name = str(self._count)
        self._count += 1
        return name

    @staticmethod
    def _dataset(parent: h5py.Group, name: str, arr: np.ndarray) -> None:
        kwargs: dict = {}
        if arr.ndim > 0 and arr.nbytes >= COMPRESSION_THRESHOLD:
            kwargs = {
                "chunks": True,
                "compression": COMPRESSION,
                "compression_opts": COMPRESSION_OPTS,
                "shuffle": True,
            }
        parent.create_dataset(name, data=np.ascontiguousarray(arr), **kwargs)

    def __call__(self, obj: Any) -> dict:
        if isinstance(obj, float):
            return {ARRAY_TAG: "float", "value": "nan" if math.isnan(obj) else _inf(obj)}
        if isinstance(obj, sitk.Image):
            return self._image(obj)
        if sp.issparse(obj):
            return self._sparse(obj)
        return self._ndarray(to_numpy(obj))

    def _ndarray(self, arr: np.ndarray) -> dict:
        if arr.dtype.names is not None:
            raise TypeError("Structured arrays cannot be stored")
        tag = {ARRAY_TAG: "ndarray", "dtype": arr.dtype.str, "shape": list(arr.shape)}
        kind = arr.dtype.kind
        if kind == "O":
            tag["dtype"] = "object"
            tag["data"] = [self.encode(item) for item in arr.reshape(-1)]
        elif kind == "U":
            tag["data"] = arr.reshape(-1).tolist()
        elif kind == "S":
            tag["data"] = [v.hex() for v in arr.reshape(-1).tolist()]
        elif kind in "biufc":
            name = self._next()
            self._dataset(self.arrays, name, arr)
            tag["dataset"] = f"arrays/{name}"
        else:
            raise TypeError(f"Arrays of dtype {arr.dtype} cannot be stored")
        return tag

    def _sparse(self, mat: Any) -> dict:
        csc = mat.tocsc()
        csc.sum_duplicates()
        name = self._next()
        sub = self.arrays.create_group(name)
        self._dataset(sub, "data", np.asarray(csc.data))
        self._dataset(sub, "indices", np.asarray(csc.indices))
        self._dataset(sub, "indptr", np.asarray(csc.indptr))
        return {
            ARRAY_TAG: "sparse",
            "format": "csc",
            "container": "matrix" if isinstance(mat, sp.spmatrix) else "array",
            "dtype": csc.dtype.str,
            "shape": [int(s) for s in csc.shape],
            "dataset": f"arrays/{name}",
        }

    def _image(self, image: sitk.Image) -> dict:
        arr = np.asarray(sitk.GetArrayViewFromImage(image))
        name = self._next()
        self._dataset(self.arrays, name, arr)
        tag = {
            ARRAY_TAG: "sitk.Image",
            "dtype": arr.dtype.str,
            "pixel_type": image.GetPixelIDTypeAsString(),
            "shape": list(arr.shape),
            "dataset": f"arrays/{name}",
            "origin": list(image.GetOrigin()),
            "spacing": list(image.GetSpacing()),
            "direction": list(image.GetDirection()),
        }
        if arr.ndim > image.GetDimension():
            tag["components"] = image.GetNumberOfComponentsPerPixel()
        return tag


def _inf(value: float) -> str:
    return "inf" if value > 0 else "-inf"


class ArrayReader:
    """Leaf decoder reading the datasets written by :class:`ArrayWriter`.

    Parameters
    ----------
    group : h5py.Group
        Group of the object holding the ``arrays`` subgroup.
    """

    def __init__(self, group: h5py.Group):
        self.group = group

    def decode(self, tree: Any) -> Any:
        """Replace the tags of ``tree`` by the arrays, sparse matrices and images they describe."""
        return decode_tree(tree, self)

    def __call__(self, leaf: dict) -> Any:
        tag = leaf[ARRAY_TAG]
        if tag == "ndarray":
            return self._ndarray(leaf)
        if tag == "sparse":
            return self._sparse(leaf)
        if tag == "sitk.Image":
            return self._image(leaf)
        if tag == "float":
            return json_leaf_decoder(leaf)
        raise ValueError(f"Unknown array tag {tag!r}")

    def _read(self, path: str) -> np.ndarray:
        dataset = self.group[path]
        if not isinstance(dataset, h5py.Dataset):
            raise ValueError(f"{dataset.name} is not a dataset")
        return dataset[()]

    def _ndarray(self, leaf: dict) -> np.ndarray:
        shape = tuple(leaf["shape"])
        if leaf["dtype"] == "object":
            out = np.empty(len(leaf["data"]), dtype=object)
            for i, item in enumerate(leaf["data"]):
                out[i] = self.decode(item)
            return out.reshape(shape)
        dtype = np.dtype(leaf["dtype"])
        if "dataset" in leaf:
            return np.asarray(self._read(leaf["dataset"]), dtype=dtype).reshape(shape)
        data = leaf["data"]
        if dtype.kind == "S":
            data = [bytes.fromhex(v) for v in data]
        return np.asarray(data, dtype=dtype).reshape(shape)

    def _sparse(self, leaf: dict) -> Any:
        if leaf.get("format", "csc") != "csc":
            raise ValueError(f"Unsupported sparse format {leaf['format']!r}")
        base = leaf["dataset"]
        cls = sp.csc_array if leaf.get("container") == "array" else sp.csc_matrix
        data = np.asarray(self._read(f"{base}/data"), dtype=np.dtype(leaf["dtype"]))
        return cls(
            (data, self._read(f"{base}/indices"), self._read(f"{base}/indptr")),
            shape=tuple(leaf["shape"]),
        )

    def _image(self, leaf: dict) -> sitk.Image:
        arr = np.asarray(self._read(leaf["dataset"]), dtype=np.dtype(leaf["dtype"]))
        components = leaf.get("components", 1)
        image = sitk.GetImageFromArray(arr.reshape(tuple(leaf["shape"])), isVector=components > 1)
        pixel_type = leaf.get("pixel_type")
        if pixel_type is not None and image.GetPixelIDTypeAsString() != pixel_type:
            image = sitk.Cast(image, _pixel_id(pixel_type))
        image.SetOrigin(leaf["origin"])
        image.SetSpacing(leaf["spacing"])
        image.SetDirection(leaf["direction"])
        return image


def write_tree(group: h5py.Group, tree: Any, *, jsonable: bool = False) -> None:
    """Store ``tree`` (JSON data, arrays, sparse matrices, images) in ``group``.

    With ``jsonable`` the remaining values (enums, paths, ...) are converted to JSON types
    like pydantic does in JSON mode, otherwise anything but plain JSON data raises ``TypeError``.
    """
    encoded = ArrayWriter(group).encode(tree)
    if jsonable:
        encoded = to_jsonable_python(encoded)
    check_json_tree(encoded)
    group.create_dataset(
        "tree", data=json.dumps(encoded, allow_nan=False), dtype=h5py.string_dtype("utf-8")
    )


def read_tree(group: h5py.Group) -> Any:
    """Load the tree stored by :func:`write_tree` with all leaves decoded."""
    text = group["tree"][()]
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    return ArrayReader(group).decode(json.loads(text))

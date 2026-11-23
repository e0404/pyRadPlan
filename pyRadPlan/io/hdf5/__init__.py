"""HDF5 import/export backend storing a complete workspace without pickle.

Loading never executes code: the file holds JSON text and plain datasets only. Unlike
pickle the files can be inspected with any HDF5 tool and are readable across versions.

File layout (``format_version`` 1)::

    /                       attributes: pyradplan_format = "workspace", format_version = 1,
                            pyradplan_version, contents = JSON list of the group names in
                            the order they were saved
    /<name>/                one group per object (ct, cst, pln, stf, dij, result, dose, ...)
                            attribute kind = "ct" | "cst" | "pln" | "stf" | "dij" | "result"
                            | "results" | "tree"
                            attribute ct_ref = "ct" on a cst whose reference CT is the saved
                            ct; its tree then omits ct_image
    /<name>/tree            UTF-8 JSON string dataset: the python-mode dump of the object
                            (kind ct, cst, pln, stf, dij, result; kind results: a mapping of
                            names to the dumps of plan results, as ``Workspace.results`` of
                            :mod:`pyRadPlan.app`) or the object itself (kind tree:
                            dicts, lists, strings, numbers, arrays, images) with every array
                            leaf replaced by a tagged dictionary
    /<name>/arrays/<n>      the array data of the larger leaves; sparse matrices are a subgroup
                            with the datasets data, indices and indptr (CSC)

Leaf tags (all carry the key ``"__array__"``)::

    {"__array__": "ndarray", "dtype": "<f8", "shape": [2, 3], "dataset": "arrays/0"}
    {"__array__": "ndarray", "dtype": "float64", "shape": [3], "data": [0.0, 1.5, "NaN"]}
    {"__array__": "ndarray", "dtype": "object", "shape": [2], "data": [<tag or None>, ...]}
    {"__array__": "ndarray", "dtype": "<U3", "shape": [2], "data": ["abc", "d"]}
    {"__array__": "sparse", "format": "csc", "container": "matrix" | "array", "dtype": ...,
     "shape": [rows, cols], "dataset": "arrays/1"}
    {"__array__": "sitk.Image", "dtype": ..., "pixel_type": "32-bit float", "shape": [z, y, x],
     "dataset": "arrays/2", "origin": [...], "spacing": [...], "direction": [...],
     "components": 3}
    {"__array__": "float", "value": "nan" | "inf" | "-inf"}

Numeric arrays below 4 KiB are stored inline in the tree, in the format of
:func:`pyRadPlan.core.json_leaf_encoder` (``"data"`` holds the flat values in C order), because
a dataset costs about 1.5 KiB of file and some milliseconds regardless of its size; a stf has
four tiny arrays per ray. The reader accepts both forms, so files written without inline
arrays load unchanged. Datasets of at least 64 KiB are stored chunked with shuffle and gzip
level 1, smaller ones uncompressed. The kinds ct, cst, pln, stf, dij and result (and every
entry of results) are rebuilt with the matching ``validate_*`` factory, so the loaded objects equal the saved ones.
"""

from .._factory import register_importer, register_exporter
from ._importer import Hdf5Importer
from ._exporter import Hdf5Exporter
from ._handler import Hdf5Handler

register_importer(Hdf5Importer)
register_exporter(Hdf5Exporter)

__all__ = ["Hdf5Importer", "Hdf5Exporter", "Hdf5Handler"]

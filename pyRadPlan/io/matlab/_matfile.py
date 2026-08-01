"""Low-level read and write of MATLAB ``.mat`` files."""

from os import PathLike

import numpy as np
import pymatreader
from scipy import io
from scipy.io.matlab import MatlabOpaque

# scipy 1.18 renamed the fields of the records holding MATLAB objects (classes, strings,
# ...) from ("s0", "s1", "s2", "arr") to the ones below, dropping the leading empty field.
# pymatreader (<= 1.2.3) still indexes them positionally and raises on the comparison
# `data[0][2] == b"string"`, which now hits the metadata array. So we map the records back
# to the old layout before handing them to pymatreader, which imports such objects on a
# best-effort basis anyway.
_OPAQUE_FIELDS_SCIPY_118 = ("_TypeSystem", "_Class", "_ObjectMetadata")
_OPAQUE_DTYPE_LEGACY = np.dtype([("s0", "O"), ("s1", "O"), ("s2", "O"), ("arr", "O")])


def _as_legacy_opaque(data: MatlabOpaque) -> MatlabOpaque:
    """Rewrite a scipy >= 1.18 opaque record in the layout pymatreader expects."""

    legacy = np.empty(data.shape, dtype=_OPAQUE_DTYPE_LEGACY)
    for index, record in np.ndenumerate(data):
        type_system, class_name, metadata = (record[name] for name in _OPAQUE_FIELDS_SCIPY_118)
        legacy[index] = (
            b"",
            type_system.encode() if isinstance(type_system, str) else type_system,
            class_name.encode() if isinstance(class_name, str) else class_name,
            metadata,
        )

    return legacy.view(MatlabOpaque)


def _restore_legacy_opaque(data):
    """Recursively map opaque records in a freshly loaded ``.mat`` structure in place."""

    if isinstance(data, dict):
        for key, value in data.items():
            data[key] = _restore_legacy_opaque(value)
    elif isinstance(data, MatlabOpaque):
        if data.dtype.names == _OPAQUE_FIELDS_SCIPY_118:
            return _as_legacy_opaque(data)
    elif isinstance(data, np.void):  # scalar element of a struct array
        for name in data.dtype.names:
            data[name] = _restore_legacy_opaque(data[name])
    elif isinstance(data, np.ndarray) and (data.dtype.names or data.dtype == object):
        for index, value in np.ndenumerate(data):
            data[index] = _restore_legacy_opaque(value)

    return data


def _load_with_opaque_workaround(path2file: PathLike) -> dict[str]:
    """Read a pre-v7.3 file like pymatreader does, but normalize opaque records first."""

    raw = io.loadmat(file_name=path2file, mat_dtype=False, squeeze_me=True, struct_as_record=True)
    return pymatreader.utils._parse_scipy_mat_dict(_restore_legacy_opaque(raw))


# Dunno. These wrappers are somewhat unnecessary...
def load(path2file: PathLike) -> dict[str]:
    """Load a .mat file and return the data as a dictionary."""

    try:
        matrad_patient = pymatreader.read_mat(filename=path2file)
    except NotImplementedError:
        matrad_patient = io.loadmat(
            file_name=path2file, mat_dtype=True, squeeze_me=True, struct_as_record=True
        )
    except Exception as exep:
        try:
            matrad_patient = _load_with_opaque_workaround(path2file)
        except Exception:
            raise ValueError(f"Could not load the .mat file: {path2file}") from exep

    return matrad_patient


def save(path2file: PathLike, the_dict: dict[str]):
    """
    Save a dictionary as a .mat file.

    Parameters
    ----------
    path2file : str
        Path to the file.
    the_dict : dict[str]
        Dictionary to be saved.
    """

    io.savemat(file_name=path2file, mdict=the_dict)

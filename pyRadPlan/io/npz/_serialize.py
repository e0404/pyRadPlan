"""Shared (de)serialization helpers for the NumPy ``.npz`` backend.

A ``.npz`` archive stores only named arrays. Geometry and per-VOI metadata are
kept in a single JSON string (stored as a 0-d ``<U`` array under ``meta``), so
the archive loads without ``allow_pickle``. 4D images (phases along the first array
axis) keep their 4D geometry; the voxel indices of a 4D VOI address the full
``(t, z, y, x)`` array in C order.
"""

import numpy as np
import SimpleITK as sitk

#: Bumped when the on-disk layout changes incompatibly.
FORMAT_VERSION = 1


def image_to_array_geom(image: sitk.Image) -> tuple[np.ndarray, dict]:
    """Return the (z, y, x) array and a geometry dict for a SimpleITK image.

    A 4D image gives a ``(t, z, y, x)`` array and 4D origin, spacing and direction.
    """
    array = sitk.GetArrayFromImage(image)
    geom = {
        "origin": list(image.GetOrigin()),
        "spacing": list(image.GetSpacing()),
        "direction": list(image.GetDirection()),
    }
    return array, geom


def array_geom_to_image(array: np.ndarray, geom: dict) -> sitk.Image:
    """Rebuild a SimpleITK image from a (z, y, x) or (t, z, y, x) array and a geometry dict."""
    image = sitk.GetImageFromArray(np.asarray(array), isVector=False)
    image.SetOrigin(tuple(geom["origin"]))
    image.SetSpacing(tuple(geom["spacing"]))
    image.SetDirection(tuple(geom["direction"]))
    return image


def geom_to_ct_kwargs(geom: dict) -> dict:
    """Translate a geometry dict into keyword arguments for ``validate_ct``."""
    spacing = geom["spacing"]
    return {
        "resolution": {"x": spacing[0], "y": spacing[1], "z": spacing[2]},
        "origin": tuple(geom["origin"]),
        "direction": tuple(geom["direction"]),
    }

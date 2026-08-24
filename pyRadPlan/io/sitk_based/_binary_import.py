"""Format-agnostic import of a CT and per-file binary structure masks.

Unlike the pyRadPlan-native SimpleITK folder layout (a single combined label map
plus a sidecar, see :mod:`pyRadPlan.io.sitk_based.base`), this module imports a
*foreign* dataset: a CT stored under an arbitrary filename plus one binary mask
per structure, possibly in a mix of formats (``.nii``/``.nrrd``/``.mha``/...) and
on a different voxel grid than the CT. CT values are taken as HU as-is; masks that
do not match the CT grid are nearest-neighbor resampled onto it.
"""

import os
import logging
from typing import Optional, Union

import numpy as np
import SimpleITK as sitk

from pyRadPlan.ct import validate_ct, CT
from pyRadPlan.cst import validate_cst, validate_voi, VOI, StructureSet
from pyRadPlan.core.resample import resample_image

from .._helpers import determine_structure_type
from .base._serialize import _check_3d, label_image_to_vois, read_sidecar

logger = logging.getLogger(__name__)

#: File extensions read as SimpleITK images when scanning a folder for masks/CT.
IMAGE_EXTENSIONS: tuple[str, ...] = (".nii.gz", ".nii", ".nrrd", ".nhdr", ".mha", ".mhd")

PathLike = Union[str, os.PathLike]


def list_image_files(path: PathLike) -> list[str]:
    """Return the SimpleITK-readable image files directly inside ``path``, sorted."""
    directory = os.fspath(path)
    if not os.path.isdir(directory):
        raise NotADirectoryError(f"Not a directory: {directory}")
    files = []
    for entry in sorted(os.listdir(directory)):
        full = os.path.join(directory, entry)
        if os.path.isfile(full) and entry.lower().endswith(IMAGE_EXTENSIONS):
            files.append(full)
    return files


def _expand_paths(structure_paths: Union[PathLike, list]) -> list[str]:
    """Flatten a path / list of paths into individual mask files (folders expanded)."""
    if isinstance(structure_paths, (str, os.PathLike)):
        structure_paths = [structure_paths]

    files: list[str] = []
    for raw in structure_paths:
        entry = os.fspath(raw)
        if os.path.isdir(entry):
            files.extend(list_image_files(entry))
        elif os.path.isfile(entry):
            files.append(entry)
        else:
            raise FileNotFoundError(f"Structure path not found: {entry}")
    return files


def read_ct_image(ct_file: PathLike) -> CT:
    """Read a CT from an image file, taking its intensities as HU."""
    path = os.fspath(ct_file)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"CT file not found: {path}")
    image = sitk.ReadImage(path)
    _check_3d(image)
    return validate_ct(cube_hu=image)


def mask_file_to_voi(
    ct: CT,
    path: PathLike,
    *,
    name: Optional[str] = None,
    voi_type: Optional[str] = None,
) -> VOI:
    """Build a VOI from a binary mask file, resampled onto the CT grid if needed.

    Parameters
    ----------
    ct : CT
        Reference CT; the mask is aligned to ``ct.cube_hu``.
    path : PathLike
        The binary mask image file.
    name : str, optional
        VOI name (defaults to the file stem).
    voi_type : str, optional
        VOI type (defaults to :func:`determine_structure_type` on the name).
    """
    path = os.fspath(path)
    stem = _file_stem(path)
    name = name or stem
    voi_type = voi_type or determine_structure_type(name)

    mask = sitk.ReadImage(path)
    _check_3d(mask)
    # Nearest-neighbor keeps the mask binary; 0 outside the original extent.
    mask = _align_to_ct(sitk.Cast(mask > 0, sitk.sitkUInt8), ct.cube_hu)

    return validate_voi(name=str(name), voi_type=voi_type, mask=mask, ct_image=ct)


#: Signed-integer images without negatives are label maps if they hold at most
#: this many distinct values, and CTs otherwise.
_MAX_LABELMAP_VALUES = 64


def infer_image_kind(image: sitk.Image) -> str:
    """Guess what a bare image represents from its pixel type and values.

    Heuristic: boolean/unsigned-integer images are masks or label maps
    (``"structures"``); images with negative values are CTs in HU (``"ct"``);
    non-negative floats are doses (``"dose"``). Non-negative signed integers are
    label maps when they hold only a few distinct values, CTs otherwise.

    Returns
    -------
    str
        One of ``"ct"``, ``"structures"`` or ``"dose"``.
    """
    arr = sitk.GetArrayViewFromImage(image)
    kind = arr.dtype.kind
    if kind in ("b", "u"):
        return "structures"
    if arr.size == 0:
        return "ct"
    if float(arr.min()) < 0:
        return "ct"
    if kind == "i":
        return "structures" if len(np.unique(arr)) <= _MAX_LABELMAP_VALUES else "ct"
    return "dose" if kind == "f" else "ct"


def _align_to_ct(image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    """Nearest-neighbor resample ``image`` onto ``reference`` if geometries differ."""
    if (
        image.GetSize() != reference.GetSize()
        or image.GetSpacing() != reference.GetSpacing()
        or image.GetOrigin() != reference.GetOrigin()
        or image.GetDirection() != reference.GetDirection()
    ):
        image = resample_image(
            image,
            interpolator=sitk.sitkNearestNeighbor,
            target_image=reference,
            extrapolate=0,
        )
    return image


def image_file_to_vois(ct: CT, path: PathLike, *, name: Optional[str] = None) -> list[VOI]:
    """Read a structure image file into one or more VOIs, aligned to the CT grid.

    A file with a single nonzero value is treated as one binary mask (VOI named
    from ``name`` or the file stem, typed via :func:`determine_structure_type`).
    A file with several distinct nonzero values is treated as a label map: one
    VOI per label, with names/types taken from a ``<stem>.json`` sidecar or
    embedded ``pyradplan_*`` header metadata when present (see
    :func:`~pyRadPlan.io.sitk_based.base._serialize.label_image_to_vois`).

    Parameters
    ----------
    ct : CT
        Reference CT; masks are aligned to ``ct.cube_hu``.
    path : PathLike
        The structure image file (binary mask or label map).
    name : str, optional
        VOI name override — only applied when the file holds a single mask.

    Returns
    -------
    list[VOI]
        The imported VOIs (one for a binary mask, several for a label map).
    """
    path = os.fspath(path)
    image = sitk.ReadImage(path)
    _check_3d(image)

    labels = [int(v) for v in np.unique(sitk.GetArrayViewFromImage(image)) if v != 0]
    if len(labels) <= 1:
        return [mask_file_to_voi(ct, path, name=name)]

    label_image = _align_to_ct(sitk.Cast(image, sitk.sitkUInt8), ct.cube_hu)
    sidecar_path = os.path.join(os.path.dirname(path), _file_stem(path) + ".json")
    return label_image_to_vois(label_image, ct, read_sidecar(sidecar_path))


def masks_to_cst(
    ct: CT,
    structure_paths: Union[PathLike, list],
    *,
    selections: Optional[list[dict]] = None,
) -> StructureSet:
    """Assemble a StructureSet from individual binary mask files.

    Parameters
    ----------
    ct : CT
        Reference CT.
    structure_paths : PathLike or list
        A folder (expanded to its image files), a mask file, or a list thereof.
    selections : list of dict, optional
        Per-file overrides, each ``{"path", "name", "voi_type"}``. When given,
        only the listed paths are imported (in order) and any entry whose
        ``voi_type`` is ``"IGNORED"`` (case-insensitive) is skipped. This is what
        the import dialog's review table produces.

    Returns
    -------
    StructureSet
        The validated structure set referencing ``ct``.
    """
    if selections is not None:
        entries = [
            (sel["path"], sel.get("name"), sel.get("voi_type"))
            for sel in selections
            if str(sel.get("voi_type", "")).upper() != "IGNORED"
        ]
    else:
        entries = [(f, None, None) for f in _expand_paths(structure_paths)]

    if not entries:
        raise ValueError("No structure masks to import.")

    vois = []
    for path, name, voi_type in entries:
        try:
            vois.append(mask_file_to_voi(ct, path, name=name, voi_type=voi_type))
        except Exception as exc:  # noqa: BLE001 - one bad mask must not abort the import
            logger.warning("Failed to import mask '%s': %s", path, exc)

    if not vois:
        raise ValueError("None of the structure masks could be imported.")
    return validate_cst(vois, ct)


def load_binary_patient(
    ct_file: PathLike,
    structure_paths: Union[PathLike, list],
    *,
    selections: Optional[list[dict]] = None,
) -> tuple[CT, StructureSet]:
    """
    Load a CT and StructureSet from a foreign binary dataset.

    Parameters
    ----------
    ct_file : PathLike
        The CT image file (any SimpleITK-readable format); values taken as HU.
    structure_paths : PathLike or list
        A folder, a mask file, or a list of folders/files. Each mask file becomes
        one VOI (named from its file stem, typed via
        :func:`~pyRadPlan.io._helpers.determine_structure_type`), resampled onto
        the CT grid if it does not already match.
    selections : list of dict, optional
        Per-file overrides (see :func:`masks_to_cst`).

    Returns
    -------
    tuple[CT, StructureSet]
        The CT and StructureSet objects.
    """
    ct = read_ct_image(ct_file)
    cst = masks_to_cst(ct, structure_paths, selections=selections)
    return ct, cst


def _file_stem(path: str) -> str:
    """Return the filename without directory or (compound) image extension."""
    name = os.path.basename(path)
    lowered = name.lower()
    for ext in IMAGE_EXTENSIONS:
        if lowered.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]

"""Shared (de)serialization for the SimpleITK-based backends.

These formats store a single image per file, so a StructureSet is written as one
label-map image (overlaps resolved by ``overlap_priority``, ``BODY`` last) plus a
JSON sidecar with the canonical VOI metadata. For NRRD/MetaImage the metadata is
additionally stamped into the image header (3D-Slicer / ``pyradplan_*`` keys) for
viewer interoperability; NIfTI cannot hold arbitrary metadata, hence the sidecar.
"""

import os
import json
import logging
from typing import Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.cst import StructureSet, VOI, validate_voi

logger = logging.getLogger(__name__)

SIDECAR_VERSION = 1

# RGB triplets (0..1) used as a fallback for 3D-Slicer segment colors.
_SLICER_COLORS = (
    "1.0 0.0 0.0",
    "0.0 1.0 0.0",
    "0.0 0.0 1.0",
    "1.0 1.0 0.0",
    "1.0 0.0 1.0",
    "0.0 1.0 1.0",
    "0.5 0.5 0.5",
    "1.0 0.5 0.0",
    "0.5 0.0 1.0",
    "0.0 0.5 0.5",
)


# --------------------------------------------------------------------------
# Plain image IO
# --------------------------------------------------------------------------


def write_image(image: sitk.Image, path: os.PathLike) -> None:
    """Write a SimpleITK image, compressing where the format supports it."""
    sitk.WriteImage(image, os.fspath(path), useCompression=True)


def read_image(path: os.PathLike) -> sitk.Image:
    """Read a SimpleITK image from a file."""
    return sitk.ReadImage(os.fspath(path))


def _check_3d(image: sitk.Image) -> None:
    if image.GetDimension() == 4:
        raise NotImplementedError("4D images are not supported by the sitk-based backends yet.")


# --------------------------------------------------------------------------
# StructureSet -> single label image (+ sidecar / interop metadata)
# --------------------------------------------------------------------------


def _sorted_vois(cst: StructureSet) -> list[tuple[int, VOI]]:
    """Order VOIs by overlap priority (lower wins), but keep BODY last."""
    items = [(i, voi) for i, voi in enumerate(cst.vois)]
    body = [it for it in items if it[1].name.upper() == "BODY"]
    others = [it for it in items if it[1].name.upper() != "BODY"]
    others.sort(key=lambda it: (it[1].overlap_priority, it[0]))
    body.sort(key=lambda it: (it[1].overlap_priority, it[0]))
    return others + body


def _extent_str(binary: sitk.Image) -> str:
    """Return a 3D-Slicer extent string ``"xmin xmax ymin ymax zmin zmax"``."""
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(binary)
    if 1 not in stats.GetLabels():
        return "0 -1 0 -1 0 -1"
    b = stats.GetBoundingBox(1)
    return f"{b[0]} {b[0] + b[3] - 1} {b[1]} {b[1] + b[4] - 1} {b[2]} {b[2] + b[5] - 1}"


def cst_to_label_image(cst: StructureSet) -> tuple[Optional[sitk.Image], dict[int, str]]:
    """Combine all VOI masks into one 1-based UInt8 label image.

    Labels follow VOI insertion order (``label = index + 1``). Overlaps are
    resolved by ``overlap_priority`` (lower wins); BODY is applied last so it
    never overwrites other structures. Returns the label image (or ``None`` for
    an empty cst) and a map of VOI index -> Slicer extent string.
    """
    if not cst.vois:
        return None, {}

    reference = cst.ct_image.cube_hu
    _check_3d(reference)

    combined = sitk.Image(reference.GetSize(), sitk.sitkUInt8)
    combined.CopyInformation(reference)

    extents: dict[int, str] = {}
    for index, voi in _sorted_vois(cst):
        binary = sitk.Cast(voi.mask > 0, sitk.sitkUInt8)
        extents[index] = _extent_str(binary)
        # Only fill voxels not yet claimed by a higher-priority VOI.
        unclaimed = sitk.Cast(sitk.Equal(combined, 0), sitk.sitkUInt8)
        combined = sitk.Add(combined, sitk.Multiply(binary, unclaimed) * (index + 1))

    return combined, extents


def build_sidecar(cst: StructureSet) -> dict:
    """Build the canonical JSON sidecar mapping label -> VOI metadata."""
    vois = {}
    for index, voi in enumerate(cst.vois):
        vois[str(index + 1)] = {
            "name": voi.name,
            "voi_type": voi.voi_type,
            "alpha_x": voi.alpha_x,
            "beta_x": voi.beta_x,
            "overlap_priority": voi.overlap_priority,
            "visible": voi.visible,
            "visible_color": list(voi.visible_color) if voi.visible_color is not None else None,
        }
    return {"format_version": SIDECAR_VERSION, "type": "StructureSet", "vois": vois}


def write_sidecar(path: os.PathLike, data: dict) -> None:
    """Write the JSON sidecar."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def read_sidecar(path: os.PathLike) -> Optional[dict]:
    """Read the JSON sidecar, or None if it does not exist."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def apply_interop_metadata(image: sitk.Image, cst: StructureSet, extents: dict[int, str]) -> None:
    """Stamp ``pyradplan_*`` and 3D-Slicer ``Segment*`` keys onto a label image.

    Only effective for formats that persist arbitrary metadata (NRRD, MetaImage).
    """
    image.SetMetaData("pyradplan_type", "StructureSet")
    image.SetMetaData("pyradplan_num_vois", str(len(cst.vois)))
    image.SetMetaData("pyradplan_voi_names", "|".join(v.name for v in cst.vois))
    image.SetMetaData("pyradplan_voi_types", "|".join(v.voi_type for v in cst.vois))
    image.SetMetaData(
        "pyradplan_overlap_priorities", "|".join(str(v.overlap_priority) for v in cst.vois)
    )
    image.SetMetaData("pyradplan_alpha_x_values", "|".join(str(v.alpha_x) for v in cst.vois))
    image.SetMetaData("pyradplan_beta_x_values", "|".join(str(v.beta_x) for v in cst.vois))

    image.SetMetaData("Segmentation_MasterRepresentation", "Binary labelmap")
    for i, voi in enumerate(cst.vois):
        if voi.visible_color is not None:
            color = " ".join(str(c / 255.0) for c in voi.visible_color)
        else:
            color = _SLICER_COLORS[i % len(_SLICER_COLORS)]
        image.SetMetaData(f"Segment{i}_LabelValue", str(i + 1))
        image.SetMetaData(f"Segment{i}_Name", voi.name)
        image.SetMetaData(f"Segment{i}_Color", color)
        image.SetMetaData(f"Segment{i}_Extent", extents.get(i, "0 -1 0 -1 0 -1"))


# --------------------------------------------------------------------------
# label image (+ sidecar) -> VOIs
# --------------------------------------------------------------------------


def _embedded_metadata(label_image: sitk.Image) -> dict[int, dict]:
    """Best-effort parse of ``pyradplan_*`` header keys into label -> metadata.

    Used for files written with embedded metadata (NRRD/MetaImage) when no JSON
    sidecar is present (e.g. 3D-Slicer style segmentations).
    """
    if not label_image.HasMetaDataKey("pyradplan_voi_names"):
        return {}
    names = label_image.GetMetaData("pyradplan_voi_names").split("|")

    def _split(key):
        return label_image.GetMetaData(key).split("|") if label_image.HasMetaDataKey(key) else []

    types = _split("pyradplan_voi_types")
    alphas = _split("pyradplan_alpha_x_values")
    betas = _split("pyradplan_beta_x_values")
    priorities = _split("pyradplan_overlap_priorities")

    out: dict[int, dict] = {}
    for i, name in enumerate(names):
        meta: dict = {"name": name}
        if i < len(types):
            meta["voi_type"] = types[i]
        if i < len(alphas):
            meta["alpha_x"] = float(alphas[i])
        if i < len(betas):
            meta["beta_x"] = float(betas[i])
        if i < len(priorities):
            meta["overlap_priority"] = int(priorities[i])
        out[i + 1] = meta
    return out


def label_image_to_vois(label_image: sitk.Image, ct, sidecar: Optional[dict]) -> list[VOI]:
    """Reconstruct VOIs from a label image, using sidecar (preferred) or header metadata."""
    arr = sitk.GetArrayViewFromImage(label_image)
    labels = [int(v) for v in np.unique(arr) if v != 0]

    voi_meta: dict = (sidecar or {}).get("vois", {})
    embedded = _embedded_metadata(label_image) if not voi_meta else {}

    vois = []
    for label in labels:
        mask = sitk.Cast(sitk.Equal(label_image, label), sitk.sitkUInt8)
        mask.CopyInformation(ct.cube_hu)

        meta = voi_meta.get(str(label)) or embedded.get(label) or {}
        kwargs = {
            "name": meta.get("name", f"Segment_{label}"),
            "voi_type": meta.get("voi_type", "OAR"),
            "mask": mask,
            "ct_image": ct,
        }
        for field in ("alpha_x", "beta_x", "overlap_priority", "visible", "visible_color"):
            if field in meta and meta[field] is not None:
                kwargs[field] = meta[field]
        vois.append(validate_voi(**kwargs))
    return vois

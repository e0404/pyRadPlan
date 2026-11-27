"""Shared (de)serialization for the SimpleITK-based backends.

A StructureSet is written as a label-map image (overlaps resolved by
``overlap_priority``, ``BODY`` last) or, to keep overlapping structures, as layers
(a 3D-Slicer ``.seg.nrrd`` for NRRD, a 4D one-hot image with one channel per VOI for
NIfTI and MetaImage), plus a JSON sidecar with the canonical VOI metadata. For
NRRD/MetaImage the metadata is additionally stamped into the image header (3D-Slicer /
``pyradplan_*`` keys) for viewer interoperability; NIfTI cannot hold arbitrary metadata,
hence the sidecar. The phases of a 4D structure set are written as separate files.
"""

import gzip
import os
import re
import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import SimpleITK as sitk

from pyRadPlan.core.resample import resample_image
from pyRadPlan.cst import StructureSet, VOI, validate_voi

logger = logging.getLogger(__name__)

SIDECAR_VERSION = 1


# --------------------------------------------------------------------------
# Plain image IO
# --------------------------------------------------------------------------


def write_image(image: sitk.Image, path: os.PathLike) -> None:
    """Write a SimpleITK image, compressing where the format supports it."""
    sitk.WriteImage(image, os.fspath(path), useCompression=True)


def read_image(path: os.PathLike) -> sitk.Image:
    """Read a SimpleITK image from a file."""
    return sitk.ReadImage(os.fspath(path))


# --------------------------------------------------------------------------
# StructureSet -> segmentation images (+ sidecar / interop metadata)
# --------------------------------------------------------------------------

SEGMENTATION_MODES = ("auto", "labelmap", "layers")


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


def num_phases(cst: StructureSet) -> int:
    """Return the number of phases (4th axis) of the VOI masks; 1 for a 3D structure set."""
    sizes = [v.mask.GetSize()[3] for v in cst.vois if v.mask.GetDimension() == 4]
    return max(sizes, default=1)


def phase_reference(cst: StructureSet) -> sitk.Image:
    """Return the 3D reference image (grid of the first CT phase) of a structure set."""
    return phase_reference_of(cst.ct_image)


def phase_masks(cst: StructureSet, phase: int = 0) -> list[sitk.Image]:
    """Binary 3D ``UInt8`` masks of all VOIs for one phase, in VOI order."""
    masks = []
    for voi in cst.vois:
        mask = voi.mask[:, :, :, phase] if voi.mask.GetDimension() == 4 else voi.mask
        masks.append(sitk.Cast(mask > 0, sitk.sitkUInt8))
    return masks


def overlapping_vois(masks_per_phase: list[list[sitk.Image]], cst: StructureSet) -> list[str]:
    """Names of the VOIs sharing at least one voxel with another VOI in any phase."""
    names: set[str] = set()
    for masks in masks_per_phase:
        arrays = [sitk.GetArrayViewFromImage(m) > 0 for m in masks]
        if len(arrays) < 2:  # noqa: PLR2004
            continue
        count = np.zeros(arrays[0].shape, dtype=np.uint16)
        for a in arrays:
            count += a
        shared = count > 1
        for voi, a in zip(cst.vois, arrays):
            if np.any(a & shared):
                names.add(voi.name)
    return [v.name for v in cst.vois if v.name in names]


def _label_dtype(num_labels: int):
    return np.uint8 if num_labels < 256 else np.uint16  # noqa: PLR2004


def _like(array: np.ndarray, reference: sitk.Image) -> sitk.Image:
    image = sitk.GetImageFromArray(array, isVector=False)
    image.CopyInformation(reference)
    return image


def masks_to_label_image(
    cst: StructureSet, masks: list[sitk.Image], reference: sitk.Image
) -> tuple[sitk.Image, dict[int, str], list[str]]:
    """Combine VOI masks into one label image; also return extents and VOIs losing voxels."""
    combined = np.zeros(sitk.GetArrayViewFromImage(reference).shape, _label_dtype(len(masks)))
    extents: dict[int, str] = {}
    lost: list[str] = []
    for index, voi in _sorted_vois(cst):
        binary = masks[index]
        selected = sitk.GetArrayViewFromImage(binary) > 0
        extents[index] = _extent_str(binary)
        # Only fill voxels not yet claimed by a higher-priority VOI.
        placed = selected & (combined == 0)
        combined[placed] = index + 1
        if np.count_nonzero(placed) < np.count_nonzero(selected):
            lost.append(voi.name)
    return _like(combined, reference), extents, lost


def cst_to_label_image(
    cst: StructureSet, phase: int = 0
) -> tuple[Optional[sitk.Image], dict[int, str]]:
    """Combine all VOI masks of one phase into one 1-based label image.

    Labels follow VOI insertion order (``label = index + 1``). Overlaps are
    resolved by ``overlap_priority`` (lower wins); BODY is applied last so it
    never overwrites other structures. Returns the label image (or ``None`` for
    an empty cst) and a map of VOI index -> Slicer extent string.
    """
    if not cst.vois:
        return None, {}
    image, extents, _ = masks_to_label_image(cst, phase_masks(cst, phase), phase_reference(cst))
    return image, extents


def assign_layers(masks: list[sitk.Image]) -> list[int]:
    """Assign each mask to the first layer in which it overlaps no other mask."""
    occupied: list[np.ndarray] = []
    layers = []
    for mask in masks:
        array = sitk.GetArrayViewFromImage(mask) > 0
        for layer, taken in enumerate(occupied):
            if not np.any(taken & array):
                occupied[layer] = taken | array
                layers.append(layer)
                break
        else:
            occupied.append(array.copy())
            layers.append(len(occupied) - 1)
    return layers


def masks_to_layers(
    masks: list[sitk.Image], reference: sitk.Image
) -> tuple[list[np.ndarray], list[int], dict[int, str]]:
    """Distribute VOI masks over non-overlapping label layers (``label = index + 1``).

    Returns the ``(z, y, x)`` layer arrays, the layer of each VOI and the Slicer extents.
    """
    layers = assign_layers(masks)
    shape = sitk.GetArrayViewFromImage(reference).shape
    arrays = [
        np.zeros(shape, _label_dtype(len(masks))) for _ in range(max(layers, default=-1) + 1)
    ]
    extents = {}
    for index, (mask, layer) in enumerate(zip(masks, layers)):
        arrays[layer][sitk.GetArrayViewFromImage(mask) > 0] = index + 1
        extents[index] = _extent_str(mask)
    return arrays, layers, extents


def masks_to_onehot(masks: list[sitk.Image], reference: sitk.Image) -> sitk.Image:
    """Stack VOI masks as the channels (4th axis) of a 4D ``UInt8`` image."""
    stacked = sitk.JoinSeries(masks)
    direction = np.eye(4)
    direction[:3, :3] = np.reshape(reference.GetDirection(), (3, 3))
    stacked.SetOrigin((*reference.GetOrigin(), 0.0))
    stacked.SetSpacing((*reference.GetSpacing(), 1.0))
    stacked.SetDirection(direction.ravel().tolist())
    return stacked


def build_sidecar(cst: StructureSet, encoding: str = "labelmap") -> dict:
    """Build the canonical JSON sidecar mapping label (or channel) -> VOI metadata.

    ``encoding`` (``"labelmap"``, ``"layers"`` or ``"onehot"``) tells importers how the
    image stores the VOIs; sidecars without it describe a label map.
    """
    vois = {}
    for index, voi in enumerate(cst.vois):
        vois[str(index + 1)] = {
            "name": voi.name,
            "voi_type": voi.voi_type,
            "alpha_x": voi.tissue_properties.alpha_x,
            "beta_x": voi.tissue_properties.beta_x,
            "tissue_properties": voi.tissue_properties.model_dump(),
            "overlap_priority": voi.overlap_priority,
            "overlap_handling": voi.overlap_handling,
            "visible": voi.visible,
            "color": list(voi.color),
        }
    return {
        "format_version": SIDECAR_VERSION,
        "type": "StructureSet",
        "encoding": encoding,
        "vois": vois,
    }


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


def interop_entries(
    cst: StructureSet,
    extents: Optional[dict[int, str]] = None,
    layers: Optional[list[int]] = None,
    encoding: str = "labelmap",
) -> dict[str, str]:
    """Header entries: ``pyradplan_*`` keys and, unless one-hot, 3D-Slicer ``Segment*`` keys."""
    entries = {
        "pyradplan_type": "StructureSet",
        "pyradplan_encoding": encoding,
        "pyradplan_num_vois": str(len(cst.vois)),
        "pyradplan_voi_names": "|".join(v.name for v in cst.vois),
        "pyradplan_voi_types": "|".join(v.voi_type for v in cst.vois),
        "pyradplan_overlap_priorities": "|".join(str(v.overlap_priority) for v in cst.vois),
    }
    tissues = [v.tissue_properties for v in cst.vois]
    entries["pyradplan_alpha_x_values"] = "|".join(str(t.alpha_x) for t in tissues)
    entries["pyradplan_beta_x_values"] = "|".join(str(t.beta_x) for t in tissues)
    entries["pyradplan_tissue_properties"] = json.dumps([t.model_dump() for t in tissues])
    if encoding == "onehot":
        return entries

    extents = extents or {}
    entries["Segmentation_MasterRepresentation"] = "Binary labelmap"
    if layers is not None:
        entries["Segmentation_ContainedRepresentationNames"] = "Binary labelmap|"
        entries["Segmentation_ReferenceImageExtentOffset"] = "0 0 0"
    for i, voi in enumerate(cst.vois):
        entries[f"Segment{i}_LabelValue"] = str(i + 1)
        entries[f"Segment{i}_Name"] = voi.name
        entries[f"Segment{i}_Color"] = " ".join(str(c / 255.0) for c in voi.color)
        entries[f"Segment{i}_Extent"] = extents.get(i, "0 -1 0 -1 0 -1")
        if layers is not None:
            entries[f"Segment{i}_ID"] = f"Segment_{i + 1}"
            entries[f"Segment{i}_Layer"] = str(layers[i])
            entries[f"Segment{i}_NameAutoGenerated"] = "0"
    return entries


def apply_interop_metadata(
    image: sitk.Image,
    cst: StructureSet,
    extents: dict[int, str],
    encoding: str = "labelmap",
) -> None:
    """Stamp ``pyradplan_*`` and 3D-Slicer ``Segment*`` keys onto a segmentation image.

    Only effective for formats that persist arbitrary metadata (NRRD, MetaImage).
    """
    for key, value in interop_entries(cst, extents, encoding=encoding).items():
        image.SetMetaData(key, value)


_NRRD_TYPES = {np.dtype(np.uint8): "unsigned char", np.dtype(np.uint16): "unsigned short"}


def write_slicer_segmentation(
    path: os.PathLike,
    layer_arrays: list[np.ndarray],
    reference: sitk.Image,
    entries: dict[str, str],
) -> None:
    """Write a 3D-Slicer segmentation (``.seg.nrrd``).

    One layer gives a 3D label map; several layers give a 4D image whose first axis
    (``kinds: list``) enumerates the layers, as Slicer stores overlapping segments.
    ``entries`` are the ``Segment*`` / ``Segmentation_*`` header fields.
    """
    multi = len(layer_arrays) > 1
    array = np.stack(layer_arrays, axis=-1) if multi else layer_arrays[0]
    array = np.ascontiguousarray(array)
    spacing = reference.GetSpacing()
    direction = np.reshape(reference.GetDirection(), (3, 3))
    axes = " ".join(
        "(" + ",".join(repr(float(v)) for v in direction[:, j] * spacing[j]) + ")"
        for j in range(3)
    )
    size = " ".join(str(s) for s in reference.GetSize())
    lines = [
        "NRRD0004",
        f"type: {_NRRD_TYPES[array.dtype]}",
        f"dimension: {4 if multi else 3}",
        "space: left-posterior-superior",
        f"sizes: {len(layer_arrays)} {size}" if multi else f"sizes: {size}",
        f"space directions: none {axes}" if multi else f"space directions: {axes}",
        "kinds: list domain domain domain" if multi else "kinds: domain domain domain",
        "endian: little",
        "encoding: gzip",
        "space origin: (" + ",".join(repr(float(v)) for v in reference.GetOrigin()) + ")",
    ]
    lines += [f"{key}:={value}" for key, value in entries.items()]
    header = ("\n".join(lines) + "\n\n").encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(gzip.compress(array.astype(array.dtype.newbyteorder("<")).tobytes()))


# --------------------------------------------------------------------------
# segmentation image (+ sidecar) -> VOIs
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
    tissues = []
    if label_image.HasMetaDataKey("pyradplan_tissue_properties"):
        try:
            tissues = json.loads(label_image.GetMetaData("pyradplan_tissue_properties"))
        except json.JSONDecodeError:
            logger.warning("Ignoring unreadable pyradplan_tissue_properties header entry.")

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
        if i < len(tissues):
            meta["tissue_properties"] = tissues[i]
        out[i + 1] = meta
    return out


_SEGMENT_LAYER_KEY = re.compile(r"Segment(\d+)_Layer$")


def _slicer_segments(image: sitk.Image) -> list[dict]:
    """Segments declared by 3D-Slicer ``Segment<i>_*`` header keys, in header order."""
    ordinals = sorted(
        int(m.group(1)) for k in image.GetMetaDataKeys() if (m := _SEGMENT_LAYER_KEY.match(k))
    )
    segments = []
    for i in ordinals:

        def _get(field, default=None, i=i):
            key = f"Segment{i}_{field}"
            return image.GetMetaData(key) if image.HasMetaDataKey(key) else default

        segment: dict = {
            "layer": int(_get("Layer", 0)),
            "label": int(_get("LabelValue", i + 1)),
            "meta": {},
        }
        if _get("Name") is not None:
            segment["meta"]["name"] = _get("Name")
        color = _get("Color")
        if color is not None:
            try:
                rgb = [float(c) for c in color.split()[:3]]
                segment["meta"]["color"] = [int(round(c * 255)) for c in rgb]
            except ValueError:
                logger.warning("Ignoring unreadable Segment%d_Color header entry.", i)
        segments.append(segment)
    return segments


@dataclass
class Segment:
    """One structure decoded from a segmentation image: its key, header metadata and mask."""

    key: int
    meta: dict
    mask: sitk.Image


def _same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return (
        a.GetSize() == b.GetSize()
        and a.GetOrigin() == b.GetOrigin()
        and a.GetSpacing() == b.GetSpacing()
        and a.GetDirection() == b.GetDirection()
    )


def _channels_and_specs(
    image: sitk.Image, voi_meta: dict
) -> tuple[list[sitk.Image], list[tuple[int, int, Optional[int], dict]]]:
    """Split an image into 3D channel images and ``(key, channel, label, meta)`` specs.

    A ``label`` of ``None`` selects every non-zero voxel of the channel (one-hot).
    Label maps only yield the labels they contain; layered and one-hot images keep
    every declared structure, also empty ones.
    """
    if image.GetDimension() == 4:  # noqa: PLR2004 - one-hot: a binary mask per channel
        channels = [image[:, :, :, k] for k in range(image.GetSize()[3])]
        return channels, [(k + 1, k, None, {}) for k in range(len(channels))]

    components = image.GetNumberOfComponentsPerPixel()
    if components > 1:  # layers of a Slicer segmentation, stored as vector components
        channels = [sitk.VectorIndexSelectionCast(image, k) for k in range(components)]
    else:
        channels = [image]

    embedded = _embedded_metadata(image) if not voi_meta else {}
    segments = _slicer_segments(image)
    if segments:
        specs = []
        for ordinal, seg in enumerate(segments, start=1):
            meta = {**embedded.get(ordinal, {}), **seg["meta"]}
            specs.append((ordinal, min(seg["layer"], len(channels) - 1), seg["label"], meta))
        return channels, specs

    arr = sitk.GetArrayViewFromImage(image)
    labels = [int(v) for v in np.unique(arr) if v != 0]
    return channels, [(label, 0, label, embedded.get(label, {})) for label in labels]


def _any(mask: sitk.Image) -> bool:
    return bool(np.any(sitk.GetArrayViewFromImage(mask)))


def _selected(channel: sitk.Image, label: Optional[int]) -> bool:
    arr = sitk.GetArrayViewFromImage(channel)
    return bool(np.any(arr == label) if label is not None else np.any(arr))


def decode_segments(
    image: sitk.Image, reference: sitk.Image, sidecar: Optional[dict] = None
) -> list[Segment]:
    """Decode a label map, Slicer layers or a one-hot image into masks on ``reference``."""
    voi_meta: dict = (sidecar or {}).get("vois", {})
    channels, specs = _channels_and_specs(image, voi_meta)

    resample = not _same_grid(channels[0], reference)
    source_filled: list[bool] = []
    if resample:
        logger.warning("Label map grid differs from the CT; resampling labels onto the CT grid.")
        source_filled = [_selected(channels[ch], label) for _, ch, label, _ in specs]
        # Preserve physical locations and discrete labels; outside the source is background.
        channels = [
            resample_image(
                c, interpolator=sitk.sitkNearestNeighbor, target_image=reference, extrapolate=0
            )
            for c in channels
        ]

    segments = []
    for n, (key, ch, label, embedded) in enumerate(specs):
        meta = voi_meta.get(str(key)) or embedded
        selected = sitk.Equal(channels[ch], label) if label is not None else channels[ch] > 0
        mask = sitk.Cast(selected, sitk.sitkUInt8)
        if resample and source_filled[n] and not _any(mask):
            logger.warning(
                "Label %s (%s) is empty on the CT grid after resampling.",
                key,
                meta.get("name", f"Segment_{key}"),
            )
        segments.append(Segment(key, meta, mask))
    return segments


def segments_to_vois(per_phase: list[list[Segment]], ct) -> list[VOI]:
    """Build VOIs from the segments of one or several phases (matched by ``Segment.key``).

    A single phase is repeated on every phase of a 4D CT; several phases need a 4D CT
    with as many phases (extra phases are ignored, with a warning, for a 3D CT).
    """
    phases = ct.num_of_ct_scen
    if len(per_phase) > 1:
        if phases == 1:
            logger.warning(
                "Found %d segmentation phases but the CT is 3D; using the first phase.",
                len(per_phase),
            )
            per_phase = per_phase[:1]
        elif len(per_phase) != phases:
            raise ValueError(
                f"The segmentation has {len(per_phase)} phases but the CT has {phases}."
            )

    keys = sorted({seg.key for segments in per_phase for seg in segments})
    vois = []
    for key in keys:
        found = [{s.key: s for s in segments}.get(key) for segments in per_phase]
        first = next(s for s in found if s is not None)
        zero = sitk.Image(first.mask.GetSize(), sitk.sitkUInt8)
        zero.CopyInformation(first.mask)
        masks = [s.mask if s is not None else zero for s in found]
        if len(masks) == 1 and phases > 1:
            masks = masks * phases
        meta = first.meta
        kwargs = {
            "name": meta.get("name", f"Segment_{key}"),
            "voi_type": meta.get("voi_type", "OAR"),
            "mask": sitk.JoinSeries(masks) if len(masks) > 1 else masks[0],
            "ct_image": ct,
        }
        # Files written before tissue properties were stored only carry alpha/beta.
        lq_fields = () if meta.get("tissue_properties") is not None else ("alpha_x", "beta_x")
        for field in (
            "tissue_properties",
            *lq_fields,
            "overlap_priority",
            "overlap_handling",
            "visible",
            "color",
        ):
            if field in meta and meta[field] is not None:
                kwargs[field] = meta[field]
        # Sidecars written before the color field was renamed use the former key.
        if "color" not in kwargs and meta.get("visible_color") is not None:
            kwargs["color"] = meta["visible_color"]
        vois.append(validate_voi(**kwargs))
    return vois


def label_image_to_vois(label_image: sitk.Image, ct, sidecar: Optional[dict]) -> list[VOI]:
    """Reconstruct VOIs from a segmentation image, using sidecar (preferred) or header metadata."""
    return segments_to_vois([decode_segments(label_image, phase_reference_of(ct), sidecar)], ct)


def phase_reference_of(ct) -> sitk.Image:
    """Return the 3D grid of a CT (the first phase of a 4D CT)."""
    image = ct.cube_hu
    return image[:, :, :, 0] if image.GetDimension() == 4 else image

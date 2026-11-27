"""Geometry helpers for the planning agent.

Pure functions on SimpleITK masks (CT grid). Distances are computed in physical
millimetres between voxel centres via signed distance maps, so margins and rings respect
anisotropic spacing but are resolved to the voxel grid: along each axis they round down
to whole voxels (see :func:`expand`). Everything returns small, JSON-friendly dicts
meant for an LLM.
"""

from typing import Literal, Optional

import numpy as np
import SimpleITK as sitk
from scipy.spatial import ConvexHull, Delaunay, QhullError

_DIRECTIONS = (  # LPS: +x left, +y posterior, +z superior
    ("left", "right"),
    ("posterior", "anterior"),
    ("superior", "inferior"),
)


# --------------------------------------------------------------------------------------
# basic mask utilities
# --------------------------------------------------------------------------------------


def as_bool(mask: sitk.Image) -> np.ndarray:
    """Mask as boolean numpy array (z, y, x)."""
    return sitk.GetArrayViewFromImage(mask).astype(bool)


def to_mask(arr: np.ndarray, reference: sitk.Image) -> sitk.Image:
    """Boolean array (z, y, x) -> uint8 mask with the reference geometry."""
    img = sitk.GetImageFromArray(arr.astype(np.uint8))
    img.CopyInformation(reference)
    return img


def voxel_cc(mask: sitk.Image) -> float:
    """Volume of one voxel in cc."""
    return float(np.prod(mask.GetSpacing())) / 1000.0


def volume_cc(arr: np.ndarray, reference: sitk.Image) -> float:
    return round(float(arr.sum()) * voxel_cc(reference), 2)


def signed_distance_mm(mask: sitk.Image) -> np.ndarray:
    """Signed distance to the mask surface in mm (negative inside), array (z, y, x)."""
    d = sitk.SignedMaurerDistanceMap(
        sitk.Cast(mask, sitk.sitkUInt8),
        insideIsPositive=False,
        squaredDistance=False,
        useImageSpacing=True,
    )
    return sitk.GetArrayFromImage(d)


def expand(mask: sitk.Image, margin_mm: float) -> np.ndarray:
    """Isotropic margin in mm (negative values shrink), resolved to the voxel grid.

    Growing adds the voxels whose centre is within *margin_mm* of the centre of a mask
    voxel; shrinking removes the mask voxels whose centre is within the margin of the
    centre of a voxel outside the mask. Along each axis a margin therefore rounds down to
    whole voxels for both signs: 1 mm on 5 mm slices changes nothing across the slices,
    5 mm changes one slice.
    """
    arr = as_bool(mask)
    if margin_mm > 0:
        return signed_distance_mm(mask) <= margin_mm
    if margin_mm == 0 or arr.all():
        return arr.copy()
    return signed_distance_mm(to_mask(~arr, mask)) > -margin_mm


def centroid_mm(arr: np.ndarray, reference: sitk.Image) -> Optional[np.ndarray]:
    """Physical (LPS, mm) center of mass of a boolean array."""
    if not arr.any():
        return None
    zyx = np.argwhere(arr).mean(axis=0)
    return np.array(reference.TransformContinuousIndexToPhysicalPoint(zyx[::-1].tolist()))


def describe_offset(vec_mm: np.ndarray, min_mm: float = 3.0) -> str:
    """'12 mm posterior, 5 mm left' for an LPS offset vector."""
    parts = [
        (abs(v), f"{abs(v):.0f} mm {pos if v > 0 else neg}")
        for v, (pos, neg) in zip(vec_mm, _DIRECTIONS)
        if abs(v) >= min_mm
    ]
    return ", ".join(p for _, p in sorted(parts, reverse=True)) or "at the center"


# --------------------------------------------------------------------------------------
# convex hull / concavity
# --------------------------------------------------------------------------------------


def convex_hull_mask(arr: np.ndarray, reference: sitk.Image) -> np.ndarray:
    """Voxels inside the convex hull of the mask (computed in mm)."""
    idx = np.argwhere(arr)
    hull_arr = np.zeros_like(arr, dtype=bool)
    if len(idx) < 5:
        hull_arr[arr] = True
        return hull_arr
    scale = np.array(reference.GetSpacing()[::-1])
    pts = idx * scale
    try:
        hull = ConvexHull(pts)
    except QhullError:  # flat (single slice) structure
        hull_arr[arr] = True
        return hull_arr
    lo, hi = idx.min(0), idx.max(0) + 1
    grid = np.stack(
        np.meshgrid(*[np.arange(a, b) for a, b in zip(lo, hi)], indexing="ij"), -1
    ).reshape(-1, 3)
    inside = Delaunay(pts[hull.vertices]).find_simplex(grid * scale) >= 0
    g = grid[inside]
    hull_arr[g[:, 0], g[:, 1], g[:, 2]] = True
    return hull_arr | arr


def target_shape(
    target_mask: sitk.Image,
    others: dict[str, sitk.Image],
    shape_parameters: Optional[dict] = None,
) -> dict:
    """Shape descriptors of a target and how the surrounding OARs sit relative to it.

    Returns solidity (volume / convex hull volume), sphericity, connected components,
    concavity volume and which structures fill it, plus per-structure distance and
    overlap. Includes plain-language hints for helper-structure decisions.
    """
    arr = as_bool(target_mask)
    vox = voxel_cc(target_mask)
    vol = float(arr.sum()) * vox
    out: dict = {"volume_cc": round(vol, 2)}
    if not arr.any():
        return {**out, "empty": True}

    # components
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(sitk.Cast(target_mask, sitk.sitkUInt8)))
    comp = sitk.GetArrayViewFromImage(cc)
    n_comp = int(comp.max())
    out["num_components"] = n_comp
    if n_comp > 1:
        out["component_volumes_cc"] = [
            round(float((comp == i).sum()) * vox, 2) for i in range(1, min(n_comp, 5) + 1)
        ]

    # surface / sphericity
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(sitk.Cast(target_mask, sitk.sitkUInt8))
    if stats.HasLabel(1):
        out["sphericity"] = round(float(stats.GetRoundness(1)), 3)
    if shape_parameters:
        out["elongation"] = round(shape_parameters["elongation"], 2)
        out["flatness"] = round(shape_parameters["flatness"], 2)
        out["extent_mm"] = [round(float(x), 1) for x in shape_parameters["bounding_box_size"]]

    # concavity
    hull = convex_hull_mask(arr, target_mask)
    concavity = hull & ~arr
    out["solidity"] = round(float(arr.sum()) / float(hull.sum()), 3)
    out["concavity_cc"] = round(float(concavity.sum()) * vox, 2)

    # neighbours
    dist = signed_distance_mm(target_mask)
    t_center = centroid_mm(arr, target_mask)
    neighbours, envelopes = [], []
    n_target = float(arr.sum())
    for name, m in others.items():
        o = as_bool(m)
        if not o.any():
            continue
        n_o = float(o.sum())
        if float((o & arr).sum()) >= 0.95 * n_target and n_o > 1.5 * n_target:
            envelopes.append(name)  # e.g. BODY typed as OAR: not a neighbour
            continue
        entry = {
            "voi": name,
            "min_distance_mm": round(float(max(dist[o].min(), 0.0)), 1),
            "overlap_cc": round(float((o & arr).sum()) * vox, 2),
            "pct_of_voi_overlapping_target": round(100 * float((o & arr).sum()) / n_o, 1),
        }
        if concavity.any():
            in_conc = float((o & concavity).sum())
            if in_conc:
                entry["pct_of_voi_in_concavity"] = round(100 * in_conc / n_o, 1)
                entry["pct_of_concavity_filled"] = round(100 * in_conc / float(concavity.sum()), 1)
        c = centroid_mm(o & ~arr, target_mask)
        if c is not None:
            entry["direction_from_target"] = describe_offset(c - t_center)
        neighbours.append(entry)
    neighbours.sort(key=lambda e: e["min_distance_mm"])
    out["neighbours"] = neighbours[:8]
    if envelopes:
        out["enclosing_structures"] = envelopes
    out["hints"] = _shape_hints(out)
    return out


def _shape_hints(s: dict) -> list[str]:
    hints = []
    if s.get("num_components", 1) > 1:
        hints.append(
            "Target has several separate parts: consider one ring per part or check that a "
            "single ring does not bridge normal tissue between them."
        )
    if s["solidity"] < 0.85:
        filled = [n["voi"] for n in s["neighbours"] if n.get("pct_of_concavity_filled", 0) > 10]
        hint = f"Target is concave (solidity {s['solidity']})."
        if filled:
            hint += (
                f" Its concavity contains {filled}: a plain ring would overlap these OARs and "
                "compete with their objectives. Use ring with exclude (OAR + margin) or "
                "skip_concavity, and rely on the OAR objectives for the inner gradient."
            )
        hints.append(hint)
    else:
        hints.append(f"Target is roughly convex (solidity {s['solidity']}): a ring is suitable.")
    if s.get("elongation", 1) > 2.5:
        hints.append(
            "Target is elongated: dose fall-off along the long axis is mostly controlled by "
            "beam/field edges; rings mainly help in the transverse directions."
        )
    close = [n["voi"] for n in s["neighbours"] if n["overlap_cc"] > 0]
    if close:
        hints.append(
            f"Target overlaps {close}: check feasibility and consider splitting the target "
            "(subtract/intersect helpers) with a lower goal in the overlap."
        )
    return hints


# --------------------------------------------------------------------------------------
# helper structure recipes
# --------------------------------------------------------------------------------------


def ring(
    target: sitk.Image,
    inner_mm: float,
    outer_mm: float,
    body: Optional[np.ndarray] = None,
    remove: Optional[np.ndarray] = None,
    skip_concavity: bool = False,
) -> np.ndarray:
    """Shell between inner_mm and outer_mm around target (follows its shape)."""
    if outer_mm <= inner_mm or inner_mm < 0:
        raise ValueError("Ring needs 0 <= inner_mm < outer_mm.")
    d = signed_distance_mm(target)
    arr = (d > inner_mm) & (d <= outer_mm)
    if skip_concavity:
        t = as_bool(target)
        arr &= ~convex_hull_mask(t, target)
    if body is not None:
        arr &= body
    if remove is not None:
        arr &= ~remove
    return arr


def dose_spots(
    dose: np.ndarray,
    region: np.ndarray,
    threshold: float,
    kind: Literal["hot", "cold"],
    reference: sitk.Image,
    labels: dict[str, np.ndarray],
    anchor: Optional[np.ndarray] = None,
    top_n: int = 3,
    min_cc: float = 0.05,
) -> dict:
    """Find connected regions with dose > threshold (hot) or < threshold (cold) in region."""
    arr = region & ((dose > threshold) if kind == "hot" else (dose < threshold))
    vox = voxel_cc(reference)
    out = {
        "kind": kind,
        "threshold": threshold,
        "total_cc": round(float(arr.sum()) * vox, 2),
        "spots": [],
    }
    if not arr.any():
        return out
    cc = sitk.RelabelComponent(
        sitk.ConnectedComponent(to_mask(arr, reference)),
        minimumObjectSize=max(1, int(min_cc / vox)),
    )
    lab = sitk.GetArrayViewFromImage(cc)
    anchor_c = centroid_mm(anchor, reference) if anchor is not None else None
    for i in range(1, min(int(lab.max()), top_n) + 1):
        spot = lab == i
        d = dose[spot]
        c = centroid_mm(spot, reference)
        entry = {
            "volume_cc": round(float(spot.sum()) * vox, 2),
            "dose_extreme": round(float(d.max() if kind == "hot" else d.min()), 3),
            "dose_mean": round(float(d.mean()), 3),
            "centroid_mm": [round(float(x), 1) for x in c],
            "inside": {
                n: round(100 * float((spot & m).sum()) / float(spot.sum()), 1)
                for n, m in labels.items()
                if (spot & m).any()
            },
        }
        if anchor_c is not None:
            entry["relative_to_target_center"] = describe_offset(c - anchor_c)
        out["spots"].append(entry)
    return out

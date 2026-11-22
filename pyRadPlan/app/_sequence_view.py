"""The delivery sequence of a result as the data a sequence viewer draws."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def has_sequence(result: Any) -> bool:
    """Whether *result* carries a delivery sequence with beams."""
    return bool(getattr(getattr(result, "sequence", None), "beams", None))


def _fluence(
    bixels: Optional[slice], geometry: Any, orientation: str, fluences: dict
) -> Optional[dict]:
    from pyRadPlan.sequencing._bev import native_fluence_map  # noqa: PLC0415

    if geometry is None or bixels is None:
        return None
    maps: dict[str, Any] = {"optimized": None, "delivered": None}
    window = None
    for mode, values in fluences.items():
        if values is None:
            continue
        grid, window = native_fluence_map(geometry, np.asarray(values)[bixels], orientation)
        maps[mode] = grid.astype(np.float32)
    if window is None:
        return None
    bixel_width = getattr(geometry, "bixel_width", None)
    return {
        "window": [float(v) for v in window],
        "bixel_width": float(bixel_width) if bixel_width else None,
        **maps,
    }


def _target(ct: Any, cst: Any, geometry: Any, orientation: str) -> Optional[dict]:
    from pyRadPlan.sequencing._bev import target_bev_points, target_outline_mask  # noqa: PLC0415

    if geometry is None or ct is None or cst is None:
        return None
    try:
        resolution = max(float(v) for v in ct.resolution.values())
        outline = target_outline_mask(
            target_bev_points(ct, cst, geometry, orientation), resolution
        )
    except Exception:  # noqa: BLE001 - a target that cannot be projected must not break the view
        logger.debug("Could not project the target into the beam's eye view", exc_info=True)
        return None
    if outline is None:
        return None
    x, y, occupied = outline
    return {
        "origin": [float(x[0]), float(y[0])],
        "spacing": [resolution, resolution],
        "mask": occupied.astype(np.uint8),
    }


def sequence_view(result: Any, stf: Any = None, ct: Any = None, cst: Any = None) -> dict[str, Any]:
    """Beams, MLC shapes, fluence maps and target projection of a result's sequence.

    All positions are in mm in the beam's eye view; a beam's horizontal axis is the travel
    direction of its leaves and its vertical axis the direction the leaf rows are stacked in.

    Returns
    -------
    dict
        ``{sequencer, num_of_levels, beams}``. Per beam: ``index``, ``label``,
        ``orientation`` (``"X"`` or ``"Y"``), ``window`` (``[travel_min, travel_max,
        stack_min, stack_max]``), ``weights`` and ``shapes`` (per shape ``leaf_width``,
        ``boundaries``, the lower edges of the leaf rows, and ``leaf_positions``, their
        left and right leaf), ``fluence`` (``window``, ``bixel_width`` and the ``optimized``
        and ``delivered`` map ``(stack, travel)``, ``NaN`` where the beam has no ray) and
        ``target`` (``origin`` and ``spacing`` of the bin centers of ``mask``, the
        ``(stack, travel)`` region the projected target covers). ``fluence`` and ``target``
        are ``None`` when the workspace lacks what they are derived from.
    """
    from pyRadPlan.sequencing._bev import (  # noqa: PLC0415
        beam_bixels,
        beam_label,
        field_window,
        shape_segments,
        stf_beam_index,
    )

    if not has_sequence(result):
        raise ValueError("The active result has no delivery sequence.")
    sequence = result.sequence
    stf_beams = list(getattr(stf, "beams", None) or [])
    fluences = {"optimized": getattr(result, "fluence", None), "delivered": sequence.fluence}

    beams = []
    for index, beam in enumerate(sequence.beams):
        segments = shape_segments(beam)
        mlcs = [segment.start.mlc for segment in segments]
        stf_beam = stf_beam_index(beam, index)
        bixels = beam_bixels(stf, stf_beam)
        geometry = stf_beams[stf_beam] if stf_beam < len(stf_beams) else None
        orientation = mlcs[0].device_orientation if mlcs else None
        beams.append(
            {
                "index": index,
                "label": beam_label(index, stf),
                "orientation": orientation,
                "window": field_window(beam),
                "weights": [float(segment.weight) for segment in segments],
                "shapes": [
                    {
                        "leaf_width": float(mlc.leaf_width),
                        "boundaries": [float(b) for b in mlc.leaf_position_boundaries],
                        "leaf_positions": np.asarray(mlc.leaf_positions, dtype=float).tolist(),
                    }
                    for mlc in mlcs
                ],
                "fluence": _fluence(bixels, geometry, orientation, fluences) if mlcs else None,
                "target": _target(ct, cst, geometry, orientation) if mlcs else None,
            }
        )
    return {
        "sequencer": sequence.sequencer,
        "num_of_levels": sequence.num_of_levels,
        "beams": beams,
    }

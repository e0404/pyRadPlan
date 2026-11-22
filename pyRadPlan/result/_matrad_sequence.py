"""Conversion between a delivery sequence and matRad's ``sequencing`` / ``apertureInfo``."""

import logging
from typing import Any, Optional

import numpy as np

from pyRadPlan.machines import MLC, Jaw

logger = logging.getLogger(__name__)


def _structs(data: Any) -> list[dict]:
    """Split a matRad struct array into one dictionary per element.

    A struct array is read as one list per field, a single struct as a plain dictionary
    whose fields hold the values themselves.
    """
    if data is None:
        return []
    if isinstance(data, np.ndarray) and data.dtype.names:
        return [{name: element[name] for name in data.dtype.names} for element in data.reshape(-1)]
    if isinstance(data, (list, tuple, np.ndarray)):
        return [dict(element) for element in data]
    values = list(data.values())
    if not values:
        return []
    if all(isinstance(v, list) for v in values) and len({len(v) for v in values}) == 1:
        return [{key: data[key][ix] for key in data} for ix in range(len(values[0]))]
    return [dict(data)]


def _struct_array(elements: list[dict]) -> np.ndarray:
    """One MATLAB struct array from dictionaries that share their keys."""
    array = np.empty(len(elements), dtype=[(key, object) for key in elements[0]])
    for ix, element in enumerate(elements):
        for key, value in element.items():
            array[ix][key] = value
    return array


def _window_jaws(window: Any) -> list[Jaw]:
    travel_min, travel_max, stack_min, stack_max = (float(v) for v in np.ravel(window)[:4])
    field_width = 2 * max(abs(travel_min), abs(travel_max), abs(stack_min), abs(stack_max))
    return [
        Jaw(device_orientation="X", positions=[travel_min, travel_max], field_width=field_width),
        Jaw(device_orientation="Y", positions=[stack_min, stack_max], field_width=field_width),
    ]


def sequence_from_matrad(data: dict) -> dict:
    """Fields of a :class:`~pyRadPlan.result.Sequence` from matRad's sequencing data.

    The segments are taken from the leaf positions of ``apertureInfo``; its bookkeeping and
    the shape masks of ``sequencing.beam`` can be rebuilt from them and are not kept.

    Parameters
    ----------
    data : dict
        matRad's ``sequencing`` struct, with ``apertureInfo`` under ``aperture_info`` or
        ``apertureInfo`` and the sequenced weights under ``fluence`` or ``w``.
    """
    out: dict[str, Any] = {}
    fluence = data.get("fluence", data.get("w"))
    if fluence is not None:
        out["fluence"] = fluence
    for key, name in (("sequencer", "sequencer"), ("numOfLevels", "num_of_levels")):
        value = data.get(key, data.get(name))
        if value is not None:
            out[name] = value

    info = data.get("aperture_info", data.get("apertureInfo"))
    if not isinstance(info, dict) or not len(_structs(info.get("beam"))):
        if data.get("beam") is not None:
            logger.warning(
                "matRad sequencing without apertureInfo carries no leaf positions; only the "
                "sequenced weights are imported."
            )
        return out

    if info.get("weightToMU") is not None:
        out["weight_to_mu"] = float(info["weightToMU"])
    leaf_width = float(info["bixelWidth"])
    num_of_leaf_pairs = int(info.get("numOfMLCLeafPairs", 0) or 0)

    beams = []
    for stf_beam, beam in enumerate(_structs(info["beam"])):
        boundaries = np.atleast_1d(np.asarray(beam["leafPairPos"], dtype=float)) - leaf_width / 2
        window = beam.get("MLCWindow")
        jaws = [] if window is None or np.isnan(window).any() else _window_jaws(window)
        segments = []
        for shape in _structs(beam.get("shape")):
            positions = np.column_stack(
                (
                    np.atleast_1d(np.asarray(shape["leftLeafPos"], dtype=float)),
                    np.atleast_1d(np.asarray(shape["rightLeafPos"], dtype=float)),
                )
            )
            mlc = MLC(
                device_orientation="X",
                number_of_elements=num_of_leaf_pairs,
                leaf_width=leaf_width,
                leaf_position_boundaries=boundaries,
                leaf_positions=positions,
            )
            segments.append(
                {
                    "weight": float(shape["weight"]),
                    "start": {"mlc": mlc, "jaws": jaws},
                    "stf_beam": stf_beam,
                }
            )
        beams.append({"segments": segments})
    out["beams"] = beams
    return out


def _jaw_window(jaws: list) -> Optional[np.ndarray]:
    """Return matRad's ``MLCWindow`` of an MLC along X from an X and a Y jaw, if both exist."""
    positions = {jaw.device_orientation: jaw.positions for jaw in jaws}
    if "X" not in positions or "Y" not in positions:
        return None
    return np.array([*positions["X"], *positions["Y"]], dtype=float)


def _exportable(sequence: Any) -> bool:
    for beam in sequence.beams:
        if not beam.segments:
            return False
        for segment in beam.segments:
            mlc = segment.start.mlc
            if segment.is_dynamic or mlc is None or mlc.leaf_positions is None:
                return False
            if mlc.device_orientation != "X":
                return False
    return bool(sequence.beams)


def sequence_to_matrad(sequence: Any, stf: Optional[Any] = None) -> dict:  # noqa: PLR0912
    """Entries of matRad's ``resultGUI`` that describe a delivery sequence.

    Parameters
    ----------
    sequence : Sequence
        The delivery sequence.
    stf : SteeringInformation, optional
        The steering information the sequence is planned on. With it, everything matRad
        derives from the beamlets is written as well: the shape masks, the beamlet index
        maps and the leaf travel limits. Without it the leaf positions, the weights and the
        field window of the jaws are.

    Returns
    -------
    dict
        ``wSequenced``, ``sequencing`` and ``apertureInfo``, as far as they are available.
    """
    out: dict[str, Any] = {}
    if sequence.fluence is not None:
        out["wSequenced"] = np.asarray(sequence.fluence, dtype=float)
    sequencing = {
        key: value
        for key, value in (
            ("sequencer", sequence.sequencer),
            ("numOfLevels", sequence.num_of_levels),
        )
        if value is not None
    }

    if not _exportable(sequence):
        if sequence.beams:
            logger.warning(
                "matRad holds static segments of an MLC along X only; the segments are not "
                "exported."
            )
        if sequencing:
            out["sequencing"] = sequencing
        return out

    info = None
    if stf is not None:
        try:
            info = sequence.aperture_info(stf)
        except (ValueError, NotImplementedError) as exc:
            logger.warning("Sequence exported without beamlet information: %s", exc)

    total_shapes = sequence.total_num_of_segments
    total_leaf_pairs = sum(
        len(segment.start.mlc.leaf_positions)
        for beam in sequence.beams
        for segment in beam.segments
    )
    vector_length = total_shapes + 2 * total_leaf_pairs
    aperture_vector = np.full(vector_length, np.nan)
    mapping_mx = np.full((vector_length, 4), np.nan)
    lim_mx = np.full((vector_length, 2), np.nan)
    lim_mx[:total_shapes] = [0.0, np.inf]

    aperture_beams, sequencing_beams = [], []
    shape_ix, leaf_ix = 0, 0
    for beam_ix, beam in enumerate(sequence.beams):
        mlc = beam.mlc
        num_rows = len(mlc.leaf_positions)
        grid = None if info is None else info.grids[beam.segments[0].stf_beam]
        columns = None
        if grid is not None:
            index_map, travel_min, stack_min = grid.bixel_index_map()
            columns = travel_min + np.arange(index_map.shape[1]) * grid.bixel_width

        shapes, masks = [], []
        for segment_ix, segment in enumerate(beam.segments):
            positions = np.asarray(segment.start.mlc.leaf_positions, dtype=float)
            rows = slice(total_shapes + leaf_ix, total_shapes + leaf_ix + num_rows)
            aperture_vector[shape_ix] = segment.weight
            aperture_vector[rows] = positions[:, 0]
            aperture_vector[rows.start + total_leaf_pairs : rows.stop + total_leaf_pairs] = (
                positions[:, 1]
            )
            mapping_mx[shape_ix, 0] = beam_ix + 1
            for half in (0, total_leaf_pairs):
                target = slice(rows.start + half, rows.stop + half)
                mapping_mx[target] = np.column_stack(
                    (
                        np.full(num_rows, beam_ix + 1),
                        np.full(num_rows, shape_ix + 1),
                        np.full(num_rows, segment_ix + 1),
                        np.arange(1, num_rows + 1),
                    )
                )
                if grid is not None:
                    lim_mx[target] = np.column_stack((grid.lim_left, grid.lim_right))
            shape = {
                "leftLeafPos": positions[:, 0],
                "rightLeafPos": positions[:, 1],
                "weight": float(segment.weight),
                "vectorOffset": float(total_shapes + leaf_ix + 1),
                "jacobiScale": 1.0,
                "weightOffset": float(shape_ix + 1),
            }
            if columns is not None:
                mask = (columns[None, :] > positions[:, :1]) & (
                    columns[None, :] < positions[:, 1:]
                )
                shape["shapeMap"] = mask.astype(float)
                masks.append(mask)
            shapes.append(shape)
            shape_ix += 1
            leaf_ix += num_rows

        aperture_beam = {
            "shape": _struct_array(shapes),
            "numOfShapes": float(len(shapes)),
            "numOfActiveLeafPairs": float(num_rows),
            "leafPairPos": mlc.leaf_position_boundaries + mlc.leaf_width / 2,
            "centralLeafPair": mlc.central_leaf_pair,
        }
        if mlc.number_of_elements >= num_rows:
            aperture_beam["isActiveLeafPair"] = mlc.is_active_leaf_pair(
                aperture_beam["centralLeafPair"]
            )
        window = _jaw_window(beam.segments[0].start.jaws)
        if window is not None:
            aperture_beam["MLCWindow"] = window
        if grid is not None:
            half = 0.5 * grid.bixel_width
            bixel_ind_map = np.where(index_map < 0, np.nan, index_map + 1.0)
            aperture_beam.update(
                {
                    "limLeft": grid.lim_left,
                    "limRight": grid.lim_right,
                    "bixelIndMap": bixel_ind_map,
                    "posOfCornerBixel": np.array([travel_min, 0.0, stack_min]),
                }
            )
            aperture_beam.setdefault(
                "MLCWindow",
                np.array(
                    [
                        grid.travel.min() - half,
                        grid.travel.max() + half,
                        grid.stack.min() - half,
                        grid.stack.max() + half,
                    ]
                ),
            )
            sequencing_beams.append(
                {
                    "numOfShapes": float(len(shapes)),
                    "shapes": np.stack(masks, axis=2).astype(float),
                    "shapesWeight": beam.weights,
                    "bixelIx": grid.bixel_offset + 1.0 + np.arange(grid.num_of_bixels),
                }
            )
        aperture_beams.append(aperture_beam)

    # A struct array needs the same fields in every element: a beam without jaws gets an
    # empty window, which the import reads as no jaws.
    if any("MLCWindow" in beam for beam in aperture_beams):
        for beam in aperture_beams:
            beam.setdefault("MLCWindow", np.full(4, np.nan))
    shared = set.intersection(*(set(beam) for beam in aperture_beams))
    aperture_beams = [{k: v for k, v in beam.items() if k in shared} for beam in aperture_beams]

    first_mlc = sequence.beams[0].mlc
    aperture_info = {
        "beam": _struct_array(aperture_beams),
        "runVMAT": False,
        "bixelWidth": float(first_mlc.leaf_width if info is None else info.grids[0].bixel_width),
        "numOfMLCLeafPairs": float(first_mlc.number_of_elements),
        "totalNumOfShapes": float(total_shapes),
        "totalNumOfLeafPairs": float(total_leaf_pairs),
        "apertureVector": aperture_vector,
        "mappingMx": mapping_mx,
        "limMx": lim_mx,
    }
    if info is not None:
        aperture_info["totalNumOfBixels"] = float(info.total_num_of_bixels)
    if sequence.weight_to_mu is not None:
        aperture_info["weightToMU"] = float(sequence.weight_to_mu)
    if sequencing_beams:
        sequencing["beam"] = _struct_array(sequencing_beams)

    out["apertureInfo"] = aperture_info
    if sequencing:
        out["sequencing"] = sequencing
    return out

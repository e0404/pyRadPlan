"""Export a pyRadPlan photon plan as a DICOM RT Plan (C-arm linac beams).

Geometry
--------
Gantry angle, couch angle, isocenter and the beam's eye view of the beams follow the IEC 61217
conventions documented in :mod:`pyRadPlan.io.dicom._export_plan`. At collimator angle 0 the
IEC BEAM LIMITING DEVICE system coincides with the IEC GANTRY system, so a beam limiting device
coordinate ``(x, y)`` at the isocenter plane is the beam's eye view position
``(x, z)`` (``ray_pos_bev = (x, 0, z)``). pyRadPlan has no collimator angle: Beam Limiting Device
Angle is always 0.

Field shapes
------------
A beam whose beamlet is a field shape (:class:`~pyRadPlan.stf.FieldShapeAsBLD` and
:class:`~pyRadPlan.stf.FieldShapeComposite` of them) is exported with its beam limiting
devices: jaws as ``ASYMX``/``ASYMY`` and multileaf collimators as ``MLCX``/``MLCY`` with Leaf
Position Boundaries and the Leaf/Jaw Positions in the Beam Limiting Device Position Sequence.
Leakage is not represented.

Device positions of ``FieldShapeAsBLD`` are in the IEC BEAM LIMITING DEVICE system, which at
collimator angle 0 is the beam's eye view (device x = BEV x, device y = BEV z), so they are
written unchanged; the tests check the exported aperture against the calculated dose.
A :class:`~pyRadPlan.stf.FieldShapeAsMask` is exported as the jaws of its bounding box (its
mask is given in the beam's eye view directly) and is not deliverable.

Bixel-based beams
-----------------
pyRadPlan has no leaf sequencing. A beam of bixels (IMRT, or a single collimated square field)
is exported as jaws forming an open field around the bixels (those with positive weight if
weights are given): for a single bixel this is exactly the collimated square field and is
deliverable. A beam of several bixels only keeps its geometry, the plan is declared not
deliverable in ``RTPlanDescription`` and ``BeamDescription`` and a warning is issued.

Weights
-------
Photon weights of pyRadPlan scale the dose per fraction of a bixel or field shape (Gy per
unit weight, unlike the ion export no particle numbers or MU) and the machine data holds no
calibration from weights to monitor units (``calculate_machine_scale`` of the examples is not
one). Therefore Beam Meterset (Type 3) is omitted, the Final Cumulative Meterset Weight is 1
and the Cumulative Meterset Weights of the two control points are 0 and 1. Weights only
select the beams (a beam without positive weight is not exported, with a warning) and the
bixels spanning the aperture; the weight of a field is written in ``BeamDescription``.
"""

import os
import warnings
from typing import Any, Optional

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

from pyRadPlan.machines.base import MLC, Jaw
from pyRadPlan.stf import FieldShape, FieldShapeAsBLD, FieldShapeAsMask, FieldShapeComposite

from ._export_common import RT_PLAN_STORAGE, ExportContext, PlanReference
from ._export_dose import _ds
from ._export_plan import (
    _resolve_machine,
    planned_fractions,
    require_head_first_supine,
    start_plan,
)


def is_photon_stf(stf) -> bool:
    """Whether all beams of the steering information use photons."""
    return len(stf.beams) > 0 and all(b.radiation_mode == "photons" for b in stf.beams)


def _is_field_beam(beam) -> bool:
    return any(isinstance(b, FieldShape) for ray in beam.rays for b in ray.beamlets)


def _leaf_pairs(positions) -> np.ndarray:
    """Return the opening ``(start, end)`` of every leaf as ``(bank 1, bank 2)`` positions."""
    positions = np.asarray(positions, dtype=float).reshape(-1, 2)
    start, end = positions[:, 0], positions[:, 1]
    closed = end <= start
    middle = 0.5 * (start + end)
    first = np.where(closed, middle, start)
    second = np.where(closed, middle, end)
    return np.stack([first, second], axis=1)


def _device_item(
    device_type: str,
    distance: Optional[float],
    boundaries: Optional[np.ndarray],
    positions: np.ndarray,
) -> tuple[Dataset, Dataset]:
    """Beam Limiting Device item and its Beam Limiting Device Position item."""
    pairs = np.asarray(positions, dtype=float).reshape(-1, 2)
    item = Dataset()
    item.RTBeamLimitingDeviceType = device_type
    if distance:
        item.SourceToBeamLimitingDeviceDistance = _ds([distance])[0]
    item.NumberOfLeafJawPairs = len(pairs)
    if boundaries is not None:
        item.LeafPositionBoundaries = _ds(boundaries)
    position = Dataset()
    position.RTBeamLimitingDeviceType = device_type
    # IEC leaf (element) subscript order: all of bank 1, then all of bank 2
    position.LeafJawPositions = _ds(np.concatenate([pairs[:, 0], pairs[:, 1]]))
    return item, position


def _mask_jaws(shape: FieldShapeAsMask) -> dict[str, tuple[float, float]]:
    """Jaws of the bounding box of the open part of a mask (beam's eye view, no mirroring)."""
    mask = np.asarray(shape.mask)
    rows, cols = np.nonzero(mask > 0.5 * mask.max())
    if rows.size == 0:
        raise ValueError("The field shape mask has no open pixel.")
    grid = np.asarray(shape.grid, dtype=float)
    half = 0.5 * float(shape.resolution)
    return {
        "ASYMX": (grid[cols.min()] - half, grid[cols.max()] + half),
        "ASYMY": (grid[rows.min()] - half, grid[rows.max()] + half),
    }


def _flatten(shape) -> list:
    if isinstance(shape, FieldShapeComposite):
        return [leaf for child in shape.shapes for leaf in _flatten(child)]
    return [shape]


def _field_devices(beam, scd: Optional[float]) -> tuple[list[tuple[Dataset, Dataset]], bool]:
    """Beam limiting devices of a field-based beam and whether it is deliverable."""
    shapes = [b for ray in beam.rays for b in ray.beamlets if isinstance(b, FieldShape)]
    if sum(len(ray.beamlets) for ray in beam.rays) != 1 or len(shapes) != 1:
        raise NotImplementedError(
            "Field-based photon beams with more than one field shape are not supported."
        )
    # a field shape lies around its ray; the generators put the single ray on the central axis
    dx, _, dz = (float(v) for v in beam.rays[0].ray_pos_bev)

    jaws: dict[str, tuple[float, float, Optional[float]]] = {}
    mlcs: list[tuple[Dataset, Dataset]] = []
    deliverable = True

    def add_jaw(device_type: str, low: float, high: float, distance: Optional[float]) -> None:
        old = jaws.get(device_type)
        if old is not None:
            low, high = max(low, old[0]), min(high, old[1])
        jaws[device_type] = (low, high, distance)

    for leaf in _flatten(shapes[0]):
        if isinstance(leaf, FieldShapeAsMask):
            deliverable = False
            for device_type, (low, high) in _mask_jaws(leaf).items():
                shift = dx if device_type == "ASYMX" else dz
                add_jaw(device_type, low + shift, high + shift, scd)
            continue
        if not isinstance(leaf, FieldShapeAsBLD):
            raise NotImplementedError(f"Unsupported field shape {type(leaf).__name__}.")
        bld = leaf.bld
        if getattr(bld, "device_angle", 0.0):
            warnings.warn(
                "Beam limiting device angles are ignored by the dose calculation; "
                "exporting with a Beam Limiting Device Angle of 0."
            )
        distance = float(bld.source_to_device_distance or 0.0) or scd
        if isinstance(bld, Jaw):
            low, high = (float(p) for p in bld.positions)
            if bld.device_orientation == "X":
                add_jaw("ASYMX", low + dx, high + dx, distance)
            else:
                add_jaw("ASYMY", low + dz, high + dz, distance)
        elif isinstance(bld, MLC):
            if bld.leaf_positions is None:
                raise ValueError("The multileaf collimator has no leaf positions.")
            if any(item.RTBeamLimitingDeviceType == bld.device_type for item, _ in mlcs):
                raise NotImplementedError(f"More than one {bld.device_type} in a field shape.")
            mlcs.append(_mlc_items(bld, distance, dx, dz))
        else:
            raise NotImplementedError(f"Unsupported beam limiting device {type(bld).__name__}.")

    items = [
        _device_item(device_type, jaws[device_type][2], None, np.array([jaws[device_type][:2]]))
        for device_type in ("ASYMX", "ASYMY")
        if device_type in jaws
    ]
    items.extend(mlcs)
    if not items:
        raise ValueError("The field shape has no beam limiting device.")
    return items, deliverable


def _mlc_items(mlc: MLC, distance: Optional[float], dx: float, dz: float):
    """Device and position items of a multileaf collimator in DICOM (beam's eye view) axes."""
    boundaries = np.asarray(mlc.leaf_position_boundaries, dtype=float)
    edges = np.append(boundaries, boundaries[-1] + mlc.leaf_width)
    positions = np.asarray(mlc.leaf_positions, dtype=float)
    if mlc.device_orientation == "X":
        # leaves move along x, the boundaries lie along y
        pairs = _leaf_pairs(positions) + dx
        boundaries = edges + dz
    else:
        # leaves move along y, the boundaries lie along x
        pairs = _leaf_pairs(positions) + dz
        boundaries = edges + dx
    return _device_item(mlc.device_type, distance, boundaries, pairs)


def _bixel_devices(beam, weights: Optional[np.ndarray], scd: Optional[float]):
    """Jaws of an open field around the (active) bixels of a beam, and the number of bixels."""
    x, z = [], []
    index = 0
    for ray in beam.rays:
        for _ in ray.beamlets:
            if weights is None or weights[index] > 0.0:
                x.append(float(ray.ray_pos_bev[0]))
                z.append(float(ray.ray_pos_bev[2]))
            index += 1
    half = 0.5 * float(beam.bixel_width)
    items = [
        _device_item("ASYMX", scd, None, np.array([[min(x) - half, max(x) + half]])),
        _device_item("ASYMY", scd, None, np.array([[min(z) - half, max(z) + half]])),
    ]
    return items, len(x)


def _beam_item(beam, number, devices, description, machine_name) -> Dataset:
    energy = float(next(b.energy for ray in beam.rays for b in ray.beamlets))

    item = Dataset()
    item.BeamNumber = number
    item.BeamName = f"Beam {number}"
    item.BeamDescription = description
    item.BeamType = "STATIC"
    item.RadiationType = "PHOTON"
    item.TreatmentMachineName = machine_name
    item.PrimaryDosimeterUnit = "MU"
    item.SourceAxisDistance = _ds([beam.sad])[0]
    item.BeamLimitingDeviceSequence = [d for d, _ in devices]
    item.TreatmentDeliveryType = "TREATMENT"
    item.NumberOfWedges = 0
    item.NumberOfCompensators = 0
    item.NumberOfBoli = 0
    item.NumberOfBlocks = 0
    item.FinalCumulativeMetersetWeight = 1.0
    item.NumberOfControlPoints = 2
    item.ReferencedPatientSetupNumber = 1

    first = Dataset()
    first.ControlPointIndex = 0
    first.NominalBeamEnergy = _ds([energy])[0]
    first.GantryAngle = _ds([float(beam.gantry_angle) % 360.0])[0]
    first.GantryRotationDirection = "NONE"
    first.BeamLimitingDeviceAngle = 0.0
    first.BeamLimitingDeviceRotationDirection = "NONE"
    first.PatientSupportAngle = _ds([float(beam.couch_angle) % 360.0])[0]
    first.PatientSupportRotationDirection = "NONE"
    first.TableTopPitchAngle = 0.0
    first.TableTopPitchRotationDirection = "NONE"
    first.TableTopRollAngle = 0.0
    first.TableTopRollRotationDirection = "NONE"
    first.IsocenterPosition = _ds(np.asarray(beam.iso_center, dtype=float))
    first.BeamLimitingDevicePositionSequence = [p for _, p in devices]
    first.CumulativeMetersetWeight = 0.0
    second = Dataset()
    second.ControlPointIndex = 1
    second.CumulativeMetersetWeight = 1.0
    item.ControlPointSequence = [first, second]
    return item


def export_photon_plan(  # noqa: PLR0913 - export options
    stf,
    pln,
    weights,
    directory: str,
    ctx: ExportContext,
    *,
    structure_set_ref: Optional[tuple[str, str]] = None,
    machine: Any = None,
    series_number: int = 4,
) -> PlanReference:
    """
    Write photon beams as a DICOM RT Plan (``RTPLAN.dcm``).

    Parameters
    ----------
    stf : SteeringInformation
        Photon steering information; field-based beams (one field shape) are exported with
        their jaws and multileaf collimators, bixel-based beams as an open field around the
        bixels (geometry only, see the module documentation).
    pln : Plan or None
        Provides ``num_of_fractions`` (1 with a warning if unavailable) and the machine.
    weights : array_like or None
        One weight per bixel (or field shape) in STF order, per fraction. Beams without
        positive weight are not exported and bixels without weight do not open the field.
        There is no meterset (see the module documentation).
    directory : str
        Output directory.
    ctx : ExportContext
        Shared identifiers; ``ctx.patient_position`` must be ``"HFS"``.
    structure_set_ref : tuple of str, optional
        ``(SOP Class UID, SOP Instance UID)`` of an RTSTRUCT exported with the plan.
    machine : Machine or str, optional
        Machine (or its name) providing the source to collimator distance.
    series_number : int
        Series number of the plan.

    Returns
    -------
    PlanReference
        Reference to the plan; the Beam Number of an STF beam is its 0-based index + 1.

    Raises
    ------
    NotImplementedError
        For patient positions other than HFS or unsupported field shapes.
    ValueError
        For non-photon beams, an invalid number of weights, or if no beam is exportable.
    """
    if not is_photon_stf(stf):
        raise ValueError("The photon RT Plan export supports photon beams only.")
    require_head_first_supine(ctx, "RT Plan")
    if weights is not None:
        weights = np.asarray(weights, dtype=float).reshape(-1)
        if weights.size != stf.total_number_of_bixels:
            raise ValueError(
                f"Number of weights ({weights.size}) does not match the number of bixels in the "
                f"stf ({stf.total_number_of_bixels})."
            )
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("Weights must be finite and non-negative.")

    fractions = planned_fractions(pln, "RT Plan")
    machine_obj = _resolve_machine(pln, "photons", machine)
    machine_name = str(stf.beams[0].machine or getattr(machine_obj, "name", "") or "")
    scd = float(getattr(machine_obj, "scd", 0.0) or 0.0) or None

    ds = start_plan(
        ctx, RT_PLAN_STORAGE, "pyRadPlan photon plan", series_number, structure_set_ref
    )

    beam_items = Sequence()
    referenced_beams = Sequence()
    skipped: list[int] = []
    geometry_only = False
    offset = 0
    for index, beam in enumerate(stf.beams):
        count = beam.total_number_of_bixels
        beam_weights = None if weights is None else weights[offset : offset + count]
        offset += count
        number = index + 1
        if beam_weights is not None and not np.any(beam_weights > 0.0):
            warnings.warn(
                f"Beam {number} has no positive weight and is not exported.", stacklevel=3
            )
            skipped.append(index)
            continue

        if _is_field_beam(beam):
            devices, deliverable = _field_devices(beam, scd)
            description = "pyRadPlan field-based beam"
            if beam_weights is not None:
                description += f", fluence weight {float(beam_weights[0]):.6g} per fraction"
            description += " (no calibration to monitor units)"
            if not deliverable:
                description += "; field shape given as mask, exported as its bounding box only"
        else:
            devices, active = _bixel_devices(beam, beam_weights, scd)
            deliverable = active == 1
            if deliverable:
                description = "pyRadPlan collimated square field (no calibration to monitor units)"
            else:
                description = (
                    f"pyRadPlan bixel-based beam ({active} bixels of {beam.bixel_width:g} mm): "
                    "open field around the bixels, geometry only, NOT deliverable"
                )
        geometry_only |= not deliverable

        beam_items.append(_beam_item(beam, number, devices, description, machine_name))
        referenced = Dataset()
        referenced.ReferencedBeamNumber = number
        referenced_beams.append(referenced)

    if not beam_items:
        raise ValueError("No beam has a positive weight; there is no plan to export.")

    if geometry_only:
        ds.RTPlanDescription = (
            "pyRadPlan photon plan with beam geometry only (no leaf sequencing, no meterset): "
            "NOT deliverable"
        )
        warnings.warn(
            "The photon plan has bixel-based or mask-based beams, for which pyRadPlan has no "
            "leaf sequencing: the RT Plan holds only the beam geometry (angles, isocenter, "
            "energy and an open field) and is not deliverable.",
            stacklevel=3,
        )

    group = Dataset()
    group.FractionGroupNumber = 1
    group.NumberOfFractionsPlanned = fractions
    group.NumberOfBeams = len(beam_items)
    group.NumberOfBrachyApplicationSetups = 0
    group.ReferencedBeamSequence = referenced_beams
    ds.FractionGroupSequence = [group]
    ds.BeamSequence = beam_items
    ds.ApprovalStatus = "UNAPPROVED"

    ds.save_as(os.path.join(directory, "RTPLAN.dcm"), enforce_file_format=True)

    return PlanReference(
        sop_instance_uid=str(ds.SOPInstanceUID),
        fraction_group_number=1,
        beam_numbers=list(range(1, len(stf.beams) + 1)),
        sop_class_uid=RT_PLAN_STORAGE,
        skipped_beams=skipped,
    )

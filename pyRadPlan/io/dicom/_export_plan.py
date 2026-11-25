"""Export a pyRadPlan ion plan (steering information and spot weights) as a DICOM RT Ion Plan.

Geometry
--------
The exported geometry follows the IEC 61217 coordinate systems used by the RT Ion Beams
module (DICOM PS3.3 C.8.8.25 and the introduction of the Radiotherapy Modules):

* The DICOM patient coordinate system is the IEC PATIENT system rotated by 90 degrees about
  its x-axis, ``(X_DICOM, Y_DICOM, Z_DICOM) = (X_IEC, -Z_IEC, Y_IEC)``. For a patient in Head
  First Supine position (the only one supported) the IEC PATIENT system coincides with the
  IEC FIXED system at Patient Support Angle 0 (Y towards the gantry bearing, Z up), so the
  fixed coordinates of a patient point relative to the isocenter are ``(x, z, -y)`` in LPS.
* The IEC GANTRY system is rotated about the Y axis of the IEC FIXED system. Gantry angle 0
  puts the source at ``+Z_FIXED`` (above the patient, beam travelling downwards); gantry
  angle 90 puts it at ``+X_FIXED``, i.e. at the patient's left for HFS (clockwise rotation
  seen from the isocenter looking at the gantry). The Scan Spot Position Map holds the
  ``(x, y)`` coordinates of the spots "as projected onto the machine isocentric plane in the
  IEC GANTRY coordinate system".
* The IEC PATIENT SUPPORT system is rotated about the vertical (``Z_FIXED``) axis, positive
  counter-clockwise seen from above. This sign follows from the standard's remark on seated
  treatments (gantry angle 90 pairs with patient support angle 270 for a patient looking
  into the beam).

pyRadPlan's own conventions (``pyRadPlan.geometry.lps``, ``Beam.source_point``) are identical
to these for Head First Supine, so angles and beam's eye view spot positions are exported
unchanged: with ``R = R_couch(c) @ R_gantry(g)``, the spot at ``ray_pos_bev = (x, 0, z)`` lies
at ``R @ ray_pos_bev`` relative to the isocenter, which is the IEC GANTRY plane position
``(x, y) = (x, z)`` mapped through the IEC chain above. The IEC GANTRY X axis is the BEV x
axis (not mirrored) and the IEC GANTRY Y axis the BEV z axis.

Weights
-------
Spot weights are per fraction (the influence matrix of pyRadPlan is always per fraction,
independent of ``Plan.dose_convention``) and are written as numbers of particles,
``w * 1e6`` (a weight is in units of 1e6 particles, as the dose engines interpret it), with
Primary Dosimeter Unit ``NP``: Scan Spot Meterset Weights, Beam Meterset and Final
Cumulative Meterset Weight are all particle numbers per fraction. Spots with zero weight and
layers without remaining spots are dropped; beams without any spot are skipped with a warning.
"""

import os
import warnings
from typing import Any, Optional

import numpy as np
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from pydicom.uid import generate_uid

from ._export_common import (
    RT_ION_PLAN_STORAGE,
    ExportContext,
    PlanReference,
    ds_list,
    populate_common,
)

#: pyRadPlan radiation mode -> (RadiationType, mass number, atomic number, charge state).
_ION_SPECIES = {
    "protons": ("PROTON", None, None, None),
    "helium": ("ION", 4, 2, 2),
    "carbon": ("ION", 12, 6, 6),
    "oxygen": ("ION", 16, 8, 8),
}

_PARTICLES_PER_WEIGHT = 1.0e6

_FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def is_ion_stf(stf) -> bool:
    """Whether all beams of the steering information use an ion radiation mode."""
    return len(stf.beams) > 0 and all(b.radiation_mode in _ION_SPECIES for b in stf.beams)


def _resolve_machine(pln, radiation_mode: str, machine: Any) -> Any:
    candidate = machine if machine is not None else getattr(pln, "machine", None)
    if candidate is None or not isinstance(candidate, str):
        return candidate
    try:
        from pyRadPlan.machines import load_from_name  # noqa: PLC0415

        return load_from_name(radiation_mode, candidate)
    except Exception:  # noqa: BLE001 - machine data is optional for the export
        return None


def _spot_size(machine: Any, energy: float, focus_ix: int, sad: float) -> Optional[float]:
    """FWHM of the spot at the isocenter in mm, from the machine's focus data if available."""
    foci = getattr(machine, "foci", None)
    if not foci:
        return None
    try:
        keys = list(foci.keys())
        key = keys[int(np.argmin(np.abs(np.asarray(keys, dtype=float) - energy)))]
        entries = foci[key]
        focus = entries[min(max(focus_ix, 0), len(entries) - 1)]
        if focus.fwhm_iso is not None and np.isfinite(focus.fwhm_iso):
            return float(focus.fwhm_iso)
        if focus.dist.min() <= sad <= focus.dist.max():
            return float(_FWHM_PER_SIGMA * np.interp(sad, focus.dist, focus.sigma))
    except Exception:  # noqa: BLE001 - spot size is Type 3
        return None
    return None


def _layers(beam, weights: np.ndarray, offset: int) -> list[dict]:
    """Group the positive-weight spots of a beam into layers of equal energy and settings."""
    layers: dict = {}
    index = offset
    for ray in beam.rays:
        x, y = float(ray.ray_pos_bev[0]), float(ray.ray_pos_bev[2])
        for beamlet in ray.beamlets:
            weight = float(weights[index])
            index += 1
            if not weight > 0.0:
                continue
            shifter = beamlet.range_shifter
            shifter_key = (
                (
                    float(shifter.eq_thickness),
                    float(shifter.source_rashi_distance),
                    int(shifter.id),
                )
                if shifter.eq_thickness > 0
                else None
            )
            key = (round(float(beamlet.energy), 2), int(beamlet.focus_ix), shifter_key)
            layer = layers.setdefault(
                key,
                {
                    "energy": float(beamlet.energy),
                    "focus_ix": int(beamlet.focus_ix),
                    "shifter": shifter_key,
                    "positions": [],
                    "particles": [],
                },
            )
            layer["positions"].append((x, y))
            layer["particles"].append(weight * _PARTICLES_PER_WEIGHT)
    # highest energy first, as a beam is usually delivered
    return sorted(layers.values(), key=lambda lay: (-lay["energy"], lay["focus_ix"]))


def _range_shifter_sequence(shifters: list) -> Sequence:
    seq = Sequence()
    for number, (thickness, source_distance, shifter_id) in enumerate(shifters, start=1):
        item = Dataset()
        item.RangeShifterNumber = number
        item.RangeShifterID = f"RS{shifter_id}" if shifter_id else f"RS{number}"
        item.RangeShifterType = "BINARY"
        item.RangeShifterDescription = (
            f"water equivalent thickness {thickness:g} mm, {source_distance:g} mm from source"
        )
        seq.append(item)
    return seq


def _range_shifter_settings(shifters: list, active, sad: float) -> Sequence:
    seq = Sequence()
    for number, shifter in enumerate(shifters, start=1):
        thickness, source_distance, _ = shifter
        item = Dataset()
        item.ReferencedRangeShifterNumber = number
        item.RangeShifterSetting = "IN" if shifter == active else "OUT"
        item.IsocenterToRangeShifterDistance = ds_list([sad - source_distance])[0]
        item.RangeShifterWaterEquivalentThickness = ds_list([thickness])[0]
        seq.append(item)
    return seq


def _control_point(  # noqa: PLR0913 - DICOM control point attributes
    index, cumulative, layer, machine, sad, shifters, spot_weights
) -> Dataset:
    cp = Dataset()
    cp.ControlPointIndex = index
    cp.NominalBeamEnergy = ds_list([layer["energy"]])[0]
    cp.CumulativeMetersetWeight = ds_list([cumulative])[0]
    cp.ScanSpotTuneID = f"focus{layer['focus_ix']}"
    positions = np.asarray(layer["positions"], dtype=float).reshape(-1, 2)
    cp.NumberOfScanSpotPositions = len(positions)
    cp.ScanSpotPositionMap = [float(v) for v in positions.reshape(-1)]
    cp.ScanSpotMetersetWeights = [float(v) for v in spot_weights]
    size = _spot_size(machine, layer["energy"], layer["focus_ix"], sad)
    if size is not None:
        cp.ScanningSpotSize = [size, size]
    cp.NumberOfPaintings = 1
    if shifters:
        cp.RangeShifterSettingsSequence = _range_shifter_settings(shifters, layer["shifter"], sad)
    return cp


def _beam_item(  # noqa: PLR0913 - DICOM beam attributes
    beam,
    number: int,
    layers: list[dict],
    machine: Any,
    machine_name: str,
    snout_position: float,
) -> tuple[Dataset, float]:
    sad = float(beam.sad)
    radiation_type, mass, atomic, charge = _ION_SPECIES[beam.radiation_mode]

    item = Dataset()
    item.BeamNumber = number
    item.BeamName = f"Beam {number}"
    item.BeamType = "STATIC"
    item.RadiationType = radiation_type
    if radiation_type == "ION":
        item.RadiationMassNumber = mass
        item.RadiationAtomicNumber = atomic
        item.RadiationChargeState = charge
    item.ScanMode = "MODULATED"
    item.TreatmentMachineName = machine_name
    item.PrimaryDosimeterUnit = "NP"
    item.VirtualSourceAxisDistances = ds_list([sad, sad])
    item.TreatmentDeliveryType = "TREATMENT"

    shifters = sorted({lay["shifter"] for lay in layers if lay["shifter"] is not None})
    item.NumberOfWedges = 0
    item.NumberOfCompensators = 0
    item.NumberOfBoli = 0
    item.NumberOfBlocks = 0
    item.NumberOfRangeModulators = 0
    item.NumberOfLateralSpreadingDevices = 0
    item.NumberOfRangeShifters = len(shifters)
    if shifters:
        item.RangeShifterSequence = _range_shifter_sequence(shifters)
    item.PatientSupportType = "TABLE"
    item.ReferencedPatientSetupNumber = 1

    # Meterset weights are particle numbers. The sums use the stored (float32) values so that
    # the cumulative weights match the spot weights.
    totals = [float(np.sum(np.asarray(lay["particles"], dtype=np.float32))) for lay in layers]
    final = float(np.sum(totals))
    item.FinalCumulativeMetersetWeight = ds_list([final])[0]
    item.NumberOfControlPoints = 2 * len(layers)

    control_points = Sequence()
    cumulative = 0.0
    for k, layer in enumerate(layers):
        first = _control_point(
            2 * k, cumulative, layer, machine, sad, shifters, layer["particles"]
        )
        if k == 0:
            first.GantryAngle = ds_list([float(beam.gantry_angle) % 360.0])[0]
            first.GantryRotationDirection = "NONE"
            first.BeamLimitingDeviceAngle = 0.0
            first.BeamLimitingDeviceRotationDirection = "NONE"
            first.PatientSupportAngle = ds_list([float(beam.couch_angle) % 360.0])[0]
            first.PatientSupportRotationDirection = "NONE"
            first.TableTopPitchAngle = 0.0
            first.TableTopPitchRotationDirection = "NONE"
            first.TableTopRollAngle = 0.0
            first.TableTopRollRotationDirection = "NONE"
            first.IsocenterPosition = ds_list(np.asarray(beam.iso_center, dtype=float))
            first.SnoutPosition = ds_list([snout_position])[0]
        control_points.append(first)
        cumulative += totals[k]
        control_points.append(
            _control_point(
                2 * k + 1,
                cumulative,
                layer,
                machine,
                sad,
                shifters,
                np.zeros(len(layer["particles"])),
            )
        )
    item.IonControlPointSequence = control_points
    return item, final


def planned_fractions(pln, plan_kind: str) -> int:
    """Return the planned fractions of the plan; one, with a warning, without a plan."""
    if pln is None or getattr(pln, "num_of_fractions", None) is None:
        warnings.warn(f"No plan given; exporting the {plan_kind} with one planned fraction.")
        return 1
    return int(pln.num_of_fractions)


def start_plan(
    ctx: ExportContext,
    sop_class_uid: str,
    plan_name: str,
    series_number: int,
    structure_set_ref: Optional[tuple[str, str]],
) -> Dataset:
    """Dataset with the identifiers, RT General Plan and RT Patient Setup of an RT (Ion) Plan."""
    ds = Dataset()
    populate_common(
        ds,
        ctx,
        sop_class_uid,
        generate_uid(),
        "RTPLAN",
        series_number,
        f"{plan_name} - research use only",
    )
    ds.OperatorsName = ""
    ds.InstanceNumber = 1

    ds.RTPlanLabel = "pyRadPlan"
    ds.RTPlanName = plan_name
    ds.RTPlanDate = ctx.export_date
    ds.RTPlanTime = ctx.export_time
    ds.PlanIntent = "RESEARCH"
    # Referenced Structure Set Sequence is required for geometry PATIENT, so a plan exported
    # without an RTSTRUCT is declared as TREATMENT_DEVICE (isocenters stay patient-based).
    ds.RTPlanGeometry = "PATIENT" if structure_set_ref is not None else "TREATMENT_DEVICE"
    if structure_set_ref is not None:
        ref = Dataset()
        ref.ReferencedSOPClassUID, ref.ReferencedSOPInstanceUID = structure_set_ref
        ds.ReferencedStructureSetSequence = [ref]

    setup = Dataset()
    setup.PatientSetupNumber = 1
    setup.PatientPosition = "HFS"
    ds.PatientSetupSequence = [setup]
    return ds


def require_head_first_supine(ctx: ExportContext, plan_kind: str) -> None:
    if (ctx.patient_position or "HFS").upper() != "HFS":
        raise NotImplementedError(
            f"{plan_kind} export supports Head First Supine only, not {ctx.patient_position!r}."
        )


def export_ion_plan(  # noqa: PLR0913 - export options
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
    Write ion beams with spot weights as a DICOM RT Ion Plan (``RTPLAN.dcm``).

    Parameters
    ----------
    stf : SteeringInformation
        Ion steering information. Photon beams are not supported.
    pln : Plan or None
        Provides ``num_of_fractions`` (1 with a warning if unavailable) and the machine.
    weights : array_like
        One weight per bixel in STF order (beam, ray, beamlet), per fraction. The number of
        particles of a spot is ``weight * 1e6`` (Primary Dosimeter Unit ``NP``).
    directory : str
        Output directory.
    ctx : ExportContext
        Shared identifiers; ``ctx.patient_position`` must be ``"HFS"``.
    structure_set_ref : tuple of str, optional
        ``(SOP Class UID, SOP Instance UID)`` of an RTSTRUCT exported with the plan.
    machine : Machine or str, optional
        Machine (or its name) providing spot sizes; defaults to the plan's machine.
    series_number : int
        Series number of the plan.

    Returns
    -------
    PlanReference
        Reference to the plan; the Beam Number of an STF beam is its 0-based index + 1.
        Beams without any positive weight are not exported (``skipped_beams``).

    Raises
    ------
    NotImplementedError
        For patient positions other than HFS.
    ValueError
        For photon beams, an invalid number of weights, or if no spot has positive weight.
    """
    if not is_ion_stf(stf):
        raise ValueError("The RT Ion Plan export supports ion beams only.")
    require_head_first_supine(ctx, "RT Ion Plan")
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != stf.total_number_of_bixels:
        raise ValueError(
            f"Number of weights ({weights.size}) does not match the number of bixels in the "
            f"stf ({stf.total_number_of_bixels})."
        )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Spot weights must be finite and non-negative.")

    fractions = planned_fractions(pln, "RT Ion Plan")

    first_beam = stf.beams[0]
    machine_obj = _resolve_machine(pln, first_beam.radiation_mode, machine)
    machine_name = str(first_beam.machine or getattr(machine_obj, "name", "") or "")
    snout = float(getattr(machine_obj, "bams_to_iso_dist", 0.0) or 0.0)

    ds = start_plan(
        ctx,
        RT_ION_PLAN_STORAGE,
        "pyRadPlan ion plan",
        series_number,
        structure_set_ref,
    )

    beam_items = Sequence()
    referenced_beams = Sequence()
    skipped: list[int] = []
    offset = 0
    for index, beam in enumerate(stf.beams):
        layers = _layers(beam, weights, offset)
        offset += beam.total_number_of_bixels
        number = index + 1
        if not layers:
            warnings.warn(
                f"Beam {number} has no spot with positive weight and is not exported.",
                stacklevel=3,
            )
            skipped.append(index)
            continue
        item, total = _beam_item(beam, number, layers, machine_obj, machine_name, snout)
        beam_items.append(item)
        referenced = Dataset()
        referenced.BeamMeterset = ds_list([total])[0]
        referenced.ReferencedBeamNumber = number
        referenced_beams.append(referenced)

    if not beam_items:
        raise ValueError("No beam has a spot with positive weight; there is no plan to export.")

    group = Dataset()
    group.FractionGroupNumber = 1
    group.NumberOfFractionsPlanned = fractions
    group.NumberOfBeams = len(beam_items)
    group.NumberOfBrachyApplicationSetups = 0
    group.ReferencedBeamSequence = referenced_beams
    ds.FractionGroupSequence = [group]
    ds.IonBeamSequence = beam_items
    ds.ApprovalStatus = "UNAPPROVED"

    ds.save_as(os.path.join(directory, "RTPLAN.dcm"), enforce_file_format=True)

    return PlanReference(
        sop_instance_uid=str(ds.SOPInstanceUID),
        fraction_group_number=1,
        beam_numbers=list(range(1, len(stf.beams) + 1)),
        sop_class_uid=RT_ION_PLAN_STORAGE,
        skipped_beams=skipped,
    )

"""DICOM RT Ion Plan import: steering information, spot weights and plan settings.

Coordinate conventions (DICOM PS3.3, RT Ion Beams Module, C.8.8.25, and IEC 61217)
-----------------------------------------------------------------------------------
The geometry is derived here from the standard and *not* from pyRadPlan's rotation
helpers; the stf is then expressed in pyRadPlan's own convention at the very end
(:func:`_equivalent_pyradplan_angles`).

* IEC FIXED system ``F``: Z points upwards and the Y axis points towards the gantry
  bearing (PS3.3 C.8.8.25.6.1: "The Y-axis of IEC GANTRY points towards the (virtual)
  gantry bearing. The Y-axis of the IEC FIXED coordinate system has to point in the same
  direction. Z-axis in IEC FIXED coordinate system is always pointing upwards."); X
  completes the right-handed system (to the right when facing the gantry).
* IEC GANTRY system ``G`` is ``F`` rotated by the Gantry Angle about Y. The radiation source
  is on ``+Z_G`` (beam travels along ``-Z_G``). The gantry angle is the "orientation of IEC
  GANTRY coordinate system with respect to IEC FIXED REFERENCE coordinate system" and
  increases clockwise when facing the gantry.
* Scan Spot Position Map (300A,0394): "(x,y) pairs that define the coordinates of the scan
  spots as projected onto the machine isocentric plane in the IEC GANTRY coordinate system
  (mm)", i.e. along ``X_G`` and ``Y_G``.
* Patient Support Angle (300A,0122): "orientation of IEC PATIENT SUPPORT (turntable)
  coordinate system with respect to IEC FIXED REFERENCE coordinate system", positive
  counter-clockwise viewed from above (IEC 61217): the patient rotates by ``R_Z(theta)``.
* Patient Position (0018,5100) and the DICOM patient system: "(X_DICOM, Y_DICOM, Z_DICOM) =
  (X_IEC, -Z_IEC, Y_IEC)" for the HFS reference orientation (PS3.3 C.7.6.2.1.1 / C.8.8.25).
  The other positions are 180 degree rotations of HFS about the IEC fixed axes.
* Isocenter Position (300A,012C) is given "in the Patient-Based Coordinate System" (LPS, mm).
* Meterset at a control point = Beam Meterset * Cumulative Meterset Weight / Final Cumulative
  Meterset Weight (PS3.3 C.8.8.14.?/C.8.8.25). For an irradiation segment (control point pair)
  "Scan Spot Meterset Weights (300A,0396) Values add up to Meterset difference between
  Control Points" of the pair.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Optional, Union

import numpy as np
import pydicom

from pyRadPlan.geometry import lps
from pyRadPlan.machines import Machine, load_from_name
from pyRadPlan.plan import IonPlan
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.stf._beam import Beam
from pyRadPlan.stf._beamlet import IonSpot
from pyRadPlan.stf._ray import Ray
from pyRadPlan.stf._rangeshifter import RangeShifter

logger = logging.getLogger(__name__)

RT_ION_PLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.8"
RT_PLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"

_ION_BY_ATOMIC_NUMBER = {2: "helium", 6: "carbon", 8: "oxygen"}

# Reference patient orientations in the IEC FIXED system at Patient Support Angle 0:
# ``f = M @ p`` with p the DICOM patient (LPS) coordinates. HFS: (X, Y, Z)_IEC = (x, z, -y).
_M_HFS = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
_PATIENT_POSITIONS = {
    "HFS": _M_HFS,
    "HFP": np.diag([-1.0, 1.0, -1.0]) @ _M_HFS,  # 180 deg about the head axis (Y_F)
    "FFS": np.diag([-1.0, -1.0, 1.0]) @ _M_HFS,  # 180 deg about the vertical (Z_F)
    "FFP": np.diag([1.0, -1.0, -1.0]) @ _M_HFS,  # 180 deg about X_F
}

ParticlesPerMU = Union[float, Callable[[float], float]]

_PARTICLES_PER_WEIGHT = 1.0e6


def plan_kind(ds: pydicom.Dataset) -> str:
    """Return ``"ion"``, ``"photon"`` or ``"other"`` for an RTPLAN dataset."""
    sop = str(getattr(ds, "SOPClassUID", "") or "")
    if sop == RT_ION_PLAN_SOP_CLASS:
        return "ion"
    if sop == RT_PLAN_SOP_CLASS:
        return "photon"
    if hasattr(ds, "IonBeamSequence"):
        return "ion"
    if hasattr(ds, "BeamSequence"):
        return "photon"
    return "other"


def _radiation_mode_of(beam: pydicom.Dataset) -> str:
    rad_type = str(getattr(beam, "RadiationType", "") or "").upper()
    if rad_type == "PROTON":
        return "protons"
    if rad_type == "ION":
        z = getattr(beam, "RadiationAtomicNumber", None)
        if z is None or int(z) not in _ION_BY_ATOMIC_NUMBER:
            raise ValueError(
                f"Unsupported ion (Radiation Atomic Number {z!r}); supported are "
                f"{sorted(_ION_BY_ATOMIC_NUMBER)} (helium, carbon, oxygen)."
            )
        return _ION_BY_ATOMIC_NUMBER[int(z)]
    raise ValueError(f"Unsupported Radiation Type {rad_type!r} in ion plan.")


def plan_summary(ds: pydicom.Dataset, path: str) -> dict:
    """Summary of an RTPLAN header for listings."""
    beams = getattr(ds, "IonBeamSequence", None) or getattr(ds, "BeamSequence", None) or []
    radiation_types = []
    for beam in beams:
        try:
            label = _radiation_mode_of(beam) if plan_kind(ds) == "ion" else None
        except ValueError:
            label = None
        label = label or str(getattr(beam, "RadiationType", "") or "")
        if label and label not in radiation_types:
            radiation_types.append(label)
    fractions = None
    groups = getattr(ds, "FractionGroupSequence", None)
    if groups:
        fractions = getattr(groups[0], "NumberOfFractionsPlanned", None)
        fractions = int(fractions) if fractions is not None else None
    return {
        "path": path,
        "label": str(getattr(ds, "RTPlanLabel", "") or ""),
        "name": str(getattr(ds, "RTPlanName", "") or ""),
        "sop_class": str(getattr(ds, "SOPClassUID", "") or ""),
        "kind": plan_kind(ds),
        "radiation_type": ", ".join(radiation_types),
        "num_beams": len(beams),
        "num_fractions": fractions,
    }


def _rot_y(angle_deg: float) -> np.ndarray:
    """Active rotation about the fixed Y axis; its columns are the gantry axes (fixed system)."""
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle_deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(angle_deg)), np.sin(np.radians(angle_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def iec_beam_frame(
    patient_position: str, gantry_angle: float, support_angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """IEC gantry axes and source direction expressed in DICOM patient (LPS) coordinates.

    Returns
    -------
    axes : ndarray (3, 3)
        Columns are ``X_G``, ``Y_G`` and ``Z_G`` in LPS (``Z_G`` points to the source).
    source_dir : ndarray (3,)
        Unit vector from the isocenter towards the source (``+Z_G``).
    """
    m = _PATIENT_POSITIONS[patient_position]
    # f = R_Z(theta) @ M @ p   =>   p = M^T R_Z(-theta) f
    to_lps = m.T @ _rot_z(-support_angle)
    fixed_axes = _rot_y(gantry_angle)  # columns X_G, Y_G, Z_G in F
    axes = to_lps @ fixed_axes
    return axes, axes[:, 2]


def _equivalent_pyradplan_angles(beam_dir: np.ndarray) -> tuple[float, float]:
    """Gantry and couch angle with ``R_couch(c) R_gantry(g) @ (0, 1, 0) = beam_dir``.

    pyRadPlan's BEV beam axis is ``+y`` and ``R = R_y(couch) @ R_z(gantry)`` (see
    :mod:`pyRadPlan.geometry.lps`), so ``d = (-cos c sin g, cos g, sin c sin g)``. Two
    solutions exist, ``(g, c)`` and ``(360 - g, c + 180)``; the one with ``|c| <= 90``
    is returned (for HFS this is the identity on the DICOM angles).
    """
    d = np.asarray(beam_dir, dtype=float)
    g = np.arccos(np.clip(d[1], -1.0, 1.0))  # [0, pi]
    if np.sin(g) < 1e-9:
        gantry, couch = np.degrees(g), 0.0
    else:
        couch = np.degrees(np.arctan2(d[2], -d[0]))
        gantry = np.degrees(g)
        if abs(couch) > 90.0 + 1e-9:
            couch = couch - 180.0 if couch > 0 else couch + 180.0
            gantry = 360.0 - gantry
    return round(gantry % 360.0, 9), round(couch, 9)


def _float_list(value, n: Optional[int] = None) -> np.ndarray:
    arr = np.atleast_1d(np.asarray([float(v) for v in np.atleast_1d(value)], dtype=float))
    if n is not None and arr.size % n != 0:
        raise ValueError(f"Expected a multiple of {n} values, got {arr.size}.")
    return arr


def _resolve_machine(machine: Any, radiation_mode: str) -> Machine:
    if machine is None:
        machine = "Generic"
    if isinstance(machine, str):
        return load_from_name(radiation_mode, machine)
    if getattr(machine, "radiation_mode", radiation_mode) != radiation_mode:
        raise ValueError(
            f"Machine radiation mode '{machine.radiation_mode}' does not match the plan's "
            f"'{radiation_mode}'."
        )
    return machine


def _map_energies(
    energies: np.ndarray, machine: Machine, mass_number: int, tolerance: float
) -> dict[float, float]:
    """Map DICOM nominal energies to machine energies.

    The DICOM energy is first compared as is, then divided by the mass number (ion plans
    may list the total kinetic energy in MeV instead of MeV/u). The tolerance is in the
    machine's energy unit.
    """
    machine_energies = np.asarray(machine.energies, dtype=float)
    mapping, missing = {}, []
    for energy in np.unique(energies):
        for candidate in (energy, energy / mass_number) if mass_number > 1 else (energy,):
            ix = int(np.argmin(np.abs(machine_energies - candidate)))
            if abs(machine_energies[ix] - candidate) <= tolerance:
                mapping[float(energy)] = float(machine_energies[ix])
                break
        else:
            missing.append(float(energy))
    if missing:
        shown = ", ".join(f"{m:.3f}" for m in missing[:5])
        if len(missing) > 5:
            shown += f", ... ({len(missing)} in total)"
        raise ValueError(
            f"Nominal beam energies [{shown}] have no machine energy within {tolerance} "
            f"(MeV/u) in machine '{machine.name}' ({machine.energies.min():.2f} to "
            f"{machine.energies.max():.2f}). Pass another machine or a larger "
            "`energy_tolerance`."
        )
    return mapping


def _focus_index(machine: Machine, energy: float, spot_size: Optional[np.ndarray]) -> int:
    """Focus whose FWHM at the isocenter is closest to the plan's Scanning Spot Size."""
    if spot_size is None:
        return 0
    foci = machine.foci.get(energy)
    if not foci:
        return 0
    fwhm = np.array([np.nan if f.fwhm_iso is None else f.fwhm_iso for f in foci], dtype=float)
    if np.all(np.isnan(fwhm)):
        return 0
    return int(np.nanargmin(np.abs(fwhm - float(np.mean(spot_size)))))


def _ppm(particles_per_mu: Optional[ParticlesPerMU], energy: float) -> float:
    value = particles_per_mu(energy) if callable(particles_per_mu) else particles_per_mu
    return float(value)


def import_ion_plan(
    ds: pydicom.Dataset,
    ct=None,
    machine: Union[str, Machine, None] = None,
    particles_per_mu: Optional[ParticlesPerMU] = None,
    energy_tolerance: float = 0.1,
) -> dict:
    """Convert an RT Ion Plan dataset into stf, spot weights and plan.

    Parameters
    ----------
    ds : pydicom.Dataset
        The RT Ion Plan.
    ct : CT, optional
        Only used to check that the isocenters lie within the CT.
    machine : str or Machine, optional
        Machine (default ``"Generic"``) providing the energies and foci.
    particles_per_mu : float or callable, optional
        Calibration in particles per MU (optionally a function of the machine energy), required
        when the beams' Primary Dosimeter Unit is ``MU``. A spot weight is in units of 1e6
        particles, as the dose engines interpret it, so the weight of a spot of ``m`` MU is
        ``m * particles_per_mu / 1e6``; the calibration is kept as ``num_particles_per_mu`` of
        the beamlets.
    energy_tolerance : float
        Maximum difference to the closest machine energy in MeV(/u).

    Returns
    -------
    dict
        ``stf`` (:class:`SteeringInformation`), ``pln`` (:class:`IonPlan`),
        ``weights`` (spot weights in stf bixel order, in units of 1e6 particles) and
        ``result`` (``{"w": weights}``, the key the workspace and matRad results use).
    """
    kind = plan_kind(ds)
    if kind == "photon":
        raise NotImplementedError(
            "Import of photon RT Plans (SOP class 1.2.840.10008.5.1.4.1.1.481.5) is not "
            "supported yet; only RT Ion Plans can be imported."
        )
    if kind != "ion":
        raise ValueError("The dataset is not an RT Ion Plan.")

    beams_ds = list(getattr(ds, "IonBeamSequence", []) or [])
    if not beams_ds:
        raise ValueError("RT Ion Plan without Ion Beam Sequence.")

    # --- fraction group: meterset per beam and number of fractions
    groups = list(getattr(ds, "FractionGroupSequence", []) or [])
    if not groups:
        raise ValueError("RT Ion Plan without Fraction Group Sequence.")
    if len(groups) > 1:
        warnings.warn(
            f"RT Ion Plan has {len(groups)} fraction groups; using the first one.",
            stacklevel=3,
        )
    group = groups[0]
    num_fractions = getattr(group, "NumberOfFractionsPlanned", None)
    if num_fractions is None or int(num_fractions) < 1:
        warnings.warn("Number of Fractions Planned missing; assuming 1.", stacklevel=3)
        num_fractions = 1
    meterset = {
        int(ref.ReferencedBeamNumber): float(getattr(ref, "BeamMeterset", 0.0) or 0.0)
        for ref in getattr(group, "ReferencedBeamSequence", [])
    }

    setups = {
        int(s.PatientSetupNumber): str(getattr(s, "PatientPosition", "") or "")
        for s in getattr(ds, "PatientSetupSequence", []) or []
    }

    selected = [b for b in beams_ds if meterset.get(int(b.BeamNumber), 0.0) > 0.0]
    if not any("BeamMeterset" in ref for ref in getattr(group, "ReferencedBeamSequence", [])):
        raise ValueError(
            "The RT Ion Plan has no Beam Meterset (300A,0086) in its Referenced Beam Sequence, "
            "which is needed to scale the spot weights."
        )
    skipped = len(beams_ds) - len(selected)
    if skipped:
        logger.info("Skipping %d beam(s) with no meterset (e.g. setup fields).", skipped)
    if not selected:
        raise ValueError("No treatment beam with a Beam Meterset > 0 in the RT Ion Plan.")

    modes = {_radiation_mode_of(b) for b in selected}
    if len(modes) != 1:
        raise ValueError(f"Beams of different radiation types are not supported: {sorted(modes)}")
    radiation_mode = modes.pop()
    mass_number = int(getattr(selected[0], "RadiationMassNumber", 1) or 1)
    mach = _resolve_machine(machine, radiation_mode)

    beam_models: list[Beam] = []
    weights: list[float] = []
    gantry_angles, couch_angles, iso_centers = [], [], []

    for beam_ds in selected:
        beam, beam_weights = _import_beam(
            beam_ds,
            beam_meterset=meterset[int(beam_ds.BeamNumber)],
            setups=setups,
            machine=mach,
            radiation_mode=radiation_mode,
            mass_number=mass_number,
            particles_per_mu=particles_per_mu,
            energy_tolerance=energy_tolerance,
        )
        beam_models.append(beam)
        weights.extend(beam_weights)
        gantry_angles.append(beam.gantry_angle)
        couch_angles.append(beam.couch_angle)
        iso_centers.append(beam.iso_center.tolist())

    if ct is not None:
        _check_isocenters(ct, iso_centers)

    stf = SteeringInformation(beams=beam_models)
    pln = IonPlan(
        radiation_mode=radiation_mode,
        machine=mach if not isinstance(machine, (str, type(None))) else mach.name,
        num_of_fractions=int(num_fractions),
        dose_convention="per_fraction",
        prop_stf={
            "generator": "IMPT",
            "gantry_angles": gantry_angles,
            "couch_angles": couch_angles,
            "iso_center": iso_centers,
        },
    )
    w = np.asarray(weights, dtype=float)
    return {"stf": stf, "pln": pln, "weights": w, "result": {"w": w}}


def _check_isocenters(ct, iso_centers) -> None:
    image = getattr(ct, "cube_hu", None)
    if image is None:
        return
    try:
        size = np.asarray(image.GetSize())
        for iso in iso_centers:
            idx = np.asarray(image.TransformPhysicalPointToContinuousIndex(tuple(iso)))
            if np.any(idx < -0.5) or np.any(idx > size - 0.5):
                warnings.warn(
                    f"Isocenter {np.round(iso, 1).tolist()} mm lies outside the CT; check that "
                    "plan and CT share the Frame of Reference.",
                    stacklevel=4,
                )
    except Exception:  # noqa: BLE001 - sanity check only
        logger.debug("Isocenter check against the CT failed.", exc_info=True)


def _first(cps, name: str, default=None):
    for cp in cps:
        if name in cp:
            return getattr(cp, name)
    return default


def _check_supported(beam_ds, name: str, unit: str, particles_per_mu) -> None:
    if str(getattr(beam_ds, "BeamType", "STATIC")).upper() != "STATIC":
        raise NotImplementedError(f"Beam '{name}': only STATIC ion beams are supported.")
    scan_mode = str(getattr(beam_ds, "ScanMode", "MODULATED") or "MODULATED").upper()
    if scan_mode not in ("MODULATED", "MODULATED_SPEC"):
        raise NotImplementedError(
            f"Beam '{name}': scan mode {scan_mode!r} is not supported (need MODULATED spots)."
        )
    scan_type = str(getattr(beam_ds, "ModulatedScanModeType", "STATIONARY") or "STATIONARY")
    if scan_type.upper() != "STATIONARY":
        raise NotImplementedError(
            f"Beam '{name}': modulated scan mode type {scan_type!r} is not supported."
        )

    for attr, label in (
        ("NumberOfLateralSpreadingDevices", "lateral spreading devices"),
        ("NumberOfRangeModulators", "range modulators"),
        ("NumberOfBlocks", "blocks"),
        ("NumberOfCompensators", "compensators"),
    ):
        if int(getattr(beam_ds, attr, 0) or 0) > 0:
            warnings.warn(f"Beam '{name}': {label} are ignored on import.", stacklevel=5)
    if getattr(beam_ds, "SnoutSequence", None):
        warnings.warn(f"Beam '{name}': the snout is ignored on import.", stacklevel=5)

    if unit not in ("NP", "MU"):
        raise ValueError(f"Beam '{name}': unsupported Primary Dosimeter Unit {unit!r}.")
    if unit == "MU" and particles_per_mu is None:
        raise ValueError(
            f"Beam '{name}' is specified in MU, which needs a particle calibration: pass "
            "`particles_per_mu` (float or callable of the energy) to the import."
        )


def _patient_position(beam_ds, setups: dict[int, str], name: str) -> str:
    number = getattr(beam_ds, "ReferencedPatientSetupNumber", None)
    position = setups.get(int(number)) if number is not None else None
    if not position and len(setups) == 1:
        position = next(iter(setups.values()))
    if not position:
        warnings.warn(f"Beam '{name}': no Patient Position found; assuming HFS.", stacklevel=5)
        position = "HFS"
    position = position.upper()
    if position not in _PATIENT_POSITIONS:
        raise NotImplementedError(
            f"Beam '{name}': patient position {position!r} is not supported "
            f"(supported: {sorted(_PATIENT_POSITIONS)})."
        )
    return position


def _check_static_geometry(cps, name: str) -> tuple[float, float]:
    """Gantry and patient support angle of a static beam; reject unsupported motion."""
    gantry = float(_first(cps, "GantryAngle", 0.0))
    support = float(_first(cps, "PatientSupportAngle", 0.0))
    for cp in cps:
        if "GantryAngle" in cp and abs(float(cp.GantryAngle) - gantry) > 1e-3:
            raise ValueError(f"Beam '{name}': gantry angle changes during the beam (arc).")
        if "PatientSupportAngle" in cp and abs(float(cp.PatientSupportAngle) - support) > 1e-3:
            raise ValueError(f"Beam '{name}': patient support angle changes during the beam.")
        for attr in (
            "TableTopPitchAngle",
            "TableTopRollAngle",
            "GantryPitchAngle",
            "TableTopEccentricAngle",
        ):
            if attr in cp and abs(float(getattr(cp, attr) or 0.0)) > 1e-6:
                raise ValueError(
                    f"Beam '{name}': non-zero {attr} ({getattr(cp, attr)}) is not supported."
                )
    return gantry, support


def _virtual_sad(beam_ds, machine: Machine, name: str) -> float:
    vsad = getattr(beam_ds, "VirtualSourceAxisDistances", None)
    if vsad is None or len(vsad) != 2 or min(float(v) for v in vsad) <= 0:
        return float(machine.sad)
    sad = float(np.mean([float(v) for v in vsad]))
    if abs(float(vsad[0]) - float(vsad[1])) > 1.0:
        warnings.warn(
            f"Beam '{name}': Virtual SAD differs in X ({vsad[0]}) and Y ({vsad[1]}); "
            f"using the mean {sad:.1f} mm.",
            stacklevel=5,
        )
    machine_sad = float(getattr(machine, "sad", sad))
    if abs(sad - machine_sad) > 0.01 * machine_sad:
        logger.warning(
            "Beam '%s': plan Virtual SAD %.1f mm differs from machine SAD %.1f mm; "
            "the plan's value is used for the beam geometry.",
            name,
            sad,
            machine_sad,
        )
    return sad


def _range_shifter_key(shifter_state: dict, name: str) -> tuple[float, float, int]:
    """(water-equivalent thickness, isocenter distance, number) of the inserted shifters.

    Thicknesses add up; distance and number are those of the first inserted shifter.
    """
    thickness, distance, first = 0.0, 0.0, 0
    for number, (setting, iso_dist, wet) in sorted(shifter_state.items()):
        if setting in ("OUT", "0", ""):
            continue
        if np.isnan(wet):
            raise ValueError(
                f"Beam '{name}': range shifter {number} is IN without a Range Shifter "
                "Water Equivalent Thickness."
            )
        if thickness == 0.0:
            distance, first = iso_dist, number
        thickness += wet
    return round(thickness, 6), round(distance, 6), first


def _collect_spots(cps, name: str):
    """Spots (x, y, DICOM energy, meterset weight, range shifter key) and the spot size.

    Spots are taken from every control point carrying positive weights. Zero-weight spots,
    such as those of the closing control point of each pair, are dropped. Energy, spot size
    and range shifter settings are carried forward from the last control point stating them.
    """
    spot_size = None
    energy = None
    shifter_state: dict[int, tuple[str, float, float]] = {}
    spots = []
    for cp in cps:
        if "NominalBeamEnergy" in cp:
            energy = float(cp.NominalBeamEnergy)
        if "ScanningSpotSize" in cp:
            spot_size = _float_list(cp.ScanningSpotSize)
        for setting in getattr(cp, "RangeShifterSettingsSequence", []) or []:
            wet = getattr(setting, "RangeShifterWaterEquivalentThickness", None)
            shifter_state[int(setting.ReferencedRangeShifterNumber)] = (
                str(getattr(setting, "RangeShifterSetting", "OUT")).upper(),
                float(getattr(setting, "IsocenterToRangeShifterDistance", 0.0) or 0.0),
                float("nan") if wet is None else float(wet),
            )

        if "ScanSpotMetersetWeights" not in cp:
            continue
        weights = _float_list(cp.ScanSpotMetersetWeights)
        if not np.any(weights > 0):
            continue
        if "ScanSpotPositionMap" not in cp or energy is None:
            raise ValueError(f"Beam '{name}': control point with weights but no spot map/energy.")
        xy = _float_list(cp.ScanSpotPositionMap, 2).reshape(-1, 2)
        if xy.shape[0] != weights.size:
            raise ValueError(f"Beam '{name}': spot map and weights differ in length.")
        rs_key = _range_shifter_key(shifter_state, name)
        spots.extend(
            (float(x), float(y), energy, float(w), rs_key)
            for (x, y), w in zip(xy, weights)
            if w > 0
        )
    if not spots:
        raise ValueError(f"Beam '{name}': no spots with positive weight.")
    return spots, spot_size


def _import_beam(
    beam_ds,
    *,
    beam_meterset: float,
    setups: dict[int, str],
    machine: Machine,
    radiation_mode: str,
    mass_number: int,
    particles_per_mu: Optional[ParticlesPerMU],
    energy_tolerance: float,
) -> tuple[Beam, list[float]]:
    name = str(getattr(beam_ds, "BeamName", "") or f"beam {beam_ds.BeamNumber}")
    unit = str(getattr(beam_ds, "PrimaryDosimeterUnit", "") or "").upper()
    _check_supported(beam_ds, name, unit, particles_per_mu)
    patient_position = _patient_position(beam_ds, setups, name)

    cps = list(beam_ds.IonControlPointSequence)
    if len(cps) < 2:
        raise ValueError(f"Beam '{name}': needs at least two control points.")
    final_cmw = float(getattr(beam_ds, "FinalCumulativeMetersetWeight", 0.0) or 0.0)
    if final_cmw <= 0.0:
        final_cmw = float(cps[-1].CumulativeMetersetWeight)
    if final_cmw <= 0.0:
        raise ValueError(f"Beam '{name}': Final Cumulative Meterset Weight is not positive.")

    gantry, support = _check_static_geometry(cps, name)
    iso = _float_list(_first(cps, "IsocenterPosition", []))
    if iso.size != 3:
        raise ValueError(f"Beam '{name}': missing Isocenter Position.")
    sad = _virtual_sad(beam_ds, machine, name)

    axes_lps, source_dir = iec_beam_frame(patient_position, gantry, support)
    py_gantry, py_couch = _equivalent_pyradplan_angles(-source_dir)
    r_py = lps.get_beam_rotation_matrix(py_gantry, py_couch)
    if not np.allclose(r_py @ np.array([0.0, 1.0, 0.0]), -source_dir, atol=1e-8):
        raise RuntimeError("Could not express the beam direction by pyRadPlan angles.")

    spots, spot_size = _collect_spots(cps, name)
    energy_map = _map_energies(
        np.array([s[2] for s in spots]), machine, mass_number, energy_tolerance
    )

    # merge spots: a ray is a lateral position, a beamlet an (energy, range shifter) on it
    rays: dict[tuple[float, float], dict] = {}
    for x, y, e_dicom, weight, rs_key in spots:
        ray = rays.setdefault((round(x, 3), round(y, 3)), {"xy": (x, y), "beamlets": {}})
        key = (energy_map[e_dicom], rs_key)
        ray["beamlets"][key] = ray["beamlets"].get(key, 0.0) + weight

    source_bev = np.array([0.0, -sad, 0.0])
    source_world = r_py @ source_bev
    ray_models, beam_weights = [], []
    for ray in rays.values():
        p_world = axes_lps[:, 0] * ray["xy"][0] + axes_lps[:, 1] * ray["xy"][1]
        pos_bev = r_py.T @ p_world
        pos_bev[1] = 0.0
        pos_world = r_py @ pos_bev
        beamlets = []
        for (energy, (thickness, distance, number)), weight in ray["beamlets"].items():
            meterset_value = beam_meterset * weight / final_cmw
            if unit == "NP":
                ppm = _PARTICLES_PER_WEIGHT
                value = meterset_value / _PARTICLES_PER_WEIGHT
            else:
                ppm = _ppm(particles_per_mu, energy)
                value = meterset_value * ppm / _PARTICLES_PER_WEIGHT
            shifter = RangeShifter(
                id=number,
                eq_thickness=thickness,
                source_rashi_distance=(sad - distance) if thickness > 0 else 0.0,
            )
            beamlets.append(
                IonSpot(
                    energy=energy,
                    num_particles_per_mu=ppm,
                    range_shifter=shifter,
                    focus_ix=_focus_index(machine, energy, spot_size),
                )
            )
            beam_weights.append(value)
        ray_models.append(
            Ray(
                beamlets=beamlets,
                ray_pos_bev=pos_bev,
                ray_pos=pos_world,
                target_point_bev=2.0 * (pos_bev - source_bev) + source_bev,
                target_point=2.0 * (pos_world - source_world) + source_world,
            )
        )

    beam = Beam(
        gantry_angle=py_gantry,
        couch_angle=py_couch,
        bixel_width=_lateral_spacing(np.array([r["xy"] for r in rays.values()])),
        radiation_mode=radiation_mode,
        machine=machine.name,
        SAD=sad,
        iso_center=iso,
        ray=ray_models,
        source_point_bev=source_bev,
        source_point=source_world,
    )
    return beam, beam_weights


def _lateral_spacing(positions: np.ndarray, default: float = 5.0) -> float:
    """Median nearest-neighbour distance of the spot positions (mm), ``default`` if unknown."""
    if len(positions) < 2:
        return default
    d = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    nearest = nearest[nearest > 1e-6]
    return float(np.median(nearest)) if nearest.size else default

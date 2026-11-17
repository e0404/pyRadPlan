"""Tests for the DICOM RT Ion Plan import.

The plans are built with pydicom (no RTPLAN test data is bundled). The geometry tests compare
against vectors derived by hand from the IEC 61217 / DICOM definitions (HFS patient, IEC fixed
system X = patient left, Y = towards the head / gantry, Z = up) and do not use pyRadPlan's
rotation helpers.
"""

import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from pyRadPlan.io import load_data, load_patient
from pyRadPlan.io.dicom import DicomImporter
from pyRadPlan.io.dicom._import_plan import (
    RT_ION_PLAN_SOP_CLASS,
    RT_PLAN_SOP_CLASS,
    import_ion_plan,
)
from pyRadPlan.machines import load_from_name

SAD = 2000.0


@pytest.fixture(scope="module")
def generic_proton():
    return load_from_name("protons", "Generic")


def _energy(machine, ix):
    return float(machine.energies[ix])


def make_plan(
    beams,
    unit="NP",
    patient_position="HFS",
    fractions=30,
    radiation="PROTON",
    atomic_number=None,
    sop_class=RT_ION_PLAN_SOP_CLASS,
    pair_control_points=True,
    two_groups=False,
):
    """Build an RT Ion Plan.

    ``beams`` is a list of dicts: ``gantry``, ``couch``, ``iso`` (x, y, z), ``layers`` (a list of
    ``(energy, [(x, y, weight), ...])``), optional ``meterset`` (default 1e9 particles or 100
    MU), ``shifter`` (``(setting, wet, iso_distance)`` applied to all layers), ``pitch``, ``roll``,
    ``spot_size``.
    """
    ds = FileDataset(None, {}, preamble=b"\0" * 128)
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = sop_class
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Modality = "RTPLAN"
    ds.PatientName = "Test^Ion"
    ds.PatientID = "ION1"
    ds.PatientBirthDate = ""
    ds.PatientSex = "O"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.StudyID = "1"
    ds.StudyDate = "20250101"
    ds.StudyTime = "120000"
    ds.AccessionNumber = ""
    ds.ReferringPhysicianName = ""
    ds.OperatorsName = ""
    structure_set = Dataset()
    structure_set.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    structure_set.ReferencedSOPInstanceUID = generate_uid()
    ds.ReferencedStructureSetSequence = Sequence([structure_set])
    ds.SeriesNumber = 1
    ds.InstanceNumber = 1
    ds.Manufacturer = "pyRadPlan"
    ds.FrameOfReferenceUID = generate_uid()
    ds.PositionReferenceIndicator = ""
    ds.RTPlanLabel = "IonPlan"
    ds.RTPlanName = "Ion test plan"
    ds.RTPlanDate = "20250101"
    ds.RTPlanTime = "120000"
    ds.RTPlanGeometry = "PATIENT"

    setup = Dataset()
    setup.PatientSetupNumber = 1
    setup.PatientPosition = patient_position
    ds.PatientSetupSequence = Sequence([setup])

    group = Dataset()
    group.FractionGroupNumber = 1
    group.NumberOfFractionsPlanned = fractions
    group.NumberOfBeams = len(beams)
    group.NumberOfBrachyApplicationSetups = 0
    group.ReferencedBeamSequence = Sequence()
    groups = [group]
    if two_groups:
        other = Dataset()
        other.FractionGroupNumber = 2
        other.NumberOfFractionsPlanned = 3
        other.NumberOfBeams = 0
        other.NumberOfBrachyApplicationSetups = 0
        groups.append(other)
    ds.FractionGroupSequence = Sequence(groups)

    ds.IonBeamSequence = Sequence()
    for number, spec in enumerate(beams, start=1):
        total_weight = sum(w for _, spots in spec["layers"] for _, _, w in spots)
        meterset = spec.get("meterset", 1e9 if unit == "NP" else 100.0)
        ref = Dataset()
        ref.ReferencedBeamNumber = number
        ref.BeamMeterset = meterset
        group.ReferencedBeamSequence.append(ref)

        beam = Dataset()
        beam.Manufacturer = "pyRadPlan"
        beam.TreatmentMachineName = "TestMachine"
        beam.PrimaryDosimeterUnit = unit
        beam.BeamNumber = number
        beam.BeamName = f"B{number}"
        beam.BeamType = "STATIC"
        beam.RadiationType = radiation
        if atomic_number is not None:
            beam.RadiationMassNumber = {2: 4, 6: 12, 8: 16}.get(atomic_number, 20)
            beam.RadiationAtomicNumber = atomic_number
            beam.RadiationChargeState = atomic_number
        beam.TreatmentDeliveryType = "TREATMENT"
        beam.PatientSupportType = "TABLE"
        beam.NumberOfWedges = 0
        beam.NumberOfCompensators = 0
        beam.NumberOfBoli = 0
        beam.NumberOfBlocks = 0
        beam.NumberOfRangeModulators = 0
        beam.NumberOfLateralSpreadingDevices = 0
        beam.ScanMode = "MODULATED"
        beam.VirtualSourceAxisDistances = [SAD, SAD]
        beam.ReferencedPatientSetupNumber = 1
        beam.FinalCumulativeMetersetWeight = total_weight

        shifter = spec.get("shifter")
        beam.NumberOfRangeShifters = 1 if shifter else 0
        if shifter:
            rs = Dataset()
            rs.RangeShifterNumber = 1
            rs.RangeShifterID = "RS1"
            rs.RangeShifterType = "BINARY"
            beam.RangeShifterSequence = Sequence([rs])

        control_points = []
        cumulative = 0.0
        for energy, spots in spec["layers"]:
            weights = [w for _, _, w in spots]
            cps = []
            for second in range(2 if pair_control_points else 1):
                cp = Dataset()
                cp.NominalBeamEnergy = round(float(energy), 5)
                cp.CumulativeMetersetWeight = cumulative + (sum(weights) if second else 0.0)
                cp.ScanSpotTuneID = "1"
                cp.NumberOfScanSpotPositions = len(spots)
                cp.ScanSpotPositionMap = [c for x, y, _ in spots for c in (x, y)]
                cp.ScanSpotMetersetWeights = [0.0] * len(spots) if second else weights
                cp.ScanningSpotSize = spec.get("spot_size", [8.0, 8.0])
                cp.NumberOfPaintings = 1
                cps.append(cp)
            cumulative += sum(weights)
            control_points.extend(cps)
        for i, cp in enumerate(control_points):
            cp.ControlPointIndex = i
            if i == 0:
                cp.GantryAngle = spec.get("gantry", 0.0)
                cp.GantryRotationDirection = "NONE"
                cp.BeamLimitingDeviceAngle = 0.0
                cp.BeamLimitingDeviceRotationDirection = "NONE"
                cp.PatientSupportAngle = spec.get("couch", 0.0)
                cp.PatientSupportRotationDirection = "NONE"
                cp.TableTopPitchAngle = spec.get("pitch", 0.0)
                cp.TableTopPitchRotationDirection = "NONE"
                cp.TableTopRollAngle = spec.get("roll", 0.0)
                cp.TableTopRollRotationDirection = "NONE"
                cp.IsocenterPosition = list(spec.get("iso", (0.0, 0.0, 0.0)))
                cp.GantryPitchAngle = 0.0
                cp.GantryPitchRotationDirection = "NONE"
                if shifter:
                    cp.RangeShifterSettingsSequence = Sequence([_shifter_setting(shifter)])
            elif shifter and spec.get("shifter_out_from") == i:
                out = Dataset()
                out.ReferencedRangeShifterNumber = 1
                out.RangeShifterSetting = "OUT"
                out.IsocenterToRangeShifterDistance = shifter[2]
                cp.RangeShifterSettingsSequence = Sequence([out])
        beam.NumberOfControlPoints = len(control_points)
        beam.IonControlPointSequence = Sequence(control_points)
        ds.IonBeamSequence.append(beam)
    return ds


def _shifter_setting(shifter):
    setting, wet, distance = shifter
    item = Dataset()
    item.ReferencedRangeShifterNumber = 1
    item.RangeShifterSetting = setting
    item.IsocenterToRangeShifterDistance = distance
    item.RangeShifterWaterEquivalentThickness = wet
    return item


def _import(ds, machine=None, **kwargs):
    return import_ion_plan(ds, machine=machine, **kwargs)


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def test_np_weights_and_structure(generic_proton):
    e0, e1 = _energy(generic_proton, 20), _energy(generic_proton, 40)
    ds = make_plan(
        [
            {
                "gantry": 0.0,
                "layers": [
                    (e0, [(0.0, 0.0, 10.0), (5.0, 0.0, 30.0)]),
                    (e1, [(0.0, 0.0, 60.0)]),
                ],
                "meterset": 2e9,
            }
        ],
        fractions=7,
    )
    out = _import(ds)
    stf, pln, w = out["stf"], out["pln"], out["weights"]

    assert out["result"]["w"] is w
    assert stf.num_of_beams == 1
    beam = stf.beams[0]
    # two lateral positions; the one at (0, 0) carries both energy layers
    assert beam.num_of_rays == 2
    assert sorted(beam.num_of_bixels_per_ray.tolist()) == [1, 2]
    assert beam.total_number_of_bixels == 3 == len(w)
    # particles = BeamMeterset * w / FinalCMW (100) -> weight = particles / 1e6
    np.testing.assert_allclose(
        sorted(w), sorted([2e9 * 0.1 / 1e6, 2e9 * 0.3 / 1e6, 2e9 * 0.6 / 1e6])
    )
    assert all(b.num_particles_per_mu == 1e6 for r in beam.rays for b in r.beamlets)
    assert pln.num_of_fractions == 7
    assert pln.dose_convention == "per_fraction"
    assert pln.radiation_mode == "protons"
    assert pln.machine == "Generic"
    assert beam.sad == SAD
    np.testing.assert_allclose(sorted(beam.energies), sorted([e0, e1]))


def test_control_point_pairs_carry_the_weights_once(generic_proton):
    e0 = _energy(generic_proton, 30)
    layers = [(e0, [(0.0, 0.0, 5.0), (4.0, 0.0, 5.0)])]
    out = _import(make_plan([{"layers": layers}], pair_control_points=True))
    # the closing control point of each pair has zero weights and must not add spots
    assert out["stf"].total_number_of_bixels == 2
    np.testing.assert_allclose(out["weights"], [0.5e9 / 1e6] * 2)


def test_identical_spots_are_merged(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan(
        [
            {
                "layers": [
                    (e0, [(1.0, 2.0, 10.0), (1.0, 2.0, 15.0)]),
                    (e0, [(1.0, 2.0, 25.0)]),
                ],
                "meterset": 1e9,
            }
        ]
    )
    out = _import(ds)
    assert out["stf"].total_number_of_bixels == 1
    np.testing.assert_allclose(out["weights"], [1e9 / 1e6])


def test_zero_weight_spots_are_dropped(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 10.0), (9.0, 9.0, 0.0)])]}])
    out = _import(ds)
    assert out["stf"].total_number_of_bixels == 1


def test_mu_needs_calibration(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 10.0)])], "meterset": 50.0}], unit="MU")
    with pytest.raises(ValueError, match="particles_per_mu"):
        _import(ds)

    out = _import(ds, particles_per_mu=2.5e6)
    np.testing.assert_allclose(out["weights"], [50.0])
    assert out["stf"].beams[0].rays[0].beamlets[0].num_particles_per_mu == 2.5e6

    out = _import(ds, particles_per_mu=lambda energy: energy * 1e4)
    assert out["stf"].beams[0].rays[0].beamlets[0].num_particles_per_mu == pytest.approx(e0 * 1e4)


def test_energy_mapping_and_errors(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0 + 0.05, [(0.0, 0.0, 1.0)])]}])
    out = _import(ds)
    assert out["stf"].beams[0].rays[0].beamlets[0].energy == e0

    e1 = _energy(generic_proton, 31)
    mid_a, mid_b = e0 + 0.4 * (e1 - e0), e0 + 0.6 * (e1 - e0)
    ds = make_plan([{"layers": [(mid_a, [(0.0, 0.0, 1.0)]), (mid_b, [(0.0, 0.0, 1.0)])]}])
    with pytest.raises(ValueError, match="energies") as exc:
        _import(ds)
    assert f"{mid_a:.3f}" in str(exc.value) and f"{mid_b:.3f}" in str(exc.value)
    out = _import(ds, energy_tolerance=5.0)
    assert out["stf"].total_number_of_bixels == 2


def test_fraction_groups(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}], fractions=5, two_groups=True)
    with pytest.warns(UserWarning, match="fraction groups"):
        out = _import(ds)
    assert out["pln"].num_of_fractions == 5


@pytest.mark.parametrize(
    "radiation,z,mode,machine_name",
    [
        ("PROTON", None, "protons", "Generic"),
        ("ION", 2, "helium", "Generic"),
        ("ION", 6, "carbon", "Generic"),
    ],
)
def test_radiation_types(radiation, z, mode, machine_name):
    machine = load_from_name(mode, machine_name)
    e0 = float(machine.energies[40])
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}], radiation=radiation, atomic_number=z)
    out = _import(ds, machine=machine)
    assert out["pln"].radiation_mode == mode
    assert out["stf"].beams[0].radiation_mode == mode


def test_ion_total_energy_is_converted_per_nucleon():
    machine = load_from_name("carbon", "Generic")
    per_u = float(machine.energies[40])
    ds = make_plan(
        [{"layers": [(per_u * 12, [(0.0, 0.0, 1.0)])]}], radiation="ION", atomic_number=6
    )
    out = _import(ds, machine=machine)
    assert out["stf"].beams[0].rays[0].beamlets[0].energy == pytest.approx(per_u)


def test_unsupported_ion():
    machine = load_from_name("carbon", "Generic")
    ds = make_plan([{"layers": [(200.0, [(0.0, 0.0, 1.0)])]}], radiation="ION", atomic_number=10)
    with pytest.raises(ValueError, match="Unsupported ion"):
        _import(ds, machine=machine)


def test_focus_follows_spot_size(generic_proton):
    e0 = _energy(generic_proton, 30)
    fwhm = generic_proton.foci[e0][0].fwhm_iso
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])], "spot_size": [fwhm, fwhm]}])
    beamlet = _import(ds)["stf"].beams[0].rays[0].beamlets[0]
    assert beamlet.focus_ix == 0


def test_range_shifter_in_and_out(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan(
        [
            {
                "layers": [
                    (e0, [(0.0, 0.0, 1.0)]),
                    (e0, [(3.0, 0.0, 1.0)]),
                ],
                "shifter": ("IN", 40.0, 300.0),
                "shifter_out_from": 2,
            }
        ]
    )
    beam = _import(ds)["stf"].beams[0]
    by_x = {round(r.ray_pos_bev[0]): r.beamlets[0].range_shifter for r in beam.rays}
    assert by_x[0].eq_thickness == pytest.approx(40.0)
    assert by_x[0].source_rashi_distance == pytest.approx(SAD - 300.0)
    assert by_x[0].id == 1
    assert by_x[3].eq_thickness == 0.0


def test_ignored_devices_warn(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}])
    ds.IonBeamSequence[0].NumberOfLateralSpreadingDevices = 1
    ds.IonBeamSequence[0].NumberOfBlocks = 1
    with pytest.warns(UserWarning) as record:
        _import(ds)
    text = " ".join(str(r.message) for r in record)
    assert "lateral spreading" in text and "blocks" in text


def test_beam_without_meterset_is_skipped(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}, {"layers": [(e0, [(0.0, 0.0, 1.0)])]}])
    ds.FractionGroupSequence[0].ReferencedBeamSequence[1].BeamMeterset = 0.0
    assert _import(ds)["stf"].num_of_beams == 1


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


def test_photon_plan_not_implemented():
    ds = Dataset()
    ds.SOPClassUID = RT_PLAN_SOP_CLASS
    ds.BeamSequence = Sequence([Dataset()])
    with pytest.raises(NotImplementedError, match="photon"):
        import_ion_plan(ds)


@pytest.mark.parametrize("key", ["pitch", "roll"])
def test_table_top_pitch_roll_rejected(generic_proton, key):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])], key: 5.0}])
    with pytest.raises(ValueError, match="Table"):
        _import(ds)


def test_gantry_pitch_rejected(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}])
    ds.IonBeamSequence[0].IonControlPointSequence[0].GantryPitchAngle = 3.0
    with pytest.raises(ValueError, match="GantryPitchAngle"):
        _import(ds)


def test_unsupported_patient_position(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)])]}], patient_position="HFDL")
    with pytest.raises(NotImplementedError, match="HFDL"):
        _import(ds)


def test_arc_rejected(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"layers": [(e0, [(0.0, 0.0, 1.0)]), (e0, [(1.0, 0.0, 1.0)])]}])
    ds.IonBeamSequence[0].IonControlPointSequence[2].GantryAngle = 10.0
    with pytest.raises(ValueError, match="gantry angle changes"):
        _import(ds)


# --------------------------------------------------------------------------
# Geometry (IEC 61217 / DICOM, hand derived for an HFS patient)
# --------------------------------------------------------------------------
#
# HFS: patient left = +x (LPS), posterior = +y, head = +z. IEC fixed: X to the patient's left
# at couch 0, Y towards the head / gantry, Z up (= anterior = -y). Gantry g: source on
# (sin g, 0, cos g) in (X, Y, Z) fixed; the gantry X axis is (cos g, 0, -sin g). A positive
# patient support angle turns the patient counter-clockwise seen from above, i.e. about the
# up axis from X towards Y.

# (gantry, couch) -> (source direction, X_gantry direction, Y_gantry direction) in LPS
_HFS_CASES = {
    (0, 0): ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
    (90, 0): ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    (180, 0): ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
    (270, 0): ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
    # patient turned CCW by 90 deg seen from above: patient left now points to fixed +Y,
    # patient head to fixed -X. The source at fixed +X therefore sits at the patient's feet.
    (90, 90): ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
    (90, -90): ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
    (0, 90): ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
}


@pytest.mark.parametrize("gantry,couch", list(_HFS_CASES))
def test_hfs_geometry(generic_proton, gantry, couch):
    src, xg, yg = (np.array(v, dtype=float) for v in _HFS_CASES[(gantry, couch)])
    e0 = _energy(generic_proton, 30)
    iso = (12.0, -7.0, 3.0)
    ds = make_plan(
        [
            {
                "gantry": float(gantry),
                "couch": float(couch),
                "iso": iso,
                "layers": [(e0, [(10.0, 5.0, 1.0), (0.0, 0.0, 1.0)])],
            }
        ]
    )
    beam = _import(ds)["stf"].beams[0]

    np.testing.assert_allclose(beam.iso_center, iso)
    np.testing.assert_allclose(beam.source_point, SAD * src, atol=1e-6)
    np.testing.assert_allclose(beam.source_point_bev, [0, -SAD, 0])

    for ray in beam.rays:
        x, y = ray.ray_pos_bev[0], ray.ray_pos_bev[2]
        # rays are expressed in the BEV of the stf angles; the world position (relative to the
        # isocenter) is what the IEC gantry system defines
        expected = 10.0 * xg + 5.0 * yg if abs(x) + abs(y) > 0 else np.zeros(3)
        np.testing.assert_allclose(ray.ray_pos, expected, atol=1e-6)
        # the ray goes from the source through the isocenter-plane position and further
        direction = ray.target_point - beam.source_point
        np.testing.assert_allclose(direction, 2.0 * (ray.ray_pos - beam.source_point), atol=1e-6)
        np.testing.assert_allclose(
            np.cross(direction, ray.ray_pos - beam.source_point), 0.0, atol=1e-6
        )
        assert np.dot(direction, -src) > 0  # towards the patient


def test_hfs_stf_angles_equal_dicom_angles(generic_proton):
    e0 = _energy(generic_proton, 30)
    ds = make_plan(
        [
            {"gantry": 270.0, "couch": 30.0, "layers": [(e0, [(0.0, 0.0, 1.0)])]},
            {"gantry": 45.0, "couch": -20.0, "layers": [(e0, [(0.0, 0.0, 1.0)])]},
        ]
    )
    out = _import(ds)
    assert [b.gantry_angle for b in out["stf"].beams] == [270.0, 45.0]
    assert [b.couch_angle for b in out["stf"].beams] == [30.0, -20.0]
    assert out["pln"].prop_stf["gantry_angles"] == [270.0, 45.0]
    assert out["pln"].prop_stf["couch_angles"] == [30.0, -20.0]


@pytest.mark.parametrize("position", ["HFP", "FFS", "FFP"])
@pytest.mark.parametrize("gantry,couch", [(0, 0), (90, 0), (90, 40), (200, -30)])
def test_other_patient_positions_geometry(generic_proton, position, gantry, couch):
    """Source and spot positions follow the patient orientation relative to the IEC axes."""
    # DICOM patient (LPS) axes expressed in the IEC fixed system at couch 0 (hand derived):
    # HFP: prone, head to gantry: left -> -X, head -> +Y, posterior -> +Z (up)
    # FFS: supine, feet to gantry: left -> -X, head -> -Y, posterior -> -Z
    # FFP: prone, feet to gantry: left -> +X, head -> -Y, posterior -> +Z
    patient_axes = {
        "HFP": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        "FFS": ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        "FFP": ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
    }[position]  # images of x_L, y_L, z_L
    to_fixed_0 = np.column_stack([np.array(a, dtype=float) for a in patient_axes])

    def rot_z(deg):
        c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])

    g = np.radians(gantry)
    z_g = np.array([np.sin(g), 0, np.cos(g)])
    x_g = np.array([np.cos(g), 0, -np.sin(g)])
    y_g = np.array([0, 1.0, 0])
    to_lps = (rot_z(couch) @ to_fixed_0).T

    e0 = _energy(generic_proton, 30)
    ds = make_plan(
        [
            {
                "gantry": float(gantry),
                "couch": float(couch),
                "layers": [(e0, [(10.0, -4.0, 1.0)])],
            }
        ],
        patient_position=position,
    )
    beam = _import(ds)["stf"].beams[0]
    np.testing.assert_allclose(beam.source_point, SAD * (to_lps @ z_g), atol=1e-6)
    np.testing.assert_allclose(beam.rays[0].ray_pos, to_lps @ (10.0 * x_g - 4.0 * y_g), atol=1e-6)


def test_bev_position_reproduced_by_stf_rotation(generic_proton):
    """ray_pos == R(stf angles) @ ray_pos_bev, as the dose engines assume."""
    from pyRadPlan.geometry import lps

    e0 = _energy(generic_proton, 30)
    ds = make_plan(
        [
            {
                "gantry": 135.0,
                "couch": 25.0,
                "layers": [(e0, [(10.0, 5.0, 1.0), (-3.0, 8.0, 1.0)])],
            }
        ],
        patient_position="FFS",
    )
    beam = _import(ds)["stf"].beams[0]
    rot = lps.get_beam_rotation_matrix(beam.gantry_angle, beam.couch_angle)
    for ray in beam.rays:
        np.testing.assert_allclose(rot @ ray.ray_pos_bev, ray.ray_pos, atol=1e-8)
        assert ray.ray_pos_bev[1] == 0.0
    np.testing.assert_allclose(rot @ beam.source_point_bev, beam.source_point, atol=1e-8)


# --------------------------------------------------------------------------
# Integration with the importer
# --------------------------------------------------------------------------


@pytest.fixture
def plan_dir(tmp_path, dicom_dir, generic_proton):
    target = tmp_path / "dicom"
    shutil.copytree(dicom_dir, target)
    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"gantry": 90.0, "iso": (0.0, 0.0, 0.0), "layers": [(e0, [(0.0, 0.0, 1.0)])]}])
    ds.save_as(target / "RTPLAN_ion.dcm", enforce_file_format=True)
    return target


def test_list_plans(plan_dir):
    importer = DicomImporter(plan_dir)
    plans = importer.list_plans()
    assert len(plans) == 1
    entry = plans[0]
    assert entry["path"].endswith("RTPLAN_ion.dcm")
    assert entry["label"] == "IonPlan"
    assert entry["name"] == "Ion test plan"
    assert entry["sop_class"] == RT_ION_PLAN_SOP_CLASS
    assert entry["radiation_type"] == "protons"
    assert entry["num_beams"] == 1
    assert entry["num_fractions"] == 30
    assert entry["kind"] == "ion"


def test_load_plan_and_load_data(plan_dir):
    importer = DicomImporter(plan_dir)
    plan = importer.load_plan()
    assert plan["stf"].num_of_beams == 1
    assert plan["pln"].radiation_mode == "protons"
    assert plan["weights"].shape == (1,)

    data = load_data(plan_dir)
    assert {"ct", "cst", "stf", "pln", "result"} <= set(data)
    np.testing.assert_allclose(data["result"]["w"], plan["weights"])


def test_load_data_without_plan_is_unchanged(dicom_dir):
    data = load_data(dicom_dir)
    assert "stf" not in data and "pln" not in data
    assert DicomImporter(dicom_dir).load_plan() is None


def test_load_data_survives_failing_plan(plan_dir):
    for f in plan_dir.glob("RTPLAN_ion.dcm"):
        ds = pydicom.dcmread(f)
        ds.IonBeamSequence[0].PrimaryDosimeterUnit = "MU"
        ds.save_as(f, enforce_file_format=True)
    with pytest.warns(UserWarning, match="Could not import the RT plan"):
        data = load_data(plan_dir)
    assert "ct" in data and "stf" not in data


def test_photon_plan_listed_but_not_loadable(tmp_path, dicom_dir):
    target = tmp_path / "dicom"
    shutil.copytree(dicom_dir, target)
    ds = make_plan([{"layers": [(100.0, [(0.0, 0.0, 1.0)])]}], sop_class=RT_PLAN_SOP_CLASS)
    del ds.IonBeamSequence
    ds.BeamSequence = Sequence([Dataset()])
    ds.save_as(target / "RTPLAN_photon.dcm", enforce_file_format=True)

    importer = DicomImporter(target)
    plans = importer.list_plans()
    assert [p["kind"] for p in plans] == ["photon"]
    assert plans[0]["sop_class"] == RT_PLAN_SOP_CLASS
    with pytest.raises(NotImplementedError, match="photon"):
        importer.load_plan()
    data = load_data(target)
    assert "ct" in data and "stf" not in data


# --------------------------------------------------------------------------
# Physics: forward dose of an imported plan
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def phantom(tg119_path_module):
    ct, cst = load_patient(tg119_path_module)
    return ct, cst


@pytest.fixture(scope="module")
def tg119_path_module():
    from importlib import resources

    return resources.files("pyRadPlan.data.phantoms").joinpath("TG119.mat")


def _centroid(dose, power=1.0):
    import SimpleITK as sitk

    arr = sitk.GetArrayFromImage(dose)  # z, y, x
    arr = np.clip(arr, 0, None) ** power
    total = arr.sum()
    zi, yi, xi = np.indices(arr.shape)
    idx = np.array([(xi * arr).sum(), (yi * arr).sum(), (zi * arr).sum()]) / total
    return np.array(dose.TransformContinuousIndexToPhysicalPoint(tuple(float(i) for i in idx)))


@pytest.mark.parametrize(
    "gantry,beam_dir,x_dir,y_dir",
    [
        (0.0, (0, 1, 0), (1, 0, 0), (0, 0, 1)),
        (90.0, (-1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ],
)
def test_forward_dose_follows_spot_and_beam(
    phantom, generic_proton, gantry, beam_dir, x_dir, y_dir
):
    from pyRadPlan import calc_dose_forward

    ct, cst = phantom
    machine = generic_proton
    ix = int(np.argmin(np.abs(machine.peak_positions - 80.0)))  # Bragg peak at ~80 mm WET
    energy = float(machine.energies[ix])
    iso = np.array([0.0, 0.0, 0.0])
    off_x, off_y = 15.0, 9.0
    ds = make_plan(
        [
            {
                "gantry": gantry,
                "iso": tuple(iso),
                "layers": [(energy, [(off_x, off_y, 1.0)])],
                "meterset": 1e9,
            }
        ]
    )
    out = _import(ds)
    pln = out["pln"]
    pln.prop_dose_calc = {
        "engine": "HongPB",
        "dose_grid": {"resolution": {"x": 4, "y": 4, "z": 4}},
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = calc_dose_forward(ct, cst, out["stf"], pln, out["weights"])
    dose = result["physical_dose"]
    arr = np.asarray(__import__("SimpleITK").GetArrayFromImage(dose))
    assert arr.max() > 0

    centroid = _centroid(dose, power=2.0)
    beam_dir = np.array(beam_dir, dtype=float)
    lateral = off_x * np.array(x_dir, dtype=float) + off_y * np.array(y_dir, dtype=float)
    expected_lateral = lateral - np.dot(lateral, beam_dir) * beam_dir

    rel = centroid - iso
    # lateral position of the dose centroid is the spot offset (coarse: 5 mm)
    rel_lat = rel - np.dot(rel, beam_dir) * beam_dir
    np.testing.assert_allclose(rel_lat, expected_lateral, atol=5.0)

    # the peak lies ~80 mm (water) behind the entry point; the phantom is water-like, the
    # entry at the phantom surface is found along the beam
    peak_index = np.unravel_index(np.argmax(arr), arr.shape)
    peak_point = np.array(
        dose.TransformIndexToPhysicalPoint(tuple(int(i) for i in peak_index[::-1]))
    )
    depth_axis = np.dot(peak_point - iso, beam_dir)
    cube = ct.cube_hu
    center_line = np.array(
        [
            sitk_hu
            for sitk_hu in [
                cube.GetPixel(
                    cube.TransformPhysicalPointToIndex(
                        tuple(float(c) for c in (iso + beam_dir * s + lateral))
                    )
                )
                for s in np.arange(-150, 150, 3.0)
            ]
        ]
    )
    inside = np.where(center_line > -500)[0]
    assert inside.size > 0
    entry = -150 + 3.0 * inside[0]
    assert depth_axis - entry == pytest.approx(80.0, abs=15.0)


# --------------------------------------------------------------------------
# dicom-validator (skipped when the standard is not available)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dicom_standard():
    pytest.importorskip("dicom_validator")
    from dicom_validator.spec_reader.edition_reader import EditionReader  # noqa: PLC0415

    cache = Path(
        os.environ.get(
            "PYRADPLAN_DICOM_STANDARD_DIR", Path.home() / ".cache" / "pyradplan-dicom-standard"
        )
    )
    try:
        cache.mkdir(parents=True, exist_ok=True)
        reader = EditionReader(str(cache))
        edition = reader.get_edition("local") or reader.get_edition("current")
        if edition is None or reader.get_edition_path(edition) is None:
            raise RuntimeError("DICOM standard unavailable")
        return reader.load_dicom_info(edition)
    except Exception as exc:  # noqa: BLE001 - any failure (e.g. offline) must skip
        pytest.skip(f"DICOM standard not available: {exc}")


def test_synthetic_plan_validates(dicom_standard, tmp_path, generic_proton):
    from dicom_validator.validator.dicom_file_validator import (  # noqa: PLC0415
        DicomFileValidator,
    )
    from dicom_validator.validator.error_handler import (  # noqa: PLC0415
        NullValidationResultHandler,
    )

    e0 = _energy(generic_proton, 30)
    ds = make_plan([{"gantry": 90.0, "layers": [(e0, [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0)])]}])
    path = tmp_path / "plan.dcm"
    ds.save_as(path, enforce_file_format=True)
    validator = DicomFileValidator(dicom_standard, error_handler=NullValidationResultHandler())
    errors = []
    for result in validator.validate(str(path)).values():
        for module, tag_errors in result.module_errors.items():
            for tag, err in tag_errors.items():
                if err.is_error():
                    errors.append((module, str(tag), err.code.name))
    assert errors == []

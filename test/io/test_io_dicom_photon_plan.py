"""Tests for the DICOM RT Plan export of photon plans."""

import warnings

import numpy as np
import pydicom
import pymatreader
import pytest
import SimpleITK as sitk

from pyRadPlan import PhotonPlan, calc_dose_forward, generate_stf, load_tg119
from pyRadPlan.geometry import lps
from pyRadPlan.io.dicom import DicomExporter
from pyRadPlan.machines import create_bld
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf
from pyRadPlan.ct import validate_ct

# Same IEC 61217 helpers and dicom-validator setup as the RT Ion Plan export tests
from test_io_dicom_plan_export import (  # noqa: F401 - dicom_standard is a fixture
    DATA,
    _iec_to_lps,
    _validation_errors,
    dicom_standard,
)

RESOLUTION = 1.0
LEAF_WIDTH = 5.0
#: asymmetric leaf openings along the device axis (device coordinates, mm)
LEAF_POSITIONS = [[-10, 10], [-15, 20], [-5, 25], [0, 30], [5, 30], [5, 25], [10, 20], [0, 5]]


def _leaf_boundaries():
    n = len(LEAF_POSITIONS)
    return np.arange(-(n // 2) * LEAF_WIDTH, (n // 2) * LEAF_WIDTH, LEAF_WIDTH)


def _mlc(orientation):
    return create_bld(
        {
            "device_type": "MLC",
            "device_orientation": orientation,
            "leaf_position_boundaries": _leaf_boundaries(),
            "leaf_positions": LEAF_POSITIONS,
            "leaf_width": LEAF_WIDTH,
            "leaf_leakage": 0.0,
        }
    )


def _jaw(orientation, positions):
    return create_bld(
        {
            "device_type": "JAW",
            "device_orientation": orientation,
            "positions": positions,
            "field_width": 120,
        }
    )


def _field_plan(gantry, couch, blds, resolution=RESOLUTION):
    ct, cst = load_tg119()
    pln = PhotonPlan(machine="Generic", num_of_fractions=3)
    pln.prop_stf = {
        "gantry_angles": gantry,
        "couch_angles": couch,
        "generator": "photonSingleBixel",
        "field_based": True,
        "blds": blds,
        "resolution": resolution,
        "energy": 6.0,
    }
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}}
    stf = generate_stf(ct, cst, pln)
    return pln, ct, cst, stf


def _imrt_plan(gantry=(0, 120), bixel_width=20):
    ct, cst = load_tg119()
    pln = PhotonPlan(machine="Generic", num_of_fractions=5)
    pln.prop_stf = {
        "gantry_angles": list(gantry),
        "couch_angles": [0] * len(gantry),
        "bixel_width": bixel_width,
    }
    pln.prop_dose_calc["dose_grid"] = {"resolution": {"x": 6.0, "y": 6.0, "z": 6.0}}
    stf = generate_stf(ct, cst, pln)
    w = np.zeros(stf.total_number_of_bixels)
    w[::3] = 1.0
    return pln, ct, cst, stf, w


def _export(tmp_path, **kwargs):
    out = tmp_path / "out"
    options = kwargs.pop("options", {})
    DicomExporter(out, **options).save(**kwargs)
    return out


@pytest.fixture(scope="module")
def field_case():
    """Two field beams (MLCX with ASYMX/ASYMY jaws) with a forward dose calculation."""
    blds = [_mlc("X"), _jaw("X", [-5, 30]), _jaw("Y", [-20, 12])]
    pln, ct, cst, stf = _field_plan([0, 90], [0, 0], blds)
    result = calc_dose_forward(ct, cst, stf, pln)
    return pln, ct, cst, stf, result


@pytest.fixture(scope="module")
def imrt_case():
    pln, ct, cst, stf, w = _imrt_plan()
    result = calc_dose_forward(ct, cst, stf, pln, w)
    return pln, ct, cst, stf, w, result


# --------------------------------------------------------------------------
# Content
# --------------------------------------------------------------------------


def test_field_plan_content(tmp_path, field_case):
    pln, ct, cst, stf, result = field_case
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")

    assert plan.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.5"
    assert plan.PlanIntent == "RESEARCH"
    assert plan.ApprovalStatus == "UNAPPROVED"
    assert plan.PatientSetupSequence[0].PatientPosition == "HFS"
    assert "IonBeamSequence" not in plan
    assert "RTPlanDescription" not in plan
    group = plan.FractionGroupSequence[0]
    assert group.NumberOfFractionsPlanned == 3
    assert group.NumberOfBeams == 2
    assert [r.ReferencedBeamNumber for r in group.ReferencedBeamSequence] == [1, 2]
    # photon weights have no calibration to MU: no meterset
    assert all("BeamMeterset" not in r for r in group.ReferencedBeamSequence)

    assert [b.BeamNumber for b in plan.BeamSequence] == [1, 2]
    for beam, stf_beam in zip(plan.BeamSequence, stf.beams):
        assert beam.RadiationType == "PHOTON"
        assert beam.BeamType == "STATIC"
        assert beam.PrimaryDosimeterUnit == "MU"
        assert beam.TreatmentMachineName == "Generic"
        assert float(beam.SourceAxisDistance) == pytest.approx(stf_beam.sad)
        assert beam.FinalCumulativeMetersetWeight == 1.0
        assert [d.RTBeamLimitingDeviceType for d in beam.BeamLimitingDeviceSequence] == [
            "ASYMX",
            "ASYMY",
            "MLCX",
        ]
        first, second = beam.ControlPointSequence
        assert float(first.NominalBeamEnergy) == pytest.approx(6.0)
        assert float(first.GantryAngle) == pytest.approx(stf_beam.gantry_angle % 360)
        assert float(first.PatientSupportAngle) == pytest.approx(stf_beam.couch_angle % 360)
        assert float(first.BeamLimitingDeviceAngle) == 0.0
        assert first.CumulativeMetersetWeight == 0.0
        assert second.CumulativeMetersetWeight == 1.0
        np.testing.assert_allclose(
            [float(v) for v in first.IsocenterPosition], stf_beam.iso_center, atol=1e-3
        )
        positions = {
            p.RTBeamLimitingDeviceType: p for p in first.BeamLimitingDevicePositionSequence
        }
        assert set(positions) == {"ASYMX", "ASYMY", "MLCX"}


def test_no_warning_for_field_plan(tmp_path, field_case):
    pln, ct, cst, stf, _ = field_case
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)


def test_jaw_and_leaf_values(tmp_path):
    """Jaw positions in mm at the isocenter plane, leaves consistent with the machine MLC."""
    mlc = _mlc("X")
    blds = [mlc, _jaw("X", [-5, 30]), _jaw("Y", [-20, 12])]
    pln, ct, cst, stf = _field_plan([0], [0], blds)
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    devices = {d.RTBeamLimitingDeviceType: d for d in beam.BeamLimitingDeviceSequence}
    positions = {
        p.RTBeamLimitingDeviceType: np.array([float(v) for v in p.LeafJawPositions])
        for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
    }

    # y is the beam's eye view z axis, x the (mirrored by pyRadPlan, see module) device axis
    np.testing.assert_allclose(positions["ASYMY"], [-20, 12])
    np.testing.assert_allclose(positions["ASYMX"], [-30, 5])
    assert devices["ASYMX"].NumberOfLeafJawPairs == 1

    n = len(LEAF_POSITIONS)
    mlcx = devices["MLCX"]
    assert mlcx.NumberOfLeafJawPairs == n
    np.testing.assert_allclose(
        [float(v) for v in mlcx.LeafPositionBoundaries],
        np.append(mlc.leaf_position_boundaries, mlc.leaf_position_boundaries[-1] + LEAF_WIDTH),
    )
    leaf = np.asarray(LEAF_POSITIONS, dtype=float)
    np.testing.assert_allclose(positions["MLCX"][:n], -leaf[:, 1])
    np.testing.assert_allclose(positions["MLCX"][n:], -leaf[:, 0])


def test_mlcy_boundaries_ascending_and_reordered(tmp_path):
    mlc = _mlc("Y")
    pln, ct, cst, stf = _field_plan([0], [0], [mlc, _jaw("X", [-30, 30])])
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    device = {d.RTBeamLimitingDeviceType: d for d in beam.BeamLimitingDeviceSequence}["MLCY"]
    boundaries = np.array([float(v) for v in device.LeafPositionBoundaries])
    assert np.all(np.diff(boundaries) > 0)
    assert len(boundaries) == len(LEAF_POSITIONS) + 1
    positions = np.array(
        [
            float(v)
            for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
            if p.RTBeamLimitingDeviceType == "MLCY"
            for v in p.LeafJawPositions
        ]
    )
    n = len(LEAF_POSITIONS)
    leaf = np.asarray(LEAF_POSITIONS, dtype=float)
    # leaf order reversed, y positions not mirrored
    np.testing.assert_allclose(positions[:n], leaf[::-1, 0])
    np.testing.assert_allclose(positions[n:], leaf[::-1, 1])


def test_closed_leaf_pair_is_degenerate(tmp_path):
    positions = [[-10, 10], [3, 3], [-5, 5], [0, -4]]
    mlc = create_bld(
        {
            "device_type": "MLC",
            "device_orientation": "X",
            "leaf_position_boundaries": np.arange(-10, 10, 5.0),
            "leaf_positions": positions,
            "leaf_width": 5.0,
        }
    )
    pln, ct, cst, stf = _field_plan([0], [0], [mlc, _jaw("Y", [-10, 10])])
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    pos = {
        p.RTBeamLimitingDeviceType: [float(v) for v in p.LeafJawPositions]
        for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
    }["MLCX"]
    assert pos[1] == pos[5] and pos[3] == pos[7]


# --------------------------------------------------------------------------
# Geometry (world-space aperture), independent of pyRadPlan's rotation code
# --------------------------------------------------------------------------


def _device_aperture(beam, extent=100.0, step=0.5):
    """Raster of the DICOM aperture in IEC GANTRY (x, y) at the isocenter plane."""
    axis = np.arange(-extent, extent + step, step)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")  # rows: y
    open_ = np.ones_like(xx, dtype=bool)
    devices = {d.RTBeamLimitingDeviceType: d for d in beam.BeamLimitingDeviceSequence}
    positions = {
        p.RTBeamLimitingDeviceType: np.array([float(v) for v in p.LeafJawPositions])
        for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
    }
    for kind, pos in positions.items():
        n = int(devices[kind].NumberOfLeafJawPairs)
        low, high = pos[:n], pos[n:]
        if kind in ("ASYMX", "X"):
            open_ &= (xx >= low[0]) & (xx <= high[0])
        elif kind in ("ASYMY", "Y"):
            open_ &= (yy >= low[0]) & (yy <= high[0])
        else:
            edges = np.array([float(v) for v in devices[kind].LeafPositionBoundaries])
            along, across = (xx, yy) if kind == "MLCX" else (yy, xx)
            leaf_open = np.zeros_like(open_)
            for i in range(n):
                in_leaf = (across >= edges[i]) & (across < edges[i + 1])
                leaf_open |= in_leaf & (along >= low[i]) & (along <= high[i])
            open_ &= leaf_open
    return axis, open_


def _centroid(axis, open_):
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.array([xx[open_].mean(), yy[open_].mean()])


def _dicom_world_center(beam):
    first = beam.ControlPointSequence[0]
    axis, open_ = _device_aperture(beam)
    x, y = _centroid(axis, open_)
    return _iec_to_lps(
        x,
        y,
        float(first.GantryAngle),
        float(first.PatientSupportAngle),
        [float(v) for v in first.IsocenterPosition],
    )


def _stf_world_center(stf_beam):
    """Aperture center of pyRadPlan's field shape, mapped with pyRadPlan's own rotation."""
    shape = stf_beam.rays[0].beamlets[0]
    mask = np.asarray(shape.mask)
    grid = np.asarray(shape.grid)
    x = (mask.sum(axis=0) * grid).sum() / mask.sum()
    z = (mask.sum(axis=1) * grid).sum() / mask.sum()
    rotation = lps.get_beam_rotation_matrix(stf_beam.gantry_angle, stf_beam.couch_angle)
    return stf_beam.iso_center + rotation @ np.array([x, 0.0, z])


def _lateral_dose_center(result, stf_beam, depth_window=12.0):
    """Dose-weighted lateral center (world) of the dose within a slab around the iso plane."""
    dose = result["physical_dose"]
    arr = sitk.GetArrayFromImage(dose)
    sp, origin = np.array(dose.GetSpacing()), np.array(dose.GetOrigin())
    k, j, i = np.indices(arr.shape)
    points = np.stack(
        [origin[0] + sp[0] * i, origin[1] + sp[1] * j, origin[2] + sp[2] * k], axis=-1
    ) - np.asarray(stf_beam.iso_center)
    axis = -np.asarray(stf_beam.source_point) / np.linalg.norm(stf_beam.source_point)
    slab = np.abs(points @ axis) <= depth_window
    weight = np.where(slab, arr, 0.0)
    return (
        np.asarray(stf_beam.iso_center)
        + (weight[..., None] * points).sum(axis=(0, 1, 2)) / weight.sum()
    )


def _asymmetric_blds(kind):
    if kind == "jaws":
        return [_jaw("X", [12, 38]), _jaw("Y", [-6, 18])]
    return [_mlc("X"), _jaw("X", [-30, 40]), _jaw("Y", [-15, 25])]


@pytest.mark.parametrize("kind", ["jaws", "mlcx"])
@pytest.mark.parametrize("gantry,couch", [(0, 0), (90, 0), (215, 0), (60, 25), (300, 340)])
def test_aperture_center_matches_stf_geometry(tmp_path, kind, gantry, couch):
    """The DICOM aperture is where pyRadPlan's field shape is (IEC chain written in the test)."""
    pln, ct, cst, stf = _field_plan([gantry], [couch], _asymmetric_blds(kind))
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    np.testing.assert_allclose(
        _dicom_world_center(beam), _stf_world_center(stf.beams[0]), atol=1.0
    )
    # the aperture is not centered: a sign error in x or y would be visible
    assert np.linalg.norm(_stf_world_center(stf.beams[0]) - stf.beams[0].iso_center) > 5


@pytest.mark.parametrize("orientation", ["X", "Y"])
def test_aperture_shape_matches_field_shape(tmp_path, orientation):
    """Leaf by leaf: the rasterized DICOM aperture equals pyRadPlan's field shape mask."""
    blds = [_mlc(orientation), _jaw("X", [-30, 40]), _jaw("Y", [-15, 25])]
    pln, ct, cst, stf = _field_plan([0], [0], blds)
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    axis, open_ = _device_aperture(beam, extent=60.0, step=1.0)

    shape = stf.beams[0].rays[0].beamlets[0]
    mask = np.asarray(shape.mask) > 0.5  # rows: BEV z (= IEC Y_G), columns: BEV x (= IEC X_G)
    grid = np.asarray(shape.grid)
    inside = np.where((axis >= grid[0]) & (axis <= grid[-1]))[0]
    pixels = np.round((axis[inside] - grid[0]) / shape.resolution).astype(int)
    # edges of leaves and jaws (half open pixels) may differ by one pixel
    assert np.mean(open_[np.ix_(inside, inside)] != mask[np.ix_(pixels, pixels)]) < 0.03


@pytest.mark.parametrize("gantry,couch", [(0, 0), (90, 0), (140, 30)])
def test_aperture_center_matches_dose(tmp_path, gantry, couch):
    """Settles the x axis of the beam limiting devices: the dose is where the DICOM aperture is."""
    pln, ct, cst, stf = _field_plan([gantry], [couch], _asymmetric_blds("jaws"), resolution=2.0)
    result = calc_dose_forward(ct, cst, stf, pln)
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    expected = _dicom_world_center(beam)
    measured = _lateral_dose_center(result, stf.beams[0])
    iso = np.asarray(stf.beams[0].iso_center)
    # the center is 25 mm off axis in x and 6 mm in y: a mirrored x would be 50 mm away
    assert np.linalg.norm(expected - iso) > 20
    np.testing.assert_allclose(measured, expected, atol=6.0)


# --------------------------------------------------------------------------
# Bixel-based (IMRT) plans: geometry only
# --------------------------------------------------------------------------


def test_imrt_plan_warns_and_exports_geometry(tmp_path, imrt_case):
    pln, ct, cst, stf, w, result = imrt_case
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert "NOT deliverable" in plan.RTPlanDescription
    assert len(plan.BeamSequence) == 2
    for index, (beam, stf_beam) in enumerate(zip(plan.BeamSequence, stf.beams)):
        assert "NOT deliverable" in beam.BeamDescription
        assert [d.RTBeamLimitingDeviceType for d in beam.BeamLimitingDeviceSequence] == [
            "ASYMX",
            "ASYMY",
        ]
        # jaws open a field around the positively weighted bixels (no mirroring: BEV)
        offset = sum(b.total_number_of_bixels for b in stf.beams[:index])
        active = [
            ray.ray_pos_bev
            for k, ray in enumerate(stf_beam.rays)
            if w[offset + k] > 0  # one bixel per ray
        ]
        half = stf_beam.bixel_width / 2
        positions = {
            p.RTBeamLimitingDeviceType: [float(v) for v in p.LeafJawPositions]
            for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
        }
        np.testing.assert_allclose(
            positions["ASYMX"],
            [min(a[0] for a in active) - half, max(a[0] for a in active) + half],
        )
        np.testing.assert_allclose(
            positions["ASYMY"],
            [min(a[2] for a in active) - half, max(a[2] for a in active) + half],
        )


def test_imrt_plan_without_weights(tmp_path):
    pln, ct, cst, stf, _ = _imrt_plan()
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert [b.BeamNumber for b in plan.BeamSequence] == [1, 2]
    assert all(
        "BeamMeterset" not in r for r in plan.FractionGroupSequence[0].ReferencedBeamSequence
    )


def test_beam_without_weight_is_skipped(tmp_path, imrt_case):
    pln, ct, cst, stf, w, result = imrt_case
    w = w.copy()
    w[: stf.beams[0].total_number_of_bixels] = 0.0
    with pytest.warns(UserWarning, match="Beam 1 has no positive weight"):
        out = _export(
            tmp_path,
            ct=ct,
            cst=cst,
            stf=stf,
            pln=pln,
            result={**result, "w": w},
            options={"per_beam_doses": True},
        )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert [b.BeamNumber for b in plan.BeamSequence] == [2]
    assert sorted(p.name for p in out.glob("RTDOSE_physical_dose_beam*.dcm")) == [
        "RTDOSE_physical_dose_beam2.dcm"
    ]


def test_single_collimated_square_field_is_deliverable(tmp_path):
    ct, cst = load_tg119()
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "generator": "photonOpenFields",
        "gantry_angles": [0, 100],
        "couch_angles": [0, 10],
        "bixel_width": 40,
    }
    stf = generate_stf(ct, cst, pln)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # one planned fraction
        out = _export(tmp_path, ct=ct, cst=cst, stf=stf, weights=[1.0, 2.0])
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert "RTPlanDescription" not in plan
    for beam in plan.BeamSequence:
        assert "collimated square field" in beam.BeamDescription
        positions = {
            p.RTBeamLimitingDeviceType: [float(v) for v in p.LeafJawPositions]
            for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
        }
        assert positions["ASYMX"] == [-20.0, 20.0]
        assert positions["ASYMY"] == [-20.0, 20.0]


def test_mask_field_shape_is_bounding_box_and_not_deliverable(tmp_path):
    mask = np.zeros((41, 41), dtype=np.float32)
    mask[20:31, 25:36] = 1.0  # rows: z, columns: x (resolution 2 mm, grid -40..40)
    ct, cst = load_tg119()
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "generator": "photonSingleBixel",
        "field_based": True,
        "field_shape": mask,
        "resolution": 2.0,
        "energy": 6.0,
    }
    stf = generate_stf(ct, cst, pln)
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    beam = pydicom.dcmread(out / "RTPLAN.dcm").BeamSequence[0]
    positions = {
        p.RTBeamLimitingDeviceType: [float(v) for v in p.LeafJawPositions]
        for p in beam.ControlPointSequence[0].BeamLimitingDevicePositionSequence
    }
    # pixel centers 10..30 (x) and 0..20 (z), half a pixel margin
    assert positions["ASYMX"] == [9.0, 31.0]
    assert positions["ASYMY"] == [-1.0, 21.0]


# --------------------------------------------------------------------------
# Standard conformance and linking
# --------------------------------------------------------------------------


def test_field_plan_validates(tmp_path, dicom_standard, field_case):
    pln, ct, cst, stf, result = field_case
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln, result=result)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []
    for path in out.glob("RTDOSE*.dcm"):
        assert _validation_errors(dicom_standard, path) == [], path.name


def test_mlcy_plan_validates(tmp_path, dicom_standard):
    pln, ct, cst, stf = _field_plan([30], [0], [_mlc("Y"), _jaw("X", [-30, 30])])
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []


def test_imrt_plan_validates(tmp_path, dicom_standard, imrt_case):
    pln, ct, cst, stf, w, result = imrt_case
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(
            tmp_path,
            ct=ct,
            cst=cst,
            stf=stf,
            pln=pln,
            result={**result, "w": w},
            options={"per_beam_doses": True},
        )
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []
    dose_files = sorted(out.glob("RTDOSE*.dcm"))
    assert len(dose_files) == 1 + len(stf.beams)
    for path in dose_files:
        assert _validation_errors(dicom_standard, path) == [], path.name


def test_linking_and_plan_references(tmp_path, field_case):
    pln, ct, cst, stf, result = field_case
    out = _export(
        tmp_path, ct=ct, cst=cst, stf=stf, pln=pln, result=result, options={"per_beam_doses": True}
    )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    struct = pydicom.dcmread(out / "RTSTRUCT.dcm")
    ct_file = pydicom.dcmread(sorted(out.glob("CT_*.dcm"))[0])
    dose = pydicom.dcmread(out / "RTDOSE_physical_dose.dcm")

    objects = [plan, struct, ct_file, dose]
    assert len({o.StudyInstanceUID for o in objects}) == 1
    assert len({o.FrameOfReferenceUID for o in objects}) == 1
    assert len({o.PatientID for o in objects}) == 1
    assert plan.RTPlanGeometry == "PATIENT"
    ref = plan.ReferencedStructureSetSequence[0]
    assert ref.ReferencedSOPInstanceUID == struct.SOPInstanceUID

    beam_files = sorted(out.glob("RTDOSE_physical_dose_beam*.dcm"))
    assert len(beam_files) == len(stf.beams)
    for path in [out / "RTDOSE_physical_dose.dcm", *beam_files]:
        ds = pydicom.dcmread(path)
        item = ds.ReferencedRTPlanSequence[0]
        assert item.ReferencedSOPInstanceUID == plan.SOPInstanceUID
        assert item.ReferencedSOPClassUID == "1.2.840.10008.5.1.4.1.1.481.5"
    ds = pydicom.dcmread(beam_files[1])
    group = ds.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0]
    assert group.ReferencedBeamSequence[0].ReferencedBeamNumber == 2


def test_plan_without_structure_set_is_treatment_device_geometry(tmp_path, field_case):
    pln, ct, cst, stf, _ = field_case
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert plan.RTPlanGeometry == "TREATMENT_DEVICE"
    assert "ReferencedStructureSetSequence" not in plan


def test_export_plan_can_be_disabled(tmp_path, field_case):
    pln, ct, cst, stf, _ = field_case
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, options={"export_plan": False})
    assert not (out / "RTPLAN.dcm").exists()


def test_wrong_number_of_weights(tmp_path, imrt_case):
    pln, ct, cst, stf, w, _ = imrt_case
    with pytest.raises(ValueError, match="Number of weights"):
        _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w[:-1])


def test_unsupported_patient_position(tmp_path, field_case):
    pln, ct, cst, stf, _ = field_case
    with pytest.raises(NotImplementedError, match="Head First Supine"):
        _export(
            tmp_path, ct=ct, stf=stf, pln=pln, options={"metadata": {"patient_position": "FFS"}}
        )


def test_photon_matrad_plan_exports_with_geometry_only_warning(tmp_path):
    """A matRad photon IMRT stf (from the test data) with its weights."""
    raw = pymatreader.read_mat(str(DATA / "photons_testData.mat"))
    pln = validate_pln(raw["pln"])
    ct = validate_ct(raw["ct"])
    stf = validate_stf(raw["stf"])
    w = np.ones(stf.total_number_of_bixels)
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert len(plan.BeamSequence) == len(stf.beams)

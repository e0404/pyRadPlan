"""Tests for the DICOM RT Ion Plan export."""

import os
from pathlib import Path

import numpy as np
import pydicom
import pymatreader
import pytest
import SimpleITK as sitk

from pyRadPlan import IonPlan, generate_stf, load_tg119
from pyRadPlan.cst import validate_cst
from pyRadPlan.ct import validate_ct
from pyRadPlan.dose import calc_dose_forward
from pyRadPlan.io import save_data
from pyRadPlan.io.dicom import DicomExporter
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf

DATA = Path(__file__).parent.parent / "data"

# --------------------------------------------------------------------------
# dicom-validator (same approach as test_io_dicom_export.py)
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


def _validation_errors(info, path):
    from dicom_validator.validator.dicom_file_validator import (  # noqa: PLC0415
        DicomFileValidator,
    )
    from dicom_validator.validator.error_handler import (  # noqa: PLC0415
        NullValidationResultHandler,
    )

    validator = DicomFileValidator(info, error_handler=NullValidationResultHandler())
    errors = []
    for result in validator.validate(path).values():
        for module, tag_errors in result.module_errors.items():
            for tag, err in tag_errors.items():
                if err.is_error():
                    errors.append((os.path.basename(path), module, str(tag), err.code.name))
    return errors


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def _load_matrad(name):
    raw = pymatreader.read_mat(str(DATA / name))
    pln = validate_pln(raw["pln"])
    ct = validate_ct(raw["ct"])
    cst = validate_cst(raw["cst"], ct=ct)
    stf = validate_stf(raw["stf"])
    w = np.atleast_1d(np.asarray(raw["resultGUI"]["w"], dtype=float))
    return pln, ct, cst, stf, w


@pytest.fixture(scope="module")
def proton_mat():
    return _load_matrad("protons_testData.mat")


@pytest.fixture(scope="module")
def carbon_mat():
    return _load_matrad("carbon_testData.mat")


def _generated_plan(gantry, couch, bixel_width=15):
    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=5)
    pln.prop_stf = {"gantry_angles": gantry, "couch_angles": couch, "bixel_width": bixel_width}
    stf = generate_stf(ct, cst, pln)
    rng = np.random.default_rng(1)
    w = rng.random(stf.total_number_of_bixels) * (rng.random(stf.total_number_of_bixels) > 0.3)
    return pln, ct, cst, stf, w


@pytest.fixture(scope="module")
def generated():
    """Two-beam proton plan on the TG119 phantom with a forward dose calculation."""
    pln, ct, cst, stf, w = _generated_plan([0, 90], [0, 0], bixel_width=20)
    result = calc_dose_forward(ct, cst, stf, pln, w)
    assert "physical_dose" in result.beam_quantities
    return pln, ct, cst, stf, w, result


@pytest.fixture(scope="module")
def many_angles():
    return _generated_plan([0, 90, 180, 270, 90, 0], [0, 0, 0, 0, 330, 45])


def _export(tmp_path, **kwargs):
    out = tmp_path / "out"
    options = kwargs.pop("options", {})
    DicomExporter(out, **options).save(**kwargs)
    return out


def _spots(stf, w):
    """Nonzero stf spots as {beam: {(energy, x, y, z) rounded: particles}} (world, rel. iso)."""
    out = []
    index = 0
    for beam in stf.beams:
        spots = {}
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                if w[index] > 0:
                    point = _isocenter_plane_point(beam, ray)
                    key = (round(beamlet.energy, 2), *np.round(point, 3))
                    spots[key] = spots.get(key, 0.0) + w[index] * 1e6
                index += 1
        out.append(spots)
    return out


# --------------------------------------------------------------------------
# IEC 61217 geometry for Head First Supine, implemented from the DICOM standard
# --------------------------------------------------------------------------


def _iec_to_lps(x, y, gantry_deg, couch_deg, iso):
    """World (LPS) point of the machine isocenter plane position (x, y) in IEC GANTRY.

    IEC FIXED: Y towards the gantry bearing, Z up, X = Y x Z. IEC GANTRY is rotated about Y
    such that the source is at +Z_FIXED for gantry 0 and at +X_FIXED for gantry 90. The
    patient support rotates the patient about Z_FIXED, counter-clockwise seen from above.
    For HFS the IEC PATIENT system equals IEC FIXED at support angle 0 and
    (X_DICOM, Y_DICOM, Z_DICOM) = (X_IEC, -Z_IEC, Y_IEC).
    """
    g, c = np.radians(gantry_deg), np.radians(couch_deg)
    z_gantry = np.array([np.sin(g), 0.0, np.cos(g)])
    y_gantry = np.array([0.0, 1.0, 0.0])
    x_gantry = np.cross(y_gantry, z_gantry)
    fixed = x * x_gantry + y * y_gantry
    # patient = Rz(-couch) fixed, as the support turns the patient by +couch
    rz_inverse = np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
    patient_iec = rz_inverse @ fixed
    lps = np.array([patient_iec[0], -patient_iec[2], patient_iec[1]])
    return np.asarray(iso, dtype=float) + lps


def _iec_axis_offset(gantry_deg, couch_deg, sad):
    """Source position relative to the isocenter (SAD along +Z_GANTRY) in LPS."""
    g, c = np.radians(gantry_deg), np.radians(couch_deg)
    fixed = sad * np.array([np.sin(g), 0.0, np.cos(g)])
    rz_inverse = np.array([[np.cos(c), np.sin(c), 0], [-np.sin(c), np.cos(c), 0], [0, 0, 1]])
    p = rz_inverse @ fixed
    return np.array([p[0], -p[2], p[1]])


def _isocenter_plane_point(beam, ray):
    """Point (relative to the isocenter) where the ray crosses the isocenter plane.

    Uses only the source point and the ray's target point of the stf.
    """
    source = np.asarray(beam.source_point, dtype=float)
    target = np.asarray(ray.target_point, dtype=float)
    normal = source / np.linalg.norm(source)
    t = -(source @ normal) / ((target - source) @ normal)
    return source + t * (target - source)


@pytest.mark.parametrize(
    "gantry,couch",
    [(0, 0), (90, 0), (180, 0), (270, 0), (0, 45), (90, 330), (270, 30), (123, 17)],
)
def test_pyradplan_source_matches_iec(gantry, couch):
    """The stf's source point is the IEC source position (no conversion needed for HFS)."""
    ct, cst = load_tg119()
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {"gantry_angles": [gantry], "couch_angles": [couch], "bixel_width": 40}
    beam = generate_stf(ct, cst, pln).beams[0]
    np.testing.assert_allclose(
        beam.source_point, _iec_axis_offset(gantry, couch, beam.sad), atol=1e-6
    )


def test_standard_anchors():
    """Head First Supine anchors of the standard: gantry 0 above, gantry 90 at the left."""
    anchors = {
        0: (0, -1, 0),  # source anterior (-y LPS), beam travelling posterior
        90: (1, 0, 0),  # source at the patient's left (+x LPS)
        180: (0, 1, 0),
        270: (-1, 0, 0),
    }
    for gantry, direction in anchors.items():
        np.testing.assert_allclose(
            _iec_axis_offset(gantry, 0, 1.0), direction, atol=1e-12, err_msg=str(gantry)
        )


def test_spot_positions_follow_iec_geometry(tmp_path, many_angles):
    pln, ct, cst, stf, w = many_angles
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    expected = _spots(stf, w)
    assert len(plan.IonBeamSequence) == len(stf.beams)

    for beam_ds, beam, expect in zip(plan.IonBeamSequence, stf.beams, expected):
        first = beam_ds.IonControlPointSequence[0]
        gantry, couch = float(first.GantryAngle), float(first.PatientSupportAngle)
        assert gantry == pytest.approx(beam.gantry_angle % 360)
        assert couch == pytest.approx(beam.couch_angle % 360)
        np.testing.assert_allclose(first.IsocenterPosition, beam.iso_center, atol=1e-6)
        found = {}
        for cp in beam_ds.IonControlPointSequence:
            weights = np.asarray(cp.ScanSpotMetersetWeights, dtype=float)
            if not weights.any():
                continue
            xy = np.asarray(cp.ScanSpotPositionMap, dtype=float).reshape(-1, 2)
            assert len(xy) == cp.NumberOfScanSpotPositions == len(weights)
            for (x, y), particles in zip(xy, weights):
                point = _iec_to_lps(x, y, gantry, couch, iso=np.zeros(3))
                key = (round(float(cp.NominalBeamEnergy), 2), *np.round(point, 3))
                found[key] = found.get(key, 0.0) + particles
        assert sorted(found) == sorted(expect)
        for key, particles in expect.items():
            assert found[key] == pytest.approx(particles, rel=1e-6)


def test_gantry_angles_cover_all_quadrants(many_angles):
    stf = many_angles[3]
    assert {b.gantry_angle for b in stf.beams} >= {0, 90, 180, 270}
    assert any(b.couch_angle != 0 for b in stf.beams)


# --------------------------------------------------------------------------
# Weights, energies, fractions
# --------------------------------------------------------------------------


def _check_weights(plan, stf, w, fractions):
    index = 0
    assert len(plan.IonBeamSequence) == len(stf.beams)
    group = plan.FractionGroupSequence[0]
    assert group.NumberOfFractionsPlanned == fractions
    assert group.NumberOfBeams == len(stf.beams)
    for number, (beam_ds, beam) in enumerate(zip(plan.IonBeamSequence, stf.beams), start=1):
        n = beam.total_number_of_bixels
        beam_w = w[index : index + n]
        energies = np.array([b.energy for r in beam.rays for b in r.beamlets])
        index += n
        nonzero = beam_w > 0
        expected = float(np.sum(beam_w[nonzero])) * 1e6

        assert beam_ds.BeamNumber == number
        assert beam_ds.PrimaryDosimeterUnit == "NP"
        assert beam_ds.FinalCumulativeMetersetWeight == pytest.approx(expected, rel=1e-6)
        ref = next(r for r in group.ReferencedBeamSequence if r.ReferencedBeamNumber == number)
        assert float(ref.BeamMeterset) == pytest.approx(expected, rel=1e-6)

        cps = beam_ds.IonControlPointSequence
        assert beam_ds.NumberOfControlPoints == len(cps) == 2 * (len(cps) // 2)
        assert cps[-1].CumulativeMetersetWeight == pytest.approx(expected, rel=1e-6)
        sums = [float(np.sum(cp.ScanSpotMetersetWeights)) for cp in cps]
        assert sum(sums) == pytest.approx(expected, rel=1e-6)
        assert sum(cp.NumberOfScanSpotPositions for cp in cps[::2]) == int(nonzero.sum())
        for cp in cps[1::2]:
            assert not np.any(cp.ScanSpotMetersetWeights)
        layer_energies = sorted(float(cp.NominalBeamEnergy) for cp in cps[::2])
        np.testing.assert_allclose(
            layer_energies, sorted(set(np.round(energies[nonzero], 6))), atol=1e-4
        )
        for k in range(0, len(cps), 2):
            assert cps[k].NominalBeamEnergy == cps[k + 1].NominalBeamEnergy
            assert list(cps[k].ScanSpotPositionMap) == list(cps[k + 1].ScanSpotPositionMap)
            assert float(cps[k + 1].CumulativeMetersetWeight) == pytest.approx(
                sum(sums[: k + 2]), rel=1e-6
            )


def test_weights_proton_reference(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, result={"w": w})
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    _check_weights(plan, stf, w, pln.num_of_fractions)
    assert plan.IonBeamSequence[0].RadiationType == "PROTON"
    assert plan.PlanIntent == "RESEARCH"
    assert plan.ApprovalStatus == "UNAPPROVED"
    assert plan.RTPlanGeometry == "TREATMENT_DEVICE"  # no RTSTRUCT exported


def test_weights_generated_with_zero_spots(tmp_path, many_angles):
    pln, ct, cst, stf, w = many_angles
    assert (w == 0).any()
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    _check_weights(pydicom.dcmread(out / "RTPLAN.dcm"), stf, w, 5)


def test_weights_from_result_and_explicit_argument(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    a = _export(tmp_path / "a", ct=ct, stf=stf, pln=pln, weights=w)
    b = _export(tmp_path / "b", ct=ct, stf=stf, pln=pln, result={"w": w})
    for out in (a, b):
        _check_weights(pydicom.dcmread(out / "RTPLAN.dcm"), stf, w, pln.num_of_fractions)


def test_ion_radiation_type(tmp_path, carbon_mat):
    pln, ct, cst, stf, w = carbon_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    beam = plan.IonBeamSequence[0]
    assert beam.RadiationType == "ION"
    assert (beam.RadiationMassNumber, beam.RadiationAtomicNumber) == (12, 6)
    assert beam.RadiationChargeState == 6
    _check_weights(plan, stf, w, pln.num_of_fractions)


def test_spot_size_and_machine_attributes(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    plan = pydicom.dcmread(_export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w) / "RTPLAN.dcm")
    beam = plan.IonBeamSequence[0]
    assert beam.TreatmentMachineName == "Generic"
    assert float(beam.VirtualSourceAxisDistances[0]) == pytest.approx(stf.beams[0].sad)
    assert beam.ScanMode == "MODULATED"
    assert beam.NumberOfRangeShifters == 0
    assert all(float(s) > 0 for cp in beam.IonControlPointSequence for s in cp.ScanningSpotSize)


def test_range_shifter(tmp_path, dicom_standard):
    from pyRadPlan.stf._rangeshifter import RangeShifter

    pln, ct, cst, stf, w = _generated_plan([0, 90], [0, 0], 20)
    stf = stf.model_copy(deep=True)
    for ray in stf.beams[0].rays:
        for index, beamlet in enumerate(ray.beamlets):
            if index % 2:
                beamlet.range_shifter = RangeShifter(
                    id=1, eq_thickness=30.0, source_rashi_distance=9500.0
                )
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    first, second = plan.IonBeamSequence
    assert second.NumberOfRangeShifters == 0 and "RangeShifterSequence" not in second
    assert first.NumberOfRangeShifters == 1
    assert first.RangeShifterSequence[0].RangeShifterNumber == 1
    settings = {
        cp.RangeShifterSettingsSequence[0].RangeShifterSetting
        for cp in first.IonControlPointSequence
    }
    assert settings <= {"IN", "OUT"}
    item = first.IonControlPointSequence[0].RangeShifterSettingsSequence[0]
    assert float(item.IsocenterToRangeShifterDistance) == pytest.approx(500.0)
    assert float(item.RangeShifterWaterEquivalentThickness) == pytest.approx(30.0)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_rtplan_validates_proton_reference(tmp_path, dicom_standard, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln, weights=w)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []


def test_rtplan_validates_generated(tmp_path, dicom_standard, many_angles):
    pln, ct, cst, stf, w = many_angles
    out = _export(tmp_path, ct=ct, cst=cst, stf=stf, pln=pln, weights=w)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []


def test_rtplan_validates_carbon(tmp_path, dicom_standard, carbon_mat):
    pln, ct, cst, stf, w = carbon_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    assert _validation_errors(dicom_standard, out / "RTPLAN.dcm") == []


@pytest.mark.parametrize("convention", ["per_fraction", "total"])
def test_rtdose_validates_and_references_plan(tmp_path, dicom_standard, generated, convention):
    pln, ct, cst, stf, w, result = generated
    pln = pln.model_copy(update={"dose_convention": convention})
    out = _export(
        tmp_path,
        ct=ct,
        cst=cst,
        stf=stf,
        pln=pln,
        result=result.model_copy(update={"fluence": np.asarray(w)}),
        options={"per_beam_doses": True},
    )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    dose_files = sorted(out.glob("RTDOSE*.dcm"))
    assert len(dose_files) == 2 + 2 * len(stf.beams)
    for path in dose_files:
        assert _validation_errors(dicom_standard, path) == [], path.name
        ds = pydicom.dcmread(path)
        item = ds.ReferencedRTPlanSequence[0]
        assert item.ReferencedSOPInstanceUID == plan.SOPInstanceUID
        assert item.ReferencedSOPClassUID == plan.SOPClassUID
        if "_beam" in path.name:
            assert ds.DoseSummationType == "BEAM"
            number = int(path.stem.rsplit("beam", 1)[1])
            group = item.ReferencedFractionGroupSequence[0]
            assert group.ReferencedBeamSequence[0].ReferencedBeamNumber == number
            assert number in [b.BeamNumber for b in plan.IonBeamSequence]
        elif convention == "per_fraction":
            assert ds.DoseSummationType == "FRACTION"
            assert item.ReferencedFractionGroupSequence[0].ReferencedFractionGroupNumber == 1
        else:
            assert ds.DoseSummationType == "PLAN"


def test_rtdose_image_validates(tmp_path, dicom_standard, generated):
    pln, ct, cst, stf, w, result = generated
    out = _export(
        tmp_path, ct=ct, dose=result["physical_dose"].values, stf=stf, pln=pln, weights=w
    )
    assert _validation_errors(dicom_standard, out / "RTDOSE.dcm") == []
    ref = pydicom.dcmread(out / "RTDOSE.dcm").ReferencedRTPlanSequence[0]
    assert ref.ReferencedSOPInstanceUID == pydicom.dcmread(out / "RTPLAN.dcm").SOPInstanceUID


# --------------------------------------------------------------------------
# Linking
# --------------------------------------------------------------------------


@pytest.mark.parametrize("anonymize", [False, True])
def test_linking(tmp_path, generated, anonymize):
    pln, ct, cst, stf, w, result = generated
    out = _export(
        tmp_path,
        ct=ct,
        cst=cst,
        stf=stf,
        pln=pln,
        result=result.model_copy(update={"fluence": np.asarray(w)}),
        options={"anonymize": anonymize},
    )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    struct = pydicom.dcmread(out / "RTSTRUCT.dcm")
    dose = pydicom.dcmread(out / "RTDOSE_physical_dose.dcm")
    ct_file = pydicom.dcmread(sorted(out.glob("CT_*.dcm"))[0])

    objects = [plan, struct, dose, ct_file]
    assert len({o.StudyInstanceUID for o in objects}) == 1
    assert len({o.FrameOfReferenceUID for o in objects}) == 1
    assert len({o.PatientID for o in objects}) == 1
    assert len({o.SeriesInstanceUID for o in objects}) == 4
    assert plan.Modality == "RTPLAN"
    assert plan.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.8"
    assert plan.file_meta.MediaStorageSOPInstanceUID == plan.SOPInstanceUID

    ref = plan.ReferencedStructureSetSequence[0]
    assert ref.ReferencedSOPInstanceUID == struct.SOPInstanceUID
    assert ref.ReferencedSOPClassUID == struct.SOPClassUID
    assert dose.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == plan.SOPInstanceUID
    assert plan.PatientSetupSequence[0].PatientPosition == "HFS"
    if anonymize:
        assert plan.PatientName == "Anonymous^Patient"


def test_plan_without_structure_set_has_no_reference(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    assert "ReferencedStructureSetSequence" not in pydicom.dcmread(out / "RTPLAN.dcm")


# --------------------------------------------------------------------------
# Behaviour
# --------------------------------------------------------------------------


def test_save_data_exports_plan(tmp_path, generated):
    pln, ct, cst, stf, w, result = generated
    target = tmp_path / "dicom"
    save_data(
        file_name=target,
        ct=ct,
        cst=cst,
        stf=stf,
        pln=pln,
        result=result.model_copy(update={"fluence": np.asarray(w)}),
        format="dcm",
    )
    assert (target / "RTPLAN.dcm").exists()
    assert (target / "RTDOSE_physical_dose.dcm").exists()


def test_export_plan_can_be_disabled(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w, options={"export_plan": False})
    assert not (out / "RTPLAN.dcm").exists()


def test_no_weights_no_plan(tmp_path, proton_mat):
    pln, ct, cst, stf, _ = proton_mat
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln)
    assert not (out / "RTPLAN.dcm").exists()


def test_plan_without_pln_warns_and_uses_one_fraction(tmp_path, proton_mat):
    _, ct, cst, stf, w = proton_mat
    with pytest.warns(UserWarning, match="one planned fraction"):
        out = _export(tmp_path, ct=ct, stf=stf, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert plan.FractionGroupSequence[0].NumberOfFractionsPlanned == 1


def test_wrong_number_of_weights(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    with pytest.raises(ValueError, match="does not match"):
        _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w[:-1])


def test_unsupported_patient_position(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    with pytest.raises(NotImplementedError, match="Head First Supine"):
        _export(
            tmp_path,
            ct=ct,
            stf=stf,
            pln=pln,
            weights=w,
            options={"metadata": {"patient_position": "FFS"}},
        )


def test_beam_without_weight_is_skipped(tmp_path, generated):
    pln, ct, cst, stf, w, result = generated
    n0 = stf.beams[0].total_number_of_bixels
    w2 = w.copy()
    w2[:n0] = 0.0
    with pytest.warns(UserWarning, match="Beam 1 has no spot"):
        out = _export(
            tmp_path,
            ct=ct,
            stf=stf,
            pln=pln,
            result=result.model_copy(update={"fluence": np.asarray(w2)}),
            options={"per_beam_doses": True},
        )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert [b.BeamNumber for b in plan.IonBeamSequence] == [2]
    beam_files = sorted(p.name for p in out.glob("RTDOSE_physical_dose_beam*.dcm"))
    assert beam_files == ["RTDOSE_physical_dose_beam2.dcm"]
    dose = pydicom.dcmread(out / beam_files[0])
    group = dose.ReferencedRTPlanSequence[0].ReferencedFractionGroupSequence[0]
    assert group.ReferencedBeamSequence[0].ReferencedBeamNumber == 2


def test_photon_stf_exports_plan_and_dose_references_it(tmp_path):
    raw = pymatreader.read_mat(str(DATA / "photons_testData.mat"))
    pln = validate_pln(raw["pln"])
    ct = validate_ct(raw["ct"])
    stf = validate_stf(raw["stf"])
    import SimpleITK as sitk  # noqa: PLC0415

    dose = sitk.Cast(ct.cube_hu * 0 + 1.0, sitk.sitkFloat64)
    with pytest.warns(UserWarning, match="not deliverable"):
        out = _export(
            tmp_path,
            ct=ct,
            stf=stf,
            pln=pln,
            result={"physical_dose": dose, "w": np.ones(stf.total_number_of_bixels)},
        )
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    assert plan.SOPClassUID == "1.2.840.10008.5.1.4.1.1.481.5"
    ds = pydicom.dcmread(out / "RTDOSE_physical_dose.dcm")
    assert ds.ReferencedRTPlanSequence[0].ReferencedSOPInstanceUID == plan.SOPInstanceUID


def _ds_elements(dataset):
    for element in dataset.iterall():
        if element.VR == "DS":
            values = element.value
            if not isinstance(values, (list, pydicom.multival.MultiValue)):
                values = [values]
            for value in values:
                yield element.keyword, str(value)


def test_all_decimal_strings_have_at_most_16_characters(tmp_path, generated):
    pln, _, _, stf, w, result = generated
    img = sitk.GetImageFromArray(np.zeros((3, 10, 10), np.float32))
    img.SetSpacing((0.976562511920929, 0.976562511920929, 1.1))
    img.SetOrigin((-249.51171875, -123.30000000000001, 0.1))
    ct = validate_ct(cube_hu=img)
    mask = np.zeros((3, 10, 10), np.uint8)
    mask[:, 3:6, 3:7] = 1
    mask_img = sitk.GetImageFromArray(mask)
    mask_img.CopyInformation(img)
    cst = validate_cst(
        [{"name": "PTV", "voi_type": "TARGET", "mask": mask_img, "ct_image": ct}], ct
    )
    dose = sitk.GetImageFromArray(np.random.default_rng(0).random((3, 10, 10)))
    dose.CopyInformation(img)

    for fmt in ("rtstruct", "seg"):
        out = tmp_path / fmt
        DicomExporter(out, structure_format=fmt).save(
            ct=ct, cst=cst, dose=dose, stf=stf, pln=pln, weights=w
        )
        files = sorted(out.glob("*.dcm"))
        assert any(f.name.startswith("CT") for f in files)
        for path in files:
            too_long = [(kw, v) for kw, v in _ds_elements(pydicom.dcmread(path)) if len(v) > 16]
            assert not too_long, f"{path.name}: {too_long[:2]}"


def test_failed_plan_export_leaves_the_previous_export_untouched(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    out = tmp_path / "out"
    DicomExporter(out).save(ct=ct, cst=cst, stf=stf, pln=pln, weights=w)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    assert "RTPLAN.dcm" in before

    bad_options = [
        ({"weights": w[:-1]}, {}, ValueError),
        ({"weights": -w}, {}, ValueError),
        ({"weights": np.zeros_like(w)}, {}, ValueError),
        ({"weights": w}, {"metadata": {"patient_position": "FFS"}}, NotImplementedError),
    ]
    for kwargs, options, error in bad_options:
        with pytest.raises(error):
            DicomExporter(out, overwrite=True, **options).save(
                ct=ct, cst=cst, stf=stf, pln=pln, **kwargs
            )
        assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_particle_number_is_weight_times_one_million(tmp_path, proton_mat):
    pln, ct, cst, stf, w = proton_mat
    for ray in stf.beams[0].rays:
        for beamlet in ray.beamlets:
            beamlet.num_particles_per_mu = 3.7e6
    out = _export(tmp_path, ct=ct, stf=stf, pln=pln, weights=w)
    plan = pydicom.dcmread(out / "RTPLAN.dcm")
    expected = float(np.sum(w[: stf.beams[0].total_number_of_bixels])) * 1e6
    assert float(plan.IonBeamSequence[0].FinalCumulativeMetersetWeight) == pytest.approx(
        expected, rel=1e-5
    )

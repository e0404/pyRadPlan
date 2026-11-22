import pytest

from pyRadPlan.app import DEFAULT_PLAN, build_plan, plan_summary


def test_default_plan_follows_the_qt_plan_form():
    pln = build_plan(None, DEFAULT_PLAN)
    summary = plan_summary(pln)
    assert summary == {
        "radiation_mode": "photons",
        "machine": "Generic",
        "num_of_fractions": 30,
        "gantry_angles": [0.0],
        "couch_angles": [0.0],
        "bixel_width": 5.0,
        "iso_center": None,
        "dose_grid_resolution": [3.0, 3.0, 3.0],
        "engine": None,
        "bio_model": "none",
    }
    assert plan_summary(None) is None


def test_partial_changes_keep_the_other_fields():
    pln = build_plan(None, DEFAULT_PLAN)
    pln.prescribed_dose = 42.0
    pln.prop_opt = {"solver": "scipy"}
    changed = build_plan(pln, {"gantry_angles": [0, 72, 144], "engine": "SVDPB"})
    summary = plan_summary(changed)
    assert summary["gantry_angles"] == [0.0, 72.0, 144.0]
    assert summary["couch_angles"] == [0.0, 0.0, 0.0]
    assert summary["engine"] == "SVDPB" and summary["bixel_width"] == 5.0
    assert changed.prescribed_dose == 42.0 and changed.prop_opt == {"solver": "scipy"}
    assert pln.prop_stf["gantry_angles"] == [0.0]


def test_single_couch_angle_applies_to_every_beam():
    pln = build_plan(None, {**DEFAULT_PLAN, "gantry_angles": [0, 90], "couch_angles": [10]})
    assert pln.prop_stf["couch_angles"] == [10.0, 10.0]
    with pytest.raises(ValueError, match="couch angles"):
        build_plan(pln, {"gantry_angles": [0, 90, 180], "couch_angles": [0, 0]})


def test_iso_center_and_dose_grid():
    pln = build_plan(None, {**DEFAULT_PLAN, "iso_center": [1, 2, 3]})
    assert plan_summary(pln)["iso_center"] == [1.0, 2.0, 3.0]
    assert plan_summary(build_plan(pln, {"iso_center": None}))["iso_center"] is None
    with pytest.raises(ValueError, match="iso_center"):
        build_plan(pln, {"iso_center": [1, 2]})
    coarse = build_plan(pln, {"dose_grid_resolution": [6, 6, 4]})
    assert coarse.prop_dose_calc["dose_grid"] == {"resolution": {"x": 6.0, "y": 6.0, "z": 4.0}}


def test_changing_the_radiation_mode_starts_from_its_defaults():
    photons = build_plan(None, {**DEFAULT_PLAN, "engine": "SVDPB"})
    protons = build_plan(photons, {"radiation_mode": "protons"})
    summary = plan_summary(protons)
    assert summary["radiation_mode"] == "protons" and summary["machine"] == "Generic"
    assert summary["bio_model"] == "constant_rbe" and summary["engine"] is None
    assert type(protons).__name__ == "IonPlan"


def test_invalid_input_raises_value_error():
    pln = build_plan(None, DEFAULT_PLAN)
    with pytest.raises(ValueError, match="Unknown plan field"):
        build_plan(pln, {"gantry": [0]})
    with pytest.raises(ValueError):
        build_plan(pln, {"num_of_fractions": 0})

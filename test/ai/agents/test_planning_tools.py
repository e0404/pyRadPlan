"""Tests for the planning agent's geometry, helper, feasibility and constraint tools."""

import math

import numpy as np
import pytest
import SimpleITK as sitk

pytest.importorskip("pydantic_ai")

from pydantic_ai import models
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pyRadPlan import load_tg119
from pyRadPlan.ai.agents._planning import (
    BooleanRecipe,
    ExcludeSpec,
    Goal,
    IsodoseRecipe,
    MarginRecipe,
    PlanningError,
    PlanningReport,
    PlanningState,
    PlanRecord,
    Protocol,
    RingRecipe,
    available_constraint_tables,
    convert_fractionation,
    create_planning_agent,
    eqd2,
    load_constraint_table,
    load_goals,
    physical_from_eqd2,
    plan_score,
    save_goals,
    select_distinct_plans,
    site_goals,
)
from pyRadPlan.ai.agents._planning import _geometry as geo
from pyRadPlan.ai.agents._planning._constraints import ConstraintTable
from pyRadPlan.ai.agents._planning._state import _cpu_backend, _dose_is_linear, sitk_array
from pyRadPlan.optimization.objectives import (
    MeanDose,
    MeanOverdosing,
    SquaredDeviation,
    SquaredOverdosing,
)
from pyRadPlan.plan import PhotonPlan

models.ALLOW_MODEL_REQUESTS = False

TG_PROTOCOL = Protocol(
    goals=[
        Goal(voi="OuterTarget", metric="D95", op=">=", value=1.9),
        Goal(voi="Core", metric="max", op="<=", value=1.0),
        Goal(voi="Core", metric="V1.5Gy", op="<=", value=5),
    ]
)


@pytest.fixture
def state():
    ct, cst = load_tg119()
    return PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=TG_PROTOCOL)


# ---- EQD2 ------------------------------------------------------------------------------


def test_eqd2_known_values():
    assert eqd2(60, 30, 3) == pytest.approx(60)  # 2 Gy fractions are their own EQD2
    assert eqd2(30, 5, 2) == pytest.approx(60)  # the cord example: 5 x 6 Gy
    assert eqd2(60, 20, 1.5) == pytest.approx(60 * 4.5 / 3.5)


@pytest.mark.parametrize("n, ab", [(5, 2), (20, 3), (39, 10), (1, 3)])
def test_physical_from_eqd2_inverts(n, ab):
    for e in (20.0, 54.0, 78.0):
        assert eqd2(physical_from_eqd2(e, n, ab), n, ab) == pytest.approx(e)


def test_convert_fractionation():
    assert convert_fractionation(70, 35, 35, 3) == 70
    # fewer, larger fractions -> lower isoeffective total dose
    assert convert_fractionation(78, 39, 20, 3) < 78


# ---- constraint tables -----------------------------------------------------------------


def _fractionation_table() -> ConstraintTable:
    """A table in physical dose for 60 Gy in 20 fractions, with relative target goals."""
    return ConstraintTable.model_validate(
        {
            "id": "test",
            "name": "test",
            "version": "1",
            "dose_reference": {"type": "fractionation"},
            "sites": {
                "prostate": {
                    "prescription": {"total_dose_gy": 60.0, "num_fractions": 20},
                    "targets": {
                        "PTV": {
                            "constraints": [
                                {"metric": "D95", "op": ">=", "value": 95, "unit": "%Rx"}
                            ]
                        }
                    },
                    "structures": {
                        "Rectum": {
                            "alpha_beta": 3,
                            "constraints": [{"metric": "V40.5Gy", "op": "<=", "value": 38}],
                        }
                    },
                }
            },
        }
    )


def test_bundled_table_loads_and_converts_everywhere():
    tables = available_constraint_tables()
    assert set(tables) == {"demo"}
    t = load_constraint_table("demo")
    assert "NOT FOR CLINICAL USE" in t.disclaimer
    for site in t.sites:
        goals, _ = site_goals(t, site, 30, 60.0)
        assert goals, site
        assert all(g.source and g.source.startswith("demo") for g in goals)


def test_eqd2_conversion_to_plan():
    t = load_constraint_table("demo")
    goals, notes = site_goals(t, "prostate", 39, 78.0, per_fraction=True)
    v70 = next(g for g in goals if g.voi == "Rectum" and g.value == 20)
    thr_per_fx = float(v70.metric[1:-2])
    # threshold, converted back to EQD2 with the rectum alpha/beta, is 70 Gy
    assert eqd2(thr_per_fx * 39, 39, 3) == pytest.approx(70, rel=1e-3)
    d95 = next(g for g in goals if g.voi == "PTV" and g.metric == "D95")
    assert d95.value == pytest.approx(78 * 0.95 / 39)  # % of the prescription, not converted
    cord_like = site_goals(t, "head_neck", 5, 35.0, per_fraction=False)[0]
    cord = next(g for g in cord_like if g.voi == "SpinalCord")
    assert eqd2(cord.value, 5, 2) == pytest.approx(50, rel=1e-3)
    assert any("NOT FOR CLINICAL USE" in n for n in notes)


def test_native_fractionation_and_relative_goals():
    goals, notes = site_goals(_fractionation_table(), "prostate", 20, 60.0)
    d95 = next(g for g in goals if g.voi == "PTV" and g.metric == "D95")
    assert d95.value == pytest.approx(60 * 0.95 / 20)
    assert any(g.metric == "V2.025Gy" for g in goals if g.voi == "Rectum")
    assert not any("converted" in n for n in notes)

    goals, notes = site_goals(_fractionation_table(), "prostate", 30, 60.0, per_fraction=False)
    thr = float(next(g for g in goals if g.voi == "Rectum").metric[1:-2])
    assert eqd2(thr, 30, 3) == pytest.approx(eqd2(40.5, 20, 3), rel=1e-3)
    assert any("converted" in n for n in notes)


def test_sbrt_fractionation_warns():
    t = load_constraint_table("demo")
    _, notes = site_goals(t, "thorax", 5, 50.0)
    assert any("unreliable" in n for n in notes)


def test_goal_file_round_trip_and_dose_conversion(tmp_path):
    goals = [
        Goal(voi="PTV", metric="D95", op=">=", value=57.0, source="95 %"),
        Goal(voi="Rectum", metric="V40Gy", op="<=", value=35, hard=False),
    ]
    path = tmp_path / "goals.json"
    save_goals(path, goals, "total", 20, name="prostate")
    text = path.read_text()
    assert '"hard"' not in text.splitlines()[-4]  # defaults are left out
    assert load_goals(path, "total", 20) == (goals, [])

    per_fx, notes = load_goals(path, "per_fraction", 20)
    assert [g.label for g in per_fx] == ["PTV D95 >= 2.85", "Rectum V2Gy <= 35"]
    assert not per_fx[1].hard and notes == []
    save_goals(path, per_fx, "per_fraction", 20)
    assert load_goals(path, "total", 20)[0] == goals

    _, notes = load_goals(path, "per_fraction", 30)
    assert "20 fractions" in notes[0]


def test_goal_file_rejects_invalid_content(tmp_path):
    path = tmp_path / "goals.json"
    path.write_text(
        '{"dose": "total", "goals": [{"voi": "PTV", "metric": "D95%", "op": ">=", "value": 1}]}'
    )
    with pytest.raises(ValueError, match="D95%"):
        load_goals(path, "total", 20)
    path.write_text('{"goals": []}')  # the dose convention is required
    with pytest.raises(ValueError, match="dose"):
        load_goals(path, "total", 20)


def test_unknown_site_and_table():
    with pytest.raises(ValueError, match="Sites"):
        site_goals(load_constraint_table("demo"), "knee", 30, 60.0)
    with pytest.raises(ValueError, match="Available"):
        load_constraint_table("nope")


# ---- shape analysis and feasibility (no dose needed) -----------------------------------


def test_target_shape_detects_concavity(state):
    shape = state.analyze_target_shape()["OuterTarget"]
    assert shape["solidity"] < 0.85 and shape["num_components"] == 1
    core = next(n for n in shape["neighbours"] if n["voi"] == "Core")
    assert core["pct_of_voi_in_concavity"] > 50
    assert shape["enclosing_structures"] == ["BODY"]  # BODY typed OAR is not a neighbour
    assert any("concave" in h and "Core" in h for h in shape["hints"])


def test_convex_shape_recommends_ring():
    arr = np.zeros((30, 30, 30), bool)
    zz, yy, xx = np.ogrid[:30, :30, :30]
    arr[(zz - 15) ** 2 + (yy - 15) ** 2 + (xx - 15) ** 2 <= 64] = True
    mask = sitk.GetImageFromArray(arr.astype(np.uint8))
    shape = geo.target_shape(mask, {})
    assert shape["solidity"] > 0.9
    assert "ring is suitable" in shape["hints"][0]


def test_feasibility_steep_and_conflict(state):
    res = state.check_feasibility()
    core_max = next(c for c in res["checks"] if c["goal"].startswith("Core max"))
    assert core_max["verdict"] in ("steep", "very steep")
    assert core_max["min_distance_mm"] > 0
    conflict = Protocol(
        goals=[
            Goal(voi="OuterTarget", metric="min", op=">=", value=1.8),
            Goal(voi="BODY", metric="max", op="<=", value=1.0),
        ]
    )
    state.protocol = conflict
    res = state.check_feasibility()
    assert res["summary"]["conflicts"] == 1


def test_feasibility_mean_budget(state):
    body = geo.as_bool(state._voi("BODY").mask)
    overlap = body & geo.as_bool(state._voi("OuterTarget").mask)
    floor_mean = 1.9 * overlap.sum() / body.sum()
    rest = (0.5 * body.sum() - 1.9 * overlap.sum()) / (body.sum() - overlap.sum())

    def check(limit):
        state.protocol = Protocol(
            goals=[
                Goal(voi="OuterTarget", metric="D95", op=">=", value=1.9),
                Goal(voi="BODY", metric="mean", op="<=", value=limit),
            ]
        )
        return state.check_feasibility()

    res = check(0.5)
    (budget,) = res["mean_budgets"]
    assert budget["mean_from_target_floors"] == pytest.approx(floor_mean, abs=0.01)
    assert budget["max_mean_outside_targets"] == pytest.approx(rest, abs=0.01)
    assert budget["target_overlap_cc"] > 0
    # the goal sees the whole BODY, its objectives only what overlap priorities leave
    assert budget["objectives_see_cc"] < budget["voi_cc"]
    body_check = next(c for c in res["checks"] if c["goal"] == "BODY mean <= 0.5")
    assert body_check["verdict"] == "see mean_budgets"

    res = check(floor_mean / 2)
    assert res["mean_budgets"][0]["verdict"] == "conflict"
    assert res["summary"]["conflicts"] == 1


def test_mean_dose_with_reference_gets_a_note(state):
    out = state.set_objectives("Core", [MeanDose(priority=1, d_ref=1.0)])
    assert "Mean Overdosing" in out["note"]
    assert "note" not in state.set_objectives("Core", [MeanDose(priority=1)])
    assert "note" not in state.set_objectives("Core", [MeanOverdosing(priority=1, d_max=1.0)])


def test_normal_tissue_keeps_a_margin_to_the_targets(state):
    near = geo.expand(state._voi("OuterTarget").mask, 3.0)
    assert state.normal_tissue_margin_mm == 5.0
    assert not (state._normal_tissue_region() & near).any()
    state.normal_tissue_margin_mm = 0.0
    assert (state._normal_tissue_region() & near).any()


def test_normal_tissue_level_follows_target_floors(state):
    rx = state.pln.prescribed_dose / state.pln.num_of_fractions
    (goal,) = state._implicit_goals()
    assert goal.value == pytest.approx(rx)  # floor 1.9 is below the prescription
    state.protocol = Protocol(
        goals=[
            Goal(voi="OuterTarget", metric="D95", op=">=", value=rx + 0.2),
            Goal(voi="Core", metric="min", op=">=", value=rx + 1.0),  # not a target
        ]
    )
    (goal,) = state._implicit_goals()
    assert goal.value == pytest.approx(rx + 0.2)
    assert "5 mm outside the targets" in goal.source


# ---- helper structures -----------------------------------------------------------------


def test_ring_follows_shape_and_respects_exclusions(state):
    plain = state.create_helper(RingRecipe(name="ring", target="OuterTarget", outer_mm=20))
    assert plain["name"] == "opt_ring"
    assert plain["pct_of_helper_in_voi"].get("Core", 0) > 0  # plain ring runs into the concavity
    smart = state.create_helper(
        RingRecipe(
            name="ring_smart",
            target="OuterTarget",
            outer_mm=20,
            exclude=[ExcludeSpec(voi="Core", margin_mm=3)],
            skip_concavity=True,
        )
    )
    assert "Core" not in smart["pct_of_helper_in_voi"]
    assert "OuterTarget" not in smart["pct_of_helper_in_voi"]
    assert smart["volume_cc"] < plain["volume_cc"]


def test_margin_and_boolean_recipes(state):
    prv = state.create_helper(MarginRecipe(name="core_prv", source="Core", margin_mm=5))
    core_cc = state.inspect_structures()["structures"][0]["volume_cc"]
    assert prv["volume_cc"] > core_cc
    split = state.create_helper(
        BooleanRecipe(kind="subtract", name="ptv_opt", a="OuterTarget", b="Core", b_margin_mm=8)
    )
    assert split["pct_of_helper_in_voi"]["OuterTarget"] == 100.0
    # the other direction: the helper is OuterTarget minus a margin, so only a part of it
    assert 0 < split["pct_of_voi_in_helper"]["OuterTarget"] < 100.0
    assert state._voi("opt_ptv_opt").voi_type == "HELPER"
    # ties with the weakest VOI it overlaps, so overlap handling cannot empty it
    prio = {v.name: v.overlap_priority for v in state.cst.vois}
    assert prio["opt_ptv_opt"] == prio["OuterTarget"]  # lies inside the target only
    assert prio["opt_core_prv"] == prio["BODY"]


def test_helper_guards(state):
    with pytest.raises(PlanningError, match="inner_mm"):
        state.create_helper(RingRecipe(name="x", target="OuterTarget", inner_mm=5, outer_mm=5))
    with pytest.raises(PlanningError, match="cc"):
        state.create_helper(MarginRecipe(name="tiny", source="Core", margin_mm=-50))
    state.create_helper(MarginRecipe(name="prv", source="Core", margin_mm=3))
    with pytest.raises(PlanningError, match="exists"):
        state.create_helper(MarginRecipe(name="opt_prv", source="Core", margin_mm=3))
    with pytest.raises(PlanningError, match="helper"):
        state.map_structures(aliases={"Core": "opt_prv"})
    with pytest.raises(PlanningError, match="not a helper"):
        state.delete_helper("Core")
    state.delete_helper("opt_prv")
    assert "opt_prv" not in state.voi_names


def test_helpers_are_never_evaluated(state):
    state.create_helper(MarginRecipe(name="prv", source="Core", margin_mm=3))
    assert "opt_prv" not in [v.name for v in state._clinical_vois()]


# ---- constraint lookup on a messy patient ----------------------------------------------


@pytest.fixture
def hn_state():
    ct, cst = load_tg119()
    core, target, body = cst.vois
    cst = cst.model_copy(
        update={
            "vois": [
                core.model_copy(update={"name": "Myelon"}),
                target.model_copy(update={"name": "PTV_70"}),
                body.model_copy(update={"name": "Aussenkontur"}),
            ]
        }
    )
    pln = PhotonPlan(num_of_fractions=35, prescribed_dose=70.0)
    return PlanningState(ct=ct, cst=cst, pln=pln, protocol=Protocol(locked=False))


def test_site_constraints_apply_and_automap(hn_state):
    hn_state.map_structures(aliases={"SpinalCord": "Myelon"})
    res = hn_state.get_site_constraints("demo", "head_neck", apply=True)
    assert res["applied"] is True
    assert res["auto_mapped"] == {"PTV": "PTV_70"}
    assert "Brainstem" in res["unmatched_structures"]
    vois = {g.voi for g in hn_state.protocol.goals}
    assert vois == {"PTV", "SpinalCord"}  # goals for absent structures dropped
    cord = next(g for g in hn_state.protocol.goals if g.voi == "SpinalCord")
    assert eqd2(cord.value * 35, 35, 2) == pytest.approx(50, rel=1e-3)  # EQD2, per fraction
    assert hn_state.check_feasibility()["checks"]


def test_locked_protocol_is_not_replaced(state):
    before = state.protocol
    res = state.get_site_constraints("demo", "prostate", apply=True)
    assert res["applied"] is False and state.protocol is before


def test_applied_site_constraints_reevaluate_the_plans(hn_state, monkeypatch):
    applied = []
    monkeypatch.setattr(PlanningState, "set_protocol", lambda self, p: applied.append(p) or [])
    hn_state.get_site_constraints("demo", "head_neck", apply=True)
    assert [p.name for p in applied] == ["demo/head_neck"]


# ---- tools that need a dose ------------------------------------------------------------


@pytest.fixture(scope="module")
def dosed():
    ct, cst = load_tg119()
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=TG_PROTOCOL, max_optimizations=50)
    s.set_beams([0, 72, 144, 216, 288], None)
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    s.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.8)])
    s.optimize()
    s.snapshot("v1")
    return s


def test_localize_hot_and_cold_spots(dosed):
    hot = dosed.localize_dose_spots("hot", 1.2)
    assert hot["spots"] and hot["spots"][0]["dose_extreme"] > 1.2
    assert "relative_to_target_center" in hot["spots"][0]
    cold = dosed.localize_dose_spots("cold", 1.9)
    assert all(s["dose_extreme"] < 1.9 for s in cold["spots"])
    assert all("OuterTarget" in s["inside"] for s in cold["spots"])
    core = dosed.localize_dose_spots("hot", 0.8, voi="Core")
    assert core["region"] == "Core"


def test_isodose_helper_keeps_dij_and_compare(dosed):
    out = dosed.create_helper(IsodoseRecipe(name="hot", threshold=1.4))
    assert out["dose_influence"] == "kept"
    dosed.set_objectives("opt_hot", [SquaredOverdosing(priority=30, d_max=1.0)])
    dosed.optimize()
    assert dosed.num_dij_calcs == 1
    cmp = dosed.compare_snapshots("v1")
    assert cmp["versions"] == ["v1", "current"]
    assert {g.label for g in TG_PROTOCOL.goals} <= {g["goal"] for g in cmp["goals"]}
    assert "opt_hot" not in cmp["dose_delta_per_voi"]
    with pytest.raises(PlanningError):
        dosed.compare_snapshots("nope")


def test_optimize_rejects_objectives_on_empty_vois(dosed):
    from pyRadPlan.cst._voi import create_voi

    ref = dosed.cst.vois[0]
    target = dosed._voi("OuterTarget")
    empty = create_voi(
        voi_type="OAR",
        name="Lens",
        mask=geo.to_mask(np.zeros_like(geo.as_bool(ref.mask)), ref.mask),
        grid=ref.grid,
    )
    # HELPER priority: overlap handling hands all of its voxels to the target
    inside = create_voi(voi_type="HELPER", name="InTarget", mask=target.mask, grid=target.grid)
    before, n_opt = dosed.cst, dosed.num_optimizations
    try:
        dosed.cst = dosed.cst.model_copy(update={"vois": [*dosed.cst.vois, empty, inside]})
        dosed.set_objectives("Lens", [SquaredOverdosing(priority=1, d_max=0.5)])
        dosed.set_objectives("InTarget", [SquaredOverdosing(priority=1, d_max=2.1)])
        with pytest.raises(PlanningError, match=r"\['Lens', 'InTarget'\] have no voxels"):
            dosed.optimize()
        assert dosed.num_optimizations == n_opt
    finally:
        dosed.cst = before


def test_optimize_reports_normal_tissue_and_optimizer(dosed):
    ev = dosed.optimize(max_iter=30)
    rx = dosed.pln.prescribed_dose / dosed.pln.num_of_fractions  # per-fraction convention
    implicit = next(g for g in ev["goals"] if g["goal"].startswith("NormalTissue"))
    assert implicit["goal"] == f"NormalTissue max <= {rx:g}" and implicit["hard"] is False
    assert set(ev["normal_tissue"]) == {"mean", "D50", "D5", "D2", "max"}
    assert ev["normal_tissue"]["max"] == pytest.approx(implicit["value"], abs=0.01)
    report = dosed.optimizer_report()
    assert report["max_iter"] == 30 and 0 < report["iterations"] <= 30
    assert ev["optimizer"]["iterations"] == report["iterations"]
    assert report["objective_final"] <= report["objective_start"]
    shares = [c["share_pct"] for c in report["contributions"]]
    assert shares == sorted(shares, reverse=True) and sum(shares) == pytest.approx(100, abs=0.5)
    assert "NormalTissue" in dosed.analyze_qis(["NormalTissue"])


def test_contributions_do_not_depend_on_the_dose_convention():
    contributions = []
    for convention, n in (("per_fraction", 30), ("total", 1)):
        ct, cst = load_tg119()
        pln = PhotonPlan(num_of_fractions=30, dose_convention=convention)
        s = PlanningState(ct=ct, cst=cst, pln=pln, protocol=TG_PROTOCOL)
        s.set_beams([0, 120, 240], None)
        for voi in s.cst.vois:
            s.set_objectives(voi.name, [])
        s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=60.0 / n)])
        s.set_objectives("Core", [MeanOverdosing(priority=10, d_max=20.0 / n)])
        s.optimize(max_iter=30)
        contributions.append([c["value"] for c in s.optimizer_report()["contributions"]])
    assert contributions[1] == pytest.approx(contributions[0], rel=1e-3)


def test_restore_brings_back_dose_and_helpers(dosed):
    dosed.set_objectives("Core", [MeanOverdosing(priority=10, d_max=0.5)])
    dosed.optimize(max_iter=30)
    saved, plan = dosed.dose, dosed.num_optimizations
    assert dosed.snapshot("v2")["plan"] == plan
    dosed.create_helper(IsodoseRecipe(name="after_v2", threshold=1.4))
    dosed.set_objectives("Core", [SquaredOverdosing(priority=1, d_max=0.5)])
    assert dosed.dose is None

    out = dosed.restore("v2")
    assert out["plan"] == plan and "no optimize needed" in out["note"]
    assert dosed.dose is saved and dosed.optimizer_report()["max_iter"] == 30
    assert "opt_after_v2" not in dosed.voi_names + dosed.created_helpers
    assert dosed.localize_dose_spots("hot", 1.2)["spots"]  # usable without optimize
    assert dosed.snapshot("v3")["plan"] == plan


def test_optimize_uses_scipy_on_the_cpu_and_is_reproducible(dosed, monkeypatch):
    import pyRadPlan.ai.agents._planning._state as pa  # noqa: PLC0415
    from pyRadPlan import settings  # noqa: PLC0415

    seen = []
    real = pa.fluence_optimization

    def spy(*args, **kwargs):
        seen.append((args[4].prop_opt["solver"]["name"], settings.xp.prefer_gpu))
        return real(*args, **kwargs)

    monkeypatch.setattr(pa, "fluence_optimization", spy)
    before = settings.xp.prefer_gpu
    dosed.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.85)])
    dosed.optimize(max_iter=40)
    first = sitk_array(dosed.dose).copy()
    with pytest.raises(PlanningError, match="exactly these objectives"):
        dosed.optimize(max_iter=40)
    monkeypatch.setattr(dosed, "_refuse_repeat", lambda run: None)
    dosed.optimize(max_iter=40)
    np.testing.assert_array_equal(sitk_array(dosed.dose), first)
    assert seen == [("scipy", False)] * 2
    assert settings.xp.prefer_gpu == before
    plan = dosed.plans[-1]
    assert plan.fluence.size == dosed.dij.total_num_of_bixels
    assert plan.dose is plan.result[plan.quantity].values
    assert not plan.result.beam_quantities  # kept plans hold the totals only


def test_cpu_backend_restores_only_what_it_changed():
    from pyRadPlan import settings  # noqa: PLC0415

    xp = settings.xp
    before = xp.prefer_gpu, xp.preferred_cpu_array_backend
    try:
        xp.prefer_gpu, xp.preferred_cpu_array_backend = True, "numpy"
        with _cpu_backend(False):
            assert xp.prefer_gpu is True
            xp.prefer_gpu = False  # set meanwhile: not undone
        assert xp.prefer_gpu is False

        xp.prefer_gpu = True
        with _cpu_backend():
            assert xp.prefer_gpu is False
        assert xp.prefer_gpu is True and xp.preferred_cpu_array_backend == "numpy"

        with _cpu_backend():
            xp.prefer_gpu = True  # set meanwhile: kept
            xp.preferred_cpu_array_backend = "array_api_strict"  # was numpy already: kept
        assert xp.prefer_gpu is True and xp.preferred_cpu_array_backend == "array_api_strict"
    finally:
        xp.prefer_gpu, xp.preferred_cpu_array_backend = before


def test_optimize_can_start_from_an_earlier_plan(dosed):
    dosed.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.8)])
    dosed.optimize(max_iter=60)
    base, cold = dosed.plans[-1].index, dosed.optimizer_report()
    ev = dosed.optimize(max_iter=60, start_from=base)
    warm = dosed.optimizer_report()
    assert ev["optimizer"]["start_from"] == base and warm["start_from"] == base
    assert warm["objective_start"] < 0.1 * cold["objective_start"]
    n_opt = dosed.num_optimizations
    with pytest.raises(PlanningError, match="No plan 999"):
        dosed.optimize(start_from=999)
    assert dosed.num_optimizations == n_opt


def test_blend_plans_mixes_doses_without_an_optimization(dosed):
    dosed.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.75)])
    dosed.optimize(max_iter=60)
    a = dosed.plans[-1]
    dosed.set_objectives("Core", [SquaredOverdosing(priority=300, d_max=0.4)])
    dosed.optimize(max_iter=60)
    b = dosed.plans[-1]
    n_opt = dosed.num_optimizations

    ev = dosed.blend_plans([a.index, b.index], [3, 1])
    blend = dosed.plans[-1]
    assert dosed.num_optimizations == n_opt and blend.index == b.index + 1
    assert ev["blend"] == {"plans": [a.index, b.index], "weights": [0.75, 0.25]}
    assert blend.blend == ev["blend"] and blend.cst is a.cst
    expected = 0.75 * sitk_array(a.dose) + 0.25 * sitk_array(b.dose)
    np.testing.assert_allclose(sitk_array(blend.dose), expected, rtol=1e-6, atol=1e-9)
    core = {p.index: {g["goal"]: g["value"] for g in p.evaluation["goals"]} for p in (a, b)}
    blended = {g["goal"]: g["value"] for g in ev["goals"]}
    mixed_max = 0.75 * core[a.index]["Core max <= 1"] + 0.25 * core[b.index]["Core max <= 1"]
    assert blended["Core max <= 1"] <= mixed_max + 1e-6
    with pytest.raises(PlanningError, match="No optimization behind"):
        dosed.optimizer_report()

    ev = dosed.optimize(max_iter=20, start_from=blend.index)
    assert ev["optimizer"]["start_from"] == blend.index


@pytest.mark.parametrize(
    "plans, weights, match",
    [
        ([1], [1.0], "two different"),
        ([1, 1], [1.0, 1.0], "two different"),
        ([1, 2], [1.0], "need 2 weights"),
        ([1, 2], [-1.0, 2.0], ">= 0"),
        ([1, 2], [0.0, 0.0], ">= 0"),
        ([1, 999], [1.0, 1.0], "No plan 999"),
    ],
)
def test_blend_plans_rejects_bad_input(dosed, plans, weights, match):
    with pytest.raises(PlanningError, match=match):
        dosed.blend_plans(plans, weights)


@pytest.fixture(scope="module")
def two_trade_offs():
    """Plan 1 covers the target better, plan 2 spares the core better; both pass the hard goal."""
    ct, cst = load_tg119()
    protocol = Protocol(
        goals=[
            Goal(voi="OuterTarget", metric="D95", op=">=", value=1.6),
            Goal(voi="Core", metric="max", op="<=", value=1.0, hard=False),
            Goal(voi="Core", metric="V1.5Gy", op="<=", value=5, hard=False),
        ]
    )
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=protocol)
    s.set_beams([0, 72, 144, 216, 288], None)
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    s.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.8)])
    s.optimize()
    s.set_objectives("Core", [SquaredOverdosing(priority=300, d_max=0.4)])
    s.optimize()
    return s


def test_pareto_front_samples_blends_for_free(two_trade_offs):
    s = two_trade_offs
    n_opt, n_plans = s.num_optimizations, len(s.plans)
    out = s.pareto_front([1, 2])
    assert s.num_optimizations == n_opt and len(s.plans) == n_plans and s.last_pareto is out
    assert out == s.pareto_front([1, 2])

    axes = {a["goal"]: a for a in out["axes"]}
    assert {"OuterTarget D95 >= 1.6", "Core max <= 1"} <= set(axes)
    assert out["flat_goals"] == ["Core V1.5Gy <= 5"]
    for p in s.plans[:2]:
        for g in p.evaluation["goals"]:
            if g["goal"] in axes:
                assert axes[g["goal"]]["plans"][p.index] == pytest.approx(g["value"], abs=1e-3)
    assert out["front_size"] >= len(out["front"]) > 1
    assert all(e["all_hard_passed"] for e in out["front"])
    # Max goals are convex in the blend: a little of plan 2 lowers the Core and NormalTissue
    # max of plan 1 without costing coverage, so plan 1 is dominated by such a blend
    assert out["dominated_plans"] == [1] and out["plans_failing_hard"] == []

    c = out["compromise"]
    assert c["weights"] == out["front"][0]["weights"] and 0 < c["weights"][0] < 1
    assert c["weakest_axes"] and not c["best_on"]
    ev = s.blend_plans([1, 2], c["weights"])
    blended = {g["goal"]: g["value"] for g in ev["goals"]}
    for goal, value in c["values"].items():
        assert blended[goal] == pytest.approx(value, abs=1e-3)


def test_pareto_front_needs_a_dose_linear_in_the_fluence(two_trade_offs, monkeypatch):
    from types import SimpleNamespace  # noqa: PLC0415

    lq = SimpleNamespace(provides=lambda *names: True)
    variable = SimpleNamespace(bio_model=lq, rbe=None, alpha_dose=1, sqrt_beta_dose=1)
    constant = SimpleNamespace(bio_model=None, rbe=1.1, alpha_dose=None, sqrt_beta_dose=None)
    assert _dose_is_linear(variable, "physical_dose") and _dose_is_linear(constant, "rbe_x_dose")
    assert not _dose_is_linear(variable, "rbe_x_dose")
    assert not two_trade_offs._nonlinear_plans  # photons

    monkeypatch.setattr(two_trade_offs, "_nonlinear_plans", {2})
    with pytest.raises(PlanningError, match="variable RBE.*blend_plans"):
        two_trade_offs.pareto_front([1, 2])


def test_blends_and_pareto_samples_are_capped(two_trade_offs, monkeypatch):
    monkeypatch.setattr(two_trade_offs, "max_pareto_samples", 25)
    assert two_trade_offs.pareto_front([1, 2], n_samples=2000)["num_blends"] <= 25
    kept = sum(1 for p in two_trade_offs.plans if p.blend)
    monkeypatch.setattr(two_trade_offs, "max_blends", kept)
    with pytest.raises(PlanningError, match="blends are kept already"):
        two_trade_offs.blend_plans([1, 2], [0.5, 0.5])
    assert sum(1 for p in two_trade_offs.plans if p.blend) == kept


def test_pareto_front_axes_and_dominated_plans(two_trade_offs):
    out = two_trade_offs.pareto_front([1, 2], axes=["Core max <= 1"])
    assert [a["goal"] for a in out["axes"]] == ["Core max <= 1"] and out["flat_goals"] == []
    assert out["dominated_plans"] == [1] and out["front"][0]["best_on"] == ["Core max <= 1"]


def test_pareto_front_keeps_hard_oar_goals_as_constraints(two_trade_offs, monkeypatch):
    import pyRadPlan.ai.agents._planning._pareto as pareto  # noqa: PLC0415

    s = two_trade_offs
    protocol = s.protocol
    hard_core = [
        g.model_copy(update={"hard": True, "value": 1.2}) if g.metric == "max" else g
        for g in protocol.goals
    ]
    s.set_protocol(Protocol(goals=hard_core))
    try:
        out = s.pareto_front([1, 2])
    finally:
        s.set_protocol(protocol)
    assert out["constraint_goals"] == ["Core max <= 1.2"]
    assert "Core max <= 1.2" not in [a["goal"] for a in out["axes"]] + out["flat_goals"]

    coarse = s.pareto_front([1, 2])
    monkeypatch.setattr(pareto, "EPS", 1e-9)
    fine = s.pareto_front([1, 2])
    assert coarse["front_size"] < fine["front_size"]


def test_pareto_front_without_a_blend_passing_hard_goals(two_trade_offs):
    s = two_trade_offs
    protocol = s.protocol
    s.set_protocol(
        Protocol(goals=[*protocol.goals, Goal(voi="Core", metric="max", op="<=", value=0.1)])
    )
    try:
        out = s.pareto_front([1, 2])
    finally:
        s.set_protocol(protocol)
    assert out["num_passing_hard"] == 0 and "compromise" not in out
    scores = [e["score"] for e in out["front"]]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.parametrize(
    "plans, axes, match",
    [
        ([1], None, "two different"),
        ([1, 1], None, "two different"),
        ([1, 999], None, "No plan 999"),
        ([1, 2], ["Core mean <= 1"], "Unknown goals"),
    ],
)
def test_pareto_front_rejects_bad_input(two_trade_offs, plans, axes, match):
    with pytest.raises(PlanningError, match=match):
        two_trade_offs.pareto_front(plans, axes)


def test_agent_sees_the_pareto_front_as_lines(two_trade_offs):
    returns = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        returns.extend(p for p in messages[-1].parts if isinstance(p, ToolReturnPart))
        if not returns:
            return ModelResponse(parts=[ToolCallPart("pareto_front", {"plans": [1, 2]})])
        report = PlanningReport(all_hard_goals_met=True, summary="x").model_dump()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    create_planning_agent(FunctionModel(fn)).run_sync("plan", deps=two_trade_offs)
    out = returns[0].content
    assert all(isinstance(line, str) for line in out["front"] + out["axes"])
    assert "% plan 1 + " in out["compromise"]["blend"]
    assert "range from its best" in out["compromise"]["weakest_axes"][0]


def test_optimize_stops_once_the_objective_stalls(dosed):
    # 50 % over 50 iterations: met right after the first window
    ev = dosed.optimize(max_iter=400, min_improvement=0.5)
    assert ev["optimizer"]["converged"] and ev["optimizer"]["iterations"] < 400
    ev = dosed.optimize(max_iter=60, min_improvement=0)
    assert not ev["optimizer"]["converged"]


def test_optimize_records_plans_with_score(dosed):
    assert dosed.plans and dosed.plans[-1].dose is not None
    ev = dosed.plans[-1].evaluation
    assert ev["score"] == plan_score(ev["goals"])
    assert all("rel_margin" in g for g in ev["goals"])


def test_optimize_refuses_exact_repeats(dosed):
    dosed.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=0.7)])
    dosed.optimize(max_iter=300, min_improvement=0.5)
    plan, n_opt = dosed.plans[-1].index, dosed.num_optimizations
    assert dosed.plans[-1].evaluation["optimizer"]["iterations"] < 300
    with pytest.raises(PlanningError, match="these objectives and settings from zero"):
        dosed.optimize(max_iter=300, min_improvement=0.5)
    with pytest.raises(PlanningError, match="these objectives and settings from zero"):
        dosed.optimize(max_iter=1000, min_improvement=0.5)
    with pytest.raises(PlanningError, match="until it converged"):
        dosed.optimize(max_iter=300, min_improvement=0.5, start_from=plan)
    assert dosed.num_optimizations == n_opt

    dosed.optimize(max_iter=60, min_improvement=0.01, start_from=plan)
    dosed.set_objectives("Core", [SquaredOverdosing(priority=11, d_max=0.7)])
    dosed.optimize(max_iter=300, min_improvement=0.5)
    assert dosed.num_optimizations == n_opt + 2


def _plan(score, rel, hard_ok=True):
    goals = [{"goal": g, "hard": True, "rel_margin": r} for g, r in rel.items()]
    evaluation = {"all_hard_passed": hard_ok, "score": score, "goals": goals}
    return PlanRecord(index=score, evaluation=evaluation)


def test_plan_score():
    rows = [
        {"hard": True, "rel_margin": 0.05},
        {"hard": False, "rel_margin": 0.9},  # clipped to +20 %
        {"hard": False, "rel_margin": -3.0},  # clipped to -100 %
    ]
    assert plan_score(rows) == pytest.approx(100 * (3 * 0.05 + 0.2 - 1.0) / 5)
    assert plan_score([{"hard": True, "rel_margin": -0.01}]) < 0


def test_select_distinct_plans():
    best = _plan(10, {"ptv": 0.05, "cord": 0.10, "parotid": 0.10})
    slightly_worse = _plan(9, {"ptv": 0.04, "cord": 0.11, "parotid": 0.09})
    other_tradeoff = _plan(5, {"ptv": 0.01, "cord": 0.05, "parotid": 0.30})
    failing = _plan(50, {"ptv": -0.2, "cord": 0.9, "parotid": 0.9}, hard_ok=False)
    # plans passing all hard goals come first, also as the alternative
    picked = select_distinct_plans([slightly_worse, failing, other_tradeoff, best])
    assert picked == [best, other_tradeoff]
    assert select_distinct_plans([slightly_worse, failing, best]) == [best, failing]
    assert select_distinct_plans([best, slightly_worse]) == [best]
    assert select_distinct_plans([]) == []
    # a third plan must beat both picks on some goal
    plans = [slightly_worse, failing, other_tradeoff, best]
    assert select_distinct_plans(plans, n=3) == [best, other_tradeoff, failing]
    assert select_distinct_plans(plans, n=1) == [best]


# ---- agent wiring ----------------------------------------------------------------------


def test_agent_can_call_new_tools(hn_state):
    script = [
        [
            ("list_constraint_tables", {}),
            ("map_structures", {"aliases": {"SpinalCord": "Myelon"}}),
        ],
        [
            (
                "get_site_constraints",
                {"table": "demo", "site": "head_neck", "apply": True},
            ),
            ("plan_versions", {"action": "save", "label": "start"}),
        ],
        [("analyze_target_shape", {}), ("check_feasibility", {})],
        [
            (
                "create_helper_structure",
                {
                    "recipe": {
                        "kind": "ring",
                        "name": "ring",
                        "target": "PTV",
                        "outer_mm": 15,
                        "exclude": [{"voi": "SpinalCord", "margin_mm": 3}],
                        "skip_concavity": True,
                    }
                },
            )
        ],
        "final",
    ]
    returns = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        returns.extend(p for p in messages[-1].parts if isinstance(p, ToolReturnPart))
        step = script[len([m for m in messages if m.kind == "response"])]
        if step == "final":
            report = PlanningReport(all_hard_goals_met=False, summary="x").model_dump()
            return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])
        return ModelResponse(parts=[ToolCallPart(n, a) for n, a in step])

    create_planning_agent(FunctionModel(fn)).run_sync("plan", deps=hn_state)
    by_tool = {r.tool_name: r.content for r in returns}
    assert by_tool["plan_versions"]["saved"] == "start"
    assert by_tool["get_site_constraints"]["applied"] is True
    assert by_tool["create_helper_structure"]["name"] == "opt_ring"
    assert "PTV_70" in by_tool["analyze_target_shape"]
    assert "opt_ring" in hn_state.created_helpers

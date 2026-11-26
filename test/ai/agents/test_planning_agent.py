"""Tests for the tool-using planning agent.

Three layers, none needs an API key:
1. pure logic (goals, protocol evaluation, error paths) - fast
2. tools on a real TG119 dose calculation - slower, shared module fixture
3. the agent loop driven by a scripted FunctionModel - checks tool wiring,
   ModelRetry self-correction and the final report
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import DeferredToolRequests, models
from pydantic_ai.messages import ModelResponse, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pyRadPlan import load_tg119
from pyRadPlan.core import ComputeCancelledError, ComputeControl
from pyRadPlan.ai.agents._planning import (
    Goal,
    PlanningError,
    PlanningReport,
    PlanningState,
    PlanRecord,
    Protocol,
    UnknownVoiError,
    continue_planning_agent,
    create_planning_agent,
    evaluate_protocol,
    planning_tools,
)
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing
from pyRadPlan.plan import PhotonPlan

models.ALLOW_MODEL_REQUESTS = False  # guarantee no real LLM is ever called here

PROTOCOL = Protocol(
    goals=[
        Goal(voi="OuterTarget", metric="D95", op=">=", value=1.8),
        Goal(voi="Core", metric="max", op="<=", value=1.6),
        Goal(voi="Core", metric="V1Gy", op="<=", value=30, hard=False),
    ]
)


@pytest.fixture
def state():
    ct, cst = load_tg119()
    return PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)


# ---- 1. pure logic -------------------------------------------------------------------


def test_goal_rejects_unknown_metric():
    with pytest.raises(ValueError):
        Goal(voi="x", metric="D95%", op=">=", value=1)


def test_protocol_reference_values():
    assert PROTOCOL.ref_vols() == [95.0]
    assert PROTOCOL.ref_doses() == [1.0]


def test_evaluate_protocol_margins():
    qi = lambda v: SimpleNamespace(value=v)
    qis = {"OuterTarget": {"D95": qi(1.7)}, "Core": {"max": qi(1.0), "V1Gy": qi(10.0)}}
    report = evaluate_protocol(PROTOCOL, qis)
    assert report["all_hard_passed"] is False
    assert report["num_failed"] == 1
    target = report["goals"][0]
    assert target["passed"] is False and target["margin"] == pytest.approx(-0.1)
    assert report["goals"][1]["margin"] == pytest.approx(0.6)


def test_error_paths_without_dose_calc(state):
    with pytest.raises(UnknownVoiError, match="Valid VOIs"):
        state.set_objectives("Rectum", [])
    with pytest.raises(PlanningError, match="No dose"):
        state.analyze_qis()
    for voi in state.cst.vois:  # TG119 ships with default objectives
        state.set_objectives(voi.name, [])
    with pytest.raises(PlanningError, match="No objectives"):
        state.optimize()
    with pytest.raises(PlanningError, match="call set_beams first"):
        state.beam_influence()
    assert state.num_dij_calcs == 0
    state.set_beams([0, 180], None)
    state.max_dij_calcs = 0
    with pytest.raises(PlanningError, match="budget"):
        state.beam_influence()


def test_beam_limits(state):
    state.min_beams, state.max_beams, state.allow_couch = 3, 5, False
    with pytest.raises(PlanningError, match="use 3 to 5"):
        state.set_beams([0, 180], None)
    with pytest.raises(PlanningError, match="use 3 to 5"):
        state.set_beams([0, 60, 120, 180, 240, 300], None)
    with pytest.raises(PlanningError, match="coplanar"):
        state.set_beams([0, 120, 240], [0, 90, 0])
    state.set_beams([0, 120, 240], [0, 0, 0])
    budget = state.case_overview()["budget"]
    assert budget["beams"] == "3-5, coplanar only (couch 0)" and budget["max_iter"] == 5000
    assert budget["optimizations_left"] == state.max_optimizations - state.num_optimizations
    state.num_dij_calcs = state.max_dij_calcs
    with pytest.raises(PlanningError, match="beams cannot change"):
        state.set_beams([0, 120, 240], None)


@pytest.fixture(scope="module")
def pooled():
    ct, cst = load_tg119()
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)
    s.set_beam_pool([0, 72, 144, 216, 288])
    s.max_dij_calcs = s.num_dij_calcs
    for voi in s.cst.vois:
        s.set_objectives(voi.name, [])
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    s.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=1.0)])
    return s


def test_candidate_beams_need_no_dose_calculation(pooled):
    out = pooled.set_beams([144, 0], None)
    assert out["from_candidates"] and pooled.num_dij_calcs == 1
    assert pooled.dij.num_of_beams == 2 and len(pooled.stf.beams) == 2
    # the subset matches a fresh calculation of the same beams (the engine gives a beam a
    # slightly different dose influence, < 0.2 % of the max, once others precede it)
    ct, cst = load_tg119()
    fresh = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)
    fresh.set_beams([144, 0], None)
    a, b = pooled.dij.physical_dose.flat[0], fresh._ensure_dij().physical_dose.flat[0]
    assert a.shape == b.shape and abs(a - b).max() < 5e-3 * abs(b).max()
    pooled.optimize(max_iter=60)
    assert pooled.num_dij_calcs == 1
    assert pooled.case_overview()["budget"]["beam_candidates"].startswith("5 beams")
    with pytest.raises(PlanningError, match="candidate beams"):
        pooled.set_beams([10, 100], None)
    with pytest.raises(PlanningError, match="twice"):
        pooled.set_beams([0, 0], None)


def test_beam_influence_reports_every_candidate(pooled):
    pooled.set_beams([72, 216], None)
    beams = pooled.beam_influence()["beams"]
    assert [b["gantry"] for b in beams] == [0.0, 72.0, 144.0, 216.0, 288.0]
    assert [b["current"] for b in beams] == [False, True, False, True, False]
    for b in beams:
        assert set(b["crossing_pct"]) <= {"Core"} and 0 <= b["clean_pct"] <= 100
    # the Core sits in the C of the target: every beam passes through it somewhere
    assert all(b["crossing_pct"].get("Core", 0) > 0 for b in beams)


def test_beams_lock_after_some_optimizations(pooled):
    pooled.lock_beams_after = pooled.num_optimizations
    assert pooled.case_overview()["budget"]["beam_changes"] == "locked"
    with pytest.raises(PlanningError, match="final"):
        pooled.set_beams([0, 144], None)
    pooled.lock_beams_after = None


def test_beam_setups_are_compared_then_one_becomes_final():
    ct, cst = load_tg119()
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)
    s.set_beam_pool([0, 72, 144, 216, 288])
    s.max_dij_calcs = s.num_dij_calcs
    s.max_beam_setups, s.trial_optimizations_per_setup = 2, 1
    for voi in s.cst.vois:
        s.set_objectives(voi.name, [])
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    with pytest.raises(PlanningError, match="No beams set"):
        s.optimize(max_iter=50)
    s.set_beams([0, 144], None)
    s.optimize(max_iter=50)
    s.set_beams([72, 216], None)
    s.optimize(max_iter=50)
    with pytest.raises(PlanningError, match="pick one of them"):
        s.set_beams([0, 288], None)
    s.set_beams([0, 144], None)  # a compared setup: allowed
    s.optimize(max_iter=60)  # its second optimization makes it final
    assert s.case_overview()["budget"]["beam_setups"]["final"] == "0/144"
    with pytest.raises(PlanningError, match="final"):
        s.set_beams([72, 216], None)


def test_setups_must_be_compared_before_one_becomes_final():
    ct, cst = load_tg119()
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)
    s.set_beam_pool([0, 72, 144, 216, 288])
    s.max_dij_calcs = s.num_dij_calcs
    s.min_beams = s.max_beams = 3
    s.max_beam_setups = s.min_beam_setups = 2
    s.trial_optimizations_per_setup, s.min_setup_difference = 1, 2
    for voi in s.cst.vois:
        s.set_objectives(voi.name, [])
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    s.set_beams([0, 72, 144], None)
    s.optimize(max_iter=40)
    with pytest.raises(PlanningError, match="2 setups need 1 optimizations each"):
        s.optimize(max_iter=50)
    with pytest.raises(PlanningError, match="at least 2 beams"):
        s.set_beams([0, 72, 216], None)
    s.set_beams([0, 216, 288], None)
    s.optimize(max_iter=40)
    s.set_beams([0, 72, 144], None)
    s.optimize(max_iter=50)  # both compared: this one makes it final
    assert s.case_overview()["budget"]["beam_setups"]["final"] == "0/72/144"


def test_excluding_structures_keeps_the_dose_influence(state):
    state.set_beams([0, 120, 240], None)
    dij = state._ensure_dij()
    state.map_structures(exclude=["BODY"])  # the targets stay the same
    assert state.dij is dij and state.num_dij_calcs == 1
    state.map_structures(voi_types={"Core": "TARGET"})  # the targets grow
    assert state.dij is None
    dij = state._ensure_dij()
    state.map_structures(exclude=["Core"])  # they shrink: the beamlets still cover them
    assert state.dij is dij and state.num_dij_calcs == 2


def test_optimize_caps_the_iterations(state):
    state.max_iter = 60
    state.set_beams([0, 120, 240], None)
    state.optimize(max_iter=500, min_improvement=0)
    assert state.history[-1]["max_iter"] == 60


def test_set_objectives_does_not_touch_other_vois(state):
    before = state.cst.vois[0].objectives
    state.set_objectives("OuterTarget", [SquaredDeviation(priority=10, d_ref=2.0)])
    assert state.cst.vois[0].objectives == before
    assert state.current_objectives()["OuterTarget"][0]["name"] == "Squared Deviation"


# ---- 2. tools on real dose -----------------------------------------------------------


@pytest.fixture(scope="module")
def planned_state():
    ct, cst = load_tg119()
    s = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=PROTOCOL)
    s.set_beams([0, 120, 240], None)
    s.set_objectives("OuterTarget", [SquaredDeviation(priority=100, d_ref=2.0)])
    s.set_objectives("Core", [SquaredOverdosing(priority=10, d_max=1.0)])
    s.influence = s.beam_influence()
    s.evaluation = s.optimize()
    return s


def test_beam_influence_per_beam_and_oar(planned_state):
    beams = planned_state.influence["beams"]
    assert [b["gantry"] for b in beams] == [0.0, 120.0, 240.0]
    for b in beams:
        ratios = b["rel_mean_dose_vs_target"]
        assert set(ratios) == {"Core", "BODY"}  # targets excluded
        assert all(0 < r < 2 for r in ratios.values())


def test_optimize_evaluates_every_goal(planned_state):
    ev = planned_state.evaluation
    assert len(ev["goals"]) == len(PROTOCOL.goals) + 1  # + implicit NormalTissue goal
    assert planned_state.dij is not None and planned_state.num_dij_calcs == 1
    # plausible plan: target near prescription, Core spared relative to target
    qis = planned_state.analyze_qis(["OuterTarget", "Core"])
    assert 1.5 < qis["OuterTarget"]["mean"] < 2.5
    assert qis["Core"]["mean"] < qis["OuterTarget"]["mean"]


def test_objective_change_reuses_dij(planned_state):
    planned_state.set_objectives("Core", [SquaredOverdosing(priority=50, d_max=0.8)])
    planned_state.optimize()
    assert planned_state.num_dij_calcs == 1  # no recomputation for objective tuning


# ---- 3. agent loop with a scripted model ---------------------------------------------


def _record(state, evaluation: dict) -> dict:
    """Count an optimization and keep its plan, as PlanningState.optimize does."""
    state.num_optimizations += 1
    state.plans.append(PlanRecord(index=len(state.plans) + 1, evaluation=evaluation))
    return evaluation


def _scripted_model(script):
    """FunctionModel replaying *script*: one list of (tool, args) per model turn."""
    seen = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        seen.append(messages[-1])
        model.last_history = messages
        step = script[len(seen) - 1]
        if step == "final":
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        info.output_tools[0].name,
                        PlanningReport(all_hard_goals_met=True, summary="done").model_dump(),
                    )
                ]
            )
        return ModelResponse(parts=[ToolCallPart(name, args) for name, args in step])

    model = FunctionModel(fn)
    return model, seen


def test_agent_loop_calls_tools(state, monkeypatch):
    # replace the expensive physics; this test is about orchestration
    monkeypatch.setattr(PlanningState, "beam_influence", lambda self: {"beams": []})
    monkeypatch.setattr(
        PlanningState,
        "optimize",
        lambda self, max_iter=500, min_improvement=1e-2, start_from=None: _record(
            self, {"all_hard_passed": True, "goals": []}
        ),
    )
    sd = {"name": "Squared Deviation", "priority": 100, "d_ref": 2.0}
    model, seen = _scripted_model(
        [
            [("case_overview", {})],
            [("set_beams", {"gantry_angles": [0, 90, 180, 270]}), ("beam_influence", {})],
            [("set_objectives", {"voi_name": "Rectum", "objectives": [sd]})],  # wrong VOI
            [("set_objectives", {"voi_name": "OuterTarget", "objectives": [sd]})],
            [("optimize", {})],
            "final",
        ]
    )
    result = create_planning_agent(model).run_sync("plan TG119", deps=state)

    assert result.output.all_hard_goals_met
    assert state.current_beams()["gantry_angles"] == [0, 90, 180, 270]
    assert state.current_objectives()["OuterTarget"][0]["parameters"]["d_ref"] == 2.0
    assert [h["action"] for h in state.history] == ["set_beams", "set_objectives"]


def test_tools_run_sequentially(state):
    # Parallel calls raced on the shared state (a snapshot saw the dose cleared midway)
    seen = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        seen.extend(info.function_tools)
        report = PlanningReport(all_hard_goals_met=False, summary="x").model_dump()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    create_planning_agent(FunctionModel(fn)).run_sync("plan", deps=state)
    assert seen and all(t.sequential for t in seen)


def test_unexpected_tool_failure_reaches_the_model(state, monkeypatch):
    def broken(self, *args, **kwargs):
        raise TypeError("float() argument must be a real number, not 'NoneType'")

    monkeypatch.setattr(PlanningState, "case_overview", broken)
    returns = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        returns.extend(p for p in messages[-1].parts if isinstance(p, ToolReturnPart))
        if not returns:
            return ModelResponse(parts=[ToolCallPart("case_overview", {})])
        report = PlanningReport(all_hard_goals_met=False, summary="x").model_dump()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    result = create_planning_agent(FunctionModel(fn)).run_sync("plan", deps=state)
    assert result.output.summary == "x"  # the run went on
    assert returns[0].content["error"].startswith("TypeError: float()")


def test_agent_tools_can_be_limited(state):
    offered = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        offered.extend(t.name for t in info.function_tools)
        report = PlanningReport(all_hard_goals_met=False, summary="x").model_dump()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    tools = ["case_overview", "set_objectives", "optimize"]
    create_planning_agent(FunctionModel(fn), tools=tools).run_sync("plan", deps=state)
    assert sorted(offered) == sorted(tools)
    with pytest.raises(ValueError, match="no_such_tool"):
        create_planning_agent(FunctionModel(fn), tools=["no_such_tool"])
    assert "set_beams" in planning_tools()


def test_listener_sees_tool_calls_and_stop_ends_the_run(state):
    events = []
    state.control = ComputeControl()
    model, _ = _scripted_model([[("case_overview", {})], [("case_overview", {})], "final"])

    def stop_after_first_call(event, **data):
        events.append((event, data.get("name")))
        if event == "tool_done":
            state.control.request_stop()

    state.listener = stop_after_first_call
    with pytest.raises(ComputeCancelledError):
        create_planning_agent(model).run_sync("plan", deps=state)
    assert events == [("tool", "case_overview"), ("tool_done", "case_overview")]


def test_agent_optimization_budget(state, monkeypatch):
    calls = []

    def fake_optimize(self, max_iter=500, min_improvement=1e-2, start_from=None):
        calls.append(1)
        return _record(self, {"all_hard_passed": False, "goals": []})

    monkeypatch.setattr(PlanningState, "optimize", fake_optimize)
    model, seen = _scripted_model([[("optimize", {})], [("optimize", {})], "final"])
    state.max_optimizations = 1
    create_planning_agent(model).run_sync("x", deps=state)
    assert len(calls) == 1
    assert "budget" in str(seen[2].parts[0].content)


def _offered_tools(state) -> dict:
    offered = {}

    def fn(messages, info: AgentInfo) -> ModelResponse:
        offered.update({t.name: t for t in info.function_tools})
        report = PlanningReport(all_hard_goals_met=False, summary="x").model_dump()
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, report)])

    create_planning_agent(FunctionModel(fn)).run_sync("plan", deps=state)
    return offered


def test_objective_schema_sent_to_the_model_is_slim(state):
    schema = _offered_tools(state)["set_objectives"].parameters_json_schema
    assert "discriminator" not in str(schema)
    for d in schema["$defs"].values():
        assert not {"quantity", "robustness"} & set(d["properties"])
    sd = schema["$defs"]["SquaredDeviation"]
    assert sd["required"][0] == "name" and "title" not in sd
    assert sd["properties"]["penalty"]["description"] == "Relative weight."


def test_objectives_from_the_slim_schema_validate(state, monkeypatch):
    sd = {"name": "Squared Deviation", "penalty": 7, "d_ref": 2.0}
    model, _ = _scripted_model(
        [[("set_objectives", {"voi_name": "OuterTarget", "objectives": [sd]})], "final"]
    )
    create_planning_agent(model).run_sync("x", deps=state)
    obj = state.cst.vois[1].objectives[0]
    assert obj.priority == 7 and obj.quantity == "physical_dose" and obj.robustness == "none"


def test_geometry_is_its_own_tool(state):
    assert "geometry" not in state.case_overview()
    assert "structure_geometry" in planning_tools()
    vois = state.structure_geometry()["vois"]
    assert [v["name"] for v in vois] == state.voi_names


def test_model_sees_goal_lines(state, monkeypatch):
    ev = {
        "all_hard_passed": False,
        "unresolved_structures": ["PTV"],
        "unresolved_goals": ["PTV D95 >= 1.8"],
        "goals": [
            {
                "goal": "Core max <= 1.6",
                "hard": True,
                "value": 1.7,
                "margin": -0.1,
                "rel_margin": -0.0625,
                "passed": False,
            },
        ],
    }

    def fake_optimize(self, max_iter=500, min_improvement=1e-2, start_from=None):
        return _record(self, ev)

    monkeypatch.setattr(PlanningState, "optimize", fake_optimize)
    model, seen = _scripted_model([[("optimize", {})], "final"])
    create_planning_agent(model).run_sync("x", deps=state)
    content = seen[1].parts[0].content
    assert content["plan"] == 1 and "unresolved_structures" not in content
    assert content["goals"] == ["Core max <= 1.6 [hard]: 1.7, margin -0.1 (-6.2 %), FAILED"]
    assert content["unresolved_goals"] == ["PTV D95 >= 1.8"] and "map" in content["note"]
    assert ev["goals"][0]["margin"] == -0.1  # the state keeps the full rows


def test_history_is_compacted_every_n_optimizations(state, monkeypatch):
    def fake_optimize(self, max_iter=500, min_improvement=1e-2, start_from=None):
        ev = {"all_hard_passed": True, "score": 1.0, "goals": [], "optimizer": {"x": 1}}
        return _record(self, ev)

    monkeypatch.setattr(PlanningState, "optimize", fake_optimize)
    events = []
    state.listener = lambda event, **data: events.append(event)
    model, seen = _scripted_model(
        [
            [("inspect_structures", {}), ("case_overview", {})],
            [("structure_geometry", {})],
            [("optimize", {})],
            [("set_beams", {"gantry_angles": [0, 180]})],
            [("optimize", {})],
            "final",
        ]
    )
    create_planning_agent(model, compact_after=2).run_sync("x", deps=state)
    assert events.count("compact") == 1 and state.num_compactions == 1

    # the final request saw the compacted history
    history = {
        p.tool_name: p.content
        for m in model.last_history
        for p in getattr(m, "parts", [])
        if isinstance(p, ToolReturnPart)
    }
    assert history["case_overview"].startswith("[Compacted")
    assert isinstance(history["structure_geometry"], dict)  # anatomy is kept
    assert isinstance(history["set_beams"], dict)  # small results are kept
    assert history["optimize"]["optimizer"] == {"x": 1}  # the latest optimization is kept


# ---- structure identification ----------------------------------------------------------


@pytest.fixture
def messy_state():
    """TG119 renamed like a real clinical export: German names, PTV with dose suffix,
    an old helper copy and a near-duplicate.
    """
    ct, cst = load_tg119()
    core, target, body = cst.vois
    vois = [
        core.model_copy(update={"name": "Myelon"}),
        target.model_copy(update={"name": "PTV_60"}),
        body.model_copy(update={"name": "Aussenkontur"}),
        core.model_copy(update={"name": "z_Myelon_PRV_old"}),
        core.model_copy(update={"name": "Myelon "}),
    ]
    cst = cst.model_copy(update={"vois": vois})
    protocol = Protocol(
        goals=[
            Goal(voi="PTV", metric="D95", op=">=", value=1.8),
            Goal(voi="SpinalCord", metric="max", op="<=", value=1.6),
        ]
    )
    return PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=protocol)


def test_inspect_structures_hints(messy_state):
    info = messy_state.inspect_structures()
    rows = {r["name"]: r for r in info["structures"]}
    assert rows["Aussenkontur"]["pct_inside_target"]["PTV_60"] > 0
    assert info["unmapped_protocol_names"] == ["PTV", "SpinalCord"]


def test_map_structures_and_evaluate_with_aliases(messy_state):
    with pytest.raises(UnknownVoiError):
        messy_state.map_structures(aliases={"PTV": "PTV60"})
    out = messy_state.map_structures(
        aliases={"PTV": "PTV_60", "SpinalCord": "Myelon"},
        voi_types={"Aussenkontur": "EXTERNAL"},
        exclude=["z_Myelon_PRV_old", "Myelon "],
    )
    assert ("Aussenkontur", "EXTERNAL") in out["vois"]
    assert len(out["vois"]) == 3
    assert messy_state.inspect_structures()["unmapped_protocol_names"] == []

    qi = lambda v: SimpleNamespace(value=v)
    qis = {"PTV_60": {"D95": qi(1.9)}, "Myelon": {"max": qi(1.0)}}
    ev = evaluate_protocol(messy_state.protocol, qis, messy_state.aliases)
    assert ev["all_hard_passed"] and ev["unresolved_structures"] == []
    unmapped = evaluate_protocol(messy_state.protocol, qis)
    assert unmapped["unresolved_structures"] == ["PTV", "SpinalCord"]
    # goals on missing structures are not evaluated and do not count in the score
    assert unmapped["goals"] == [] and unmapped["num_failed"] == 0
    assert len(unmapped["unresolved_goals"]) == len(messy_state.protocol.goals)


def test_cannot_exclude_mapped_structure(messy_state):
    messy_state.map_structures(aliases={"SpinalCord": "Myelon"})
    with pytest.raises(PlanningError, match="mapped"):
        messy_state.map_structures(exclude=["Myelon"])


def test_refused_mapping_changes_nothing(messy_state):
    with pytest.raises(PlanningError, match="TARGET"):
        messy_state.map_structures(aliases={"SpinalCord": "Myelon"}, exclude=["PTV_60"])
    assert messy_state.aliases == {} and messy_state.excluded == []


def test_snapshot_restore(state):
    state.set_objectives("Core", [SquaredOverdosing(priority=1, d_max=1.0)])
    state.snapshot("v1")
    state.set_objectives("Core", [])
    out = state.restore("v1")
    assert state.current_objectives()["Core"][0]["penalty"] == 1
    assert out["plan"] is None and "call optimize" in out["note"]


# ---- 4. planner reviews --------------------------------------------------------------


def _fake_optimize(monkeypatch, passed=False):
    def optimize(self, max_iter=500, min_improvement=1e-2, start_from=None):
        return _record(self, {"all_hard_passed": passed, "goals": []})

    monkeypatch.setattr(PlanningState, "optimize", optimize)


def _returned(message, tool: str) -> str:
    return " ".join(
        str(p.content)
        for p in message.parts
        if isinstance(p, ToolReturnPart) and p.tool_name == tool
    )


def test_review_pauses_and_continue_passes_feedback(state, monkeypatch):
    _fake_optimize(monkeypatch)
    model, seen = _scripted_model([[("optimize", {})], [("optimize", {})], "final"])
    state.review_every = 1
    paused = create_planning_agent(model).run_sync("x", deps=state)

    assert isinstance(paused.output, DeferredToolRequests)
    (call,) = paused.output.approvals
    assert paused.output.metadata[call.tool_call_id] == {"reason": "review"}
    assert state.num_optimizations == 1

    done = continue_planning_agent(state, paused, "Spare the rectum more.", model=model)
    assert done.output.all_hard_goals_met and state.num_optimizations == 2
    assert "Spare the rectum more." in _returned(seen[2], "optimize")
    assert state.feedback is None
    assert done.usage.requests > paused.usage.requests  # usage adds up over the calls


def test_feedback_reaches_the_model_when_the_optimize_is_refused(state, monkeypatch):
    def refuse(self, max_iter=500, min_improvement=1e-2, start_from=None):
        raise PlanningError("Plan 1 was optimized with exactly these objectives.")

    _fake_optimize(monkeypatch)
    model, seen = _scripted_model([[("optimize", {})], [("optimize", {})], "final"])
    state.review_every = 1
    paused = create_planning_agent(model).run_sync("x", deps=state)
    monkeypatch.setattr(PlanningState, "optimize", refuse)
    continue_planning_agent(state, paused, "Spare the rectum more.", model=model)

    sent = " ".join(str(getattr(p, "content", "")) for p in seen[2].parts)
    assert "exactly these objectives" in sent and "Spare the rectum more." in sent
    assert state.feedback is None


def test_redirect_skips_the_pending_call(state, monkeypatch):
    _fake_optimize(monkeypatch)
    model, seen = _scripted_model(
        [[("set_beams", {"gantry_angles": [0, 180]})], [("optimize", {})], "final"]
    )
    state.pause_requested = True
    paused = create_planning_agent(model).run_sync("x", deps=state)
    assert paused.output.metadata[paused.output.approvals[0].tool_call_id]["reason"] == "pause"

    continue_planning_agent(state, paused, "Use five beams.", action="redirect", model=model)
    assert "planner reviewed the plans and says: Use five beams." in _returned(
        seen[1], "set_beams"
    )
    assert state.current_beams()["gantry_angles"] != [0, 180]
    assert state.num_optimizations == 1 and not state.pause_requested


def test_finish_asks_for_the_report(state, monkeypatch):
    _fake_optimize(monkeypatch)
    model, seen = _scripted_model([[("optimize", {})], [("optimize", {})], "final"])
    state.pause_requested = True
    paused = create_planning_agent(model).run_sync("x", deps=state)
    continue_planning_agent(state, paused, action="finish", model=model)
    assert "ends the run" in _returned(seen[1], "optimize")
    assert "planner ended the run" in _returned(seen[2], "optimize")
    assert state.num_optimizations == 0


def test_continue_drops_calls_of_disabled_tools(state, monkeypatch):
    model, seen = _scripted_model([[("set_beams", {"gantry_angles": [0, 180]})], "final"])
    state.pause_requested = True
    paused = create_planning_agent(model).run_sync("x", deps=state)
    continue_planning_agent(state, paused, model=model, tools=["optimize", "set_objectives"])
    assert "disabled" in _returned(seen[1], "set_beams")


def test_budget_and_hard_goal_reviews(state, monkeypatch):
    _fake_optimize(monkeypatch, passed=True)
    model, _ = _scripted_model([[("optimize", {})]] * 4 + ["final"])
    state.review_every, state.max_optimizations = 10, 1
    out = create_planning_agent(model).run_sync("x", deps=state).output
    assert out.metadata[out.approvals[0].tool_call_id]["reason"] == "budget"

    state.mark_reviewed()
    assert state.review_reason("optimize") is None  # budget seen, not raised again
    state.max_optimizations = 2
    state._reviewed_hard_pass = False
    assert state.review_reason("optimize") == "hard_goals"
    state.mark_reviewed()
    state.num_optimizations, state.max_optimizations = 11, 5
    assert state.review_reason("optimize") == "budget"
    state.max_optimizations = 20
    assert state.review_reason("optimize") == "review"
    assert state.review_reason("set_beams") is None  # only a pause stops set_beams


def test_continue_after_the_report(state):
    model, seen = _scripted_model(["final", "final"])
    done = create_planning_agent(model).run_sync("x", deps=state)
    with pytest.raises(ValueError, match="prompt"):
        continue_planning_agent(state, done, model=model)
    again = continue_planning_agent(state, done, "Try seven beams.", model=model)
    assert isinstance(again.output, PlanningReport)
    prompt = [p for p in seen[1].parts if isinstance(p, UserPromptPart)]
    assert prompt and prompt[0].content == "Try seven beams."


def test_changed_goals_reevaluate_plans_and_reach_the_agent(planned_state):
    s = planned_state
    before = s.protocol
    goal = Goal(voi="Core", metric="mean", op="<=", value=0.1, hard=False)
    try:
        changes = s.set_protocol(before.model_copy(update={"goals": [*before.goals, goal]}))
        assert changes == ["added Core mean <= 0.1 (soft)"]
        labels = [g["goal"] for g in s.plans[0].evaluation["goals"]]
        assert "Core mean <= 0.1" in labels
        assert "normal_tissue" in s.plans[0].evaluation  # kept from the optimization
        s.snapshot("goal_change")
        assert s.set_protocol(before) == ["removed Core mean <= 0.1 (soft)"]
        labels = [g["goal"] for g in s.snapshots["goal_change"]["evaluation"]["goals"]]
        assert "Core mean <= 0.1" not in labels  # saved versions are re-evaluated too
    finally:
        s.snapshots.pop("goal_change", None)
        s.set_protocol(before)


def test_continue_reports_changed_goals(state):
    model, seen = _scripted_model(["final", "final"])
    done = create_planning_agent(model).run_sync("x", deps=state)
    goal = Goal(voi="Core", metric="max", op="<=", value=1.2)
    protocol = PROTOCOL.model_copy(update={"goals": [goal]})
    continue_planning_agent(state, done, model=model, protocol=protocol)
    prompt = next(p for p in seen[1].parts if isinstance(p, UserPromptPart)).content
    assert "changed the goals" in prompt and "added Core max <= 1.2 (hard)" in prompt
    assert state.protocol is protocol

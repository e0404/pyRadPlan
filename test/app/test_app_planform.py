import json

import pytest

from pyRadPlan.app import Session, SessionManager, dispatch
from pyRadPlan.app._profiles import WEB_DEFAULT_PLAN


@pytest.fixture
def web_client(client):
    """The shared client fixture, but with a web-profile session."""
    client.session = client.ok("session.create", {"profile": "web"}, session=False)["result"][
        "session"
    ]
    client.ok("data.load_phantom", {"name": "TG119"})
    return client


def test_profiles_give_the_default_plan():
    native = Session(profile="native")
    web = Session(profile="web")
    assert native.default_plan["dose_grid_resolution"] == [3.0, 3.0, 3.0]
    assert web.default_plan == WEB_DEFAULT_PLAN
    assert web.default_plan["dose_grid_resolution"] == [5.0, 5.0, 5.0]
    with pytest.raises(ValueError, match="Unknown profile"):
        Session(profile="cloud")


def test_web_profile_applies_5mm_and_total(web_client):
    detail = web_client.ok("plan.get", {"detail": True})["result"]
    assert detail["dose_grid_resolution"] == [5.0, 5.0, 5.0]
    assert detail["dose_convention"] == "total"
    assert detail["machine_source"] == "name"
    assert len(detail["resolved_iso_center"]) == 3
    state = web_client.ok("session.state")["result"]
    assert state["options"]["profile"] == "web"


def test_locked_fields_are_rejected():
    manager = SessionManager()
    options = {"locked_fields": ["dose_grid_resolution"]}
    created = dispatch(
        {"id": 1, "method": "session.create", "params": {"profile": "web", "options": options}},
        manager=manager,
    )
    sid = created["result"]["session"]
    assert created["result"]["options"]["locked_fields"] == ["dose_grid_resolution"]

    def call(method, params):
        request = {"id": 2, "session": sid, "method": method, "params": params}
        return dispatch(request, manager=manager)

    assert call("data.load_phantom", {"name": "TG119"})["ok"]
    refused = call("plan.set", {"dose_grid_resolution": [3, 3, 3]})
    assert refused["error"]["code"] == "locked"
    assert call("plan.set", {"dose_grid_resolution": [5, 5, 5], "num_of_fractions": 3})["ok"]
    fields = {f["name"]: f for f in call("plan.options", {})["result"]["fields"]}
    assert fields["dose_grid_resolution"]["editable"] is False
    manager.close_all()


def test_plan_options_list_the_registries(web_client):
    options = web_client.ok("plan.options")["result"]
    assert options["radiation_mode"] == "photons"
    assert [c["value"] for c in options["engines"]] == ["SVDPB", "CCC"]
    assert "Generic" in [c["value"] for c in options["machines"]]
    assert json.loads(options["default_bio_model"])["model"] == "none"
    assert {"nomScen", "wcScen"} <= {c["value"] for c in options["scenario_models"]}
    assert "scipy" in [c["value"] for c in options["solvers"]]
    assert options["beam_presets"] and options["beam_presets"][0]["typical"]
    fields = {f["name"]: f for f in options["fields"]}
    assert fields["num_of_fractions"]["min"] == 1 and fields["num_of_fractions"]["advanced"]
    assert fields["run_sequencing"]["editable"] is False

    protons = web_client.ok("plan.options", {"radiation_mode": "protons"})["result"]
    assert json.loads(protons["default_bio_model"])["model"] == "constant_rbe"
    assert "HongPB" in [c["value"] for c in protons["engines"]]


@pytest.mark.parametrize(
    ("kind", "name", "field"),
    [
        ("engine", "SVDPB", "enable_dij_sampling"),
        ("problem", "nonlin_fluence", "p_norm"),
        ("solver", "scipy", "method"),
        ("scenario", "wcScen", "setup"),
    ],
)
def test_algorithm_forms(web_client, kind, name, field):
    form = web_client.ok("plan.form", {"kind": kind, "name": name})["result"]
    assert form["id"] == f"{kind}:{name}" and form["title"].startswith("Configure ")
    assert field in {f["name"] for f in form["fields"]}


def test_problem_form_hides_solver_settings(web_client):
    form = web_client.ok("plan.form", {"kind": "problem", "name": "nonlin_fluence"})["result"]
    names = {f["name"] for f in form["fields"]}
    assert not names & {"solver", "convert_dose_objectives", "max_iter", "display"}


def test_bio_model_form_of_another_mode(web_client):
    params = {"kind": "bio_model", "name": "constant_rbe", "radiation_mode": "protons"}
    form = web_client.ok("plan.form", params)["result"]
    assert [f["name"] for f in form["fields"]] == ["rbe"]


def test_plan_set_v1_round_trip(web_client):
    changes = {
        "dose_convention": "per_fraction",
        "engine_config": {"values": {"enable_dij_sampling": False}},
        "scenarios": {"model": "wcScen", "values": {"wc_sigma": 2.0}},
        "problem": {"name": "nonlin_fluence", "values": {"p_norm": 10.0}},
        "solver": {"name": "scipy", "values": {"max_iter": 20}},
    }
    web_client.ok("plan.set", changes)
    detail = web_client.ok("plan.get", {"detail": True})["result"]
    assert detail["dose_convention"] == "per_fraction"
    assert detail["engine_config"]["values"] == {"enable_dij_sampling": False}
    assert detail["scenario"] == {"model": "wcScen", "values": {"wc_sigma": 2.0}}
    assert detail["problem"]["values"] == {"p_norm": 10.0}
    assert detail["solver"] == {"name": "scipy", "values": {"max_iter": 20}}
    assert detail["dose_grid_resolution"] == [5.0, 5.0, 5.0]

    solver_form = web_client.ok("plan.form", {"kind": "solver", "name": "scipy"})["result"]
    assert solver_form["values"]["max_iter"] == 20 and "max_iter" in solver_form["explicit"]

    session = web_client.manager.get(web_client.session)
    instance = session.workspace.pln.scenarios
    web_client.ok("plan.set", {"scenarios": {"model": "wcScen", "values": {"wc_sigma": 2.0}}})
    assert session.workspace.pln.scenarios is not instance
    assert session.workspace.pln.scenarios.wc_sigma == 2.0


def test_plan_set_validation_details(web_client):
    bad = {"problem": {"name": "nonlin_fluence", "values": {"p_norm": "abc"}}}
    error = web_client.call("plan.set", bad)["error"]
    assert error["type"] == "ValidationError" and error["code"] == "invalid"
    assert error["details"][0]["field"] == "problem.p_norm"
    error = web_client.call("plan.set", {"num_of_fractions": 0})["error"]
    assert error["type"] == "ValidationError" and error["details"]


def test_plan_set_rejects_an_engine_of_another_mode(web_client):
    error = web_client.call("plan.set", {"radiation_mode": "photons", "engine": "HongPB"})["error"]
    assert error["type"] == "ValidationError"
    assert error["details"] == [
        {"field": "engine", "message": "dose engine 'HongPB' is not available for photons"}
    ]
    assert web_client.ok("plan.get", {"detail": True})["result"]["engine"] != "HongPB"
    web_client.ok("plan.set", {"radiation_mode": "protons", "engine": "HongPB"})


def test_bio_model_option_selects_base_data(web_client):
    options = web_client.ok("plan.options", {"radiation_mode": "protons"})["result"]
    option = options["default_bio_model"]
    web_client.ok(
        "plan.set",
        {
            "radiation_mode": "protons",
            "bio_model": {"model": "constant_rbe", "option": option, "values": {"rbe": 1.2}},
        },
    )
    detail = web_client.ok("plan.get", {"detail": True})["result"]
    assert detail["bio"]["model"] == "constant_rbe"
    assert detail["bio"]["values"] == {"rbe": 1.2}
    assert detail["bio"]["option"] == option
    assert web_client.ok("plan.tissue_check")["result"] == {"marker": "", "issues": []}


def test_bio_model_options_say_whether_they_are_configurable(web_client):
    options = web_client.ok("plan.options", {"radiation_mode": "protons"})["result"]
    configurable = {o["identifier"]: o["configurable"] for o in options["bio_models"]}
    assert configurable["none"] is False
    assert configurable["MCN"] is True


def test_comparison_key_tells_values_apart():
    import enum

    import numpy as np

    from pyRadPlan.app._forms import comparison_key

    class Mode(enum.Enum):
        A = "a"

    assert comparison_key({"b": (1, 2), "a": np.float64(1.5)}) == comparison_key(
        {"a": 1.5, "b": [1, 2]}
    )
    assert comparison_key(Mode.A) == comparison_key("a")
    assert comparison_key(float("nan")) != comparison_key(None)
    assert comparison_key({"rbe": 1.0}) == '{"rbe": 1.0}'


def test_changes_from_form_sends_only_what_differs():
    from pyRadPlan.app._planform import changes_from_form, plan_form_values
    from pyRadPlan.optimization.solvers import get_solver
    from pyRadPlan.plan import PhotonPlan

    solver = get_solver("scipy")
    pln = PhotonPlan(
        prop_stf={"gantry_angles": [0.0], "couch_angles": [0.0], "bixel_width": 5.0},
        prop_opt={"solver": solver, "legacy": 1},
    )
    form = plan_form_values(pln)
    assert form["engine"] == "SVDPB" and form["machine"] == "Generic"
    assert changes_from_form(pln, form) == {}
    form["gantry_angles"] = [0.0, 90.0]
    form["couch_angles"] = [0.0, 0.0]
    assert set(changes_from_form(pln, form)) == {"gantry_angles", "couch_angles"}
    form["engine_config"] = {"values": {"kernel_cutoff": 60.0}}
    assert {"engine", "engine_config"} <= set(changes_from_form(pln, form))
    assert changes_from_form(None, form) == form
    assert changes_from_form(pln, {**form, "radiation_mode": "protons"})["engine"] == "SVDPB"

    session = Session(auto_plan=False)
    session.workspace.pln = pln
    session.set_plan(changes_from_form(pln, {**plan_form_values(pln), "num_of_fractions": 5}))
    assert session.workspace.pln.num_of_fractions == 5
    assert session.workspace.pln.prop_opt["solver"] is solver
    assert session.workspace.pln.prop_opt["legacy"] == 1


def test_plan_set_keeps_problem_entries_the_form_does_not_show():
    from pyRadPlan.plan import PhotonPlan

    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(
        prop_opt={"convert_dose_objectives": False, "legacy": 1, "p_norm": 2.0}
    )
    session.set_plan({"problem": {"name": "nonlin_fluence", "values": {"p_norm": 3.0}}})
    assert session.workspace.pln.prop_opt == {
        "p_norm": 3.0,
        "convert_dose_objectives": False,
        "legacy": 1,
    }


def test_problem_settings_of_the_plan_detail_can_be_sent_back_edited():
    from pyRadPlan.plan import PhotonPlan

    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(
        prop_opt={"convert_dose_objectives": False, "legacy": 1, "max_iter": 50}
    )
    values = session.plan_detail()["problem"]["values"]
    assert "convert_dose_objectives" not in values and "legacy" not in values
    session.set_plan({"problem": {"name": "nonlin_fluence", "values": {**values, "p_norm": 3.0}}})
    assert session.workspace.pln.prop_opt == {
        "p_norm": 3.0,
        "max_iter": 50,
        "convert_dose_objectives": False,
        "legacy": 1,
    }


def test_unshown_problem_entries_sent_unchanged_are_kept():
    from pyRadPlan.app._forms import FormValidationError
    from pyRadPlan.plan import PhotonPlan

    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(prop_opt={"convert_dose_objectives": False, "legacy": 1})
    sent = {"convert_dose_objectives": False, "legacy": 1, "p_norm": 3.0}
    session.set_plan({"problem": {"name": "nonlin_fluence", "values": sent}})
    assert session.workspace.pln.prop_opt == sent
    with pytest.raises(FormValidationError, match="convert_dose_objectives"):
        session.set_plan(
            {"problem": {"name": "nonlin_fluence", "values": {"convert_dose_objectives": True}}}
        )


def test_a_mode_change_keeps_scenario_model_and_problem_settings():
    from pyRadPlan.plan import PhotonPlan

    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(
        scenarios={"model": "wcScen", "wc_sigma": 2.0},
        prop_opt={"solver": {"name": "scipy", "max_iter": 7}, "convert_dose_objectives": False},
    )
    session.set_plan({"radiation_mode": "protons"})
    pln = session.workspace.pln
    assert pln.radiation_mode == "protons"
    assert pln.scenarios.short_name == "wcScen" and pln.scenarios.wc_sigma == 2.0
    assert pln.prop_opt == {
        "solver": {"name": "scipy", "max_iter": 7},
        "convert_dose_objectives": False,
    }


def test_a_mode_change_keeps_an_explicit_scenario_set():
    from pyRadPlan.app._planform import changes_from_form, plan_form_values
    from pyRadPlan.plan import PhotonPlan
    from pyRadPlan.scenarios import ScenarioSet, SetupUncertainty

    scenarios = ScenarioSet.from_points(
        dimensions=[SetupUncertainty()], points={"setup": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
    )
    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(scenarios=scenarios)
    session.set_plan({"radiation_mode": "protons"})
    assert session.workspace.pln.scenarios.short_name == "explicit"

    session.workspace.pln = PhotonPlan(scenarios=scenarios)
    form = {**plan_form_values(session.workspace.pln), "radiation_mode": "protons"}
    form.update(engine=None, engine_config={"values": {}}, bio_model="none")
    session.set_plan(changes_from_form(session.workspace.pln, form))
    assert session.workspace.pln.scenarios.short_name == "explicit"


def test_a_mode_change_sending_scenarios_and_problem_uses_them():
    from pyRadPlan.plan import PhotonPlan

    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(
        scenarios={"model": "wcScen"}, prop_opt={"convert_dose_objectives": False}
    )
    session.set_plan(
        {
            "radiation_mode": "protons",
            "scenarios": {"model": "nomScen", "values": {}},
            "problem": {"name": "nonlin_fluence", "values": {"p_norm": 4.0}},
        }
    )
    pln = session.workspace.pln
    assert pln.scenarios.short_name == "nomScen"
    assert pln.prop_opt == {"p_norm": 4.0, "convert_dose_objectives": False}


def test_engine_settings_the_engine_form_does_not_show_are_kept():
    from pyRadPlan.app._planform import changes_from_form, plan_form_values
    from pyRadPlan.plan import PhotonPlan

    hidden = {"select_voxels_in_scenarios": False, "voxel_sub_ix": [1, 2, 3]}
    session = Session(auto_plan=False)
    session.workspace.pln = PhotonPlan(prop_dose_calc={"engine": "SVDPB", **hidden})
    assert session.plan_detail()["engine_config"]["values"] == {}
    form = plan_form_values(session.workspace.pln)
    assert form["engine_config"] == {"values": {}}
    assert changes_from_form(session.workspace.pln, form) == {}

    session.set_plan({"engine": "SVDPB", "engine_config": {"values": {"kernel_cutoff": 60.0}}})
    prop = session.workspace.pln.prop_dose_calc
    assert {k: prop[k] for k in hidden} == hidden and prop["kernel_cutoff"] == 60.0
    session.set_plan({"engine_config": {"values": {**hidden, "kernel_cutoff": 50.0}}})
    prop = session.workspace.pln.prop_dose_calc
    assert {k: prop[k] for k in hidden} == hidden and prop["kernel_cutoff"] == 50.0


def test_plan_set_rejects_a_plain_bio_model_specification():
    from pyRadPlan.app._forms import FormValidationError

    session = Session(auto_plan=False)
    with pytest.raises(FormValidationError, match="model, option and values"):
        session.set_plan(
            {"radiation_mode": "protons", "bio_model": {"model": "constant_rbe", "rbe": 1.2}}
        )

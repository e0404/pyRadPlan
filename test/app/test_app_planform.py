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
    assert [c["value"] for c in options["engines"]] == ["SVDPB"]
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

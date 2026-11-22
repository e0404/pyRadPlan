import json

import numpy as np
import pytest

from pyRadPlan.app import Session, SessionManager, available_methods, dispatch


def _json_only(response):
    """The response without its buffers serializes as strict JSON."""
    json.dumps({k: v for k, v in response.items() if k != "buffers"}, allow_nan=False)


def test_protocol_methods_are_registered():
    expected = {
        "app.info",
        "session.create",
        "session.close",
        "session.state",
        "data.load_phantom",
        "data.load_file",
        "plan.get",
        "plan.set",
        "workflow.generate_stf",
        "workflow.calc_dose_influence",
        "workflow.optimize",
        "workflow.recalc",
        "workflow.run_all",
        "view.volume",
        "view.structure_masks",
        "analysis.dvh",
    }
    assert expected <= set(available_methods())


def test_app_info(client):
    info = client.ok("app.info", session=False)["result"]
    assert "TG119" in info["phantoms"]
    assert "scipy" in info["solvers"]
    assert "Generic" in info["machines"]["photons"]
    assert info["engines"]["photons"]
    _json_only({"result": info})


def test_session_lifecycle_and_state(client):
    state = client.ok("session.state")["result"]
    assert state["session"] == client.session
    assert state["has"] == dict.fromkeys(("ct", "cst", "pln", "stf", "dij", "result"), False)
    assert state["stages"]["data"] == "missing"
    v0_actions = ("generate_stf", "calc_dose_influence", "optimize", "recalc")
    assert not any(state["actions"][name] for name in v0_actions)
    assert state["status"] == "No data loaded" and state["busy"] is None
    assert client.events_of("state")[0]["session"] == client.session

    assert client.ok("session.close")["result"] == {}
    response = client.call("session.state")
    assert not response["ok"] and response["error"]["type"] == "UnknownSessionError"


def test_unknown_method_and_malformed_requests(client):
    response = client.call("no.such.method")
    assert response == {
        "id": response["id"],
        "ok": False,
        "error": {
            "type": "UnknownMethodError",
            "message": "Unknown method 'no.such.method'",
            "traceback": response["error"]["traceback"],
        },
    }
    assert "UnknownMethodError" in response["error"]["traceback"]
    _json_only(response)

    assert dispatch("nonsense")["error"]["type"] == "TypeError"
    no_traceback = dispatch({"id": 1, "method": "x"}, with_traceback=False)
    assert "traceback" not in no_traceback["error"] and no_traceback["id"] == 1


def test_parameter_errors(client):
    missing = client.call("data.load_phantom", {})
    assert missing["error"]["type"] == "TypeError" and "name" in missing["error"]["message"]
    unexpected = client.call("session.state", {"bogus": 1})
    assert unexpected["error"]["type"] == "TypeError"
    assert "bogus" in unexpected["error"]["message"]
    reserved = client.call("plan.set", {"session": "s9"})
    assert reserved["error"]["type"] == "TypeError"


def test_precondition_errors_are_reported(client):
    response = client.call("workflow.optimize", {})
    assert not response["ok"]
    assert response["error"]["type"] == "PreconditionError"
    assert response["error"]["code"] == "not_available"
    assert "dose influence" in response["error"]["message"]
    assert client.call("workflow.run_all")["error"]["code"] == "no_data"


class _Issue:
    def __init__(self, kind, is_error=True):
        self.kind, self.is_error = kind, is_error

    def __str__(self):
        return f"issue {self.kind}"


@pytest.mark.parametrize("kind", ["unsupported", "invalid_reference"])
def test_failing_tissue_check_offers_the_physical_dose_only(client, monkeypatch, kind):
    session = client.manager.get(client.session)
    client.ok("data.load_phantom", {"name": "TG119"})
    client.ok("plan.set", {"radiation_mode": "protons"})
    assert session.workspace.pln.bio_model.model == "constant_rbe"
    monkeypatch.setattr(session, "_tissue_issues", lambda pln, cst: [_Issue(kind)])
    calls = []
    monkeypatch.setattr("pyRadPlan.calc_dose_influence", lambda *a: calls.append(a))
    monkeypatch.setattr("pyRadPlan.generate_stf", lambda *a: "stf")

    error = client.call("workflow.calc_dose_influence")["error"]
    expected = "invalid_reference" if kind == "invalid_reference" else "bio_model_unsupported"
    assert error["code"] == expected and not calls

    response = client.call("workflow.calc_dose_influence", {"physical_only": True})
    if kind == "invalid_reference":
        assert response["error"]["code"] == "invalid_reference" and not calls
    else:
        assert response["ok"] and len(calls) == 1
        assert session.workspace.pln.bio_model.model == "none"
        assert response["result"]["plan"]["bio_model"] == "none"


def test_unknown_phantom_and_quantity(client):
    assert client.call("data.load_phantom", {"name": "nope"})["error"]["type"] == (
        "FileNotFoundError"
    )
    error = client.call("view.volume", {"name": "ct"})["error"]
    assert error["type"] == "LookupError"


def test_keyboard_interrupt_becomes_an_error_response(client, monkeypatch):
    session = client.manager.get(client.session)

    def interrupted():
        raise KeyboardInterrupt

    monkeypatch.setattr(session, "generate_stf", interrupted)
    response = client.call("workflow.generate_stf")
    assert not response["ok"] and response["error"]["type"] == "KeyboardInterrupt"


def test_load_file_from_buffers(client):
    from pyRadPlan.io import phantom_path

    raw = phantom_path("TG119").read_bytes()
    data = np.frombuffer(raw, dtype=np.uint8)
    params = {"filename": "C:\\uploads\\TG119.mat", "data": {"$array": 0, "dtype": "uint8"}}
    response = client.ok("data.load_file", params, buffers=[data])
    state = response["result"]
    assert state["has"]["ct"] and state["has"]["cst"] and state["has"]["pln"]
    assert [s["name"] for s in state["structures"]] == ["Core", "OuterTarget", "BODY"]
    assert state["ct"]["shape"] == [129, 167, 167]
    assert state["ct"]["spacing"] == [3.0, 3.0, 2.5]
    assert response["buffers"] == []
    _json_only(response)


def test_load_phantom_applies_the_default_plan_and_plan_roundtrip(client):
    state = client.ok("data.load_phantom", {"name": "TG119"})["result"]
    assert state["stages"]["data"] == "done" and state["stages"]["plan"] == "done"
    assert state["actions"]["generate_stf"] and not state["actions"]["optimize"]
    plan = client.ok("plan.get")["result"]
    assert plan["radiation_mode"] == "photons" and plan["gantry_angles"] == [0.0]

    state = client.ok("plan.set", {"gantry_angles": [0, 90], "num_of_fractions": 5})["result"]
    assert state["plan"]["gantry_angles"] == [0.0, 90.0]
    assert state["plan"]["couch_angles"] == [0.0, 0.0]
    assert state["plan"]["num_of_fractions"] == 5

    bad = client.call("plan.set", {"gantry_angles": [0, 90], "couch_angles": [0, 0, 0]})
    assert bad["error"]["type"] == "ValueError"
    assert client.ok("plan.get")["result"]["gantry_angles"] == [0.0, 90.0]


def test_events_only_reach_the_emit_of_the_running_request():
    manager = SessionManager()
    first, second = [], []
    created = dispatch({"id": 1, "method": "session.create"}, first.append, manager=manager)
    sid = created["result"]["session"]
    load = {"id": 2, "session": sid, "method": "data.load_phantom", "params": {"name": "TG119"}}
    dispatch(load, first.append, manager=manager)
    count = len(first)
    plan = {"id": 3, "session": sid, "method": "plan.set", "params": {"num_of_fractions": 3}}
    dispatch(plan, second.append, manager=manager)
    assert len(first) == count and second
    assert all(event["session"] == sid for event in first + second)
    manager.close_all()


def test_session_manager():
    manager = SessionManager(max_sessions=2)
    a = manager.create()
    b = manager.create("custom")
    assert isinstance(a, Session) and manager.ids() == [a.id, "custom"]
    with pytest.raises(RuntimeError):
        manager.create()
    manager.close(b.id)
    with pytest.raises(ValueError):
        manager.create(a.id)
    with pytest.raises(KeyError):
        manager.get("custom")
    manager.close_all()
    assert len(manager) == 0


def test_warning_reports_are_marked_and_repeated_once(client):
    """Reports carry ``report``; one already logged as a record is resent with ``repeat``."""
    session = client.manager.get(client.session)
    session._busy = object()  # an operation is running
    session._log("warning", "already logged", "pyRadPlan.dose")
    session._warn("already logged")
    session._warn("only reported")
    session._busy = None
    log = [
        e
        for e in session.history_since(0)["log"]
        if e["message"] in ("already logged", "only reported")
    ]
    assert [(e["message"], e.get("report", False), e.get("repeat", False)) for e in log] == [
        ("already logged", False, False),
        ("already logged", True, True),
        ("only reported", True, False),
    ]

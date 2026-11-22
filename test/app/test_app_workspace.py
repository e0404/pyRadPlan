import pytest

from pyRadPlan.app import WORKSPACE_KEYS, Workspace


def test_keys_match_the_qt_workspace_manager():
    assert WORKSPACE_KEYS == ("ct", "cst", "pln", "stf", "dij", "result")
    assert Workspace().keys == WORKSPACE_KEYS


def test_set_notifies_with_the_changed_key():
    ws = Workspace()
    seen = []
    ws.subscribe(seen.append)
    ws.ct = "ct"
    ws.set("pln", "pln")
    assert seen == [["ct"], ["pln"]]
    assert ws.ct == "ct" and ws.get("pln") == "pln"


def test_set_many_notifies_once_and_ignores_unknown_keys():
    ws = Workspace()
    seen = []
    ws.subscribe(seen.append)
    ws.set_many(stf=1, dij=2, other=3)
    assert seen == [["stf", "dij"]]
    ws.set_many(other=3)
    assert seen == [["stf", "dij"]]


def test_clear_and_has():
    ws = Workspace()
    ws.set_many(ct=1, cst=2, pln=3)
    assert ws.has("ct", "cst") and not ws.has("ct", "stf")
    assert ws.present() == {"ct", "cst", "pln"}
    seen = []
    ws.subscribe(seen.append)
    ws.clear(["cst"])
    assert seen == [["cst"]] and ws.cst is None and ws.has("ct")
    ws.clear()
    assert seen[-1] == list(WORKSPACE_KEYS) and not ws.present()


def test_refresh_reports_every_key():
    ws = Workspace()
    seen = []
    ws.subscribe(seen.append)
    ws.refresh()
    assert seen == [list(WORKSPACE_KEYS)]


def test_unsubscribe_stops_notifications_and_is_idempotent():
    ws = Workspace()
    seen = []
    unsubscribe = ws.subscribe(seen.append)
    ws.ct = 1
    unsubscribe()
    unsubscribe()
    ws.ct = 2
    assert seen == [["ct"]]


def test_a_failing_listener_does_not_stop_the_others():
    ws = Workspace()
    seen = []

    def broken(_keys):
        raise RuntimeError("boom")

    ws.subscribe(broken)
    ws.subscribe(seen.append)
    ws.dij = 1
    assert seen == [["dij"]]


def test_listener_receives_a_copy_of_the_changed_keys():
    ws = Workspace()
    seen = []

    def mutate(keys):
        keys.append("x")

    ws.subscribe(mutate)
    ws.subscribe(seen.append)
    ws.ct = 1
    assert seen == [["ct"]]


def test_unknown_keys_raise():
    ws = Workspace()
    with pytest.raises(KeyError):
        ws.get("dose")
    with pytest.raises(KeyError):
        ws.set("dose", 1)
    with pytest.raises(AttributeError):
        ws.dose  # noqa: B018


def test_snapshot_is_a_copy():
    ws = Workspace()
    ws.ct = 1
    snapshot = ws.snapshot()
    snapshot["ct"] = 2
    assert ws.ct == 1

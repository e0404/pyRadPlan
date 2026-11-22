import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.workspace import WorkspaceManager


def test_properties_emit_changed(qapp):
    ws = WorkspaceManager()
    received = []
    ws.workspace_changed.connect(received.append)

    ws.ct = object()
    assert received[-1] == ["ct"]
    ws.pln = object()
    assert received[-1] == ["pln"]


def test_set_many_emits_once(qapp):
    ws = WorkspaceManager()
    received = []
    ws.workspace_changed.connect(received.append)

    ws.set_many(ct=object(), cst=object(), pln=object())
    assert len(received) == 1
    assert set(received[0]) == {"ct", "cst", "pln"}


def test_has(qapp):
    ws = WorkspaceManager()
    assert not ws.has("ct")
    ws.ct = object()
    assert ws.has("ct")
    assert not ws.has("ct", "cst")
    ws.cst = object()
    assert ws.has("ct", "cst")


def test_clear(qapp):
    ws = WorkspaceManager()
    ws.set_many(ct=object(), cst=object())
    received = []
    ws.workspace_changed.connect(received.append)

    ws.clear(["ct"])
    assert ws.ct is None
    assert ws.cst is not None
    assert received[-1] == ["ct"]

    ws.clear()
    assert not ws.has("cst")


def test_refresh_emits_all_keys(qapp):
    ws = WorkspaceManager()
    received = []
    ws.workspace_changed.connect(received.append)
    ws.refresh()
    assert set(received[-1]) == set(ws.keys)


def test_singleton(qapp):
    assert WorkspaceManager.instance() is WorkspaceManager.instance()


def _dose(value=1.0):
    import SimpleITK as sitk

    return sitk.Image([4, 4, 2], sitk.sitkFloat32) + value


def test_results_are_kept_by_name(qapp):
    from pyRadPlan.result import PlanResult

    ws = WorkspaceManager()
    received = []
    ws.workspace_changed.connect(received.append)

    ws.add_result("measured", {"physical_dose": _dose()})
    assert received == [["result"]]
    assert ws.result is None and not ws.has("result")
    assert isinstance(ws.results["measured"], PlanResult)

    ws.result = {"physical_dose": _dose(2.0)}
    assert ws.active_result == "current"
    assert list(ws.results) == ["measured", "current"]
    assert [tag for tag, _ in ws.result_entries()] == ["", "measured"]
    assert ws.result_entries()[0][1] is ws.result

    # Replacing or dropping the active result leaves the others alone
    measured = ws.results["measured"]
    ws.result = {"physical_dose": _dose(3.0)}
    ws.result = None
    assert ws.results == {"measured": measured}

    ws.results["other"] = measured  # a copy; the workspace is unchanged
    assert list(ws.results) == ["measured"]


def test_active_result_can_be_switched(qapp):
    ws = WorkspaceManager()
    ws.result = {"physical_dose": _dose()}
    ws.add_result("plan_b", {"physical_dose": _dose(2.0)})
    plan_b = ws.results["plan_b"]

    ws.active_result = "plan_b"
    assert ws.result is plan_b
    assert [tag for tag, _ in ws.result_entries()] == ["", "current"]

    with pytest.raises(KeyError):
        ws.active_result = "missing"

    ws.remove_result("plan_b")
    assert ws.result is None
    assert list(ws.results) == ["current"]


def test_result_names_are_checked(qapp):
    ws = WorkspaceManager()
    for name in ("", "two words", "a/b", None):
        with pytest.raises(ValueError, match="Invalid result name"):
            ws.add_result(name, {"physical_dose": _dose()})
    assert ws.results == {}


def test_set_many_replaces_all_results_and_clear_drops_them(qapp):
    ws = WorkspaceManager()
    ws.result = {"physical_dose": _dose()}
    ws.add_result("old", {"physical_dose": _dose()})
    received = []
    ws.workspace_changed.connect(received.append)

    ws.set_many(
        ct=object(),
        results={"a": {"physical_dose": _dose()}, "b": {"physical_dose": _dose()}},
        active_result="b",
    )
    assert received == [["ct", "result"]]
    assert list(ws.results) == ["a", "b"]
    assert ws.result is ws.results["b"]

    ws.clear(["result"])
    assert ws.results == {} and ws.result is None
    assert ws.active_result == "current"


def test_the_session_tracks_a_change_before_the_widgets_hear_of_it(qapp):
    ws = WorkspaceManager()
    session = ws.session
    assert session.workspace is ws.store and session.profile == "native"
    assert not session.auto_plan and session.solver_preference == ()
    ws.set_many(ct=object(), cst=object(), pln=object(), stf=object(), dij=object())
    flags = []
    ws.workspace_changed.connect(lambda keys: flags.append(session.tracker.dij_stale))

    ws.pln = object()

    assert flags == [True]
    assert ws.store.pln is ws.pln and ws.present() == ws.store.present()

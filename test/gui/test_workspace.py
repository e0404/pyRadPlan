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

import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.widgets._base import WorkspaceWidget
from pyRadPlan.gui.workspace import WorkspaceManager


class _RecordingWidget(WorkspaceWidget):
    _watched_keys = ("ct", "cst")

    def __init__(self, workspace=None, parent=None):
        super().__init__(workspace, parent)
        self.calls: list[list] = []
        self.initialize()

    def _do_update(self, changed_keys):
        self.calls.append(list(changed_keys))


def test_initialize_triggers_full_update(qapp):
    w = _RecordingWidget(WorkspaceManager())
    assert w.calls == [[]]


def test_watched_keys_filter(qapp):
    ws = WorkspaceManager()
    w = _RecordingWidget(ws)
    w.calls.clear()

    ws.pln = object()  # not watched
    assert w.calls == []

    ws.ct = object()  # watched
    assert w.calls == [["ct"]]


def test_hold_updates_suppresses_self(qapp):
    ws = WorkspaceManager()
    w = _RecordingWidget(ws)
    w.calls.clear()

    with w.hold_updates():
        ws.ct = object()
    assert w.calls == []

    # Updates resume afterwards
    ws.cst = object()
    assert w.calls == [["cst"]]


def test_other_widgets_still_update_during_hold(qapp):
    ws = WorkspaceManager()
    writer = _RecordingWidget(ws)
    listener = _RecordingWidget(ws)
    writer.calls.clear()
    listener.calls.clear()

    with writer.hold_updates():
        ws.ct = object()

    assert writer.calls == []
    assert listener.calls == [["ct"]]


def test_update_failure_emits_signal_without_raising(qapp):
    class _Boom(WorkspaceWidget):
        def __init__(self, workspace=None, parent=None):
            super().__init__(workspace, parent)
            self.initialize()

        def _do_update(self, changed_keys):
            raise RuntimeError("boom")

    messages = []
    w = _Boom(WorkspaceManager())
    w.update_failed.connect(messages.append)
    w.workspace.ct = object()  # triggers _do_update -> raises -> caught
    assert any("boom" in m for m in messages)

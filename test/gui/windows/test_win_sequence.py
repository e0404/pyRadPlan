from pyRadPlan.gui.windows._sequence_win import SequenceWindow


def test_sequence_window_init(qapp):
    win = SequenceWindow()
    assert win is not None
    assert win.viewer is not None
    assert win.windowTitle() == "pyRadPlan Sequence Viewer"

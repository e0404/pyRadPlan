import numpy as np
import pytest

from pyRadPlan.gui.windows._main_win import MainWindow
from pyRadPlan.gui.windows._result_win import QuantityWindow
from pyRadPlan.gui.workspace import WorkspaceManager


def test_quantity_window_init(qapp):
    win = QuantityWindow()
    assert win is not None
    assert win.viewer is not None
    assert win.windowTitle() == "pyRadPlan Plan Result Viewer"


def test_quantity_window_load():
    # TODO
    pass


def _open_analysis_window(parent=None):
    from pyRadPlan.gui.windows._analysis_win import show_analysis

    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    return show_analysis(
        quantities={"Dose": np.ones((4, 4, 4))}, masks={"Target": mask}, parent=parent
    )


def test_closing_quantity_window_closes_analysis_windows(qapp):
    from pyRadPlan.gui.windows._analysis_win import _OPEN_WINDOWS, close_all_analysis_windows

    win = QuantityWindow()
    analysis = _open_analysis_window(parent=win.viewer)
    try:
        assert analysis in _OPEN_WINDOWS
        win.close()
        assert analysis not in _OPEN_WINDOWS
    finally:
        close_all_analysis_windows()


@pytest.mark.parametrize("close_main", [False, True])
def test_closing_viewer_preserves_unrelated_analysis_windows(qapp, close_main):
    """Only analyses opened by the closing viewer should disappear."""
    from pyRadPlan.gui.windows._analysis_win import _OPEN_WINDOWS, close_all_analysis_windows

    main = MainWindow(WorkspaceManager())
    result = QuantityWindow(WorkspaceManager())

    def open_from_viewer(viewer):
        viewer.quantity_widget._quantities = {"Dose": np.ones((4, 4, 4))}
        viewer.quantity_widget._masks = {"Target": np.ones((4, 4, 4), dtype=bool)}
        viewer._on_show_analysis()
        return viewer._analysis_window

    try:
        main_analyses = [open_from_viewer(main._viewer) for _ in range(2)]
        result_analyses = [open_from_viewer(result.viewer) for _ in range(2)]
        standalone = _open_analysis_window()
        closing = main if close_main else result
        owned = main_analyses if close_main else result_analyses
        unrelated = result_analyses if close_main else main_analyses

        assert all(w.parent() is None for w in [*owned, *unrelated, standalone])
        assert closing.close()

        assert all(w not in _OPEN_WINDOWS and not w.isVisible() for w in owned)
        assert all(w in _OPEN_WINDOWS and w.isVisible() for w in [*unrelated, standalone])
    finally:
        main.close()
        result.close()
        close_all_analysis_windows()

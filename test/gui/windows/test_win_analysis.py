import pytest

from pyRadPlan.gui.widgets._analysis_widget import AnalysisWidget
from pyRadPlan.gui.windows._analysis_win import (
    _OPEN_WINDOWS,
    AnalysisWindow,
    close_all_analysis_windows,
    show_analysis,
)
import numpy as np
import SimpleITK as sitk


@pytest.fixture(autouse=True)
def _close_analysis_windows():
    yield
    close_all_analysis_windows()


def test_analysis_window_init(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    quantities = {"Dose": dose}
    import SimpleITK as sitk

    masks = {
        voi.name: sitk.GetArrayFromImage(voi.mask)
        if isinstance(voi.mask, sitk.Image)
        else np.asarray(voi.mask)
        for voi in cst.vois
    }

    window = AnalysisWindow(quantities=quantities, masks=masks)
    assert window is not None
    assert window.widget is not None


def test_show_analysis(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    quantities = {"Dose": dose}
    import SimpleITK as sitk

    masks = {
        voi.name: sitk.GetArrayFromImage(voi.mask)
        if isinstance(voi.mask, sitk.Image)
        else np.asarray(voi.mask)
        for voi in cst.vois
    }

    window = show_analysis(quantities=quantities, masks=masks)

    assert isinstance(window, AnalysisWindow)
    window.close()


def test_analysis_window_is_top_level(qapp, test_data_photons):
    """The window must have no Qt parent, or Windows denies it a taskbar button.

    A parented window gets a native owner HWND and is then unreachable via the
    taskbar once other applications are in front of it. Setting Qt.Window does
    not help; only the absence of a parent does.
    """
    from PySide6.QtWidgets import QWidget

    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)
    masks = {voi.name: sitk.GetArrayFromImage(voi.mask) for voi in cst.vois}

    host = QWidget()
    window = AnalysisWindow(quantities={"Dose": dose}, masks=masks, parent=host)

    assert window.parent() is None
    assert window.windowHandle() is None or window.windowHandle().transientParent() is None


def test_analysis_window_survives_without_caller_reference(qapp, test_data_photons):
    """Having no Qt parent must not let the window be garbage collected."""
    import gc
    import weakref

    from pyRadPlan.gui.windows._analysis_win import _OPEN_WINDOWS

    ct, cst, result = test_data_photons
    dose = np.swapaxes(result["physicalDose"], 0, 1)
    masks = {voi.name: sitk.GetArrayFromImage(voi.mask) for voi in cst.vois}

    window = show_analysis(quantities={"Dose": dose}, masks=masks)
    ref = weakref.ref(window)
    assert window in _OPEN_WINDOWS

    del window
    gc.collect()
    assert ref() is not None, "window was collected while still open"

    ref().close()
    assert ref() not in _OPEN_WINDOWS


def _small_window():
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[1:3, 1:3, 1:3] = True
    return show_analysis(quantities={"Dose": np.ones((4, 4, 4))}, masks={"Target": mask})


def test_close_all_analysis_windows(qapp):
    windows = [_small_window(), _small_window()]
    assert all(w in _OPEN_WINDOWS for w in windows)

    close_all_analysis_windows()

    assert not _OPEN_WINDOWS
    assert not any(w.isVisible() for w in windows)


def test_failing_set_data_does_not_register_window(qapp, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(AnalysisWidget, "set_data", fail)
    before = set(_OPEN_WINDOWS)

    with pytest.raises(RuntimeError, match="boom"):
        _small_window()

    assert _OPEN_WINDOWS == before

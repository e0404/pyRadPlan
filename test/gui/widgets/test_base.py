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


@pytest.mark.parametrize("locale_name", ["de_DE", "en_US"])
def test_adaptive_spin_box_edits_round_trip_in_the_widget_locale(qapp, locale_name):
    from PySide6.QtCore import QLocale, Qt
    from PySide6.QtTest import QTest

    from pyRadPlan.gui.widgets._base import AdaptiveDoubleSpinBox

    spin = AdaptiveDoubleSpinBox()
    spin.setLocale(QLocale(locale_name))
    spin.setRange(-1.0e9, 1.0e9)
    spin.setValue(60.0)
    spin.show()
    spin.setFocus()

    # Overwrite "60" of "60.0" / "60,0" with "45", as when editing the value in place.
    spin.lineEdit().setSelection(0, 2)
    QTest.keyClicks(spin, "45")
    QTest.keyClick(spin, Qt.Key.Key_Return)

    assert spin.value() == 45.0
    spin.setValue(12345.5)
    assert spin.valueFromText(spin.text()) == 12345.5


@pytest.mark.parametrize("locale_name", ["ccp_BD", "ff_Adlm_SN", "ar_EG", "de_DE", "en_US"])
def test_adaptive_spin_box_small_values_round_trip_in_any_locale(qapp, locale_name):
    """Locales with native digits outside the BMP (Chakma, Adlam) included."""
    from PySide6.QtCore import QLocale, Qt
    from PySide6.QtTest import QTest

    from pyRadPlan.gui.widgets._base import AdaptiveDoubleSpinBox

    spin = AdaptiveDoubleSpinBox()
    spin.setLocale(QLocale(locale_name))
    spin.setRange(-1.0e9, 1.0e9)
    for value in (0.000123, -0.0456, 1.5, 12345.5):
        spin.setValue(value)
        assert spin.valueFromText(spin.text()) == pytest.approx(value)

    spin.setValue(0.000123)
    spin.show()
    spin.setFocus()
    spin.lineEdit().selectAll()
    # Inserted rather than typed: QTest has no key codes for separators such as the
    # Arabic decimal separator and crashes on them.
    spin.lineEdit().insert(spin.textFromValue(0.000456))
    QTest.keyClick(spin, Qt.Key.Key_Return)

    assert spin.value() == pytest.approx(0.000456)


def test_adaptive_spin_box_size_ignores_a_wide_range(qapp):
    from pyRadPlan.gui.widgets._base import AdaptiveDoubleSpinBox

    narrow = AdaptiveDoubleSpinBox()
    narrow.setRange(-1000.0, 1000.0)
    wide = AdaptiveDoubleSpinBox()
    wide.setRange(-1.0e9, 1.0e9)

    assert wide.sizeHint().width() <= narrow.sizeHint().width()

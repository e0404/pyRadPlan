import sys

import pytest
from PySide6.QtWidgets import QApplication, QWidget

import pyRadPlan.gui.windows._main_win as main_win
from pyRadPlan.gui import apps
from pyRadPlan.gui.windows._splash_win import (
    BUILD_WINDOW_WEIGHT,
    SHOW_WINDOW_WEIGHT,
    StartupSplash,
)


@pytest.fixture
def splash(qapp):
    splash = StartupSplash(total_weight=10.0)
    splash.show()
    yield splash
    splash.close()


@pytest.fixture
def recorded_steps(monkeypatch):
    """Record every splash step as (splash, message, weight)."""
    steps = []
    original = StartupSplash.step

    def step(self, message, weight=1.0):
        steps.append((self, message, weight))
        original(self, message, weight)

    monkeypatch.setattr(StartupSplash, "step", step)
    return steps


def test_splash_starts_empty(splash):
    assert splash.progress() == 0.0
    assert splash.message() == ""


def test_step_completes_the_running_phase(splash):
    splash.step("Loading A…", 4.0)
    assert splash.message() == "Loading A…"
    assert splash.progress() == 0.0

    splash.step("Loading B…", 6.0)
    assert splash.message() == "Loading B…"
    assert splash.progress() == pytest.approx(0.4)


def test_progress_is_capped(splash):
    for message in ("A", "B", "C"):
        splash.step(message, 8.0)
    assert splash.progress() == 1.0


def test_finish_fills_bar_and_closes_when_window_shown(splash):
    splash.step("Loading A…", 4.0)
    window = QWidget()
    window.show()
    splash.finish(window)
    assert splash.progress() == 1.0
    assert not splash.isVisible()
    window.close()


def test_launch_main_window_advances_and_finishes_splash(qapp, monkeypatch, recorded_steps):
    monkeypatch.setattr(main_win, "MainWindow", lambda workspace: QWidget())
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)
    splash = StartupSplash(BUILD_WINDOW_WEIGHT + SHOW_WINDOW_WEIGHT)
    splash.show()

    main_win.launch_main_window(None, splash=splash)

    assert [weight for *_, weight in recorded_steps] == [BUILD_WINDOW_WEIGHT, SHOW_WINDOW_WEIGHT]
    assert splash.progress() == 1.0
    assert not splash.isVisible()


@pytest.fixture
def captured_launch(monkeypatch):
    """Replace the main-window launcher and record the splash it receives."""
    launched = {}

    def fake_launch(workspace, splash=None):
        launched.update(workspace=workspace, splash=splash, visible=splash.isVisible())

    monkeypatch.setattr(main_win, "launch_main_window", fake_launch)
    return launched


@pytest.fixture
def unloaded_module(tmp_path, monkeypatch):
    """Name of an importable module that has not been imported yet."""
    (tmp_path / "_splash_probe_module.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "_splash_probe_module", raising=False)
    return "_splash_probe_module"


def test_gui_hands_visible_splash_to_main_window_and_closes_it(qapp, captured_launch):
    apps.gui()

    assert captured_launch["workspace"] is None
    assert captured_launch["visible"]
    assert not captured_launch["splash"].isVisible()


def test_gui_announces_only_pending_imports(
    qapp, monkeypatch, captured_launch, recorded_steps, unloaded_module
):
    monkeypatch.setattr(
        apps,
        "_STARTUP_IMPORTS",
        (("Loaded already", "sys", 5.0), ("Loading probe…", unloaded_module, 3.0)),
    )

    apps.gui()

    assert [(message, weight) for _, message, weight in recorded_steps] == [
        ("Loading probe…", 3.0)
    ]
    assert unloaded_module in sys.modules


def test_gui_closes_splash_when_startup_fails(qapp, monkeypatch, recorded_steps):
    monkeypatch.setattr(
        apps, "_STARTUP_IMPORTS", (("Loading broken…", "pyRadPlan._no_such_module", 1.0),)
    )

    with pytest.raises(ModuleNotFoundError):
        apps.gui()

    splash, message, _ = recorded_steps[0]
    assert message == "Loading broken…"
    assert not splash.isVisible()

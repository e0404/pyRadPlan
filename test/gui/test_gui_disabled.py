"""Tests for disabling the GUI via the PYRADPLAN_GUI_DISABLED environment variable."""

import subprocess
import sys

import pytest

CODE = (
    "import pyRadPlan.gui as g; print(g.GUI_AVAILABLE, g.GUI_DISABLED);"
    "g.launch_viewer(None, None, None)"
)


@pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
def test_gui_disabled_env(value, monkeypatch):
    """GUI reported unavailable and launching it fails with a clear message."""
    monkeypatch.setenv("PYRADPLAN_GUI_DISABLED", value)
    proc = subprocess.run(
        [sys.executable, "-c", CODE], capture_output=True, text=True, check=False
    )
    assert proc.stdout.strip() == "False True"
    assert proc.returncode != 0
    assert "PYRADPLAN_GUI_DISABLED" in proc.stderr


def test_gui_not_disabled_by_default(monkeypatch):
    """Without the variable the GUI is not flagged as disabled."""
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    proc = subprocess.run(
        [sys.executable, "-c", "import pyRadPlan.gui as g; print(g.GUI_DISABLED)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "False"

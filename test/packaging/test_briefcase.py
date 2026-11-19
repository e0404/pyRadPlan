"""Checks of the Briefcase desktop bundle against the library."""

from pathlib import Path
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, where pytest brings tomli
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def test_bundle_requires_match_gui_extra():
    root = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    bundle = tomllib.loads((ROOT / "packaging/briefcase/pyproject.toml").read_text("utf-8"))
    requires = bundle["tool"]["briefcase"]["app"]["pyradplan_app"]["requires"]

    assert set(root["project"]["optional-dependencies"]["gui"]) <= set(requires)


def test_app_loads_qt_before_simpleitk(monkeypatch):
    """The app imports Qt first, so on Windows Qt uses the C++ runtime PySide6 ships."""
    pytest.importorskip("PySide6")
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import pyradplan_app.app; "
        "mods = list(sys.modules); "
        "print(mods.index('PySide6.QtCore') < mods.index('SimpleITK'))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "packaging/briefcase/src")],
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "True"

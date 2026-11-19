"""Checks of the Briefcase desktop bundle against the library."""

import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10, where pytest brings tomli
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]


def _load_run_briefcase():
    path = ROOT / "packaging/briefcase/scripts/run_briefcase.py"
    spec = importlib.util.spec_from_file_location("run_briefcase", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_copy_msvc_runtime(tmp_path):
    """All msvcp140 DLLs of shiboken6 are copied, replacing older copies."""
    shiboken = tmp_path / "app_packages/shiboken6"
    shiboken.mkdir(parents=True)
    names = ["msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll"]
    for name in [*names, "vcruntime140.dll"]:
        (shiboken / name).write_text(f"new {name}")
    target = tmp_path / "src"
    target.mkdir()
    (target / "msvcp140.dll").write_text("old")

    copied = _load_run_briefcase().copy_msvc_runtime(tmp_path / "app_packages", target)

    assert sorted(p.name for p in copied) == names
    assert (target / "msvcp140.dll").read_text() == "new msvcp140.dll"
    assert not (target / "vcruntime140.dll").exists()


def test_copy_msvc_runtime_fails_without_runtime(tmp_path):
    """A bundle without the runtime fails the build instead of silently shipping without it."""
    (tmp_path / "shiboken6").mkdir()
    with pytest.raises(FileNotFoundError):
        _load_run_briefcase().copy_msvc_runtime(tmp_path, tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="PySide6 ships the runtime on Windows only")
def test_pyside6_ships_msvc_runtime_in_shiboken6():
    """The layout the bundle relies on holds for the installed PySide6, e.g. after an update."""
    spec = importlib.util.find_spec("shiboken6")
    if spec is None:
        pytest.skip("PySide6 is not installed")
    assert (Path(spec.origin).parent / "msvcp140.dll").is_file()

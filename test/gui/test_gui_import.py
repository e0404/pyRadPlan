"""Tests for how pyRadPlan.gui loads Qt and reports GUI import failures.

Each test runs in a fresh interpreter, as module import order and DLL loading are process-wide.
"""

import subprocess
import sys
import textwrap

import pytest

# Makes importing the given module raise the given exception, before any other import.
FAIL_IMPORT = """
import importlib.abc, sys

class _Fail(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == {module!r}:
            raise {exc}
        return None

sys.meta_path.insert(0, _Fail())
"""

# Makes the given module behave as if it were not installed, for import and find_spec alike.
MISSING = """
import sys
sys.modules[{module!r}] = None
"""

REPORT = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import pyRadPlan.gui as g
print("AVAILABLE", g.GUI_AVAILABLE)
print("WARNINGS", [str(w.message) for w in caught if w.category is RuntimeWarning])
try:
    g.main([])
except ImportError as exc:
    print("ERROR", exc)
"""


def _run(code):
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout


def _run_with_failing_import(module, exc, monkeypatch):
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    return _run(FAIL_IMPORT.format(module=module, exc=exc) + REPORT)


def test_qt_imported_before_simpleitk(monkeypatch):
    """Qt is loaded before SimpleITK, so Qt's own C++ runtime wins on Windows."""
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    out = _run(
        """
        import sys
        import pyRadPlan.gui
        mods = list(sys.modules)
        print(mods.index("PySide6.QtCore") < mods.index("SimpleITK"))
        """
    )
    assert out.strip() == "True"


@pytest.mark.skipif(sys.platform != "win32", reason="MSVC runtime loading is Windows-only")
def test_qt_runs_on_bundled_msvc_runtime(monkeypatch):
    """The msvcp140.dll in use is the one PySide6 ships, not the system's."""
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    out = _run(
        """
        import ctypes
        import pyRadPlan.gui
        k = ctypes.WinDLL("kernel32")
        k.GetModuleHandleW.restype = ctypes.c_void_p
        k.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
        k.GetModuleFileNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint]
        buf = ctypes.create_unicode_buffer(1024)
        k.GetModuleFileNameW(k.GetModuleHandleW("msvcp140.dll"), buf, 1024)
        print(buf.value)
        """
    )
    path = out.strip().lower()
    assert "site-packages" in path
    assert "pyside6" in path or "shiboken6" in path


@pytest.mark.parametrize("module", ["PySide6", "pyqtgraph"])
def test_missing_dependency_suggests_extra(module, monkeypatch):
    """A missing GUI dependency asks to install the extra, without a warning."""
    monkeypatch.delenv("PYRADPLAN_GUI_DISABLED", raising=False)
    out = _run(MISSING.format(module=module) + REPORT)
    assert "AVAILABLE False" in out
    assert "WARNINGS []" in out
    assert "pip install 'pyRadPlan[gui]'" in out


def test_broken_qt_reports_real_error(monkeypatch):
    """A Qt load failure is reported as such, not as a missing extra."""
    out = _run_with_failing_import(
        "PySide6.QtCore",
        'ImportError("DLL load failed while importing QtCore: '
        'The specified procedure could not be found.")',
        monkeypatch,
    )
    assert "AVAILABLE False" in out
    warnings_line = next(line for line in out.splitlines() if line.startswith("WARNINGS"))
    assert "DLL load failed" in warnings_line
    error_line = next(line for line in out.splitlines() if line.startswith("ERROR"))
    assert "DLL load failed" in error_line
    assert "Visual C++ Redistributable" in error_line
    assert "pip install" not in error_line


def test_unrelated_missing_module_is_not_reported_as_missing_extra(monkeypatch):
    """A missing module outside the 'gui' extra is reported as a broken installation."""
    out = _run_with_failing_import(
        "SimpleITK", 'ModuleNotFoundError(name="SimpleITK")', monkeypatch
    )
    assert "AVAILABLE False" in out
    error_line = next(line for line in out.splitlines() if line.startswith("ERROR"))
    assert "failed to load" in error_line
    assert "pip install" not in error_line

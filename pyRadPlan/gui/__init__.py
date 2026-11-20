"""Qt-based visualization components for pyRadPlan.

This subpackage provides graphical user interface (GUI) applications for visualizing
and analyzing radiotherapy plans and quantity distributions.
"""

import os
import warnings

#: Top-level modules of the 'gui' extra. Only their absence means the extra is not installed;
#: any other import failure is a broken installation and is reported as such.
GUI_DEPENDENCIES = frozenset({"PySide6", "shiboken6", "pyqtgraph", "contourpy"})

# Set PYRADPLAN_GUI_DISABLED=1 to report the GUI as unavailable, e.g. when running the examples
# non-interactively (docs notebook execution, headless CI) so they fall back to static plots.
GUI_DISABLED = os.environ.get("PYRADPLAN_GUI_DISABLED", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

try:
    if GUI_DISABLED:
        raise ImportError("pyRadPlan GUI disabled via PYRADPLAN_GUI_DISABLED")
    # Qt has to be loaded before SimpleITK and the like: on Windows they link the system's
    # msvcp140.dll, and as a DLL already loaded by name is reused, Qt would otherwise run on
    # that one instead of the newer runtime PySide6 ships, and fails if the system's is too old.
    from PySide6 import QtCore  # noqa: F401
    from .apps import gui, launch_viewer, launch_sequence_viewer, analysis_viewer, main

    GUI_AVAILABLE = True
except ImportError as _exc:
    GUI_AVAILABLE = False
    _import_error = _exc
    _missing_dependency = (
        isinstance(_exc, ModuleNotFoundError)
        and (_exc.name or "").partition(".")[0] in GUI_DEPENDENCIES
    )
    if GUI_DISABLED or _missing_dependency:
        _gui_error_message = (
            "pyRadPlan GUI requires the 'gui' optional dependencies (PySide6, pyqtgraph). "
            "Install them with: pip install 'pyRadPlan[gui]'"
        )
    else:
        _gui_error_message = (
            f"pyRadPlan GUI dependencies are installed but failed to load: {_exc}. On Windows, "
            "'DLL load failed' usually means an outdated Microsoft Visual C++ Redistributable; "
            "install the latest x64 version."
        )
        warnings.warn(_gui_error_message, RuntimeWarning, stacklevel=2)

    def _missing_gui(*_args, **_kwargs):
        if GUI_DISABLED:
            raise RuntimeError(
                "pyRadPlan GUI is disabled via the PYRADPLAN_GUI_DISABLED environment variable."
            )
        raise ImportError(_gui_error_message) from _import_error

    gui = launch_viewer = launch_sequence_viewer = analysis_viewer = main = _missing_gui


__all__ = [
    "gui",
    "launch_viewer",
    "launch_sequence_viewer",
    "analysis_viewer",
    "main",
    "GUI_AVAILABLE",
    "GUI_DISABLED",
]

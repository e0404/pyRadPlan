"""Qt-based visualization components for pyRadPlan.

This subpackage provides graphical user interface (GUI) applications for visualizing
and analyzing radiotherapy plans and quantity distributions.
"""

import os

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
    from .apps import gui, launch_viewer, analysis_viewer, main

    GUI_AVAILABLE = True
except ImportError as _exc:
    GUI_AVAILABLE = False
    _import_error = _exc

    def _missing_gui(*_args, **_kwargs):
        if GUI_DISABLED:
            raise RuntimeError(
                "pyRadPlan GUI is disabled via the PYRADPLAN_GUI_DISABLED environment variable."
            )
        raise ImportError(
            "pyRadPlan GUI requires the 'gui' optional dependencies (PySide6, pyqtgraph). "
            "Install them with: pip install 'pyRadPlan[gui]'"
        ) from _import_error

    gui = launch_viewer = analysis_viewer = main = _missing_gui


__all__ = [
    "gui",
    "launch_viewer",
    "analysis_viewer",
    "main",
    "GUI_AVAILABLE",
    "GUI_DISABLED",
]

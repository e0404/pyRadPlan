"""Qt-based visualization components for pyRadPlan.

This subpackage provides graphical user interface (GUI) applications for visualizing
and analyzing radiotherapy plans and quantity distributions.
"""

try:
    from .apps import gui, launch_viewer, analysis_viewer
except ImportError as _exc:
    _import_error = _exc

    def _missing_gui(*_args, **_kwargs):
        raise ImportError(
            "pyRadPlan GUI requires the 'gui' optional dependencies (PySide6, pyqtgraph). "
            "Install them with: pip install 'pyRadPlan[gui]'"
        ) from _import_error

    gui = launch_viewer = analysis_viewer = _missing_gui


__all__ = [
    "gui",
    "launch_viewer",
    "analysis_viewer",
]

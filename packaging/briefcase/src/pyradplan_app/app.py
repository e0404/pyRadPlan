"""Start the pyRadPlan GUI from the bundled application.

The whole GUI lives in :mod:`pyRadPlan.gui`; this module only hands the command
line over to it so the desktop bundle and ``pyRadPlanGUI`` behave the same.
"""

import sys

from pyRadPlan.gui import main as gui_main


def main() -> None:
    """Launch the pyRadPlan main window and exit with its status."""
    sys.exit(gui_main(sys.argv[1:]))

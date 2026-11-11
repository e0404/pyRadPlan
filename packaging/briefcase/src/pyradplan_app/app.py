"""Start the pyRadPlan GUI from the bundled application.

The whole GUI lives in :mod:`pyRadPlan.gui`; this module only hands the command
line over to it so the desktop bundle and ``pyRadPlanGUI`` behave the same.
"""

import os
from pathlib import Path
import sys

from pyRadPlan import settings
from pyRadPlan.gui import main as gui_main

#: Base data the app ships in addition to pyRadPlan's, the phantoms of matRad that the build
#: downloads with scripts/fetch_matrad_phantoms.py.
BASE_DATA = Path(__file__).with_name("data")


def add_bundled_base_data() -> None:
    """Search the app's base data folder after the shared ones the user configured."""
    if BASE_DATA.is_dir():
        roots = [settings.base_data_path, str(BASE_DATA)]
        settings.base_data_path = os.pathsep.join(root for root in roots if root)


def main() -> None:
    """Launch the pyRadPlan main window and exit with its status."""
    add_bundled_base_data()
    # The Linux desktop entry passes every selected file (Exec=... %F); open the first one.
    sys.exit(gui_main(sys.argv[1:2]))

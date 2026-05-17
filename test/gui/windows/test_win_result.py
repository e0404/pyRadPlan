import numpy as np
import os
from pyRadPlan.gui.windows._result_win import QuantityWindow


def test_quantity_window_init(qapp):
    win = QuantityWindow()
    assert win is not None
    assert win.viewer is not None
    assert win.windowTitle() == "pyRadPlan Plan Result Viewer"


def test_quantity_window_load():
    # TODO
    pass

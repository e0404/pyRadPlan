from pyRadPlan.gui.widgets.analysis._dvh import DVHPlotWidget
from pyRadPlan.analysis._dvh import DVH
import numpy as np


def test_dvh_plot_widget_init(qapp):
    widget = DVHPlotWidget()
    assert widget is not None
    assert widget.figure is not None
    assert widget.canvas is not None


def test_dvh_plot_widget_plot(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    dvhs = [DVH.compute(quantity=dose, mask=voi.mask, name=voi.name) for voi in cst.vois]

    widget = DVHPlotWidget()
    widget.plot(dvhs)

    # Check if axes were created
    assert len(widget.figure.axes) > 0

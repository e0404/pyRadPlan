from pyRadPlan.gui.widgets._analysis_widget import AnalysisWidget
from pyRadPlan.analysis._dvh import DVH
import numpy as np


def test_analysis_widget_init(qapp):
    widget = AnalysisWidget()
    assert widget is not None
    assert widget.tabs.count() == 2


def test_analysis_widget_plot(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dose = np.swapaxes(result["physicalDose"], 0, 1)

    quantities = {"Dose": dose}
    import SimpleITK as sitk

    masks = {
        voi.name: sitk.GetArrayFromImage(voi.mask)
        if isinstance(voi.mask, sitk.Image)
        else np.asarray(voi.mask)
        for voi in cst.vois
    }

    widget = AnalysisWidget()
    widget.set_data(quantities=quantities, masks=masks)

    assert widget.dvh_widget is not None
    assert widget.qi_widget is not None

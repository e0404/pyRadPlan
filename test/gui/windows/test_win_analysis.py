from pyRadPlan.gui.windows._analysis_win import AnalysisWindow, show_analysis
from pyRadPlan.analysis._dvh import DVH
import numpy as np


def test_analysis_window_init(qapp, test_data_photons):
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

    window = AnalysisWindow(quantities=quantities, masks=masks)
    assert window is not None
    assert window.widget is not None


def test_show_analysis(qapp, test_data_photons):
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

    window = show_analysis(quantities=quantities, masks=masks)

    assert isinstance(window, AnalysisWindow)
    window.close()

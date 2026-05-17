import numpy as np
import SimpleITK as sitk
from pyRadPlan.gui.widgets._result_widget import ViewingWidget


def test_viewing_widget_init(qapp):
    widget = ViewingWidget()
    assert widget is not None
    assert widget.quantity_widget is not None
    assert widget.vis_widget is not None
    assert widget.opts_widget is not None
    assert widget.vois_widget is not None


def test_viewing_widget_set_data(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    # Prepare data (X, Y, Z)
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)

    if isinstance(result, dict) and "physicalDose" in result:
        dose_vol = np.swapaxes(result["physicalDose"], 0, 1)
    else:
        # Fallback for testing if result structure varies
        dose_vol = np.zeros_like(ct_vol)

    widget = ViewingWidget()
    widget.set_data(ct_volume=ct_vol, quantity_volume=dose_vol)

    assert widget.quantity_widget._ct is not None
    # Check if quantity was set (ViewingWidget sets active quantity if available)
    assert widget.quantity_widget._active_quantity_name is not None


def test_viewing_widget_set_vois(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = ViewingWidget()
    widget.set_vois(cst.vois)

    assert len(widget.vois_widget._voi_checkboxes) == len(cst.vois)
    # Check if colors propagated to quantity widget
    assert len(widget.quantity_widget._voi_colors) > 0


def test_viewing_widget_signals(qapp):
    widget = ViewingWidget()

    # Test signal propagation from child widgets
    received = []
    widget.overlay_toggled.connect(lambda n, c: received.append((n, c)))

    # Simulate signal from vis_widget
    widget.vis_widget.overlay_toggled.emit("CT", False)

    assert len(received) > 0
    assert received[-1] == ("CT", False)


def test_viewing_widget_set_plane(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)

    widget = ViewingWidget()
    widget.set_data(ct_volume=ct_vol)

    widget.set_plane("Sagittal")
    assert widget.quantity_widget._plane == "Sagittal"

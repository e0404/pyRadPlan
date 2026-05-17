import numpy as np
import SimpleITK as sitk
from pyRadPlan.gui.widgets.result.quantity_widget import QuantityWidget


def test_quantity_widget_init(qapp):
    widget = QuantityWidget()
    assert widget is not None
    assert widget._plane == "Axial"


def test_quantity_widget_set_data(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    # Prepare data
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)
    dose_vol = np.swapaxes(result["physicalDose"], 0, 1)

    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol, quantity_volume=dose_vol)

    assert widget._ct is not None
    assert "Physical quantity" in widget._quantities
    assert widget._active_quantity_name == "Physical quantity"
    assert widget.slice_slider.isEnabled()


def test_quantity_widget_set_masks(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu)

    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol)

    masks = {}
    for voi in cst.vois:
        mask_arr = sitk.GetArrayFromImage(voi.mask)
        masks[voi.name] = mask_arr

    widget.set_masks(masks)

    assert len(widget._masks) > 0
    # Check if mask exists (using first VOI name)
    first_voi_name = cst.vois[0].name
    if first_voi_name in widget._masks:
        assert widget._masks[first_voi_name].shape == ct_vol.shape


def test_quantity_widget_plane_change(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)

    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol)

    widget.set_plane("Sagittal")
    assert widget._plane == "Sagittal"

    # Check slider range update
    axis = widget._PLANE_MAP["Sagittal"]
    assert widget.slice_slider.maximum() == ct_vol.shape[axis] - 1


def test_quantity_widget_visualization_options(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)
    dose_vol = np.swapaxes(result["physicalDose"], 0, 1)

    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol, quantity_volume=dose_vol)

    widget.set_isolines([10.0, 20.0])
    assert widget._isoline_levels == [10.0, 20.0]

    widget.set_opacity(0.8)
    assert widget._quantity_opacity == 0.8

    widget.set_active_mode("ct")
    assert widget._active_mode == "ct"

    widget.set_colormap("viridis", mode="quantity")
    assert widget._quantity_colormap == "viridis"


def test_quantity_widget_signals(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    ct_vol = sitk.GetArrayFromImage(ct.cube_hu).transpose(2, 1, 0)

    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol)

    # Test slice_changed signal
    received_slice = []
    widget.slice_changed.connect(received_slice.append)

    # Slider starts at mid (5 for size 10), so set to something else
    widget.slice_slider.setValue(0)
    assert len(received_slice) > 0
    assert received_slice[-1] == 0

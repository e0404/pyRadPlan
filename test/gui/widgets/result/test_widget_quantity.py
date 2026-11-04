import pytest
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


def test_new_image_opens_at_the_isocenter_slice(qapp):
    widget = QuantityWidget()
    widget.set_data(ct_volume=np.zeros((20, 10, 10)), isocenter=np.array([4.2, 3.0, 7.0]))
    assert widget.slice_slider.value() == 4

    widget.set_plane("Coronal")
    assert widget.slice_slider.value() == 7

    widget.set_data(ct_volume=np.zeros((20, 10, 10)))
    assert widget.slice_slider.value() == 5  # no isocenter: middle slice


def test_new_result_keeps_the_slice(qapp):
    ct_vol = np.zeros((20, 10, 10))
    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol, isocenter=np.array([4.0, 3.0, 7.0]))
    widget.slice_slider.setValue(12)

    widget.set_data(ct_volume=ct_vol, quantity_volume=np.ones_like(ct_vol))

    assert widget.slice_slider.value() == 12


def test_isocenter_arriving_with_the_same_image_positions_the_slice(qapp):
    ct_vol = np.zeros((20, 10, 10))
    widget = QuantityWidget()
    widget.set_data(ct_volume=ct_vol)
    assert widget.slice_slider.value() == 10

    widget.set_data(ct_volume=ct_vol, isocenter=np.array([4.0, 3.0, 7.0]))
    widget.set_isocenter(np.array([4.0, 3.0, 7.0]))
    assert widget.slice_slider.value() == 4

    widget.slice_slider.setValue(15)
    widget.set_data(ct_volume=ct_vol, isocenter=np.array([6.0, 3.0, 7.0]))
    assert widget.slice_slider.value() == 15


def test_isocenter_known_later_positions_the_slice_until_the_user_scrolls(qapp):
    widget = QuantityWidget()
    widget.set_data(ct_volume=np.zeros((20, 10, 10)))
    assert widget.slice_slider.value() == 10

    widget.set_isocenter(np.array([4.0, 3.0, 7.0]))
    assert widget.slice_slider.value() == 4

    widget.slice_slider.setValue(15)
    widget.set_isocenter(np.array([6.0, 3.0, 7.0]))
    assert widget.slice_slider.value() == 15


def test_isocenter_cross_and_scroll_hint_on_nearby_slices(qapp):
    from pyRadPlan.gui.widgets.result.quantity_widget import (
        _ISO_CROSS,
        _ISO_HALF_DOWN,
        _ISO_HALF_UP,
        _ISO_HINT_SLICES,
    )

    widget = QuantityWidget()
    widget.set_data(ct_volume=np.zeros((20, 10, 10)), isocenter=np.array([10.0, 3.0, 7.0]))
    widget.set_isocenter_visible(True)

    def symbol_at(idx):
        widget.slice_slider.setValue(idx)
        item = widget._isocenter_item
        return None if item is None else item.opts["symbol"]

    assert symbol_at(10) is _ISO_CROSS
    assert symbol_at(10 - _ISO_HINT_SLICES) is _ISO_HALF_UP  # scroll up to reach it
    assert symbol_at(10 + _ISO_HINT_SLICES) is _ISO_HALF_DOWN
    assert symbol_at(10 - _ISO_HINT_SLICES - 1) is None
    assert symbol_at(10 + _ISO_HINT_SLICES + 1) is None


def test_isocenter_and_beams_sit_on_the_isocenter_voxel_center(qapp):
    from pyRadPlan.gui.widgets.result.quantity_widget import _BeamWedgeItem

    # Viewer (z, x, y) voxel coordinates; the image draws voxel i over [i, i + 1].
    iso = np.array([10.0, 4.0, 7.0])
    widget = QuantityWidget()
    widget.set_data(ct_volume=np.zeros((21, 21, 21)), isocenter=iso)
    widget.set_ct_geometry((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    widget.set_plane("Axial")
    widget.set_isocenter_visible(True)
    widget.set_beams_visible(True)
    widget.set_beams([{"iso_center": iso, "source_direction": np.array([0.0, 1.0, 0.0])}])

    marker = widget._isocenter_item.points()[0].pos()
    assert (marker.x(), marker.y()) == pytest.approx((4.5, 7.5))
    wedge = next(item for item in widget._beam_items if isinstance(item, _BeamWedgeItem))
    assert (wedge._target.x(), wedge._target.y()) == pytest.approx((4.5, 7.5))
    # The source lies on the largest circle around the isocenter within the voxel centers.
    assert wedge._source.x() == pytest.approx(4.5 + 4.0)


def _beam_widget(spacing=(1.0, 1.0, 1.0)):
    widget = QuantityWidget()
    widget.set_data(ct_volume=np.zeros((21, 21, 21)))
    widget.set_ct_geometry((0.0, 0.0, 0.0), spacing)
    widget.set_beams_visible(True)
    return widget


def _show_beam(widget, direction):
    """Draw one beam through the volume centre; *direction* in viewer (z, x, y) voxels."""
    from pyRadPlan.gui.widgets.result.quantity_widget import _BeamWedgeItem

    widget.set_beams(
        [{"iso_center": np.array([10.0, 10.0, 10.0]), "source_direction": np.array(direction)}]
    )
    wedges = [item for item in widget._beam_items if isinstance(item, _BeamWedgeItem)]
    return wedges[0].widths if wedges else None


def test_beam_wedge_widens_towards_the_viewer(qapp):
    widget = _beam_widget()
    widget.set_plane("Axial")

    # In the axial view (seen from the feet) +z points into the screen.
    source_width, iso_width = _show_beam(widget, [0.0, 1.0, 0.0])
    assert source_width == pytest.approx(iso_width)
    source_width, iso_width = _show_beam(widget, [-0.5, 1.0, 0.0])
    assert source_width > iso_width  # source inferior: in front of the plane
    source_width, iso_width = _show_beam(widget, [0.5, 1.0, 0.0])
    assert source_width < iso_width  # source superior: behind the plane


def test_beam_wedge_uses_the_physical_angle(qapp):
    # The same voxel direction is steeper in mm when the slices are thicker.
    thin = _show_beam(_beam_widget(spacing=(1.0, 1.0, 1.0)), [-0.5, 1.0, 0.0])
    thick = _show_beam(_beam_widget(spacing=(1.0, 1.0, 3.0)), [-0.5, 1.0, 0.0])
    assert thick[0] > thin[0]


def test_beam_along_the_view_direction_gets_a_marker(qapp):
    import pyqtgraph as pg

    widget = _beam_widget()
    widget.set_plane("Axial")

    def marker_symbols():
        return [
            [str(spot.symbol()) for spot in item.points()]
            for item in widget._beam_items
            if isinstance(item, pg.ScatterPlotItem)
        ]

    assert _show_beam(widget, [1.0, 0.0, 0.0]) is None
    assert marker_symbols() == [["o", "o"]]  # source behind: beam comes out, ⊙
    assert _show_beam(widget, [-1.0, 0.0, 0.0]) is None
    assert marker_symbols() == [["o", "x"]]  # source in front: beam goes in, ⊗


def test_beam_wedge_paints(qapp):
    widget = _beam_widget()
    widget.resize(300, 300)
    _show_beam(widget, [-0.5, 1.0, 0.3])
    assert not widget._plot_widget.grab().isNull()


def test_beam_label_marks_the_out_of_plane_direction(qapp):
    import pyqtgraph as pg

    widget = _beam_widget()
    widget.set_plane("Axial")

    def label():
        (text,) = [i.toPlainText() for i in widget._beam_items if isinstance(i, pg.TextItem)]
        return text

    _show_beam(widget, [0.0, 1.0, 0.0])
    assert label() == "#0"
    _show_beam(widget, [-0.05, 1.0, 0.0])  # about 3 deg: the taper alone suffices
    assert label() == "#0"
    _show_beam(widget, [-0.5, 1.0, 0.0])  # source in front: the beam goes in
    assert label() == "#0 ⊗"
    _show_beam(widget, [0.5, 1.0, 0.0])  # source behind: the beam comes out
    assert label() == "#0 ⊙"

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

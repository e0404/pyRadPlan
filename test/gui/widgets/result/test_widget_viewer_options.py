from pyRadPlan.gui.widgets.result.viewer_options_widget import ViewerOptionsWidget


def test_viewer_options_init(qapp):
    widget = ViewerOptionsWidget()
    assert widget is not None
    assert widget.cmap_mode_btn.isChecked()  # Default is Quantity


def test_viewer_options_sync_ui(qapp):
    widget = ViewerOptionsWidget()

    # Sync for Quantity
    widget.sync_ui(
        mode="quantity", data_range=(0.0, 50.0), window_level=(25.0, 50.0), colormap="jet"
    )
    assert widget.wc_spin.value() == 25.0
    assert widget.ww_spin.value() == 50.0
    assert widget.cmap_combo.currentText() == "jet"
    assert not widget.ct_preset_combo.isEnabled()

    # Sync for CT
    widget.sync_ui(
        mode="ct", data_range=(-1000.0, 3000.0), window_level=(40.0, 400.0), colormap="gray"
    )
    assert widget.wc_spin.value() == 40.0
    assert widget.ww_spin.value() == 400.0
    assert widget.cmap_combo.currentText() == "gray"
    assert widget.ct_preset_combo.isEnabled()


def test_viewer_options_signals(qapp):
    widget = ViewerOptionsWidget()

    # Test colormap change
    received_cmap = []
    widget.colormap_changed.connect(lambda name, mode: received_cmap.append((name, mode)))

    widget.cmap_combo.setCurrentText("viridis")
    assert len(received_cmap) > 0
    assert received_cmap[-1][0] == "viridis"

    # Test window level change
    received_wl = []
    widget.window_level_changed.connect(lambda c, w, m: received_wl.append((c, w, m)))

    widget.wc_spin.setValue(100.0)
    assert len(received_wl) > 0
    assert received_wl[-1][0] == 100.0

    # Test opacity change
    received_opacity = []
    widget.opacity_changed.connect(received_opacity.append)

    widget.opacity_slider.setValue(50)
    assert len(received_opacity) > 0
    assert received_opacity[-1] == 0.5


def test_viewer_options_presets(qapp):
    widget = ViewerOptionsWidget()
    widget.sync_ui(mode="ct", data_range=(-1000, 1000), window_level=None, colormap="gray")

    widget.ct_preset_combo.setCurrentText("Lung")
    # Lung preset: -600, 1500
    assert widget.wc_spin.value() == -600.0
    assert widget.ww_spin.value() == 1500.0

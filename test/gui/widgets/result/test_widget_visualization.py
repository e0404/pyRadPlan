from pyRadPlan.gui.widgets.result.visualization_widget import VisualizationWidget


def test_visualization_widget_init(qapp):
    widget = VisualizationWidget()
    assert widget is not None
    assert widget.use_ct_checkbox.isChecked()
    assert widget.use_quantity_checkbox.isChecked()


def test_visualization_widget_signals(qapp):
    widget = VisualizationWidget()

    # Test overlay toggled
    received_overlay = []
    widget.overlay_toggled.connect(lambda name, state: received_overlay.append((name, state)))

    widget.use_ct_checkbox.setChecked(False)
    assert len(received_overlay) > 0
    assert received_overlay[-1] == ("CT", False)

    widget.use_quantity_checkbox.setChecked(False)
    assert received_overlay[-1] == ("quantity", False)

    # Test isolines toggled
    received_isolines = []
    widget.isolines_toggled.connect(received_isolines.append)

    widget.isolines_checkbox.setChecked(True)
    assert len(received_isolines) > 0
    assert received_isolines[-1] is True


def test_visualization_widget_quantity_selector(qapp):
    widget = VisualizationWidget()
    keys = [
        "physical_dose",
        "rbe_x_dose",
        "physical_dose_beam 0",
        "physical_dose_beam 1",
        "alpha",
    ]
    widget.update_quantity_selector(keys, active="physical_dose_beam 1")
    selector = widget.quantity_selector

    groups = [selector.quantity_combo.itemText(i) for i in range(selector.quantity_combo.count())]
    assert groups == ["dose", "RBExDose", "alpha"]
    components = [
        selector.component_combo.itemText(i) for i in range(selector.component_combo.count())
    ]
    assert components == ["full", "beam 0", "beam 1"]
    assert selector.current_key() == "physical_dose_beam 1"

    received = []
    widget.quantity_changed.connect(received.append)
    selector.component_combo.setCurrentIndex(0)
    selector.quantity_combo.setCurrentIndex(1)
    assert received == ["physical_dose", "rbe_x_dose"]
    # a group without beams disables the component selector
    assert not selector.component_combo.isEnabled()


def test_beams_toggle_requires_available_beams(qapp):
    widget = VisualizationWidget()
    # Nothing to draw before a plan defines gantry angles.
    assert not widget.beams_checkbox.isEnabled()

    received = []
    widget.beams_toggled.connect(received.append)

    widget.set_beams_available(True)
    assert widget.beams_checkbox.isEnabled()
    widget.beams_checkbox.setChecked(True)
    assert received[-1] is True

    widget.beams_checkbox.setChecked(False)
    assert received[-1] is False

    widget.set_beams_available(False)
    assert not widget.beams_checkbox.isEnabled()

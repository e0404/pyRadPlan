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


def _components(selector):
    combo = selector.component_combo
    return [combo.itemText(i) for i in range(combo.count())]


def test_quantity_selector_lists_labelled_scenarios(qapp):
    from pyRadPlan.gui.widgets._display_names import IDENTIFIER_ROLE

    widget = VisualizationWidget()
    keys = [
        "physical_dose",
        "physical_dose_beam 0",
        "physical_dose_scenario 0",
        "physical_dose_scenario 1",
        "physical_dose_scenario 3",
        "rbe_x_dose",
        "rbe_x_dose_scenario 1",
    ]
    widget.update_quantity_selector(keys, active="physical_dose_scenario 1")
    selector = widget.quantity_selector
    assert _components(selector) == ["full", "beam 0", "scenario 0", "scenario 1", "scenario 3"]
    assert selector.current_key() == "physical_dose_scenario 1"
    assert "scenario" in selector.component_combo.toolTip()

    received = []
    widget.quantity_changed.connect(received.append)
    widget.set_scenario_labels(["nominal", "setup (+3.0, 0.0, 0.0) mm", "derived"])
    assert _components(selector) == [
        "full",
        "beam 0",
        "scenario 0: nominal",
        "scenario 1: setup (+3.0, 0.0, 0.0) mm",
        "scenario 3",
    ]
    assert selector.current_key() == "physical_dose_scenario 1"
    assert selector.current_label() == "dose, scenario 1: setup (+3.0, 0.0, 0.0) mm"
    assert selector.component_combo.itemData(3, IDENTIFIER_ROLE) == "physical_dose_scenario 1"
    assert received == []
    assert selector.scenario_keys() == (
        "physical_dose_scenario 0",
        "physical_dose_scenario 1",
        "physical_dose_scenario 3",
    )

    # The chosen scenario is kept when switching to a quantity that has it.
    selector.quantity_combo.setCurrentIndex(1)
    assert received == ["rbe_x_dose_scenario 1"]
    assert selector.scenario_keys() == ("rbe_x_dose_scenario 1",)


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

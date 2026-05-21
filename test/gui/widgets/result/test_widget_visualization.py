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

    quantities = ["Dose 1", "Dose 2", "LET"]
    widget.update_quantity_selector(quantities, active="Dose 2")

    assert widget.quantity_selector.count() == 3
    assert widget.quantity_selector.currentText() == "Dose 2"

    # Test signal
    received_qty = []
    widget.quantity_changed.connect(received_qty.append)

    widget.quantity_selector.setCurrentText("LET")
    assert len(received_qty) > 0
    assert received_qty[-1] == "LET"

from pyRadPlan.gui.widgets.result.vois_widget import VOIsWidget


def test_vois_widget_init(qapp):
    widget = VOIsWidget()
    assert widget is not None


def test_vois_widget_set_vois(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    assert len(widget._voi_checkboxes) == len(cst.vois)

    # Check if some are selected by default (heuristics)
    selected = widget.selected_vois()
    assert len(selected) > 0


def test_vois_widget_signals(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    # Test selection changed
    received_selection = []
    widget.selection_changed.connect(received_selection.append)

    # Toggle first VOI
    first_voi_name = cst.vois[0].name
    cb = widget._voi_checkboxes[first_voi_name]

    # Flip state
    new_state = not cb.isChecked()
    cb.setChecked(new_state)

    assert len(received_selection) > 0
    if new_state:
        assert first_voi_name in received_selection[-1]
    else:
        assert first_voi_name not in received_selection[-1]


def test_vois_widget_colors(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    colors = widget.get_voi_colors()
    assert len(colors) == len(cst.vois)
    assert isinstance(colors[cst.vois[0].name], tuple)
    assert len(colors[cst.vois[0].name]) == 3

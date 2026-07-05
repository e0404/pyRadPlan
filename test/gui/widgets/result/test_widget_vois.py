import pytest

from PySide6.QtWidgets import QDialog, QGroupBox

from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog, VOIsWidget


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


def test_vois_widget_tooltips(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    voi = cst.vois[0]
    tooltip = widget._voi_checkboxes[voi.name].toolTip()
    assert voi.name in tooltip
    assert voi.voi_type in tooltip
    assert "Overlap priority" in tooltip
    assert str(voi.overlap_priority) in tooltip
    assert f"{voi.alpha_x:g}" in tooltip
    assert f"{voi.beta_x:g}" in tooltip


def _group_titles(widget: VOIsWidget) -> list[str]:
    """Titles of the group boxes currently held by the VOIs layout."""
    layout = widget._vois_layout
    return [
        item.widget().title()
        for item in (layout.itemAt(i) for i in range(layout.count()))
        if isinstance(item.widget(), QGroupBox)
    ]


def test_vois_widget_group_by_overlap(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    selected_before = set(widget.selected_vois())

    widget._group_combo.setCurrentIndex(1)

    assert len(widget._voi_checkboxes) == len(cst.vois)
    assert set(widget.selected_vois()) == selected_before

    priorities = sorted({int(v.overlap_priority) for v in cst.vois})
    assert _group_titles(widget) == [f"Overlap priority {p}" for p in priorities]

    # Switching back restores the type grouping
    widget._group_combo.setCurrentIndex(0)
    assert not any(t.startswith("Overlap priority") for t in _group_titles(widget))
    assert len(widget._voi_checkboxes) == len(cst.vois)


def test_voi_metadata_dialog_applies_values(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]

    dialog = VOIMetadataDialog(voi)
    dialog.alpha_spin.setValue(0.2)
    dialog.beta_spin.setValue(0.02)
    dialog.priority_spin.setValue(3)
    dialog.accept()

    assert voi.alpha_x == pytest.approx(0.2)
    assert voi.beta_x == pytest.approx(0.02)
    assert voi.overlap_priority == 3


def test_vois_widget_edit_emits_metadata_changed(qapp, test_data_photons, monkeypatch):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    monkeypatch.setattr(VOIMetadataDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    received = []
    widget.metadata_changed.connect(received.append)

    name = cst.vois[0].name
    widget._edit_voi(name)

    assert received == [name]
    assert len(widget._voi_checkboxes) == len(cst.vois)


def test_voi_metadata_dialog_changes_type(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]

    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)

    dialog = VOIMetadataDialog(voi)
    dialog.type_combo.setCurrentText(new_type)
    dialog.accept()

    new_voi = dialog.updated_voi
    assert new_voi is not voi
    assert new_voi.voi_type == new_type
    assert new_voi.name == voi.name
    assert new_voi.mask is voi.mask
    assert new_voi.alpha_x == pytest.approx(voi.alpha_x)
    assert new_voi.overlap_priority == voi.overlap_priority
    assert new_voi.objectives == voi.objectives
    # The original VOI is untouched
    assert voi.voi_type != new_type


def test_vois_widget_type_change_emits_voi_replaced(qapp, test_data_photons, monkeypatch):
    ct, cst, result = test_data_photons

    widget = VOIsWidget()
    widget.set_vois(cst.vois)

    voi = cst.vois[0]
    new_type = next(t for t in ("TARGET", "OAR") if t != voi.voi_type)

    def fake_exec(self):
        self.type_combo.setCurrentText(new_type)
        self.accept()
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(VOIMetadataDialog, "exec", fake_exec)

    replaced = []
    widget.voi_replaced.connect(lambda n, v: replaced.append((n, v)))

    widget._edit_voi(voi.name)

    assert len(replaced) == 1
    name, new_voi = replaced[0]
    assert name == voi.name
    assert new_voi is not voi
    assert new_voi.voi_type == new_type
    assert widget._voi_by_name[voi.name] is new_voi
    assert len(widget._voi_checkboxes) == len(cst.vois)

import math
from types import SimpleNamespace

import numpy as np
import pytest

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog, QGroupBox

from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog, VOIsWidget
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)


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
    assert f"Tissue model: {voi.tissue_properties.model}" in tooltip


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

    # The VOI is recreated: it mirrors alpha_x/beta_x of its tissue properties and
    # rejects assignments that leave the two inconsistent.
    updated = dialog.updated_voi
    assert updated is not voi
    assert updated.alpha_x == pytest.approx(0.2)
    assert updated.beta_x == pytest.approx(0.02)
    assert updated.tissue_properties.alpha_x == pytest.approx(0.2)
    assert updated.tissue_properties.beta_x == pytest.approx(0.02)
    assert updated.overlap_priority == 3
    assert type(updated.tissue_properties) is TissueProperties


def test_voi_metadata_dialog_switches_to_mkm_tissue(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]

    dialog = VOIMetadataDialog(voi)
    assert dialog.tissue_combo.currentText() == "Generic"
    assert set(dialog._tissue_spins) == {"alpha_x", "beta_x"}

    dialog.tissue_combo.setCurrentText("MKM")
    assert set(dialog._tissue_spins) == {
        "alpha_x",
        "beta_x",
        "r_nucleus",
        "r_domain",
        "alpha_0",
        "beta_0",
    }
    for name, value in (
        ("r_nucleus", 3.9),
        ("r_domain", 0.32),
        ("alpha_0", 0.172),
        ("beta_0", 0.0615),
    ):
        dialog._tissue_spins[name].setValue(value)
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.model == "MKM"
    assert tissue.r_nucleus == pytest.approx(3.9)
    assert tissue.r_domain == pytest.approx(0.32)
    assert tissue.alpha_0 == pytest.approx(0.172)
    assert tissue.beta_0 == pytest.approx(0.0615)


def test_voi_metadata_dialog_groups_model_and_reference_params(qapp, test_data_photons):
    from pyRadPlan.gui.widgets._tissue_params import MODEL_GROUP_TITLE, REFERENCE_GROUP_TITLE

    _ct, cst, _result = test_data_photons
    dialog = VOIMetadataDialog(cst.vois[0])

    def group_of(name):
        spin = dialog._tissue_spins[name]
        return next(
            group
            for group in (dialog.model_group, dialog.reference_group)
            if group.isAncestorOf(spin)
        )

    assert (
        dialog.reference_group.title() == REFERENCE_GROUP_TITLE == "Reference radiation (for RBE)"
    )
    # The reference-only tissue has no model parameters
    assert dialog.model_group.isHidden()
    assert group_of("alpha_x") is dialog.reference_group

    dialog.tissue_combo.setCurrentText("LEM")
    assert dialog.model_group.title() == f"{MODEL_GROUP_TITLE} · LEM"
    assert group_of("d_t") is dialog.model_group
    assert group_of("beta_x") is dialog.reference_group
    assert "model takes them as inputs" in dialog.reference_note.text()
    assert dialog.derive_button is None

    dialog.tissue_combo.setCurrentText("mMKM")
    for name in ("r_nucleus", "r_domain", "alpha_0", "beta_0"):
        assert group_of(name) is dialog.model_group
    note = dialog.reference_note.text()
    assert "z*_1D,ref" in note and "200 kV X-rays" in note and "4.2 keV/µm" in note
    assert "200 kV X-rays" in dialog.alpha_spin.toolTip()
    assert dialog.derive_button is not None and dialog.derive_button.text() == "Derive from model"


def test_voi_metadata_dialog_derives_reference_params(qapp, test_data_photons):
    from pyRadPlan.gui.widgets._tissue_params import param_source
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _ct, cst, _result = test_data_photons

    # A structure with explicit reference parameters keeps them until derived on request
    explicit = VOIMetadataDialog(_with_tissue(cst.vois[0], TissueProperties(alpha_x=0.2)))
    explicit.tissue_combo.setCurrentText("MKM")
    for name, value in (("alpha_0", 0.313), ("beta_0", 0.0615), ("r_domain", 0.34)):
        explicit._tissue_spins[name].setValue(value)
    assert explicit.alpha_spin.value() == pytest.approx(0.2)
    assert explicit.derive_reference_params()["alpha_x"] > 0.4
    assert param_source(explicit.alpha_spin)[0] == "estimated"

    dialog = VOIMetadataDialog(_with_tissue(cst.vois[0], TissueProperties()))
    dialog.tissue_combo.setCurrentText("MKM")

    # Derived live while the model parameters are entered
    dialog._tissue_spins["alpha_0"].setValue(0.313)
    dialog._tissue_spins["beta_0"].setValue(0.0615)
    dialog._tissue_spins["r_domain"].setValue(0.34)
    expected = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.313, beta_0=0.0615, r_domain=0.34
    )
    assert dialog.alpha_spin.value() == pytest.approx(expected["alpha_x"], abs=1e-4)
    assert dialog.beta_spin.value() == pytest.approx(expected["beta_x"], abs=1e-4)
    source, note = param_source(dialog.alpha_spin)
    assert source == "estimated" and "z_1D,ref" in note and "200 kV X-rays" in note
    assert "italic" in dialog.alpha_spin.styleSheet()

    # An override is kept while the model parameters change, until derived again
    dialog.alpha_spin.setValue(0.3)
    dialog._tissue_spins["alpha_0"].setValue(0.2)
    assert dialog.alpha_spin.value() == pytest.approx(0.3)
    assert param_source(dialog.alpha_spin)[0] == "explicit"
    dialog.derive_button.click()
    derived = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.2, beta_0=0.0615, r_domain=0.34
    )
    assert dialog.alpha_spin.value() == pytest.approx(derived["alpha_x"], abs=1e-4)
    assert param_source(dialog.alpha_spin)[0] == "estimated"

    # mMKM derives with the saturation-corrected z*_1D once r_nucleus is known
    dialog._tissue_spins["r_nucleus"].setValue(4.1)
    dialog.tissue_combo.setCurrentText("mMKM")
    mmkm = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.2, beta_0=0.0615, r_domain=0.34, r_nucleus=4.1, model="mMKM"
    )
    assert dialog.alpha_spin.value() == pytest.approx(mmkm["alpha_x"], abs=1e-4)
    assert "z*_1D,ref" in param_source(dialog.alpha_spin)[1]

    dialog.accept()
    tissue = dialog.updated_voi.tissue_properties
    assert tissue.model == "mMKM"
    assert tissue.reference_source == "estimated"
    assert tissue.alpha_x == pytest.approx(mmkm["alpha_x"], abs=1e-9)
    assert tissue.beta_x == pytest.approx(mmkm["beta_x"], abs=1e-9)


def test_voi_metadata_dialog_reports_missing_baseline_params(qapp, test_data_photons, monkeypatch):
    _ct, cst, _result = test_data_photons

    shown: list[str] = []
    monkeypatch.setattr(
        "pyRadPlan.gui.widgets.result.vois_widget.QMessageBox.information",
        lambda *args, **kwargs: shown.append(args[2]),
    )

    dialog = VOIMetadataDialog(cst.vois[0])
    dialog.tissue_combo.setCurrentText("MKM")
    # beta_0 and r_domain are left unset, so nothing is derivable
    dialog._tissue_spins["alpha_0"].setValue(0.313)
    assert dialog.derive_reference_params() == {}
    assert shown and "model parameters" in shown[0]


def test_voi_metadata_dialog_keeps_values_across_tissue_models(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    dialog.alpha_spin.setValue(0.3)
    dialog.tissue_combo.setCurrentText("LEM")

    assert dialog.alpha_spin.value() == pytest.approx(0.3)


def _class_picks(dialog):
    """Tissue classes of the base data the tissue picker offers."""
    return [pick for pick in dialog._picks if pick.kind == "class"]


def _preset_index(dialog, cell_line, model):
    """Index of a cell line preset in the editor's list of presets."""
    return next(
        i
        for i, preset in enumerate(dialog._presets)
        if preset[0].group == (cell_line.upper(), model)
    )


def _machine_stub():
    """Machine stand-in holding its tissue classes as an array, like real machines do."""
    return SimpleNamespace(
        name="Generic",
        tissue_properties=np.asarray(
            [
                LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0),
                MKMTissueProperties(r_nucleus=3.9, r_domain=0.32),
            ],
            dtype=object,
        ),
    )


def test_voi_metadata_dialog_applies_machine_tissue_class(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    machine = _machine_stub()

    dialog = VOIMetadataDialog(cst.vois[0], machine)
    classes = _class_picks(dialog)
    assert len(classes) == 2
    assert "r_domain = 0.32" in classes[1].label

    dialog.alpha_spin.setValue(0.2)
    dialog.beta_spin.setValue(0.04)
    dialog.apply_machine_tissue(1)
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.r_nucleus == pytest.approx(3.9)
    assert tissue.r_domain == pytest.approx(0.32)
    # The MKM base data leaves the reference parameters open, so they survive
    assert tissue.alpha_x == pytest.approx(0.2)
    assert tissue.beta_x == pytest.approx(0.04)
    # Picking a base data class is the way to obtain a tissue the model can match
    assert tissue.matches(machine.tissue_properties[1])


def test_voi_metadata_dialog_applies_machine_lem_class(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    machine = _machine_stub()

    dialog = VOIMetadataDialog(cst.vois[0], machine)
    dialog.apply_machine_tissue(0)
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, LEMTissueProperties)
    assert tissue.d_t == pytest.approx(30.0)
    assert tissue.r_nucleus == pytest.approx(5.0)
    assert tissue.alpha_x == pytest.approx(0.1)
    assert tissue.matches(machine.tissue_properties[0])


def test_voi_metadata_dialog_applies_cell_line_preset(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    assert dialog.preset_combo is not None
    labels = [dialog.preset_combo.itemText(i) for i in range(dialog.preset_combo.count())]
    # Grouped by provenance, each preset named with its source
    assert labels[0] == "Presets · LEM I"
    assert labels[1].startswith("HSG · LEM I (Elsässer 2007)")
    # The LEM variants share the model name "LEM", so they are told apart by the label
    assert any(label.startswith("HSG · LEM II (Elsässer 2007)") for label in labels)

    # A preset names its source in the label, in full with its notes in the tooltip
    index = labels.index(next(label for label in labels if label.startswith("CHO · MKM")))
    assert labels[index].startswith("CHO · MKM (Kase 2008)")
    tooltip = dialog.preset_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    assert "Source: Kase et al." in tooltip and "Sect. 3.3 (p. 52)" in tooltip
    assert "α_x = 0.228, β_x = 0.02 (published, Elsässer 2007)" in tooltip
    # A published preset names its source and the reference radiation it refers to
    hsg = labels.index(next(label for label in labels if label.startswith("HSG · mMKM")))
    assert labels[hsg].startswith("HSG · mMKM (Inaniwa 2010, derived reference)")
    assert labels[hsg + 1].startswith("HSG · mMKM (Inaniwa 2010 / Furusawa 2000 reference)")
    hsg_tooltip = dialog.preset_combo.itemData(hsg, Qt.ItemDataRole.ToolTipRole)
    assert "10.1088/0031-9155/55/22/008" in hsg_tooltip
    assert "derived from the model for 200 kV X-rays (y_D = 4.2 keV/µm)" in hsg_tooltip
    lem_index = labels.index(next(label for label in labels if label.startswith("HSG · LEM I ")))
    lem_tooltip = dialog.preset_combo.itemData(lem_index, Qt.ItemDataRole.ToolTipRole)
    assert "Phys. Med. Biol. 63 (2018) 08NT01" in lem_tooltip

    dialog.apply_cell_line_preset(_preset_index(dialog, "CHO", "MKM"))
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.cell_line == "CHO"
    assert "10.1088/0031-9155/53/1/003" in tissue.cell_line_source
    # A preset describes a whole tissue, so every parameter of its model is taken over,
    # with the published reference parameters as they are
    assert tissue.r_nucleus == pytest.approx(4.1)
    assert tissue.r_domain == pytest.approx(0.2)
    assert tissue.alpha_0 == pytest.approx(0.228)
    assert tissue.beta_0 == pytest.approx(0.02)
    assert tissue.reference_sources == {"alpha_x": "published", "beta_x": "published"}
    assert (tissue.alpha_x, tissue.beta_x) == (0.228, 0.02)

    # "Derive from model" switches them to the derived values
    dialog = VOIMetadataDialog(dialog.updated_voi)
    assert "published (Elsässer 2007)" in dialog.alpha_spin.toolTip()
    assert "italic" not in dialog.alpha_spin.styleSheet()
    dialog.derive_button.click()
    dialog.accept()
    tissue = dialog.updated_voi.tissue_properties
    assert tissue.reference_source == "estimated"
    assert tissue.alpha_x == pytest.approx(0.228 + 0.02 * 0.1602 * 4.2 / (math.pi * 0.2**2))
    assert tissue.beta_x == pytest.approx(0.02)


def test_voi_metadata_dialog_preset_switches_tissue_model(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    assert dialog.tissue_combo.currentText() == "Generic"

    dialog.apply_cell_line_preset(_preset_index(dialog, "V79", "LEM_II"))
    assert dialog.tissue_combo.currentText() == "LEM"
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, LEMTissueProperties)
    assert tissue.cell_line == "V79"
    assert tissue.d_t == pytest.approx(5.5)
    assert tissue.r_nucleus == pytest.approx(4.0)


def test_voi_metadata_dialog_without_machine_has_no_picker(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dialog = VOIMetadataDialog(cst.vois[0])
    assert _class_picks(dialog) == []
    # Nothing to compare against, so the base data row stays hidden
    assert dialog._comparison_row.isHidden()


def test_vois_widget_passes_machine_to_editor(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    machine = _machine_stub()
    captured = []

    widget = VOIsWidget()
    widget.set_vois(cst.vois)
    widget.set_machine_provider(lambda: machine)

    def fake_exec(self):
        captured.append(_class_picks(self))
        return QDialog.DialogCode.Rejected

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VOIMetadataDialog, "exec", fake_exec)
        widget._edit_voi(cst.vois[0].name)

    assert len(captured) == 1
    assert len(captured[0]) == 2


def test_vois_widget_passes_the_dose_engine_to_editor(qapp, test_data_photons):
    from pyRadPlan.dose.engines import ParticleTOPASMCEngine

    ct, cst, result = test_data_photons
    captured = []
    widget = VOIsWidget()
    widget.set_vois(cst.vois)
    widget.set_engine_provider(lambda: ParticleTOPASMCEngine)

    def fake_exec(self):
        captured.append(self._engine)
        return QDialog.DialogCode.Rejected

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VOIMetadataDialog, "exec", fake_exec)
        widget._edit_voi(cst.vois[0].name)

    assert captured == [ParticleTOPASMCEngine]


def test_voi_metadata_dialog_leaves_optional_tissue_parameters_unset(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    dialog.tissue_combo.setCurrentText("LEM")
    dialog._tissue_spins["r_nucleus"].setValue(5.0)  # d_t left at its "unset" minimum
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert isinstance(tissue, LEMTissueProperties)
    assert tissue.r_nucleus == pytest.approx(5.0)
    assert math.isnan(tissue.d_t)


def test_voi_metadata_dialog_edits_the_lem_version(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    # The reference-only tissue has no parameter with a fixed set of choices
    assert dialog._tissue_choices == {}

    dialog.tissue_combo.setCurrentText("LEM")
    assert set(dialog._tissue_choices) == {"version"}
    dialog._tissue_choices["version"].setCurrentText("III")
    dialog._tissue_spins["d_t"].setValue(19.0)
    dialog._tissue_spins["r_nucleus"].setValue(5.0)
    dialog.accept()

    tissue = dialog.updated_voi.tissue_properties
    assert type(tissue) is LEMTissueProperties
    assert tissue.version == "III"
    assert tissue.d_t == pytest.approx(19.0)


def test_voi_metadata_dialog_keeps_the_version_across_tissue_models(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    dialog.tissue_combo.setCurrentText("LEM")
    dialog._tissue_choices["version"].setCurrentText("II")
    dialog.tissue_combo.setCurrentText("MKM")
    dialog.tissue_combo.setCurrentText("LEM")

    assert dialog._tissue_choices["version"].currentText() == "II"


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


def _with_tissue(voi, tissue):
    """Copy of a VOI carrying the given tissue properties."""
    from pyRadPlan.cst import create_voi

    data = dict(voi)
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    return create_voi(data)


def test_voi_metadata_dialog_unchanged_keeps_the_voi(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]

    dialog = VOIMetadataDialog(voi)
    dialog.accept()

    assert dialog.updated_voi is voi


def test_voi_metadata_dialog_keeps_untouched_values_exactly(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = _with_tissue(cst.vois[0], TissueProperties(alpha_x=0.123456, beta_x=0.054321))

    dialog = VOIMetadataDialog(voi)
    # Switching the model back and forth must not round the values either
    dialog.tissue_combo.setCurrentText("LEM")
    dialog.tissue_combo.setCurrentText("Generic")
    dialog.accept()

    assert dialog.updated_voi is voi
    assert dialog.tissue_properties().alpha_x == 0.123456


def test_voi_metadata_dialog_unchanged_keeps_a_tissue_with_unset_parameters(
    qapp, test_data_photons
):
    from pyRadPlan.biology.tissues import MKMTissueProperties

    ct, cst, result = test_data_photons
    tissue = MKMTissueProperties(alpha_x=0.1, beta_x=0.05, r_nucleus=3.9, r_domain=0.32)
    voi = _with_tissue(cst.vois[0], tissue)

    dialog = VOIMetadataDialog(voi)
    dialog.accept()

    assert dialog.updated_voi is voi


def test_voi_metadata_dialog_tissue_form_has_no_leftover_rows(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])

    def rows():
        return dialog._model_form.rowCount() + dialog._reference_form.rowCount()

    before = rows()
    dialog.tissue_combo.setCurrentText("MKM")
    dialog.tissue_combo.setCurrentText("Generic")

    assert rows() == before


def test_voi_metadata_dialog_drops_the_cell_line_of_a_changed_tissue(qapp, test_data_photons):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0])
    dialog.apply_cell_line_preset(_preset_index(dialog, "CHO", "MKM"))
    assert dialog.tissue_properties().cell_line == "CHO"

    # Another model no longer describes the cell line
    dialog.tissue_combo.setCurrentText("Generic")
    assert dialog.tissue_properties().cell_line is None

    # Neither do changed parameters of its own model
    dialog.tissue_combo.setCurrentText("MKM")
    assert dialog.tissue_properties().cell_line == "CHO"
    dialog._tissue_spins["r_domain"].setValue(0.3)
    assert dialog.tissue_properties().cell_line is None


def test_voi_editor_unchanged_ok_leaves_the_workspace_untouched(qapp, test_data_photons):
    from pyRadPlan.gui.widgets._result_widget import ViewingWidget
    from pyRadPlan.gui.widgets.workflow import WorkflowWidget
    from pyRadPlan.gui.workspace import WorkspaceManager

    ct, cst, _ = test_data_photons
    ws = WorkspaceManager()
    workflow = WorkflowWidget(ws)
    viewing = ViewingWidget(ws)
    ws.set_many(ct=ct, cst=cst, pln=object(), stf=object(), dij=object(), result={})
    assert not workflow._result_stale

    changes = []
    ws.workspace_changed.connect(changes.append)

    def fake_exec(self):
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VOIMetadataDialog, "exec", fake_exec)
        viewing.vois_widget._edit_voi(cst.vois[0].name)

    assert changes == []
    assert not workflow._result_stale
    assert not workflow._dij_stale


def _answer_message_boxes(monkeypatch, button_text=None):
    """Answer every message box by clicking `button_text`; fail on one without it."""
    from PySide6.QtWidgets import QMessageBox

    shown = []

    def exec_(box):
        shown.append(f"{box.text()}\n{box.informativeText()}")
        assert button_text is not None, f"unexpected message box: {shown[-1]}"
        next(b for b in box.buttons() if b.text() == button_text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return shown


def _oxygen_machine():
    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly

    return load_machine_quietly("oxygen", "Generic")


def test_voi_metadata_dialog_ranks_the_provenance_of_the_selected_model_first(
    qapp, test_data_photons
):
    ct, cst, result = test_data_photons

    dialog = VOIMetadataDialog(cst.vois[0], _oxygen_machine(), "zstar_kernels")
    assert [t.cell_line for t in dialog._machine_tissues] == ["Generic_MKM"]
    # Only the two HSG · mMKM presets are of the mMKM provenance of the z* kernels, and both
    # match the class exactly
    preferred = [pick for pick in dialog._picks if pick.preferred]
    assert [pick.kind for pick in preferred] == ["class", "preset", "preset"]
    assert all(pick.exact for pick in preferred)
    texts = [dialog.preset_combo.itemText(i) for i in range(dialog.preset_combo.count())]
    assert texts[1].startswith("✓ Generic_MKM · mMKM (base data)")
    assert texts[2].startswith("✓ HSG · mMKM (Inaniwa 2010, derived reference)")
    assert texts[3].startswith("✓ HSG · mMKM (Inaniwa 2010 / Furusawa 2000 reference)")
    assert texts.index("Other presets · MKM") > 4

    dialog = VOIMetadataDialog(cst.vois[0], _oxygen_machine(), "LEM")
    preferred = [pick for pick in dialog._picks if pick.preferred]
    assert {pick.provenance for pick in preferred if pick.kind == "preset"} == {"LEM_I"}
    assert len(_class_picks(dialog)) == 2


@pytest.mark.parametrize(("answer", "applied"), [("Keep editing", False), ("Apply anyway", True)])
def test_voi_metadata_dialog_asks_before_applying_a_failing_tissue(
    qapp, monkeypatch, test_data_photons, answer, applied
):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]
    dialog = VOIMetadataDialog(voi, _oxygen_machine(), "zstar_kernels", cst=cst.vois)
    dialog.alpha_spin.setValue(0.2)
    shown = _answer_message_boxes(monkeypatch, answer)

    dialog.accept()

    assert len(shown) == 1 and voi.name in shown[0]
    # Only the edited structure's issue is reported, though the others fail as well
    assert not any(other.name + ":" in shown[0] for other in cst.vois[1:])
    assert dialog.tissue_combo.styleSheet()
    assert "No matching tissue class" in dialog.tissue_combo.toolTip()
    assert (dialog.result() == QDialog.DialogCode.Accepted) == applied
    assert (dialog.updated_voi is not voi) == applied


def test_voi_metadata_dialog_checks_with_the_edited_voi_substituted(
    qapp, monkeypatch, test_data_photons
):
    from pyRadPlan.gui.widgets._tissue_params import tissue_issues_quietly
    from pyRadPlan.gui.widgets.result import vois_widget

    ct, cst, result = test_data_photons
    checked = []

    def recording_check(machine, bio_model, structures, engine=None):
        checked.append(list(structures))
        return tissue_issues_quietly(machine, bio_model, structures, engine)

    monkeypatch.setattr(vois_widget, "tissue_issues_quietly", recording_check)
    machine = _oxygen_machine()
    dialog = VOIMetadataDialog(cst.vois[0], machine, "LEM", cst=cst.vois)
    dialog.apply_machine_tissue(1)
    _answer_message_boxes(monkeypatch)  # the LEM I class matches, nothing to ask

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    (structures,) = checked
    assert len(structures) == len(cst.vois)
    edited = dict(structures)[cst.vois[0].name]
    assert edited.matches(machine.tissue_properties[2])


def _mkm_voi(voi, model="MKM", r_domain=0.32):
    tissue = MKMTissueProperties(
        model=model,
        alpha_x=0.2,
        beta_x=0.05,
        r_nucleus=3.9,
        r_domain=r_domain,
        alpha_0=0.2,
        beta_0=0.05,
    )
    return _with_tissue(voi, tissue)


def test_voi_metadata_dialog_adopts_a_preset_of_another_provenance(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dialog = VOIMetadataDialog(cst.vois[0], _oxygen_machine(), "zstar_kernels", cst=cst.vois)

    note = dialog.apply_cell_line_preset(_preset_index(dialog, "CHO", "MKM"))

    assert note == "values from CHO · MKM; adopted as mMKM"
    assert dialog.status_label.text() == note
    assert dialog.tissue_combo.currentText() == "mMKM"
    assert dialog.tissue_properties().model == "mMKM"

    # Overwriting the preset with the base data class then matches the base data
    dialog.apply_machine_tissue(0)
    tissue = dialog.tissue_properties()
    assert tissue.model == "mMKM" and tissue.r_domain == pytest.approx(0.32)
    assert tissue.alpha_0 == pytest.approx(0.228)
    dialog.refresh_comparison()
    assert dialog.comparison().matched


def test_voi_metadata_dialog_tints_parameters_differing_from_the_base_data(
    qapp, test_data_photons
):
    from pyRadPlan.gui.widgets._tissue_params import (
        PROVENANCE_WARNING_COLOR,
        editor_mismatch,
        editor_mismatch_is_warning,
    )

    ct, cst, result = test_data_photons
    voi = _mkm_voi(cst.vois[0], model="mMKM", r_domain=0.3)
    dialog = VOIMetadataDialog(voi, _oxygen_machine(), "zstar_kernels", cst=cst.vois)

    spins = dialog._tissue_spins
    assert editor_mismatch(spins["r_domain"]) == "base data: r_domain = 0.32"
    assert editor_mismatch(spins["r_nucleus"]) == ""
    for name in ("alpha_x", "beta_x", "alpha_0", "beta_0"):
        assert editor_mismatch(spins[name]) == ""  # undeclared by the class
    assert editor_mismatch(dialog.tissue_combo) == ""
    assert "Closest base data class" in dialog.comparison_label.text()

    # Fixing the value clears the tint once the edits come to a rest
    spins["r_domain"].setValue(0.32)
    dialog.refresh_comparison()
    assert editor_mismatch(dialog._tissue_spins["r_domain"]) == ""
    assert dialog.comparison_label.text().startswith("Matches the base data class")

    # Another provenance label tints the model box softly: a warning only
    dialog.tissue_combo.setCurrentText("MKM")
    dialog.refresh_comparison()
    assert editor_mismatch(dialog.tissue_combo).startswith("base data: mMKM (tissue: MKM)")
    assert editor_mismatch_is_warning(dialog.tissue_combo)
    assert PROVENANCE_WARNING_COLOR in dialog.tissue_combo.styleSheet()
    assert dialog.comparison().matched
    assert dialog.comparison_label.text().startswith("Matches the base data class")


def test_voi_metadata_dialog_match_base_data(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = _mkm_voi(cst.vois[0], r_domain=0.3)
    dialog = VOIMetadataDialog(voi, _oxygen_machine(), "zstar_kernels", cst=cst.vois)
    assert dialog.match_button.isEnabled()

    dialog.match_button.click()

    tissue = dialog.tissue_properties()
    assert tissue.model == "mMKM" and tissue.r_domain == pytest.approx(0.32)
    assert tissue.alpha_x == pytest.approx(0.2)  # explicit, kept
    assert dialog.comparison().matched


def _mkm_labelled_dialog(cst):
    dialog = VOIMetadataDialog(cst.vois[0], _oxygen_machine(), "zstar_kernels", cst=cst.vois)
    dialog.tissue_combo.setCurrentText("MKM")
    for name, value in (
        ("r_nucleus", 3.9),
        ("r_domain", 0.32),
        ("alpha_0", 0.2),
        ("beta_0", 0.05),
    ):
        dialog._tissue_spins[name].setValue(value)
    dialog.refresh_comparison()
    return dialog


def test_voi_metadata_dialog_accepts_a_differing_model_label_without_asking(
    qapp, monkeypatch, test_data_photons
):
    from pyRadPlan.gui.widgets._tissue_params import editor_mismatch_is_warning

    ct, cst, result = test_data_photons
    dialog = _mkm_labelled_dialog(cst)
    _answer_message_boxes(monkeypatch)  # fails if a box is shown

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    tissue = dialog.updated_voi.tissue_properties
    assert tissue.model == "MKM" and tissue.alpha_0 == pytest.approx(0.2)
    # The check highlights the model box as a warning, not an error
    assert editor_mismatch_is_warning(dialog.tissue_combo)
    assert "is labelled MKM" in dialog.tissue_combo.toolTip()


def test_voi_metadata_dialog_relabels_as_the_base_data_model(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    dialog = _mkm_labelled_dialog(cst)
    assert not dialog.relabel_button.isHidden()
    assert dialog.relabel_button.text() == "Relabel as mMKM"

    dialog.relabel_button.click()

    tissue = dialog.tissue_properties()
    assert tissue.model == "mMKM" and tissue.alpha_0 == pytest.approx(0.2)
    assert not dialog.comparison().provenance_mismatch
    assert dialog.relabel_button.isHidden()
    assert dialog.relabel() is False


def test_voi_metadata_dialog_shows_where_reference_values_come_from(qapp, test_data_photons):
    from pyRadPlan.gui.widgets._tissue_params import param_source

    ct, cst, result = test_data_photons
    voi = _with_tissue(cst.vois[0], TissueProperties())

    dialog = VOIMetadataDialog(voi)
    assert param_source(dialog.alpha_spin) == ("default", "default 0.1 / 0.05")
    assert "italic" in dialog.alpha_spin.styleSheet()

    # The source survives a model switch, and untouched values keep it
    dialog.tissue_combo.setCurrentText("MKM")
    assert param_source(dialog.alpha_spin)[0] == "default"
    dialog._tissue_spins["r_domain"].setValue(0.32)
    assert dialog.tissue_properties().reference_source == "default"

    for name, value in (("alpha_0", 0.3), ("beta_0", 0.05)):
        dialog._tissue_spins[name].setValue(value)
    assert param_source(dialog.alpha_spin) == (
        "estimated",
        "derived: α_0 + β_0 · z_1D,ref of 200 kV X-rays (y_D = 4.2 keV/µm)",
    )
    assert param_source(dialog.beta_spin) == ("estimated", "derived: β_0")
    dialog.accept()
    assert dialog.updated_voi.tissue_properties.reference_source == "estimated"

    dialog = VOIMetadataDialog(voi)
    dialog.alpha_spin.setValue(0.3)
    assert param_source(dialog.alpha_spin) == ("explicit", "")
    assert "italic" not in dialog.alpha_spin.styleSheet()
    # The untouched beta_x still holds the default, which the tissue records
    assert dialog.tissue_properties().reference_source == "default"


def test_voi_metadata_dialog_no_op_ok_keeps_the_voi_with_base_data(qapp, test_data_photons):
    ct, cst, result = test_data_photons
    voi = cst.vois[0]
    dialog = VOIMetadataDialog(voi, _oxygen_machine(), "zstar_kernels", cst=cst.vois)

    dialog.accept()

    assert dialog.updated_voi is voi


# ------------------------------------------------------------------ reference radiation
_MKM_BASELINE = dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615)


@pytest.fixture
def test_photons(monkeypatch):
    """A further registered reference radiation, removed again after the test."""
    from pyRadPlan.biology.tissues import (
        _reference,
        ParameterSource,
        register_reference_radiation,
    )

    monkeypatch.setattr(
        _reference, "_REFERENCE_RADIATIONS", dict(_reference._REFERENCE_RADIATIONS)
    )
    return register_reference_radiation(
        "test_photons",
        "Test photons",
        y_d=2.0,
        source=ParameterSource("Doe et al., J. Test. 1 (2020) 1", short="Doe 2020"),
    )


def test_voi_metadata_dialog_offers_the_reference_radiations(
    qapp, test_data_photons, test_photons
):
    from pyRadPlan.gui.widgets._tissue_params import reference_radiation_value
    from pyRadPlan.biology.tissues import (
        DEFAULT_REFERENCE_RADIATION,
        registered_reference_radiations,
    )

    _ct, cst, _result = test_data_photons
    voi = _with_tissue(cst.vois[0], MKMTissueProperties(**_MKM_BASELINE))
    dialog = VOIMetadataDialog(voi)
    combo = dialog.reference_radiation_combo
    assert combo is not None and dialog.reference_group.isAncestorOf(combo)
    names = [combo.itemData(i) for i in range(combo.count())]
    assert names == [radiation.name for radiation in registered_reference_radiations()]
    assert reference_radiation_value(combo) == DEFAULT_REFERENCE_RADIATION
    index = names.index("test_photons")
    assert combo.itemText(index) == "Test photons (Doe 2020)"
    tooltip = combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    assert "y_D = 2 keV/µm" in tooltip and "Doe et al., J. Test." in tooltip

    # Showing the tissue changes nothing
    dialog.accept()
    assert dialog.updated_voi is voi

    # Only models deriving their reference parameters refer to a reference radiation
    dialog = VOIMetadataDialog(voi)
    dialog.set_reference_radiation("test_photons")
    dialog.tissue_combo.setCurrentText("LEM")
    assert dialog.reference_radiation_combo is None
    with pytest.raises(ValueError, match="no reference radiation"):
        dialog.set_reference_radiation("test_photons")
    # ... and the selection survives switching the model back and forth
    dialog.tissue_combo.setCurrentText("mMKM")
    assert reference_radiation_value(dialog.reference_radiation_combo) == "test_photons"


def test_voi_metadata_dialog_rederives_for_the_reference_radiation(
    qapp, test_data_photons, test_photons
):
    from pyRadPlan.gui.widgets._tissue_params import param_source

    _ct, cst, _result = test_data_photons
    voi = _with_tissue(cst.vois[0], MKMTissueProperties(**_MKM_BASELINE))
    expected = MKMTissueProperties(reference_radiation="test_photons", **_MKM_BASELINE)
    assert expected.alpha_x != voi.tissue_properties.alpha_x

    dialog = VOIMetadataDialog(voi)
    dialog.set_reference_radiation("test_photons")
    assert dialog.alpha_spin.value() == pytest.approx(expected.alpha_x, abs=1e-4)
    source, note = param_source(dialog.alpha_spin)
    assert source == "estimated" and "Test photons" in note
    assert "Test photons" in dialog.reference_note.text()
    assert "Doe et al." in dialog.reference_radiation_combo.toolTip()
    dialog.accept()
    tissue = dialog.updated_voi.tissue_properties
    assert tissue.reference_radiation == "test_photons"
    assert tissue.alpha_x == pytest.approx(expected.alpha_x, rel=1e-9)
    assert tissue.reference_source == "estimated"

    # An explicit override is kept, but the reference radiation still changes the tissue
    dialog = VOIMetadataDialog(voi)
    dialog.alpha_spin.setValue(0.3)
    dialog.beta_spin.setValue(0.06)
    dialog.set_reference_radiation("test_photons")
    assert dialog.alpha_spin.value() == pytest.approx(0.3)
    dialog.accept()
    tissue = dialog.updated_voi.tissue_properties
    assert (tissue.reference_radiation, tissue.alpha_x) == ("test_photons", 0.3)
    assert tissue.reference_source == "explicit"

    # Selecting the original radiation again is no change at all
    dialog = VOIMetadataDialog(voi)
    dialog.set_reference_radiation("test_photons")
    dialog.set_reference_radiation(voi.tissue_properties.reference_radiation)
    dialog.accept()
    assert dialog.updated_voi is voi


def test_voi_metadata_dialog_preset_selects_its_reference_radiation(
    qapp, test_data_photons, test_photons
):
    from pyRadPlan.gui.widgets._tissue_params import reference_radiation_value
    from pyRadPlan.biology.tissues import (
        CELL_LINE_PRESETS,
        UNPUBLISHED_SOURCE,
        register_cell_line_preset,
    )

    saved = dict(CELL_LINE_PRESETS)
    try:
        preset = register_cell_line_preset(
            "TestCells",
            "mMKM",
            source=UNPUBLISHED_SOURCE,
            reference_radiation="test_photons",
            **_MKM_BASELINE,
        )
        _ct, cst, _result = test_data_photons
        dialog = VOIMetadataDialog(cst.vois[0])
        dialog.apply_cell_line_preset(_preset_index(dialog, "TESTCELLS", "mMKM"))
    finally:
        CELL_LINE_PRESETS.clear()
        CELL_LINE_PRESETS.update(saved)
    assert reference_radiation_value(dialog.reference_radiation_combo) == "test_photons"
    tissue = dialog.tissue_properties()
    assert (tissue.cell_line, tissue.reference_radiation) == ("TestCells", "test_photons")
    assert tissue == preset.tissue()


def test_voi_metadata_dialog_labels_presets_without_a_confirmed_source(qapp, test_data_photons):
    from pyRadPlan.biology.tissues import (
        CELL_LINE_PRESETS,
        ParameterSource,
        register_cell_line_preset,
    )

    saved = dict(CELL_LINE_PRESETS)
    try:
        register_cell_line_preset(
            "TestCells", "MKM", source=ParameterSource(note="Origin unknown."), **_MKM_BASELINE
        )
        _ct, cst, _result = test_data_photons
        dialog = VOIMetadataDialog(cst.vois[0])
    finally:
        CELL_LINE_PRESETS.clear()
        CELL_LINE_PRESETS.update(saved)
    labels = [dialog.preset_combo.itemText(i) for i in range(dialog.preset_combo.count())]
    index = labels.index(next(label for label in labels if label.startswith("TestCells · MKM")))
    assert labels[index].startswith("TestCells · MKM (unverified)")
    tooltip = dialog.preset_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)
    assert "Source: Unverified" in tooltip and "Origin unknown." in tooltip

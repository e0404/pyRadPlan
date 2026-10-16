import math
from typing import get_args

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from pyRadPlan._settings import GuiUiMode, get_settings
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.plan import PlanWidget, UIMode
from pyRadPlan.machines import get_available_machines
from pyRadPlan.plan import IonPlan, PhotonPlan, Plan
from pyRadPlan.io import load_tg119


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


def _make_widget():
    ws = WorkspaceManager()
    return PlanWidget(workspace=ws), ws


def _answer_message_boxes(monkeypatch, button_text=None):
    """
    Answer every message box by clicking the button labelled `button_text`.

    Returns the texts of the boxes shown; without `button_text`, showing one fails.
    """
    from PySide6.QtWidgets import QMessageBox

    shown = []

    def exec_(box):
        shown.append(f"{box.text()}\n{box.informativeText()}")
        assert button_text is not None, f"unexpected message box: {shown[-1]}"
        next(b for b in box.buttons() if b.text() == button_text).click()
        return 0

    monkeypatch.setattr(QMessageBox, "exec", exec_)
    return shown


def _preset_index(dialog, cell_line, model):
    """Index of a cell line preset in the tissue editor's list of presets."""
    return next(
        i
        for i, preset in enumerate(dialog._presets)
        if preset[0].group == (cell_line.upper(), model)
    )


def _preferred_preset_provenances(dialog):
    """Provenances of the presets the tissue picker offers above its separator."""
    return {p.provenance for p in dialog._picks if p.kind == "preset" and p.preferred}


def _picker_texts(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _type_gantry(widget, text):
    """Replace the gantry angles keystroke by keystroke and commit with Enter."""
    widget._txt_gantry.selectAll()
    QTest.keyClicks(widget._txt_gantry, text)
    QTest.keyClick(widget._txt_gantry, Qt.Key_Return)


def test_constructs_with_empty_workspace(qapp):
    widget, ws = _make_widget()
    assert widget.workspace is ws
    assert ws.pln is None
    assert widget._cmb_radiation.currentText() == "photons"


def test_ai_beam_button_enabled_matches_availability(qapp):
    widget, _ = _make_widget()
    assert widget._btn_ai_beams.isEnabled() == (widget._ai_disabled_reason is None)
    if widget._ai_disabled_reason is not None:
        assert widget._btn_ai_beams.toolTip() == widget._ai_disabled_reason


def test_do_update_populates_fields_photon(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(
        num_of_fractions=25,
        machine="Generic",
        prop_stf={"gantry_angles": [0, 90, 180], "couch_angles": [0, 0, 0]},
    )

    assert widget._cmb_radiation.currentText() == "photons"
    assert widget._spn_fractions.value() == 25
    assert widget._cmb_machine.currentText() == "Generic"
    assert widget._spn_num_beams.value() == 3


def test_machine_box_lists_machine_files_of_the_mode(qapp):
    widget, _ = _make_widget()

    def listed():
        return [widget._cmb_machine.itemText(i) for i in range(widget._cmb_machine.count())]

    assert listed() == get_available_machines("photons")["photons"]

    widget._cmb_radiation.setCurrentText("VHEE")
    VHEE_machines = get_available_machines("VHEE")["VHEE"]
    assert listed() == VHEE_machines
    # The machine added next to VHEE_Generic.mat has to be selectable
    assert "Focused" in VHEE_machines
    assert widget._cmb_machine.currentText() == "Generic"

    widget._cmb_machine.setCurrentText("Focused")
    widget._cmb_radiation.setCurrentText("protons")
    # A machine the new mode does not have falls back to the default
    assert widget._cmb_machine.currentText() == "Generic"


def test_do_update_populates_fields_ion(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="protons", num_of_fractions=10)
    assert widget._cmb_radiation.currentText() == "protons"
    assert widget._spn_fractions.value() == 10


def test_switching_radiation_mode_builds_correct_subclass(qapp):
    widget, ws = _make_widget()

    widget._cmb_radiation.setCurrentText("protons")
    widget._txt_gantry.setText("0 180")
    widget._on_apply()
    assert isinstance(ws.pln, IonPlan)
    assert ws.pln.radiation_mode == "protons"

    widget._cmb_radiation.setCurrentText("photons")
    widget._on_apply()
    assert isinstance(ws.pln, PhotonPlan)
    assert ws.pln.radiation_mode == "photons"


def test_couch_angles_follow_gantry_angles(qapp):
    widget, ws = _make_widget()

    _type_gantry(widget, "0 90 180 270")
    assert widget._txt_couch.text() == "0 0 0 0"

    widget._on_apply()
    assert ws.pln.prop_stf["couch_angles"] == [0.0] * 4


def test_added_beam_gets_a_zero_couch_angle(qapp):
    widget, _ = _make_widget()

    _type_gantry(widget, "0 90")
    widget._txt_couch.setText("0 30")
    # an added gantry angle appends a zero couch angle, the existing ones stay
    _type_gantry(widget, "0 90 180")
    assert widget._txt_couch.text() == "0 30 0"
    # a removed one drops its couch angle again
    _type_gantry(widget, "0 90")
    assert widget._txt_couch.text() == "0 30"


def test_retyping_gantry_angles_keeps_couch_angles(qapp):
    widget, _ = _make_widget()
    _type_gantry(widget, "0 90 180")
    widget._txt_couch.setText("10 20 30")

    # the intermediate text ("", "0", "0 1", ...) must not trim the couch angles
    _type_gantry(widget, "0 120 240")
    assert widget._txt_couch.text() == "10 20 30"

    # neither does clearing the field and leaving it
    widget._txt_gantry.clear()
    widget._txt_gantry.editingFinished.emit()
    assert widget._txt_couch.text() == "10 20 30"


def test_single_couch_angle_is_repeated_for_added_beams(qapp):
    widget, _ = _make_widget()
    widget._txt_couch.setText("10")

    _type_gantry(widget, "0 120 240")
    assert widget._txt_couch.text() == "10 10 10"

    widget._spn_num_beams.setValue(4)
    assert widget._txt_couch.text() == "10 10 10 0"


def test_num_beams_generates_equally_spaced_gantry_angles(qapp):
    widget, ws = _make_widget()

    widget._spn_num_beams.setValue(8)
    assert widget._txt_gantry.text() == "0 45 90 135 180 225 270 315"
    # the couch angles follow along
    assert widget._txt_couch.text() == "0 0 0 0 0 0 0 0"

    widget._on_apply()
    assert ws.pln.prop_stf["gantry_angles"] == [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]


def test_num_beams_tracks_manually_entered_gantry_angles(qapp):
    widget, _ = _make_widget()

    widget._txt_gantry.setText("0 45 300")
    assert widget._spn_num_beams.value() == 3
    # an unparsable entry disables the spin box instead of showing a stale count
    widget._txt_gantry.setText("0 abc")
    assert not widget._spn_num_beams.isEnabled()
    widget._txt_gantry.setText("0 90")
    assert widget._spn_num_beams.isEnabled()
    assert widget._spn_num_beams.value() == 2


def test_sync_from_workspace_keeps_couch_angles(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(
        prop_stf={"gantry_angles": [0, 90], "couch_angles": [0, 45]},
    )
    assert widget._txt_couch.text() == "0 45"
    assert widget._spn_num_beams.value() == 2


def test_editing_fields_and_apply_writes_valid_pln(qapp):
    widget, ws = _make_widget()

    widget._cmb_radiation.setCurrentText("photons")
    widget._spn_fractions.setValue(15)
    widget._txt_gantry.setText("0, 72, 144, 216, 288")
    widget._txt_couch.setText("0")
    widget._spn_bixel.setValue(5.0)
    widget._spn_res_x.setValue(4.0)
    widget._spn_res_y.setValue(4.0)
    widget._spn_res_z.setValue(4.0)

    widget._on_apply()

    pln = ws.pln
    assert isinstance(pln, Plan)
    assert pln.num_of_fractions == 15
    assert len(pln.prop_stf["gantry_angles"]) == 5
    assert pln.prop_stf["couch_angles"] == [0.0] * 5
    assert pln.prop_dose_calc["dose_grid"]["resolution"]["x"] == 4.0


def test_scenario_combo_wired_to_mult_scen(qapp):
    widget, ws = _make_widget()

    items = [widget._cmb_scenario.itemText(i) for i in range(widget._cmb_scenario.count())]
    assert items[0] == "nomScen"
    assert {"wcScen", "impScen", "rndScen"} <= set(items)
    # unimplemented scenario models are listed but not selectable
    assert not widget._cmb_scenario.model().item(items.index("wcScen")).isEnabled()

    widget._on_apply()
    assert ws.pln.mult_scen.short_name == "nomScen"


def test_placeholder_controls_disabled(qapp):
    widget, _ = _make_widget()
    assert not widget._btn_tissue.isEnabled()
    assert not widget._chk_sequencing.isEnabled()
    assert not widget._chk_dao.isEnabled()
    assert not widget._chk_conf3d.isEnabled()
    assert not widget._cmb_quantity.isEnabled()
    assert _combo_items(widget._cmb_quantity) == ["auto"]


def _combo_items(combo):
    return [combo.itemText(i) for i in range(combo.count())]


def _combo_data(combo):
    return [combo.itemData(i) for i in range(combo.count())]


def test_bio_model_combo_follows_radiation_mode(qapp):
    widget, _ = _make_widget()
    assert widget._cmb_bio_model.isEnabled()

    widget._cmb_radiation.setCurrentText("photons")
    assert widget._bio_model_name() == "none"
    assert "WED" not in _combo_data(widget._cmb_bio_model)

    widget._cmb_radiation.setCurrentText("protons")
    items = _combo_data(widget._cmb_bio_model)
    assert items[0] == "none"
    assert {"constant_rbe", "WED", "MCN", "CAR", "LSM"} <= set(items)
    assert "HEL" not in items
    assert "LEM" not in items  # aliases are not listed
    assert widget._bio_model_name() == "constant_rbe"  # per-mode default

    widget._cmb_radiation.setCurrentText("helium")
    assert "HEL" in _combo_data(widget._cmb_bio_model)
    assert widget._bio_model_name() == "HEL"

    widget._cmb_radiation.setCurrentText("carbon")
    assert widget._bio_model_name() == "alpha_beta_kernels"

    widget._cmb_radiation.setCurrentText("oxygen")
    assert widget._bio_model_name() == "alpha_beta_kernels"

    widget._cmb_radiation.setCurrentText("VHEE")
    assert widget._bio_model_name() == "none"


def _selected_entry_enabled(widget):
    combo = widget._cmb_bio_model
    return combo.model().item(combo.currentIndex()).isEnabled()


def test_bio_model_selection_survives_mode_switch_when_supported(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._select_bio_model("LSM")
    widget._cmb_radiation.setCurrentText("helium")
    assert widget._bio_model_name() == "LSM"
    assert _selected_entry_enabled(widget)
    widget._cmb_radiation.setCurrentText("photons")
    assert widget._bio_model_name() == "none"


@pytest.mark.parametrize(
    ("model", "radiation_mode", "expected"),
    [
        ("LSM", "carbon", "α/β kernels · LEM IV"),
        ("MCN", "oxygen", "α/β kernels · LEM I"),
        ("WED", "carbon", "α/β kernels · LEM IV"),
    ],
)
def test_mode_switch_replaces_a_model_the_new_mode_cannot_run(
    qapp, model, radiation_mode, expected
):
    """An explicit selection the new context cannot run gives way to its default."""
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._select_bio_model(model)
    replaced = widget._cmb_bio_model.currentText()

    widget._cmb_radiation.setCurrentText(radiation_mode)

    assert widget._cmb_bio_model.currentText() == expected
    assert _selected_entry_enabled(widget)
    assert "not supported" not in widget._cmb_bio_model.currentText()
    note = widget._lbl_status.text()
    assert replaced in note and expected in note
    assert "not available" in note


@pytest.mark.parametrize(
    ("radiation_mode", "expected"),
    [("carbon", "α/β kernels · LEM IV"), ("oxygen", "α/β kernels · LEM I")],
)
def test_mode_switch_from_the_proton_default_picks_the_ion_default(qapp, radiation_mode, expected):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    assert widget._bio_model_name() == "constant_rbe"
    widget._cmb_radiation.setCurrentText(radiation_mode)
    assert widget._cmb_bio_model.currentText() == expected
    assert _selected_entry_enabled(widget)


def test_mode_switch_from_carbon_kernels_to_photons_selects_none(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("carbon")
    assert widget._bio_model_name() == "alpha_beta_kernels"
    widget._cmb_radiation.setCurrentText("photons")
    assert widget._bio_model_name() == "none"
    assert _selected_entry_enabled(widget)


def test_engine_switch_disables_kernel_formalisms_and_replaces_the_selection(qapp):
    widget, ws = _make_widget()
    widget._cmb_radiation.setCurrentText("carbon")
    widget._cmb_engine.setCurrentText("HongPB")
    assert widget._cmb_bio_model.currentText() == "α/β kernels · LEM IV"

    widget._cmb_engine.setCurrentText("FRED")

    entries = {label: (enabled, tooltip) for label, enabled, tooltip in _bio_entries(widget)}
    enabled, tooltip = entries["α/β kernels · LEM IV"]
    assert not enabled
    assert "FRED does not use the machine's α/β kernels" in tooltip
    # FRED scores LET itself, which the carbon machine does not tabulate
    assert entries["Linear scaling (LET)"][0]
    assert widget._bio_model_name() == "none"
    assert _selected_entry_enabled(widget)
    note = widget._lbl_status.text()
    assert "α/β kernels · LEM IV" in note and "FRED" in note
    widget._on_apply()
    assert ws.pln.bio_model.model == "none"
    assert ws.pln.prop_dose_calc["engine"] == "FRED"

    # Back on the pencil-beam engine the kernels are offered again; the choice stays
    widget._cmb_engine.setCurrentText("HongPB")
    entries = {label: enabled for label, enabled, _tooltip in _bio_entries(widget)}
    assert entries["α/β kernels · LEM IV"]
    assert not entries["Linear scaling (LET)"]
    assert widget._bio_model_name() == "none"


def test_engine_switch_keeps_a_model_the_engine_supports(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._select_bio_model("WED")
    widget._lbl_status.setText("")
    widget._cmb_engine.setCurrentText("FRED")
    assert widget._bio_model_name() == "WED"
    assert _selected_entry_enabled(widget)
    assert "not available" not in widget._lbl_status.text()

    # TOPAS scores the LET of protons itself
    widget._cmb_engine.setCurrentText("TOPAS")
    assert widget._bio_model_name() == "WED"
    assert "not available" not in widget._lbl_status.text()


def test_restoring_a_plan_the_engine_cannot_run_keeps_its_model(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(
        radiation_mode="carbon",
        bio_model="alpha_beta_kernels",
        prop_dose_calc={"engine": "FRED"},
    )

    combo = widget._cmb_bio_model
    assert widget._cmb_engine.currentText() == "FRED"
    assert combo.currentText() == "α/β kernels (not supported by the dose engine FRED)"
    assert not _selected_entry_enabled(widget)
    tooltip = combo.itemData(combo.currentIndex(), Qt.ItemDataRole.ToolTipRole)
    assert "FRED does not use the machine's α/β kernels" in tooltip
    assert not widget.has_pending_changes()

    widget._on_apply()
    assert ws.pln.bio_model.model == "alpha_beta_kernels"

    # Switching to an engine that can run it selects its regular entry
    widget._cmb_engine.setCurrentText("HongPB")
    assert combo.currentText() == "α/β kernels · LEM IV"
    assert _selected_entry_enabled(widget)


def test_tissue_indicator_follows_the_engine(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    ws.pln = IonPlan(
        radiation_mode="oxygen",
        bio_model="alpha_beta_kernels",
        prop_dose_calc={"engine": "TOPAS"},
    )
    assert not widget._lbl_tissue_warning.isHidden()
    assert "TOPAS does not use the machine's α/β kernels" in widget._lbl_tissue_warning.toolTip()

    widget._cmb_engine.setCurrentText("HongPB")
    assert widget._lbl_tissue_warning.isHidden()


def test_tissue_editor_follows_the_engine(qapp, test_data_protons):
    from pyRadPlan.dose.engines import ParticleTOPASMCEngine
    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    machine = load_machine_quietly("oxygen", "Generic")
    dialog = TissueParametersDialog(cst, machine, "zstar_kernels")
    assert [t.cell_line for t in dialog._machine_tissues] == ["Generic_MKM"]
    # TOPAS does not read the z* kernels, so their classes are no restriction
    dialog = TissueParametersDialog(cst, machine, "zstar_kernels", engine=ParticleTOPASMCEngine)
    assert len(dialog._machine_tissues) == 3


def test_plan_widget_passes_the_engine_to_the_tissue_editor(qapp, monkeypatch, tg119):
    from pyRadPlan.gui.widgets.plan import _plan_widget

    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    widget._cmb_radiation.setCurrentText("oxygen")
    widget._cmb_engine.setCurrentText("TOPAS")
    opened = {}

    class _Dialog:
        def __init__(self, cst, machine, bio_model, parent, engine=None):
            opened["engine"] = engine

        def exec(self):
            return 0

    monkeypatch.setattr(_plan_widget, "TissueParametersDialog", _Dialog)
    widget._on_tissue_parameters()
    assert opened["engine"].short_name == "TOPAS"


def test_apply_writes_bio_model_to_pln(qapp):
    widget, ws = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._txt_gantry.setText("0")
    widget._select_bio_model("MCN")
    widget._on_apply()
    assert isinstance(ws.pln, IonPlan)
    assert ws.pln.bio_model.model == "MCN"


def test_do_update_restores_bio_model_and_apply_keeps_parameters(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="protons", bio_model={"model": "constant_rbe", "rbe": 1.0})
    assert widget._bio_model_name() == "constant_rbe"

    # unchanged selection: the parametrised instance is kept on Apply
    widget._txt_gantry.setText("0")
    widget._on_apply()
    assert ws.pln.bio_model.rbe == 1.0

    # changing the selection replaces the model
    widget._select_bio_model("WED")
    widget._on_apply()
    assert ws.pln.bio_model.model == "WED"


def test_dose_convention_round_trip(qapp):
    widget, ws = _make_widget()
    assert widget._cmb_dose_convention.currentData() == "total"

    widget._txt_gantry.setText("0")
    widget._on_apply()
    assert ws.pln.dose_convention == "total"

    widget._cmb_dose_convention.setCurrentIndex(
        widget._cmb_dose_convention.findData("per_fraction")
    )
    widget._on_apply()
    assert ws.pln.dose_convention == "per_fraction"

    ws.pln = PhotonPlan(num_of_fractions=5, dose_convention="total")
    assert widget._cmb_dose_convention.currentData() == "total"
    ws.pln = PhotonPlan(num_of_fractions=5, dose_convention="per_fraction")
    assert widget._cmb_dose_convention.currentData() == "per_fraction"


def test_bio_model_parameters_round_trip(qapp):
    widget, ws = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._txt_gantry.setText("0")
    widget._select_bio_model("WED")
    assert widget._btn_bio_config.isEnabled()
    widget._bio_params["WED"] = {"p1": 0.5}
    widget._on_apply()
    assert ws.pln.bio_model.model == "WED"
    assert ws.pln.bio_model.p1_WED == 0.5

    widget._select_bio_model("none")
    assert not widget._btn_bio_config.isEnabled()
    # A model with nothing to configure says so, rather than greying out unexplained
    assert "no parameters to configure" in widget._btn_bio_config.toolTip()
    widget._select_bio_model("WED")
    assert "Configure" in widget._btn_bio_config.toolTip()

    ws.pln = IonPlan(radiation_mode="protons", bio_model={"model": "constant_rbe", "rbe": 1.0})
    assert widget._bio_params["constant_rbe"] == {"rbe": 1.0}
    assert widget._capture_state()["bio_params"] == '{"rbe": 1.0}'


def test_bio_model_config_model():
    from pyRadPlan.biology.models import ConstantRBEModel, EmptyModel, Wedenberg

    assert set(Wedenberg.config_model().model_fields) == {"p0", "p1", "p2", "quantity_variants"}
    assert Wedenberg.config_model()().p1 == 0.434
    assert ConstantRBEModel.config_model()(rbe=1.0).rbe == 1.0
    assert not EmptyModel.config_model().model_fields


def test_tissue_dialog_edits_cst(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    widget, ws = _make_widget()
    assert not widget._btn_tissue.isEnabled()
    ws.cst = cst
    assert widget._btn_tissue.isEnabled()

    dialog = TissueParametersDialog(cst)
    dialog.set_values(0, 0.2, 0.04)
    dialog.accept()
    assert dialog.changed
    assert cst.vois[0].alpha_x == pytest.approx(0.2)
    assert cst.vois[0].beta_x == pytest.approx(0.04)
    assert cst.vois[0].tissue_properties.alpha_x == pytest.approx(0.2)
    assert cst.vois[0].tissue_properties.beta_x == pytest.approx(0.04)

    dialog = TissueParametersDialog(cst)
    dialog.accept()
    assert not dialog.changed


def test_tissue_dialog_keeps_auto_assigned_color(qapp, test_data_protons):
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    ct, cst, _ = test_data_protons
    # The matRad fixture ships explicit colors, so build a VOI with an auto-assigned one
    voi = create_voi(voi_type="TARGET", name="auto", ct_image=ct, mask=cst.vois[0].mask)
    auto_cst = StructureSet(ct_image=ct, vois=[voi])
    color = tuple(auto_cst.vois[0].color)
    assert "color" not in auto_cst.vois[0].model_fields_set

    dialog = TissueParametersDialog(auto_cst)
    dialog.set_values(0, 0.2, 0.04)
    dialog.accept()

    assert dialog.changed
    assert auto_cst.vois[0].alpha_x == pytest.approx(0.2)
    assert tuple(auto_cst.vois[0].color) == color
    assert "color" not in auto_cst.vois[0].model_fields_set


def test_tissue_dialog_edits_model_parameters(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties, TissueProperties

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    assert dialog._model_combos[0].currentText() == "Generic"

    dialog.set_tissue_model(0, "MKM")
    dialog.set_param(0, "r_nucleus", 3.9)
    dialog.set_param(0, "r_domain", 0.32)
    dialog.set_param(0, "alpha_0", 0.172)
    dialog.accept()

    assert dialog.changed
    tissue = cst.vois[0].tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.model == "MKM"
    assert tissue.r_nucleus == pytest.approx(3.9)
    assert tissue.r_domain == pytest.approx(0.32)
    assert tissue.alpha_0 == pytest.approx(0.172)
    assert math.isnan(tissue.beta_0)  # left unset
    # Structures that were not edited keep their reference-only tissue
    assert type(cst.vois[1].tissue_properties) is TissueProperties


def test_tissue_dialog_sets_one_parameter_for_all_structures(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    assert dialog.bulk_param_combo.findData("r_nucleus") < 0
    for row in range(len(cst.vois) - 1):
        dialog.set_tissue_model(row, "MKM")
    assert dialog.bulk_param_combo.findData("r_nucleus") >= 0

    dialog.bulk_param_combo.setCurrentIndex(dialog.bulk_param_combo.findData("r_nucleus"))
    dialog.bulk_value_spin.setValue(4.2)
    changed = dialog.set_param_for_rows()
    assert changed == list(range(len(cst.vois) - 1))
    assert "left unchanged" in dialog.status_label.text()
    dialog.accept()

    for voi in cst.vois[:-1]:
        assert isinstance(voi.tissue_properties, MKMTissueProperties)
        assert voi.tissue_properties.r_nucleus == pytest.approx(4.2)
    assert not isinstance(cst.vois[-1].tissue_properties, MKMTissueProperties)


def test_tissue_dialog_derives_reference_params(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import param_source
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    dialog.set_tissue_model(0, "MKM")
    dialog.set_param(0, "r_nucleus", 4.1)
    dialog.set_param(0, "r_domain", 0.34)
    dialog.set_param(0, "alpha_0", 0.313)
    dialog.set_param(0, "beta_0", 0.0615)

    expected = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.313, beta_0=0.0615, r_domain=0.34
    )
    # The structure's reference parameters were explicit, so they are kept until derived
    assert param_source(dialog._spins[0]["alpha_x"])[0] == "explicit"
    assert dialog.derive_reference_params(0) == pytest.approx(expected, abs=1e-4)
    assert param_source(dialog._spins[0]["alpha_x"])[0] == "estimated"

    # Derived values follow the model parameters live, and the model (mMKM: z*_1D,ref)
    dialog.set_param(0, "alpha_0", 0.2)
    dialog.set_tissue_model(0, "mMKM")
    mmkm = MKMTissueProperties.estimate_reference_parameters(
        alpha_0=0.2, beta_0=0.0615, r_domain=0.34, r_nucleus=4.1, model="mMKM"
    )
    assert dialog._spins[0]["alpha_x"].value() == pytest.approx(mmkm["alpha_x"], abs=1e-4)
    assert "z*_1D,ref" in dialog._spins[0]["alpha_x"].toolTip()
    expected = mmkm

    # A row of another model has nothing to derive from
    dialog.set_tissue_model(1, "Generic")
    assert dialog.derive_reference_params(1) == {}

    dialog.accept()
    tissue = cst.vois[0].tissue_properties
    assert tissue.alpha_x == pytest.approx(expected["alpha_x"], abs=1e-4)
    assert tissue.beta_x == pytest.approx(expected["beta_x"], abs=1e-4)


def test_machine_summary_is_reread_after_base_data_changes(monkeypatch, tmp_path):
    """Files replaced through pyRadPlan keep size and mtime within a clock tick; reread them."""
    from pyRadPlan.core import add_base_data_file, base_data_generation
    from pyRadPlan.gui.widgets import _tissue_params

    monkeypatch.setenv("PYRADPLAN_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(_tissue_params, "_MACHINE_SUMMARY_CACHE", {})
    key = ("protons", "Generic")
    info = _tissue_params.load_machine_quietly(*key)
    assert _tissue_params.load_machine_quietly(*key) is info

    generation = base_data_generation()
    dummy = tmp_path / "SP_dummy.mat"
    dummy.write_bytes(b"")
    add_base_data_file("stopping_power_tables", dummy)
    assert base_data_generation() == generation + 1
    assert _tissue_params.load_machine_quietly(*key) is not info


def test_machine_base_data_is_read_once_per_file(monkeypatch):
    """The editors keep what they read off a machine, so opening one does not re-read it."""
    from pyRadPlan.gui.widgets import _tissue_params

    key = ("protons", "Generic")
    monkeypatch.setattr(_tissue_params, "_MACHINE_SUMMARY_CACHE", {})
    reads = []
    real_load_from_name = _tissue_params.load_from_name

    def counting_load_from_name(radiation_mode, machine_name):
        reads.append((radiation_mode, machine_name))
        return real_load_from_name(radiation_mode, machine_name)

    monkeypatch.setattr(_tissue_params, "load_from_name", counting_load_from_name)

    info = _tissue_params.load_machine_quietly(*key)
    assert info is not None
    assert info.radiation_mode == "protons"
    # The machine itself is not held on to, only what the editors read off it
    assert _tissue_params.load_machine_quietly(*key) is info
    assert reads == [key]
    assert _tissue_params.available_tissue_classes(info) == list(info.tissue_properties)

    # A base data file replaced on disk is read again rather than served stale
    stamp, cached = _tissue_params._MACHINE_SUMMARY_CACHE[key]
    _tissue_params._MACHINE_SUMMARY_CACHE[key] = ((*stamp[:-1], stamp[-1] + 1), cached)
    assert _tissue_params.load_machine_quietly(*key) is not info
    assert reads == [key, key]

    # A machine without a file is reported as missing every time, never cached as one
    assert _tissue_params.load_machine_quietly("protons", "NoSuchMachine") is None
    assert _tissue_params.load_machine_quietly("protons", "NoSuchMachine") is None
    assert reads.count(("protons", "NoSuchMachine")) == 2


def test_tissue_dialog_offers_the_tissue_classes_of_the_rbe_table(
    qapp, monkeypatch, test_data_protons
):
    """A tabulated RBE model offers the classes of its table, not the machine's MKM class."""
    from types import SimpleNamespace

    from pyRadPlan.biology.models import TabulatedAlphaBetaModel
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _, cst, _ = test_data_protons
    machine = SimpleNamespace(
        name="Generic",
        radiation_mode="carbon",
        provided_quantities=lambda: ["fluence"],
        tissue_properties=np.asarray([MKMTissueProperties(r_nucleus=3.9, r_domain=0.32)]),
    )
    table_tissues = list(TabulatedAlphaBetaModel().table_kernel_set.tissues)
    assert table_tissues, "the shipped RBE table provides no tissue class"

    dialog = TissueParametersDialog(cst, machine, "dose_average_alpha_beta")
    assert dialog.preset_combo is not None
    assert dialog._machine_tissues == table_tissues
    # Presets of the table's provenance (LEM I) are offered first
    assert _preferred_preset_provenances(dialog) == {"LEM_I"}

    # The table class can be applied, which is what makes the model's table usable
    _answer_message_boxes(monkeypatch)
    dialog.apply_machine_tissue(0, 0)
    dialog.accept()
    assert cst.vois[0].tissue_properties.matches(table_tissues[0])


def test_tissue_dialog_offers_no_rbe_tables_without_fluence_spectra(qapp, test_data_protons):
    """The tables are dose-averaged over fluence spectra, so a machine without them gets none."""
    from types import SimpleNamespace

    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _, cst, _ = test_data_protons
    machine = SimpleNamespace(
        name="Generic",
        radiation_mode="carbon",
        provided_quantities=lambda: ["physical_dose"],
        tissue_properties=np.asarray([MKMTissueProperties(r_nucleus=3.9, r_domain=0.32)]),
    )

    dialog = TissueParametersDialog(cst, machine, "dose_average_alpha_beta")
    assert dialog._machine_tissues == [machine.tissue_properties[0]]


def test_tissue_dialog_derive_button_needs_an_mkm_structure(qapp, test_data_protons):
    """Only MKM / mMKM derive their reference parameters, so the button waits for such a row."""
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    assert dialog.derive_button.text() == "Derive from model"

    for row in range(len(cst.vois)):
        dialog.set_tissue_model(row, "Generic")
    assert not dialog.derive_button.isEnabled()
    # The tooltip says why instead of describing what the button would do
    assert "no structure" in dialog.derive_button.toolTip()

    # One MKM structure is enough, wherever it sits in the table
    dialog.set_tissue_model(len(cst.vois) - 1, "mMKM")
    assert dialog.derive_button.isEnabled()
    assert "model parameters" in dialog.derive_button.toolTip()

    # A model taking its reference parameters as given switches it off again
    dialog.set_tissue_model(len(cst.vois) - 1, "LEM")
    assert not dialog.derive_button.isEnabled()


def test_tissue_dialog_groups_model_and_reference_columns(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    dialog.set_tissue_model(0, "MKM")
    dialog.set_tissue_model(1, "LEM")

    columns = {name: dialog._param_column(name) for name in dialog._fields}
    reference = [columns["alpha_x"], columns["beta_x"]]
    model = [columns[name] for name in ("d_t", "r_nucleus", "r_domain", "alpha_0", "beta_0")]
    # The model parameters come first, the reference parameters last
    assert max(model) < min(reference)
    header = dialog._table.horizontalHeaderItem(columns["alpha_x"])
    assert header.text() == "α_x (ref.) [Gy⁻¹]"
    assert header.toolTip().startswith("Reference radiation (for RBE)")
    assert (
        dialog._table.horizontalHeaderItem(columns["d_t"]).toolTip().startswith("Model parameters")
    )
    # The reference cells say how they come about for the row's model
    assert "200 kV X-rays" in dialog._spins[0]["alpha_x"].toolTip()
    assert "inputs" in dialog._spins[1]["alpha_x"].toolTip()


def test_tissue_dialog_offers_machine_tissue_classes(qapp, test_data_protons):
    from types import SimpleNamespace

    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import LEMTissueProperties, MKMTissueProperties

    _, cst, _ = test_data_protons
    machine = SimpleNamespace(
        name="Generic",
        # Machines hold their tissue classes as an array, as the shipped ion machines do
        tissue_properties=np.asarray(
            [
                LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0),
                MKMTissueProperties(r_nucleus=3.9, r_domain=0.32),
            ],
            dtype=object,
        ),
    )

    dialog = TissueParametersDialog(cst, machine)
    classes = [p for p in dialog._picks if p.kind == "class"]
    assert len(classes) == 2
    assert "D_t = 30" in classes[0].label
    # The classes lead the picker
    assert _picker_texts(dialog.preset_combo)[1] == classes[0].label

    dialog.set_values(0, 0.2, 0.04)
    dialog.apply_machine_tissue(0, 1)
    dialog.accept()

    tissue = cst.vois[0].tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.r_nucleus == pytest.approx(3.9)
    assert tissue.r_domain == pytest.approx(0.32)
    # The MKM base data leaves the reference parameters open, so they survive
    assert tissue.alpha_x == pytest.approx(0.2)
    assert tissue.beta_x == pytest.approx(0.04)
    # Picking a base data class is the way to obtain a tissue the model can match
    assert tissue.matches(machine.tissue_properties[1])


def test_tissue_dialog_edits_the_lem_version(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import LEMTissueProperties

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    dialog.set_tissue_model(0, "LEM")
    # The version only concerns LEM, so its column appears with the model
    assert not dialog._table.isColumnHidden(dialog._param_column("version"))

    dialog.set_choice(0, "version", "II")
    dialog.set_param(0, "d_t", 6.0)
    dialog.set_param(0, "r_nucleus", 5.0)
    dialog.accept()

    assert dialog.changed
    tissue = cst.vois[0].tissue_properties
    assert type(tissue) is LEMTissueProperties
    assert tissue.version == "II"
    assert tissue.d_t == pytest.approx(6.0)


def test_tissue_dialog_leaves_parameters_of_other_models_empty(qapp, test_data_protons):
    """A column shown for one row's model stays empty in the rows not using it."""
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    dialog.show()
    dialog.set_tissue_model(0, "LEM")
    dialog.set_tissue_model(1, "MKM")
    qapp.processEvents()

    # Both models are in use, so their columns are shown for either row to fill
    assert not dialog._table.isColumnHidden(dialog._param_column("version"))
    assert not dialog._table.isColumnHidden(dialog._param_column("r_domain"))

    # The version and D_t are LEM's, the domain radius is MKM's
    assert not dialog._choices[0]["version"].isHidden()
    assert dialog._choices[1]["version"].isHidden()
    assert not dialog._spins[0]["d_t"].isHidden()
    assert dialog._spins[1]["d_t"].isHidden()
    assert dialog._spins[0]["r_domain"].isHidden()
    assert not dialog._spins[1]["r_domain"].isHidden()

    dialog.close()


def test_tissue_dialog_offers_cell_line_presets(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import MKMTissueProperties

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    assert dialog.preset_combo is not None
    labels = _picker_texts(dialog.preset_combo)
    assert any(label.startswith("HSG · LEM I (Elsässer 2007)") for label in labels)

    # Presets are labelled with their source, several of one cell line and provenance each
    assert any(label.startswith("T1 · MKM (Kase 2008)") for label in labels)
    for source in ("Manganaro 2018", "Elsässer 2008"):
        assert sum(label.startswith(f"HSG · LEM III ({source})") for label in labels) == 1
    hsg_mkm = [label for label in labels if label.startswith("HSG · MKM (")]
    assert [label.split(")")[0] for label in hsg_mkm] == [
        "HSG · MKM (Okamoto 2011",
        "HSG · MKM (Kase 2008",
        "HSG · MKM (Kase 2006",
    ]

    dialog.apply_cell_line_preset(0, _preset_index(dialog, "T1", "MKM"))
    dialog.accept()

    tissue = cst.vois[0].tissue_properties
    assert isinstance(tissue, MKMTissueProperties)
    assert tissue.cell_line == "T1"
    assert "Phys. Med. Biol. 53 (2008) 37" in tissue.cell_line_source
    assert tissue.r_nucleus == pytest.approx(3.5)
    assert tissue.r_domain == pytest.approx(0.35)
    # The published X-ray values of Furusawa et al. 2000, not derived from the model
    assert tissue.reference_source == "published"
    assert (tissue.alpha_x, tissue.beta_x) == (0.0305, 0.0585)
    # Structures the preset was not applied to keep their reference-only tissue
    assert type(cst.vois[1].tissue_properties) is not MKMTissueProperties


def test_tissue_dialog_labels_presets_without_a_confirmed_source(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import CELL_LINE_PRESETS, register_cell_line_preset

    _, cst, _ = test_data_protons
    saved = dict(CELL_LINE_PRESETS)
    try:
        register_cell_line_preset("TestCells", "MKM", **_MKM_BASELINE)
        dialog = TissueParametersDialog(cst)
        labels = _picker_texts(dialog.preset_combo)
    finally:
        CELL_LINE_PRESETS.clear()
        CELL_LINE_PRESETS.update(saved)
    assert any(label.startswith("TestCells · MKM (unverified)") for label in labels)


def test_tissue_dialog_preset_applies_to_all_rows_when_none_selected(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    labels = _picker_texts(dialog.preset_combo)
    dialog.preset_combo.setCurrentIndex(
        next(i for i, label in enumerate(labels) if label.startswith("CHO · LEM III"))
    )
    dialog._on_apply_pick()
    dialog.accept()

    for voi in cst.vois:
        assert voi.tissue_properties.cell_line == "CHO"
        assert voi.tissue_properties.d_t == pytest.approx(55.0)


def test_tissue_dialog_edit_marks_dose_influence_stale(qapp, test_data_protons):
    """A tissue edit in the plan widget must reach the workflow widget's indicators."""
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.gui.widgets.workflow import WorkflowWidget

    ct, cst, _ = test_data_protons
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    workflow = WorkflowWidget(ws)
    ws.set_many(ct=ct, cst=cst, pln=object(), stf=object(), dij=object(), result={})
    assert not workflow._dij_stale

    def fake_exec(self):
        self.set_values(0, 0.42, 0.04)
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TissueParametersDialog, "exec", fake_exec)
        widget._on_tissue_parameters()

    assert ws.cst.vois[0].tissue_properties.alpha_x == pytest.approx(0.42)
    assert workflow._dij_stale
    assert "recalculate" in workflow._lbl_status.text().lower()


def test_tissue_dialog_without_machine_has_no_picker(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    assert not any(pick.kind == "class" for pick in dialog._picks)
    # Without base data nothing is preferred: all presets, grouped by provenance
    assert not any(pick.preferred for pick in dialog._picks)
    assert "separator" not in [
        dialog.preset_combo.itemData(i, Qt.ItemDataRole.AccessibleDescriptionRole)
        for i in range(dialog.preset_combo.count())
    ]
    assert _picker_texts(dialog.preset_combo)[0] == "Presets · LEM I"


def test_tissue_dialog_shows_only_relevant_parameter_columns(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons

    dialog = TissueParametersDialog(cst)
    assert dialog._table.isColumnHidden(dialog._param_column("r_domain"))
    assert not dialog._table.isColumnHidden(dialog._param_column("alpha_x"))

    dialog.set_tissue_model(0, "MKM")
    assert not dialog._table.isColumnHidden(dialog._param_column("r_domain"))
    assert dialog._table.isColumnHidden(dialog._param_column("d_t"))


def test_iso_center_auto_omits_key(qapp):
    widget, ws = _make_widget()
    assert widget._chk_iso_auto.isChecked()
    assert not widget._txt_iso.isEnabled()

    widget._on_apply()
    assert "iso_center" not in ws.pln.prop_stf


def test_iso_center_manual_written_to_prop_stf(qapp):
    widget, ws = _make_widget()
    widget._chk_iso_auto.setChecked(False)
    assert widget._txt_iso.isEnabled()
    widget._txt_iso.setText("100 120.5 80")

    widget._on_apply()
    assert ws.pln.prop_stf["iso_center"] == [100.0, 120.5, 80.0]


def test_iso_center_restored_from_pln(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(
        num_of_fractions=30,
        machine="Generic",
        prop_stf={"iso_center": [50.0, 60.0, 70.0]},
    )
    assert not widget._chk_iso_auto.isChecked()
    assert widget._txt_iso.text() == "50 60 70"

    # per-beam iso centers fall back to auto
    ws.pln = PhotonPlan(
        num_of_fractions=30,
        machine="Generic",
        prop_stf={"iso_center": [[50.0, 60.0, 70.0], [10.0, 20.0, 30.0]]},
    )
    assert widget._chk_iso_auto.isChecked()


def test_per_beam_iso_centers_kept_on_apply(qapp):
    widget, ws = _make_widget()
    isos = [[50.0, 60.0, 70.0], [10.0, 20.0, 30.0]]
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0.0, 90.0], "iso_center": isos})
    assert widget._txt_iso.text() == "multiple iso centers"

    widget._spn_fractions.setValue(widget._spn_fractions.value() + 1)
    widget._on_apply()
    assert ws.pln.prop_stf["iso_center"] == isos


def test_per_beam_iso_centers_must_match_beam_count(qapp):
    widget, ws = _make_widget()
    original = PhotonPlan(
        prop_stf={"gantry_angles": [0.0, 90.0], "iso_center": [[5.0, 6.0, 7.0], [1.0, 2.0, 3.0]]}
    )
    ws.pln = original

    widget._txt_gantry.setText("0 90 180")
    widget._on_apply()
    assert ws.pln is original  # apply failed with status error
    assert "per-beam iso centers" in widget._lbl_status.text()


def test_toggling_auto_drops_per_beam_iso_centers(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(
        prop_stf={"gantry_angles": [0.0, 90.0], "iso_center": [[5.0, 6.0, 7.0], [1.0, 2.0, 3.0]]}
    )

    widget._chk_iso_auto.setChecked(False)
    widget._chk_iso_auto.setChecked(True)
    assert widget._btn_apply.text() == "Apply *"  # pending: the per-beam iso centers go
    widget._on_apply()
    assert "iso_center" not in ws.pln.prop_stf


def test_empty_iso_center_restored_as_auto(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(prop_stf={"iso_center": []})
    assert widget._chk_iso_auto.isChecked()


def test_auto_iso_center_shown_after_data_load(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)

    assert widget._chk_iso_auto.isChecked()
    assert not widget._txt_iso.isEnabled()
    shown = [float(t) for t in widget._txt_iso.text().split()]
    assert shown == pytest.approx(cst.target_center_of_mass(), rel=1e-4)


def test_toggling_auto_refills_iso_center(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)

    widget._chk_iso_auto.setChecked(False)
    widget._txt_iso.setText("1 2 3")
    widget._chk_iso_auto.setChecked(True)

    shown = [float(t) for t in widget._txt_iso.text().split()]
    assert shown == pytest.approx(cst.target_center_of_mass(), rel=1e-4)


def test_default_plan_auto_applied_on_data_load(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    assert ws.pln is None

    ws.set_many(ct=ct, cst=cst)

    assert isinstance(ws.pln, PhotonPlan)
    # the form matches the applied plan, so no pending-apply highlight remains
    assert widget._btn_apply.text() == "Apply"
    assert widget._lbl_status.text() == "Default plan applied."


def test_invalid_iso_center_rejected(qapp):
    widget, ws = _make_widget()
    widget._chk_iso_auto.setChecked(False)
    widget._txt_iso.setText("1 2")

    widget._on_apply()
    assert ws.pln is None  # apply failed with status error


def test_engine_combo_follows_radiation_mode(qapp):
    widget, ws = _make_widget()

    photon_engines = [widget._cmb_engine.itemText(i) for i in range(widget._cmb_engine.count())]
    assert "SVDPB" in photon_engines

    widget._cmb_radiation.setCurrentText("protons")
    proton_engines = [widget._cmb_engine.itemText(i) for i in range(widget._cmb_engine.count())]
    assert "SVDPB" not in proton_engines
    assert proton_engines  # Hong PB, FRED, TOPAS register for protons


def test_apply_writes_engine_and_config_to_prop_dose_calc(qapp):
    widget, ws = _make_widget()

    widget._cmb_engine.setCurrentText("SVDPB")
    # simulate values confirmed in the [...] config popup
    widget._engine_props["SVDPB"] = {"random_seed": 42, "kernel_cutoff": 60.0}
    widget._on_apply()

    prop = ws.pln.prop_dose_calc
    assert prop["engine"] == "SVDPB"
    assert prop["random_seed"] == 42
    assert prop["kernel_cutoff"] == 60.0
    assert "resolution" in prop["dose_grid"]


def test_do_update_restores_engine_and_config_from_pln(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(
        num_of_fractions=30,
        machine="Generic",
        prop_dose_calc={"engine": "SVDPB", "random_seed": 7},
    )

    assert widget._cmb_engine.currentText() == "SVDPB"
    assert widget._engine_props["SVDPB"] == {"random_seed": 7}


def test_engine_config_dialog_round_trip(qapp):
    from pyRadPlan.gui.widgets import ConfigFormDialog

    widget, ws = _make_widget()
    widget._cmb_engine.setCurrentText("SVDPB")

    engine_cls = widget._engines["SVDPB"]
    dialog = ConfigFormDialog(engine_cls.config_model(), initial={"random_seed": 3})
    assert "kernel_cutoff" in dialog.form._editors
    dialog.form._set_value("enable_dij_sampling", False)
    assert dialog.values() == {"random_seed": 3, "enable_dij_sampling": False}


def test_invalid_gantry_does_not_crash(qapp):
    widget, ws = _make_widget()
    failures = []
    widget.update_failed.connect(failures.append)

    widget._txt_gantry.setText("")
    widget._on_apply()

    assert ws.pln is None
    assert failures


def test_use_ct_grid_button(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)

    widget._on_use_ct_grid()
    res = ct.grid.resolution
    assert widget._spn_res_x.value() == pytest.approx(res["x"])
    assert widget._spn_res_y.value() == pytest.approx(res["y"])
    assert widget._spn_res_z.value() == pytest.approx(res["z"])


def test_hold_updates_guard_prevents_recursion(qapp):
    widget, ws = _make_widget()

    calls = []
    original = widget._do_update

    def _counting_update(changed):
        calls.append(changed)
        original(changed)

    widget._do_update = _counting_update

    widget._cmb_radiation.setCurrentText("protons")
    widget._txt_gantry.setText("0 180")
    widget._on_apply()

    assert isinstance(ws.pln, IonPlan)
    # apply happens inside hold_updates so the widget must not re-enter _do_update
    assert calls == []


def test_apply_preserves_unedited_plan_settings(qapp):
    widget, ws = _make_widget()
    original = PhotonPlan(
        prescribed_dose=72.0,
        prop_opt={"solver": {"name": "scipy", "max_iter": 17}, "convert_dose_objectives": False},
        prop_seq={"custom": 7},
        prop_stf={"gantry_angles": [0.0], "energy": 6.0, "iso_center": [1.0, 2.0, 3.0]},
    )
    ws.pln = original
    widget._txt_gantry.setText("0 180")
    widget._chk_iso_auto.setChecked(True)
    widget._on_apply()

    assert ws.pln.prop_opt == original.prop_opt
    assert ws.pln.prop_seq == original.prop_seq
    assert ws.pln.prescribed_dose == 72.0
    assert ws.pln.prop_stf["energy"] == 6.0
    assert ws.pln.prop_stf["gantry_angles"] == [0.0, 180.0]
    assert "iso_center" not in ws.pln.prop_stf
    assert original.prop_stf["iso_center"] == [1.0, 2.0, 3.0]
    assert original.prop_stf["gantry_angles"] == [0.0]


def test_modality_switch_resets_sequencing_and_steering_settings(qapp):
    widget, ws = _make_widget()
    original = PhotonPlan(
        prop_opt={"solver": "scipy"},
        prop_seq={"custom": 7},
        prop_stf={"gantry_angles": [0.0], "energy": 6.0},
    )
    ws.pln = original
    widget._cmb_radiation.setCurrentText("protons")
    widget._on_apply()

    assert isinstance(ws.pln, IonPlan)
    assert ws.pln.prop_seq == {}
    assert "energy" not in ws.pln.prop_stf
    assert ws.pln.prop_opt == original.prop_opt
    assert original.prop_seq == {"custom": 7}


def test_ion_bixel_width_round_trip(qapp):
    widget, ws = _make_widget()

    widget._cmb_radiation.setCurrentText("protons")
    assert widget._lbl_bixel.text() == "Bixel width / spot spacing [mm]:"
    widget._txt_gantry.setText("0")
    widget._spn_bixel.setValue(3.0)
    widget._on_apply()

    assert ws.pln.prop_stf["bixel_width"] == 3.0
    assert "longitudinal_spot_spacing" not in ws.pln.prop_stf

    reloaded, _ = _make_widget()
    reloaded.workspace.pln = ws.pln
    assert reloaded._spn_bixel.value() == 3.0


def test_preset_longitudinal_spacing_is_kept(qapp):
    widget, ws = _make_widget()

    widget._cmb_radiation.setCurrentText("protons")
    widget._txt_gantry.setText("0")
    widget._on_apply()
    pln = ws.pln
    pln.prop_stf["longitudinal_spot_spacing"] = 1.5
    ws.pln = pln

    widget._spn_bixel.setValue(4.0)
    widget._on_apply()

    assert ws.pln.prop_stf["bixel_width"] == 4.0
    assert ws.pln.prop_stf["longitudinal_spot_spacing"] == 1.5


# ----------------------------------------------------------------------
# Presentation mode (default / educational)
# ----------------------------------------------------------------------


def _shown(widget) -> bool:
    # isHidden reflects the explicit visibility flag even while the parent
    # widget itself is never shown (as in these offscreen tests).
    return not widget.isHidden()


def _locked(widget) -> bool:
    # Educational mode disables the container around advanced controls; the
    # control's own enabled flag (e.g. the iso field under "Auto") is separate.
    return not widget.parentWidget().isEnabled()


@pytest.fixture
def gui_settings():
    """Hand out the GUI settings and restore them after the test."""
    settings = get_settings().gui
    snapshot = settings.model_dump()
    yield settings
    for name, value in snapshot.items():
        setattr(settings, name, value)


def test_ui_mode_enum_matches_settings_literal():
    assert {mode.value for mode in UIMode} == set(get_args(GuiUiMode))


def test_default_mode_unlocks_all_controls(qapp, gui_settings):
    gui_settings.ui_mode = "default"
    widget, _ = _make_widget()
    assert widget.ui_mode() is UIMode.DEFAULT
    for editor in (
        widget._cmb_machine,
        widget._spn_bixel,
        widget._cmb_engine,
        widget._cmb_bio_model,
        widget._spn_res_x,
        widget._chk_sequencing,
    ):
        assert _shown(editor)
        assert not _locked(editor)


def test_initial_mode_follows_gui_settings(qapp, gui_settings):
    gui_settings.ui_mode = "educational"
    widget, _ = _make_widget()
    assert widget.ui_mode() is UIMode.EDUCATIONAL
    assert _locked(widget._cmb_machine)
    assert not _locked(widget._txt_gantry)


def test_educational_mode_locks_advanced_controls(qapp):
    widget, _ = _make_widget()
    widget.set_ui_mode(UIMode.EDUCATIONAL)

    basic = (
        widget._cmb_radiation,
        widget._txt_gantry,
        widget._txt_couch,
        widget._spn_num_beams,
        widget._btn_ai_beams,
        widget._btn_apply,
        widget._spn_bixel,
        widget._txt_iso,
        widget._chk_iso_auto,
        widget._cmb_bio_model,
        widget._btn_bio_config,
        widget._btn_tissue,
        widget._spn_res_x,
        widget._btn_ct_grid,
    )
    advanced = (
        widget._cmb_machine,
        widget._cmb_engine,
        widget._btn_engine_config,
        widget._spn_fractions,
        widget._cmb_scenario,
        widget._cmb_quantity,
        widget._chk_sequencing,
        widget._chk_dao,
        widget._chk_conf3d,
    )
    for editor in basic:
        assert not _locked(editor), editor
    # advanced configuration stays visible (greyed out) but is not editable
    for editor in advanced:
        assert _shown(editor), editor
        assert _locked(editor), editor
        assert not editor.isEnabled(), editor


def test_educational_mode_classifies_every_tracked_field(qapp):
    # Guards the lock policy against drift: a newly tracked field must be
    # deliberately added to the basic set here or registered as advanced.
    basic = {
        "radiation",
        "gantry",
        "couch",
        "bixel",
        "iso_auto",
        "iso_per_beam",
        "iso",
        "bio_model",
        "bio_params",
        "res_x",
        "res_y",
        "res_z",
    }
    widget, _ = _make_widget()
    widget.set_ui_mode(UIMode.EDUCATIONAL)
    for key, editor in widget._field_widgets.items():
        assert _shown(editor), key
        assert _locked(editor) == (key not in basic), key


def test_mode_round_trip_restores_all_controls(qapp):
    widget, _ = _make_widget()
    widget.set_ui_mode("educational")  # plain strings are accepted
    assert widget.ui_mode() is UIMode.EDUCATIONAL

    widget.set_ui_mode("default")
    assert widget.ui_mode() is UIMode.DEFAULT
    assert widget._cmb_machine.isEnabled()
    assert widget._cmb_scenario.isEnabled()
    # controls disabled for their own reasons stay disabled
    assert not widget._chk_sequencing.isEnabled()
    assert not widget._cmb_quantity.isEnabled()


def test_set_ui_mode_emits_signal_once_per_change(qapp):
    widget, _ = _make_widget()
    seen = []
    widget.mode_changed.connect(seen.append)

    widget.set_ui_mode(UIMode.EDUCATIONAL)
    widget.set_ui_mode(UIMode.EDUCATIONAL)  # no-op: must not emit again
    widget.set_ui_mode("default")
    assert seen == [UIMode.EDUCATIONAL, UIMode.DEFAULT]

    with pytest.raises(ValueError):
        widget.set_ui_mode("expert")


def test_educational_mode_apply_writes_valid_pln(qapp):
    widget, ws = _make_widget()
    widget.set_ui_mode(UIMode.EDUCATIONAL)

    widget._txt_gantry.setText("0 90 180")
    widget._spn_bixel.setValue(7.0)
    widget._on_apply()

    assert isinstance(ws.pln, Plan)
    assert [float(a) for a in ws.pln.prop_stf["gantry_angles"]] == [0.0, 90.0, 180.0]
    assert ws.pln.prop_stf["bixel_width"] == 7.0
    # locked advanced fields kept their defaults and still reach the plan
    assert ws.pln.num_of_fractions == 30


def test_educational_mode_keeps_locked_field_values(qapp):
    widget, ws = _make_widget()
    widget._spn_fractions.setValue(5)
    widget.set_ui_mode(UIMode.EDUCATIONAL)

    widget._txt_gantry.setText("0")
    widget._on_apply()

    assert ws.pln.num_of_fractions == 5


def test_educational_mode_isocenter_editable(qapp):
    widget, ws = _make_widget()
    widget.set_ui_mode(UIMode.EDUCATIONAL)
    widget._chk_iso_auto.setChecked(False)
    widget._txt_iso.setText("10 20 30")
    assert widget._txt_iso.isEnabled()

    widget._on_apply()

    assert ws.pln.prop_stf["iso_center"] == [10.0, 20.0, 30.0]


@pytest.mark.parametrize("mode", [UIMode.EDUCATIONAL, "educational"])
def test_constructor_mode_overrides_settings(qapp, mode):
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws, ui_mode=mode)
    assert widget.ui_mode() is UIMode.EDUCATIONAL
    assert _locked(widget._cmb_machine)
    assert not _locked(widget._txt_gantry)


def test_switching_mode_does_not_mark_form_modified(qapp):
    widget, ws = _make_widget()
    ws.pln = PhotonPlan(prop_stf={"gantry_angles": [0.0]})
    widget.set_ui_mode(UIMode.EDUCATIONAL)
    widget.set_ui_mode(UIMode.DEFAULT)

    assert widget._btn_apply.text() == "Apply"
    assert widget._lbl_status.text() == ""


def test_bio_model_combo_shows_display_names(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    assert widget._bio_model_name() == "constant_rbe"
    assert widget._cmb_bio_model.currentText() == "Constant RBE"
    assert "Wedenberg (LET)" in _combo_items(widget._cmb_bio_model)


def test_angle_preset_overwrites_angles(qapp):
    widget, ws = _make_widget()
    button = widget._btn_angle_preset
    assert button.text() == "Choose preset…"

    widget._txt_couch.setText("0 10")
    action = next(a for a in button.menu().actions() if a.data().name == "4-field box")
    action.trigger()
    assert widget._txt_gantry.text() == "0 90 180 270"
    assert widget._txt_couch.text() == "0 0 0 0"
    assert widget._spn_num_beams.value() == 4
    assert button.text() == "Choose preset…"

    widget._on_apply()
    assert [float(a) for a in ws.pln.prop_stf["gantry_angles"]] == [0.0, 90.0, 180.0, 270.0]


def test_angle_presets_ordered_by_radiation_mode(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("photons")
    photons = [a.data().name for a in widget._btn_angle_preset.menu().actions()]
    assert photons.index("5-field IMRT star") < photons.index("Lateral opposing")

    widget._cmb_radiation.setCurrentText("protons")
    protons = [a.data().name for a in widget._btn_angle_preset.menu().actions()]
    assert protons.index("Lateral opposing") < protons.index("5-field IMRT star")
    assert None not in protons  # no separator or placeholder entries


def test_angle_presets_follow_an_imported_plan(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="protons")
    names = [a.data().name for a in widget._btn_angle_preset.menu().actions()]
    assert names.index("Lateral opposing") < names.index("5-field IMRT star")


@pytest.mark.parametrize("confirm", [False, True])
def test_angle_preset_keyboard_navigation_waits_for_confirmation(qapp, confirm):
    widget, ws = _make_widget()
    widget._on_apply()
    original = widget.edited_beam_angles()
    button = widget._btn_angle_preset
    menu = button.menu()
    menu.popup(button.mapToGlobal(QPoint(0, button.height())))
    try:
        QTest.keyClick(menu, Qt.Key_Down)
        QTest.keyClick(menu, Qt.Key_Down)
        assert menu.activeAction() == menu.actions()[1]
        assert widget.edited_beam_angles() == original
        assert not widget.has_pending_changes()

        preset = menu.activeAction().data()
        QTest.keyClick(menu, Qt.Key_Return if confirm else Qt.Key_Escape)
        assert not menu.isVisible()
        expected = (list(preset.gantry_angles), list(preset.couch_angles)) if confirm else original
        assert widget.edited_beam_angles() == expected
        assert widget.has_pending_changes() == confirm
        assert ws.pln.prop_stf["gantry_angles"] == original[0]
        assert ws.pln.prop_stf["couch_angles"] == original[1]
    finally:
        menu.close()


def test_angle_preset_editable_in_educational_mode(qapp):
    widget, _ = _make_widget()
    widget.set_ui_mode(UIMode.EDUCATIONAL)
    assert widget._btn_angle_preset.isEnabled()
    assert widget._btn_angle_preset.parentWidget().isEnabled()


def test_set_gantry_angle_edits_the_form_without_applying(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    _type_gantry(widget, "0 120 240")
    widget._on_apply()

    edited = []
    widget.beam_angles_edited.connect(edited.append)
    widget.set_gantry_angle(1, 100.0)

    assert widget._txt_gantry.text() == "0 100 240"
    assert edited[-1] == ([0.0, 100.0, 240.0], [0.0, 0.0, 0.0])
    assert ws.pln.prop_stf["gantry_angles"] == [0.0, 120.0, 240.0]
    assert widget._btn_apply.text() == "Apply *"
    assert widget._btn_discard.isEnabled()

    widget.set_gantry_angle(7, 10.0)  # no such beam
    assert widget._txt_gantry.text() == "0 100 240"


def test_discard_resets_the_form_to_the_applied_plan(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    assert not widget._btn_discard.isEnabled()
    ws.set_many(ct=ct, cst=cst)
    _type_gantry(widget, "0 120 240")
    widget._on_apply()

    _type_gantry(widget, "10 20")
    widget._spn_fractions.setValue(widget._spn_fractions.value() + 1)
    edited = []
    widget.beam_angles_edited.connect(edited.append)
    widget._btn_discard.click()

    assert widget._txt_gantry.text() == "0 120 240"
    assert widget._spn_fractions.value() == ws.pln.num_of_fractions
    assert edited[-1] == ([0.0, 120.0, 240.0], [0.0, 0.0, 0.0])
    assert widget._btn_apply.text() == "Apply"
    assert not widget._btn_discard.isEnabled()


@pytest.mark.parametrize("overrides", [{}, {"kernel_cutoff": 60.0}])
@pytest.mark.parametrize("explicit_engine", [False, True])
def test_discard_restores_engine_overrides_including_defaults(qapp, overrides, explicit_engine):
    widget, ws = _make_widget()
    prop_dose_calc = {"engine": "SVDPB", **overrides} if explicit_engine else dict(overrides)
    ws.pln = PhotonPlan(prop_dose_calc=prop_dose_calc)
    original = widget._build_plan().prop_dose_calc

    # Simulate accepting edits in the engine configuration dialog.
    widget._engine_props["SVDPB"] = {"kernel_cutoff": 50.0, "random_seed": 7}
    widget._update_dirty_state()
    assert widget.has_pending_changes()
    widget._btn_discard.click()

    assert not widget.has_pending_changes()
    assert widget._build_plan().prop_dose_calc == original
    widget._spn_fractions.setValue(widget._spn_fractions.value() + 1)
    widget._on_apply()
    assert ws.pln.prop_dose_calc == original


@pytest.mark.parametrize("plan_cls", [PhotonPlan, IonPlan])
@pytest.mark.parametrize("dose_grid", [None, {}, {"resolution": {"x": 2.0}}])
def test_discard_restores_implicit_geometry(qapp, plan_cls, dose_grid):
    widget, ws = _make_widget()
    ws.pln = plan_cls(prop_dose_calc={"dose_grid": dose_grid})
    original = widget._build_plan()
    applied = ws.pln
    widget._txt_gantry.setText("45 135")
    widget._txt_couch.setText("10 20")
    widget._spn_bixel.setValue(9.0)
    for spin in (widget._spn_res_x, widget._spn_res_y, widget._spn_res_z):
        spin.setValue(9.0)
    assert widget.has_pending_changes()

    edited = []
    widget.beam_angles_edited.connect(edited.append)
    widget._btn_discard.click()

    assert ws.pln is applied
    assert not widget.has_pending_changes()
    assert not widget._btn_discard.isEnabled()
    assert widget._build_plan().prop_stf == original.prop_stf
    assert widget._build_plan().prop_dose_calc == original.prop_dose_calc
    assert edited[-1] == (original.prop_stf["gantry_angles"], original.prop_stf["couch_angles"])

    # A later unrelated edit must not apply any discarded geometry.
    widget._spn_fractions.setValue(widget._spn_fractions.value() + 1)
    widget._on_apply()
    assert ws.pln.prop_stf == original.prop_stf
    assert ws.pln.prop_dose_calc == original.prop_dose_calc


def _select_other_bio_model(widget) -> str:
    """Select a biological model other than the applied one, without applying it."""
    applied = widget._bio_model_name()
    index = next(
        i
        for i in range(widget._cmb_bio_model.count())
        if widget._cmb_bio_model.itemData(i) != applied
    )
    widget._cmb_bio_model.setCurrentIndex(index)
    assert widget.has_pending_changes()
    return widget._bio_model_name()


def test_tissue_dialog_accept_keeps_unapplied_plan_edits(qapp, test_data_protons):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    ct, cst, _ = test_data_protons
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    assert ws.pln is not None
    chosen = _select_other_bio_model(widget)

    def fake_exec(self):
        self.set_values(0, 0.42, 0.04)
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TissueParametersDialog, "exec", fake_exec)
        widget._on_tissue_parameters()

    assert ws.cst.vois[0].alpha_x == pytest.approx(0.42)
    assert widget._bio_model_name() == chosen
    assert widget.has_pending_changes()

    # A plan change still resets the form to the plan
    ws.pln = ws.pln
    assert widget._bio_model_name() != chosen
    assert not widget.has_pending_changes()


def test_voi_editor_accept_keeps_unapplied_plan_edits(qapp, test_data_protons):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.widgets._result_widget import ViewingWidget
    from pyRadPlan.gui.widgets.result.vois_widget import VOIMetadataDialog

    ct, cst, _ = test_data_protons
    widget, ws = _make_widget()
    viewing = ViewingWidget(ws)
    ws.set_many(ct=ct, cst=cst)
    chosen = _select_other_bio_model(widget)

    def fake_exec(self):
        self.alpha_spin.setValue(0.42)
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(VOIMetadataDialog, "exec", fake_exec)
        viewing.vois_widget._edit_voi(cst.vois[0].name)

    assert ws.cst.vois[0].alpha_x == pytest.approx(0.42)
    assert widget._bio_model_name() == chosen
    assert widget.has_pending_changes()


def test_tissue_dialog_unchanged_ok_keeps_exact_values(qapp, test_data_protons):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.cst import create_voi
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.gui.widgets.workflow import WorkflowWidget
    from pyRadPlan.biology.tissues import TissueProperties

    ct, cst, _ = test_data_protons
    tissue = TissueProperties(alpha_x=0.123456, beta_x=0.054321)
    data = dict(cst.vois[0])
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    cst.vois = [create_voi(data), *cst.vois[1:]]

    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    workflow = WorkflowWidget(ws)
    ws.set_many(ct=ct, cst=cst, pln=object(), stf=object(), dij=object(), result={})
    changes = []
    ws.workspace_changed.connect(changes.append)

    def fake_exec(self):
        self.accept()
        return QDialog.DialogCode.Accepted

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TissueParametersDialog, "exec", fake_exec)
        widget._on_tissue_parameters()

    assert changes == []
    assert cst.vois[0].alpha_x == 0.123456
    assert cst.vois[0].beta_x == 0.054321
    assert not workflow._dij_stale and not workflow._result_stale


def test_tissue_dialog_drops_the_cell_line_after_a_model_switch(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    dialog.apply_cell_line_preset(0, _preset_index(dialog, "CHO", "MKM"))
    assert dialog.tissue_properties(0).cell_line == "CHO"

    dialog.set_tissue_model(0, "Generic")
    dialog.accept()

    assert cst.vois[0].tissue_properties.cell_line is None


# ----------------------------------------------------------------------
# Biological model options of the machine
# ----------------------------------------------------------------------


def _bio_entries(widget):
    """(label, enabled, tooltip) of each model dropdown entry."""
    combo = widget._cmb_bio_model
    return [
        (
            combo.itemText(i),
            combo.model().item(i).isEnabled(),
            combo.itemData(i, Qt.ItemDataRole.ToolTipRole) or "",
        )
        for i in range(combo.count())
    ]


def _two_kernel_set_machine():
    """Summary of a carbon machine with α/β kernels of LEM I (default) and of mMKM."""
    from pyRadPlan.biology.models import BioMachineSummary, KernelSetInfo, TissueClassInfo
    from pyRadPlan.biology.tissues import LEMTissueProperties, TissueProperties

    lem = LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30, r_nucleus=5.0)
    mmkm = TissueProperties.from_dict(
        {"model": "mMKM", "alpha_x": 0.1, "beta_x": 0.05, "r_nucleus": 3.9, "r_domain": 0.32}
    )
    sets = {
        "lem1": KernelSetInfo(
            id="lem1",
            kind="alpha_beta",
            fields=("alpha", "beta"),
            provenance="LEM_I",
            default=True,
            declared=True,
            tissue_classes=(TissueClassInfo(lem, "lem1", 0),),
        ),
        "mmkm": KernelSetInfo(
            id="mmkm",
            kind="alpha_beta",
            fields=("alpha_mMKM", "beta_mMKM"),
            provenance="mMKM",
            declared=True,
            tissue_classes=(TissueClassInfo(mmkm, "mmkm", 0),),
        ),
    }
    return BioMachineSummary(
        name="TwoSets",
        radiation_mode="carbon",
        kernel_sets=sets,
        tissue_properties=(lem, mmkm),
        quantities=("physical_dose", "alpha", "beta"),
    )


@pytest.mark.parametrize(
    ("radiation_mode", "expected", "default"),
    [
        ("oxygen", ["α/β kernels · LEM I", "z* kernels · mMKM"], "α/β kernels · LEM I"),
        ("carbon", ["α/β kernels · LEM IV"], "α/β kernels · LEM IV"),
        ("protons", ["Constant RBE", "Wedenberg (LET)"], "Constant RBE"),
        ("photons", ["None (physical dose)"], "None (physical dose)"),
    ],
)
def test_bio_model_dropdown_lists_the_options_of_the_machine(
    qapp, radiation_mode, expected, default
):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText(radiation_mode)
    entries = {label: (enabled, tooltip) for label, enabled, tooltip in _bio_entries(widget)}

    for label in expected:
        assert entries[label][0], f"{label} should be available on {radiation_mode}"
    assert widget._cmb_bio_model.currentText() == default
    # Entries the machine cannot run are listed disabled, their tooltip saying why
    disabled = [tooltip for enabled, tooltip in entries.values() if not enabled]
    if radiation_mode != "photons":
        assert disabled
    assert all("Not available:" in tooltip for tooltip in disabled)


def test_carbon_alpha_beta_tables_are_disabled_without_fluence_spectra(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("carbon")
    entry = next(e for e in _bio_entries(widget) if e[0].startswith("α/β tables"))
    assert not entry[1]
    assert "fluence spectra" in entry[2]


def test_unknown_machine_lists_the_plain_formalisms(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("oxygen")
    widget._cmb_machine.setEditText("NoSuchMachine")
    widget._cmb_machine.lineEdit().editingFinished.emit()

    entries = _bio_entries(widget)
    assert all(enabled for _label, enabled, _tooltip in entries)
    assert "α/β kernels" in [label for label, *_ in entries]
    assert not any("·" in label for label, *_ in entries)
    assert widget._bio_model_name() == "alpha_beta_kernels"


def test_machine_models_follow_finished_edits_not_keystrokes(qapp, monkeypatch):
    from pyRadPlan.gui.widgets.plan import _plan_widget

    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("carbon")
    loads = []
    real_load = _plan_widget.load_machine_quietly

    def counting_load(radiation_mode, machine_name):
        loads.append(machine_name)
        return real_load(radiation_mode, machine_name)

    monkeypatch.setattr(_plan_widget, "load_machine_quietly", counting_load)
    widget._cmb_machine.lineEdit().selectAll()
    QTest.keyClicks(widget._cmb_machine.lineEdit(), "Gen")
    assert loads == []
    assert widget._bio_models_key == ("carbon", "Generic")

    widget._cmb_machine.lineEdit().editingFinished.emit()
    assert widget._bio_models_key == ("carbon", "Gen")
    assert "Gen" in loads
    # The name does not resolve to a machine, so the entries are the plain formalisms
    assert widget._cmb_bio_model.currentText() == "α/β kernels"


@pytest.fixture(scope="module")
def oxygen_machine():
    """The Generic oxygen base data, renamed so no machine file of that name exists."""
    from pyRadPlan.machines import load_from_name

    return load_from_name("oxygen", "Generic").model_copy(update={"name": "Loaded"})


def _machine_items(widget):
    return [widget._cmb_machine.itemText(i) for i in range(widget._cmb_machine.count())]


def test_plan_machine_object_is_listed_and_lists_its_bio_models(qapp, oxygen_machine, monkeypatch):
    from pyRadPlan.gui.widgets.plan import _plan_widget

    reference, _ = _make_widget()
    reference._cmb_radiation.setCurrentText("oxygen")
    generic_entries = _bio_entries(reference)

    loads = []
    real_load = _plan_widget.load_machine_quietly

    def counting_load(radiation_mode, machine_name):
        loads.append(machine_name)
        return real_load(radiation_mode, machine_name)

    monkeypatch.setattr(_plan_widget, "load_machine_quietly", counting_load)
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="oxygen", machine=oxygen_machine)

    assert widget._cmb_machine.currentText() == "Loaded (from plan)"
    assert _machine_items(widget)[0] == "Loaded (from plan)"
    assert "Generic" in _machine_items(widget)
    # The models are those of the carried base data, not the plain formalisms of an unknown
    # name, and its file is never looked up
    assert _bio_entries(widget) == generic_entries
    assert any("·" in label for label, *_ in _bio_entries(widget))
    assert widget._current_machine() is widget._plan_machine_summary
    assert widget._current_machine().name == "Loaded"
    assert not any("Loaded" in name for name in loads)
    assert not widget._lbl_tissue_warning.isVisible()


def test_apply_keeps_the_plan_machine_object_until_another_is_picked(qapp, oxygen_machine):
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="oxygen", machine=oxygen_machine, num_of_fractions=5)

    widget._spn_fractions.setValue(7)
    widget._on_apply()
    assert ws.pln.num_of_fractions == 7
    assert ws.pln.machine is oxygen_machine
    assert widget._cmb_machine.currentText() == "Loaded (from plan)"

    # Another mode has no entry for it; returning offers it again
    widget._cmb_radiation.setCurrentText("carbon")
    assert not any("from plan" in item for item in _machine_items(widget))
    widget._cmb_radiation.setCurrentText("oxygen")
    assert _machine_items(widget)[0] == "Loaded (from plan)"

    widget._cmb_machine.setCurrentText("Generic")
    widget._on_machine_changed()
    widget._on_apply()
    assert ws.pln.machine == "Generic"
    assert widget._cmb_machine.currentText() == "Generic"
    assert not any("from plan" in item for item in _machine_items(widget))


def test_selecting_an_option_sets_its_base_data_parameters(qapp, monkeypatch):
    from pyRadPlan.gui.widgets.plan import _plan_widget

    machine = _two_kernel_set_machine()
    monkeypatch.setattr(_plan_widget, "load_machine_quietly", lambda *_args: machine)
    widget, ws = _make_widget()
    widget._cmb_radiation.setCurrentText("carbon")
    labels = [label for label, *_ in _bio_entries(widget)]
    assert {"α/β kernels · LEM I", "α/β kernels · mMKM"} <= set(labels)
    # The default formalism maps to its first available entry
    assert widget._cmb_bio_model.currentText() == "α/β kernels · LEM I"
    widget._on_apply()
    assert ws.pln.bio_model.data_model == "LEM_I"

    widget._cmb_bio_model.setCurrentIndex(labels.index("α/β kernels · mMKM"))
    assert widget.has_pending_changes()
    assert widget._cmb_bio_model.styleSheet()  # marked modified
    assert not widget._btn_bio_config.styleSheet()  # its parameters are unchanged
    widget._on_apply()
    assert ws.pln.bio_model.model == "alpha_beta_kernels"
    assert ws.pln.bio_model.data_model == "mMKM"

    # Restoring a plan selects the entry of its base data
    ws.pln = IonPlan(radiation_mode="carbon", bio_model={"model": "LEM", "data_model": "LEM_I"})
    assert widget._cmb_bio_model.currentText() == "α/β kernels · LEM I"
    assert not widget.has_pending_changes()


def test_restoring_a_plan_with_an_unsupported_model_keeps_it(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(
        radiation_mode="oxygen", bio_model={"model": "alpha_beta_kernels", "data_model": "mMKM"}
    )

    combo = widget._cmb_bio_model
    assert combo.currentText() == "α/β kernels · mMKM (not supported by this machine)"
    assert not combo.model().item(combo.currentIndex()).isEnabled()
    assert widget._bio_model_name() == "alpha_beta_kernels"
    assert not widget.has_pending_changes()

    # Apply keeps the plan's model instead of silently switching to the LEM I kernels
    widget._txt_gantry.setText("0 90")
    widget._on_apply()
    assert ws.pln.bio_model.model == "alpha_beta_kernels"
    assert ws.pln.bio_model.data_model == "mMKM"

    # Choosing an entry of the machine replaces it with that entry's base data
    widget._select_bio_model("alpha_beta_kernels")
    assert combo.currentText() == "α/β kernels · LEM I"
    widget._on_apply()
    assert ws.pln.bio_model.data_model is None


def test_restoring_a_plan_selects_the_entry_of_an_alias(qapp):
    widget, ws = _make_widget()
    ws.pln = IonPlan(radiation_mode="oxygen", bio_model="MKM")
    assert widget._cmb_bio_model.currentText() == "z* kernels · mMKM"
    assert widget._bio_model_name() == "zstar_kernels"


def test_tissue_indicator_follows_the_model(qapp, tg119):
    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    widget._cmb_radiation.setCurrentText("oxygen")
    assert widget._cmb_bio_model.currentText() == "α/β kernels · LEM I"
    assert widget._lbl_tissue_warning.isHidden()

    # The structures carry reference parameters only, which no z* kernel class matches
    widget._select_bio_model("zstar_kernels")
    assert not widget._lbl_tissue_warning.isHidden()
    tooltip = widget._lbl_tissue_warning.toolTip()
    assert "z* kernels · mMKM" in tooltip
    assert all(voi.name in tooltip for voi in cst.vois)
    assert widget._lbl_status.text() != tooltip  # a label of its own

    widget._select_bio_model("none")
    assert widget._lbl_tissue_warning.isHidden()


def test_tissue_indicator_follows_the_structures(qapp, tg119):
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.biology.tissues import MKMTissueProperties

    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    widget._cmb_radiation.setCurrentText("oxygen")
    widget._select_bio_model("zstar_kernels")
    assert not widget._lbl_tissue_warning.isHidden()

    tissue = MKMTissueProperties(
        model="mMKM",
        alpha_x=0.1,
        beta_x=0.05,
        r_nucleus=3.9,
        r_domain=0.32,
        alpha_0=0.1,
        beta_0=0.05,
    )
    vois = []
    for voi in cst.vois:
        data = dict(voi)
        data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
        vois.append(create_voi(data))
    ws.cst = StructureSet(ct_image=ct, vois=vois)
    assert widget._lbl_tissue_warning.isHidden()


def test_tissue_editor_ranks_the_provenance_of_the_selected_model_first(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import CELL_LINE_PRESETS

    _, cst, _ = test_data_protons
    machine = load_machine_quietly("oxygen", "Generic")

    dialog = TissueParametersDialog(cst, machine, "zstar_kernels")
    assert [t.cell_line for t in dialog._machine_tissues] == ["Generic_MKM"]
    # The HSG · mMKM preset (Inaniwa 2010) is of the mMKM provenance of the z* kernels; all
    # other presets are still offered
    assert _preferred_preset_provenances(dialog) == {"mMKM"}
    assert len([p for p in dialog._picks if p.kind == "preset"]) == len(CELL_LINE_PRESETS)

    dialog = TissueParametersDialog(cst, machine, "alpha_beta_kernels")
    assert _preferred_preset_provenances(dialog) == {"LEM_I"}
    assert [t.cell_line for t in dialog._machine_tissues] == [
        "Generic_LEMax01",
        "Generic_LEMax05",
    ]
    # The preferred entries lead, the others follow grouped by provenance
    preferred = [p.preferred for p in dialog._picks]
    assert preferred == sorted(preferred, reverse=True)

    # Without a model reading tissue-dependent base data, nothing but the classes leads
    dialog = TissueParametersDialog(cst, machine, "constant_rbe")
    assert len(dialog._presets) == len(CELL_LINE_PRESETS)
    assert len(dialog._machine_tissues) == 3
    assert _preferred_preset_provenances(dialog) == set()
    dialog = TissueParametersDialog(cst)
    assert len(dialog._presets) == len(CELL_LINE_PRESETS)


@pytest.mark.parametrize(("answer", "applied"), [("Keep editing", False), ("Apply anyway", True)])
def test_tissue_editor_asks_before_applying_failing_tissues(
    qapp, monkeypatch, test_data_protons, answer, applied
):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly
    from pyRadPlan.gui.widgets.plan._tissue_dialog import ISSUE_BRUSH, TissueParametersDialog

    _, cst, _ = test_data_protons
    alpha_x = cst.vois[0].alpha_x
    machine = load_machine_quietly("oxygen", "Generic")
    dialog = TissueParametersDialog(cst, machine, "zstar_kernels")
    dialog.set_values(0, 0.2, 0.04)
    shown = _answer_message_boxes(monkeypatch, answer)

    dialog.accept()

    assert len(shown) == 1 and cst.vois[0].name in shown[0]
    item = dialog._table.item(0, 0)
    assert item.background() == ISSUE_BRUSH
    assert "No matching tissue class" in item.toolTip()
    assert (dialog.result() == QDialog.DialogCode.Accepted) == applied
    assert dialog.changed == applied
    assert cst.vois[0].alpha_x == pytest.approx(0.2 if applied else alpha_x)


def test_tissue_editor_accepts_matching_tissues_without_asking(
    qapp, monkeypatch, test_data_protons
):
    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    machine = load_machine_quietly("oxygen", "Generic")
    dialog = TissueParametersDialog(cst, machine, "alpha_beta_kernels")
    for row in range(len(cst.vois)):
        dialog.apply_machine_tissue(row, 1)
    _answer_message_boxes(monkeypatch)  # fails if a box is shown

    dialog.accept()

    assert dialog.changed
    assert all(voi.alpha_x == pytest.approx(0.5) for voi in cst.vois)


# ----------------------------------------------------------------------
# Ranked tissue picker, live comparison, quick fixes and reference sources
# ----------------------------------------------------------------------


def _oxygen_dialog(cst, bio_model="zstar_kernels"):
    from pyRadPlan.gui.widgets._tissue_params import load_machine_quietly
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    return TissueParametersDialog(cst, load_machine_quietly("oxygen", "Generic"), bio_model)


def _set_mkm_row(dialog, row, model, r_nucleus=3.9, r_domain=0.32):
    dialog.set_tissue_model(row, model)
    dialog.set_param(row, "r_nucleus", r_nucleus)
    dialog.set_param(row, "r_domain", r_domain)
    dialog.set_param(row, "alpha_0", 0.2)
    dialog.set_param(row, "beta_0", 0.05)


def test_tissue_picker_offers_mmkm_classes_first_and_mkm_presets_below(qapp, test_data_protons):
    from PySide6.QtGui import QFont

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    combo = dialog.preset_combo
    texts = _picker_texts(combo)
    roles = [
        combo.itemData(i, Qt.ItemDataRole.AccessibleDescriptionRole) for i in range(len(texts))
    ]

    assert texts[0] == "Base data · z* kernels · mMKM"
    assert not combo.model().item(0).isEnabled()
    assert texts[1].startswith("✓ Generic_MKM · mMKM (base data)")
    assert isinstance(combo.itemData(1, Qt.ItemDataRole.FontRole), QFont)
    assert combo.currentIndex() == 1  # the first entry that can be applied
    # The HSG · mMKM presets (Inaniwa 2010) have the geometry of the class: exact matches
    assert texts[2].startswith("✓ HSG · mMKM (Inaniwa 2010, derived reference)")
    assert texts[3].startswith("✓ HSG · mMKM (Inaniwa 2010 / Furusawa 2000 reference)")
    assert isinstance(combo.itemData(2, Qt.ItemDataRole.FontRole), QFont)
    assert isinstance(combo.itemData(3, Qt.ItemDataRole.FontRole), QFont)
    separator = roles.index("separator")
    assert separator == 4
    mkm_header = texts.index("Other presets · MKM")
    assert mkm_header > separator
    hsg = next(i for i, t in enumerate(texts) if t.startswith("HSG · MKM (Okamoto 2011)"))
    assert hsg > mkm_header
    cho = next(i for i, t in enumerate(texts) if t.startswith("CHO · MKM (Kase 2008)"))
    assert cho > mkm_header
    assert not texts[hsg].startswith("✓")


def test_applying_an_mkm_preset_for_mmkm_adopts_mmkm(qapp, test_data_protons):
    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)

    note = dialog.apply_cell_line_preset(0, _preset_index(dialog, "CHO", "MKM"))

    assert note == "values from CHO · MKM; adopted as mMKM"
    assert dialog.status_label.text() == note
    assert dialog._model_combos[0].currentText() == "mMKM"
    tissue = dialog.tissue_properties(0)
    assert tissue.model == "mMKM" and tissue.cell_line == "CHO"
    assert tissue.r_domain == pytest.approx(0.2)
    assert tissue.alpha_0 == pytest.approx(0.228)


def test_applying_the_published_mmkm_preset_matches_the_oxygen_base_data(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import reference_radiation_value
    from pyRadPlan.biology.tissues import TissueProperties

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    dialog.set_reference_radiation(0, "co60")
    note = dialog.apply_cell_line_preset(0, _preset_index(dialog, "HSG", "mMKM"))

    assert note == "values from HSG · mMKM"
    assert reference_radiation_value(dialog._radiation_combos[0]) == "xray_200kV"
    tissue = dialog.tissue_properties(0)
    assert tissue == TissueProperties.from_preset("HSG", "mMKM")
    assert "Inaniwa" in tissue.cell_line_source
    dialog.refresh_comparison()
    assert dialog.comparison(0).matched


def test_preset_then_base_data_class_matches_the_base_data(qapp, monkeypatch, test_data_protons):
    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    combo = dialog.preset_combo
    texts = _picker_texts(combo)
    combo.setCurrentIndex(next(i for i, t in enumerate(texts) if t.startswith("CHO · MKM")))
    dialog._on_apply_pick()  # no row selected: every structure
    combo.setCurrentIndex(1)  # the Generic_MKM class
    dialog._on_apply_pick()
    _answer_message_boxes(monkeypatch)  # fails if a box is shown

    dialog.accept()

    machine_class = dialog._machine_tissues[0]
    for voi in cst.vois:
        tissue = voi.tissue_properties
        assert tissue.model == "mMKM"
        assert tissue.r_nucleus == pytest.approx(3.9) and tissue.r_domain == pytest.approx(0.32)
        # The preset's own parameters stay
        assert tissue.alpha_0 == pytest.approx(0.228)
        assert tissue.matches(machine_class, ignore_undeclared=True)


def test_tissue_table_tints_parameters_differing_from_the_base_data(qapp, test_data_protons):
    from PySide6.QtGui import QBrush

    from pyRadPlan.gui.widgets._tissue_params import (
        MISMATCH_COLOR,
        PROVENANCE_WARNING_COLOR,
        editor_mismatch,
        editor_mismatch_is_warning,
    )
    from pyRadPlan.gui.widgets.plan._tissue_dialog import ISSUE_BRUSH, WARNING_BRUSH

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    _set_mkm_row(dialog, 0, "mMKM", r_domain=0.3)
    dialog.refresh_comparison()

    spins = dialog._spins[0]
    assert editor_mismatch(spins["r_domain"]) == "base data: r_domain = 0.32"
    assert "base data: r_domain = 0.32" in spins["r_domain"].toolTip()
    assert "background" in spins["r_domain"].styleSheet()
    assert editor_mismatch(spins["r_nucleus"]) == ""
    # The class leaves these undeclared, so they are never tinted
    for name in ("alpha_x", "beta_x", "alpha_0", "beta_0"):
        assert editor_mismatch(spins[name]) == ""
    assert editor_mismatch(dialog._model_combos[0]) == ""
    item = dialog._table.item(0, 0)
    assert item.background() == ISSUE_BRUSH
    assert "Closest base data class: mMKM / Generic_MKM" in item.toolTip()

    # The MKM label is another provenance than the mMKM kernels: a soft warning tint
    dialog.set_param(0, "r_domain", 0.32)
    dialog.set_tissue_model(0, "MKM")
    dialog.refresh_comparison()
    assert editor_mismatch(spins["r_domain"]) == ""
    combo = dialog._model_combos[0]
    assert editor_mismatch(combo).startswith("base data: mMKM (tissue: MKM)")
    assert "Evaluated with the base data's model" in combo.toolTip()
    assert editor_mismatch_is_warning(combo)
    assert PROVENANCE_WARNING_COLOR in combo.styleSheet()
    assert MISMATCH_COLOR not in combo.styleSheet()
    assert dialog.comparison(0).matched
    assert item.background() == WARNING_BRUSH
    assert item.toolTip().startswith("Matches the base data class")

    dialog.set_tissue_model(0, "mMKM")
    dialog.refresh_comparison()
    assert editor_mismatch(dialog._model_combos[0]) == ""
    assert item.background() == QBrush()
    assert item.toolTip().startswith("Matches the base data class")


def test_tissue_table_updates_the_highlighting_while_editing(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import editor_mismatch

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    _set_mkm_row(dialog, 0, "mMKM", r_domain=0.3)
    assert dialog._comparison_timer.isActive()  # debounced, nothing blocks

    QTest.qWait(dialog._comparison_timer.interval() + 100)
    assert editor_mismatch(dialog._spins[0]["r_domain"])


def test_match_base_data_applies_the_closest_class(qapp, test_data_protons):
    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    for row in (0, 1):
        _set_mkm_row(dialog, row, "MKM", r_domain=0.3)
    dialog.refresh_comparison()
    assert dialog.match_button.isEnabled()

    # The action applies to the selected rows
    dialog._table.selectRow(1)
    dialog.match_button.click()
    assert dialog.comparison(1).matched
    assert not dialog.comparison(0).matched

    assert dialog.match_base_data([0]) == [0]
    tissue = dialog.tissue_properties(0)
    assert tissue.model == "mMKM" and tissue.r_domain == pytest.approx(0.32)
    assert tissue.alpha_0 == pytest.approx(0.2)  # describes the tissue, kept
    assert dialog.comparison(0).matched


def test_tissue_editor_accepts_a_differing_model_label_without_asking(
    qapp, monkeypatch, test_data_protons
):
    from PySide6.QtWidgets import QDialog

    from pyRadPlan.gui.widgets.plan._tissue_dialog import WARNING_BRUSH

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    for row in range(len(cst.vois)):
        _set_mkm_row(dialog, row, "MKM")
    _answer_message_boxes(monkeypatch)  # fails if a box is shown

    dialog.accept()

    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.changed
    assert all(voi.tissue_properties.model == "MKM" for voi in cst.vois)
    item = dialog._table.item(0, 0)
    assert item.background() == WARNING_BRUSH
    assert "is labelled MKM" in item.toolTip()


def test_tissue_editor_relabels_rows_as_the_base_data_model(qapp, test_data_protons):
    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst)
    for row in range(len(cst.vois)):
        _set_mkm_row(dialog, row, "MKM")
    dialog.refresh_comparison()
    assert not dialog.relabel_button.isHidden()
    assert dialog.relabel_button.text() == "Relabel as mMKM"

    # The action applies to the selected rows
    dialog._table.selectRow(1)
    dialog.relabel_button.click()
    assert dialog.tissue_properties(1).model == "mMKM"
    assert dialog.tissue_properties(1).alpha_0 == pytest.approx(0.2)
    assert dialog.tissue_properties(0).model == "MKM"

    assert dialog.relabel([0]) == [0]
    assert not any(dialog.comparison(row).provenance_mismatch for row in (0, 1))
    assert dialog.relabel_button.isHidden()


def test_tissue_editor_tints_a_differing_lem_version_softly(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import editor_mismatch, editor_mismatch_is_warning

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst, "alpha_beta_kernels")
    dialog.apply_machine_tissue(0, 1)
    dialog.set_choice(0, "version", "IV")
    dialog.refresh_comparison()

    comparison = dialog.comparison(0)
    assert comparison.matched and comparison.provenance_mismatch
    version = dialog._choices[0]["version"]
    assert editor_mismatch(version) == "base data: Version = I"
    assert editor_mismatch_is_warning(version)
    assert editor_mismatch_is_warning(dialog._model_combos[0])
    assert dialog.relabel_button.text() == "Relabel as LEM I"


def test_tissue_indicator_distinguishes_warnings_from_errors(qapp, tg119):
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.biology.tissues import MKMTissueProperties

    ct, cst = tg119
    widget, ws = _make_widget()
    ws.set_many(ct=ct, cst=cst)
    widget._cmb_radiation.setCurrentText("oxygen")
    widget._select_bio_model("zstar_kernels")
    assert widget.tissue_marker_state == "error"
    assert widget._lbl_tissue_warning.text() == "⚠"

    tissue = MKMTissueProperties(
        model="MKM", r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05
    )
    vois = []
    for voi in cst.vois:
        data = dict(voi)
        data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
        vois.append(create_voi(data))
    ws.cst = StructureSet(ct_image=ct, vois=vois)

    assert widget.tissue_marker_state == "warning"
    assert not widget._lbl_tissue_warning.isHidden()
    assert widget._lbl_tissue_warning.text() == "ℹ"
    tooltip = widget._lbl_tissue_warning.toolTip()
    assert "is evaluated" in tooltip and "is labelled MKM" in tooltip
    assert "cannot be evaluated" not in tooltip


def test_tissue_editor_shows_where_reference_values_come_from(qapp, test_data_protons):
    from pyRadPlan.cst import create_voi
    from pyRadPlan.gui.widgets._tissue_params import param_source
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import TissueProperties

    _, cst, _ = test_data_protons
    data = dict(cst.vois[0])
    tissue = TissueProperties()
    assert tissue.reference_source == "default"
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    cst.vois = [create_voi(data), *cst.vois[1:]]

    dialog = TissueParametersDialog(cst)
    alpha = dialog._spins[0]["alpha_x"]
    assert param_source(alpha) == ("default", "default 0.1 / 0.05")
    assert "italic" in alpha.styleSheet()
    assert "default 0.1 / 0.05" in alpha.toolTip()

    # Untouched values keep their source through other edits
    dialog.set_tissue_model(0, "MKM")
    dialog.set_param(0, "r_domain", 0.32)
    assert dialog.tissue_properties(0).reference_source == "default"

    # Deriving them from the model parameters marks them as estimated, live
    dialog.set_param(0, "alpha_0", 0.3)
    dialog.set_param(0, "beta_0", 0.05)
    assert param_source(alpha)[0] == "estimated"
    assert "derived: α_0 + β_0 · z_1D,ref" in alpha.toolTip()
    assert dialog.tissue_properties(0).reference_source == "estimated"

    # Editing a value makes it explicit
    alpha.setValue(0.42)
    assert param_source(alpha) == ("explicit", "")
    assert "italic" not in alpha.styleSheet()
    assert param_source(dialog._spins[0]["beta_x"])[0] == "estimated"
    dialog.set_values(0, 0.42, 0.06)
    dialog.accept()
    assert cst.vois[0].tissue_properties.reference_source == "explicit"


def test_tissue_editor_keeps_an_alpha_x_override_alone(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import param_source
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    dialog.set_tissue_model(0, "MKM")
    for name, value in _MKM_BASELINE.items():
        dialog.set_param(0, name, value)
    dialog.derive_reference_params(0)
    alpha = dialog._spins[0]["alpha_x"]
    assert param_source(alpha)[0] == "estimated"
    alpha.setValue(0.42)
    assert param_source(dialog._spins[0]["beta_x"])[0] == "estimated"
    dialog.accept()

    tissue = cst.vois[0].tissue_properties
    assert tissue.alpha_x == pytest.approx(0.42)
    assert tissue.reference_sources == {"alpha_x": "explicit", "beta_x": "estimated"}
    tissue.beta_0 = 0.07
    assert (tissue.alpha_x, tissue.beta_x) == (pytest.approx(0.42), 0.07)


def test_tissue_editor_shows_published_reference_values_as_given(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import param_source
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    index = next(
        i
        for i, (preset, _tissue) in enumerate(dialog._presets)
        if preset.key == ("HSG", "mMKM", "Inaniwa 2010 / Furusawa 2000 reference")
    )
    dialog.apply_cell_line_preset(0, index)
    alpha = dialog._spins[0]["alpha_x"]
    assert param_source(alpha) == ("published", "published (Furusawa 2000)")
    assert "italic" not in alpha.styleSheet()
    assert "published (Furusawa 2000)" in alpha.toolTip()
    # Published values do not follow the model parameters
    dialog.set_param(0, "r_domain", 0.4)
    assert alpha.value() == pytest.approx(0.313)
    dialog.accept()
    tissue = cst.vois[0].tissue_properties
    assert (tissue.alpha_x, tissue.beta_x, tissue.reference_source) == (0.313, 0.0615, "published")


def test_tissue_editor_marks_reference_values_of_a_base_data_class(qapp, test_data_protons):
    from pyRadPlan.gui.widgets._tissue_params import param_source

    _, cst, _ = test_data_protons
    dialog = _oxygen_dialog(cst, "alpha_beta_kernels")
    dialog.apply_machine_tissue(0, 0)

    source, note = param_source(dialog._spins[0]["alpha_x"])
    assert source == "base_data"
    assert note == "from base data class Generic_LEMax01"
    assert dialog.tissue_properties(0).reference_source == "base_data"


def test_tissue_editor_no_op_ok_with_base_data_leaves_the_structures(
    qapp, monkeypatch, test_data_protons
):
    _, cst, _ = test_data_protons
    vois = list(cst.vois)
    dialog = _oxygen_dialog(cst)
    _answer_message_boxes(monkeypatch)  # failing, but unedited tissues are not asked about

    dialog.accept()

    assert not dialog.changed
    assert all(a is b for a, b in zip(cst.vois, vois))


# ------------------------------------------------------------------ reference radiation
_MKM_BASELINE = dict(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615)


@pytest.fixture
def test_photons(monkeypatch):
    """A further registered reference radiation, removed again after the test."""
    from pyRadPlan.biology.tissues import (
        ParameterSource,
        _reference,
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


def _mkm_cst(cst):
    """Copy of a structure set whose first structure carries an MKM tissue."""
    from pyRadPlan.cst import StructureSet, create_voi
    from pyRadPlan.biology.tissues import MKMTissueProperties

    tissue = MKMTissueProperties(**_MKM_BASELINE)
    data = dict(cst.vois[0])
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    return StructureSet(ct_image=cst.ct_image, vois=[create_voi(data), *cst.vois[1:]])


def test_tissue_dialog_selects_the_reference_radiation(qapp, test_data_protons, test_photons):
    from pyRadPlan.gui.widgets._tissue_params import param_source, reference_radiation_value
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import DEFAULT_REFERENCE_RADIATION, MKMTissueProperties

    _, cst, _ = test_data_protons
    cst = _mkm_cst(cst)
    vois = list(cst.vois)
    dialog = TissueParametersDialog(cst)
    column = dialog._param_column("reference_radiation")
    header = dialog._table.horizontalHeaderItem(column).text()
    assert header == "Reference radiation (ref.)"
    assert column < dialog._param_column("alpha_x")
    assert not dialog._table.isColumnHidden(column)
    assert reference_radiation_value(dialog._radiation_combos[0]) == DEFAULT_REFERENCE_RADIATION
    assert not dialog._radiation_combos[0].isHidden()
    assert dialog._radiation_combos[1].isHidden()  # a reference-only tissue

    # Showing the tissues changes nothing
    dialog.accept()
    assert not dialog.changed
    assert all(a is b for a, b in zip(cst.vois, vois))

    dialog = TissueParametersDialog(cst)
    dialog.set_reference_radiation(0, "test_photons")
    expected = MKMTissueProperties(reference_radiation="test_photons", **_MKM_BASELINE)
    alpha = dialog._spins[0]["alpha_x"]
    assert alpha.value() == pytest.approx(expected.alpha_x, abs=1e-4)
    assert "Test photons" in param_source(alpha)[1]
    assert "Test photons" in alpha.toolTip()
    dialog.accept()
    assert dialog.changed
    tissue = cst.vois[0].tissue_properties
    assert tissue.reference_radiation == "test_photons"
    assert tissue.alpha_x == pytest.approx(expected.alpha_x, rel=1e-9)
    assert cst.vois[0].alpha_x == tissue.alpha_x

    # Back to the default radiation: derived anew, a change again
    dialog = TissueParametersDialog(cst)
    dialog.set_reference_radiation(0, DEFAULT_REFERENCE_RADIATION)
    dialog.accept()
    assert dialog.changed
    assert cst.vois[0].tissue_properties == MKMTissueProperties(**_MKM_BASELINE)


def test_tissue_dialog_hides_the_reference_radiation_without_mkm(qapp, test_data_protons):
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog

    _, cst, _ = test_data_protons
    dialog = TissueParametersDialog(cst)
    column = dialog._param_column("reference_radiation")
    dialog.set_tissue_model(0, "Generic")
    dialog.set_tissue_model(1, "Generic")
    assert all(
        dialog._row_tissue_class(row).__name__ != "MKMTissueProperties"
        for row in range(len(cst.vois))
    ) and dialog._table.isColumnHidden(column)
    dialog.set_tissue_model(1, "mMKM")
    assert not dialog._table.isColumnHidden(column)
    assert not dialog._radiation_combos[1].isHidden()
    with pytest.raises(ValueError, match="Unknown reference radiation"):
        dialog.set_reference_radiation(1, "nope")


def test_tissue_dialog_preset_selects_its_reference_radiation(
    qapp, test_data_protons, test_photons
):
    from pyRadPlan.gui.widgets._tissue_params import reference_radiation_value
    from pyRadPlan.gui.widgets.plan._tissue_dialog import TissueParametersDialog
    from pyRadPlan.biology.tissues import (
        CELL_LINE_PRESETS,
        UNPUBLISHED_SOURCE,
        register_cell_line_preset,
    )

    _, cst, _ = test_data_protons
    saved = dict(CELL_LINE_PRESETS)
    try:
        register_cell_line_preset(
            "TestCells",
            "MKM",
            source=UNPUBLISHED_SOURCE,
            reference_radiation="test_photons",
            **_MKM_BASELINE,
        )
        dialog = TissueParametersDialog(cst)
        dialog.apply_cell_line_preset(1, _preset_index(dialog, "TESTCELLS", "MKM"))
    finally:
        CELL_LINE_PRESETS.clear()
        CELL_LINE_PRESETS.update(saved)
    assert reference_radiation_value(dialog._radiation_combos[1]) == "test_photons"
    tissue = dialog.tissue_properties(1)
    assert (tissue.cell_line, tissue.reference_radiation) == ("TestCells", "test_photons")

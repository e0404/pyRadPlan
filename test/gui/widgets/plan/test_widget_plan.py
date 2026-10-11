from typing import get_args

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest

from pyRadPlan._settings import GuiUiMode, get_settings
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.plan import PlanWidget, UIMode
from pyRadPlan.plan import IonPlan, PhotonPlan, Plan
from pyRadPlan.io import load_tg119


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


def _make_widget():
    ws = WorkspaceManager()
    return PlanWidget(workspace=ws), ws


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
    assert widget._bio_model_name() == "kernel_based_lq"

    widget._cmb_radiation.setCurrentText("oxygen")
    assert widget._bio_model_name() == "kernel_based_lq"

    widget._cmb_radiation.setCurrentText("VHEE")
    assert widget._bio_model_name() == "none"


def test_bio_model_selection_survives_mode_switch_when_supported(qapp):
    widget, _ = _make_widget()
    widget._cmb_radiation.setCurrentText("protons")
    widget._select_bio_model("LSM")
    widget._cmb_radiation.setCurrentText("carbon")
    assert widget._bio_model_name() == "LSM"
    widget._cmb_radiation.setCurrentText("photons")
    assert widget._bio_model_name() == "none"


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

    ws.pln = IonPlan(radiation_mode="protons", bio_model={"model": "constant_rbe", "rbe": 1.0})
    assert widget._bio_params["constant_rbe"] == {"rbe": 1.0}
    assert widget._capture_state()["bio_params"] == '{"rbe": 1.0}'


def test_bio_model_config_model():
    from pyRadPlan.bio_models import ConstantRBEModel, EmptyModel, Wedenberg

    assert set(Wedenberg.config_model().model_fields) == {"p0", "p1", "p2"}
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

    dialog = TissueParametersDialog(cst)
    dialog.accept()
    assert not dialog.changed


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
    assert "Wedenberg" in _combo_items(widget._cmb_bio_model)


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

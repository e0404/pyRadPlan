import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.gui.widgets.plan import PlanWidget
from pyRadPlan.plan import IonPlan, PhotonPlan, Plan
from pyRadPlan.io import load_tg119


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


def _make_widget():
    ws = WorkspaceManager()
    return PlanWidget(workspace=ws), ws


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
    assert widget._lbl_beams.text() == "3 beams"


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
    assert not widget._cmb_bio_model.isEnabled()
    assert not widget._cmb_quantity.isEnabled()
    assert not widget._btn_tissue.isEnabled()
    assert not widget._chk_sequencing.isEnabled()
    assert not widget._chk_dao.isEnabled()
    assert not widget._chk_conf3d.isEnabled()
    assert widget._cmb_quantity.count() > 0  # populated from available quantities


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

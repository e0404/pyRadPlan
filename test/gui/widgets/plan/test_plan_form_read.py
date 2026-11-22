"""The Qt plan form shows a plan as the session describes it.

For many plans: the form synchronised from the plan describes it unchanged
(``changes_from_form`` finds nothing to send, an unedited Apply writes the same plan),
its selections are those of ``plan_detail`` and its option lists those of
``plan_options``.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("PySide6")

from pyRadPlan.app._planform import changes_from_form, option_id
from pyRadPlan.gui.widgets._display_names import current_identifier
from pyRadPlan.gui.widgets.plan import PlanWidget
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.io import load_tg119
from pyRadPlan.plan import IonPlan, PhotonPlan

from test_plan_form_parity import plan_diff


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


def _explicit_scenarios() -> Any:
    from pyRadPlan.scenarios import ScenarioSet, SetupUncertainty

    return ScenarioSet.from_points(
        dimensions=[SetupUncertainty()], points={"setup": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
    )


def _loaded_machine() -> Any:
    from pyRadPlan.machines import load_from_name

    return load_from_name("oxygen", "Generic").model_copy(update={"name": "Loaded"})


_GRID = {"dose_grid": {"resolution": {"x": 3.0, "y": 3.0, "z": 3.0}}}


def _stf(**extra: Any) -> dict:
    return {"gantry_angles": [0.0], "couch_angles": [0.0], "bixel_width": 5.0, **extra}


#: (id, plan factory); every plan carries what the form shows (angles, bixel, grid).
PLANS = [
    ("photons", lambda: PhotonPlan(prop_stf=_stf(), prop_dose_calc=dict(_GRID))),
    (
        "photons-beams",
        lambda: PhotonPlan(
            num_of_fractions=5,
            dose_convention="per_fraction",
            prop_stf=_stf(gantry_angles=[0.0, 72.0, 144.0], couch_angles=[0.0, 10.0, 20.0]),
            prop_dose_calc={"engine": "SVDPB", "kernel_cutoff": 60.0, **_GRID},
        ),
    ),
    (
        "photons-manual-iso",
        lambda: PhotonPlan(prop_stf=_stf(iso_center=[1.0, 2.0, 3.0]), prop_dose_calc=dict(_GRID)),
    ),
    (
        "photons-per-beam-iso",
        lambda: PhotonPlan(
            prop_stf=_stf(
                gantry_angles=[0.0, 180.0],
                couch_angles=[0.0, 0.0],
                iso_center=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            ),
            prop_dose_calc=dict(_GRID),
        ),
    ),
    (
        "photons-wc-scenarios",
        lambda: PhotonPlan(
            scenarios={"model": "wcScen", "wc_sigma": 2.0},
            prop_stf=_stf(),
            prop_dose_calc=dict(_GRID),
        ),
    ),
    (
        "photons-explicit-scenarios",
        lambda: PhotonPlan(
            scenarios=_explicit_scenarios(), prop_stf=_stf(), prop_dose_calc=dict(_GRID)
        ),
    ),
    (
        "photons-solver",
        lambda: PhotonPlan(
            prop_opt={"problem": "nonlin_fluence", "solver": {"name": "scipy", "max_iter": 7}},
            prop_stf=_stf(),
            prop_dose_calc=dict(_GRID),
        ),
    ),
    (
        "protons-constant-rbe",
        lambda: IonPlan(
            radiation_mode="protons",
            bio_model={"model": "constant_rbe", "rbe": 1.2},
            prop_stf=_stf(bixel_width=4.0),
            prop_dose_calc={"engine": "HongPB", **_GRID},
        ),
    ),
    (
        "protons-default-model",
        lambda: IonPlan(radiation_mode="protons", prop_stf=_stf(), prop_dose_calc=dict(_GRID)),
    ),
    (
        "carbon-fred",
        lambda: IonPlan(
            radiation_mode="carbon",
            prop_stf=_stf(),
            prop_dose_calc={"engine": "FRED", **_GRID},
        ),
    ),
    (
        "oxygen-loaded-machine",
        lambda: IonPlan(
            radiation_mode="oxygen",
            machine=_loaded_machine(),
            prop_stf=_stf(),
            prop_dose_calc=dict(_GRID),
        ),
    ),
    (
        "oxygen-unsupported-model",
        lambda: IonPlan(
            radiation_mode="oxygen",
            bio_model={"model": "alpha_beta_kernels", "data_model": "mMKM"},
            prop_stf=_stf(),
            prop_dose_calc=dict(_GRID),
        ),
    ),
    (
        "vhee-focused",
        lambda: IonPlan(
            radiation_mode="VHEE", machine="Focused", prop_stf=_stf(), prop_dose_calc=dict(_GRID)
        ),
    ),
]


@pytest.fixture(params=[p for _, p in PLANS], ids=[i for i, _ in PLANS])
def shown(request, qapp, tg119):
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    ws.set_many(ct=tg119[0], cst=tg119[1])
    ws.pln = request.param()
    return widget, ws


def test_the_form_describes_the_plan_unchanged(shown):
    widget, ws = shown
    assert not widget.has_pending_changes()
    assert changes_from_form(ws.pln, widget._form_changes()) == {}


def test_an_unedited_apply_keeps_the_plan(shown):
    widget, ws = shown
    plan = ws.pln
    widget._on_apply()
    assert not widget._status_is_error, widget._lbl_status.text()
    assert plan_diff(plan, ws.pln) == []


def _items(combo) -> list[tuple[Any, str]]:
    return [(combo.itemData(i), combo.itemText(i)) for i in range(combo.count())]


def test_the_selections_are_the_plan_detail(shown):
    widget, ws = shown
    detail = ws.session.plan_detail()
    options = ws.session.plan_options()
    assert widget._cmb_radiation.currentText() == detail["radiation_mode"]
    assert widget._spn_fractions.value() == detail["num_of_fractions"]
    assert widget._cmb_dose_convention.currentData() == detail["dose_convention"]
    engine = detail["engine_config"]["name"] or options["engines"][0]["value"]
    assert current_identifier(widget._cmb_engine) == engine
    assert current_identifier(widget._cmb_scenario) == detail["scenario"]["model"]
    assert current_identifier(widget._cmb_problem) == detail["problem"]["name"]
    option = widget._option_at(widget._cmb_bio_model.currentIndex())
    if option is None:
        # Qt keeps a model the context cannot run as an extra entry; the detail names the
        # (disabled) option reading its base data, if there is one.
        assert widget._unsupported_bio_model is not None
        disabled = {c["value"] for c in options["bio_models"] if c.get("enabled") is False}
        assert detail["bio"]["option"] in disabled | {None}
    else:
        assert option_id(option) == detail["bio"]["option"]
    machine = widget._cmb_machine.currentText()
    if detail["machine_source"] == "plan":
        assert machine == options["machines"][0]["label"]
    else:
        assert machine == detail["machine"]


def _bio_entry(widget) -> tuple[Any, str]:
    option = widget._option_at(widget._cmb_bio_model.currentIndex())
    return (option_id(option) if option is not None else None), widget._cmb_bio_model.currentText()


def test_the_bio_selection_is_the_plan_options(shown):
    widget, ws = shown
    selection = ws.session.plan_options()["bio_selection"]
    option, label = _bio_entry(widget)
    assert selection["option"] == option
    if selection["unsupported"] is not None:
        assert label == selection["unsupported"]["label"]
        assert widget._unsupported_bio_model is not None


@pytest.mark.parametrize("target", ["photons", "protons", "carbon"])
def test_a_mode_change_selects_the_bio_model_of_the_plan_options(shown, target):
    from pyRadPlan.biology._editing import SELECTION_PARAMETERS

    widget, ws = shown
    if target == ws.pln.radiation_mode:
        pytest.skip("not a mode change")
    option, label = _bio_entry(widget)
    name = widget._bio_model_name()
    current = {
        "radiation_mode": ws.pln.radiation_mode,
        "model": name,
        "option": option,
        "values": {
            k: v
            for k, v in widget._bio_params.get(name, {}).items()
            if k not in SELECTION_PARAMETERS
        },
        "label": label.split(" (not supported")[0],
    }
    widget._cmb_radiation.setCurrentText(target)
    selection = ws.session.plan_options(
        radiation_mode=target,
        machine=widget._cmb_machine.currentText(),
        engine=current_identifier(widget._cmb_engine),
        current_bio=current,
    )["bio_selection"]
    assert selection["option"] == _bio_entry(widget)[0]
    note = selection["note"]
    if note is not None:
        assert widget._lbl_status.text() == note["text"]
        assert widget._lbl_status.toolTip() == note["reason"]


def test_the_option_lists_are_the_plan_options(shown):
    widget, ws = shown
    options = ws.session.plan_options()
    machines = [widget._cmb_machine.itemText(i) for i in range(widget._cmb_machine.count())]
    assert machines == [c["label"] for c in options["machines"]]
    for combo, key in (
        (widget._cmb_engine, "engines"),
        (widget._cmb_problem, "problems"),
        (widget._cmb_scenario, "scenario_models"),
    ):
        assert _items(combo) == [(c["value"], c["label"]) for c in options[key]], key
    listed = [
        (option_id(o), widget._cmb_bio_model.itemText(i), o.available)
        for i, o in enumerate(widget._bio_options)
        if o is not None
    ]
    expected = [(c["value"], c["label"], c.get("enabled", True)) for c in options["bio_models"]]
    assert listed == expected
    presets = [a.data().name for a in widget._btn_angle_preset.menu().actions()]
    assert presets == [p["name"] for p in options["beam_presets"]]

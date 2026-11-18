from importlib.util import find_spec
from types import SimpleNamespace

import numpy as np
import pytest

from pyRadPlan.app import (
    StageTracker,
    Workspace,
    available_actions,
    merge_loaded_data,
    nested_fraction,
    recalculation_angles_match,
    stage_status,
    status_text,
)
from pyRadPlan.app._stages import progress_detail
from pyRadPlan.app._workspace import WORKSPACE_KEYS


class Tracked:
    """A workspace with a tracker following it, as a session keeps one."""

    def __init__(self) -> None:
        self.ws = Workspace()
        self.tracker = StageTracker()
        self.ws.subscribe(self._update)

    def _update(self, keys):
        self.tracker = self.tracker.update(keys, self.ws.present(), self.ws.cst)

    def stages(self):
        return stage_status(self.ws.present(), self.tracker)


def _all_loaded(tracked, cst=None):
    tracked.ws.set_many(
        ct=object(), cst=cst or object(), pln=object(), stf=object(), dij=object(), result={}
    )


def test_loading_everything_at_once_leaves_products_current():
    t = Tracked()
    _all_loaded(t)
    assert t.stages() == dict.fromkeys(("data", "plan", "stf", "dij", "optimization"), "done")


def test_plan_change_marks_downstream_stale_and_recomputing_clears_it():
    t = Tracked()
    _all_loaded(t)

    t.ws.pln = object()
    assert t.tracker.stf_stale and t.tracker.dij_stale and t.tracker.result_stale
    assert t.stages()["dij"] == "stale" and t.stages()["optimization"] == "stale"

    t.ws.set_many(stf=object(), dij=object())
    assert not t.tracker.stf_stale and not t.tracker.dij_stale and t.tracker.result_stale

    t.ws.result = {"w": 1}
    assert t.stages()["optimization"] == "done"


def test_moved_beams_outdate_the_dose_influence():
    t = Tracked()
    _all_loaded(t)
    t.ws.stf = object()
    assert not t.tracker.stf_stale
    assert t.tracker.dij_stale and t.tracker.result_stale


def test_removed_products_are_missing_not_stale():
    t = Tracked()
    _all_loaded(t)
    t.ws.pln = object()
    t.ws.clear(["dij", "result"])
    assert t.stages()["dij"] == "missing" and not t.tracker.dij_stale
    assert t.stages()["optimization"] == "missing"


def test_ct_change_outdates_the_products():
    """The one deliberate deviation from the Qt widget (see StageTracker.update)."""
    t = Tracked()
    _all_loaded(t)
    t.ws.ct = object()
    assert t.tracker.stf_stale and t.tracker.dij_stale and t.tracker.result_stale


def _tg119():
    from pyRadPlan.io import load_tg119

    return load_tg119()


def test_only_tissue_changes_of_the_structures_outdate_the_dose_influence():
    from pyRadPlan.biology.tissues import MKMTissueProperties
    from pyRadPlan.cst import create_voi

    _, cst = _tg119()
    t = Tracked()
    _all_loaded(t, cst)

    cst.vois[0].objectives = []
    t.ws.cst = cst
    assert not t.tracker.dij_stale and t.tracker.result_stale

    t.ws.set_many(dij=object(), result={})
    voi = cst.vois[0]
    tissue = MKMTissueProperties(alpha_x=0.2, beta_x=0.04, r_nucleus=3.9, r_domain=0.32)
    data = dict(voi)
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    cst.vois = [create_voi(data), *cst.vois[1:]]
    t.ws.cst = cst
    assert t.tracker.dij_stale and t.tracker.result_stale
    assert not t.tracker.stf_stale


def test_renaming_a_structure_keeps_the_dose_influence():
    _, cst = _tg119()
    t = Tracked()
    _all_loaded(t, cst)
    cst.vois = [cst.vois[0].model_copy(update={"name": "Renamed"}), *cst.vois[1:]]
    t.ws.cst = cst
    assert not t.tracker.dij_stale


def test_available_actions_follow_the_qt_buttons():
    assert available_actions(set()) == dict.fromkeys(
        ("generate_stf", "calc_dose_influence", "optimize", "recalc"), False
    )
    patient_plan = {"ct", "cst", "pln"}
    actions = available_actions(patient_plan)
    assert actions["generate_stf"] and actions["calc_dose_influence"]
    assert not actions["optimize"] and not actions["recalc"]
    actions = available_actions(patient_plan | {"stf", "dij"})
    assert actions["optimize"] and not actions["recalc"]
    assert available_actions(set(WORKSPACE_KEYS))["recalc"]


@pytest.mark.parametrize(
    ("present", "tracker", "text"),
    [
        (set(), StageTracker(), "No data loaded"),
        ({"ct", "cst"}, StageTracker(), "configure a plan"),
        ({"ct", "cst", "pln"}, StageTracker(), "Ready for dose influence"),
        ({"ct", "cst", "pln", "stf", "dij"}, StageTracker(dij_stale=True), "outdated"),
        ({"ct", "cst", "pln", "stf", "dij"}, StageTracker(), "Ready for optimization"),
        (set(WORKSPACE_KEYS), StageTracker(result_stale=True), "re-optimize"),
        (set(WORKSPACE_KEYS), StageTracker(), "result available"),
    ],
)
def test_status_text(present, tracker, text):
    assert text in status_text(present, tracker)


def test_new_ct_replaces_the_whole_patient():
    payload, warnings = merge_loaded_data({"ct": 1, "cst": 2}, True, WORKSPACE_KEYS)
    assert payload == {"ct": 1, "cst": 2, "pln": None, "stf": None, "dij": None, "result": None}
    assert not warnings


def test_import_without_ct_keeps_the_patient():
    payload, _ = merge_loaded_data({"pln": 3, "dij": None}, True, WORKSPACE_KEYS)
    assert payload == {"pln": 3}


def test_structures_need_a_ct():
    payload, warnings = merge_loaded_data({"cst": 2}, False, WORKSPACE_KEYS)
    assert payload == {} and len(warnings) == 1
    payload, warnings = merge_loaded_data({"cst": 2}, True, WORKSPACE_KEYS)
    assert payload == {"cst": 2} and not warnings


def test_bare_dose_becomes_a_result():
    payload, _ = merge_loaded_data({"dose": "image"}, True, WORKSPACE_KEYS)
    assert payload == {"result": {"physical_dose": "image"}}


def _stf(*angles):
    beams = [SimpleNamespace(gantry_angle=g, couch_angle=c) for g, c in angles]
    return SimpleNamespace(beams=beams)


@pytest.mark.parametrize(
    ("prop_stf", "match"),
    [
        ({"gantry_angles": [360, -180], "couch_angles": [0]}, True),
        ({"gantry_angles": [0, 180]}, True),
        ({}, True),
        ({"gantry_angles": [90, 180], "couch_angles": [0, 0]}, False),
        ({"gantry_angles": [0], "couch_angles": [0]}, False),
        ({"gantry_angles": [float("nan"), 180]}, False),
    ],
)
def test_recalculation_angles_match(prop_stf, match):
    pln = SimpleNamespace(prop_stf=prop_stf)
    assert recalculation_angles_match(pln, _stf((0.0, 0.0), (180.0, 0.0))) is match


def test_nested_fraction_and_detail():
    level = SimpleNamespace
    outer = level(name="Beam", current=1, total=2, fraction=0.5)
    inner = level(name="Ray", current=50, total=100, fraction=0.5)
    assert nested_fraction([outer, inner]) == pytest.approx(0.75)
    assert nested_fraction([level(name="Load", current=3, total=None, fraction=None)]) is None
    assert progress_detail([outer, inner]) == "Beam 1/2 · Ray 50/100"


@pytest.mark.skipif(find_spec("PySide6") is None, reason="needs the gui extra")
def test_tracker_agrees_with_the_qt_workflow_widget():
    """Replays workspace changes on the Qt widget and the tracker, comparing the flags."""
    from PySide6.QtWidgets import QApplication

    from pyRadPlan.gui.widgets.workflow import WorkflowWidget
    from pyRadPlan.gui.workspace import WorkspaceManager

    _app = QApplication.instance() or QApplication([])
    _, cst = _tg119()
    manager = WorkspaceManager()
    widget = WorkflowWidget(manager)
    tracker = StageTracker()

    def apply(**changes):
        nonlocal tracker
        if changes.pop("_clear", False):
            manager.clear(list(changes))
        else:
            manager.set_many(**changes)
        present = {k for k in WORKSPACE_KEYS if manager.has(k)}
        tracker = tracker.update(list(changes), present, manager.cst)
        assert (tracker.dij_stale, tracker.result_stale) == (
            widget._dij_stale,
            widget._result_stale,
        ), changes

    rng = np.random.default_rng(7)
    apply(ct=object(), cst=cst, pln=object(), stf=object(), dij=object(), result={})
    for _ in range(60):
        key = rng.choice(["cst", "pln", "stf", "dij", "result", "stf+dij", "clear"])
        if key == "stf+dij":
            apply(stf=object(), dij=object())
        elif key == "clear":
            apply(_clear=True, **{str(rng.choice(["dij", "result"])): None})
        elif key == "cst":
            apply(cst=cst)
        else:
            apply(**{str(key): object() if key != "result" else {}})
    widget.deleteLater()

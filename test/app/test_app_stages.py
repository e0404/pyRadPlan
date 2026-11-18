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
    tissue_signature,
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


#: What each product is computed from; ``"tissue"`` is the structures' tissue properties.
_UPSTREAM = {
    "stf": ("ct", "pln"),
    "dij": ("ct", "pln", "stf", "tissue"),
    "result": ("ct", "pln", "cst", "stf", "dij"),
}


class _VersionOracle:
    """Staleness by version counters: a product is stale once an input changed since."""

    def __init__(self) -> None:
        self.versions = dict.fromkeys((*WORKSPACE_KEYS, "tissue"), 0)
        self.inputs: dict[str, dict[str, int]] = {}

    def write(self, keys, present, tissue_changed):
        for key in keys:
            self.versions[key] += 1
        if tissue_changed:
            self.versions["tissue"] += 1
        for product, upstream in _UPSTREAM.items():
            if product not in present:
                self.inputs.pop(product, None)
            elif product in keys:
                self.inputs[product] = {key: self.versions[key] for key in upstream}

    def stale(self, product):
        recorded = self.inputs.get(product)
        return recorded is not None and any(self.versions[k] != v for k, v in recorded.items())


def test_tracker_agrees_with_input_versions_over_random_changes():
    """Replays random workspace writes and compares the flags with a version oracle."""
    from pyRadPlan.biology.tissues import MKMTissueProperties
    from pyRadPlan.cst import create_voi

    _, cst = _tg119()
    voi = cst.vois[0]
    tissue = MKMTissueProperties(alpha_x=0.2, beta_x=0.04, r_nucleus=3.9, r_domain=0.32)
    data = dict(voi)
    data.update(tissue_properties=tissue, alpha_x=tissue.alpha_x, beta_x=tissue.beta_x)
    other = SimpleNamespace(vois=[create_voi(data), *cst.vois[1:]])
    renamed = SimpleNamespace(vois=[voi.model_copy(update={"name": "x"}), *cst.vois[1:]])
    structure_sets = [cst, other, renamed]

    t = Tracked()
    oracle = _VersionOracle()
    rng = np.random.default_rng(7)

    def apply(keys, clear=False):
        before = tissue_signature(t.ws.cst)
        if clear:
            t.ws.clear(keys)
        else:
            t.ws.set_many(**{key: values[key]() for key in keys})
        oracle.write(keys, t.ws.present(), tissue_signature(t.ws.cst) != before)
        for product, flag in (
            ("stf", "stf_stale"),
            ("dij", "dij_stale"),
            ("result", "result_stale"),
        ):
            assert getattr(t.tracker, flag) == oracle.stale(product), (keys, product)

    values = {
        "ct": object,
        "cst": lambda: structure_sets[int(rng.integers(3))],
        "pln": object,
        "stf": object,
        "dij": object,
        "result": dict,
    }
    apply(list(WORKSPACE_KEYS))
    for _ in range(300):
        choice = rng.integers(4)
        if choice == 0:
            apply([str(rng.choice(["stf", "dij", "result"]))], clear=True)
        elif choice == 1:
            apply([str(k) for k in rng.choice(WORKSPACE_KEYS, size=2, replace=False)])
        else:
            apply([str(rng.choice(WORKSPACE_KEYS))])

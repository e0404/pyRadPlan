import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.app import Session, Workspace
from pyRadPlan.app._results import (
    loaded_results,
    quantity_meta,
    result_catalog,
    result_components,
    result_keys_meta,
    sanitize_tag,
)
from pyRadPlan.result import PlanResult, create_result

TINY_PLAN = {"gantry_angles": [0.0, 180.0], "bixel_width": 10.0, "dose_grid_resolution": [8, 8, 8]}


def _image(value=1.0, size=2, spacing=1.0):
    image = sitk.GetImageFromArray(np.full((size, size, size), value, dtype=np.float32))
    image.SetSpacing((spacing,) * 3)
    return image


@pytest.fixture(scope="module")
def optimized(client_factory):
    client = client_factory()
    client.session = client.ok("session.create", session=False)["result"]["session"]
    client.ok("data.load_phantom", {"name": "TG119"})
    client.ok("plan.set", TINY_PLAN)
    client.ok("workflow.run_all", {"solver": "scipy", "max_iter": 5})
    yield client
    client.manager.close_all()


def test_results_catalog_and_revisions(optimized):
    state = optimized.ok("session.state")["result"]
    results = state["results"]
    assert set(results) == {
        "groups",
        "scenario_labels",
        "default",
        "names",
        "active",
        "tags",
        "default_tag",
    }
    assert results["names"] == ["current"] and results["active"] == "current"
    assert results["tags"] == []
    dose = next(g for g in results["groups"] if g["base"] == "physical_dose" and not g["tag"])
    assert set(dose) == {
        "base",
        "tag",
        "label",
        "meta_label",
        "unit",
        "components",
        "scenario_labels",
    }
    assert dose["label"] == "dose" and dose["meta_label"] == "Dose" and dose["unit"] == "Gy"
    assert dose["scenario_labels"] == results["scenario_labels"]
    kinds = [c["kind"] for c in dose["components"]]
    assert kinds[0] == "full" and kinds.count("beam") == 2
    assert dose["components"][1] == {
        "key": "physical_dose_beam 0",
        "label": "beam 0",
        "kind": "beam",
    }
    assert results["default"] == "physical_dose"
    assert results["default_tag"] == "photons_30fx"
    assert state["quantities"] == ["physical_dose"]
    revisions = state["revisions"]
    assert {"ct", "physical_dose", "physical_dose_beam 1"} <= set(revisions["volumes"])
    assert revisions["workspace"]["result"] >= 1


def test_snapshot_is_a_result_of_its_own_and_survives_reoptimization(optimized):
    session = optimized.manager.get(optimized.session)
    before = optimized.ok("session.state")["result"]["revisions"]["volumes"]
    active = session.workspace.result
    result = optimized.ok("workflow.snapshot", {"tag": "first try"})["result"]
    assert result["tag"] == "first_try"
    state = result["state"]
    assert state["results"]["names"] == ["current", "first_try"]
    assert state["results"]["active"] == "current"
    assert state["results"]["tags"] == ["first_try"]
    tagged = [g for g in state["results"]["groups"] if g["tag"] == "first_try"]
    assert tagged and tagged[0]["label"] == "dose (first_try)"
    assert tagged[0]["components"][0]["key"] == "first_try/physical_dose"
    assert "first_try/physical_dose" in state["quantities"]
    volumes = state["revisions"]["volumes"]
    assert volumes["physical_dose"] == before["physical_dose"]
    assert volumes["first_try/physical_dose"] == before["physical_dose"]
    assert optimized.call("workflow.snapshot", {"tag": "current"})["error"]["type"] == "ValueError"

    snapshot = session.workspace.results["first_try"]
    assert session.workspace.result is active and snapshot is not active
    assert snapshot["physical_dose"].values is active["physical_dose"].values
    np.testing.assert_array_equal(snapshot.fluence, active.fluence)

    state = optimized.ok("workflow.optimize", {"solver": "scipy", "max_iter": 3})["result"]
    assert session.workspace.results["first_try"] is snapshot
    assert session.workspace.result is not active
    assert state["results"]["tags"] == ["first_try"]
    assert state["revisions"]["volumes"]["physical_dose"] != before["physical_dose"]
    assert state["revisions"]["volumes"]["first_try/physical_dose"] == before["physical_dose"]
    catalog = optimized.ok("objectives.catalog")["result"]
    assert catalog["reference_images"] == [
        {"value": {"$result": "physical_dose"}, "label": "dose"},
        {"value": {"$result": "first_try/physical_dose"}, "label": "dose (first_try)"},
    ]

    volume = optimized.ok("view.volume", {"name": "first_try/physical_dose_beam 1"})["result"]
    assert volume["unit"] == "Gy"
    dvh = optimized.ok("analysis.dvh", {"quantity": "first_try/physical_dose"})["result"]
    assert dvh["unit"] == "Gy" and dvh["label"] == "Dose"

    optimized.ok("workflow.snapshot", {"tag": "first try"})
    assert session.workspace.results["first_try"] is not snapshot
    assert list(session.workspace.results) == ["current", "first_try"]


def test_history_has_numbered_log_and_optimization_points(optimized):
    history = optimized.ok("session.history")["result"]
    seqs = [event["seq"] for event in history["log"]]
    assert seqs == sorted(seqs) and len(set(seqs)) == len(seqs)
    optimization = history["optimization"]
    assert optimization["finished"] and optimization["outcome"] == "done"
    assert optimization["points"] and "objective" in optimization["points"][-1]
    later = optimized.ok("session.history", {"since": seqs[-1]})["result"]
    assert later["log"] == []


def test_pause_needs_a_running_operation(optimized):
    error = optimized.call("session.pause")["error"]
    assert error["code"] == "not_available"
    assert optimized.ok("session.cancel")["result"] == {}


def test_result_key_helpers():
    image = _image()
    result = create_result(
        {
            "a": image,
            "a_beam": [image, image],
            "a_scenario": [image, None, image],
            "b_scenario": [None, None, None],
            "w": [1, 2],
        }
    )
    assert list(result_components(result)) == [
        "a",
        "a_beam 0",
        "a_beam 1",
        "a_scenario 0",
        "a_scenario 2",
    ]
    assert result_components(None) == {}
    other = create_result({"a": _image(2.0)})
    components = result_components([("", result), ("other", other)])
    assert list(components)[-1] == "other/a"
    assert components["other/a"] is other["a"].values
    assert list(result_components([("other", other)])) == ["other/a"]

    assert sanitize_tag(" my run! ") == "my_run"
    assert sanitize_tag("scenario") == "scenario"
    with pytest.raises(ValueError):
        sanitize_tag(" /! ")


def test_catalog_lists_every_result_with_its_own_scenario_labels():
    image = _image()
    robust = create_result(
        {
            "physical_dose": image,
            "physical_dose_scenario": [image, image],
            "scenario_labels": ("nominal", "shifted"),
        }
    )
    imported = create_result({"physical_dose": _image(2.0, size=3)})
    ws = Workspace()
    ws.set_many(results={"imported": imported, "plan": robust}, active_result="plan")

    class Dij:
        def scenario_labels(self):
            return ["from", "the", "dij"]

    catalog = result_catalog(ws.result_entries(), Dij(), None, list(ws.results), ws.active_result)
    assert catalog["names"] == ["imported", "plan"]
    assert catalog["active"] == "plan" and catalog["tags"] == ["imported"]
    assert catalog["default"] == "physical_dose" and catalog["default_tag"] == "result"
    assert catalog["scenario_labels"] == ["nominal", "shifted"]
    assert [(g["tag"], g["label"]) for g in catalog["groups"]] == [
        ("", "dose"),
        ("imported", "dose (imported)"),
    ]
    active, other = catalog["groups"]
    assert active["scenario_labels"] == ["nominal", "shifted"]
    assert [c["key"] for c in active["components"]] == [
        "physical_dose",
        "physical_dose_scenario 0",
        "physical_dose_scenario 1",
    ]
    assert other["scenario_labels"] == []
    assert other["components"] == [
        {"key": "imported/physical_dose", "label": "full", "kind": "full"}
    ]

    ws.set_many(results={"plan": create_result({"physical_dose": image})}, active_result="plan")
    catalog = result_catalog(ws.result_entries(), Dij(), None, list(ws.results), "plan")
    assert catalog["scenario_labels"] == ["from", "the", "dij"]
    assert catalog["groups"][0]["scenario_labels"] == ["from", "the", "dij"]

    ws.set_many(results={"imported": imported})
    catalog = result_catalog(ws.result_entries(), None, None, list(ws.results), ws.active_result)
    assert catalog["active"] == "current" and catalog["tags"] == ["imported"]
    assert catalog["default"] == "imported/physical_dose"

    empty = result_catalog()
    assert empty["groups"] == [] and empty["names"] == [] and empty["default"] is None


@pytest.mark.parametrize(
    ("key", "meta", "beam_only"),
    [
        ("rbe_x_dose", ("RBE-weighted Dose", "Gy (RBE)"), ("RBE-weighted Dose", "Gy (RBE)")),
        ("physical_dose_beam", ("Dose", "Gy"), ("Dose", "Gy")),
        ("let_scenario", ("LET", "keV/μm"), ("", "")),
        ("physical_dose_beam_scenario", ("Dose", "Gy"), ("", "")),
        ("physical_dose_saved", ("Quantity", ""), ("", "")),
        ("saved/physical_dose", ("Dose", "Gy"), ("Dose", "Gy")),
        ("saved/let_beam", ("LET", "keV/μm"), ("LET", "keV/μm")),
    ],
)
def test_quantity_meta_resolves_component_keys(key, meta, beam_only):
    """One table serves the viewers (any component key) and the save dialog (per beam)."""
    assert quantity_meta(key) == meta
    assert quantity_meta(key, ("", ""), suffixes=("_beam",)) == beam_only


def test_result_keys_meta_resolves_the_keys_of_other_results():
    keys = [
        "rbe_x_dose",
        "physical_dose",
        "carbon/rbe_x_dose",
        "carbon/physical_dose",
        "carbon/let",
        "carbon/rbe_x_dose_beam 0",
        "carbon/rbe_x_dose_scenario 2",
        "mystery",
    ]
    meta = result_keys_meta(keys)
    assert meta["rbe_x_dose"] == meta["carbon/rbe_x_dose"] == ("RBE-weighted Dose", "Gy (RBE)")
    assert meta["carbon/rbe_x_dose_beam 0"] == meta["rbe_x_dose"]
    assert meta["carbon/rbe_x_dose_scenario 2"] == meta["rbe_x_dose"]
    assert meta["carbon/physical_dose"] == ("Dose", "Gy")
    assert meta["carbon/let"] == ("LET", "keV/μm")
    assert meta["mystery"] == ("Quantity", "")


def test_results_survive_a_workspace_pickle(optimized):
    session = optimized.manager.get(optimized.session)
    names = list(session.workspace.results)
    saved = optimized.call("export.save", {"object": "workspace", "format": "pickle"})
    if not saved["ok"]:
        pytest.skip(f"export.save unavailable: {saved['error']['message']}")
    params = {"filename": saved["result"]["filename"], "data": saved["result"]["data"]}
    state = optimized.ok("data.load_file", params, buffers=saved["buffers"])["result"]
    assert state["results"]["names"] == names
    assert state["results"]["active"] == "current"
    assert state["results"]["tags"] == ["first_try"]
    assert list(session.workspace.results) == names


def test_loaded_results_of_session_files_plan_results_and_doses():
    result, snapshot = create_result({"physical_dose": _image()}), create_result({"let": _image()})
    session_file = {"results": {"plan": result, "first": snapshot}, "active_result": "plan"}
    assert loaded_results(session_file) == ({"plan": result, "first": snapshot}, "plan")
    results, active = loaded_results({"result": {"physical_dose": _image()}})
    assert list(results) == ["current"] and active is None
    assert isinstance(results["current"], PlanResult)
    assert loaded_results({"result": result}) == ({"current": result}, None)
    dose = _image()
    assert loaded_results({"dose": dose}) == ({"imported": {"physical_dose": dose}}, None)
    assert loaded_results({"pln": 1}) == ({}, None)


def test_session_saved_up_to_0_9_is_split_into_separate_results():
    dose, snap, snap_beam, imported = _image(1.0), _image(2.0), _image(3.0), _image(4.0)
    legacy = {
        "physical_dose": dose,
        "physical_dose_beam": [dose, dose],
        "physical_dose_first_try": snap,
        "physical_dose_beam_first_try": [snap_beam, snap_beam],
        "rbe_x_dose_first_try": snap,
        "import_plan_dose": imported,
        "w": np.ones(3),
        # A 0.9 snapshot copied every entry, also the weights and the scenario lists
        "w_first_try": np.zeros(3),
        "physical_dose_scenario_first_try": [snap, None],
        "scenario_labels_first_try": ("nominal", "shifted"),
    }
    results, active = loaded_results({"result": legacy, "saved_result_tags": ["first_try"]})
    assert active is None
    assert list(results) == ["current", "first_try", "plan_dose"]
    current, first_try, plan_dose = results.values()
    assert current.stored_quantities == ["physical_dose"]
    assert current.stored_beam_quantities == ["physical_dose"]
    assert current["physical_dose"].values is dose
    np.testing.assert_array_equal(current.fluence, np.ones(3))
    assert first_try.stored_quantities == ["physical_dose", "rbe_x_dose"]
    assert first_try["physical_dose"].values is snap
    assert [d.values for d in first_try.beam_quantities["physical_dose"]] == [snap_beam] * 2
    np.testing.assert_array_equal(first_try.fluence, np.zeros(3))
    assert first_try.scenario_labels == ("nominal", "shifted")
    assert first_try.stored_scenario_quantities == ["physical_dose"]
    assert plan_dose.stored_quantities == ["physical_dose"]
    assert plan_dose["physical_dose"].values is imported

    session = Session(auto_plan=False)
    session.merge_loaded({"result": legacy, "saved_result_tags": ["first_try"]})
    assert list(session.workspace.results) == ["current", "first_try", "plan_dose"]
    assert session.state()["results"]["tags"] == ["first_try", "plan_dose"]

    only_import, _ = loaded_results({"result": {"import_dose": imported}})
    assert list(only_import) == ["dose"]


def test_workspace_file_of_version_0_9_loads(tmp_path):
    import pickle

    from pyRadPlan.io import load_data

    dose = _image(1.0)
    path = tmp_path / "workspace.pkl"
    legacy = {
        "result": {
            "physical_dose": dose,
            "w": np.ones(3),
            "physical_dose_a": dose,
            "w_a": np.ones(3),
        },
        "saved_result_tags": ["a"],
    }
    path.write_bytes(pickle.dumps(legacy))

    session = Session(auto_plan=False)
    session.merge_loaded(load_data(path))

    assert list(session.workspace.results) == ["current", "a"]
    np.testing.assert_array_equal(session.workspace.results["a"].fluence, np.ones(3))


def test_loading_without_a_ct_keeps_the_other_results_and_a_ct_replaces_them():
    session = Session(auto_plan=False)
    ws = session.workspace
    plan, snapshot = create_result({"physical_dose": _image()}), create_result({"let": _image()})
    ws.set_many(ct=object(), results={"plan": plan, "first": snapshot}, active_result="plan")
    ws.pln = object()
    assert session.tracker.result_stale

    session.merge_loaded({"dose": _image(3.0, size=3)})
    assert list(ws.results) == ["plan", "first", "imported"]
    assert ws.result is plan and session.tracker.result_stale

    loaded = create_result({"physical_dose": _image(4.0)})
    session.merge_loaded({"result": loaded, "pln": object()})
    assert list(ws.results) == ["plan", "first", "imported"]
    assert ws.result is loaded and ws.results["first"] is snapshot
    assert ws.active_result == "plan" and not session.tracker.result_stale

    session.merge_loaded({"ct": object(), "results": {"other": loaded}, "active_result": "other"})
    assert list(ws.results) == ["other"] and ws.active_result == "other"
    session.merge_loaded({"ct": object()})
    assert ws.results == {} and ws.active_result == "current"
    assert not session.state()["has"]["result"]

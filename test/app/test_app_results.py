import pytest

from pyRadPlan.app._results import quantity_meta, result_components, sanitize_tag

TINY_PLAN = {"gantry_angles": [0.0, 180.0], "bixel_width": 10.0, "dose_grid_resolution": [8, 8, 8]}


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
    groups = state["results"]["groups"]
    dose = next(g for g in groups if g["base"] == "physical_dose" and not g["tag"])
    assert dose["label"] == "dose" and dose["meta_label"] == "Dose" and dose["unit"] == "Gy"
    kinds = [c["kind"] for c in dose["components"]]
    assert kinds[0] == "full" and kinds.count("beam") == 2
    assert dose["components"][1]["key"] == "physical_dose_beam 0"
    assert state["results"]["default"] == "physical_dose"
    assert state["results"]["default_tag"] == "photons_30fx"
    revisions = state["revisions"]
    assert {"ct", "physical_dose", "physical_dose_beam 1"} <= set(revisions["volumes"])
    assert revisions["workspace"]["result"] >= 1


def test_snapshot_groups_and_survives_reoptimization(optimized):
    before = optimized.ok("session.state")["result"]["revisions"]["volumes"]
    result = optimized.ok("workflow.snapshot", {"tag": "first try"})["result"]
    assert result["tag"] == "first_try"
    state = result["state"]
    assert state["results"]["tags"] == ["first_try"]
    tagged = [g for g in state["results"]["groups"] if g["tag"] == "first_try"]
    assert tagged and tagged[0]["label"] == "dose (first_try)"
    assert state["revisions"]["volumes"]["physical_dose"] == before["physical_dose"]
    assert optimized.call("workflow.snapshot", {"tag": "beam"})["error"]["type"] == "ValueError"

    state = optimized.ok("workflow.optimize", {"solver": "scipy", "max_iter": 3})["result"]
    session = optimized.manager.get(optimized.session)
    assert "physical_dose_first_try" in session.workspace.result
    assert state["results"]["tags"] == ["first_try"]
    assert state["revisions"]["volumes"]["physical_dose"] != before["physical_dose"]
    assert state["revisions"]["volumes"]["physical_dose_first_try"] == before["physical_dose"]
    catalog = optimized.ok("objectives.catalog")["result"]
    assert {"$result": "physical_dose"} in [c["value"] for c in catalog["reference_images"]]


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
    import numpy as np  # noqa: PLC0415

    volume = np.zeros((2, 2, 2))
    components = result_components({"a": volume, "a_beam": [volume, None, volume], "w": [1, 2]})
    assert list(components) == ["a", "a_beam 0", "a_beam 2"]
    assert sanitize_tag(" my run! ") == "my_run"
    with pytest.raises(ValueError):
        sanitize_tag("scenario")


@pytest.mark.parametrize(
    ("key", "meta", "beam_only"),
    [
        ("rbe_x_dose", ("RBE-weighted Dose", "Gy (RBE)"), ("RBE-weighted Dose", "Gy (RBE)")),
        ("physical_dose_beam", ("Dose", "Gy"), ("Dose", "Gy")),
        ("let_scenario", ("LET", "keV/μm"), ("", "")),
        ("physical_dose_beam_scenario", ("Dose", "Gy"), ("", "")),
        ("physical_dose_saved", ("Quantity", ""), ("", "")),
    ],
)
def test_quantity_meta_resolves_component_keys(key, meta, beam_only):
    """One table serves the viewers (any component key) and the save dialog (per beam)."""
    assert quantity_meta(key) == meta
    assert quantity_meta(key, ("", ""), suffixes=("_beam",)) == beam_only


def test_snapshot_tags_survive_a_workspace_pickle(optimized):
    saved = optimized.call("export.save", {"object": "workspace", "format": "pickle"})
    if not saved["ok"]:
        pytest.skip(f"export.save unavailable: {saved['error']['message']}")
    params = {"filename": saved["result"]["filename"], "data": saved["result"]["data"]}
    state = optimized.ok("data.load_file", params, buffers=saved["buffers"])["result"]
    assert state["results"]["tags"] == ["first_try"]

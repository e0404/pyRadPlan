"""Scenario results of a robust optimization through the protocol (viewer and analysis)."""

import numpy as np
import pytest

ROBUST_PLAN = {
    "gantry_angles": [0.0],
    "bixel_width": 10.0,
    "dose_grid_resolution": [8.0, 8.0, 8.0],
    "scenarios": {
        "model": "rndScen",
        "values": {"n_samples": 2, "include_nominal": True, "seed": 1},
    },
    "solver": {"name": "scipy", "values": {"max_iter": 5}},
}


@pytest.fixture(scope="module")
def robust(client_factory):
    client = client_factory()
    client.session = client.ok("session.create", session=False)["result"]["session"]
    client.ok("data.load_phantom", {"name": "TG119"})
    client.ok("plan.set", ROBUST_PLAN)
    for structure in client.ok("cst.get")["result"]["structures"]:
        for objective in structure["objectives"]:
            client.ok(
                "objectives.update",
                {
                    "structure": structure["name"],
                    "index": objective["index"],
                    "changes": {"robustness": "voxel_worst_case"},
                },
            )
    client.ok("workflow.calc_dose_influence")
    optimized = client.ok("workflow.optimize")["result"]
    yield client, optimized
    client.manager.close_all()


def _dose_group(state):
    return next(
        g for g in state["results"]["groups"] if g["base"] == "physical_dose" and not g["tag"]
    )


def test_robust_result_lists_labelled_scenario_components(robust):
    client, state = robust
    labels = state["results"]["scenario_labels"]
    scenario_keys = [c["key"] for c in _dose_group(state)["components"] if c["kind"] == "scenario"]
    assert len(labels) > 1 and labels[0] == "nominal"
    assert scenario_keys == [f"physical_dose_scenario {i}" for i in range(len(labels))]
    assert all(key not in state["quantities"] for key in scenario_keys)
    assert set(scenario_keys) <= set(state["revisions"]["volumes"])
    assert state["optimization"]["solver"] == "scipy"
    detail = client.ok("cst.get")["result"]["structures"]
    assert {o["robustness"] for s in detail for o in s["objectives"]} == {"voxel_worst_case"}


def test_scenario_volumes_on_the_ct_grid(robust):
    client, state = robust
    full = client.ok("view.volume", {"name": "physical_dose"})
    nominal = client.ok("view.volume", {"name": "physical_dose_scenario 0"})
    shifted = client.ok("view.volume", {"name": "physical_dose_scenario 1"})
    assert shifted["result"]["grid"] == full["result"]["grid"]
    assert shifted["result"]["unit"] == "Gy"
    arrays = [np.frombuffer(r["buffers"][0], dtype=np.float32) for r in (full, nominal, shifted)]
    assert np.allclose(arrays[0], arrays[1], rtol=1e-4, atol=1e-4)
    assert not np.allclose(arrays[0], arrays[2], rtol=1e-3, atol=1e-3)


def test_scenario_dvh_sets_and_qi(robust):
    client, state = robust
    scenario_keys = [c["key"] for c in _dose_group(state)["components"] if c["kind"] == "scenario"]
    keys = ["physical_dose", *scenario_keys]
    params = {"keys": keys, "num_points": 200, "range": "per_structure"}
    sets = client.ok("analysis.dvh", params)["result"]["sets"]
    assert [s["key"] for s in sets] == keys
    assert sets[2]["name"] == "dose, scenario 1" and sets[2]["label"] == "Dose"
    assert all(len(s["curves"]) == len(sets[0]["curves"]) for s in sets)
    qi = client.ok("analysis.qi", {"quantity": "physical_dose_scenario 1"})["result"]
    assert qi["name"] == "dose, scenario 1" and qi["unit"] == "Gy"
    assert {row["name"] for row in qi["rows"]} == {c["name"] for c in sets[0]["curves"]}


def test_snapshot_keeps_scenario_components(robust):
    client, _ = robust
    state = client.ok("workflow.snapshot", {"tag": "robust"})["result"]["state"]
    tagged = next(g for g in state["results"]["groups"] if g["tag"] == "robust")
    kinds = [c["kind"] for c in tagged["components"]]
    assert kinds.count("scenario") == len(state["results"]["scenario_labels"])
    assert tagged["scenario_labels"] == state["results"]["scenario_labels"]
    assert tagged["scenario_labels"] == _dose_group(state)["scenario_labels"]
    assert client.ok("view.volume", {"name": "robust/physical_dose_scenario 1"})["ok"]

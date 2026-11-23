import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

import pyRadPlan.app
from pyRadPlan.app import Session, decode


def _arrays(response):
    return decode(response["result"], response["buffers"])


def test_stages_after_each_step(tg119_pipeline):
    _, states, _ = tg119_pipeline
    expected = {
        "data.load_phantom": ("done", "done", "missing", "missing", "missing"),
        "plan.set": ("done", "done", "missing", "missing", "missing"),
        "workflow.generate_stf": ("done", "done", "done", "missing", "missing"),
        "workflow.calc_dose_influence": ("done", "done", "done", "done", "missing"),
        "workflow.optimize": ("done",) * 5,
    }
    for method, stages in expected.items():
        assert tuple(states[method]["stages"].values()) == stages, method
    final = states["workflow.optimize"]
    assert final["actions"] == {
        "generate_stf": True,
        "calc_dose_influence": True,
        "optimize": True,
        "recalc": True,
        "snapshot": True,
        "load_dij": True,
        "save_pln": True,
        "save_dij": True,
        "save_cst": True,
        "save_result": True,
        "save_workspace": True,
    }
    assert final["quantities"] == ["physical_dose"]
    assert final["status"] == "Optimization result available"
    assert final["optimization"]["solver"] == "scipy"
    assert 0 < final["optimization"]["num_iter"] <= 20
    assert final["optimization"]["objective"] > 0


def test_pipeline_is_fast(tg119_pipeline):
    _, _, timings = tg119_pipeline
    assert sum(timings.values()) < 60, timings


def test_progress_and_state_events(tg119_pipeline):
    client, _, _ = tg119_pipeline
    progress = client.events_of("progress")
    labels = {event["label"] for event in progress}
    assert {"Calculating dose influence", "Optimizing fluence"} <= labels
    for event in progress:
        assert event["fraction"] is None or 0.0 <= event["fraction"] <= 1.0
    dose = [e for e in progress if e["label"] == "Calculating dose influence"]
    fractions = [e["fraction"] for e in dose if e["fraction"] is not None]
    assert fractions[-1] == 1.0
    assert any("Ray" in e["detail"] for e in dose)
    iterations = [e for e in progress if e.get("data", {}).get("iteration") is not None]
    assert iterations and all("objective" in e["data"] for e in iterations)

    busy = [e["state"]["busy"] for e in client.events_of("state")]
    assert "Optimizing fluence" in busy and busy[-1] is None
    assert any(e["level"] == "warning" for e in client.events_of("log"))


def test_ct_volume_and_grid(tg119_pipeline):
    client, states, _ = tg119_pipeline
    response = client.ok("view.volume", {"name": "ct"})
    volume = _arrays(response)
    ct_state = states["workflow.optimize"]["ct"]
    assert response["result"]["data"] == {
        "$array": 0,
        "dtype": "float32",
        "shape": [129, 167, 167],
    }
    assert volume["grid"]["shape"] == ct_state["shape"] == [129, 167, 167]
    assert volume["grid"]["spacing"] == [3.0, 3.0, 2.5]
    assert volume["grid"]["origin"] == ct_state["origin"]
    assert volume["grid"]["direction"] == [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]
    assert volume["unit"] == "HU"

    session = client.manager.get(client.session)
    reference = sitk.GetArrayFromImage(session.workspace.ct.cube_hu).astype(np.float32)
    np.testing.assert_array_equal(volume["data"], reference)
    assert volume["range"] == [float(reference.min()), float(reference.max())]


def test_dose_volume_is_on_the_ct_grid_and_cached(tg119_pipeline):
    client, _, _ = tg119_pipeline
    volume = _arrays(client.ok("view.volume", {"name": "physical_dose"}))
    assert volume["data"].shape == (129, 167, 167)
    assert volume["unit"] == "Gy"
    assert volume["range"][0] >= 0 and volume["range"][1] > 0

    session = client.manager.get(client.session)
    image = session.workspace.result["physical_dose"].values
    np.testing.assert_allclose(volume["data"], sitk.GetArrayFromImage(image), rtol=1e-6)
    assert session.volume("physical_dose")["data"] is session.volume("physical_dose")["data"]

    missing = client.call("view.volume", {"name": "rbe_x_dose"})
    assert missing["error"]["type"] == "KeyError"


def test_structure_masks(tg119_pipeline):
    client, _, _ = tg119_pipeline
    response = client.ok("view.structure_masks", {"names": ["OuterTarget", "Core"]})
    masks = _arrays(response)
    assert [m["name"] for m in masks["masks"]] == ["OuterTarget", "Core"]
    session = client.manager.get(client.session)
    vois = {voi.name: voi for voi in session.workspace.cst.vois}
    for mask in masks["masks"]:
        assert mask["data"].dtype == np.uint8 and mask["data"].shape == (129, 167, 167)
        expected = sitk.GetArrayFromImage(vois[mask["name"]].mask) != 0
        np.testing.assert_array_equal(mask["data"].astype(bool), expected)
        assert mask["color"] == list(vois[mask["name"]].color)
    assert masks["grid"]["shape"] == [129, 167, 167]

    assert len(_arrays(client.ok("view.structure_masks"))["masks"]) == 3
    assert client.call("view.structure_masks", {"names": ["nope"]})["error"]["type"] == "KeyError"


def test_dvh_matches_the_analysis_package(tg119_pipeline):
    from pyRadPlan.analysis import DVH

    client, _, _ = tg119_pipeline
    dvh = _arrays(client.ok("analysis.dvh", {"num_points": 200}))
    assert dvh["quantity"] == "physical_dose" and dvh["unit"] == "Gy"
    assert [c["name"] for c in dvh["curves"]] == ["Core", "OuterTarget", "BODY"]

    session = client.manager.get(client.session)
    dose = sitk.GetArrayFromImage(session.workspace.result["physical_dose"].values)
    dose = dose.astype(np.float32)
    max_value = None
    for curve in dvh["curves"]:
        assert curve["dose"].shape == curve["volume"].shape == (200,)
        assert curve["volume"][0] == pytest.approx(100.0)
        assert np.all(np.diff(curve["volume"]) <= 1e-4)
        max_value = max(max_value or 0.0, curve["stats"]["max"])
    for voi, curve in zip(session.workspace.cst.vois, dvh["curves"]):
        reference = DVH.compute(dose, voi.mask, num_points=200, max_value=max_value * 1.05)
        np.testing.assert_allclose(curve["volume"], reference.cum_volume, atol=1e-3)
        np.testing.assert_allclose(curve["dose"], reference.bins, rtol=1e-5)
        inside = dose[sitk.GetArrayFromImage(voi.mask) != 0]
        assert curve["stats"]["mean"] == pytest.approx(float(inside.mean()), rel=1e-5)


def test_staleness_run_all_and_recalc(client, coarse_plan):
    client.ok("data.load_phantom", {"name": "TG119"})
    client.ok("plan.set", coarse_plan)
    state = client.ok("workflow.run_all", {"solver": "scipy", "max_iter": 10})["result"]
    assert set(state["stages"].values()) == {"done"}
    session = client.manager.get(client.session)
    dij, optimized = session.workspace.dij, session.workspace.result
    weights = optimized.fluence
    assert optimized.stored_beam_quantities == ["physical_dose"]

    state = client.ok("plan.set", {"num_of_fractions": 5})["result"]
    assert state["stages"]["stf"] == "stale" and state["stages"]["dij"] == "stale"
    assert state["stages"]["optimization"] == "stale"
    assert state["actions"]["optimize"] and "outdated" in state["status"]

    state = client.ok("workflow.snapshot", {"tag": "before"})["result"]["state"]
    assert state["stages"]["optimization"] == "stale"
    snapshot = session.workspace.results["before"]

    state = client.ok("workflow.recalc")["result"]
    assert state["stages"]["optimization"] == "done" and state["stages"]["dij"] == "stale"
    recalculated = session.workspace.result
    assert recalculated is not optimized
    np.testing.assert_array_equal(recalculated.fluence, weights)
    assert list(session.workspace.results) == ["current", "before"]
    assert session.workspace.results["before"] is snapshot
    assert snapshot["physical_dose"].values is optimized["physical_dose"].values
    assert recalculated["physical_dose"].values is not optimized["physical_dose"].values

    state = client.ok("workflow.run_all", {"solver": "scipy", "max_iter": 10})["result"]
    assert set(state["stages"].values()) == {"done"}
    assert session.workspace.dij is not dij
    assert session.workspace.result is not recalculated
    assert session.workspace.results["before"] is snapshot


def test_prepared_operations_touch_the_workspace_only_on_commit(coarse_plan):
    session = Session()
    session.load_phantom("TG119")
    session.set_plan(coarse_plan)
    seen = []
    session.workspace.subscribe(seen.append)
    operation = session.prepare("generate_stf")
    stf = operation.compute()
    assert session.workspace.stf is None and not seen
    operation.commit(stf)
    assert session.workspace.stf is stf and seen == [["stf"]]
    with pytest.raises(ValueError, match="Unknown operation"):
        session.prepare("explode")
    session.close()
    assert not session.workspace.present()


def test_importing_the_app_layer_does_not_import_qt():
    check = (
        "import sys, pyRadPlan.app\n"
        "loaded = [m for m in sys.modules if m.startswith(('PySide6', 'pyRadPlan.gui'))]\n"
        "assert not loaded, loaded\n"
    )
    subprocess.run([sys.executable, "-c", check], check=True)


def test_the_app_layer_uses_no_qt_threads_or_processes():
    forbidden = ("PySide6", "multiprocessing", "concurrent", "pyRadPlan.gui")
    starters = {"Thread", "Timer", "start_new_thread"}
    for path in Path(pyRadPlan.app.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                if isinstance(node, ast.Attribute):
                    assert node.attr not in starters, f"{path.name} starts a thread"
                continue
            for name in names:
                assert not name.startswith(forbidden), f"{path.name} imports {name}"

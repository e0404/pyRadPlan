import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.app import Session
from pyRadPlan.app._views import resolve_structure
from pyRadPlan.cst import recreate_voi


@pytest.fixture
def duplicate_session(client):
    client.ok("data.load_phantom", {"name": "TG119"})
    session = client.manager.get(client.session)
    ws = session.workspace
    cst = ws.cst
    vois = list(cst.vois)
    vois[1] = recreate_voi(vois[1], name=vois[0].name)
    cst.vois = vois
    ws.cst = cst
    dose = sitk.Cast(ws.ct.cube_hu, sitk.sitkFloat32)
    ramp = sitk.GetImageFromArray(
        np.arange(sitk.GetArrayViewFromImage(dose).size, dtype=np.float32).reshape(
            sitk.GetArrayViewFromImage(dose).shape
        )
    )
    ramp.CopyInformation(dose)
    ws.result = {"physical_dose": ramp}
    return client, session


def _array(response, descriptor):
    buffer = response["buffers"][descriptor["$array"]]
    return np.frombuffer(buffer, dtype=descriptor["dtype"]).reshape(descriptor["shape"])


def test_masks_of_same_named_structures_differ(duplicate_session):
    client, session = duplicate_session
    vois = session.workspace.cst.vois
    assert vois[0].name == vois[1].name
    response = client.ok("view.structure_masks", {"names": [0, 1]})
    first, second = response["result"]["masks"]
    assert (first["index"], second["index"]) == (0, 1)
    expected = [session.views.mask_array(i) for i in (0, 1)]
    assert not np.array_equal(expected[0], expected[1])
    np.testing.assert_array_equal(_array(response, first["data"]), expected[0])
    np.testing.assert_array_equal(_array(response, second["data"]), expected[1])
    bits = client.ok("view.structure_masks", {"encoding": "bits"})["result"]["masks"]
    assert [m["num_voxels"] for m in bits[:2]] == [int(m.sum()) for m in expected]
    assert bits[0]["bit"] != bits[1]["bit"]


def test_state_and_cst_get_report_each_structure(duplicate_session):
    client, session = duplicate_session
    vois = session.workspace.cst.vois
    state = client.ok("session.state")["result"]
    counts = [s["num_voxels"] for s in state["structures"]]
    assert counts[0] != counts[1]
    assert [s["index"] for s in state["structures"]] == list(range(len(vois)))
    details = client.ok("cst.get", {"names": [1]})["result"]["structures"]
    assert [d["index"] for d in details] == [1]
    assert details[0]["num_voxels"] == counts[1]


def test_dvh_and_qi_per_position(duplicate_session):
    client, _ = duplicate_session
    result = client.ok("analysis.dvh", {"names": [0, 1], "quantity": "physical_dose"})["result"]
    first, second = result["curves"]
    assert (first["index"], second["index"]) == (0, 1)
    assert first["stats"]["mean"] != second["stats"]["mean"]
    rows = client.ok("analysis.qi", {"quantity": "physical_dose", "names": [0, 1]})["result"][
        "rows"
    ]
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[0]["values"] != rows[1]["values"]


def test_ambiguous_names_are_refused(duplicate_session):
    client, session = duplicate_session
    name = session.workspace.cst.vois[0].name
    for method, params in (
        ("view.structure_masks", {"names": [name]}),
        ("cst.get", {"names": [name]}),
        ("cst.update_structure", {"name": name, "changes": {"visible": False}}),
    ):
        error = client.call(method, params)["error"]
        assert error["type"] == "KeyError" and "ambiguous" in error["message"]
    assert client.call("cst.get", {"names": [99]})["error"]["type"] == "KeyError"
    assert client.call("cst.get", {"names": [True]})["error"]["type"] == "TypeError"


def test_update_structure_by_position_changes_only_that_one(duplicate_session):
    client, session = duplicate_session
    client.ok("cst.update_structure", {"name": 1, "changes": {"color": [1, 2, 3]}})
    vois = session.workspace.cst.vois
    assert list(vois[1].color) == [1, 2, 3]
    assert list(vois[0].color) != [1, 2, 3]


def test_objectives_by_position(duplicate_session):
    client, session = duplicate_session
    before = [len(v.objectives or []) for v in session.workspace.cst.vois]
    client.ok("objectives.add", {"structure": 1})
    after = [len(v.objectives or []) for v in session.workspace.cst.vois]
    assert after[1] == before[1] + 1 and after[0] == before[0]


def test_resolve_structure_unit():
    class V:
        def __init__(self, name):
            self.name = name

    vois = [V("a"), V("b"), V("a")]
    assert resolve_structure(vois, "b") == 1
    assert resolve_structure(vois, 2) == 2
    with pytest.raises(KeyError):
        resolve_structure(vois, "a")
    with pytest.raises(KeyError):
        resolve_structure(vois, "c")

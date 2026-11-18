import pytest


@pytest.fixture
def loaded(client):
    client.ok("data.load_phantom", {"name": "TG119"})
    return client


def _objectives(client, name):
    structures = client.ok("cst.get", {"names": [name]})["result"]["structures"]
    return structures[0]["objectives"]


def test_catalog(loaded):
    catalog = loaded.ok("objectives.catalog")["result"]
    by_name = {o["name"]: o for o in catalog["objectives"]}
    assert {"Squared Deviation", "Squared Overdosing", "Min DVH", "Variance"} <= set(by_name)
    min_dvh = {p["name"]: p for p in by_name["Min DVH"]["parameters"]}
    assert min_dvh["v_min"]["unit"] == "%" and min_dvh["v_min"]["max"] == 100
    assert min_dvh["d"]["unit_from"] == "quantity" and min_dvh["d"]["min"] == 0
    mean = {p["name"]: p for p in by_name["Mean Dose"]["parameters"]}
    assert mean["f_diff"]["type"] == "choice"
    assert {p["name"]: p for p in by_name["Squared Mimicking"]["parameters"]}["d_ref"][
        "type"
    ] == "image_ref"
    assert [c["value"] for c in by_name["Variance"]["robustness"]] == ["scenario_stack"]
    assert by_name["Variance"]["scenario_aware"] is True
    assert "physical_dose" in [q["value"] for q in catalog["quantities"]]
    assert catalog["default_by_type"]["TARGET"] == "Squared Deviation"


def test_add_update_change_type_remove(loaded):
    before = _objectives(loaded, "Core")
    added = loaded.ok("objectives.add", {"structure": "Core"})["result"]["structure"]
    assert len(added["objectives"]) == len(before) + 1
    new = added["objectives"][-1]
    assert new["name"] == "Squared Overdosing"
    index = new["index"]

    changes = {"priority": 7.0, "parameters": {"d_max": 12.5}}
    updated = loaded.ok(
        "objectives.update", {"structure": "Core", "index": index, "changes": changes}
    )["result"]["structure"]["objectives"][index]
    assert updated["priority"] == 7.0 and updated["parameters"]["d_max"] == 12.5

    changed = loaded.ok(
        "objectives.change_type", {"structure": "Core", "index": index, "name": "Max DVH"}
    )["result"]["structure"]["objectives"][index]
    assert changed["name"] == "Max DVH" and changed["priority"] == 7.0
    assert set(changed["parameters"]) == {"d", "v_max"}

    bad = {"structure": "Core", "index": index, "changes": {"parameters": {"v_max": 150}}}
    error = loaded.call("objectives.update", bad)["error"]
    assert error["type"] == "ValidationError" and error["details"][0]["field"].endswith("v_max")
    assert _objectives(loaded, "Core")[index]["parameters"]["v_max"] == 50.0

    loaded.ok("objectives.remove", {"structure": "Core", "index": index})
    assert len(_objectives(loaded, "Core")) == len(before)
    assert loaded.call("objectives.remove", {"structure": "Core", "index": 99})["error"]


def test_set_quantity_all(loaded):
    loaded.ok("objectives.set_quantity_all", {"quantity": "rbe_x_dose"})
    structures = loaded.ok("cst.get")["result"]["structures"]
    assert {o["quantity"] for s in structures for o in s["objectives"]} == {"rbe_x_dose"}
    assert loaded.call("objectives.set_quantity_all", {"quantity": "nope"})["error"]
    structures = loaded.ok("cst.get")["result"]["structures"]
    assert {o["quantity"] for s in structures for o in s["objectives"]} == {"rbe_x_dose"}


def _twin_session():
    """A TG119 session whose second structure is named like the first."""
    from pyRadPlan.app import Session
    from pyRadPlan.cst import recreate_voi

    session = Session(auto_plan=False)
    session.load_phantom("TG119")
    cst = session.workspace.cst
    vois = list(cst.vois)
    vois[1] = recreate_voi(vois[1], name=vois[0].name)
    cst.vois = vois
    session.workspace.cst = cst
    return session, vois[0].name


def test_a_position_tells_apart_structures_of_the_same_name():
    from pyRadPlan.app._objectives import objectives_of

    session, name = _twin_session()
    first, second = session.workspace.cst.vois[:2]
    before = (len(objectives_of(first)), len(objectives_of(second)))
    assert session.objectives_edit("add", name, detail=False, voi_index=1) is None
    first, second = session.workspace.cst.vois[:2]
    assert (len(objectives_of(first)), len(objectives_of(second))) == (before[0], before[1] + 1)
    index = len(objectives_of(second)) - 1
    session.objectives_edit(
        "update", name, index=index, changes={"priority": 7.0}, voi_index=1, detail=False
    )
    assert objectives_of(session.workspace.cst.vois[1])[index].priority == 7.0
    with pytest.raises(KeyError):
        session.objectives_edit("add", "no such name", voi_index=1)
    with pytest.raises(ValueError, match="Unknown objectives operation"):
        session.objectives_edit("set_quantity_all", name)


def test_empty_and_invalid_entries_are_skipped():
    from pyRadPlan.app._objectives import objectives_of
    from pyRadPlan.optimization.objectives import get_objective

    class _Voi:
        objectives = [None, [], (), "no such objective", get_objective("Squared Overdosing")]

    assert [o.name for o in objectives_of(_Voi())] == ["Squared Overdosing"]


def test_reference_values_pair_cubes_with_the_ct_grid():
    import numpy as np
    import SimpleITK as sitk

    from pyRadPlan.app._objectives import reference_images, reference_values

    class _Ct:
        grid = "grid"

    image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    cube = np.zeros((2, 2, 2))
    result = {"physical_dose": image, "let": cube, "w": np.ones(3), "physical_dose_beam": [image]}
    values = reference_values(result, _Ct())
    assert [(key, label) for key, label, _ in values] == [
        ("physical_dose", "dose"),
        ("let", "LETd"),
    ]
    assert values[0][2] is image and values[1][2][0] is cube and values[1][2][1] == "grid"
    assert [key for key, *_ in reference_values(result, None)] == ["physical_dose"]
    assert reference_images(result, None) == [
        {"value": {"$result": "physical_dose"}, "label": "dose"}
    ]

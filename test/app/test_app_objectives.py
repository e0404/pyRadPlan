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

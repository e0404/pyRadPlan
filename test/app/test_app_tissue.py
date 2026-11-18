import math

import pytest


@pytest.fixture
def loaded(client):
    client.ok("data.load_phantom", {"name": "TG119"})
    return client


def test_cst_get_details(loaded):
    structures = loaded.ok("cst.get")["result"]["structures"]
    assert [s["name"] for s in structures] == ["Core", "OuterTarget", "BODY"]
    target = structures[1]
    assert target["type"] == "TARGET" and target["num_voxels"] > 0
    assert target["tissue"]["model"] == "Generic"
    assert set(target["tissue"]["values"]) == {"alpha_x", "beta_x"}
    assert target["objectives"][0]["name"]
    assert len(target["center_of_mass"]) == 3 and target["tissue_summary"]
    state = loaded.ok("session.state")["result"]
    assert state["structures"][1]["num_voxels"] == target["num_voxels"]
    assert state["structures"][1]["num_objectives"] == len(target["objectives"])
    error = loaded.call("cst.get", {"names": ["nope"]})["error"]
    assert error["type"] == "KeyError"


def test_update_structure_colour_type_priority(loaded):
    before = loaded.ok("session.state")["result"]["revisions"]
    result = loaded.ok(
        "cst.update_structure",
        {"name": "Core", "changes": {"color": [1, 2, 3], "overlap_priority": 5, "visible": False}},
    )["result"]
    assert result["structure"]["color"] == [1, 2, 3]
    assert result["structure"]["overlap_priority"] == 5
    assert result["structure"]["visible"] is False
    after = result["state"]["revisions"]
    assert after["workspace"]["cst"] == before["workspace"]["cst"] + 1
    assert after["masks"] == before["masks"]

    result = loaded.ok("cst.update_structure", {"name": "Core", "changes": {"type": "helper"}})
    assert result["result"]["structure"]["type"] == "HELPER"
    assert loaded.call("cst.update_structure", {"name": "Core", "changes": {"x": 1}})["error"]


def test_tissue_schema_and_mkm_derivation(loaded):
    schema = loaded.ok("tissue.schema")["result"]
    models = {m["label"]: m for m in schema["models"]}
    assert set(models) == {"Generic", "LEM", "MKM", "mMKM"}
    mkm = {f["name"]: f for f in models["MKM"]["fields"]}
    assert {"r_domain", "alpha_0", "beta_0", "reference_radiation", "alpha_x"} <= set(mkm)
    assert mkm["r_domain"]["nullable"] and mkm["reference_radiation"]["type"] == "choice"
    assert schema["reference_radiations"]

    draft = {
        "model": "MKM",
        "values": {"r_nucleus": 4.0, "r_domain": 0.3, "alpha_0": 0.2, "beta_0": 0.05},
        "sources": {},
    }
    rows = [{"structure": "Core", "tissue": draft}]
    evaluated = loaded.ok("tissue.evaluate", {"rows": rows, "op": "derive"})["result"]
    tissue = evaluated["rows"][0]["tissue"]
    assert tissue["sources"]["alpha_x"] == "estimated"
    assert tissue["values"]["alpha_x"] > 0.2 and math.isclose(tissue["values"]["beta_x"], 0.05)
    assert evaluated["can_derive"] is True and evaluated["notes"]


def test_tissue_apply_writes_and_gates_errors(loaded, monkeypatch):
    rows = [{"structure": "Core", "tissue": {"model": "Generic", "values": {"alpha_x": 0.3,
                                                                        "beta_x": 0.03}}}]  # fmt: skip
    session = loaded.manager.get(loaded.session)
    from pyRadPlan.app import _tissue

    monkeypatch.setattr(_tissue, "check_errors", lambda *a, **k: [_Issue()])
    error = loaded.call("tissue.apply", {"rows": rows})["error"]
    assert error["code"] == "tissue_errors" and error["details"][0]["kind"] == "base_data"
    assert session.workspace.cst.vois[0].alpha_x != 0.3

    loaded.ok("tissue.apply", {"rows": rows, "apply_anyway": True})
    voi = session.workspace.cst.vois[0]
    assert voi.alpha_x == pytest.approx(0.3) and voi.beta_x == pytest.approx(0.03)
    assert voi.tissue_properties.alpha_x == pytest.approx(0.3)


class _Issue:
    kind = "base_data"
    structure = "Core"
    is_error = True
    suggestion = None
    message = "no base data"

    def __str__(self):
        return self.message


def test_tissue_calls_use_the_unapplied_plan_form(loaded):
    """``draft`` evaluates against the plan form's selection, as ``plan.tissue_check`` does."""
    draft = {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "LEM"}
    schema = loaded.ok("tissue.schema", {"draft": draft})["result"]
    assert "LEM" in schema["option_label"]
    tissue = loaded.ok("cst.get", {"names": ["Core"]})["result"]["structures"][0]["tissue"]
    rows = [{"structure": "Core", "tissue": tissue}]
    with_draft = loaded.ok("tissue.evaluate", {"rows": rows, "draft": draft})["result"]["rows"][0]
    applied = loaded.ok("tissue.evaluate", {"rows": rows})["result"]["rows"][0]
    # The photon plan has no RBE base data to compare with; the carbon LEM draft has.
    assert applied["comparison"] is None
    assert "LEM" in with_draft["comparison"]["closest_class"]

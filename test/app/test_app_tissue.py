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


def test_tissue_evaluate_names_the_pick_a_tissue_corresponds_to(loaded):
    draft = {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "LEM"}
    schema = loaded.ok("tissue.schema", {"draft": draft})["result"]
    tissue = loaded.ok("cst.get", {"names": ["Core"]})["result"]["structures"][0]["tissue"]
    rows = [{"structure": "Core", "tissue": tissue}]

    def evaluate(rows, **params):
        result = loaded.ok("tissue.evaluate", {"rows": rows, "draft": draft, **params})
        return result["result"]["rows"][0]

    # The default tissue (0.1 / 0.05) matches the LEM IV class of those reference parameters
    assert evaluate(rows)["comparison"]["matched"]
    assert evaluate(rows)["pick"] == "class:0"
    other = {"model": "Generic", "values": {"alpha_x": 0.3, "beta_x": 0.05}}
    assert evaluate([{"structure": "Core", "tissue": other}])["pick"] is None

    class_id = next(p["id"] for p in schema["picks"] if p["kind"] == "base_data")
    applied = evaluate(rows, op="apply_pick", pick=class_id)
    assert applied["pick"] == class_id
    edited = {**applied["tissue"], "values": {**applied["tissue"]["values"], "alpha_x": 0.123}}
    assert evaluate([{"structure": "Core", "tissue": edited}])["pick"] is None

    preset_id = next(p["id"] for p in schema["picks"] if p["kind"] == "preset")
    applied = evaluate(rows, op="apply_pick", pick=preset_id)
    assert applied["pick"] == preset_id
    # A tissue draft carries no cell line, so the preset is found again by its parameters
    again = evaluate([{"structure": "Core", "tissue": applied["tissue"]}])
    assert again["pick"] == preset_id


PICK_CONTEXTS = [
    {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "LEM"},
    {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "alpha_beta_kernels"},
    {"radiation_mode": "protons", "machine": "Generic", "bio_model": "MCN"},
    {"radiation_mode": "oxygen", "machine": "Generic", "bio_model": "zstar_kernels"},
]


@pytest.mark.parametrize("draft", PICK_CONTEXTS, ids=lambda d: d["bio_model"])
def test_every_pick_round_trips_through_its_draft(loaded, draft):
    """Applying a pick and evaluating the returned draft again names the same pick."""
    schema = loaded.ok("tissue.schema", {"draft": draft})["result"]
    assert schema["picks"]
    tissue = loaded.ok("cst.get", {"names": ["Core"]})["result"]["structures"][0]["tissue"]
    rows = [{"structure": "Core", "tissue": tissue}]
    ambiguous = []
    for pick in schema["picks"]:
        applied = loaded.ok(
            "tissue.evaluate",
            {"rows": rows, "draft": draft, "op": "apply_pick", "pick": pick["id"]},
        )["result"]["rows"][0]
        assert applied["pick"] == pick["id"]
        assert applied["tissue"]["cell_line"] == pick["cell_line"] or pick["kind"] == "base_data"
        again = loaded.ok(
            "tissue.evaluate",
            {"rows": [{"structure": "Core", "tissue": applied["tissue"]}], "draft": draft},
        )["result"]["rows"][0]
        if again["pick"] != pick["id"]:
            # Presets whose applied tissues are identical (e.g. LEM I and LEM II presets of a
            # cell line adopted by MKM base data keep only the same reference parameters)
            # cannot be told apart by the tissue; the pick named must give the same tissue.
            other = loaded.ok(
                "tissue.evaluate",
                {"rows": rows, "draft": draft, "op": "apply_pick", "pick": again["pick"]},
            )["result"]["rows"][0]
            assert other["tissue"] == applied["tissue"], pick["label"]
            ambiguous.append(pick["id"])
    assert len(ambiguous) < len(schema["picks"]) / 2


def test_identity_survives_writing_and_reopening(loaded):
    draft = {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "LEM"}
    schema = loaded.ok("tissue.schema", {"draft": draft})["result"]
    preset = next(p for p in schema["picks"] if p["kind"] == "preset" and p["cell_line"])
    rows = [{"structure": "Core", "tissue": {"model": "Generic", "values": {}}}]
    applied = loaded.ok(
        "tissue.evaluate", {"rows": rows, "draft": draft, "op": "apply_pick", "pick": preset["id"]}
    )["result"]["rows"][0]["tissue"]
    assert applied["cell_line"] == preset["cell_line"] and applied["cell_line_source"]
    session = loaded.manager.get(loaded.session)

    for write in ("update_structure", "tissue.apply"):
        loaded.ok(
            "cst.update_structure",
            {"name": "Core", "changes": {"tissue": {"model": "Generic", "values": {}}}},
        )
        if write == "update_structure":
            loaded.ok("cst.update_structure", {"name": "Core", "changes": {"tissue": applied}})
        else:
            loaded.ok("tissue.apply", {"rows": [{"structure": "Core", "tissue": applied}]})
        stored = session.workspace.cst.vois[0].tissue_properties
        assert stored.cell_line == preset["cell_line"]
        assert stored.cell_line_source == applied["cell_line_source"]
        reopened = loaded.ok("cst.get", {"names": ["Core"]})["result"]["structures"][0]["tissue"]
        assert reopened["cell_line"] == preset["cell_line"]
        row = loaded.ok(
            "tissue.evaluate",
            {"rows": [{"structure": "Core", "tissue": reopened}], "draft": draft},
        )["result"]["rows"][0]
        assert row["pick"] == preset["id"]


def test_identity_is_dropped_when_parameters_change(loaded):
    draft = {"radiation_mode": "carbon", "machine": "Generic", "bio_model": "LEM"}
    schema = loaded.ok("tissue.schema", {"draft": draft})["result"]
    preset = next(p for p in schema["picks"] if p["kind"] == "preset" and p["cell_line"])
    rows = [{"structure": "Core", "tissue": {"model": "Generic", "values": {}}}]
    applied = loaded.ok(
        "tissue.evaluate", {"rows": rows, "draft": draft, "op": "apply_pick", "pick": preset["id"]}
    )["result"]["rows"][0]["tissue"]
    field = next(name for name, value in applied["values"].items() if isinstance(value, float))
    edited = {**applied, "values": {**applied["values"], field: applied["values"][field] + 0.01}}
    row = loaded.ok(
        "tissue.evaluate",
        {
            "rows": [{"structure": "Core", "tissue": edited}],
            "draft": draft,
            "edited": {"structure": "Core", "field": field},
        },
    )["result"]["rows"][0]
    assert row["tissue"]["cell_line"] is None and row["tissue"]["cell_line_source"] is None


def test_draft_without_identity_keeps_the_cell_line_of_the_structure(loaded):
    session = loaded.manager.get(loaded.session)
    tissue = session.workspace.cst.vois[0].tissue_properties
    legacy = {"model": "Generic", "values": {"alpha_x": tissue.alpha_x, "beta_x": tissue.beta_x}}
    loaded.ok("tissue.apply", {"rows": [{"structure": "Core", "tissue": legacy}]})
    assert session.workspace.cst.vois[0].tissue_properties.cell_line == tissue.cell_line


def test_display_edits_outdate_nothing_and_are_accepted_while_busy():
    from pyRadPlan.app import Operation, PreconditionError, Session

    session = Session(auto_plan=False)
    session.load_phantom("TG119")
    session.workspace.result = {"physical_dose": object()}
    name = session.workspace.cst.vois[0].name
    events = []
    session.add_listener(events.append)
    before = session.revisions.as_dict()["workspace"]["cst"]

    detail = session.update_structure(name, {"color": [9, 8, 7], "visible": False})

    assert detail["color"] == [9, 8, 7] and detail["visible"] is False
    assert session.stages()["optimization"] == "done"
    assert session.revisions.as_dict()["workspace"]["cst"] == before + 1
    assert events[-1]["event"] == "state"
    assert events[-1]["state"]["structures"][0]["color"] == [9, 8, 7]

    operation = Operation("calc", "Calculating", lambda: None, lambda value: None)
    session.begin(operation)
    assert session.update_structure(name, {"color": [1, 1, 1]}, detail=False) is None
    with pytest.raises(PreconditionError) as error:
        session.update_structure(name, {"type": "OAR"})
    assert error.value.code == "busy"
    applied = []
    session.call_when_idle(lambda: applied.append(session.busy))
    assert applied == []
    session.abort(operation)
    assert applied == [False]
    assert tuple(session.workspace.cst.vois[0].color) == (1, 1, 1)


def test_tissue_apply_takes_tissue_properties_and_positions():
    from pyRadPlan.app import Session
    from pyRadPlan.biology.tissues import TissueProperties
    from pyRadPlan.cst import recreate_voi

    session = Session(auto_plan=False)
    session.load_phantom("TG119")
    cst = session.workspace.cst
    vois = list(cst.vois)
    vois[1] = recreate_voi(vois[1], name=vois[0].name)
    cst.vois = vois
    session.workspace.cst = cst
    tissue = TissueProperties(alpha_x=0.3, beta_x=0.03)

    session.tissue_apply(
        [{"structure": vois[0].name, "voi_index": 1, "tissue": tissue}], apply_anyway=True
    )

    stored = session.workspace.cst.vois
    assert stored[1].tissue_properties == tissue and stored[1].alpha_x == 0.3
    assert stored[0].tissue_properties == vois[0].tissue_properties
    with pytest.raises(KeyError):
        session.tissue_apply([{"structure": "nope", "voi_index": 1, "tissue": tissue}])

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

    from pyRadPlan.result import create_result

    image = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    let = sitk.Image([2, 2, 2], sitk.sitkFloat32)
    result = create_result(
        {"physical_dose": image, "let": let, "w": np.ones(3), "physical_dose_beam": [image]}
    )
    measured = sitk.Image([3, 3, 3], sitk.sitkFloat32)
    entries = [("", result), ("measured", create_result({"physical_dose": measured}))]
    values = reference_values(entries)
    assert [(key, label) for key, label, _ in values] == [
        ("physical_dose", "dose"),
        ("let", "LETd"),
        ("measured/physical_dose", "dose (measured)"),
    ]
    assert values[0][2] is image and values[1][2] is let and values[2][2] is measured
    assert reference_values(None) == []
    assert reference_images(result) == [
        {"value": {"$result": "physical_dose"}, "label": "dose"},
        {"value": {"$result": "let"}, "label": "LETd"},
    ]


def _scripted_objective(monkeypatch):
    """Register an objective only code can create: it needs a ``handle`` no editor has."""
    from typing import Any, ClassVar, Literal

    from pydantic import Field, model_validator

    from pyRadPlan.optimization.objectives import Objective
    from pyRadPlan.optimization.objectives._factory import OBJECTIVES

    class Scripted(Objective):
        name: Literal["Scripted"] = "Scripted"
        selectable: ClassVar[bool] = False
        loads: ClassVar[int] = 0
        model: str
        handle: Any = Field(default=None, exclude=True)

        @model_validator(mode="before")
        @classmethod
        def _load(cls, data: Any) -> Any:
            if isinstance(data, dict) and data.get("handle") is None:
                Scripted.loads += 1
                data = {**data, "handle": object()}
            return data

        def compute_objective(self, values):
            return values.sum()

        def compute_gradient(self, values):
            return values

    monkeypatch.setitem(OBJECTIVES, "Scripted", Scripted)
    return Scripted


def _script_objective(client, structure, objective):
    session = client.manager.get(client.session)
    cst = session.workspace.cst
    voi = next(voi for voi in cst.vois if voi.name == structure)
    voi.objectives = [*voi.objectives, objective]
    session.workspace.cst = cst
    return session, len(voi.objectives) - 1


def test_objectives_created_in_code_are_not_offered(loaded, monkeypatch):
    from pyRadPlan.app import _objectives as editor
    from pyRadPlan.optimization.objectives import get_available_objectives

    scripted = _scripted_objective(monkeypatch)
    hidden = set(get_available_objectives()) - set(get_available_objectives(selectable_only=True))
    assert {"Scripted", "Outcome Model", "Outcome LR", "Outcome scikit-learn"} <= hidden

    names = {o["name"] for o in loaded.ok("objectives.catalog")["result"]["objectives"]}
    assert names == set(get_available_objectives(selectable_only=True))

    monkeypatch.setitem(editor.DEFAULT_BY_TYPE, "*", "Scripted")
    added = loaded.ok("objectives.add", {"structure": "Core"})["result"]["structure"]
    new = added["objectives"][-1]
    assert new["name"] in names

    for name in ("Scripted", "Outcome LR"):
        request = {"structure": "Core", "index": new["index"], "name": name}
        error = loaded.call("objectives.change_type", request)["error"]
        assert error["type"] == "ValidationError" and error["details"][0]["field"] == "name"
        assert "cannot be selected" in error["message"]
    assert _objectives(loaded, "Core")[new["index"]]["name"] == new["name"]
    assert scripted.loads == 0


def test_an_objective_created_in_code_is_shown_and_edited_without_reloading(loaded, monkeypatch):
    from pyRadPlan.app._objectives import objectives_of

    scripted = _scripted_objective(monkeypatch)
    objective = scripted(model="some model")
    assert scripted.loads == 1
    session, index = _script_objective(loaded, "Core", objective)

    shown = _objectives(loaded, "Core")[index]
    assert shown["name"] == "Scripted" and shown["parameters"] == {}

    changes = {"priority": 3.0, "quantity": "rbe_x_dose", "robustness": "expected_value"}
    updated = loaded.ok(
        "objectives.update", {"structure": "Core", "index": index, "changes": changes}
    )["result"]["structure"]["objectives"][index]
    assert {key: updated[key] for key in changes} == changes
    loaded.ok("objectives.set_quantity_all", {"quantity": "physical_dose"})
    stored = objectives_of(next(v for v in session.workspace.cst.vois if v.name == "Core"))[index]
    assert isinstance(stored, scripted) and stored.handle is objective.handle
    assert (stored.priority, stored.quantity) == (3.0, "physical_dose")
    assert scripted.loads == 1

    request = {"structure": "Core", "index": index, "name": "Max DVH"}
    error = loaded.call("objectives.change_type", request)["error"]
    assert error["type"] == "ValidationError" and "cannot be changed" in error["message"]
    assert _objectives(loaded, "Core")[index]["name"] == "Scripted"

    loaded.ok("objectives.remove", {"structure": "Core", "index": index})
    assert "Scripted" not in [o["name"] for o in _objectives(loaded, "Core")]


def test_a_preloaded_outcome_objective_keeps_its_model_when_edited(loaded):
    import numpy as np

    linear_model = pytest.importorskip("sklearn.linear_model")

    from pyRadPlan.ai.modelhub import OutcomeFeature, OutcomeModelSpec
    from pyRadPlan.app._objectives import objectives_of
    from pyRadPlan.cst import validate_cst
    from pyRadPlan.optimization.objectives import OutcomeLR

    estimator = linear_model.LogisticRegression()
    estimator.coef_ = np.asarray([[0.1]])
    estimator.intercept_ = np.asarray([0.05])
    estimator.classes_ = np.asarray([0, 1])
    feature = OutcomeFeature(name="core_DoseMean", segment="Core", function="Dose Mean")
    spec = OutcomeModelSpec(label="test", outcome_type="NTCP", features=[feature])
    session, index = _script_objective(loaded, "Core", OutcomeLR(model=(estimator, spec)))

    assert _objectives(loaded, "Core")[index]["name"] == "Outcome LR"
    request = {"structure": "Core", "index": index, "changes": {"priority": 4.0}}
    loaded.ok("objectives.update", request)
    core = next(v for v in validate_cst(session.workspace.cst).vois if v.name == "Core")
    stored = objectives_of(core)[index]
    assert stored.priority == 4.0 and stored.model == "<preloaded>"
    assert stored.estimator is estimator and stored.spec is spec
    assert stored.coef.tolist() == [0.1] and stored.intercept == 0.05

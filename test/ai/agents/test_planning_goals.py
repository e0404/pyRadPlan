"""Goal and constraint-table validation, goal files, evaluation of goals without a value."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.ai.agents._planning import _geometry as geo
from pyRadPlan.ai.agents._planning._constraints import (
    ConstraintTable,
    Goal,
    Protocol,
    _scale_goal,
    load_constraint_table,
    load_goals,
    save_goals,
    site_goals,
)
from pyRadPlan.ai.agents._planning._evaluation import (
    PlanRecord,
    evaluate_protocol,
    plan_score,
    select_distinct_plans,
    voxel_metric,
)

# ---- evaluation ----------------------------------------------------------------------


def _qis(**voxels):
    keys = ("mean", "max", "D95")
    return {
        voi: {k: SimpleNamespace(value=voxel_metric(np.asarray(v, dtype=float), k)) for k in keys}
        for voi, v in voxels.items()
    }


def test_goal_on_an_empty_structure_is_unresolved_and_keeps_the_score_finite():
    protocol = Protocol(
        goals=[
            Goal(voi="PTV", metric="D95", op=">=", value=1.0),
            Goal(voi="Empty", metric="max", op="<=", value=1.0),
            Goal(voi="Empty", metric="mean", op="<=", value=0.5, hard=False),
            Goal(voi="Missing", metric="max", op="<=", value=1.0),
        ]
    )
    report = evaluate_protocol(protocol, _qis(PTV=[1.2, 1.2, 1.2], Empty=[]))
    assert [r["goal"] for r in report["goals"]] == ["PTV D95 >= 1"]
    assert report["score"] == pytest.approx(20.0) and report["all_hard_passed"]
    assert report["empty_structures"] == ["Empty"]
    assert report["unresolved_structures"] == ["Missing"]
    assert report["unresolved_goals"] == [
        "Empty max <= 1",
        "Empty mean <= 0.5",
        "Missing max <= 1",
    ]
    assert "empty_structures" not in evaluate_protocol(protocol, _qis(PTV=[1.2], Empty=[0.1]))


def test_plan_score_is_finite_for_a_margin_that_is_not():
    rows = [{"hard": True, "rel_margin": math.nan}, {"hard": False, "rel_margin": 0.2}]
    assert plan_score(rows) == pytest.approx(100 * (3 * -1.0 + 0.2) / 4)


def _plan(index, score, margin, all_hard_passed=True):
    goals = [{"goal": "PTV D95 >= 1", "rel_margin": margin}]
    evaluation = {"all_hard_passed": all_hard_passed, "score": score, "goals": goals}
    return PlanRecord(index=index, evaluation=evaluation)


@pytest.mark.parametrize("order", [(1, 2, 3), (3, 1, 2), (2, 3, 1)])
def test_distinct_plans_rank_a_score_that_is_not_finite_last(order):
    plans = {1: _plan(1, math.nan, math.nan), 2: _plan(2, 5.0, 0.05), 3: _plan(3, 12.0, 0.12)}
    picks = select_distinct_plans([plans[i] for i in order], n=3)
    assert [p.index for p in picks] == [3]

    ties = [_plan(1, math.nan, 0.0), _plan(2, math.nan, 0.1), _plan(3, math.nan, 0.2)]
    assert [p.index for p in select_distinct_plans(ties, n=1)] == [1]


# ---- goal files ----------------------------------------------------------------------


def _goal_file(tmp_path, dose, fractions, goals):
    path = tmp_path / "goals.json"
    save_goals(path, goals, dose, fractions)
    return path


def test_per_fraction_goal_file_converts_with_its_own_fractions(tmp_path):
    goals = [
        Goal(voi="PTV", metric="D95", op=">=", value=1.9),
        Goal(voi="Rectum", metric="V1.5Gy", op="<=", value=35),
    ]
    path = _goal_file(tmp_path, "per_fraction", 30, goals)

    total, notes = load_goals(path, "total", 35)
    assert [g.label for g in total] == ["PTV D95 >= 57", "Rectum V45Gy <= 35"]
    assert len(notes) == 1 and "30 fractions" in notes[0] and "the plan has 35" in notes[0]
    assert "converted between per-fraction and total with the file's 30" in notes[0]
    assert "not adapted to the plan's fractionation" in notes[0]

    per_fraction, notes = load_goals(path, "per_fraction", 35)
    assert per_fraction == goals
    assert len(notes) == 1 and "used as written" in notes[0] and "converted" not in notes[0]

    assert load_goals(path, "per_fraction", 30) == (goals, [])


def test_total_goal_file_converts_with_its_own_fractions(tmp_path):
    goals = [
        Goal(voi="PTV", metric="D95", op=">=", value=57.0),
        Goal(voi="Rectum", metric="V45Gy", op="<=", value=35),
    ]
    path = _goal_file(tmp_path, "total", 30, goals)

    per_fraction, notes = load_goals(path, "per_fraction", 35)
    assert [g.label for g in per_fraction] == ["PTV D95 >= 1.9", "Rectum V1.5Gy <= 35"]
    assert len(notes) == 1 and "with the file's 30 fractions" in notes[0]

    total, notes = load_goals(path, "total", 35)
    assert total == goals and len(notes) == 1 and "used as written" in notes[0]


def test_goal_file_without_fractions_uses_the_plan_fractions(tmp_path):
    path = _goal_file(
        tmp_path, "per_fraction", None, [Goal(voi="PTV", metric="min", op=">=", value=2)]
    )
    total, notes = load_goals(path, "total", 35)
    assert total[0].value == 70 and notes == []


def test_goal_and_table_files_are_read_as_utf8(tmp_path):
    goal = {"voi": "Hörnerv", "metric": "max", "op": "<=", "value": 1.0, "source": "≤ 45 Gy"}
    path = tmp_path / "goals.json"
    text = json.dumps({"dose": "per_fraction", "goals": [goal]}, ensure_ascii=False)
    path.write_text(text, encoding="utf-8")
    assert load_goals(path, "per_fraction", 30)[0] == [Goal(**goal)]

    path = tmp_path / "table.json"
    table = {**_table(), "name": "Tabelle für Hörnerv ≤ 45 Gy"}
    path.write_text(json.dumps(table, ensure_ascii=False), encoding="utf-8")
    assert load_constraint_table(path).name == table["name"]


def test_scaled_volume_metric_has_no_exponent():
    goal = Goal(voi="Body", metric="V0.00002Gy", op="<=", value=5)
    scaled = _scale_goal(goal, 1 / 30)
    assert scaled.metric == "V0.000000666667Gy"
    assert Goal.model_validate(scaled.model_dump()) == scaled


# ---- validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("update", "match"),
    [
        ({"metric": "D150"}, "0 to 100"),
        ({"metric": "D100.5"}, "0 to 100"),
        ({"metric": "V20Gy", "value": 120}, "0 to 100"),
        ({"metric": "D95%"}, "Unsupported metric"),
        ({"value": -1}, "greater than or equal to 0"),
        ({"value": math.nan}, "finite"),
        ({"Hard": False}, "Extra inputs"),
    ],
)
def test_goal_rejects_invalid_content(update, match):
    with pytest.raises(ValueError, match=match):
        Goal(**{"voi": "PTV", "metric": "D95", "op": ">=", "value": 1.0, **update})


def test_goal_accepts_the_metric_limits():
    for metric in ("D0", "D100", "V0Gy", "mean", "max", "min"):
        Goal(voi="PTV", metric=metric, op=">=", value=100)


def test_goal_file_and_protocol_reject_unknown_keys(tmp_path):
    path = tmp_path / "goals.json"
    path.write_text(json.dumps({"dose": "total", "fraction": 30, "goals": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="fraction"):
        load_goals(path, "total", 30)
    with pytest.raises(ValueError, match="Extra inputs"):
        Protocol(goal=[])


def _table(constraint=None, **structure):
    constraint = constraint or {"metric": "max", "op": "<=", "value": 50}
    return {
        "id": "t",
        "name": "t",
        "version": "1",
        "dose_reference": {"type": "EQD2"},
        "sites": {"site": {"structures": {"Cord": {"constraints": [constraint], **structure}}}},
    }


@pytest.mark.parametrize(
    ("constraint", "match"),
    [
        ({"metric": "max", "op": "<=", "value": 50, "unit": "%"}, "unit '%'"),
        ({"metric": "D2", "op": "<=", "value": 50, "unit": "%"}, "unit '%'"),
        ({"metric": "V20Gy", "op": "<=", "value": 30, "unit": "%Rx"}, "unit '%Rx'"),
        ({"metric": "V20Gy", "op": "<=", "value": 30, "unit": "Gy"}, "unit 'Gy'"),
        ({"metric": "V20Gy", "op": "<=", "value": 130}, "0 to 100"),
        ({"metric": "D150", "op": "<=", "value": 30}, "0 to 100"),
        ({"metric": "max", "op": "<=", "value": -1}, "greater than or equal to 0"),
        ({"metric": "max", "op": "<=", "value": 50, "Hard": False}, "Extra inputs"),
    ],
)
def test_table_rejects_an_invalid_constraint(constraint, match):
    with pytest.raises(ValueError, match=match):
        ConstraintTable.model_validate(_table(constraint))


@pytest.mark.parametrize("alpha_beta", [0, -2])
def test_table_rejects_a_non_positive_alpha_beta(alpha_beta):
    with pytest.raises(ValueError, match="greater than 0"):
        ConstraintTable.model_validate(_table(alpha_beta=alpha_beta))
    table = ConstraintTable.model_validate(_table())
    with pytest.raises(ValueError, match="default_alpha_beta"):
        site_goals(table, "site", 30, default_alpha_beta=alpha_beta)


def test_table_rejects_unknown_keys_and_accepts_valid_units():
    with pytest.raises(ValueError, match="Extra inputs"):
        ConstraintTable.model_validate({**_table(), "sources": "x"})
    with pytest.raises(ValueError, match="Extra inputs"):
        ConstraintTable.model_validate(_table(alphabeta=2))
    for constraint in (
        {"metric": "max", "op": "<=", "value": 50, "unit": "Gy"},
        {"metric": "V20Gy", "op": "<=", "value": 30, "unit": "%"},
        {"metric": "D95", "op": ">=", "value": 95, "unit": "%Rx"},
    ):
        table = ConstraintTable.model_validate(_table(constraint))
        assert len(site_goals(table, "site", 30, 60.0)[0]) == 1


def test_table_alpha_beta_is_used_as_given():
    low = site_goals(ConstraintTable.model_validate(_table(alpha_beta=0.5)), "site", 5)[0][0]
    default = site_goals(ConstraintTable.model_validate(_table()), "site", 5)[0][0]
    assert low.value < default.value


def test_tiny_table_volume_threshold_has_no_exponent():
    constraint = {"metric": "V0.001Gy", "op": "<=", "value": 30}
    goal = site_goals(ConstraintTable.model_validate(_table(constraint)), "site", 30)[0][0]
    assert "e" not in goal.metric
    assert Goal.model_validate(goal.model_dump()) == goal


def test_demo_table_states_what_its_values_are():
    table = load_constraint_table("demo")
    for text in (table.disclaimer, table.source):
        assert "QUANTEC/RTOG-style" in text and "not validated" in text
    assert "NOT FOR CLINICAL USE" in table.disclaimer


# ---- geometry ------------------------------------------------------------------------


def _box(spacing=(1.0, 1.0, 5.0)):
    arr = np.zeros((12, 24, 24), dtype=bool)  # z, y, x
    arr[4:8, 6:18, 6:18] = True
    mask = sitk.GetImageFromArray(arr.astype(np.uint8))
    mask.SetSpacing(spacing)
    return mask, arr


def _extent(arr):
    return [(int(i.min()), int(i.max()) + 1) for i in np.nonzero(arr)]


def test_margins_round_down_to_whole_voxels_for_both_signs():
    mask, arr = _box()
    assert _extent(geo.expand(mask, 1.0)) == [(4, 8), (5, 19), (5, 19)]
    assert _extent(geo.expand(mask, -1.0)) == [(4, 8), (7, 17), (7, 17)]
    assert _extent(geo.expand(mask, 5.0))[0] == (3, 9)
    assert _extent(geo.expand(mask, -5.0))[0] == (5, 7)
    assert _extent(geo.expand(mask, 4.9))[0] == (4, 8)
    assert _extent(geo.expand(mask, -4.9))[0] == (4, 8)
    np.testing.assert_array_equal(geo.expand(mask, 0.0), arr)
    np.testing.assert_array_equal(geo.expand(sitk.Cast(mask, sitk.sitkUInt8), -0.5), arr)


@pytest.mark.parametrize("margin", [1.0, 3.0, 5.0])
def test_shrinking_undoes_growing_a_box(margin):
    mask, arr = _box()
    grown = geo.to_mask(geo.expand(mask, margin), mask)
    assert sitk.GetArrayViewFromImage(grown).sum() > arr.sum()
    np.testing.assert_array_equal(geo.expand(grown, -margin), arr)


def test_shrinking_a_full_mask_keeps_it():
    mask = sitk.GetImageFromArray(np.ones((3, 3, 3), dtype=np.uint8))
    assert geo.expand(mask, -2.0).all()

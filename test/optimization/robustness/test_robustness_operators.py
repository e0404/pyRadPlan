"""Expected-quantity and worst-case robustness operators."""

import logging

import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    SquaredUnderdosing,
    get_objective,
)
from pyRadPlan.optimization.robustness import (
    CompositeWorstCase,
    ExpectedQuantity,
    ObjectiveWorstCase,
    RobustnessOptions,
    RobustTermEntry,
    VoxelWorstCase,
    VoxelWorstCaseInverted,
    get_robustness_operator,
    smooth_max,
)

VALUES = np.array([[2.0, 0.5, 4.0], [1.0, 1.5, 3.0], [0.0, 3.0, 3.5]])
METHODS = ["logsumexp", "pnorm", "none"]


def _assert_fd(grad, fd):
    for g, f in zip(grad, fd):
        np.testing.assert_allclose(np.asarray(g), f, rtol=1e-5, atol=1e-7)


def test_expected_quantity(three_scenarios, operator_fd):
    objective = SquaredDeviation(d_ref=2.0, priority=3.0)
    op = get_robustness_operator("expected_quantity", three_scenarios)
    assert isinstance(op, ExpectedQuantity)
    assert op.scenario_indices == [0, 1, 2]
    entries = [RobustTermEntry(objective, "TARGET")]
    values = [xp.asarray(VALUES)]
    expected = np.array([0.5, 0.3, 0.2]) @ VALUES
    assert float(op.evaluate(entries, values)) == pytest.approx(
        3.0 * np.mean((expected - 2.0) ** 2)
    )
    grad = op.gradient(entries, values)
    assert grad[0].shape == (3, 3)
    np.testing.assert_allclose(
        np.asarray(grad[0]),
        3.0 * np.array([0.5, 0.3, 0.2])[:, None] * (2.0 * (expected - 2.0) / 3.0)[None, :],
    )
    _assert_fd(grad, operator_fd(op, entries, values))


@pytest.mark.parametrize(
    ("cls", "voi_type", "reduction"),
    [
        (VoxelWorstCase, "TARGET", np.min),
        (VoxelWorstCase, "OAR", np.max),
        (VoxelWorstCaseInverted, "TARGET", np.max),
        (VoxelWorstCaseInverted, "OAR", np.min),
    ],
)
def test_voxel_worst_case_selection(three_scenarios, operator_fd, cls, voi_type, reduction):
    objective = SquaredDeviation(d_ref=2.0, priority=2.0)
    op = cls(three_scenarios)
    entries = [RobustTermEntry(objective, voi_type)]
    values = [xp.asarray(VALUES)]
    worst = reduction(VALUES, axis=0)
    assert float(op.evaluate(entries, values)) == pytest.approx(2.0 * np.mean((worst - 2.0) ** 2))
    grad = np.asarray(op.gradient(entries, values)[0])
    arg = np.argmin(VALUES, axis=0) if reduction is np.min else np.argmax(VALUES, axis=0)
    one_hot = (np.arange(3)[:, None] == arg[None, :]).astype(float)
    np.testing.assert_allclose(grad, one_hot * 2.0 * 2.0 * (worst - 2.0) / 3.0)
    _assert_fd([grad], operator_fd(op, entries, values))


def test_voxel_worst_case_ties_take_first_scenario(three_scenarios):
    op = VoxelWorstCase(three_scenarios)
    values = [xp.asarray(np.array([[1.0, 2.0], [1.0, 1.0], [3.0, 1.0]]))]
    entries = [RobustTermEntry(SquaredDeviation(d_ref=0.0), "TARGET")]
    grad = np.asarray(op.gradient(entries, values)[0])
    np.testing.assert_array_equal(grad != 0, [[True, False], [False, True], [False, False]])


def test_voxel_worst_case_unknown_type_warns_once(three_scenarios, caplog):
    op = VoxelWorstCase(three_scenarios)
    entries = [RobustTermEntry(SquaredDeviation(d_ref=2.0), "EXTERNAL")]
    values = [xp.asarray(VALUES)]
    with caplog.at_level(logging.WARNING):
        value = float(op.evaluate(entries, values))
        op.gradient(entries, values)
    assert value == pytest.approx(np.mean((VALUES.max(axis=0) - 2.0) ** 2))
    assert sum("EXTERNAL" in r.getMessage() for r in caplog.records) == 1


def test_voxel_worst_case_refuses_several_ct_phases(make_set):
    four_d = make_set([(0.0, 0.0, 0.0)] * 2 + [(3.0, 0.0, 0.0)], ct_phase=[0, 1, 0])
    for cls in (VoxelWorstCase, VoxelWorstCaseInverted):
        with pytest.raises(NotImplementedError, match="CT phases"):
            cls(four_d)
    one_phase_selected = make_set(
        [(0.0, 0.0, 0.0)] * 2 + [(3.0, 0.0, 0.0)], weights=[0.5, 0.0, 0.5], ct_phase=[0, 1, 0]
    )
    assert VoxelWorstCase(one_phase_selected).scenario_indices == [0, 2]


@pytest.mark.parametrize("method", METHODS)
def test_objective_worst_case(three_scenarios, operator_fd, method):
    objective = SquaredDeviation(d_ref=2.0, priority=1.5)
    op = get_robustness_operator(
        "OWC", three_scenarios, RobustnessOptions(max_approximation=method, p_norm=8.0)
    )
    assert isinstance(op, ObjectiveWorstCase)
    entries = [RobustTermEntry(objective)]
    values = [xp.asarray(VALUES)]
    per_scenario = 1.5 * np.mean((VALUES - 2.0) ** 2, axis=1)
    expected, dmax = smooth_max(xp.asarray(per_scenario), method, 8.0)
    assert float(op.evaluate(entries, values)) == pytest.approx(float(expected))
    if method == "none":
        assert float(op.evaluate(entries, values)) == pytest.approx(per_scenario.max())
    grad = op.gradient(entries, values)
    np.testing.assert_allclose(
        np.asarray(grad[0]),
        np.asarray(dmax)[:, None] * 1.5 * 2.0 * (VALUES - 2.0) / 3.0,
    )
    _assert_fd(grad, operator_fd(op, entries, values))


@pytest.mark.parametrize("method", METHODS)
def test_composite_worst_case_two_entries(three_scenarios, operator_fd, method):
    first = SquaredUnderdosing(d_min=3.0, priority=2.0)
    second = SquaredOverdosing(d_max=1.0, priority=0.5)
    op = get_robustness_operator(
        "COWC", three_scenarios, RobustnessOptions(max_approximation=method, p_norm=10.0)
    )
    assert isinstance(op, CompositeWorstCase)
    assert op.scope == "composite"
    entries = [RobustTermEntry(first, "TARGET"), RobustTermEntry(second, "OAR")]
    values_first = np.array([[2.0, 3.5], [1.0, 2.5], [2.8, 1.5]])
    values_second = VALUES
    values = [xp.asarray(values_first), xp.asarray(values_second)]

    composite = 2.0 * np.mean(np.minimum(values_first - 3.0, 0.0) ** 2, axis=1) + 0.5 * np.mean(
        np.maximum(values_second - 1.0, 0.0) ** 2, axis=1
    )
    expected, _ = smooth_max(xp.asarray(composite), method, 10.0)
    assert float(op.evaluate(entries, values)) == pytest.approx(float(expected))

    grad = op.gradient(entries, values)
    assert [g.shape for g in grad] == [(3, 2), (3, 3)]
    _assert_fd(grad, operator_fd(op, entries, values))


def test_composite_worst_case_exact_max_without_positive_values(three_scenarios):
    objective = SquaredOverdosing(d_max=10.0)
    op = CompositeWorstCase(three_scenarios)
    entries = [RobustTermEntry(objective, "OAR")]
    values = [xp.asarray(VALUES)]
    assert float(op.evaluate(entries, values)) == 0.0
    np.testing.assert_array_equal(np.asarray(op.gradient(entries, values)[0]), 0.0)


def test_operators_use_scenarios_with_weight(make_set):
    scen_set = make_set(
        [(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)], weights=[0.6, 0.0, 0.4]
    )
    for name in ("PROB", "VWWC", "VWWC_INV", "OWC", "COWC"):
        assert get_robustness_operator(name, scen_set).scenario_indices == [0, 2]


@pytest.mark.parametrize(
    ("alias", "canonical", "matrad"),
    [
        ("OWC", "objective_worst_case", "OWC"),
        ("COWC", "composite_worst_case", "COWC"),
        ("VWWC", "voxel_worst_case", "VWWC"),
        ("VWWC_INV", "voxel_worst_case_inverted", "VWWC_INV"),
    ],
)
def test_alias_normalization_and_export(alias, canonical, matrad):
    objective = SquaredDeviation(robustness=alias, d_ref=60.0)
    assert objective.robustness == canonical
    exported = objective.to_matrad()
    assert exported["robustness"] == matrad
    assert get_objective(dict(exported)).robustness == canonical
    assert SquaredDeviation(robustness=canonical).robustness == canonical
    assert canonical in SquaredDeviation.available_robustness()


def test_prob_alias_warns_and_exports_none(caplog):
    with caplog.at_level(logging.WARNING):
        objective = get_objective(
            {
                "className": "DoseObjectives.matRad_SquaredDeviation",
                "parameters": [60.0],
                "penalty": 10.0,
                "robustness": "PROB",
            }
        )
    assert objective.robustness == "expected_quantity"
    assert sum("variance" in r.getMessage() for r in caplog.records) == 1

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        SquaredDeviation(robustness="expected_quantity")
        exported = objective.to_matrad()
    assert not any("variance" in r.getMessage() for r in caplog.records)
    assert exported["robustness"] == "none"

"""Robustness operator contract: registry, objective field and the two basic operators."""

import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, get_objective
from pyRadPlan.optimization.robustness import (
    ExpectedValue,
    NoRobustness,
    RobustnessOperator,
    RobustnessOptions,
    RobustTermEntry,
    ScenarioStack,
    available_robustness,
    get_robustness_operator,
    get_robustness_operator_class,
    register_robustness_operator,
    resolve_robustness_name,
)
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)


def _set(setup, weights=None, ct_phase=None):
    points = {"setup": setup}
    phases = CTPhaseUncertainty()
    if ct_phase is not None:
        points["ct_phase"] = [[p] for p in ct_phase]
        unique = sorted(set(ct_phase))
        phases = CTPhaseUncertainty(phases=unique, probabilities=[1.0 / len(unique)] * len(unique))
    return ScenarioSet.from_points(
        (phases, SetupUncertainty(), RangeUncertainty()), points, weights=weights
    )


@pytest.fixture
def three_scenarios():
    return _set([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, -3.0, 0.0)], weights=[0.5, 0.3, 0.2])


def _finite_difference(operator, entries, values, eps=1e-6):
    base = np.asarray(values[0], dtype=float)
    grad = np.zeros_like(base)
    for s in range(base.shape[0]):
        for v in range(base.shape[1]):
            plus = base.copy()
            plus[s, v] += eps
            minus = base.copy()
            minus[s, v] -= eps
            f_plus = float(operator.evaluate(entries, [xp.asarray(plus)]))
            f_minus = float(operator.evaluate(entries, [xp.asarray(minus)]))
            grad[s, v] = (f_plus - f_minus) / (2 * eps)
    return grad


def test_registry_and_aliases():
    names = available_robustness()
    assert names[0] == "none"
    assert {"expected_value", "scenario_stack"} <= set(names)
    assert resolve_robustness_name("STOCH") == "expected_value"
    assert resolve_robustness_name("stochastic") == "expected_value"
    assert resolve_robustness_name(" None ") == "none"
    assert resolve_robustness_name("owc") == resolve_robustness_name("OWC")
    assert resolve_robustness_name("Stoch") == "expected_value"
    assert get_robustness_operator_class("STOCH") is ExpectedValue
    with pytest.raises(ValueError, match="Unknown robustness"):
        resolve_robustness_name("unknown")
    with pytest.raises(ValueError, match="scenario set"):
        get_robustness_operator("none")


def test_register_rejects_conflicts():
    class _Dup(RobustnessOperator):
        name = "none"

        def evaluate(self, entries, values):
            return 0.0

        def gradient(self, entries, values):
            return []

    with pytest.raises(ValueError, match="already registered"):
        register_robustness_operator(_Dup)
    with pytest.raises(TypeError):
        register_robustness_operator(object)
    # re-registering the same class is a no-op
    assert register_robustness_operator(NoRobustness) is NoRobustness


def test_objective_robustness_field_normalizes_and_exports():
    obj = SquaredDeviation(
        robustness="OWC" if "objective_worst_case" in available_robustness() else "STOCH"
    )
    assert obj.robustness in ("objective_worst_case", "expected_value")
    assert SquaredDeviation().robustness == "none"
    assert "none" in SquaredDeviation.available_robustness()
    assert "scenario_stack" not in SquaredDeviation.available_robustness()
    with pytest.raises(ValueError):
        SquaredDeviation(robustness="scenario_stack")
    with pytest.raises(ValueError):
        SquaredDeviation(robustness="bogus")

    exported = SquaredDeviation(robustness="STOCH", d_ref=60).to_matrad()
    assert exported["robustness"] == "STOCH"
    back = get_objective(dict(exported))
    assert back.robustness == "expected_value"
    assert (
        get_objective(
            {
                "className": "DoseObjectives.matRad_SquaredDeviation",
                "parameters": [60],
                "penalty": 10,
            }
        ).robustness
        == "none"
    )


def test_none_operator_uses_nominal_scenarios(three_scenarios):
    objective = SquaredOverdosing(d_ref=1.0, priority=2.0)
    op = get_robustness_operator("none", three_scenarios)
    assert isinstance(op, NoRobustness)
    assert op.scenario_indices == [0]
    values = xp.asarray(np.array([[2.0, 3.0]]))
    entries = [RobustTermEntry(objective, "OAR")]
    expected = 2.0 * float(objective.compute_objective(values[0, ...]))
    assert float(op.evaluate(entries, [values])) == pytest.approx(expected)
    grad = op.gradient(entries, [values])[0]
    assert grad.shape == (1, 2)
    np.testing.assert_allclose(np.asarray(grad), _finite_difference(op, entries, [values]))
    with pytest.raises(ValueError, match="stacked"):
        op.evaluate(entries, [xp.asarray(np.ones((3, 2)))])


def test_none_operator_sums_nominal_ct_phases():
    scen_set = _set(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0)],
        weights=[0.5, 0.5, 0.0],
        ct_phase=[0, 1, 0],
    )
    op = NoRobustness(scen_set)
    assert op.scenario_indices == [0, 1]
    objective = SquaredDeviation(d_ref=1.0)
    values = xp.asarray(np.array([[2.0, 2.0], [0.0, 0.0]]))
    entries = [RobustTermEntry(objective)]
    assert float(op.evaluate(entries, [values])) == pytest.approx(1.0 + 1.0)


def test_expected_value_operator(three_scenarios):
    objective = SquaredDeviation(d_ref=1.0, priority=3.0)
    op = get_robustness_operator("STOCH", three_scenarios, RobustnessOptions())
    assert op.scenario_indices == [0, 1, 2]
    np.testing.assert_allclose(op.weights, [0.5, 0.3, 0.2])
    values = xp.asarray(np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]]))
    entries = [RobustTermEntry(objective, "TARGET")]
    per_scenario = [float(objective.compute_objective(values[s, ...])) for s in range(3)]
    expected = 3.0 * np.dot([0.5, 0.3, 0.2], per_scenario)
    assert float(op.evaluate(entries, [values])) == pytest.approx(expected)
    grad = op.gradient(entries, [values])[0]
    assert grad.shape == (3, 2)
    np.testing.assert_allclose(
        np.asarray(grad), _finite_difference(op, entries, [values]), rtol=1e-6, atol=1e-8
    )


def test_expected_value_skips_zero_weight_scenarios():
    scen_set = _set([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0)], weights=[0.6, 0.0, 0.4])
    op = ExpectedValue(scen_set)
    assert op.scenario_indices == [0, 2]
    np.testing.assert_allclose(op.weights, [0.6, 0.4])


def test_objective_scope_operators_take_one_entry(three_scenarios):
    op = ExpectedValue(three_scenarios)
    entries = [RobustTermEntry(SquaredDeviation()), RobustTermEntry(SquaredDeviation())]
    values = [xp.asarray(np.ones((3, 2)))] * 2
    with pytest.raises(ValueError, match="exactly one"):
        op.evaluate(entries, values)


def test_scenario_stack_passes_weights(three_scenarios):
    class _Spread(SquaredDeviation):
        scenario_aware = True

        def compute_objective(self, values, weights=None):
            xp_ = __import__("array_api_compat").array_namespace(values)
            w = xp_.asarray(np.asarray(weights), dtype=values.dtype)
            mean = xp_.sum(values * w[:, None], axis=0)
            return xp_.sum(w[:, None] * (values - mean) ** 2)

        def compute_gradient(self, values, weights=None):
            xp_ = __import__("array_api_compat").array_namespace(values)
            w = xp_.asarray(np.asarray(weights), dtype=values.dtype)
            mean = xp_.sum(values * w[:, None], axis=0)
            return 2.0 * w[:, None] * (values - mean)

    objective = _Spread(robustness="none", priority=2.0)
    assert objective.robustness == "scenario_stack"
    assert _Spread.available_robustness() == ["scenario_stack"]
    op = get_robustness_operator(objective.robustness, three_scenarios)
    assert isinstance(op, ScenarioStack)
    values = xp.asarray(np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0]]))
    entries = [RobustTermEntry(objective)]
    grad = op.gradient(entries, [values])[0]
    np.testing.assert_allclose(
        np.asarray(grad), _finite_difference(op, entries, [values]), rtol=1e-6, atol=1e-8
    )


def test_nominal_indices_on_sets():
    scen_set = _set(
        [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
        ct_phase=[0, 1, 0, 1],
    )
    np.testing.assert_array_equal(scen_set.nominal_indices(), [0, 1, 3])
    assert scen_set.nominal_index() == 0

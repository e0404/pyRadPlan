"""Scenario-aware variance objective."""

import numpy as np
import pytest
import array_api_strict as xp
from pydantic import TypeAdapter

from pyRadPlan.optimization.objectives import (
    Variance,
    get_available_objectives,
    get_objective,
    get_objectives_union,
)
from pyRadPlan.optimization.robustness import RobustTermEntry, ScenarioStack

VALUES = np.array([[2.0, 0.5, 4.0], [1.0, 1.5, 3.0], [0.0, 3.0, 3.5]])
WEIGHTS = np.array([0.5, 0.3, 0.2])


def test_variance_closed_form_uniform_default():
    objective = Variance()
    values = xp.asarray(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert float(objective.compute_objective(values)) == pytest.approx(1.0)
    np.testing.assert_allclose(
        np.asarray(objective.compute_gradient(values)), [[-0.5, -0.5], [0.5, 0.5]]
    )
    uniform = objective.compute_objective(values, weights=[0.5, 0.5])
    assert float(uniform) == pytest.approx(1.0)


def test_variance_weighted_value_and_gradient(finite_difference):
    objective = Variance()
    values = xp.asarray(VALUES)
    mean = WEIGHTS @ VALUES
    expected = np.mean(np.average((VALUES - mean) ** 2, axis=0, weights=WEIGHTS))
    assert float(objective.compute_objective(values, weights=WEIGHTS)) == pytest.approx(expected)
    grad = objective.compute_gradient(values, weights=xp.asarray(WEIGHTS))
    assert grad.shape == (3, 3)
    np.testing.assert_allclose(
        np.asarray(grad), 2.0 * WEIGHTS[:, None] * (VALUES - mean) / VALUES.shape[1]
    )
    fd = finite_difference(lambda v: objective.compute_objective(v[0], weights=WEIGHTS), [values])
    np.testing.assert_allclose(np.asarray(grad), fd[0], rtol=1e-6, atol=1e-8)
    with pytest.raises(ValueError, match="one weight per scenario"):
        objective.compute_objective(values, weights=[0.5, 0.5])


def test_variance_single_scenario_is_zero():
    objective = Variance()
    values = xp.asarray(np.array([1.0, 2.0]))
    assert float(objective.compute_objective(values)) == 0.0
    np.testing.assert_array_equal(np.asarray(objective.compute_gradient(values)), 0.0)


def test_variance_through_scenario_stack(three_scenarios, operator_fd):
    objective = Variance(priority=4.0, robustness="OWC")
    assert objective.robustness == "scenario_stack"
    assert Variance.available_robustness() == ["scenario_stack"]
    op = ScenarioStack(three_scenarios)
    entries = [RobustTermEntry(objective)]
    values = [xp.asarray(VALUES)]
    mean = WEIGHTS @ VALUES
    expected = 4.0 * np.mean(np.average((VALUES - mean) ** 2, axis=0, weights=WEIGHTS))
    assert float(op.evaluate(entries, values)) == pytest.approx(expected)
    grad = op.gradient(entries, values)
    np.testing.assert_allclose(
        np.asarray(grad[0]), operator_fd(op, entries, values)[0], rtol=1e-6, atol=1e-8
    )


def test_variance_registration_and_export():
    assert get_available_objectives()["Variance"] is Variance
    assert isinstance(get_objective("Variance"), Variance)
    assert isinstance(get_objective({"name": "Variance", "penalty": 2.0}), Variance)
    assert Variance().to_matrad() is None
    union = get_objectives_union(exclude_image_references=True)
    restored = TypeAdapter(union).validate_python(Variance(priority=3.0).model_dump())
    assert isinstance(restored, Variance)
    assert restored.priority == 3.0

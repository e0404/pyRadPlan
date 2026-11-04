"""Shared helpers for the robustness operator tests."""

import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)


def _make_set(setup, weights=None, ct_phase=None):
    points = {"setup": setup}
    phases = CTPhaseUncertainty()
    if ct_phase is not None:
        points["ct_phase"] = [[p] for p in ct_phase]
        unique = sorted(set(ct_phase))
        phases = CTPhaseUncertainty(phases=unique, probabilities=[1.0 / len(unique)] * len(unique))
    return ScenarioSet.from_points(
        (phases, SetupUncertainty(), RangeUncertainty()), points, weights=weights
    )


def _finite_difference(evaluate, values, eps=1e-6):
    bases = [np.asarray(v, dtype=float) for v in values]
    grads = []
    for j, base in enumerate(bases):
        grad = np.zeros_like(base)
        for index in np.ndindex(base.shape):
            shifted = []
            for sign in (1.0, -1.0):
                perturbed = [b.copy() for b in bases]
                perturbed[j][index] += sign * eps
                shifted.append(float(evaluate([xp.asarray(p) for p in perturbed])))
            grad[index] = (shifted[0] - shifted[1]) / (2 * eps)
        grads.append(grad)
    return grads


@pytest.fixture
def make_set():
    """Build a scenario set from setup shifts, weights and optional CT phases."""
    return _make_set


@pytest.fixture
def finite_difference():
    """Central differences of ``evaluate(values)`` w.r.t. every entry of ``values``."""
    return _finite_difference


@pytest.fixture
def operator_fd(finite_difference):
    """Finite-difference cotangents of an operator for the given entries."""

    def _fd(operator, entries, values, eps=1e-6):
        return finite_difference(lambda v: operator.evaluate(entries, v), values, eps)

    return _fd


@pytest.fixture
def three_scenarios(make_set):
    return make_set([(0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (0.0, -3.0, 0.0)], weights=[0.5, 0.3, 0.2])

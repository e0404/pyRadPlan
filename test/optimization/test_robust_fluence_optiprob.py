"""Robust terms in the non-linear fluence planning problem."""

import logging

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core import xp_utils
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    SquaredUnderdosing,
)
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.optimization.robustness import (
    RobustnessOperator,
    RobustnessOptions,
    ScenarioStack,
    _factory,
    register_robustness_operator,
)
from pyRadPlan.quantities import QuantityResolver
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)
from pyRadPlan.stf import validate_stf

# Scales the matrix to doses of order one for fluences of order one.
DOSE_SCALE = 1000.0
SCALES = (1.0, 0.85, 1.2)
WEIGHTS = (0.5, 0.3, 0.2)


class _SquaredScenarioSum(RobustnessOperator):
    """Test composite: ``sum_s w_s (sum_j p_j f_js)^2``, coupling the entries per scenario."""

    name = "test_squared_scenario_sum"
    scope = "composite"

    def _totals(self, entries, values):
        xp = array_api_compat.array_namespace(values[0])
        total = None
        for entry, value in zip(entries, values):
            objective = entry.objective
            f = xp.stack(
                [objective.compute_objective(value[s, ...]) for s in range(self.num_scenarios)]
            )
            total = objective.priority * f if total is None else total + objective.priority * f
        return xp, total, xp.asarray(np.asarray(self.weights), dtype=total.dtype)

    def evaluate(self, entries, values):
        self._check(entries, values)
        xp, total, weights = self._totals(entries, values)
        return xp.sum(weights * total**2)

    def gradient(self, entries, values):
        self._check(entries, values)
        xp, total, weights = self._totals(entries, values)
        coefficient = 2.0 * weights * total
        grads = []
        for entry, value in zip(entries, values):
            objective = entry.objective
            g = xp.stack(
                [objective.compute_gradient(value[s, ...]) for s in range(self.num_scenarios)]
            )
            grads.append(objective.priority * coefficient[:, None] * g)
        return grads


class _Spread(SquaredDeviation):
    """Test scenario-aware objective: weighted spread of the voxel values over the scenarios."""

    scenario_aware = True

    def compute_objective(self, values, weights=None):
        xp = array_api_compat.array_namespace(values)
        w = xp.asarray(np.asarray(weights), dtype=values.dtype)[:, None]
        mean = xp.sum(values * w, axis=0)
        return xp.sum(w * (values - mean) ** 2) / values.shape[1]

    def compute_gradient(self, values, weights=None):
        xp = array_api_compat.array_namespace(values)
        w = xp.asarray(np.asarray(weights), dtype=values.dtype)[:, None]
        mean = xp.sum(values * w, axis=0)
        return 2.0 * w * (values - mean) / values.shape[1]


@pytest.fixture
def composite_operator():
    register_robustness_operator(_SquaredScenarioSum)
    yield _SquaredScenarioSum.name
    _factory._OPERATORS.pop(_SquaredScenarioSum.name, None)
    _factory._ALIASES.pop(_SquaredScenarioSum.name, None)


@pytest.fixture
def proton_case(test_data_protons_raw):
    """
    The small proton case with its physical-dose matrix only.

    The dose grid is placed on the CT (the stored one lacks the CT's origin, which would move
    the target off its dose when the structures are resampled onto it).
    """
    tmp = test_data_protons_raw
    pln = tmp["pln"]
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    stf = validate_stf(tmp["stf"])
    dij = validate_dij(tmp["dij"])
    dij.dose_grid = ct.grid
    for name in dij.quantities:
        if name != "physical_dose":
            setattr(dij, name, None)
    return pln, ct, cst, stf, dij


def _scenario_set():
    return ScenarioSet.from_points(
        (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()),
        {"setup": [[0, 0, 0], [3, 0, 0], [-3, 0, 0]]},
        weights=list(WEIGHTS),
    )


def _dij_with(dij, scales, scenarios=None):
    """Copy of ``dij`` with one scaled copy of its matrix per scenario slot."""
    base = DOSE_SCALE * dij.physical_dose.flat[0]
    out = dij.model_copy(deep=True)
    container = np.empty((len(scales),), dtype=object)
    for i, scale in enumerate(scales):
        container[i] = scale * base
    out.physical_dose = container
    out.scenarios = scenarios
    return out


def _problem(case, dij, cst=None, **prop_opt):
    pln, ct, case_cst, stf, _ = case
    pln = dict(pln)
    pln["prop_opt"] = {
        "solver": {"name": "scipy", "max_iter": 1, "display": False},
        "convert_dose_objectives": False,
        **prop_opt,
    }
    prob = NonLinearFluencePlanningProblem(pln)
    prob._ct = ct
    prob._cst = case_cst if cst is None else cst
    prob._stf = stf
    prob._dij = dij
    prob._initialize()
    return prob


def _set_robustness(cst, robustness):
    for voi in cst.vois:
        for obj in voi.objectives:
            obj.robustness = robustness


def _fluence(prob, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 1.0, prob._dij.total_num_of_bixels)
    return xp_utils.from_numpy(prob._array_backend, x)


def _value(prob, x):
    return float(prob._objective_function(x))


def test_none_evaluates_only_the_nominal_scenario(proton_case):
    pln, ct, cst, stf, dij = proton_case
    single = _problem(proton_case, _dij_with(dij, (1.0,)))
    robust = _problem(proton_case, _dij_with(dij, SCALES, _scenario_set()))
    x = _fluence(single)

    assert _value(robust, x) == pytest.approx(_value(single, x), rel=1e-12)
    nominal = [int(i) for i in robust._scenario_set.nominal_indices()]
    assert nominal == [0]
    for q in robust._quantities:
        assert q.scenarios.tolist() == nominal
        assert all(q._q_cache.flat[s] is None for s in range(1, len(SCALES)))


def test_expected_value_is_the_weighted_sum_of_scenario_values(proton_case):
    pln, ct, cst, stf, dij = proton_case
    _set_robustness(cst, "expected_value")
    robust = _problem(proton_case, _dij_with(dij, SCALES, _scenario_set()))
    x = _fluence(robust)

    _set_robustness(cst, "none")
    expected = sum(
        w * _value(_problem(proton_case, _dij_with(dij, (scale,))), x)
        for scale, w in zip(SCALES, WEIGHTS)
    )
    assert _value(robust, x) == pytest.approx(expected, rel=1e-12)
    for q in robust._quantities:
        assert q.scenarios.tolist() == [0, 1, 2]


@pytest.fixture
def mixed_problem(proton_case, composite_operator):
    """
    Problem mixing ``none``, ``expected_value``, a scenario-aware objective and a composite.

    The composite acts on a target objective on ``rbe_x_dose`` and a body objective on
    ``physical_dose``.
    """
    pln, ct, cst, stf, dij = proton_case
    target, body = cst.vois[0], cst.vois[1]
    assert (target.voi_type, body.voi_type) == ("TARGET", "OAR")
    target.objectives = [
        SquaredDeviation(priority=2.0, d_ref=1.0),
        _Spread(priority=5.0, d_ref=0.0),
        SquaredUnderdosing(
            priority=1.5, d_min=1.6, quantity="rbe_x_dose", robustness=composite_operator
        ),
    ]
    body.objectives = [
        SquaredOverdosing(priority=3.0, d_max=0.1, robustness="expected_value"),
        SquaredOverdosing(priority=0.5, d_max=0.05, robustness=composite_operator),
    ]
    robust_dij = _dij_with(dij, SCALES, _scenario_set())
    robust_dij.bio_model = {"model": "constant_rbe", "rbe": 1.1}
    return _problem(proton_case, robust_dij, cst=cst)


def test_terms_group_objectives(mixed_problem):
    prob = mixed_problem
    operators = [type(term.operator).__name__ for term in prob._terms]
    assert operators == ["NoRobustness", "ScenarioStack", "_SquaredScenarioSum", "ExpectedValue"]
    assert isinstance(prob._terms[1].operator, ScenarioStack)
    composite = prob._terms[2]
    assert [item.voi_type for item in composite.items] == ["TARGET", "OAR"]
    assert composite.quantities == ["rbe_x_dose", "physical_dose"]
    assert prob._num_objectives == 4
    assert prob._objectives_per_quantity == {"physical_dose": [0, 1, 2, 3], "rbe_x_dose": [2]}
    assert prob._q_cache_index[2] == {"rbe_x_dose": 0, "physical_dose": 2}
    for q in prob._quantities:
        assert q.scenarios.tolist() == [0, 1, 2]


def test_mixed_gradient_matches_finite_differences(mixed_problem):
    prob = mixed_problem
    rng = np.random.default_rng(1)
    for seed in range(3):
        x = _fluence(prob, seed)
        gradient = xp_utils.to_numpy(prob._objective_gradient(x))
        x_np = xp_utils.to_numpy(x)
        for _ in range(2):
            direction = rng.standard_normal(x_np.size)
            direction /= np.linalg.norm(direction)
            h = 1e-4 * np.linalg.norm(x_np)
            f_plus = _value(prob, xp_utils.from_numpy(prob._array_backend, x_np + h * direction))
            f_minus = _value(prob, xp_utils.from_numpy(prob._array_backend, x_np - h * direction))
            numeric = (f_plus - f_minus) / (2 * h)
            assert gradient @ direction == pytest.approx(numeric, rel=1e-5)


def test_jacobian_has_one_row_per_term(mixed_problem):
    prob = mixed_problem
    x = _fluence(prob)
    gradient = xp_utils.to_numpy(prob._objective_gradient(x))

    prob.bypass_objective_jacobian = False
    prob._grad_cache = prob._grad_cache_intermediate = None
    for q in prob._quantities:
        q._w_grad_cache = None
    jacobian = xp_utils.to_numpy(prob._objective_jacobian(x))
    values = xp_utils.to_numpy(prob._objective_functions(x))

    num_objectives = sum(len(objs) for _, objs in prob._objective_list)
    assert num_objectives == 5
    assert jacobian.shape == (len(prob._terms), prob._dij.total_num_of_bixels)
    assert values.shape == (len(prob._terms),)
    np.testing.assert_allclose(jacobian.sum(axis=0), gradient, rtol=1e-6, atol=1e-12)


def test_multi_slot_dij_without_scenario_set(proton_case, caplog):
    pln, ct, cst, stf, dij = proton_case
    legacy = _dij_with(dij, SCALES)
    cst.vois[1].objectives[0].robustness = "expected_value"
    with pytest.raises(ValueError, match="attach_scenarios"):
        _problem(proton_case, legacy)

    cst.vois[1].objectives[0].robustness = "none"
    with caplog.at_level(logging.WARNING):
        prob = _problem(proton_case, legacy)
    assert "slot 0" in caplog.text
    single = _problem(proton_case, _dij_with(dij, (1.0,)))
    x = _fluence(prob)
    assert _value(prob, x) == pytest.approx(_value(single, x), rel=1e-12)
    for q in prob._quantities:
        assert q.scenarios.tolist() == [0]


def test_robustness_options_reach_the_operators(proton_case):
    pln, ct, cst, stf, dij = proton_case
    assert {"max_approximation", "p_norm"} <= set(
        NonLinearFluencePlanningProblem.config_model().model_fields
    )
    prob = _problem(
        proton_case,
        _dij_with(dij, SCALES, _scenario_set()),
        max_approximation="pnorm",
        p_norm=12.0,
    )
    assert prob._terms
    for term in prob._terms:
        assert term.operator.options == RobustnessOptions(max_approximation="pnorm", p_norm=12.0)


def test_resolver_widens_dependencies_to_their_consumers(proton_case):
    pln, ct, cst, stf, dij = proton_case
    robust_dij = _dij_with(dij, SCALES, _scenario_set())
    robust_dij.bio_model = {"model": "constant_rbe", "rbe": 1.1}
    resolver = QuantityResolver(robust_dij, scenarios={"rbe_x_dose": [2, 0], "physical_dose": [1]})
    resolver.resolve(["rbe_x_dose", "physical_dose"])
    assert resolver.instances["rbe_x_dose"].scenarios.tolist() == [0, 2]
    assert resolver.instances["physical_dose"].scenarios.tolist() == [0, 1, 2]

    resolver = QuantityResolver(robust_dij, scenarios={"rbe_x_dose": [1]})
    resolver.resolve(["rbe_x_dose"])
    assert resolver.instances["physical_dose"].scenarios.tolist() == [1]

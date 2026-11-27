"""Parameterized POLO instances (one per risk structure) under robust optimization."""

import numpy as np
import pytest

from pyRadPlan.core.xp_utils import from_numpy, to_numpy
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.optimization.objectives import NTCPPOLO
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.plan import validate_pln
from pyRadPlan.quantities import QuantityResolver
from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    RangeUncertainty,
    ScenarioSet,
    SetupUncertainty,
)
from pyRadPlan.stf import validate_stf

SCALES = (1.0, 1.1, 0.9)


@pytest.fixture
def robust_case(test_data_protons_raw):
    """Return the proton case with one scaled dose and LETxDose matrix per setup scenario."""
    tmp = test_data_protons_raw
    pln = validate_pln(tmp["pln"])
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    dij.dose_grid.origin = np.array(ct.cube_hu.GetOrigin())
    for voi in cst.vois:
        voi.objectives = []
    vois = {voi.name: voi for voi in cst.vois}
    # Body keeps its region in every run, whether or not Target carries an objective.
    vois["Body"].overlap_handling = "keep"

    for name in ("physical_dose", "let_dose"):
        base = getattr(dij, name).flat[0]
        container = np.empty((len(SCALES),), dtype=object)
        for i, scale in enumerate(SCALES):
            container[i] = scale * base
        setattr(dij, name, container)
    dij.scenarios = ScenarioSet.from_points(
        (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty()),
        {"setup": [[0, 0, 0], [3, 0, 0], [-3, 0, 0]]},
        weights=[0.5, 0.25, 0.25],
    )
    return pln, ct, cst, dij, stf, vois


def _problem(case, objectives):
    pln, ct, cst, dij, stf, vois = case
    for name, objs in objectives.items():
        vois[name].objectives = objs
    prob = NonLinearFluencePlanningProblem(pln)
    prob._ct, prob._cst, prob._dij, prob._stf = ct, cst, dij, stf
    prob.bypass_objective_jacobian = False  # one jacobian row per term
    prob._initialize()
    return prob


def _fluence(prob, seed=0):
    # A course dose inside the 2 Gy domain of the model, but short of saturating it.
    rng = np.random.default_rng(seed)
    x = 500.0 * rng.uniform(0.8, 1.2, prob._dij.total_num_of_bixels)
    return from_numpy(prob._array_backend, x)


def _assert_jacobian_matches_finite_differences(prob, x, seed=1):
    xp = prob._array_backend
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(prob._dij.total_num_of_bixels)
    h = 1e-3 * float(np.linalg.norm(to_numpy(x))) / np.linalg.norm(v)
    step = from_numpy(xp, h * v)
    f_plus = to_numpy(prob._objective_functions(x + step))
    f_minus = to_numpy(prob._objective_functions(x - step))
    jac = to_numpy(prob._objective_jacobian(x))
    np.testing.assert_allclose(jac @ v, (f_plus - f_minus) / (2 * h), rtol=1e-3)


def _polo_by_risk(prob):
    return {q.risk_voi_name: q for q in prob._quantities if q.identifier == "polo"}


def test_resolver_scenarios_follow_instance_keys(robust_case):
    _, _, cst, dij, _, _ = robust_case
    resolver = QuantityResolver(dij, context={"cst": cst, "num_of_fractions": 1})
    on_body = resolver.key("polo", {"risk_voi_name": "Body"})
    on_target = resolver.key("polo", {"risk_voi_name": "Target"})
    assert on_body != on_target

    resolver.scenarios = {on_body: [0], on_target: [2]}
    body, target = resolver.resolve(
        ["polo", "polo"], [{"risk_voi_name": "Body"}, {"risk_voi_name": "Target"}]
    )

    assert body is resolver.instances[on_body]
    assert target is resolver.instances[on_target]
    assert body.scenarios.tolist() == [0]
    assert target.scenarios.tolist() == [2]
    # The shared dependency evaluates what either consumer reads.
    assert body.dependencies["physical_dose"] is target.dependencies["physical_dose"]
    assert body.dependencies["physical_dose"].scenarios.tolist() == [0, 2]


@pytest.mark.parametrize("robustness", ["STOCH", "OWC", "VWWC"])
def test_parameterized_polo_objectives_are_robust_per_key(robust_case, robustness):
    on_target = NTCPPOLO(priority=1.0, risk_voi_name="Target", robustness=robustness)
    on_body = NTCPPOLO(priority=2.0, risk_voi_name="Body", robustness=robustness)
    prob = _problem(robust_case, {"Target": [on_target], "Body": [on_body]})

    polo = _polo_by_risk(prob)
    assert set(polo) == {"Body", "Target"}
    for q in polo.values():
        assert q.scenarios.tolist() == [0, 1, 2]
    assert len(prob._terms) == 2
    assert [term.quantities for term in prob._terms] == [
        [key] for key in prob._objectives_per_quantity if key.startswith("polo[")
    ]

    x = _fluence(prob)
    f = to_numpy(prob._objective_functions(x))
    assert np.all(f > 0)
    _assert_jacobian_matches_finite_differences(prob, x)

    # Each robust term behaves exactly as it does alone.
    f_target = to_numpy(
        _problem(robust_case, {"Target": [on_target], "Body": []})._objective_functions(x)
    )
    f_body = to_numpy(
        _problem(robust_case, {"Target": [], "Body": [on_body]})._objective_functions(x)
    )
    np.testing.assert_allclose(f, [f_target[0], f_body[0]], rtol=1e-6)


def test_composite_term_reads_both_polo_instances(robust_case):
    on_target = NTCPPOLO(priority=1.0, risk_voi_name="Target", robustness="COWC")
    on_body = NTCPPOLO(priority=2.0, risk_voi_name="Body", robustness="COWC")
    prob = _problem(robust_case, {"Target": [on_target], "Body": [on_body]})

    assert len(prob._terms) == 1
    keys = prob._terms[0].quantities
    assert len(keys) == 2 and all(key.startswith("polo[") for key in keys)
    # One cache row per quantity, both belonging to the single term.
    assert prob._q_cache_index == [{keys[0]: 0, keys[1]: 0}]

    x = _fluence(prob)
    assert to_numpy(prob._objective_functions(x))[0] > 0
    _assert_jacobian_matches_finite_differences(prob, x)

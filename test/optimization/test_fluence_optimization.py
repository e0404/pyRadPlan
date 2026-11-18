"""Tests for the PlanResult returned by ``fluence_optimization``."""

import numpy as np
import SimpleITK as sitk

from pyRadPlan import fluence_optimization
from pyRadPlan.result import PlanResult


def _optimize(case):
    pln, ct, cst, stf, dij = case
    pln.prop_opt = {"solver": "scipy", "display": False, "max_iter": 5}
    return fluence_optimization(ct, cst, stf, dij, pln)


def test_returns_plan_result(small_proton_case):
    result = _optimize(small_proton_case)

    assert isinstance(result, PlanResult)
    assert result.fluence.shape == (small_proton_case[4].total_num_of_bixels,)
    assert isinstance(result.physical_dose.values, sitk.Image)
    assert result.used_optimizer == "scipy"
    assert "num_iter" in result.info


def test_total_dose_convention_scales_the_result(small_proton_case):
    pln, ct, cst, stf, dij = small_proton_case
    pln.num_of_fractions = 30
    pln.dose_convention = "total"

    result = _optimize(small_proton_case)

    per_fraction = dij.compute_result_ct_grid(result.fluence)
    np.testing.assert_allclose(
        sitk.GetArrayFromImage(result.physical_dose.values),
        30 * sitk.GetArrayFromImage(per_fraction.physical_dose.values),
        rtol=1e-5,
    )


def test_solver_info_is_reduced_to_scalars():
    """Arrays and solver objects of the raw solver output are not kept on the result."""
    from scipy.optimize import minimize

    from pyRadPlan.optimization._fluence_optimization import _summarize_solver_info

    raw = dict(
        minimize(lambda x: float((x**2).sum()), np.ones(3), jac=lambda x: 2 * x, method="L-BFGS-B")
    )
    raw["num_iter"] = np.int64(raw["nit"])

    summary = _summarize_solver_info(raw)

    assert {"fun", "nit", "status", "message", "success", "num_iter"} <= set(summary)
    assert not {"x", "jac", "hess_inv"} & set(summary)
    assert all(isinstance(v, (bool, int, float, str)) for v in summary.values())
    assert type(summary["num_iter"]) is int
    assert _summarize_solver_info(None) == {}

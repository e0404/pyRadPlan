"""Plan.dose_convention: objective scaling in the planning problem."""

import pytest

from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf


@pytest.fixture
def proton_case(test_data_protons_raw):
    tmp = test_data_protons_raw
    pln = validate_pln(tmp["pln"])
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    for voi in cst.vois:
        voi.objectives = []
    cst.vois[0].objectives = [SquaredDeviation(d_ref=60.0, priority=1.0)]
    cst.vois[1].objectives = [SquaredOverdosing(d_max=30.0, priority=1.0)]
    return pln, ct, cst, dij, stf


def _collected_objectives(pln, ct, cst, dij, stf):
    prob = NonLinearFluencePlanningProblem(pln)
    prob._ct, prob._cst, prob._dij, prob._stf = ct, cst, dij, stf
    prob._initialize()
    return [obj for _, objs in prob._objective_list for obj in objs]


def test_per_fraction_leaves_objective_doses_untouched(proton_case):
    pln, ct, cst, dij, stf = proton_case
    pln.num_of_fractions = 30
    pln.dose_convention = "per_fraction"
    objs = _collected_objectives(pln, ct, cst, dij, stf)
    assert objs[0].d_ref == pytest.approx(60.0)
    assert objs[1].d_max == pytest.approx(30.0)


def test_total_scales_objective_doses_by_fractions(proton_case):
    pln, ct, cst, dij, stf = proton_case
    pln.num_of_fractions = 30
    pln.dose_convention = "total"
    objs = _collected_objectives(pln, ct, cst, dij, stf)
    assert objs[0].d_ref == pytest.approx(2.0)
    assert objs[1].d_max == pytest.approx(1.0)
    # the user's objectives in the structure set are not modified
    assert cst.vois[0].objectives[0].d_ref == pytest.approx(60.0)
    assert cst.vois[1].objectives[0].d_max == pytest.approx(30.0)

    # a second run scales from the original values again, not from the scaled ones
    objs = _collected_objectives(pln, ct, cst, dij, stf)
    assert objs[0].d_ref == pytest.approx(2.0)


def test_legacy_conversion_rejects_explicit_nonphysical_quantity(proton_case):
    pln, ct, cst, dij, stf = proton_case
    cst.vois[0].objectives = [SquaredDeviation(d_ref=4.0, priority=1.0, quantity="effect")]

    with pytest.raises(
        ValueError,
        match="VOI .* objective 1 .* quantity 'effect'.*convert_dose_objectives=False",
    ):
        _collected_objectives(pln, ct, cst, dij, stf)


def test_legacy_conversion_accepts_explicit_default_quantity(proton_case):
    pln, ct, cst, dij, stf = proton_case
    dij.alpha_dose = dij.physical_dose.copy()
    dij.sqrt_beta_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredDeviation(d_ref=4.0, priority=1.0, quantity="rbe_x_dose")]

    objs = _collected_objectives(pln, ct, cst, dij, stf)

    assert objs[0].quantity == "rbe_x_dose"


def test_disabled_legacy_conversion_preserves_literal_quantities(proton_case):
    pln, ct, cst, dij, stf = proton_case
    pln.prop_opt["convert_dose_objectives"] = False
    dij.alpha_dose = dij.physical_dose.copy()
    dij.sqrt_beta_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredDeviation(d_ref=4.0, priority=1.0, quantity="effect")]

    objs = _collected_objectives(pln, ct, cst, dij, stf)

    assert [obj.quantity for obj in objs] == ["effect", "physical_dose"]
    assert cst.vois[0].objectives[0].quantity == "effect"

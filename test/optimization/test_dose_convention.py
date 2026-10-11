"""Plan.dose_convention: objective scaling in the planning problem."""

import pytest

from pyRadPlan.core import WarningReport
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


def _collect_with_warnings(pln, ct, cst, dij, stf):
    """Collected objective quantities and the messages of the emitted warning reports."""
    prob = NonLinearFluencePlanningProblem(pln)
    prob._ct, prob._cst, prob._dij, prob._stf = ct, cst, dij, stf
    reports = []
    prob.add_report_observer(reports.append)
    prob._initialize()
    quantities = [obj.quantity for _, objs in prob._objective_list for obj in objs]
    return quantities, [r.message for r in reports if isinstance(r, WarningReport)]


def test_legacy_conversion_silent_without_biological_model(proton_case, caplog):
    pln, ct, cst, dij, stf = proton_case

    quantities, warnings = _collect_with_warnings(pln, ct, cst, dij, stf)

    assert quantities == ["physical_dose", "physical_dose"]
    assert warnings == []
    assert "physical-dose objective" not in caplog.text


def test_legacy_conversion_converts_physical_dose_and_warns(proton_case, caplog):
    pln, ct, cst, dij, stf = proton_case
    dij.bio_model = "constant_rbe"

    quantities, warnings = _collect_with_warnings(pln, ct, cst, dij, stf)

    assert quantities == ["rbe_x_dose", "rbe_x_dose"]
    assert len(warnings) == 1
    assert "2 physical-dose objective(s) were converted to 'rbe_x_dose'" in warnings[0]
    assert warnings[0] in caplog.text
    # the user's objectives in the structure set keep their quantity
    assert cst.vois[0].objectives[0].quantity == "physical_dose"


@pytest.mark.parametrize("biological", ["rbe_x_dose", "effect"])
def test_legacy_conversion_keeps_physical_dose_next_to_biological_objectives(
    proton_case, caplog, biological
):
    pln, ct, cst, dij, stf = proton_case
    dij.bio_model = "constant_rbe"
    dij.alpha_dose = dij.physical_dose.copy()
    dij.sqrt_beta_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredDeviation(d_ref=4.0, priority=1.0, quantity=biological)]

    quantities, warnings = _collect_with_warnings(pln, ct, cst, dij, stf)

    assert quantities == [biological, "physical_dose"]
    assert len(warnings) == 1
    assert "1 physical-dose objective(s) are not converted" in warnings[0]
    assert repr([biological]) in warnings[0]
    assert warnings[0] in caplog.text


def test_legacy_conversion_uses_non_biological_quantities_literally(proton_case):
    pln, ct, cst, dij, stf = proton_case
    dij.bio_model = "constant_rbe"
    dij.let_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredOverdosing(d_max=2.0, priority=1.0, quantity="let_dose")]

    quantities, warnings = _collect_with_warnings(pln, ct, cst, dij, stf)

    assert quantities == ["let_dose", "rbe_x_dose"]
    assert len(warnings) == 1
    assert "were converted" in warnings[0]


def test_disabled_legacy_conversion_preserves_literal_quantities(proton_case):
    pln, ct, cst, dij, stf = proton_case
    pln.prop_opt["convert_dose_objectives"] = False
    dij.alpha_dose = dij.physical_dose.copy()
    dij.sqrt_beta_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredDeviation(d_ref=4.0, priority=1.0, quantity="effect")]

    objs = _collected_objectives(pln, ct, cst, dij, stf)

    assert [obj.quantity for obj in objs] == ["effect", "physical_dose"]
    assert cst.vois[0].objectives[0].quantity == "effect"

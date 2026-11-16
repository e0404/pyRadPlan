"""Plan.dose_convention: objective scaling in the planning problem."""

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.core import WarningReport
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.optimization.objectives import (
    DoseUniformity,
    SquaredDeviation,
    SquaredMimicking,
    SquaredOverdosing,
)
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.plan import validate_pln
from pyRadPlan.quantities import QUANTITIES, Dose
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


def _reference_image(grid, value, as_sitk):
    array = np.full(tuple(grid.dimensions[::-1]), value, dtype=np.float32)
    if not as_sitk:
        return (array, grid)
    image = sitk.GetImageFromArray(array)
    image.SetOrigin(grid.origin)
    image.SetSpacing(grid.resolution_vector)
    image.SetDirection(grid.direction.ravel())
    return image


@pytest.mark.parametrize("as_sitk", [True, False], ids=["sitk", "array_grid"])
@pytest.mark.parametrize(("convention", "expected"), [("per_fraction", 60.0), ("total", 2.0)])
def test_dose_convention_scales_image_references(proton_case, as_sitk, convention, expected):
    pln, ct, cst, dij, stf = proton_case
    pln.num_of_fractions = 30
    pln.dose_convention = convention
    d_ref = _reference_image(dij.dose_grid, 60.0, as_sitk)
    cst.vois[0].objectives = [SquaredMimicking(d_ref=d_ref)]

    objs = _collected_objectives(pln, ct, cst, dij, stf)

    cached = to_numpy(objs[0]._resampled_image_reference_cache["d_ref"])
    assert cached.size > 0
    assert np.allclose(cached, expected)
    values = cached + 1.0
    assert np.allclose(to_numpy(objs[0]._image_reference("d_ref", values)), expected)

    # the user's objective keeps its image and gets no (scaled) evaluation cache
    original = cst.vois[0].objectives[0]
    if as_sitk:
        assert original.d_ref is d_ref
    else:
        assert original.d_ref[0] is d_ref[0]
    assert original._resampled_image_reference_cache == {}
    source = sitk.GetArrayViewFromImage(d_ref) if as_sitk else d_ref[0]
    assert np.allclose(source, 60.0)


def test_image_reference_normalization_requires_preprocessing(proton_case):
    _, _, _, dij, _ = proton_case
    obj = SquaredMimicking(d_ref=_reference_image(dij.dose_grid, 60.0, True))

    with pytest.raises(ValueError, match="preprocess_image_reference_parameters"):
        obj.normalize_to_fraction_num(30)


@pytest.mark.parametrize(
    ("quantity", "expected"),
    [
        ("physical_dose", 6.0 / 30),
        ("rbe_x_dose", 6.0 / 30),
        ("effect", 6.0 / 30),
        ("alpha_dose", 6.0 / 30),
        ("sqrt_beta_dose", 6.0 / np.sqrt(30)),
        ("let_dose", 6.0 / 30),
        ("let", 6.0),
    ],
)
def test_total_normalization_follows_quantity_fraction_scaling(proton_case, quantity, expected):
    pln, ct, cst, dij, stf = proton_case
    pln.num_of_fractions = 30
    pln.dose_convention = "total"
    pln.prop_opt["convert_dose_objectives"] = False
    dij.bio_model = "constant_rbe"
    dij.alpha_dose = dij.physical_dose.copy()
    dij.sqrt_beta_dose = dij.physical_dose.copy()
    dij.let_dose = dij.physical_dose.copy()
    cst.vois[0].objectives = [SquaredOverdosing(d_max=6.0, quantity=quantity)]

    objs = _collected_objectives(pln, ct, cst, dij, stf)

    assert objs[0].d_max == pytest.approx(expected)
    assert cst.vois[0].objectives[0].d_max == pytest.approx(6.0)


@pytest.mark.parametrize("identifier", sorted(QUANTITIES))
def test_normalization_inverts_result_scaling(identifier):
    """A total-course reference normalized to one fraction reports back as itself."""
    obj = SquaredOverdosing(d_max=6.0, quantity=identifier).normalize_to_fraction_num(30)
    reported = Dij._scale_result_to_fractions({identifier: obj.d_max}, 30)[identifier]
    assert reported == pytest.approx(6.0)


def test_undeclared_quantity_fraction_scaling_fails_for_total(monkeypatch):
    class Custom(Dose):
        identifier = "custom_dose"
        fraction_exponent = None

    monkeypatch.setitem(QUANTITIES, Custom.identifier, Custom)
    obj = SquaredOverdosing(d_max=6.0, quantity=Custom.identifier)

    assert obj.model_copy().normalize_to_fraction_num(1).d_max == pytest.approx(6.0)
    with pytest.raises(ValueError, match="fraction_exponent"):
        obj.normalize_to_fraction_num(30)
    # without reference parameters there is nothing to convert
    DoseUniformity(quantity=Custom.identifier).normalize_to_fraction_num(30)


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


@pytest.mark.parametrize("partial_lq", [False, True])
def test_legacy_conversion_skipped_when_dij_cannot_provide_rbe_x_dose(
    proton_case, caplog, partial_lq
):
    """An alpha/beta model without (complete) LQ matrices cannot yield rbe_x_dose."""
    pln, ct, cst, dij, stf = proton_case
    dij.bio_model = "kernel_based_lq"
    if partial_lq:
        dij.alpha_dose = dij.physical_dose.copy()

    quantities, warnings = _collect_with_warnings(pln, ct, cst, dij, stf)

    assert quantities == ["physical_dose", "physical_dose"]
    assert warnings == []
    assert "physical-dose objective" not in caplog.text


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

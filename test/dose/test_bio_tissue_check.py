"""The dose engine runs BiologicalModel.check_tissues: both always agree."""

import logging
import math

import numpy as np
import pytest

from pyRadPlan import calc_dose_influence
from pyRadPlan.biology.models import BioModelUnavailableError, create_bio_model
from pyRadPlan.core import WarningReport, observe_reports
from pyRadPlan.dose.engines import DoseEngineBase
from pyRadPlan.biology.tissues import (
    LEMTissueProperties,
    MKMTissueProperties,
    TissueProperties,
)

MKM_OXYGEN = MKMTissueProperties(
    model="mMKM", r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615
)
LEM_OXYGEN = LEMTissueProperties(alpha_x=0.5, beta_x=0.05, d_t=30.0, r_nucleus=5.0)

CASES = [
    # (radiation mode, model, tissue of every structure, check reports errors)
    ("carbon", "alpha_beta_kernels", TissueProperties(alpha_x=0.1, beta_x=0.05), False),
    ("carbon", "alpha_beta_kernels", TissueProperties(alpha_x=0.2, beta_x=0.05), True),
    ("carbon", "alpha_beta_kernels", LEMTissueProperties(alpha_x=0.5, version="IV"), False),
    # Dt / rNucleus the carbon LEM IV classes leave undeclared are not compared
    (
        "carbon",
        "alpha_beta_kernels",
        LEMTissueProperties(alpha_x=0.5, version="IV", d_t=30.0, r_nucleus=5.0),
        False,
    ),
    # Another LEM version, matching by its parameters: a warning only
    ("carbon", "alpha_beta_kernels", LEM_OXYGEN, False),
    ("carbon", "zstar_kernels", MKM_OXYGEN, True),
    ("oxygen", "alpha_beta_kernels", LEM_OXYGEN, False),
    ("oxygen", "alpha_beta_kernels", MKM_OXYGEN, True),
    ("oxygen", "zstar_kernels", MKM_OXYGEN, False),
    ("oxygen", "zstar_kernels", MKM_OXYGEN.model_copy(update={"model": "MKM"}), False),
    ("oxygen", "zstar_kernels", MKM_OXYGEN.model_copy(update={"beta_0": math.nan}), True),
    ("oxygen", "LEM", LEM_OXYGEN.model_copy(update={"version": "II"}), False),
    ("oxygen", "zstar_kernels", LEM_OXYGEN, True),
    ("protons", "MCN", TissueProperties(alpha_x=0.1, beta_x=0.0), True),
    ("protons", "WED", TissueProperties(alpha_x=0.1, beta_x=0.05), False),
]


def _set_tissue(cst, tissue):
    for i, voi in enumerate(cst.vois):
        cst.vois[i] = voi.model_copy(
            update={
                "tissue_properties": tissue,
                "alpha_x": tissue.alpha_x,
                "beta_x": tissue.beta_x,
            }
        )


@pytest.mark.parametrize(("mode", "model", "tissue", "has_errors"), CASES)
def test_requested_bio_dose_raises_exactly_when_the_check_reports_errors(
    request, mode, model, tissue, has_errors
):
    pln, ct, cst, stf, _dij, _result = request.getfixturevalue(f"test_data_{mode}")
    pln.bio_model = model
    pln.prop_dose_calc["calc_bio_dose"] = True
    _set_tissue(cst, tissue)

    machine = DoseEngineBase.load_machine(pln.radiation_mode, pln.machine)
    issues = create_bio_model(model).check_tissues(machine, cst)
    assert bool([issue for issue in issues if issue.is_error]) is has_errors

    if has_errors:
        with pytest.raises(BioModelUnavailableError) as info:
            calc_dose_influence(ct, cst, stf, pln)
        for issue in issues:
            assert issue.message in str(info.value)
    else:
        dij = calc_dose_influence(ct, cst, stf, pln)
        assert dij.bio_model == create_bio_model(model)
        assert dij.alpha_dose is not None


@pytest.mark.parametrize(
    ("mode", "model", "tissue"),
    [(mode, model, tissue) for mode, model, tissue, has_errors in CASES if has_errors],
)
def test_automatic_bio_dose_falls_back_to_physical_dose_on_errors(request, mode, model, tissue):
    pln, ct, cst, stf, _dij, _result = request.getfixturevalue(f"test_data_{mode}")
    pln.bio_model = model
    _set_tissue(cst, tissue)

    dij = calc_dose_influence(ct, cst, stf, pln)

    assert dij.bio_model is None
    assert dij.alpha_dose is None


def test_invalid_reference_always_raises(test_data_carbon):
    pln, ct, cst, stf, _dij, _result = test_data_carbon
    pln.bio_model = "alpha_beta_kernels"
    _set_tissue(cst, TissueProperties(alpha_x=0.1, beta_x=-0.05))

    issues = create_bio_model("alpha_beta_kernels").check_tissues(None, cst)
    assert {issue.kind for issue in issues} == {"invalid_reference"}
    with pytest.raises(ValueError, match="beta_x must be finite and non-negative"):
        calc_dose_influence(ct, cst, stf, pln)


WARNING_CASES = [
    # (radiation mode, model, tissue labelled with another provenance, relabelled tissue)
    ("oxygen", "zstar_kernels", MKM_OXYGEN.model_copy(update={"model": "MKM"}), MKM_OXYGEN),
    ("carbon", "alpha_beta_kernels", LEM_OXYGEN, LEM_OXYGEN.model_copy(update={"version": "IV"})),
]


def _effect(dij):
    weights = np.ones(dij.alpha_dose.flat[0].shape[1])
    return np.asarray(dij.alpha_dose.flat[0] @ weights)


@pytest.mark.parametrize("calc_bio_dose", ["auto", True])
@pytest.mark.parametrize(("mode", "model", "tissue", "relabelled"), WARNING_CASES)
def test_differing_provenance_label_warns_and_evaluates_the_model(
    request, caplog, mode, model, tissue, relabelled, calc_bio_dose
):
    pln, ct, cst, stf, _dij, _result = request.getfixturevalue(f"test_data_{mode}")
    pln.bio_model = model
    pln.prop_dose_calc["calc_bio_dose"] = calc_bio_dose

    _set_tissue(cst, relabelled)
    reference = calc_dose_influence(ct, cst, stf, pln)

    _set_tissue(cst, tissue)
    machine = DoseEngineBase.load_machine(pln.radiation_mode, pln.machine)
    issues = create_bio_model(model).check_tissues(machine, cst)
    assert issues and {issue.kind for issue in issues} == {"provenance_differs"}

    reports = []
    with caplog.at_level(logging.WARNING), observe_reports(reports.append):
        dij = calc_dose_influence(ct, cst, stf, pln)

    assert dij.bio_model == create_bio_model(model)
    assert dij.alpha_dose is not None
    assert np.any(_effect(reference) > 0)
    np.testing.assert_allclose(_effect(dij), _effect(reference))
    warnings = [r.message for r in reports if isinstance(r, WarningReport)]
    assert len(warnings) == len(issues)
    assert all("is labelled" in message for message in warnings)
    assert any("is labelled" in record.getMessage() for record in caplog.records)

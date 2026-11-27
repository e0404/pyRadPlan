"""POLO in the planning problem: fit-domain warnings and risk structures."""

import numpy as np
import pytest

from pyRadPlan.core import WarningReport
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.optimization.objectives import (
    HellingerConvPOLO,
    LogSumExpConvPOLO,
    NTCPPOLO,
    SquaredDeviation,
)
from pyRadPlan.optimization.problems import NonLinearFluencePlanningProblem
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf


@pytest.fixture
def proton_case(test_data_protons_raw):
    """'Target' (overlap priority 1) lies inside 'Body' (2), which loses those voxels."""
    tmp = test_data_protons_raw
    pln = validate_pln(tmp["pln"])
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    dij = validate_dij(tmp["dij"])
    stf = validate_stf(tmp["stf"])
    # The reference dij places its dose grid at the origin rather than on the ct, which would
    # shift every structure on the dose grid; align them so voxel counts carry over.
    dij.dose_grid.origin = np.array(ct.cube_hu.GetOrigin())
    for voi in cst.vois:
        voi.objectives = []
    vois = {voi.name: voi for voi in cst.vois}
    return pln, ct, cst, dij, stf, vois


def _initialize(pln, ct, cst, dij, stf):
    """The initialized problem and the messages of its warning reports."""
    prob = NonLinearFluencePlanningProblem(pln)
    prob._ct, prob._cst, prob._dij, prob._stf = ct, cst, dij, stf
    reports = []
    prob.add_report_observer(reports.append)
    prob._initialize()
    return prob, [r.message for r in reports if isinstance(r, WarningReport)]


def test_polo_risk_structure_ignores_overlap_handling(proton_case, caplog):
    pln, ct, cst, dij, stf, vois = proton_case
    vois["Target"].objectives = [NTCPPOLO(priority=1.0, risk_voi_name="Body")]

    prob, warnings = _initialize(pln, ct, cst, dij, stf)

    # The risk structure keeps the voxels overlap handling removes from 'Body' as a region.
    polo = next(q for q in prob._quantities if q.identifier == "polo")
    assert np.count_nonzero(np.asarray(polo.binary_risk_mask)) == len(vois["Body"].indices_numpy)
    assert warnings == []


def test_polo_warns_on_region_without_risk_voxels(proton_case, caplog):
    pln, ct, cst, dij, stf, vois = proton_case
    vois["Body"].objectives = [NTCPPOLO(priority=1.0, risk_voi_name="Target")]
    # Only a VOI with objectives takes voxels from the body.
    vois["Target"].objectives = [SquaredDeviation(priority=1.0, d_ref=2.0)]

    _, warnings = _initialize(pln, ct, cst, dij, stf)

    assert len(warnings) == 1
    assert "no voxel of the risk structure 'Target'" in warnings[0]
    assert "overlap_handling='ignore' on 'Body'" in warnings[0]
    assert warnings[0] in caplog.text


def test_polo_warns_on_region_inside_gtv(proton_case):
    pln, ct, cst, dij, stf, vois = proton_case
    vois["Target"].name = "GTV"
    vois["Target"].objectives = [NTCPPOLO(priority=1.0, risk_voi_name="Body")]

    _, warnings = _initialize(pln, ct, cst, dij, stf)

    n_gtv = len(vois["Target"].indices_numpy)
    assert warnings == [
        f"POLO on 'GTV': {n_gtv} voxels lie inside the GTV (GTV), where the model was not fitted."
    ]


def test_polo_warns_on_radiation_mode_other_than_protons(proton_case):
    pln, ct, cst, dij, stf, vois = proton_case
    vois["Target"].objectives = [NTCPPOLO(priority=1.0, risk_voi_name="Body")]
    for beam in stf.beams:
        beam.radiation_mode = "helium"

    _, warnings = _initialize(pln, ct, cst, dij, stf)

    assert warnings == [
        "POLO on 'Target': the plan's radiation mode is helium, but the model was fitted on "
        "proton plans."
    ]


def test_polo_objectives_with_different_risk_structures_are_independent(proton_case):
    pln, ct, cst, dij, stf, vois = proton_case
    # Keep the body's region the same whether or not the target carries an objective.
    vois["Body"].overlap_handling = "keep"

    def evaluate(objectives):
        for name, objs in objectives.items():
            vois[name].objectives = objs
        prob, warnings = _initialize(pln, ct, cst, dij, stf)
        assert warnings == []
        prob.bypass_objective_jacobian = False  # one gradient row per objective
        xp = prob._array_backend
        # A course dose inside the 2 Gy domain of the model, but short of saturating it.
        x = 500.0 * xp.ones(dij.total_num_of_bixels, dtype=xp.float64)
        f = to_numpy(prob._objective_functions(x))
        return prob, f, to_numpy(prob._objective_jacobian(x))

    on_target = NTCPPOLO(priority=1.0, risk_voi_name="Target")
    on_body = NTCPPOLO(priority=2.0, risk_voi_name="Body")
    prob, f, jac = evaluate({"Target": [on_target], "Body": [on_body]})

    polo = {q.risk_voi_name: q for q in prob._quantities if q.identifier == "polo"}
    assert set(polo) == {"Body", "Target"}
    for name, q in polo.items():
        n_risk = np.count_nonzero(np.asarray(q.binary_risk_mask))
        assert n_risk == len(vois[name].indices_numpy)
    # Only the POLO evaluation is duplicated; the dose influence products stay shared.
    for dep in ("physical_dose", "let_dose"):
        assert polo["Body"].dependencies[dep] is polo["Target"].dependencies[dep]

    # Each objective behaves exactly as it does alone.
    _, f_target, jac_target = evaluate({"Target": [on_target], "Body": []})
    _, f_body, jac_body = evaluate({"Target": [], "Body": [on_body]})
    assert np.all(f > 0) and np.all(np.abs(jac).max(axis=1) > 0)
    np.testing.assert_allclose(f, [f_target[0], f_body[0]], rtol=1e-6)
    np.testing.assert_allclose(jac, np.vstack([jac_target, jac_body]), rtol=1e-5)


def test_polo_objectives_forward_their_dose_threshold(proton_case):
    pln, ct, cst, dij, stf, vois = proton_case
    vois["Target"].objectives = [
        NTCPPOLO(risk_voi_name="Body"),
        LogSumExpConvPOLO(risk_voi_name="Body"),
        HellingerConvPOLO(risk_voi_name="Body", dose_threshold=2.0),
    ]

    prob, _ = _initialize(pln, ct, cst, dij, stf)

    # Each default reaches its own instance of the quantity, and so does an override.
    thresholds = sorted(
        (q.identifier, q.dose_threshold) for q in prob._quantities if "polo" in q.identifier
    )
    assert thresholds == [("linear_polo", 0.0), ("linear_polo", 2.0), ("polo", 2.0)]

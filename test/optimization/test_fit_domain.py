"""Fit-domain warnings of quantities, raised by the planning problem."""

import numpy as np
import pytest

from pyRadPlan.core import WarningReport
from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.optimization.objectives import NTCPPOLO
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

    _, warnings = _initialize(pln, ct, cst, dij, stf)

    assert len(warnings) == 1
    assert "no voxel of the risk structure 'Target'" in warnings[0]
    assert "apply_overlap=None on 'Body'" in warnings[0]
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

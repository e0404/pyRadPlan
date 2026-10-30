import pytest

from pyRadPlan.ct import validate_ct
from pyRadPlan.cst import validate_cst
from pyRadPlan.dij import validate_dij
from pyRadPlan.plan import validate_pln
from pyRadPlan.stf import validate_stf


@pytest.fixture
def small_proton_case(test_data_protons_raw):
    """A complete, small optimization case (2000 voxels, 12 bixels) with a precomputed dij.

    Reusing the matRad reference data keeps the whole optimization pipeline in play without
    running a dose calculation. The dij is a truncated reference matrix, so it is only suitable
    for exercising the machinery, not for asserting convergence behaviour.
    """
    tmp = test_data_protons_raw

    pln = validate_pln(tmp["pln"])
    ct = validate_ct(tmp["ct"])
    cst = validate_cst(tmp["cst"], ct=ct)
    stf = validate_stf(tmp["stf"])
    dij = validate_dij(tmp["dij"])

    return pln, ct, cst, stf, dij

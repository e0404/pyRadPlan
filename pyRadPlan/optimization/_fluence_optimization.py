from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.result import PlanResult, validate_result
from .problems import get_problem_from_pln


def fluence_optimization(
    ct: CT, cst: StructureSet, stf: SteeringInformation, dij: Dij, pln: Plan
) -> PlanResult:
    """
    Trigger fluence optimization using the configuration stored in the pln object.

    Parameters
    ----------
    ct : CT
        CT object.
    cst : StructureSet
        StructureSet object.
    stf : SteeringInformation
        SteeringInformation object.
    dij : Dij
        Dij object.
    pln : Plan
        Plan object.

    Returns
    -------
    PlanResult
        Result object containing the requested dose cubes and information.
    """

    _ct = validate_ct(ct)
    _cst = validate_cst(cst)
    _stf = validate_stf(stf)
    _dij = validate_dij(dij)
    _pln = validate_pln(pln)

    planning_prob = get_problem_from_pln(_pln)

    fluence, _result_info = planning_prob.solve(_ct, _cst, _stf, _dij)
    cubes = dij.compute_result_ct_grid(fluence)
    result = validate_result(cubes, fluence=fluence, info=_result_info)
    return result

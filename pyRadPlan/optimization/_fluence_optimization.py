from typing import TypedDict

import numpy as np

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.stf import SteeringInformation, validate_stf

from .problems import get_problem_from_pln


class OptInfo(TypedDict, total=False):
    """Solver-specific optimization information populated during optimization.

    Pass an empty dict to ``fluence_optimization`` to have it filled with
    the following keys after the solve completes:

    Attributes
    ----------
    obj_history : list[float]
        Objective function value at each evaluation.
    num_iter : int
        Number of iterations performed by the solver.
    result_info : dict
        Solver-specific result information (convergence status, iteration
        count, etc.).
    """

    obj_history: list[float]
    num_iter: int
    result_info: dict


def fluence_optimization(
    ct: CT, cst: StructureSet, stf: SteeringInformation, dij: Dij, pln: Plan, **kwargs
) -> np.ndarray:
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

    Additional keyword arguments:

    opt_info : OptInfo, optional
        If provided (e.g. as an empty ``{}``), it will be populated with
        ``obj_history``, ``n_iter``, and ``result_info`` after optimization.

    Returns
    -------
    np.ndarray
        The optimized fluence map.
    """

    _ct = validate_ct(ct)
    _cst = validate_cst(cst)
    _stf = validate_stf(stf)
    _dij = validate_dij(dij)
    _pln = validate_pln(pln)

    planning_prob = get_problem_from_pln(_pln)

    opt_info: OptInfo = kwargs.get("opt_info", None)
    if opt_info is not None:
        planning_prob.obj_history = []

    x, result_info = planning_prob.solve(_ct, _cst, _stf, _dij)

    if opt_info is not None:
        opt_info["obj_history"] = planning_prob.obj_history
        opt_info["result_info"] = result_info
        # Extract iteration count from solver-specific result
        if isinstance(result_info, dict):
            opt_info["num_iter"] = result_info.get("nit", result_info.get("iter_count", 0))

    return x

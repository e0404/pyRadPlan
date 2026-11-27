import logging
from collections.abc import Mapping, Sequence
from typing import Any, Optional, TypedDict, Union

import numpy as np

from pyRadPlan.ct import CT, validate_ct
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.plan import Plan, validate_pln
from pyRadPlan.dij import Dij, validate_dij
from pyRadPlan.stf import SteeringInformation, validate_stf
from pyRadPlan.result import PlanResult
from .problems import get_problem_from_pln

logger = logging.getLogger(__name__)


def _summarize_solver_info(solver_info: Any) -> dict[str, Any]:
    """
    Reduce the raw solver output to its scalar entries.

    Solvers also report arrays and solver objects (the solution, gradients, Hessian
    approximations) that are either stored on the result already or cannot be saved.
    """
    if not isinstance(solver_info, Mapping):
        return {}
    summary: dict[str, Any] = {}
    for key, raw in solver_info.items():
        value = raw
        if isinstance(raw, np.generic) or (isinstance(raw, np.ndarray) and raw.ndim == 0):
            value = raw.item()
        elif isinstance(raw, bytes):
            value = raw.decode(errors="replace")
        if isinstance(value, (bool, int, float, str)):
            summary[str(key)] = value
    return summary


class OptInfo(TypedDict, total=False):
    """Solver-specific optimization information populated during optimization.

    Pass an empty dict to ``fluence_optimization`` to have it filled with
    the following keys after the solve completes:

    Attributes
    ----------
    obj_history : list[float]
        Objective function value at each evaluation. Absent if the planning problem does not
        record one (``records_obj_history``), rather than empty.
    num_iter : int
        Number of iterations performed by the solver. Absent if the solver does not report
        one under ``result_info["num_iter"]``.
    result_info : dict
        Solver-specific result information (convergence status, iteration
        count, etc.).
    objective_info : list[dict]
        Auxiliary information reported by the individual objectives (e.g. NTCP histories from
        outcome objectives). Absent when no objective reports any.
    """

    obj_history: list[float]
    num_iter: int
    result_info: dict
    objective_info: list[dict]


def fluence_optimization(  # noqa: PLR0913
    ct: CT,
    cst: StructureSet,
    stf: SteeringInformation,
    dij: Dij,
    pln: Plan,
    *,
    opt_info: Optional[OptInfo] = None,
    scenarios: Union[bool, Sequence[int]] = False,
    initial_fluence: Optional[np.ndarray] = None,
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
    opt_info : OptInfo, optional
        If provided (e.g. as an empty ``{}``), it will be populated with
        ``obj_history``, ``num_iter`` and ``result_info`` after optimization.
    scenarios : bool or sequence of int, default False
        Scenarios to also report the optimized quantities for, as
        ``result.scenario_quantities`` (see :meth:`Dij.compute_result_ct_grid`).
        ``True`` selects every scenario of a dij with several scenarios; by default
        only the nominal scenario is reported.
    initial_fluence : np.ndarray, optional
        Fluence (one weight per bixel of *dij*) the solver starts from, e.g. the result of an
        earlier optimization with the same beams. Default: all zeros.

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

    records_history = getattr(planning_prob, "records_obj_history", False)

    if opt_info is not None:
        # The caller owns this dictionary and may reuse it across solves. The optional keys are
        # dropped up front so a key this solve does not fill cannot be read as its result - and
        # so a solve that raises leaves no stale values behind either.
        for key in ("obj_history", "num_iter", "result_info", "objective_info"):
            opt_info.pop(key, None)

    if opt_info is not None and records_history:
        planning_prob.obj_history = []

    # Only passed when given, so problems overriding solve() without x0 keep working
    start = {} if initial_fluence is None else {"x0": initial_fluence}
    fluence, result_info = planning_prob.solve(_ct, _cst, _stf, _dij, **start)

    if opt_info is not None:
        # Only a problem that declares it actually fills the history; reporting an empty list
        # for one that does not would be indistinguishable from a solve without evaluations.
        if records_history:
            opt_info["obj_history"] = planning_prob.obj_history
        else:
            logger.warning(
                "Planning problem '%s' does not record an objective history, so "
                "opt_info['obj_history'] is left unset.",
                getattr(planning_prob, "short_name", type(planning_prob).__name__),
            )
        opt_info["result_info"] = result_info
        # Every solver normalizes its iteration count under "num_iter"; a third-party solver
        # that does not is left without the key rather than reported as zero iterations.
        if isinstance(result_info, dict) and "num_iter" in result_info:
            opt_info["num_iter"] = int(result_info["num_iter"])
        # Left unset when no objective reports anything, like the other optional keys.
        collect = getattr(planning_prob, "collect_objective_info", None)
        objective_info = collect() if callable(collect) else None
        if objective_info:
            opt_info["objective_info"] = objective_info

    result = _dij.compute_result_ct_grid(
        fluence, num_of_fractions=_pln.result_dose_factor, scenarios=scenarios
    )
    result.info = _summarize_solver_info(result_info)
    # solve() resolves the solver from its identifier, so read it back afterwards.
    solver = getattr(planning_prob, "solver", None)
    if solver is not None:
        result.used_optimizer = getattr(solver, "short_name", str(solver))
    return result

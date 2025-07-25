"""Factory methods to manage available Problems."""

import warnings
import logging
from typing import Union, Type, Optional

from pyRadPlan.plan import Plan, validate_pln
from ._optiprob import PlanningProblem

PROBLEMS: dict[str, Type[PlanningProblem]] = {}

logger = logging.getLogger(__name__)


def register_problem(prob_cls: Type[PlanningProblem]) -> None:
    """
    Register a new problem.

    Parameters
    ----------
    prob_cls : type
        A PlanningProblem class.
    """
    if not issubclass(prob_cls, PlanningProblem):
        raise ValueError("PlanningProblem must be a subclass of PlanningProblem.")

    if prob_cls.short_name is None:
        raise ValueError("PlanningProblem must have a 'short_name' attribute.")

    if prob_cls.name is None:
        raise ValueError("PlanningProblem must have a 'name' attribute.")

    prob_name = prob_cls.short_name
    if prob_name in PROBLEMS:
        warnings.warn(f"PlanningProblem '{prob_name}' is already registered.")
    else:
        PROBLEMS[prob_name] = prob_cls


def get_available_problems(
    pln: Optional[Union[Plan, str]] = None,
) -> dict[str, Type[PlanningProblem]]:
    """
    Get a list of available planning problems based on the plan.

    Parameters
    ----------
    pln : Plan, optional
        A Plan object.

    Returns
    -------
    list
        A list of available planning problems.
    """

    if pln is None:
        return PROBLEMS
    pln = validate_pln(pln)
    return {
        name: cls
        for name, cls in PROBLEMS.items()
        if pln.radiation_mode in cls.possible_radiation_modes
    }


def get_problem(problem_desc: Union[str, dict, PlanningProblem]) -> Type[PlanningProblem]:
    """
    Return a problem instance based on a descriptive parameter.

    Parameters
    ----------
    problem_desc : Union[str, dict, PlanningProblem, Plan]
        A string with the problem name, a dictionary with the problem configuration or a problem
        instance

    Returns
    -------
    PlanningProblem
        A problem instance
    """
    if isinstance(problem_desc, str):
        problem = PROBLEMS[problem_desc]()
    elif isinstance(problem_desc, dict):
        raise NotImplementedError(
            "PlanningProblem configuration from dictionary not implemented yet."
        )
    elif isinstance(problem_desc, PlanningProblem):
        problem = problem_desc
    else:
        raise ValueError(f"Invalid problem description: {problem_desc}")

    return problem


def get_problem_from_pln(pln: Union[Plan, dict]) -> Type[PlanningProblem]:
    """
    Get the appropriate problem based on the plan.

    Parameters
    ----------
    pln : Plan
        A Plan object.

    Returns
    -------
    Problem
        A Planning Problem object.
    """
    pln = validate_pln(pln)
    problems = get_available_problems(pln)

    if len(problems) <= 0:
        raise ValueError(
            f"No PlanningProblem available for radiation mode '{pln.radiation_mode}'."
        )

    problem_names = list(problems.keys())

    # Did the user provide an problem in the pln?
    if isinstance(pln.prop_opt, PlanningProblem):
        # The user provided an problem object, so lets use it
        # but warn the user if it is not in the available problems
        problem_name = pln.prop_opt.short_name
        if problem_name not in problems:
            warnings.warn(f"Engine '{problem_name}' seems not to be valid for Plan setup.")
        return pln.prop_opt

    if isinstance(pln.prop_opt, dict):
        # The user provided a dictionary with problem parameters,
        # so we need to find the problem name
        if "problem" in pln.prop_opt:
            if pln.prop_opt["problem"] in problems:
                return problems[pln.prop_opt["problem"]](pln)
            warnings.warn(f"Engine '{pln.prop_opt['problem']}' not available for Plan.")

        # If no problem name was found, we choose the first as default
        logger.warning(
            "No PlanningProblem specified in Plan. Using first available problem %s.",
            problem_names[0],
        )
        return problems[problem_names[0]](pln)

    raise ValueError(f"No problem available for radiation mode '{pln.radiation_mode}'.")

"""Factory methods to manage available solver implementations."""

import warnings
import logging
from typing import Union, Type
from ._base_solvers import SolverBase

SOLVERS = {}

logger = logging.getLogger(__name__)


def register_solver(solver_cls: Type[SolverBase]) -> None:
    """
    Register a new solver.

    Parameters
    ----------
    solver_cls : type
        A Dose Solver class.
    """
    if not issubclass(solver_cls, SolverBase):
        raise ValueError("Solver must be a subclass of SolverBase.")

    if solver_cls.short_name is None:
        raise ValueError("Solver must have a 'short_name' attribute.")

    if solver_cls.name is None:
        raise ValueError("Solver must have a 'name' attribute.")

    solver_name = solver_cls.short_name
    if solver_name in SOLVERS:
        warnings.warn(f"Solver '{solver_name}' is already registered.")
    else:
        SOLVERS[solver_name] = solver_cls


def get_available_solvers() -> dict[str, Type[SolverBase]]:
    """
    Get a list of available solvers based on the plan.

    Returns
    -------
    list
        A list of available solvers.
    """
    return SOLVERS


def get_solver(solver_desc: Union[str, dict, SolverBase]):
    """
    Return a solver instance based on a descriptive parameter.

    Parameters
    ----------
    solver_desc : Union[str, dict, SolverBase]
        A string with the solver name, a dictionary with the solver configuration or a solver
        instance

    Returns
    -------
    SolverBase
        A solver instance
    """
    if isinstance(solver_desc, str):
        solver = SOLVERS[solver_desc]()
    elif isinstance(solver_desc, dict):
        raise NotImplementedError("Solver configuration from dictionary not implemented yet.")
    elif isinstance(solver_desc, SolverBase):
        solver = solver_desc
    else:
        raise ValueError(f"Invalid solver description: {solver_desc}")

    return solver

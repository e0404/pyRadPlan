"""Factory methods to manage available solver implementations."""

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
        logger.warning("Solver '%s' is already registered.", solver_name)
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
        instance. When a dictionary is given, the ``"name"`` key selects the solver and any other
        keys are assigned as attributes. Dict-typed attributes (e.g. ``options``) are merged into
        the solver's defaults rather than replacing them.

    Returns
    -------
    SolverBase
        A solver instance
    """
    if isinstance(solver_desc, str):
        if solver_desc not in SOLVERS:
            raise ValueError(f"Solver '{solver_desc}' not registered. Available: {list(SOLVERS)}")
        return SOLVERS[solver_desc]()

    if isinstance(solver_desc, SolverBase):
        return solver_desc

    if isinstance(solver_desc, dict):
        cfg = dict(solver_desc)
        name = cfg.pop("name", None)
        if name is None:
            raise ValueError("Solver configuration dictionary must include a 'name' key.")
        if name not in SOLVERS:
            raise ValueError(f"Solver '{name}' not registered. Available: {list(SOLVERS)}")
        solver = SOLVERS[name]()
        for key, value in cfg.items():
            if not hasattr(solver, key):
                logger.warning("Property '%s' not found on solver '%s'.", key, name)
                continue
            current = getattr(solver, key)
            if isinstance(current, dict) and isinstance(value, dict):
                merged = dict(current)
                merged.update(value)
                setattr(solver, key, merged)
            else:
                setattr(solver, key, value)
        return solver

    raise ValueError(f"Invalid solver description: {solver_desc}")

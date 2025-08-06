"""Optimization solvers for treatment planning problems."""

from ._factory import register_solver, get_available_solvers, get_solver

try:
    from ._ipopt import OptimizerIpopt

    register_solver(OptimizerIpopt)

except ImportError:
    OptimizerIpopt = None

from ._base_solvers import SolverBase, NonLinearOptimizer
from ._scipy_solver import OptimizerSciPy

register_solver(OptimizerSciPy)


__all__ = [
    "OptimizerIpopt",
    "OptimizerSciPy",
    "SolverBase",
    "NonLinearOptimizer",
    "register_solver",
    "get_available_solvers",
    "get_solver",
]

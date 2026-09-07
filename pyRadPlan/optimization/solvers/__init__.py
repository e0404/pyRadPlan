"""Optimization solvers for treatment planning problems."""

import logging

from ...util.openmp import blocked_by_openmp
from ._factory import register_solver, get_available_solvers, get_solver
from ._base_solvers import SolverBase, NonLinearOptimizer
from ._scipy_solver import OptimizerSciPy

logger = logging.getLogger(__name__)

#: Why IPOPT is unavailable in this process, or ``None`` when it can be used.  The
#: ``ipyopt`` wheel vendors its own Intel OpenMP runtime, which aborts the whole
#: process (``OMP: Error #15``) when another one -- PyTorch ships one too -- is
#: already loaded.  The clash only turns fatal once IPOPT opens its first parallel
#: region, i.e. mid-solve, so the solver is not registered at all when it is
#: detected.  ``KMP_DUPLICATE_LIB_OK=TRUE`` overrides this (see
#: :mod:`pyRadPlan.util.openmp`).
IPOPT_DISABLED_REASON = blocked_by_openmp("ipyopt")

if IPOPT_DISABLED_REASON is None:
    try:
        from ._ipopt import OptimizerIpopt

        register_solver(OptimizerIpopt)

    except ImportError:
        OptimizerIpopt = None
        IPOPT_DISABLED_REASON = "ipyopt is not installed"
else:
    OptimizerIpopt = None
    logger.warning(
        "IPOPT is unavailable: %s. Loading it would abort the process once it starts "
        "solving. Set KMP_DUPLICATE_LIB_OK=TRUE before starting Python to use it anyway "
        "(unsafe: it may crash or silently return wrong results).",
        IPOPT_DISABLED_REASON,
    )

register_solver(OptimizerSciPy)


__all__ = [
    "OptimizerIpopt",
    "OptimizerSciPy",
    "IPOPT_DISABLED_REASON",
    "SolverBase",
    "NonLinearOptimizer",
    "register_solver",
    "get_available_solvers",
    "get_solver",
]

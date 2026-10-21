"""ipax solver interface.

``ipax`` is an Array-API-native interior-point optimizer. Its public ``solve``
infers the array backend from ``x0`` and consumes the objective / gradient
callables directly, so it can run entirely on a GPU backend with arrays staying
on device.

When :attr:`OptimizerIpax.force_cpu_inner_solve` is set, ipax instead runs its
interior-point linear algebra on the CPU (NumPy) while each objective / gradient
evaluation is round-tripped to the compute backend (e.g. GPU) and back, mirroring
how :class:`OptimizerSciPy` / :class:`OptimizerIpopt` handle CPU-only solvers.

ipax is a declared dependency; its sparse-direct accelerations (``ipax[sparse]``)
are optional. The import guard in the package ``__init__`` only keeps a
stripped-down environment importable.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Annotated, Any, ClassVar

import array_api_compat
import numpy as np

import ipax

from ...core import AlgorithmParameterMetadata, xp_utils
from ...core.xp_utils.typing import Array
from ._base_solvers import NonLinearOptimizer

logger = logging.getLogger(__name__)

#: Statuses that leave a usable iterate: a certified solution, a point certified
#: within the acceptable band, or a user-requested stop.
_SUCCESS_STATES = frozenset({ipax.Status.OPTIMAL, ipax.Status.ACCEPTABLE, ipax.Status.STOPPED})


class OptimizerIpax(NonLinearOptimizer):
    """ipax interior-point solver interface.

    Attributes
    ----------
    options : ipax.Options
        Solver options. ``max_iter`` and ``max_time`` are overwritten from the
        generic :attr:`max_iter` / :attr:`max_time` attributes at solve time;
        everything else (Hessian mode, linear-solve route, μ schedule, the
        RT-scale acceptable-stopping band, …) is taken from this object. A
        dictionary given for it through :func:`get_solver` overrides only the
        entries it names. ipax terminates on scaled KKT residuals and a relative
        objective change, so :attr:`abs_obj_tol` is unused.
    """

    name: ClassVar[str] = "Array-API Interior Point (ipax)"
    short_name: ClassVar[str] = "ipax"
    gpu_compatible: ClassVar[bool] = True

    allow_keyboard_cancel = True

    options: Annotated[
        ipax.Options,
        AlgorithmParameterMetadata(hidden_entries=("max_iter", "max_time")),
    ]
    #: ipax terminates on scaled KKT residuals and a relative objective change.
    abs_obj_tol: Annotated[float, AlgorithmParameterMetadata(configurable=False)]
    #: Run ipax's interior-point linear algebra in NumPy, converting each
    #: objective / gradient return to CPU before feeding it to the solver.
    #: Defaults to ``False``, so ipax solves natively on the compute backend.
    #: Note this also keeps the sparse route on the host factorization; only
    #: the native path reaches cuDSS.
    force_cpu_inner_solve: bool

    def __init__(self):
        super().__init__()

        self.force_cpu_inner_solve = False

        # Linear solver options, used only once ``linsolve="auto"`` picks the
        # Krylov route (from ~10k bixels). On a bound-only L-BFGS block both
        # ``jacobi`` and ``auto`` apply ipax's exact Woodbury inverse directly
        # (reported as ``pc=lbfgs-exact``) instead of ~30 CG iterations;
        # ``auto`` additionally picks the L-BFGS preconditioner once inequality
        # rows add a Gram term.
        krylov = ipax.KrylovOptions(
            method="cg",
            preconditioner="auto",
        )

        lbfgs = ipax.LBFGSOptions(memory=10, seed_formula="scalar1")

        # Interpolated backtracking: each rejected trial costs a dose
        # projection, and on fluence maps it roughly halves the objective
        # evaluations at the same optimum compared to plain halving.
        line_search = ipax.LineSearchOptions(backtrack_interpolation=True)

        # Acceptable-stopping band, loosened from ipax's 1e-6/15-iteration
        # default: the tail is not worth the dose-matrix evaluations. The dual
        # band is the loosest: on fluence maps the scaled dual infeasibility
        # plateaus at 1e-3..1e-2 once the objective has settled (gradients are
        # scaled to at most 100, so 1e-2 is 1e-4 of the initial gradient), and
        # a 1e-3 band left the solve stalling in failed line searches until
        # max_iter (TG119: ACCEPTABLE after ~220 iterations at a 1e-5 relative
        # objective instead of 500).
        #
        # ``f_tol`` is an absolute objective *level* test (|f| <= tol), so it
        # stays off. ``f_rel_change_tol`` guards the looser dual band against
        # stopping while the objective still moves; enabling it declines
        # ipax's terminal KKT certificate on max_iter/stalled exits. ipax
        # compares relative objective changes, so :attr:`abs_obj_tol` is not
        # consumed here.
        acceptable = ipax.AcceptableStoppingOptions(
            constr_viol_tol=1e-3,
            dual_inf_tol=1e-2,
            compl_inf_tol=1e-3,
            f_rel_change_tol=1e-4,
            n_iter=5,
        )

        # L-BFGS Hessian + auto linear-solve route is the right default for the
        # large, sparse fluence-map problems pyRadPlan throws at it.
        self.options = ipax.Options(
            acceptable=acceptable,
            hessian="lbfgs",
            mu_schedule="adaptive",
            globalization="filter",
            linsolve="auto",
            corrections=ipax.CorrectionsOptions(method="gondzio"),
            krylov=krylov,
            lbfgs=lbfgs,
            line_search=line_search,
        )

    def _bound_array(self, xp, b: Any, x0: Array, *, is_lower: bool) -> Array | None:
        """Broadcast a scalar/array bound to a full ``(n,)`` vector for ipax.

        ``±inf`` collapses to ``None`` (ipax's "unbounded" sentinel) so the
        barrier never sees a redundant bound.
        """
        if b is None:
            return None
        # Bounds must share x0's namespace/device. Under force_cpu_inner_solve x0
        # is round-tripped to NumPy, so deriving the device from x0 (rather than
        # self.device, which holds the original backend) keeps them consistent.
        barr = xp.asarray(b, dtype=x0.dtype, device=array_api_compat.device(x0))
        if barr.ndim == 0:
            value = float(barr)
            # Scalar ±inf → unbounded on that side.
            if is_lower and value == -np.inf:
                return None
            if not is_lower and value == np.inf:
                return None
            # The Array API only takes a Python scalar as fill value.
            barr = xp.full_like(x0, value)
        return barr

    def _callback(self, info: ipax.IterationInfo) -> bool:
        """Ipax iteration hook: push status upward and honor cancellation.

        Returning ``True`` requests early termination (ipax → ``Status.STOPPED``).
        """
        record = info.record
        data = {
            "iteration": int(record.iteration),
            "objective": float(record.objective),
            "constraint_violation": float(record.primal_infeasibility),
            "dual_inf": float(record.dual_infeasibility),
            "step": float(record.alpha_primal),
            "kkt_error": float(record.kkt_error),
            "mu": float(record.mu),
        }

        # _emit_status returns True to *continue*; ipax wants True to *stop*.
        cont = self._emit_status(message=f"iteration {data['iteration']}", **data)
        return not cont or self._keyboard_listener.stop_event.is_set()

    def _solve_problem(self, x0: Array) -> tuple[Array, dict]:
        if isinstance(x0, list):
            x0 = np.asarray(x0)

        xp = array_api_compat.array_namespace(x0)

        objective = self.objective
        gradient = self.gradient
        backend_xp = None
        device = array_api_compat.device(x0)
        bounds = self.bounds

        if self.force_cpu_inner_solve and not array_api_compat.is_numpy_array(x0):
            # The objective/gradient (dij & linear algebra) live on the compute
            # backend (e.g. GPU), but ipax should run its interior-point solve on
            # the CPU. Run ipax in NumPy and round-trip each inner evaluation:
            # copy x onto the backend device, evaluate, copy the result back.
            backend_xp = xp

            def objective(x: np.ndarray) -> np.ndarray:
                xb = xp_utils.from_numpy(backend_xp, x, device=device)
                return xp_utils.to_numpy(self.objective(xb))

            def gradient(x: np.ndarray) -> np.ndarray:
                xb = xp_utils.from_numpy(backend_xp, x, device=device)
                return xp_utils.to_numpy(self.gradient(xb))

            x0 = xp_utils.to_numpy(x0)
            xp = array_api_compat.array_namespace(x0)
            bounds = tuple(
                xp_utils.to_numpy(b) if array_api_compat.is_array_api_obj(b) else b for b in bounds
            )

        lower = self._bound_array(xp, bounds[0], x0, is_lower=True)
        upper = self._bound_array(xp, bounds[1], x0, is_lower=False)

        problem = ipax.FunctionProblem(
            int(x0.shape[0]),
            objective,
            gradient=gradient,
            bounds=(lower, upper),
        )

        options = replace(
            self.options,
            max_iter=self.max_iter,
            max_time=float(self.max_time),
        )

        result = ipax.solve(problem, x0, options=options, callback=self._callback)

        logger.info(
            "ipax finished: status=%s objective=%.6e kkt=%.3e iters=%d route=%s",
            result.status.value,
            result.objective,
            result.kkt_error,
            result.n_iter,
            result.routes.linear_solver if result.routes is not None else result.linear_solver,
        )
        if result.status not in _SUCCESS_STATES:
            logger.warning("ipax did not converge (%s): %s", result.status.value, result.message)

        x = result.x
        if backend_xp is not None:
            # Hand the solution back on the original compute backend/device.
            x = xp_utils.from_numpy(backend_xp, xp_utils.to_numpy(x), device=device)

        result_info = {
            "status": result.status.value,
            "message": result.message,
            "objective": float(result.objective),
            "num_iter": int(result.n_iter),
            "kkt_error": float(result.kkt_error),
            "solve_time": float(result.solve_time),
        }

        return x, result_info

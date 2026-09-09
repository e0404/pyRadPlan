"""SciPy solver Class."""

import logging
from typing import Callable, Union

import array_api_compat
import numpy as np

from ...core.xp_utils.typing import Array

from scipy.optimize import minimize, Bounds

from ._base_solvers import NonLinearOptimizer
from ...core import xp_utils

logger = logging.getLogger(__name__)


class OptimizerSciPy(NonLinearOptimizer):
    """
    SciPy solver configuration class.

    Attributes
    ----------
    options : dict
        Options passed to :func:`scipy.optimize.minimize` by name, including the method's own
        tolerances (``ftol``, ``gtol``, ``xtol``, ...), which SciPy lets a caller set
        independently of one another. An entry here wins over :attr:`abs_obj_tol` for that
        tolerance, as it does in SciPy itself.

        The exception is the method's iteration cap, which is set from :attr:`max_iter` — it is
        the generic knob a planning problem forwards — and a value placed here for it is
        reported and ignored.
    method : Union[str, Callable]
        The solver method
    display : bool, default=True
        Whether to log the objective value at each iteration
    abs_obj_tol : float, default=1e-5
        Objective tolerance, handed to ``minimize(tol=...)``, which maps it onto whichever
        tolerance options the chosen *method* understands and no explicit entry in
        :attr:`options` already covers. Defaults to ``1e-5``, the tolerance this solver
        effectively ran with before, rather than the base class' ``1e-6``.
    """

    name = "SciPy minimize"
    short_name = "scipy"
    gpu_compatible = False

    allow_keyboard_cancel = True

    options: dict[str]
    method: Union[str, Callable]
    display: bool

    #: Default objective tolerance. Deliberately not the base class' 1e-6: it is the tolerance
    #: this solver effectively ran with while `abs_obj_tol` never reached SciPy, so making the
    #: attribute work does not silently tighten every existing plan.
    _DEFAULT_ABS_OBJ_TOL = 1e-5

    def __init__(self):
        # Left empty on purpose: seeding a tolerance here would take precedence over
        # `abs_obj_tol` (as any explicit entry does), leaving the attribute with no effect.
        self.options = {}

        self.method = "L-BFGS-B"
        self.display = True

        self._iter_count = 0

        super().__init__()

        self.abs_obj_tol = self._DEFAULT_ABS_OBJ_TOL

    def _callback(self, intermediate_result):
        # scipy passes either the current iterate (ndarray) or, for solvers that
        # support it, an OptimizeResult carrying the objective value (`.fun`).
        self._iter_count += 1
        data = {"iteration": self._iter_count}
        fun = getattr(intermediate_result, "fun", None)
        if fun is not None:
            data["objective"] = float(fun)

        # SciPy's own `disp` option is deprecated (removal in 1.18) and already silent for
        # L-BFGS-B, so iteration output is emitted here through the logger instead.
        if self.display:
            if fun is not None:
                logger.info("Iteration %d: objective = %.6e", self._iter_count, data["objective"])
            else:
                logger.info("Iteration %d", self._iter_count)

        cont = self._emit_status(message=f"iteration {self._iter_count}", **data)
        if not cont or self._keyboard_listener.stop_event.is_set():
            raise StopIteration("Optimization cancelled by user")

    #: Methods whose iteration cap is not called ``maxiter``. SciPy's TNC takes ``maxfun`` and
    #: reports ``maxiter`` as an unknown option, so it would simply not be capped. Every other
    #: method shipped by :func:`scipy.optimize.minimize` accepts ``maxiter``, including COBYLA,
    #: which maps it onto its own function-evaluation budget.
    _MAX_ITER_OPTIONS = {"tnc": "maxfun"}

    #: Option names :func:`scipy.optimize.minimize` fills in from its ``tol`` argument, across
    #: all methods. It does so with ``setdefault``, so any of these present in ``options`` takes
    #: precedence over :attr:`abs_obj_tol` for that particular tolerance. They are left alone -
    #: SciPy exposes several tolerances per method that a caller may want set independently, and
    #: a single objective tolerance cannot express that - but a collision is reported.
    _TOLERANCE_OPTIONS = frozenset(
        {"ftol", "gtol", "xtol", "xatol", "fatol", "tol", "final_tr_radius", "barrier_tol"}
    )

    def _max_iter_option(self) -> str:
        """Name of the option capping iterations for the configured :attr:`method`."""
        if not isinstance(self.method, str):
            return "maxiter"  # a custom callable receives the options verbatim
        return self._MAX_ITER_OPTIONS.get(self.method.lower(), "maxiter")

    def _effective_options(self) -> dict:
        """Assemble the options for one solve without touching :attr:`options`.

        Working on a copy keeps ``max_iter`` from overwriting a stored iteration cap the caller
        wrote, so what the caller configured stays readable after a solve.
        """
        options = dict(self.options)

        iter_option = self._max_iter_option()
        if iter_option in options:
            logger.warning(
                "SciPy option '%s' in `options` is ignored: the iteration cap is set from the "
                "solver attribute 'max_iter' (%r). Configure the attribute instead.",
                iter_option,
                self.max_iter,
            )
        options[iter_option] = self.max_iter

        # A tolerance the caller set explicitly stays authoritative; saying so keeps a solver
        # whose abs_obj_tol was configured too from looking as though it took effect.
        shadowing = sorted(self._TOLERANCE_OPTIONS.intersection(options))
        if shadowing and self.abs_obj_tol != self._DEFAULT_ABS_OBJ_TOL:
            logger.warning(
                "SciPy option(s) %s in `options` take precedence over the solver attribute "
                "'abs_obj_tol' (%r), which therefore does not apply to them.",
                ", ".join(f"'{option}'" for option in shadowing),
                self.abs_obj_tol,
            )

        return options

    def _solve_problem(self, x0: Array) -> tuple[Array, dict]:
        """
        Solve the problem.

        Parameters
        ----------
        x0 : np.ndarray
            Initial guess for the decision variables.

        Returns
        -------
        result : dict
        """

        options = self._effective_options()
        self._iter_count = 0

        if isinstance(x0, list):
            x0 = np.asarray(x0)

        xp = array_api_compat.array_namespace(x0)

        x0 = xp_utils.to_numpy(x0)
        bounds = [xp_utils.to_numpy(xp.asarray(b)) for b in self.bounds]
        bounds = Bounds(lb=bounds[0], ub=bounds[1])

        device = self.device

        def scipy_objective(x: Array):
            return xp_utils.to_numpy(self.objective(xp_utils.from_numpy(xp, x, device=device)))

        def scipy_gradient(x: Array):
            return xp_utils.to_numpy(self.gradient(xp_utils.from_numpy(xp, x, device=device)))

        # Initialize the SciPy solution function and its arguments
        result = minimize(
            x0=x0,
            fun=scipy_objective,
            method=self.method,
            jac=scipy_gradient,
            # constraints=self.constraints,
            # hess=self.hessian,
            tol=self.abs_obj_tol,
            bounds=bounds,
            callback=self._callback,
            options=options,
        )

        if self.display:
            logger.info("SciPy finished: %s", result.get("message", ""))

        # Normalize the iteration count under the name every solver reports it as. Not all
        # SciPy methods populate "nit", so the callback's own count serves as the fallback.
        result["num_iter"] = int(result.get("nit", self._iter_count))

        return xp_utils.from_numpy(xp, result["x"]), result

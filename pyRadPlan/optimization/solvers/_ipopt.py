"""
Ipopt solver for non-linear optimization problems.

Notes
-----
Not installed by default. Uses ipyopt because it provides linux wheels
"""

from numpy.typing import NDArray

from ipyopt import Problem
from importlib.metadata import version as _pkg_version


import logging
import re
import numpy as np
import array_api_compat

from ...core.xp_utils.typing import Array

from ._base_solvers import NonLinearOptimizer
from ...core import xp_utils

logger = logging.getLogger(__name__)


class OptimizerIpopt(NonLinearOptimizer):
    """
    IPOPT solver interface.

    Attributes
    ----------
    options : dict
        Options for IPOPT
    """

    name = "Interior Point Optimizer"
    short_name = "ipopt"
    gpu_compatible = False

    allow_keyboard_cancel = True

    options: dict[str]

    def __init__(self):
        self.result = None

        super().__init__()

        self.options = {
            "print_level": 5,
            "print_user_options": "no",
            "print_options_documentation": "no",
            "tol": 1e-10,
            "dual_inf_tol": 1e-4,
            "constr_viol_tol": 1e-4,
            "compl_inf_tol": 1e-4,
            "acceptable_iter": 5,
            "acceptable_tol": self.abs_obj_tol,
            "acceptable_constr_viol_tol": 1e-2,
            "acceptable_dual_inf_tol": 1e10,
            "acceptable_compl_inf_tol": 1e10,
            "acceptable_obj_change_tol": 1e-4,
            "max_iter": self.max_iter,
            "max_cpu_time": float(self.max_time),
            "mu_strategy": "adaptive",
            "hessian_approximation": "limited-memory",
            "limited_memory_max_history": 20,
            "limited_memory_initialization": "scalar2",
            "linear_solver": "mumps",
            "print_timing_statistics": "yes",
        }

    def _solve_problem(
        self,
        x0: Array,
    ) -> tuple[Array, dict]:
        self.options.update(
            {
                "max_iter": self.max_iter,
                "max_cpu_time": float(self.max_time),
                "acceptable_tol": self.abs_obj_tol,
            }
        )

        xp = array_api_compat.array_namespace(x0)

        x0 = xp_utils.to_numpy(x0)

        x0 = np.asarray(x0)

        def ipopt_objective(x: NDArray) -> NDArray[np.float64]:
            return xp_utils.to_numpy(self.objective(xp_utils.from_numpy(xp, x)))

        def ipopt_derivative(x: NDArray, out: Array) -> NDArray[np.float64]:
            out[()] = xp_utils.to_numpy(self.gradient(xp_utils.from_numpy(xp, x))).astype(
                np.float64
            )
            return out

        # Build Ipopt problem via helper to centralize validation & option fallbacks
        nlp = self._validate_ipopt_problem(
            {
                "n": x0.size,
                "eval_f": ipopt_objective,
                "eval_grad_f": ipopt_derivative,
                "intermediate_callback": self._callback,
                "ipopt_options": self.options,
            }
        )

        x, _, status = nlp.solve(x0=x0)

        return xp_utils.from_numpy(xp, x), status

    def _callback(self, *cb_args):  # Ipopt provides many args; we only need cancel flag
        # Optional: could inspect cb_args[1] for iter_count, cb_args[2] for obj_value, etc.
        if self._keyboard_listener.stop_event.is_set():
            return False  # abort
        return True  # continue

    def _validate_ipopt_problem(self, cfg: dict) -> Problem:
        """Create and return an ipyopt Problem instance with version-based option handling."""

        required = [
            "n",
            "eval_f",
            "eval_grad_f",
            "ipopt_options",
            "intermediate_callback",
        ]
        missing = [k for k in required if k not in cfg]
        if missing:
            raise ValueError(f"Missing Ipopt problem fields: {missing}")

        ipopt_options = cfg.get("ipopt_options", self.options)

        # Determine ipyopt version (fall back if unavailable)
        version_str = None
        version_str = _pkg_version("ipyopt")

        def _parse(v: str) -> tuple[int, int, int]:
            m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", v)
            if not m:
                return (0, 0, 0)
            g1, g2, g3 = m.groups()
            return (int(g1), int(g2 or 0), int(g3 or 0))

        # Adjust if upstream changes become known. Chosen conservatively.
        cutoff = (0, 12, 0)
        if version_str is not None:
            parsed = _parse(version_str)
            supports_print = parsed >= cutoff
        else:
            supports_print = True

        # Normalize timing statistics option name
        if supports_print:
            if (
                "timing_statistics" in ipopt_options
                and "print_timing_statistics" not in ipopt_options
            ):
                ipopt_options["print_timing_statistics"] = ipopt_options.pop("timing_statistics")
        else:
            if "print_timing_statistics" in ipopt_options:
                val = ipopt_options.pop("print_timing_statistics")
                ipopt_options.setdefault("timing_statistics", val)
            # Ensure we don't carry both keys unintentionally
            if "print_timing_statistics" in ipopt_options:
                del ipopt_options["print_timing_statistics"]

        return Problem(
            n=cfg["n"],
            x_l=cfg.get("x_l", np.zeros(cfg["n"], dtype=float)),
            x_u=cfg.get("x_u", np.full(cfg["n"], np.inf, dtype=float)),
            m=cfg.get("m", 0),
            g_l=cfg.get("g_l", np.empty((0,))),
            g_u=cfg.get("g_u", np.empty((0,))),
            eval_f=cfg.get("eval_f"),
            eval_grad_f=cfg.get("eval_grad_f"),
            eval_g=cfg.get("eval_g", lambda _x, _out: None),
            eval_jac_g=cfg.get("eval_jac_g", lambda _x, _out: None),
            eval_h=cfg.get("eval_h", None),
            sparsity_indices_jac_g=cfg.get("sparsity_indices_jac_g", (np.array([]), np.array([]))),
            sparsity_indices_h=cfg.get("sparsity_indices_h", (np.array([]), np.array([]))),
            intermediate_callback=cfg.get("intermediate_callback", None),
            ipopt_options=ipopt_options,
        )

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
from contextlib import nullcontext
from typing import Literal
import numpy as np
import array_api_compat

from ...core.xp_utils.typing import Array

from ._base_solvers import NonLinearOptimizer
from ...core import xp_utils
from ...util._jupyter import detect_jupyter
from ...util.logging_utils import native_output_to_logger
from ...util.openmp import blocked_by_openmp

logger = logging.getLogger(__name__)


class OptimizerIpopt(NonLinearOptimizer):
    """
    IPOPT solver interface.

    Attributes
    ----------
    options : dict
        Options passed to IPOPT by name. ``max_iter``, ``max_cpu_time`` and ``acceptable_tol``
        are not taken from here: they are derived from the generic solver attributes
        :attr:`max_iter`, :attr:`max_time` and :attr:`abs_obj_tol` on every solve, and a value
        placed here for one of them is reported and ignored.
    display : bool, default=True
        Whether to show solver output. Sets IPOPT's ``print_level`` to 0 when disabled.
    output_mode : {"auto", "native", "logging"}, default="auto"
        Where IPOPT's output goes. It is written by the IPOPT library to the process' standard
        output descriptor, which Jupyter does not display, so ``"auto"`` leaves it there in a
        terminal and captures it into this module's logger in a notebook. ``"native"`` and
        ``"logging"`` force one or the other regardless of the environment.
    """

    name = "Interior Point Optimizer"
    short_name = "ipopt"
    gpu_compatible = False

    allow_keyboard_cancel = True

    options: dict[str]
    display: bool
    output_mode: Literal["auto", "native", "logging"]

    def __init__(self):
        self.result = None
        self.display = True
        self.output_mode = "auto"

        super().__init__()

        self._iter_count = 0

        self.options = {
            "print_level": 5,
            "print_user_options": "no",
            "print_options_documentation": "no",
            "tol": 1e-10,
            "dual_inf_tol": 1e-4,
            "constr_viol_tol": 1e-4,
            "compl_inf_tol": 1e-4,
            "acceptable_iter": 5,
            "acceptable_constr_viol_tol": 1e-2,
            "acceptable_dual_inf_tol": 1e10,
            "acceptable_compl_inf_tol": 1e10,
            "acceptable_obj_change_tol": 1e-4,
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
        # Re-checked here, not just at registration: another OpenMP-carrying package
        # (PyTorch) may only have been imported after this module. IPOPT initializes
        # its runtime at the first parallel region, so without this the process would
        # abort inside nlp.solve() with no catchable exception.
        blocked = blocked_by_openmp("ipyopt")
        if blocked is not None:
            raise RuntimeError(
                f"Refusing to run IPOPT: {blocked}. Initializing the second OpenMP runtime "
                "would abort the process (OMP: Error #15). Set KMP_DUPLICATE_LIB_OK=TRUE "
                "before starting Python to run it anyway (unsafe), or pick another solver."
            )

        options = self._effective_options()

        self._iter_count = 0

        xp = array_api_compat.array_namespace(x0)

        x0 = xp_utils.to_numpy(x0)

        x0 = np.asarray(x0)

        device = self.device

        def ipopt_objective(x: NDArray) -> NDArray[np.float64]:
            return xp_utils.to_numpy(self.objective(xp_utils.from_numpy(xp, x, device=device)))

        def ipopt_derivative(x: NDArray, out: Array) -> NDArray[np.float64]:
            out[()] = xp_utils.to_numpy(
                self.gradient(xp_utils.from_numpy(xp, x, device=device))
            ).astype(np.float64)
            return out

        # Build Ipopt problem via helper to centralize validation & option fallbacks
        nlp = self._validate_ipopt_problem(
            {
                "n": x0.size,
                "eval_f": ipopt_objective,
                "eval_grad_f": ipopt_derivative,
                "intermediate_callback": self._callback,
                "ipopt_options": options,
            }
        )

        with self._output_context():
            x, obj_value, status = nlp.solve(x0=x0)

        # ipyopt returns the status as a bare integer code; the solver interface promises a
        # dictionary, so the code is wrapped together with the information IPOPT does not
        # report back itself (the iteration count is only visible from the callback).
        result_info = {
            "status": int(status),
            "objective": float(obj_value),
            "num_iter": self._iter_count,
        }

        return xp_utils.from_numpy(xp, x), result_info

    # IPOPT options that mirror a generic solver attribute; the attribute is the source of truth.
    _ATTRIBUTE_BACKED_OPTIONS = {
        "max_iter": "max_iter",
        "max_cpu_time": "max_time",
        "acceptable_tol": "abs_obj_tol",
    }

    def _effective_options(self) -> dict:
        """Assemble the options for one solve without touching :attr:`options`.

        Working on a copy keeps ``display=False`` from permanently overwriting ``print_level``
        and keeps the option-name normalization in :meth:`_validate_ipopt_problem` from leaking
        back into the stored dictionary.
        """
        options = dict(self.options)

        for option, attribute in self._ATTRIBUTE_BACKED_OPTIONS.items():
            if option in options:
                logger.warning(
                    "IPOPT option '%s' in `options` is ignored: it is set from the solver "
                    "attribute '%s' (%r). Configure the attribute instead.",
                    option,
                    attribute,
                    getattr(self, attribute),
                )

        options["max_iter"] = self.max_iter
        options["max_cpu_time"] = float(self.max_time)
        options["acceptable_tol"] = self.abs_obj_tol

        if not self.display:
            options["print_level"] = 0

        return options

    def _output_context(self):
        """Return the context routing IPOPT's output according to :attr:`output_mode`."""
        mode = self.output_mode
        if mode not in ("auto", "native", "logging"):
            raise ValueError(
                f"Unknown output_mode '{mode}', expected 'auto', 'native' or 'logging'."
            )

        if mode == "auto":
            # Only a notebook actually loses the native output, so leave it alone elsewhere.
            mode = "logging" if detect_jupyter() else "native"

        if mode == "native" or not self.display:
            return nullcontext()

        return native_output_to_logger(self.short_name, target=logger)

    def _callback(self, *cb_args):
        # Ipopt's intermediate callback args:
        # (alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm,
        #  regularization_size, alpha_du, alpha_pr, ls_trials)
        data = {}
        if len(cb_args) >= 10:
            data = {
                "iteration": int(cb_args[1]),
                "objective": float(cb_args[2]),
                "constraint_violation": float(cb_args[3]),
                "dual_inf": float(cb_args[4]),
                "step": float(cb_args[9]),
            }
        message = f"iteration {data['iteration']}" if data else ""

        if data:
            # IPOPT does not report the iteration count back from solve(), so it is taken here.
            self._iter_count = data["iteration"]

        cont = self._emit_status(message=message, **data)
        if not cont or self._keyboard_listener.stop_event.is_set():
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

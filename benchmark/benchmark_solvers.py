"""Compare fluence-optimization solvers on photon and proton plans, on CPU and GPU.

Every solver configuration in :data:`SOLVERS` optimizes the same dose influence matrices, and
each run reports iterations, objective/gradient evaluations, timings (split into dose
evaluations and solver overhead) and the final objective, re-evaluated at the returned
fluence so it is comparable across solvers.

Run it as a script (it collects no pytest tests)::

    python benchmark/benchmark_solvers.py
    python benchmark/benchmark_solvers.py --cases photons-5beam protons-1beam --backends cpu
    python benchmark/benchmark_solvers.py --solvers ipax scipy --repeats 5 --csv results.csv
    python benchmark/benchmark_solvers.py --solver "ipax-200=ipax:max_iter=200"

A solver configuration is a label mapped to the ``pln.prop_opt["solver"]`` value, i.e. a
registered solver name or a ``{"name": ..., **attributes}`` dict. Add one by editing
:data:`SOLVERS` or on the command line with ``--solver LABEL=NAME[:ATTR=VALUE,...]``.

Dose influence matrices are recomputed per process unless ``--cache-dir`` is given.
Optimization runs are not bit-reproducible run to run, so use several repeats.
"""

from __future__ import annotations

import argparse
import ast
import csv
import logging
import pickle
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from pyRadPlan import (
    IonPlan,
    PhotonPlan,
    calc_dose_influence,
    generate_stf,
    load_tg119,
    settings,
    xp_utils,
)
from pyRadPlan.optimization.problems import get_problem_from_pln
from pyRadPlan.optimization.solvers import get_available_solvers

logger = logging.getLogger("benchmark_solvers")


# %% Solver configurations
@dataclass(frozen=True)
class SolverConfig:
    """A solver configuration to benchmark.

    ``solver`` is passed as ``pln.prop_opt["solver"]``. ``backends`` restricts the
    configuration to some compute backends (``None``: all).
    """

    solver: str | dict
    backends: tuple[str, ...] | None = None

    @property
    def name(self) -> str:
        """The registered name of the solver."""
        return self.solver if isinstance(self.solver, str) else self.solver["name"]


SOLVERS: dict[str, SolverConfig] = {
    "scipy": SolverConfig("scipy"),
    "ipopt": SolverConfig("ipopt"),
    "ipax": SolverConfig("ipax"),
    # Compare the native GPU solve with host linear algebra and GPU dose evaluations.
    "ipax-cpu": SolverConfig({"name": "ipax", "force_cpu_inner_solve": True}, ("gpu",)),
}


# %% Benchmark cases
@dataclass(frozen=True)
class Case:
    """A treatment plan whose dose influence matrix the solvers optimize."""

    description: str
    make_plan: Callable[[], Any]


def _photon_plan(num_of_beams: int, bixel_width: float | None = None) -> PhotonPlan:
    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
        "couch_angles": np.zeros((num_of_beams,)),
    }
    if bixel_width is not None:
        pln.prop_stf["bixel_width"] = bixel_width
    return pln


def _proton_plan(gantry_angles: list[float]) -> IonPlan:
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_stf = {
        "gantry_angles": gantry_angles,
        "couch_angles": [0.0] * len(gantry_angles),
    }
    return pln


CASES: dict[str, Case] = {
    "photons-5beam": Case("TG119, 5 photon beams, 5 mm bixels", lambda: _photon_plan(5)),
    "photons-9beam-3mm": Case(
        "TG119, 9 photon beams, 3 mm bixels", lambda: _photon_plan(9, bixel_width=3.0)
    ),
    "protons-1beam": Case("TG119, 1 proton beam (0 deg)", lambda: _proton_plan([0.0])),
    "protons-3beam": Case(
        "TG119, 3 proton beams (0/120/240 deg)", lambda: _proton_plan([0.0, 120.0, 240.0])
    ),
}


# %% Backends
def _gpu_available() -> bool:
    return xp_utils.cupy_available() or xp_utils.pytorch_gpu_available()


def _set_backend(backend: str) -> None:
    if backend == "cpu":
        settings.xp.prefer_gpu = False
        settings.xp.preferred_cpu_array_backend = "numpy"
    elif backend == "gpu":
        settings.xp.prefer_gpu = True
    else:
        raise ValueError(f"Unknown backend '{backend}' (use 'cpu' or 'gpu')")


# %% Running
@dataclass
class RunResult:
    """Measurements of one optimization run."""

    case: str
    backend: str
    array_backend: str
    solver: str
    repeat: int
    num_bixels: int
    status: str
    num_iter: int | None
    obj_evals: int
    grad_evals: int
    objective: float
    total_time: float
    solver_time: float
    obj_time: float
    grad_time: float

    @property
    def overhead_time(self) -> float:
        """Solver time not spent evaluating objectives or gradients."""
        return self.solver_time - self.obj_time - self.grad_time


#: Readable names for the integer status codes of the solvers that report one.
_STATUS_CODES = {
    "scipy": {0: "converged", 1: "max_iter", 2: "stopped"},
    "ipopt": {0: "optimal", 1: "acceptable", -1: "max_iter", -4: "max_time", 2: "infeasible"},
}


def _status(solver: str, result_info: Any) -> str:
    if not isinstance(result_info, dict) or "status" not in result_info:
        return "?"
    status = result_info["status"]
    if isinstance(status, str):
        return status
    status = int(status)
    return _STATUS_CODES.get(solver, {}).get(status, f"status {status}")


def run_once(
    case: str, ct, cst, stf, dij, pln, backend: str, label: str, repeat: int, max_iter=None
) -> RunResult:
    """Optimize once and return the measurements."""
    pln.prop_opt = {"solver": SOLVERS[label].solver, "display": False}
    if max_iter is not None:
        pln.prop_opt["max_iter"] = max_iter

    problem = get_problem_from_pln(pln)
    t0 = time.perf_counter()
    x, result_info = problem.solve(ct, cst, stf, dij)
    total_time = time.perf_counter() - t0

    obj_times = list(problem._obj_times)
    grad_times = list(problem._deriv_times)

    xp = problem._array_backend
    final = problem._objective_function(xp_utils.from_numpy(xp, x, device=problem._device))

    num_iter = result_info.get("num_iter") if isinstance(result_info, dict) else None
    return RunResult(
        case=case,
        backend=backend,
        array_backend=xp.__name__.removeprefix("array_api_compat."),
        solver=label,
        repeat=repeat,
        num_bixels=dij.total_num_of_bixels,
        status=_status(SOLVERS[label].name, result_info),
        num_iter=None if num_iter is None else int(num_iter),
        obj_evals=len(obj_times),
        grad_evals=len(grad_times),
        objective=float(final),
        total_time=total_time,
        solver_time=float(problem._solve_time),
        obj_time=float(sum(obj_times)),
        grad_time=float(sum(grad_times)),
    )


def prepare_case(name: str, ct, cst, cache_dir: str | None = None):
    """Steering information and dose influence matrix of a case (computed on the CPU)."""
    cache = None if cache_dir is None else Path(cache_dir) / f"{name}.pkl"
    if cache is not None and cache.exists():
        with cache.open("rb") as f:
            pln, stf, dij = pickle.load(f)
        logger.warning("%s: loaded dij from %s", name, cache)
        return pln, stf, dij

    _set_backend("cpu")
    pln = CASES[name].make_plan()
    t0 = time.perf_counter()
    stf = generate_stf(ct, cst, pln)
    dij = calc_dose_influence(ct, cst, stf, pln)
    logger.warning(
        "%s: %d bixels, dij in %.1f s", name, dij.total_num_of_bixels, time.perf_counter() - t0
    )

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("wb") as f:
            pickle.dump((pln, stf, dij), f)
    return pln, stf, dij


# %% Reporting
def _fmt_time(seconds: float) -> str:
    return f"{seconds:8.2f}"


def summarize(results: list[RunResult]) -> list[dict]:
    """One row per (case, backend, solver): medians over repeats, gap to the best objective."""
    groups: dict[tuple, list[RunResult]] = {}
    for r in results:
        groups.setdefault((r.case, r.backend, r.solver), []).append(r)

    best: dict[tuple, float] = {}
    for r in results:
        key = (r.case, r.backend)
        best[key] = min(best.get(key, np.inf), r.objective)

    rows = []
    for (case, backend, solver), runs in groups.items():
        med = lambda attr: statistics.median(getattr(r, attr) for r in runs)  # noqa: E731
        ref = best[(case, backend)]
        statuses = Counter(r.status for r in runs)
        iters = [r.num_iter for r in runs if r.num_iter is not None]
        rows.append(
            {
                "case": case,
                "backend": f"{backend} ({runs[0].array_backend})",
                "solver": solver,
                "bixels": runs[0].num_bixels,
                "runs": len(runs),
                "status": ", ".join(f"{s} x{n}" if n > 1 else s for s, n in statuses.items()),
                "iter": statistics.median(iters) if iters else None,
                "f_evals": med("obj_evals"),
                "g_evals": med("grad_evals"),
                "total_s": med("total_time"),
                "solver_s": med("solver_time"),
                "eval_s": statistics.median(r.obj_time + r.grad_time for r in runs),
                "overhead_s": med("overhead_time"),
                "objective": med("objective"),
                "gap_to_best": (med("objective") - ref) / max(abs(ref), 1e-300),
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    """Print the summary rows as a table."""
    header = (
        f"{'case':18s} {'backend':14s} {'solver':12s} {'status':22s} {'iter':>6s} "
        f"{'f_ev':>6s} {'g_ev':>6s} {'total':>8s} {'solver':>8s} {'evals':>8s} "
        f"{'ovrhd':>8s} {'objective':>13s} {'gap':>9s}"
    )
    print()
    print("Medians over repeats; times in seconds. 'evals' is time spent in objective and")
    print("gradient evaluations, 'ovrhd' the rest of the solver time; 'gap' is the relative")
    print("difference to the best objective any run reached on that case and backend.")
    print(header)
    print("-" * len(header))
    last_group = None
    for row in rows:
        group = (row["case"], row["backend"])
        if last_group is not None and group != last_group:
            print()
        last_group = group
        iters = "-" if row["iter"] is None else f"{row['iter']:6.0f}"
        print(
            f"{row['case']:18s} {row['backend']:14s} {row['solver']:12s} {row['status'][:22]:22s} "
            f"{iters:>6s} {row['f_evals']:6.0f} {row['g_evals']:6.0f} "
            f"{_fmt_time(row['total_s'])} {_fmt_time(row['solver_s'])} "
            f"{_fmt_time(row['eval_s'])} {_fmt_time(row['overhead_s'])} "
            f"{row['objective']:13.6e} {row['gap_to_best']:9.2e}"
        )


def write_csv(path: str, results: list[RunResult]) -> None:
    """Write every run, one row each."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(asdict(results[0])) + ["overhead_time"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({**asdict(r), "overhead_time": r.overhead_time})


# %% Command line
def _parse_solver_option(spec: str) -> tuple[str, SolverConfig]:
    """Parse ``LABEL=NAME[:ATTR=VALUE,...]`` (values are Python literals or plain strings)."""
    label, _, rest = spec.partition("=")
    if not rest:
        label, rest = spec, spec
    name, _, attrs = rest.partition(":")
    if not attrs:
        return label, SolverConfig(name)
    solver: dict[str, Any] = {"name": name}
    for item in attrs.split(","):
        key, _, value = item.partition("=")
        try:
            solver[key.strip()] = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            solver[key.strip()] = value.strip()
    return label, SolverConfig(solver)


def parse_args(argv=None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    parser.add_argument("--backends", nargs="+", choices=["cpu", "gpu"], default=["cpu", "gpu"])
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=None,
        help=f"Labels from the configuration table (default: all available): {list(SOLVERS)}",
    )
    parser.add_argument(
        "--solver",
        action="append",
        default=[],
        metavar="LABEL=NAME[:ATTR=VALUE,...]",
        help="Add a solver configuration, e.g. 'ipax-200=ipax:max_iter=200'. Repeatable.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--warmup", type=int, default=1, help="Untimed runs per backend before measuring."
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Default: each solver's own.")
    parser.add_argument("--cache-dir", default=None, help="Cache dose influence matrices here.")
    parser.add_argument("--csv", default=None, help="Write every run to this CSV file.")
    return parser.parse_args(argv)


def main(argv=None) -> list[RunResult]:
    """Run the benchmark and print the summary; returns every run."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    # Per-run notes (problem/solver defaults, non-convergence) would drown the progress lines;
    # the status column reports convergence.
    logging.getLogger("pyRadPlan").setLevel(logging.ERROR)

    for spec in args.solver:
        label, config = _parse_solver_option(spec)
        SOLVERS[label] = config

    registered = get_available_solvers()
    labels = args.solvers or list(SOLVERS)
    unavailable = [label for label in labels if SOLVERS[label].name not in registered]
    if unavailable:
        print(f"Skipping unavailable solvers: {unavailable}")
    labels = [label for label in labels if label not in unavailable]
    for label in labels:
        # get_solver only warns about an unknown attribute and the warning is silenced below,
        # so a typo would quietly benchmark the defaults.
        solver = SOLVERS[label].solver
        if isinstance(solver, dict):
            instance = registered[solver["name"]]()
            unknown = [key for key in solver if key != "name" and not hasattr(instance, key)]
            if unknown:
                raise SystemExit(f"Solver '{label}': unknown attributes {unknown}")

    backends = list(args.backends)
    if "gpu" in backends and not _gpu_available():
        print("No GPU array backend available, skipping 'gpu'.")
        backends.remove("gpu")

    previous = settings.xp.model_dump()
    ct, cst = load_tg119()
    results: list[RunResult] = []
    try:
        for case in args.cases:
            pln, stf, dij = prepare_case(case, ct, cst, args.cache_dir)
            for backend in backends:
                _set_backend(backend)
                case_labels = [
                    label
                    for label in labels
                    if SOLVERS[label].backends is None or backend in SOLVERS[label].backends
                ]
                for label in case_labels:
                    for _ in range(args.warmup):
                        run_once(case, ct, cst, stf, dij, pln, backend, label, -1, args.max_iter)
                for repeat in range(args.repeats):
                    # Interleave solvers so drifting machine load hits all of them alike.
                    for label in case_labels:
                        r = run_once(
                            case, ct, cst, stf, dij, pln, backend, label, repeat, args.max_iter
                        )
                        results.append(r)
                        print(
                            f"{case:18s} {backend:4s} {label:12s} #{repeat}: {r.status:12s} "
                            f"it={r.num_iter} f_ev={r.obj_evals} obj={r.objective:.6e} "
                            f"t={r.total_time:.2f}s",
                            flush=True,
                        )
    finally:
        for key, value in previous.items():
            setattr(settings.xp, key, value)

    if results:
        print_table(summarize(results))
        if args.csv:
            write_csv(args.csv, results)
            print(f"\nWrote {len(results)} runs to {args.csv}")
    return results


if __name__ == "__main__":
    main()

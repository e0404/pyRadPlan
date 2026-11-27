"""Optimization tools of the planning state: objectives, optimize, plans, snapshots."""

from contextlib import contextmanager
from typing import Any, Iterator, Optional

import numpy as np

from pyRadPlan._settings import get_settings
from pyRadPlan.core import ComputeControl, StatusReport, observe_control, observe_reports
from pyRadPlan.dij import Dij
from pyRadPlan.optimization._fluence_optimization import fluence_optimization
from pyRadPlan.optimization.objectives import MeanDose, get_available_objectives
from pyRadPlan.optimization.solvers import SolverBase, get_available_solvers
from pyRadPlan.quantities._rbe_x_dose import RBExDose
from pyRadPlan.result import PlanResult

from .._cst_agents import _objectives_by_voi
from . import _geometry as geo
from ._evaluation import PlanRecord
from ._state_base import PlanningError, UnknownVoiError, sitk_array

# optimize stops once the best objective improved too little over this many iterations
_STALL_WINDOW = 50


def _solver_name(solver: Any) -> Optional[str]:
    """Registry name of a prop_opt solver entry (name, config dict or instance)."""
    if isinstance(solver, dict):
        return solver.get("name")
    if isinstance(solver, SolverBase):
        return solver.short_name
    return solver


@contextmanager
def _cpu_backend(enabled: bool = True) -> Iterator[None]:
    """Run the block with numpy on the CPU (the backend is a global setting).

    GPU reductions are not deterministic, so the same objectives could give different
    plans; the agent needs reproducible results to judge its changes. Only the settings
    this changed are put back, and only while they still hold the value set here, so a
    change made meanwhile (another thread, the user) survives.
    """
    if not enabled:
        yield
        return
    xp = get_settings().xp
    wanted = {"prefer_gpu": False, "preferred_cpu_array_backend": "numpy"}
    before = {k: getattr(xp, k) for k, v in wanted.items() if getattr(xp, k) != v}
    for k in before:
        setattr(xp, k, wanted[k])
    try:
        yield
    finally:
        for k, value in before.items():
            if getattr(xp, k) == wanted[k]:
                setattr(xp, k, value)


def _dose_is_linear(dij: Any, quantity: str) -> bool:
    """Whether *quantity* of a result of *dij* is linear in the fluence.

    Physical dose and constant-RBE weighted dose are; an RBE-weighted dose from the LQ
    effect (variable RBE) is not.
    """
    if quantity != "rbe_x_dose":
        return True
    has_lq = (
        getattr(dij, "alpha_dose", None) is not None
        and getattr(dij, "sqrt_beta_dose", None) is not None
    )
    return RBExDose.choose_path(dij, has_effect=has_lq, strict=False) != "effect"


class OptimizeMixin:
    """Objectives, optimization, the kept plans and plan versions."""

    def current_objectives(self) -> dict[str, list[dict]]:
        return _objectives_by_voi(self.cst)

    def set_objectives(self, voi_name: str, objectives: list[Any]) -> dict:
        """Replace all objectives of one VOI. Penalties are relative weights.

        Mean Dose pulls the mean towards d_ref from both sides; for an upper limit use
        Mean Overdosing (d_max) or Mean Dose with d_ref 0.
        """
        if voi_name not in self.voi_names:
            raise UnknownVoiError(voi_name, self.voi_names)
        # Objectives created in code (outcome models) load estimators when validated
        refused = sorted(
            {str(getattr(o, "name", o)) for o in objectives if not getattr(o, "selectable", False)}
        )
        if refused:
            raise PlanningError(
                f"{refused} cannot be set here. Use: "
                f"{sorted(get_available_objectives(selectable_only=True))}."
            )
        vois = [
            v.model_copy(update={"objectives": list(objectives)}) if v.name == voi_name else v
            for v in self.cst.vois
        ]
        self.cst = self.cst.model_copy(update={"vois": vois})
        self.dose = self.last_evaluation = None
        self._log("set_objectives", voi=voi_name, n=len(objectives))
        out = {"voi": voi_name, "objectives": self.current_objectives()[voi_name]}
        if self._voi(voi_name).voi_type != "TARGET" and any(
            isinstance(o, MeanDose) and o.d_ref > 0 for o in objectives
        ):
            out["note"] = (
                "Mean Dose with d_ref > 0 also pushes a lower mean up to d_ref. For an upper "
                "limit use Mean Overdosing (d_max) or d_ref 0."
            )
        return out

    def optimize(
        self,
        max_iter: int = 500,
        min_improvement: float = 1e-2,
        start_from: Optional[int] = None,
    ) -> dict:
        """Optimize fluence and evaluate every goal (margin >= 0: passed).

        Includes a soft NormalTissue goal: max of the patient beyond a margin around the
        targets (case_overview) below the highest target dose level.
        max_iter: ~500 to compare settings, 1000-2000 for promising plans. Stops early once
        the objective improved by less than min_improvement (relative) over 50 iterations;
        0 disables.
        start_from: plan number (same beams) whose fluence the solver starts from instead of
        zero. Worth it after small changes, above all to DVH objectives (Min / Max DVH), whose
        result depends on the start: the plan changes only where the objectives ask for it
        and needs fewer iterations. Start from zero after large changes.
        An exact repeat of an earlier optimization (same objectives, start and settings, or
        continuing a converged plan with its own objectives) is refused without using one.
        """
        if self.num_optimizations >= self.max_optimizations:
            raise PlanningError(
                f"Optimization budget of {self.max_optimizations} used up. "
                "Return your final report now."
            )
        if not any(v.objectives for v in self.cst.vois):
            raise PlanningError("No objectives defined. Call set_objectives first.")
        max_iter = min(max_iter, self.max_iter)
        run = {
            "start_from": start_from,
            "max_iter": max_iter,
            "min_improvement": min_improvement,
            "setup": (self.solver, self.cpu_only, repr(self.pln.prop_opt)),
        }
        self._refuse_repeat(run)
        self._check_setup_trials()
        dij = self._ensure_dij()
        initial = None if start_from is None else self._plan_fluence(start_from, dij)
        # An objective over zero voxels is NaN, and the solver stops at iteration 0. The
        # optimizer sees the VOIs after overlap priorities, on the dose grid.
        apply_overlap = bool((self.pln.prop_opt or {}).get("apply_overlap", True))
        idx = self._dose_grid_indices(dij, apply_overlap=apply_overlap)
        empty = [v.name for v in self.cst.vois if v.objectives and not idx[v.name].size]
        if empty:
            res = "x".join(f"{r:g}" for r in dij.dose_grid.resolution.values())
            raise PlanningError(
                f"{empty} have no voxels left on the {res} mm dose grid once overlapping "
                "voxels go to structures with a higher overlap priority (lower number), so "
                "their objectives cannot be evaluated. Remove them (set_objectives with []) "
                "or put the objectives on a margin helper (PRV) around the structure."
            )

        prop_opt = dict(self.pln.prop_opt or {})
        solver = prop_opt.get("solver")
        if self.solver is not None and _solver_name(solver) != self.solver:
            solver = self.solver
        if solver is None:
            # Without a configured solver the optimizer falls back the same way
            available = list(get_available_solvers())
            solver = "ipopt" if "ipopt" in available else available[0]
        if isinstance(solver, (str, dict)):
            solver = {"name": solver} if isinstance(solver, str) else dict(solver)
            prop_opt["solver"] = {**solver, "max_iter": max_iter}
        opt_info: dict = {}
        self._notify("phase", phase="optimize")
        # A stop keeps the last iterate, so stopping on a stalled objective ends the run
        # like convergence would. This control replaces self.control while the solver
        # runs, so a stop request on that is passed on.
        stall = ComputeControl()
        best: list[float] = []

        def watch(report: Any) -> None:
            if self.control is not None and self.control.stop_requested:
                stall.request_stop()
            f = report.data.get("objective") if isinstance(report, StatusReport) else None
            if f is None or not np.isfinite(f):
                return
            best.append(min(f, best[-1]) if best else f)
            if min_improvement > 0 and len(best) > _STALL_WINDOW:
                if best[-_STALL_WINDOW - 1] - best[-1] < min_improvement * abs(best[-1]):
                    stall.request_stop()

        with observe_reports(watch), observe_control(stall), _cpu_backend(self.cpu_only):
            result = fluence_optimization(
                self.ct,
                self.cst,
                self.stf,
                dij,
                self.pln.model_copy(update={"prop_opt": prop_opt}),
                opt_info=opt_info,
                initial_fluence=initial,
            )
        aborted = self.control is not None and self.control.stop_requested
        converged = stall.stop_requested and not aborted
        result, quantity = self._use_result(result)
        self.num_optimizations += 1

        self.last_evaluation = self._evaluate_dose()
        self.last_optimizer = self._optimizer_summary(opt_info, max_iter, apply_overlap)
        self.last_optimizer["converged"] = converged
        self.last_optimizer["start_from"] = start_from
        self.last_evaluation["optimizer"] = {
            k: self.last_optimizer[k]
            for k in ("iterations", "max_iter", "converged", "improvement_last_20pct")
        }
        if start_from is not None:
            self.last_evaluation["optimizer"]["start_from"] = start_from
        self._add_plan(
            result, quantity, cst=self.cst, objectives=self.current_objectives(), run=run
        )
        self._log(
            "optimize",
            max_iter=max_iter,
            start_from=start_from,
            all_hard_passed=self.last_evaluation["all_hard_passed"],
            num_failed=self.last_evaluation["num_failed"],
        )
        return self.last_evaluation

    def _refuse_repeat(self, run: dict) -> None:
        """Refuse an optimization that can only return an earlier plan, before it costs one.

        On the CPU the solver is deterministic: the same objectives (on the same structures)
        from the same start with the same settings give the same plan, and continuing a plan
        that converged with its own objectives returns it again unless min_improvement is
        now stricter.
        """
        if not self.cpu_only:
            return
        objectives = self.current_objectives()
        masks = {v.name: v.mask for v in self.cst.vois}
        for p in self.plans:
            prev = p.run
            if prev is None or prev["setup"] != run["setup"]:
                continue
            if p.objectives != objectives or p.beams != self.current_beams():
                continue
            if any(masks.get(v.name) is not v.mask for v in p.cst.vois if v.objectives):
                continue
            iterations = (p.evaluation.get("optimizer") or {}).get("iterations")
            stopped_early = iterations is not None and iterations < prev["max_iter"]
            if (
                prev["start_from"] == run["start_from"]
                and prev["min_improvement"] == run["min_improvement"]
                and (
                    prev["max_iter"] == run["max_iter"]
                    or (stopped_early and run["max_iter"] >= iterations)
                )
            ):
                start = "zero" if run["start_from"] is None else f"plan {run['start_from']}"
                raise PlanningError(
                    f"Plan {p.index} was optimized with exactly these objectives and "
                    f"settings from {start}; the solver is deterministic, so this would "
                    "return the same plan. Not run, no optimization used: change objectives "
                    "or settings, start from another plan, or blend plans."
                )
            if (
                run["start_from"] == p.index
                and stopped_early
                and run["min_improvement"] >= prev["min_improvement"]
            ):
                raise PlanningError(
                    f"Plan {p.index} was optimized with exactly these objectives until it "
                    "converged; continuing it returns the same plan. Not run, no optimization "
                    "used: change objectives, or lower min_improvement to converge further."
                )

    def _plan(self, index: int) -> PlanRecord:
        plan = next((p for p in self.plans if p.index == index), None)
        if plan is None:
            raise PlanningError(f"No plan {index}. Plans: {[p.index for p in self.plans]}")
        return plan

    def _plan_fluence(self, index: int, dij: Dij) -> np.ndarray:
        """Return the fluence of plan *index*, checked against the current beams and dij."""
        plan = self._plan(index)
        if plan.beams != self.current_beams():
            raise PlanningError(f"Plan {index} has other beams than the current ones.")
        fluence = plan.fluence
        if fluence is None or fluence.size != dij.total_num_of_bixels:
            raise PlanningError(f"Plan {index} does not fit the current dose influence.")
        return fluence

    def _use_result(self, result: PlanResult) -> tuple[PlanResult, str]:
        """Make the dose of a CT-grid *result* the current dose.

        Returns the result without its per-beam and per-scenario quantities, which every
        kept plan would otherwise hold in memory, and the quantity the goals are judged on.
        """
        result = result.model_copy(update={"beam_quantities": {}, "scenario_quantities": {}})
        quantity = "rbe_x_dose" if "rbe_x_dose" in result else "physical_dose"
        self.dose = result[quantity].values
        return result, quantity

    def _add_plan(self, result: PlanResult, quantity: str, **data: Any) -> None:
        """Record *result*, the current dose, as a new plan (see :attr:`plans`)."""
        plan = PlanRecord(
            index=len(self.plans) + 1,
            beams=self.current_beams(),
            evaluation=self.last_evaluation,
            result=result,
            quantity=quantity,
            **data,
        )
        self.plans.append(plan)
        if not _dose_is_linear(self.dij, quantity):
            self._nonlinear_plans.add(plan.index)
        self._notify("plan", plan=plan)

    def optimizer_report(self) -> dict:
        """Report on the last optimization: convergence and objective contributions.

        Solver status, iterations, objective start / end, improvement over the last 20 %
        (large: not converged, raise max_iter), each objective's share of the total.
        """
        if self.last_optimizer is None:
            raise PlanningError(
                "No optimization behind the current dose (none yet, or a blend). "
                "Call optimize first."
            )
        self._log("optimizer_report")
        return self.last_optimizer

    def _optimizer_summary(self, opt_info: dict, max_iter: int, apply_overlap: bool) -> dict:
        hist = [float(f) for f in opt_info.get("obj_history", [])]
        info = opt_info.get("result_info", {})
        iterations = opt_info.get("num_iter", info.get("num_iter"))
        improvement = None
        if len(hist) > 1 and hist[-1]:
            improvement = round((hist[int(0.8 * (len(hist) - 1))] - hist[-1]) / abs(hist[-1]), 4)

        dose = sitk_array(self.dose)
        cst = self.cst.apply_overlap_priorities() if apply_overlap else self.cst
        # As the solver sees them: one fraction for total-course references
        n = self.pln.num_of_fractions if self.pln.dose_convention == "total" else 1
        rows = []
        for v in cst.vois:
            values = dose[geo.as_bool(v.mask)] / n
            for o in v.objectives:
                solved = o.model_copy().normalize_to_fraction_num(n) if n > 1 else o
                f = float(o.priority * solved.compute_objective(values)) if values.size else 0.0
                rows.append({"voi": v.name, "objective": type(o).__name__, "value": f})
        total = sum(r["value"] for r in rows) or 1.0
        for r in rows:
            r["share_pct"] = round(100 * r["value"] / total, 1)
            r["value"] = round(r["value"], 5)
        return {
            "status": info.get("message", info.get("status")),
            "iterations": iterations,
            "max_iter": max_iter,
            "hit_iteration_limit": iterations is not None and iterations >= max_iter,
            "objective_start": round(hist[0], 5) if hist else None,
            "objective_final": round(hist[-1], 5) if hist else None,
            "improvement_last_20pct": improvement,
            "contributions": sorted(rows, key=lambda r: r["share_pct"], reverse=True),
        }

    # ---- plan versions -----------------------------------------------------------------
    def _current_plan_index(self) -> Optional[int]:
        """Index of the optimization the current dose comes from, None without a dose."""
        if self.dose is None:
            return None
        return next((p.index for p in reversed(self.plans) if p.dose is self.dose), None)

    def snapshot(self, label: str) -> dict:
        """Save the current objectives, helpers, dose and evaluation under *label*."""
        plan = self._current_plan_index()
        self.snapshots[label] = {
            "cst": self.cst,
            "helpers": list(self.created_helpers),
            "beams": self.current_beams(),
            "evaluation": self.last_evaluation,
            "qis": self._clinical_qi_summary(),
            "dose": self.dose,
            "optimizer": self.last_optimizer,
            "plan": plan,
        }
        self._log("snapshot", label=label, plan=plan)
        return {"saved": label, "plan": plan, "snapshots": list(self.snapshots)}

    def restore(self, label: str) -> dict:
        """Return to the objectives, helpers and dose of snapshot *label*.

        Beams and the clinical structures (map_structures) must be unchanged. A snapshot
        saved with a dose needs no new optimize.
        """
        if label not in self.snapshots:
            raise PlanningError(f"No snapshot '{label}'. Available: {list(self.snapshots)}")
        snap = self.snapshots[label]
        if snap["beams"] != self.current_beams():
            raise PlanningError("Beams changed since this snapshot; set_beams first.")
        saved = {
            (v.name, v.voi_type)
            for v in snap["cst"].vois
            if v.voi_type != "HELPER" and v.name not in snap["helpers"]
        }
        if saved != {(v.name, v.voi_type) for v in self._clinical_vois()}:
            raise PlanningError(
                f"map_structures changed the structures or their types since '{label}' was "
                "saved, so its objectives, dose and evaluation no longer fit. Not restored: "
                "set the objectives again and save a new version."
            )
        self.cst = snap["cst"]
        self.created_helpers = list(snap["helpers"])
        self.dose, self.last_evaluation = snap["dose"], snap["evaluation"]
        self.last_optimizer = snap["optimizer"]
        self._log("restore", label=label, plan=snap["plan"])
        out = {"restored": label, "plan": snap["plan"], "evaluation": snap["evaluation"]}
        if self.dose is None:
            out["note"] = "Saved without a dose: call optimize to compute one."
        else:
            of_plan = f" of plan {snap['plan']}" if snap["plan"] is not None else ""
            out["note"] = f"Dose{of_plan} restored; no optimize needed."
        return out

    def _clinical_qi_summary(self) -> Optional[dict]:
        if self.dose is None:
            return None
        qis = self._qis()
        return {
            v.name: {k: round(float(qis[v.name][k].value), 3) for k in ("mean", "max")}
            for v in self._clinical_vois()
            if v.name in qis
        }

    def compare_snapshots(self, a: str, b: str = "current") -> dict:
        """Compare two versions: goal values and mean/max dose per clinical VOI, b - a."""

        def get(label):
            if label == "current":
                return {"evaluation": self.last_evaluation, "qis": self._clinical_qi_summary()}
            if label not in self.snapshots:
                raise PlanningError(f"No snapshot '{label}'. Available: {list(self.snapshots)}")
            return self.snapshots[label]

        sa, sb = get(a), get(b)
        if not sa["evaluation"] or not sb["evaluation"]:
            raise PlanningError("Both versions need an optimized dose (call optimize first).")
        ga_by = {g["goal"]: g for g in sa["evaluation"]["goals"]}
        goals = []
        for gb in sb["evaluation"]["goals"]:
            ga = ga_by.get(gb["goal"])
            if ga is None:  # goal not present in version a (protocol changed)
                goals.append({"goal": gb["goal"], "passed": [None, gb["passed"]]})
                continue
            row = {"goal": gb["goal"], "passed": [ga["passed"], gb["passed"]]}
            if "value" in ga and "value" in gb:
                row["value"] = [ga["value"], gb["value"]]
                row["delta"] = round(gb["value"] - ga["value"], 3)
            goals.append(row)
        vois = {}
        for name in sorted(set(sa["qis"] or {}) & set(sb["qis"] or {})):
            vois[name] = {
                k: round(sb["qis"][name][k] - sa["qis"][name][k], 3) for k in ("mean", "max")
            }
        self._log("compare_snapshots", a=a, b=b)
        return {
            "versions": [a, b],
            "hard_passed": [
                sa["evaluation"]["all_hard_passed"],
                sb["evaluation"]["all_hard_passed"],
            ],
            "goals": goals,
            "dose_delta_per_voi": vois,
        }

"""Trade-off tools of the planning state: plan blends and the Pareto front."""

from types import SimpleNamespace
from typing import Optional

import numpy as np

from . import _geometry as geo
from . import _pareto as pareto
from ._constraints import qi_key
from ._evaluation import evaluate_protocol, voxel_metric
from ._state_base import NORMAL_TISSUE, PlanningError, sitk_array


class ParetoMixin:
    """Blends of kept plans and the Pareto front between them."""

    def blend_plans(self, plans: list[int], weights: list[float]) -> dict:
        """Mix the fluences of earlier plans (same beams) and evaluate the mix like a plan.

        Costs no optimization: the dose is computed from the mixed fluence, so the blend is
        an exact, deliverable plan between its sources. Blend plans that pass the hard goals
        but differ in their trade-offs (e.g. one sparing the cord, one the parotids) to find
        plans in between; blending near-identical plans gains nothing. Physical (and
        constant-RBE) dose is linear in the fluence, so max, min and mean goals that all
        sources pass also pass in the blend; D<x> and V<x> goals can change either way.
        With a variable RBE the dose is not linear, so any goal can change (and
        pareto_front, which relies on the linearity, is refused).
        Weights (>= 0) are normalized to sum 1. The blend becomes the current
        plan: save it with plan_versions or use it as start_from. The number of blends
        kept is limited (pareto_front compares blends without keeping them).
        """
        num_blends = sum(1 for p in self.plans if p.blend)
        if num_blends >= self.max_blends:
            raise PlanningError(
                f"{num_blends} blends are kept already (at most {self.max_blends}). "
                "pareto_front compares blends without keeping them; continue with the "
                "plans you have."
            )
        if len(plans) < 2 or len(set(plans)) != len(plans):
            raise PlanningError("Give at least two different plan numbers.")
        if len(weights) != len(plans):
            raise PlanningError(f"{len(plans)} plans need {len(plans)} weights.")
        w = np.asarray(weights, dtype=float)
        if (w < 0).any() or not w.sum() > 0:
            raise PlanningError("Weights must be >= 0 and not all 0.")
        w = w / w.sum()
        if self.dij is None:
            raise PlanningError(
                "No dose influence loaded (new beams, or a helper outside its region): "
                "optimize first."
            )
        fluences = [self._plan_fluence(i, self.dij) for i in plans]
        fluence = sum(wi * f for wi, f in zip(w, fluences))
        result, quantity = self._use_result(
            self.dij.compute_result_ct_grid(fluence, num_of_fractions=self.pln.result_dose_factor)
        )
        weights_out = [round(float(wi), 4) for wi in w]

        self.last_evaluation = self._evaluate_dose()
        self.last_evaluation["blend"] = {"plans": list(plans), "weights": weights_out}
        self.last_optimizer = None
        main = self._plan(plans[int(np.argmax(w))])
        self._add_plan(
            result,
            quantity,
            cst=main.cst,
            objectives=main.objectives,
            blend=self.last_evaluation["blend"],
        )
        self._log("blend_plans", plans=list(plans), weights=weights_out)
        return self.last_evaluation

    def pareto_front(
        self,
        plans: list[int],
        axes: Optional[list[str]] = None,
        n_samples: int = 300,
        top_n: int = 6,
    ) -> dict:
        """Sample blends of plans (same beams) and return their Pareto front.

        Costs no optimization and changes no plan. Only blends passing all hard goals count.
        Needs a dose that is linear in the fluence (physical or constant-RBE weighted, see
        blend_plans): with a variable RBE it is refused, blend_plans stays exact.
        Trade-off axes are the target goals, soft goals and mean goals whose value differs
        between the plans by >= 1 % of its limit (or the goal labels in axes); other hard
        goals (OAR max and the like) only need to pass (constraint_goals). Axis values are
        raw goal values, not the clipped margins of the score, so sparing far below a limit
        counts; differences below 1 % of a limit do not count (epsilon dominance), so the
        front holds blends that differ noticeably. Returns the axes (best / worst on the
        front, value per plan), up to top_n
        spread-out front blends (weights in the order of plans: blend_plans(plans,
        weights) makes one a plan), the compromise blend (closest to the best on every
        axis at once) with its weakest axes, the plans a blend beats on every axis
        (dominated_plans: they extend nothing) and the goals no plan changes (flat_goals).
        Grow the front where the compromise is weakest: blend it, optimize start_from that
        blend with its weakest axes tightened, and pass the result here with the others.
        """
        if len(plans) < 2 or len(set(plans)) != len(plans):
            raise PlanningError("Give at least two different plan numbers.")
        sources = [self._plan(i) for i in plans]
        if any(p.beams != sources[0].beams for p in sources):
            raise PlanningError("All plans need the same beams.")
        nonlinear = [p.index for p in sources if p.index in self._nonlinear_plans]
        if nonlinear:
            raise PlanningError(
                f"The dose of plans {nonlinear} is RBE-weighted with a variable RBE and not "
                "linear in the fluence, so blends cannot be sampled from the plan doses. "
                "Use blend_plans, which computes each blend exactly."
            )
        n_samples = min(n_samples, self.max_pareto_samples)
        protocol = self._full_protocol()
        masks = {v.name: geo.as_bool(v.mask) for v in self.cst.vois}
        masks[NORMAL_TISSUE] = self._normal_tissue_region()
        keys: dict[str, set[str]] = {}
        for g in protocol.goals:
            voi = self.aliases.get(g.voi, g.voi)
            if voi in masks:
                keys.setdefault(voi, set()).add(qi_key(g.metric))
        if not keys:
            raise PlanningError("No goal can be evaluated: map the protocol structures first.")
        doses = [sitk_array(p.dose) for p in sources]
        # Doses are linear in the fluence, so a blend's dose is the blend of the doses
        voxels = {voi: np.stack([d[masks[voi]] for d in doses]) for voi in keys}

        def evaluate(w: np.ndarray) -> dict:
            qis = {}
            for voi, metric_keys in keys.items():
                v = w @ voxels[voi]
                qis[voi] = {k: SimpleNamespace(value=voxel_metric(v, k)) for k in metric_keys}
            return evaluate_protocol(protocol, qis, self.aliases)

        n = len(plans)
        weights = pareto.sample_weights(n, n_samples)
        evals = [evaluate(w / w.sum()) for w in weights]

        labels = [r["goal"] for r in evals[0]["goals"]]
        goal_of = {g.label: g for g in protocol.goals}
        values = np.array([[r["value"] for r in e["goals"]] for e in evals], dtype=float)
        sign = np.array([1.0 if goal_of[lbl].op == "<=" else -1.0 for lbl in labels])
        cost = sign * values
        limit = np.array([max(abs(goal_of[lbl].value), 1e-6) for lbl in labels])

        def trade_off(label: str) -> bool:
            g = goal_of[label]
            voi = self.aliases.get(g.voi, g.voi)
            target = voi in self.voi_names and self._voi(voi).voi_type == "TARGET"
            return target or not g.hard or g.metric == "mean"

        constraints = [lbl for lbl in labels if not trade_off(lbl)]
        if axes is None:
            spread = np.ptp(values[:n], axis=0)
            ax = [
                g
                for g, lbl in enumerate(labels)
                if trade_off(lbl) and spread[g] >= pareto.FLAT * limit[g]
            ]
        else:
            unknown = [a for a in axes if a not in labels]
            if unknown:
                raise PlanningError(f"Unknown goals {unknown}. Goals: {labels}")
            ax = [labels.index(a) for a in dict.fromkeys(axes)]
        if not ax:
            raise PlanningError(
                "The plans differ on no goal by >= 1 % of its limit: optimize plans with "
                "different trade-offs first."
            )

        def entry(s: int) -> dict:
            e = evals[s]
            return {
                # As sampled: blend_plans normalizes them the same way
                "weights": [round(float(w), 2) for w in weights[s]],
                "score": e["score"],
                "all_hard_passed": e["all_hard_passed"],
                "num_failed": e["num_failed"],
                "values": {labels[g]: round(float(values[s, g]), 3) for g in ax},
            }

        out: dict = {
            "plans": list(plans),
            "num_blends": len(weights),
            "constraint_goals": [] if axes is not None else constraints,
            "flat_goals": []
            if axes is not None
            else [lbl for g, lbl in enumerate(labels) if g not in ax and lbl not in constraints],
        }
        feasible = np.flatnonzero([e["all_hard_passed"] for e in evals])
        out["num_passing_hard"] = len(feasible)
        if not feasible.size:
            best = sorted(range(len(evals)), key=lambda s: evals[s]["score"], reverse=True)
            out["front"] = [entry(s) for s in best[:top_n]]
            out["note"] = "No blend passes all hard goals; front holds the best blends by score."
            self._log("pareto_front", plans=list(plans), front=0)
            return out

        c = cost[feasible][:, ax]
        front = feasible[pareto.epsilon_front(c, pareto.EPS * limit[ax])]
        fc = cost[front][:, ax]
        best, worst = fc.min(axis=0), fc.max(axis=0)
        regret, picked = pareto.spread_picks(fc, top_n)
        compromise = picked[0]

        out["axes"] = [
            {
                "goal": labels[g],
                "best": round(float(sign[g] * best[j]), 3),
                "worst": round(float(sign[g] * worst[j]), 3),
                "plans": {p: round(float(values[i, g]), 3) for i, p in enumerate(plans)},
            }
            for j, g in enumerate(ax)
        ]
        out["front_size"] = len(front)
        out["front"] = [
            {
                **entry(int(front[k])),
                "best_on": [labels[g] for j, g in enumerate(ax) if regret[k, j] == 0],
            }
            for k in picked
        ]
        order = np.argsort(-regret[compromise])
        out["compromise"] = {
            **out["front"][0],
            "weakest_axes": [
                {
                    "goal": labels[ax[j]],
                    "pct_of_range_from_best": round(100 * regret[compromise, j], 1),
                }
                for j in order[:2]
                if regret[compromise, j] > 0
            ],
        }
        # Plain Pareto dominance, no epsilon: one blend at least as good on every axis and
        # better on one is enough, however small the difference
        beaten = pareto.dominated(cost[:n][:, ax], c)
        out["dominated_plans"] = [p for i, p in enumerate(plans) if beaten[i]]
        out["plans_failing_hard"] = [
            p for i, p in enumerate(plans) if not evals[i]["all_hard_passed"]
        ]
        self.last_pareto = out
        self._log("pareto_front", plans=list(plans), front=len(front))
        self._notify("pareto", result=out)
        return out

"""Evaluation of plans against a protocol, plan scores and the plans a run keeps."""

from typing import Any, Mapping, Optional

import numpy as np
from pydantic import BaseModel, Field

from pyRadPlan.cst import StructureSet
from pyRadPlan.result import PlanResult

from ._constraints import D_RE, V_RE, Protocol, qi_key

# Relative margins (margin / goal value) are clipped to this range in the score, so one
# goal far inside its limit cannot hide others failing
_SCORE_CLIP = (-1.0, 0.2)
_HARD_WEIGHT = 3.0
#: A further plan of select_distinct_plans must beat each pick by this relative margin
DISTINCT_MIN_GAIN = 0.05


def evaluate_protocol(
    protocol: Protocol, qis: Mapping[str, Any], aliases: Optional[dict[str, str]] = None
) -> dict:
    """Evaluate *protocol* on *qis*; returns a compact, LLM-friendly report.

    *qis* maps VOI names to their QIs by metric key, each with a ``value`` (e.g. a
    :class:`~pyRadPlan.analysis.QICollection`). *aliases* maps protocol structure
    names to VOI names. Goals whose structure cannot be found are not evaluated and do
    not count in the score; they are listed under ``unresolved_goals`` (their structures
    under ``unresolved_structures``).
    """
    aliases = aliases or {}
    rows = []
    unresolved = []
    for g in protocol.goals:
        voi = aliases.get(g.voi, g.voi)
        if voi not in qis:
            unresolved.append(g)
            continue
        value = float(qis[voi][qi_key(g.metric)].value)
        margin = value - g.value if g.op == ">=" else g.value - value
        rows.append(
            {
                "goal": g.label,
                "hard": g.hard,
                "value": round(value, 3),
                "margin": round(margin, 3),  # >= 0 means passed
                "rel_margin": round(margin / max(abs(g.value), 1e-6), 4),
                "passed": margin >= 0,
            }
        )
    return {
        "all_hard_passed": all(r["passed"] for r in rows if r["hard"]),
        "num_failed": sum(not r["passed"] for r in rows),
        "score": plan_score(rows),
        "unresolved_structures": sorted({g.voi for g in unresolved}),
        "unresolved_goals": [g.label for g in unresolved],
        "goals": rows,
    }


def voxel_metric(voxels: np.ndarray, key: str) -> float:
    """Compute QI *key* (``mean``, ``max``, ``D95``, ``V20Gy``, ...) of voxel values.

    Computed like :class:`~pyRadPlan.analysis.QICollection` does, without building QI
    models: :meth:`PlanningState.pareto_front` evaluates hundreds of blends per call.
    """
    if not voxels.size:
        return float("nan")
    if m := D_RE.match(key):
        return float(np.percentile(voxels, 100 - float(m.group(1))))
    if m := V_RE.match(key):
        return float((voxels >= float(m.group(1))).sum() / voxels.size * 100.0)
    return float({"mean": np.mean, "max": np.max, "min": np.min, "std": np.std}[key](voxels))


def plan_score(goal_rows: list[dict]) -> float:
    """Score a plan from its evaluated goals: higher is better, >= 0 when all goals pass.

    Weighted mean of the relative goal margins (margin / goal value), each clipped to
    [-100 %, +20 %], in percent; hard goals weigh three times as much as soft ones.
    """
    lo, hi = _SCORE_CLIP
    num = den = 0.0
    for r in goal_rows:
        w = _HARD_WEIGHT if r["hard"] else 1.0
        num += w * min(max(r["rel_margin"], lo), hi)
        den += w
    return round(100 * num / den, 2) if den else 0.0


class PlanRecord(BaseModel):
    """One plan of a planning agent run: an optimization or a blend of earlier plans."""

    index: int = Field(description="Plan number, counted over optimizations and blends.")
    evaluation: dict = Field(description="Goal evaluation (see evaluate_protocol).")
    beams: dict[str, list] = Field(
        default_factory=lambda: {"gantry_angles": [], "couch_angles": []}
    )
    result: Optional[PlanResult] = Field(None, description="Dose and fluence on the CT grid.")
    quantity: str = Field("physical_dose", description="The quantity the goals are judged on.")
    cst: Optional[StructureSet] = Field(None, description="Structures and objectives.")
    objectives: dict[str, list[dict]] = Field(
        default_factory=dict, description="Objectives per VOI as the agent sees them."
    )
    run: Optional[dict] = Field(None, description="Optimizer settings; None for a blend.")
    blend: Optional[dict] = Field(None, description="Source plans and weights of a blend.")

    @property
    def dose(self) -> Any:
        """Image of :attr:`quantity`, None without a result."""
        return None if self.result is None else self.result[self.quantity].values

    @property
    def fluence(self) -> Optional[np.ndarray]:
        """Fluence of the plan, None without a result."""
        return None if self.result is None else self.result.fluence

    def rel_margins(self) -> dict[str, float]:
        """Relative margin of every evaluated goal, by goal label."""
        return {r["goal"]: r.get("rel_margin", -1.0) for r in self.evaluation["goals"]}


def select_distinct_plans(
    plans: list[PlanRecord], n: int = 2, min_gain: float = DISTINCT_MIN_GAIN
) -> list[PlanRecord]:
    """Pick up to *n* good plans that are each good in a different way.

    Plans are ranked by (all hard goals passed, score) and the best one is picked first.
    Each further plan must beat every plan picked so far by at least *min_gain* (relative
    margin, 0.05 = 5 % of the goal value) on at least one goal. Returns fewer than *n*
    plans if there are no such alternatives, none if *plans* is empty.
    """
    ranked = sorted(
        plans,
        key=lambda p: (p.evaluation["all_hard_passed"], p.evaluation["score"]),
        reverse=True,
    )
    picks: list[PlanRecord] = []
    for p in ranked:
        if len(picks) == n:
            break
        p_rel = p.rel_margins()
        if all(
            any(v - q.rel_margins().get(g, np.inf) >= min_gain for g, v in p_rel.items())
            for q in picks
        ):
            picks.append(p)
    return picks

"""The planning state: everything the agent works on, with one method per tool.

All tools are plain methods on :class:`PlanningState`, so they can be tested and used
without any LLM; :mod:`._agent` registers them with pydantic-ai.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from pyRadPlan.core import ComputeControl
from pyRadPlan.cst import StructureSet
from pyRadPlan.ct import CT
from pyRadPlan.dij import Dij
from pyRadPlan.plan import Plan, validate_pln

from ._constraints import Protocol
from ._evaluation import PlanRecord
from ._state_beams import BeamsMixin
from ._state_optimize import OptimizeMixin
from ._state_pareto import ParetoMixin
from ._state_protocol import ProtocolMixin
from ._state_structures import StructuresMixin

# Kept importable from here; the tools that use them live in the mixin modules
from ._state_base import (  # noqa: F401
    _NO_DOSE,
    NORMAL_TISSUE,
    PlanningError,
    UnknownVoiError,
    sitk_array,
)
from ._state_beams import _CROSSING_FRACTION, _beam_subset, _setup_text  # noqa: F401
from ._state_optimize import (  # noqa: F401
    _STALL_WINDOW,
    _cpu_backend,
    _dose_is_linear,
    _solver_name,
)
from ._state_protocol import (  # noqa: F401
    _FALLOFF_OK,
    _FALLOFF_STEEP,
    _MEAN_TIGHT,
    _NEAR_MAX_PCT,
)

logger = logging.getLogger(__name__)


@dataclass
class PlanningState(StructuresMixin, ProtocolMixin, BeamsMixin, OptimizeMixin, ParetoMixin):
    """Everything the agent works on. Lives in Python, never in the chat.

    The plan is copied, so setting beams never changes the caller's plan: read the
    beams from ``state.pln``.
    """

    ct: CT
    cst: StructureSet
    pln: Plan
    protocol: Protocol
    max_dij_calcs: int = 3
    # Limits of set_beams and optimize, set by the user
    min_beams: int = 1
    max_beams: int = 30
    allow_couch: bool = True
    max_iter: int = 5000
    max_optimizations: int = 8
    # Blends kept as plans (each holds a CT-grid dose); pareto_front keeps none
    max_blends: int = 20
    max_pareto_samples: int = 2000
    # set_beams is refused once this many optimizations are used (None: never)
    lock_beams_after: Optional[int] = None
    # Comparing beam setups: at most max_beam_setups get optimizations, each up to
    # trial_optimizations_per_setup; one more makes that setup final (None: no limit)
    max_beam_setups: Optional[int] = None
    trial_optimizations_per_setup: Optional[int] = None
    # A setup becomes final only once this many setups had all their trial optimizations
    min_beam_setups: Optional[int] = None
    # A new setup must differ from every compared one in at least this many beams
    min_setup_difference: Optional[int] = None
    # Planner reviews (see review_reason): every this many optimizations, 0 never
    review_every: int = 0
    # NormalTissue (implicit goal, QIs) is the patient this far outside the targets
    normal_tissue_margin_mm: float = 5.0
    # Solver of every optimize, over the plan's prop_opt (None: the plan's or the default)
    solver: Optional[str] = "scipy"
    # Optimize with numpy on the CPU: the same inputs give the same plan
    cpu_only: bool = True

    stf: Any = None
    dij: Optional[Dij] = None
    dose: Any = None
    last_evaluation: Optional[dict] = None
    history: list[dict] = field(default_factory=list)
    num_dij_calcs: int = 0
    num_optimizations: int = 0
    aliases: dict[str, str] = field(default_factory=dict)  # protocol name -> VOI name
    excluded: list[str] = field(default_factory=list)
    snapshots: dict[str, dict] = field(default_factory=dict)
    created_helpers: list[str] = field(default_factory=list)
    plans: list[PlanRecord] = field(default_factory=list)  # every optimization and blend
    last_optimizer: Optional[dict] = None  # see optimizer_report
    last_pareto: Optional[dict] = None  # see pareto_front
    # set by a host (e.g. the GUI): a stop request ends the run at the next tool call
    control: Optional[ComputeControl] = None
    # called as listener(event, **data) on tool calls, dij/optimize phases and new plans
    listener: Optional[Callable[..., None]] = None
    num_compactions: int = 0  # see compact_after of create_planning_agent
    # Set by a host: pause before the next optimize / set_beams
    pause_requested: bool = False
    finish_requested: bool = False  # the planner ended the run: only the report is left
    feedback: Optional[str] = None  # planner feedback for the next optimize result
    _reviewed_at: int = 0  # num_optimizations at the last review
    _reviewed_budget: int = 0  # max_optimizations at the last review
    _reviewed_hard_pass: bool = False
    _dij_region: Optional[np.ndarray] = None  # CT-grid voxels covered by the dij
    _normal_tissue: Optional[tuple] = None  # (cache key, mask), see _normal_tissue_region
    # Candidate beams with one dose influence (set_beam_pool): gantry_angles,
    # couch_angles, stf, dij, region
    beam_pool: Optional[dict] = None
    _influence: Optional[tuple] = None  # (dij, VOI names, rows), see beam_influence
    # Plans whose dose is not linear in the fluence (variable RBE), see pareto_front
    _nonlinear_plans: set = field(default_factory=set)

    def __post_init__(self) -> None:
        self.pln = validate_pln(self.pln).model_copy(deep=True)

    # ---- helpers ---------------------------------------------------------------------
    @property
    def voi_names(self) -> list[str]:
        return [v.name for v in self.cst.vois]

    def _log(self, action: str, **info: Any) -> None:
        self.history.append({"step": len(self.history) + 1, "action": action, **info})

    def _notify(self, event: str, **data: Any) -> None:
        if self.listener is None:
            return
        try:
            self.listener(event, **data)
        except Exception:  # a broken listener must not end the run
            logger.exception("Planning listener failed on %s", event)

    # ---- planner reviews -------------------------------------------------------------
    def review_reason(self, tool: str) -> Optional[str]:
        """Why the planner should review before *tool* runs, None to go on.

        'pause' (requested), and before an optimize with ``review_every`` set: 'budget'
        (used up), 'hard_goals' (all passed the first time), 'review' (every N plans).
        """
        if self.pause_requested:
            return "pause"
        if tool != "optimize" or not self.review_every:
            return None
        n = self.num_optimizations
        if n >= self.max_optimizations and self._reviewed_budget != self.max_optimizations:
            return "budget"
        if not self._reviewed_hard_pass and self._any_plan_passes():
            return "hard_goals"
        if n - self._reviewed_at >= self.review_every:
            return "review"
        return None

    def mark_reviewed(self) -> None:
        """Record a review: the reasons that applied now are not raised again."""
        self.pause_requested = False
        self._reviewed_at = self.num_optimizations
        self._reviewed_budget = self.max_optimizations
        self._reviewed_hard_pass = self._any_plan_passes()

    def _any_plan_passes(self) -> bool:
        return any(p.evaluation["all_hard_passed"] for p in self.plans)

    # ---- planning tools ----------------------------------------------------------------
    def case_overview(self) -> dict:
        """Summarize plan settings, protocol, aliases, helpers, beams, objectives, budget."""
        return {
            "radiation_mode": self.pln.radiation_mode,
            "fractionation": self._fractionation(),
            "protocol": [g.tagged_label for g in self.protocol.goals],
            "protocol_locked": self.protocol.locked,
            "structure_aliases": self.aliases,
            "normal_tissue": f"patient more than {self.normal_tissue_margin_mm:g} mm outside "
            "the targets",
            "helpers": self.created_helpers,
            "overlap_priorities": {
                "note": "Overlapping voxels count only for the objectives of the VOI with the "
                "lower number (ties: both); goals are evaluated on the full VOIs.",
                **{v.name: v.overlap_priority for v in self.cst.vois},
            },
            "current_beams": self.current_beams(),
            "current_objectives": self.current_objectives(),
            "budget": {
                "dij_calcs_left": self.max_dij_calcs - self.num_dij_calcs,
                "optimizations_left": self.max_optimizations - self.num_optimizations,
                "beams": f"{self.min_beams}-{self.max_beams}"
                + ("" if self.allow_couch else ", coplanar only (couch 0)"),
                "max_iter": self.max_iter,
                **self._beam_budget(),
            },
        }

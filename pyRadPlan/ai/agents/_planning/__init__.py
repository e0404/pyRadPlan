"""The tool-using planning agent.

- :mod:`._state`: :class:`PlanningState` with one method per tool (usable without an LLM)
- :mod:`._agent`: the pydantic-ai agent, its prompt, runs and planner reviews
- :mod:`._evaluation`: protocol evaluation, plan scores and the kept :class:`PlanRecord`
- :mod:`._constraints`: goals, constraint tables and EQD2 conversion
- :mod:`._pareto`, :mod:`._geometry`, :mod:`._recipes`, :mod:`._views`:
  Pareto math, mask geometry, helper recipes and the views of the tool results the model sees
"""

from ._agent import (
    PlanningReport,
    compact_history,
    continue_planning_agent,
    create_planning_agent,
    planning_tools,
    run_planning_agent,
)
from ._constraints import (
    Goal,
    GoalFile,
    Protocol,
    available_constraint_tables,
    convert_fractionation,
    eqd2,
    load_constraint_table,
    load_goals,
    physical_from_eqd2,
    save_goals,
    site_goals,
)
from ._evaluation import PlanRecord, evaluate_protocol, plan_score, select_distinct_plans
from ._recipes import BooleanRecipe, ExcludeSpec, IsodoseRecipe, MarginRecipe, RingRecipe
from ._state import PlanningError, PlanningState, UnknownVoiError

__all__ = [
    "BooleanRecipe",
    "ExcludeSpec",
    "Goal",
    "GoalFile",
    "IsodoseRecipe",
    "MarginRecipe",
    "PlanRecord",
    "PlanningError",
    "PlanningReport",
    "PlanningState",
    "Protocol",
    "RingRecipe",
    "UnknownVoiError",
    "available_constraint_tables",
    "compact_history",
    "continue_planning_agent",
    "convert_fractionation",
    "create_planning_agent",
    "eqd2",
    "evaluate_protocol",
    "load_constraint_table",
    "load_goals",
    "physical_from_eqd2",
    "plan_score",
    "planning_tools",
    "run_planning_agent",
    "save_goals",
    "select_distinct_plans",
    "site_goals",
]

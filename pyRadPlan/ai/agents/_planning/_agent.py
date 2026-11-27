"""The pydantic-ai agent around :class:`PlanningState`: tools, prompt, runs and reviews.

Unlike :func:`generate_beam_angles` / :func:`generate_voi_objectives` (one structured LLM
call each), this agent runs a loop: the model picks a tool, pyRadPlan executes it, the
model sees the (compact) result and decides what to do next, until it returns a
:class:`PlanningReport`.
"""

import copy
import dataclasses
import functools
import inspect
import logging
from typing import Annotated, Any, Callable, Collection, Literal, Optional

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import (
    Agent,
    ApprovalRequired,
    DeferredToolRequests,
    DeferredToolResults,
    ModelRetry,
    RunContext,
    ToolDefinition,
    ToolDenied,
    capture_run_messages,
)
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelRequest, ToolReturnPart
from pydantic_ai.usage import RunUsage, UsageLimits
from pydantic_core import to_json

from pyRadPlan.core import ComputeCancelledError
from pyRadPlan.optimization.objectives import get_objectives_union

from .._models import resolve_model
from .._usage import log_run_usage
from ._constraints import Protocol
from ._recipes import HelperRecipe
from ._state import PlanningError, PlanningState
from ._views import (
    model_beam_influence,
    model_comparison,
    model_evaluation,
    model_optimizer_report,
    model_pareto,
    slim_schema,
    with_hidden_defaults,
)

logger = logging.getLogger(__name__)

# Tools that wait for the planner when a review is due (see PlanningState.review_reason)
_REVIEW_TOOLS = frozenset({"optimize", "set_beams"})
# Tools refused once the planner ended the run
_PRODUCING_TOOLS = frozenset({"optimize", "blend_plans"})
# Pending call results of continue_planning_agent, by action
_DENIALS = {
    "redirect": "Not run: the planner reviewed the plans and says: {}",
    "finish": "Not run: the planner ends the run. Write your final report now. {}",
}
_FINISH = "The planner ended the run. Return your final report now."
# Tool calls allowed per run_planning_agent / continue_planning_agent call unless given:
# this many, plus this many per optimization of the budget
_TOOL_CALLS_BASE = 40
_TOOL_CALLS_PER_OPTIMIZATION = 20


class PlanningRunError(Exception):
    """A planning run ended early without a report; what it did so far is kept.

    Raised by :func:`run_planning_agent` and :func:`continue_planning_agent` when the
    pydantic-ai run fails: a usage limit is reached (requests, tool calls), the model
    misbehaves (e.g. keeps sending invalid arguments) or the provider / connection fails.
    The original exception is the ``__cause__``. The :class:`PlanningState` keeps its
    plans; pass this error to :func:`continue_planning_agent` in place of a result to go
    on from where the run stopped. A stop request (``ComputeCancelledError``) is not
    wrapped.

    Attributes
    ----------
    messages : list[ModelMessage]
        The whole conversation up to the failure.
    usage : RunUsage
        Usage over all calls so far.
    output : None
        No report (same attribute as a run result).
    """

    def __init__(self, messages: list[ModelMessage], usage: RunUsage, cause: BaseException):
        super().__init__(
            f"The planning run stopped early ({type(cause).__name__}: {cause}). "
            "Continue it with continue_planning_agent."
        )
        self.messages = messages
        self.usage = usage
        self.output = None

    def all_messages(self) -> list[ModelMessage]:
        """Return the conversation, as a run result does."""
        return self.messages

    @property
    def limit_exceeded(self) -> bool:
        """Whether a usage limit ended the run, so continuing with a new limit goes on."""
        return isinstance(self.__cause__, UsageLimitExceeded)


class PlanningReport(BaseModel):
    """Final answer of the planning agent."""

    all_hard_goals_met: bool
    summary: str = Field(description="What the final plan looks like, in 2-4 sentences.")
    trade_offs: list[str] = Field(
        default_factory=list, description="Trade-offs made and why, one per item."
    )
    suggestions_for_planner: list[str] = Field(
        default_factory=list, description="What a human planner should check or try next."
    )
    # Filled by pyRadPlan, not by the model: goals may be mapped to any clinical VOI
    goal_aliases: SkipJsonSchema[dict[str, str]] = Field(
        default_factory=dict,
        description="Protocol structure name -> VOI its goals were evaluated on.",
    )


PLANNING_SYSTEM_PROMPT = """
You are a radiotherapy treatment planning agent working in pyRadPlan.
Goal: a plan that passes all hard goals of the protocol and is as good as possible on
soft goals. Numbers (constraints, conversions, doses) must come from tools, never from
memory. Doses follow the plan's dose_convention (default per fraction).

Work like an experienced planner:
1. Structures: inspect_structures. Names are messy (languages, abbreviations, laterality,
   helper/old copies, near-duplicates). map_structures maps every protocol name to exactly
   one clinical VOI, fixes wrong types and excludes junk. If a mapping is genuinely
   ambiguous, say so in the report instead of guessing.
2. Goals: if the protocol is empty and unlocked, use list_constraint_tables and
   get_site_constraints for the site and fractionation. Map its structure names with
   map_structures, then get_site_constraints(apply=True); goals on unmapped names are dropped.
3. Anatomy: case_overview, structure_geometry, analyze_target_shape and check_feasibility.
   Conflicts (target/OAR overlap with incompatible goals) cannot be optimized away: split
   the target with subtract/intersect helpers and a lower goal in the overlap, or report it.
4. Beams: set_beams within the beam limits of the budget (case_overview), then
   beam_influence shows which beams run through OARs. Dose influence calculations are expensive
   and limited (every beam change needs one, except among candidate beams in
   case_overview); settle beams before tuning.
5. Objectives: set_objectives per VOI, optimize, read the goal margins. Tissue outside the
   targets is healthy even where no goal names it: read the NormalTissue QIs like an OAR's
   (e.g. D5 close to the prescription means a large volume gets target dose) and protect
   it with objectives. Use helpers only where the shape analysis supports them.
6. Iterate: change one or two things per step and say why. optimizer_report shows whether
   the optimizer converged and which objectives dominate (the weight to change);
   localize_dose_spots shows where a goal fails. plan_versions saves good states,
   compares a change against them and restores one if a change made things worse.
   After a small change, optimize(start_from=<plan>) continues from that plan's fluence.
7. Use the optimization budget: optimizations_left (case_overview, every optimize result)
   says how many remain. Passing all hard goals is not the end: approximate the Pareto
   front of the trade-offs, the plans where no goal improves without another getting
   worse.
   a) Anchors: from a plan passing the hard goals, optimize one plan per trade-off (e.g.
      target coverage, cord / brainstem, each parotid, larynx, body mean), each pushing
      its own goals hard while keeping the hard goals. Make them clearly different, not
      small variations of each other.
   b) pareto_front(plans=[...]) blends them for free and returns the front, the
      compromise blend and its weakest axes. Raw goal values count there, so it also
      shows sparing the score cannot (margins above +20 % do not raise the score).
   c) Spend each further optimization where the front is weakest: blend_plans with the
      compromise weights, optimize(start_from=<that blend>) with the weakest axes
      tightened, and pass the result to pareto_front with the others. A plan under
      dominated_plans extended nothing: try another direction instead of repeating it.
      Flat goals are trade-offs no plan explored yet: anchor one if it matters.
   d) Deliver plans from the front (blend_plans turns a front blend into a plan).
   Stop early only when several optimizations in a row extended nothing and you can name
   why the remaining failed goals cannot be reached. Report honestly, including failed
   goals and trade-offs.

The user's task overrides these working rules where they differ (e.g. how long to
iterate, which plans to deliver), but never the budgets or the hard goals.

A human planner may review between steps. Their feedback overrides your own plan; the
goals stay binding unless the planner changes them. Mention the feedback and how you
acted on it in the report.
"""


async def _prepare_slim_schema(
    ctx: RunContext[PlanningState], tool_def: ToolDefinition
) -> ToolDefinition:
    return dataclasses.replace(
        tool_def, parameters_json_schema=slim_schema(tool_def.parameters_json_schema)
    )


# Results that stay valid while optimizing (anatomy) are never compacted
_DURABLE_TOOLS = frozenset({"analyze_target_shape", "check_feasibility", "structure_geometry"})
# Smaller results (confirmations of set_beams, set_objectives, ...) are kept as they are
_COMPACT_MIN_CHARS = 300


def _compact_return(part: ToolReturnPart, step: int) -> ToolReturnPart:
    content = part.content
    if part.tool_name in _DURABLE_TOOLS or len(to_json(content)) < _COMPACT_MIN_CHARS:
        return part
    if part.tool_name in _PRODUCING_TOOLS and isinstance(content, dict) and "goals" in content:
        keep = (
            "plan",
            "blend",
            "all_hard_passed",
            "num_failed",
            "score",
            "goals",
            "normal_tissue",
            "optimizations_left",
        )
        digest = {k: content[k] for k in keep if k in content}
        digest["note"] = "Compacted: optimizer details dropped."
        return dataclasses.replace(part, content=digest)
    return dataclasses.replace(
        part,
        content=f"[Compacted: this {part.tool_name} result (before plan {step}) is outdated. "
        "Call the tool again if you need it.]",
    )


def compact_history(messages: list[ModelMessage], step: int) -> list[ModelMessage]:
    """Shorten tool results before the request holding the latest optimize result.

    Optimization results keep score and goal lines, anatomy results and small results
    stay, everything else becomes a short note. Tool calls, the model's own text and
    retry prompts are kept, so the model still knows what it tried.
    """
    last = None
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, ToolReturnPart) and p.tool_name == "optimize" for p in msg.parts
        ):
            last = i
    if last is None:
        return messages
    out = []
    for i, msg in enumerate(messages):
        if i < last and isinstance(msg, ModelRequest):
            parts = [
                _compact_return(p, step) if isinstance(p, ToolReturnPart) else p for p in msg.parts
            ]
            out.append(dataclasses.replace(msg, parts=parts))
        else:
            out.append(msg)
    return out


def _tool_decorator(agent: Agent, tools: Optional[Collection[str]], registered: set) -> Callable:
    """Return the decorator registering a planning tool with *agent* (if in *tools*)."""

    def tool(method: Any = None, description: Optional[str] = None, prepare: Any = None):
        # Tools share the state, so calls of one response run in order, not in parallel.
        # Errors reach the model as results instead of ending the run: a PlanningError
        # (a refusal the model can act on) and any other failure. Neither uses up the
        # retries, which are left to arguments failing validation. A stop request ends
        # the run; once the planner ended it, no further plans are made.
        def register(fn):
            registered.add(fn.__name__)
            if tools is not None and fn.__name__ not in tools:
                return fn

            @functools.wraps(fn)
            def safe(*args, **kwargs):
                state: PlanningState = args[0].deps
                if state.control is not None and state.control.stop_requested:
                    raise ComputeCancelledError("The planning agent was stopped on request.")
                if fn.__name__ in _REVIEW_TOOLS and not args[0].tool_call_approved:
                    if reason := state.review_reason(fn.__name__):
                        state._notify("review", name=fn.__name__, reason=reason)
                        raise ApprovalRequired(metadata={"reason": reason})
                state._notify("tool", name=fn.__name__, args=kwargs)
                error = None
                try:
                    if state.finish_requested and fn.__name__ in _PRODUCING_TOOLS:
                        raise PlanningError(_FINISH)
                    return fn(*args, **kwargs)
                except (ModelRetry, ComputeCancelledError) as e:
                    error = str(e)
                    raise
                except PlanningError as e:
                    error = str(e)
                    return {"error": error, "refused": True}
                except Exception as e:
                    logger.exception("Planning tool %s failed", fn.__name__)
                    error = f"{type(e).__name__}: {e}"
                    return {
                        "error": error,
                        "note": "Unexpected failure inside pyRadPlan, not a wrong argument. "
                        "The state may be partly changed (check case_overview); do not "
                        "repeat the same call unchanged.",
                    }
                finally:
                    state._notify("tool_done", name=fn.__name__, error=error)

            return agent.tool(
                description=description or inspect.getdoc(method), sequential=True, prepare=prepare
            )(safe)

        return register

    return tool


def create_planning_agent(
    model: Any = None,
    tools: Optional[Collection[str]] = None,
    compact_after: int = 0,
) -> Agent:
    """Build the pydantic-ai agent with the planning tools registered.

    *tools* limits the agent to these tool names (all if omitted, see
    :func:`planning_tools`). Tool descriptions are the docstrings of the
    :class:`PlanningState` methods. The run ends with a :class:`PlanningReport`, or with
    ``DeferredToolRequests`` when the planner should review (see
    :meth:`PlanningState.review_reason` and :func:`continue_planning_agent`).

    A call the state refuses (:class:`PlanningError`: unknown VOI, budget used up, ...)
    comes back to the model as the result ``{"error": ..., "refused": True}``.

    *compact_after* > 0 compacts the message history every that many optimizations (see
    :func:`compact_history`), so later requests send fewer tokens. Each compaction
    rewrites the history, so a provider's prompt cache is rebuilt from the first
    shortened message on; 0 never compacts.
    """
    model = resolve_model(model)

    def compact(ctx: RunContext[PlanningState], messages: list[ModelMessage]):
        state = ctx.deps
        due = state.num_optimizations // compact_after
        if due <= state.num_compactions:
            return messages
        state.num_compactions = due
        compacted = compact_history(messages, len(state.plans))
        state._notify("compact", num_optimizations=state.num_optimizations)
        return compacted

    agent = Agent(
        model,
        deps_type=PlanningState,
        output_type=[PlanningReport, DeferredToolRequests],
        system_prompt=PLANNING_SYSTEM_PROMPT,
        retries=2,
        # Ignored by non-Anthropic models
        model_settings={
            "anthropic_cache_instructions": True,
            "anthropic_cache_tool_definitions": True,
            "anthropic_cache_messages": True,
        },
        capabilities=[ProcessHistory(compact)] if compact_after > 0 else None,
    )

    @agent.output_validator
    def add_goal_aliases(ctx: RunContext[PlanningState], output: Any) -> Any:
        if isinstance(output, PlanningReport):
            state = ctx.deps
            names = {g.voi for g in state.protocol.goals}
            output.goal_aliases = {
                name: voi for name, voi in state.aliases.items() if name in names and voi != name
            }
        return output

    registered: set[str] = set()
    tool = _tool_decorator(agent, tools, registered)
    _register_structure_tools(tool)
    _register_plan_tools(tool)
    unknown = set(tools or ()) - registered
    if unknown:
        raise ValueError(f"Unknown planning tools {sorted(unknown)}. Known: {sorted(registered)}")
    return agent


def _register_structure_tools(tool: Callable) -> None:
    """Register the tools on structures, anatomy, helpers and goals."""

    @tool(PlanningState.inspect_structures)
    def inspect_structures(ctx: RunContext[PlanningState]) -> dict:
        return ctx.deps.inspect_structures()

    @tool(PlanningState.map_structures)
    def map_structures(
        ctx: RunContext[PlanningState],
        aliases: Optional[dict[str, str]] = None,
        voi_types: Optional[dict[str, Literal["TARGET", "OAR", "HELPER", "EXTERNAL"]]] = None,
        exclude: Optional[list[str]] = None,
    ) -> dict:
        return ctx.deps.map_structures(aliases, voi_types, exclude)

    @tool(
        description=(
            "Plan versions. save: remember the current objectives, helpers, dose and "
            "evaluation under label. restore: return to all of these (beams must be "
            "unchanged); a version saved with a dose needs no new optimize. "
            "compare: goal values and mean/max dose per clinical VOI of compare_to "
            "('current' or a label) minus label."
        ),
    )
    def plan_versions(
        ctx: RunContext[PlanningState],
        action: Literal["save", "restore", "compare"],
        label: str,
        compare_to: str = "current",
    ) -> dict:
        if action == "save":
            return ctx.deps.snapshot(label)
        if action == "restore":
            out = ctx.deps.restore(label)
            return {**out, "evaluation": model_evaluation(out["evaluation"])}
        return model_comparison(ctx.deps.compare_snapshots(label, compare_to))

    @tool(PlanningState.analyze_target_shape)
    def analyze_target_shape(ctx: RunContext[PlanningState], target: Optional[str] = None) -> dict:
        return ctx.deps.analyze_target_shape(target)

    @tool(PlanningState.check_feasibility)
    def check_feasibility(ctx: RunContext[PlanningState]) -> dict:
        return ctx.deps.check_feasibility()

    @tool(PlanningState.create_helper, prepare=_prepare_slim_schema)
    def create_helper_structure(ctx: RunContext[PlanningState], recipe: HelperRecipe) -> dict:
        return ctx.deps.create_helper(recipe)

    @tool(PlanningState.delete_helper)
    def delete_helper_structure(ctx: RunContext[PlanningState], name: str) -> dict:
        return ctx.deps.delete_helper(name)

    @tool(PlanningState.localize_dose_spots)
    def localize_dose_spots(
        ctx: RunContext[PlanningState],
        kind: Literal["hot", "cold"],
        threshold: float,
        voi: Optional[str] = None,
    ) -> dict:
        return ctx.deps.localize_dose_spots(kind, threshold, voi)

    @tool(PlanningState.list_constraint_tables)
    def list_constraint_tables(ctx: RunContext[PlanningState]) -> dict:
        return ctx.deps.list_constraint_tables()

    @tool(PlanningState.get_site_constraints)
    def get_site_constraints(
        ctx: RunContext[PlanningState], table: str, site: str, apply: bool = False
    ) -> dict:
        return ctx.deps.get_site_constraints(table, site, apply)


def _register_plan_tools(tool: Callable) -> None:
    """Register the tools on beams, objectives, optimization and plans."""
    # Without the objectives created in code (outcome models): validating those loads
    # pickled estimators or downloads models, which no model output may trigger
    objective_union = get_objectives_union(exclude_image_references=True, selectable_only=True)

    @tool(PlanningState.case_overview)
    def case_overview(ctx: RunContext[PlanningState]) -> dict:
        return ctx.deps.case_overview()

    @tool(PlanningState.structure_geometry)
    def structure_geometry(ctx: RunContext[PlanningState]) -> dict:
        return ctx.deps.structure_geometry()

    @tool(PlanningState.set_beams)
    def set_beams(
        ctx: RunContext[PlanningState],
        gantry_angles: list[float],
        couch_angles: Optional[list[float]] = None,
    ) -> dict:
        return ctx.deps.set_beams(gantry_angles, couch_angles)

    @tool(PlanningState.beam_influence)
    def beam_influence(ctx: RunContext[PlanningState]) -> dict:
        return model_beam_influence(ctx.deps.beam_influence())

    @tool(PlanningState.set_objectives, prepare=_prepare_slim_schema)
    def set_objectives(
        ctx: RunContext[PlanningState], voi_name: str, objectives: list[objective_union]
    ) -> dict:
        return ctx.deps.set_objectives(voi_name, [with_hidden_defaults(o) for o in objectives])

    @tool(PlanningState.optimize)
    def optimize(
        ctx: RunContext[PlanningState],
        max_iter: Annotated[int, Field(ge=50, le=5000)] = 500,
        min_improvement: Annotated[float, Field(ge=0, le=0.1)] = 1e-2,
        start_from: Optional[int] = None,
    ) -> dict:
        state = ctx.deps
        feedback, state.feedback = state.feedback, None
        # A refused call carries the planner's feedback instead of dropping it
        note = f" Planner feedback: {feedback}" if feedback else ""
        try:
            evaluation = state.optimize(max_iter, min_improvement, start_from)
        except PlanningError as e:
            if not feedback:
                raise
            raise PlanningError(f"{e}{note}") from e
        except Exception:
            state.feedback = feedback  # not a refusal: keep it for the next optimize
            raise
        out = model_evaluation(evaluation, plan=state.plans[-1].index)
        out["optimizations_left"] = state.max_optimizations - state.num_optimizations
        return {"planner_feedback": feedback, **out} if feedback else out

    @tool(PlanningState.blend_plans)
    def blend_plans(
        ctx: RunContext[PlanningState], plans: list[int], weights: list[float]
    ) -> dict:
        state = ctx.deps
        return model_evaluation(state.blend_plans(plans, weights), plan=state.plans[-1].index)

    @tool(PlanningState.pareto_front)
    def pareto_front(
        ctx: RunContext[PlanningState],
        plans: list[int],
        axes: Optional[list[str]] = None,
        n_samples: Annotated[int, Field(ge=20, le=2000)] = 300,
        top_n: Annotated[int, Field(ge=1, le=12)] = 6,
    ) -> dict:
        return model_pareto(ctx.deps.pareto_front(plans, axes, n_samples, top_n))

    @tool(PlanningState.optimizer_report)
    def optimizer_report(ctx: RunContext[PlanningState]) -> dict:
        return model_optimizer_report(ctx.deps.optimizer_report())

    @tool(PlanningState.analyze_qis)
    def analyze_qis(
        ctx: RunContext[PlanningState],
        vois: Optional[list[str]] = None,
        ref_vols: Optional[list[float]] = None,
        ref_doses: Optional[list[float]] = None,
    ) -> dict:
        return ctx.deps.analyze_qis(vois, ref_vols, ref_doses)


def planning_tools() -> dict[str, str]:
    """Name and description of every planning tool, in registration order."""
    from pydantic_ai.models.test import TestModel  # noqa: PLC0415

    toolset = create_planning_agent(TestModel()).toolsets[0]
    return {name: t.description for name, t in toolset.tools.items()}


def run_planning_agent(
    state: PlanningState,
    task: str,
    model: Any = None,
    max_optimizations: Optional[int] = None,
    request_limit: int = 40,
    tool_calls_limit: Optional[int] = None,
    tools: Optional[Collection[str]] = None,
    compact_after: int = 0,
):
    """Run the agent on *state* and return the pydantic-ai result.

    ``.output`` is the :class:`PlanningReport`, or ``DeferredToolRequests`` if the run
    paused for a review (resume with :func:`continue_planning_agent`). *state* holds
    plan, objectives, dose and history; what an earlier run left of the planner's end
    of the run and feedback is cleared.

    *max_optimizations* replaces the budget of *state* if given. *request_limit* caps
    the model requests and *tool_calls_limit* the tool calls of this call (default:
    derived from the optimization budget). Reaching a limit, or any other failure of
    the run, raises a :class:`PlanningRunError` holding the conversation so far, which
    :func:`continue_planning_agent` takes in place of a result.
    """
    if max_optimizations is not None:
        state.max_optimizations = max_optimizations
    state.finish_requested = False
    state.feedback = None
    agent = create_planning_agent(model, tools=tools, compact_after=compact_after)
    return _run(agent, state, task, (request_limit, tool_calls_limit), "run_planning_agent")


def continue_planning_agent(
    state: PlanningState,
    previous: Any,
    prompt: str = "",
    action: Literal["continue", "redirect", "finish"] = "continue",
    model: Any = None,
    request_limit: int = 40,
    tools: Optional[Collection[str]] = None,
    compact_after: int = 0,
    protocol: Optional[Protocol] = None,
    tool_calls_limit: Optional[int] = None,
):
    """Continue the run that returned (or raised) *previous* on the same *state*.

    After a pause, *action* decides the pending call: ``continue`` runs it with *prompt*
    as planner feedback, ``redirect`` skips it and passes *prompt* on, ``finish`` asks
    for the final report. After a report, *prompt* is the next task. After a
    :class:`PlanningRunError`, the run goes on where it stopped, *prompt* (optional) is
    passed to the model and ``finish`` asks for the report. A changed *protocol*
    re-evaluates every plan and is reported to the agent. Budgets are those of *state*;
    *request_limit* and *tool_calls_limit* count the requests and tool calls of this
    call, so a continued run has them anew.
    """
    notes = []
    if protocol is not None and (changes := state.set_protocol(protocol)):
        notes.append("The planner changed the goals: " + "; ".join(changes) + ".")
    text = " ".join([*notes, prompt.strip()]).strip()
    state.mark_reviewed()
    state.finish_requested = action == "finish"
    agent = create_planning_agent(model, tools=tools, compact_after=compact_after)
    history = previous.all_messages()
    pending = previous.output
    limits = (request_limit, tool_calls_limit)
    if isinstance(previous, PlanningRunError):
        if action == "finish":
            text = f"{_FINISH} {text}".strip()
        # A request the model has not answered yet goes on without a new prompt
        if not text and not (history and isinstance(history[-1], ModelRequest)):
            text = "Continue."
        return _run(
            agent, state, text or None, limits, "continue_planning_agent", previous, history
        )
    if not isinstance(pending, DeferredToolRequests):
        if not text:
            raise ValueError("Give a prompt to continue a finished run.")
        return _run(agent, state, text, limits, "continue_planning_agent", previous, history)

    approvals: dict[str, Any] = {}
    for call in pending.approvals:
        if tools is not None and call.tool_name not in tools:
            approvals[call.tool_call_id] = ToolDenied("Not run: the planner disabled this tool.")
        elif action == "continue":
            approvals[call.tool_call_id] = True
        else:
            approvals[call.tool_call_id] = ToolDenied(_DENIALS[action].format(text).strip())
    if action == "continue" and text:
        if "optimize" in (c.tool_name for c in pending.approvals):
            state.feedback = text
        else:  # no optimize result to carry it: deny the call with the feedback instead
            approvals = {
                c.tool_call_id: ToolDenied(_DENIALS["redirect"].format(text))
                for c in pending.approvals
            }
    return _run(
        agent,
        state,
        None,
        limits,
        "continue_planning_agent",
        previous,
        history,
        DeferredToolResults(approvals=approvals),
    )


def _run(agent, state, prompt, limits, operation, previous=None, history=None, results=None):
    # The limits count this call's requests and tool calls; the usage adds up over all calls
    request_limit, tool_calls_limit = limits
    if tool_calls_limit is None:
        tool_calls_limit = (
            _TOOL_CALLS_BASE + _TOOL_CALLS_PER_OPTIMIZATION * state.max_optimizations
        )
    usage = copy.deepcopy(previous.usage) if previous is not None else RunUsage()
    model_name = str(getattr(agent.model, "model_name", agent.model))
    with capture_run_messages() as messages:
        try:
            result = agent.run_sync(
                prompt,
                deps=state,
                message_history=history,
                deferred_tool_results=results,
                usage=usage,
                usage_limits=UsageLimits(
                    request_limit=usage.requests + request_limit,
                    tool_calls_limit=usage.tool_calls + tool_calls_limit,
                ),
            )
        except ComputeCancelledError:
            raise
        except Exception as e:
            error = PlanningRunError(list(messages) or list(history or []), usage, e)
            log_run_usage(error, model_name, operation=operation)
            raise error from e
    log_run_usage(result, model_name, operation=operation)
    return result

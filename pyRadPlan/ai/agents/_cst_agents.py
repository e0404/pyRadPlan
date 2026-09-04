import json
import math
from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent

from pyRadPlan.analysis import QICollection
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.optimization.objectives import Objective, get_objectives_union
from pyRadPlan.plan._plans import Plan
from pyRadPlan._settings import get_settings

from ._settings import load_ai_env
from ._usage import log_run_usage


OBJECTIVES_SYSTEM_PROMPT = """
        You are a radiotherapy treatment planning assistant.
        Given a treatment site and a list of Volumes of Interest (VOIs), suggest typical
        optimization objectives following standard clinical practice.

        The available objective types and the meaning of their parameters are fully
        described by the output schema. All dosimetric parameter values must be given as
        dose per fraction. If no prescribed dose is given, prescribe 2 Gy per fraction
        to the target. Assign a priority (weight) to every objective and leave the
        'quantity' field at its default.
        """

OBJECTIVES_ADAPT_PROMPT = """
        Additionally, quality indicators (QIs) computed per VOI from the dose
        distribution of a previous optimization run are provided. Assume that run used
        the objectives currently present in the structure set (also provided). Instead
        of proposing a generic template, adapt those objectives: adjust dose parameters
        and priorities, and add or remove objectives, to fix shortcomings the QIs
        reveal (insufficient target coverage or homogeneity, hot spots, avoidable OAR
        dose). Keep objectives that the QIs show to be working. Assume the QI dose
        values are given per fraction, on the same scale as the objectives, unless the
        user context states otherwise.
        """


def objectives_system_prompt(adapt: bool = False) -> str:
    """Return the system prompt for objective suggestion.

    Parameters
    ----------
    adapt : bool, optional
        Whether quality indicators from a previous optimization run are provided
        and the existing objectives should be adapted rather than created anew.
    """
    if adapt:
        return OBJECTIVES_SYSTEM_PROMPT + OBJECTIVES_ADAPT_PROMPT
    return OBJECTIVES_SYSTEM_PROMPT


def _objective_summary(obj) -> dict:
    """Compact JSON-friendly description of a single objective."""
    if not isinstance(obj, Objective):
        return {"definition": str(obj)}
    return {
        "name": obj.name,
        "priority": obj.priority,
        "quantity": obj.quantity,
        "parameters": dict(zip(obj.parameter_names, obj.parameters)),
    }


def _objectives_by_voi(cst: StructureSet) -> dict[str, list[dict]]:
    """Return the current objectives per VOI as compact JSON-friendly dicts."""
    return {
        voi.name: [_objective_summary(obj) for obj in voi.objectives or [] if obj is not None]
        for voi in cst.vois
    }


def _qi_summary(qis: QICollection) -> dict[str, dict[str, str]]:
    """Return QI values per structure as ``{structure: {metric: "value unit"}}``."""
    return {
        name: {
            metric: f"{qi.value:.4g} {qi.unit:~}"
            for metric, qi in structure.items()
            if math.isfinite(qi.value)
        }
        for name, structure in qis.structures.items()
    }


def cst_context_summary(pln: Plan, cst: StructureSet, qis: Optional[QICollection] = None) -> dict:
    """Return a JSON-serialisable summary of *cst*/*pln* for an LLM data context.

    Only metadata (VOI names, types and existing objective counts) is included;
    the voxel masks and any other numpy arrays held on the VOIs are deliberately
    left out so they are never serialised or sent to the model.

    When *qis* is given, the summary additionally carries the current objectives
    per VOI and the quality indicators, mirroring what
    :func:`generate_voi_objectives` sends for QI-based adaptation.
    """
    summary = {
        "radiation_mode": pln.radiation_mode,
        "prescribed_dose_gy": getattr(pln, "prescribed_dose", None),
        "num_of_fractions": pln.num_of_fractions,
        "vois": [
            {
                "name": voi.name,
                "type": voi.voi_type,
                "num_existing_objectives": len(voi.objectives or []),
            }
            for voi in cst.vois
        ],
    }
    if qis is not None:
        summary["current_objectives"] = _objectives_by_voi(cst)
        summary["quality_indicators"] = _qi_summary(qis)
    return summary


def _create_output_model(voi_names: tuple[str, ...]) -> type[BaseModel]:
    """
    Create the agent output model for objective suggestions.

    The model is built from the registered objective classes themselves (as a discriminated
    union), so the LLM sees the exact parameter schema, defaults, bounds and descriptions of
    pyRadPlan's objectives, and validated output directly yields Objective instances.

    Parameters
    ----------
    voi_names : tuple of str
        Names of the VOIs in the structure set. Restricts the suggestions to existing VOIs.

    Returns
    -------
    type[BaseModel]
        The output model with per-VOI objective suggestions.
    """
    objective_union = get_objectives_union(exclude_image_references=True)

    voi_objectives_model = create_model(
        "VoiObjectives",
        voi_name=(
            Literal[voi_names],
            Field(description="Name of the VOI the objectives apply to."),
        ),
        objectives=(
            list[objective_union],
            Field(description="Optimization objectives suggested for this VOI."),
        ),
    )

    return create_model(
        "OptimizationObjectives",
        objectives=(
            list[voi_objectives_model],
            Field(description="Suggested optimization objectives per VOI."),
        ),
    )


def generate_voi_objectives(  # noqa: PLR0913
    pln: Plan,
    cst: StructureSet,
    treatment_site: str,
    additional_context: Optional[str] = None,
    model: Optional[str] = None,
    clear_existing: bool = True,
    qis: Optional[QICollection] = None,
) -> StructureSet:
    """
    Generate VOI objectives for a given treatment site using an AI agent.

    Parameters
    ----------
    pln : Plan
        The plan object.
    cst : StructureSet
        The structure set object.
    treatment_site : str
        The treatment site (e.g., "prostate", "head and neck").
    additional_context : str, optional
        Additional clinical context or considerations to guide objective generation.
    model : str, optional
        The AI model to use (e.g., "gpt-4o", "gemini-1.5-pro", "claude-sonnet-4-5").
        If None, uses ``settings.ai.agents_model`` (``PYRADPLAN_AI_AGENTS_MODEL``).
    clear_existing : bool, optional
        Whether to clear existing objectives on the VOIs before assigning new ones.
    qis : QICollection, optional
        Per-organ quality indicators from a previous optimization run. When given,
        the agent assumes the QIs resulted from optimizing with the objectives
        currently present in *cst* and adapts those objectives instead of
        suggesting fresh ones. The QI dose values are assumed to be per fraction
        (the scale the objectives use); state deviations in *additional_context*.

    Returns
    -------
    StructureSet
        A new structure set with generated objectives.  The input *cst* is not
        modified, so a failing model call never destroys existing objectives.
    """

    adapt_block = ""
    if qis is not None:
        if not any(voi.name in qis for voi in cst.vois):
            raise ValueError(
                "None of the QI collection's structures match a VOI in the structure set."
            )
        adapt_block = f"""
        Objectives used in the previous optimization run (per VOI):
        {json.dumps(_objectives_by_voi(cst), indent=2, default=str)}

        Quality indicators per VOI computed from the resulting dose distribution:
        {json.dumps(_qi_summary(qis), indent=2, default=str)}
        """

    # Work on copied VOIs with fresh objectives lists (masks stay shared) so
    # the caller's structure set is untouched until the run has succeeded.
    vois = [
        voi.model_copy(update={"objectives": [] if clear_existing else list(voi.objectives or [])})
        for voi in cst.vois
    ]
    cst = cst.model_copy(update={"vois": vois})

    load_ai_env()
    effective_model = model or get_settings().ai.agents_model

    output_model = _create_output_model(tuple(voi.name for voi in cst.vois))

    system_prompt = objectives_system_prompt(adapt=qis is not None)

    voi_list_str = "\n".join(f"- {voi.name} (Type: {voi.voi_type})" for voi in cst.vois)

    task_instruction = (
        "Adapt the previous objectives based on the quality indicators."
        if qis is not None
        else "For each VOI, suggest one or more objectives if appropriate."
    )

    user_prompt = f"""
        Treatment site: {treatment_site}
        Prescribed dose (total): {pln.prescribed_dose} Gy
        Number of fractions: {pln.num_of_fractions}

        Available VOIs:
        {voi_list_str}
        {adapt_block}
        Additional context: {additional_context or "None given"}

        {task_instruction}
        """

    agent = Agent(effective_model, output_type=output_model, system_prompt=system_prompt)

    result = agent.run_sync(user_prompt=user_prompt)
    log_run_usage(result, effective_model, operation="generate_voi_objectives")

    vois_by_name = {voi.name: voi for voi in cst.vois}
    for suggestion in result.output.objectives:
        vois_by_name[suggestion.voi_name].objectives.extend(suggestion.objectives)

    return validate_cst(cst)

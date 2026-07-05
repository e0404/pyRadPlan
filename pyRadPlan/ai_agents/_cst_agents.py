from typing import Literal, Optional

from pydantic import BaseModel, Field, create_model
from pydantic_ai import Agent

from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.optimization.objectives import get_objectives_union
from pyRadPlan.plan._plans import Plan
from ._settings import AiSettings, load_ai_env
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


def cst_context_summary(pln: Plan, cst: StructureSet) -> dict:
    """Return a JSON-serialisable summary of *cst*/*pln* for an LLM data context.

    Only metadata (VOI names, types and existing objective counts) is included;
    the voxel masks and any other numpy arrays held on the VOIs are deliberately
    left out so they are never serialised or sent to the model.
    """
    return {
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


def generate_voi_objectives(
    pln: Plan,
    cst: StructureSet,
    treatment_site: str,
    additional_context: Optional[str] = None,
    model: Optional[str] = None,
    clear_existing: bool = True,
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
        If None, uses ``PYRADPLAN_AI_MODEL`` from the environment, falling back to
        the default defined in :class:`AiSettings`.
    clear_existing : bool, optional
        Whether to clear existing objectives on the VOIs before assigning new ones.

    Returns
    -------
    StructureSet
        A new structure set with generated objectives.  The input *cst* is not
        modified, so a failing model call never destroys existing objectives.
    """

    # Work on copied VOIs with fresh objectives lists (masks stay shared) so
    # the caller's structure set is untouched until the run has succeeded.
    vois = [
        voi.model_copy(update={"objectives": [] if clear_existing else list(voi.objectives or [])})
        for voi in cst.vois
    ]
    cst = cst.model_copy(update={"vois": vois})

    load_ai_env()
    effective_model = model or AiSettings().model

    output_model = _create_output_model(tuple(voi.name for voi in cst.vois))

    system_prompt = OBJECTIVES_SYSTEM_PROMPT

    voi_list_str = "\n".join(f"- {voi.name} (Type: {voi.voi_type})" for voi in cst.vois)

    user_prompt = f"""
        Treatment site: {treatment_site}
        Prescribed dose (total): {pln.prescribed_dose} Gy
        Number of fractions: {pln.num_of_fractions}

        Available VOIs:
        {voi_list_str}

        Additional context: {additional_context or "None given"}

        For each VOI, suggest one or more objectives if appropriate.
        """

    agent = Agent(effective_model, output_type=output_model, system_prompt=system_prompt)

    result = agent.run_sync(user_prompt=user_prompt)
    log_run_usage(result, effective_model, operation="generate_voi_objectives")

    vois_by_name = {voi.name: voi for voi in cst.vois}
    for suggestion in result.output.objectives:
        vois_by_name[suggestion.voi_name].objectives.extend(suggestion.objectives)

    return validate_cst(cst)

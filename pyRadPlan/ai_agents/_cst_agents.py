from typing import List, Literal, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from pyRadPlan.cst import StructureSet, validate_cst
from pyRadPlan.optimization import objectives as opt_obj
from ._settings import AiSettings


class ObjectiveParams(BaseModel):
    d_ref: Optional[float] = None
    d_max: Optional[float] = None
    d_min: Optional[float] = None
    d: Optional[float] = None
    v_min: Optional[float] = None
    v_max: Optional[float] = None
    eud_ref: Optional[float] = None
    k: Optional[float] = None
    priority: float


class VoiObjectiveSuggestion(BaseModel):
    voi_name: str
    objective_type: Literal[
        "SquaredDeviation",
        "SquaredOverdosing",
        "SquaredUnderdosing",
        "MeanDose",
        "EUD",
        "MinDVH",
        "MaxDVH",
        "DoseUniformity",
    ]
    parameters: ObjectiveParams


class OptimizationObjectives(BaseModel):
    objectives: List[VoiObjectiveSuggestion]


def generate_voi_objectives(
    cst: StructureSet,
    treatment_site: str,
    additional_context: Optional[str] = None,
    model: Optional[str] = None,
) -> StructureSet:
    """
    Generate VOI objectives for a given treatment site using an AI agent.

    Parameters
    ----------
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

    Returns
    -------
    StructureSet
        The updated structure set with generated objectives.
    """
    effective_model = model or AiSettings().model

    voi_list_str = "\n".join(f"- {voi.name} (Type: {voi.voi_type})" for voi in cst.vois)

    prompt = f"""
        You are a radiotherapy treatment planning assistant.
        Given a treatment site and a list of Volumes of Interest (VOIs),
        suggest typical optimization objectives.

        Treatment Site: {treatment_site}

        Available VOIs:
        {voi_list_str}

        For each VOI, suggest one or more objectives if appropriate.
        Use standard radiotherapy constraints.

        Only use the following objective types and their corresponding parameters:
        - SquaredDeviation: d_ref (reference dose)
        - SquaredOverdosing: d_max (maximum dose)
        - SquaredUnderdosing: d_min (minimum dose)
        - MeanDose: d_ref (reference mean dose)
        - EUD: eud_ref (reference EUD), k (exponent)
        - MinDVH: d (dose), v_min (min volume %)
        - MaxDVH: d (dose), v_max (max volume %)
        - DoseUniformity: (no parameters besides priority)

        All objectives must have a 'priority' (weight).
        """

    agent = Agent(effective_model, output_type=OptimizationObjectives, system_prompt=prompt)

    result = agent.run_sync(
        user_prompt=f"Treatment site: {treatment_site}, Additional context: {additional_context or 'None given'}"
    )

    suggestions = result.output.objectives

    # Map suggestions to cst
    for suggestion in suggestions:
        voi = next((v for v in cst.vois if v.name == suggestion.voi_name), None)
        if voi:
            obj_class = getattr(opt_obj, suggestion.objective_type, None)
            if obj_class:
                params = suggestion.parameters.model_dump(exclude_none=True)
                try:
                    objective_instance = obj_class(**params)
                    if voi.objectives is None:
                        voi.objectives = []
                    voi.objectives.append(objective_instance)
                except Exception as e:
                    print(
                        f"Failed to create objective {suggestion.objective_type} for {voi.name}: {e}"
                    )

    return validate_cst(cst)

from typing import List, Optional
from pydantic_ai import Agent
from pyRadPlan.plan import Plan, validate_pln


def generate_beam_angles(
    pln: Plan,
    treatment_site: str,
    additional_context: Optional[str] = "None given",
    model: Optional[str] = None,
) -> Plan:
    """
    Generate beam angles for a given treatment site using an AI agent.

    Parameters
    ----------
    pln : Plan
        The treatment plan object.
    treatment_site : str
        The treatment site (e.g., "prostate", "head and neck").
    additional_context : str, optional
        Additional clinical context or considerations to guide angle generation.
    model : str, optional
        The name of the AI model to use (e.g., "gpt-4", "gemini-1.5-pro").
        If None, uses the default model defined in ai_agents.MODEL_NAME.

    Returns
    -------
    Plan
        The updated treatment plan with generated beam angles.
    """
    from pyRadPlan import ai_agents

    # Validate backend configuration
    ai_agents.validate_ai_backend()

    model_name = model or ai_agents.MODEL_NAME

    prompt = f"""
        You are a radiotherapy treatment planning assistant.
        Given a treatment site, suggest typical gantry angles (in degrees)
        used for photon IMRT or proton IMPT beam setups.
        The radiation mode is {pln.radiation_mode}.
        You may respect the additional context provided by the user.
        In any case: Return only a Python list of floats.
        """

    agent = Agent(model_name, output_type=List[float], system_prompt=prompt)

    gantry_angles = agent.run_sync(
        user_prompt=f"Treatment site: {treatment_site}, Additional context: {additional_context}"
    ).output

    # Update plan
    if pln.prop_stf is None:
        pln.prop_stf = {}

    pln.prop_stf["gantry_angles"] = gantry_angles
    # Assume couch angles are 0 for now
    pln.prop_stf["couch_angles"] = [0.0] * len(gantry_angles)

    return validate_pln(pln)

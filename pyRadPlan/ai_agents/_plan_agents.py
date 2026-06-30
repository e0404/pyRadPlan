from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from pyRadPlan.plan import Plan, validate_pln
from ._settings import AiSettings
from ._usage import log_run_usage

Angle = Annotated[float, Field(ge=0.0, lt=360.0)]


class BeamSetup(BaseModel):
    """Beam setup as matching lists of gantry and couch angles."""

    gantry_angles: list[Angle] = Field(description="Gantry angles in degrees [0, 360).")
    couch_angles: list[Angle] = Field(
        default_factory=list,
        description=(
            "Couch angles in degrees [0, 360), one per gantry angle. "
            "Omit or use 0 for coplanar beams."
        ),
    )

    @model_validator(mode="after")
    def _match_couch_angles(self):
        if not self.couch_angles:
            self.couch_angles = [0.0] * len(self.gantry_angles)
        if len(self.couch_angles) != len(self.gantry_angles):
            raise ValueError("couch_angles must have the same length as gantry_angles")
        return self


def generate_beam_angles(
    pln: Plan,
    treatment_site: str,
    additional_context: Optional[str] = None,
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
        The AI model to use (e.g., "gpt-4o", "gemini-1.5-pro", "claude-sonnet-4-5").
        If None, uses ``PYRADPLAN_AI_MODEL`` from the environment, falling back to
        the default defined in :class:`AiSettings`.

    Returns
    -------
    Plan
        The updated treatment plan with generated beam angles.
    """
    effective_model = model or AiSettings().model

    prompt = f"""
        You are a radiotherapy treatment planning assistant.
        Given a treatment site, suggest a typical beam setup (gantry and couch angles)
        used for photon IMRT or proton IMPT.
        The radiation mode is {pln.radiation_mode}.
        You may respect the additional context provided by the user.
        """

    agent = Agent(effective_model, output_type=BeamSetup, system_prompt=prompt)

    result = agent.run_sync(
        user_prompt=f"Treatment site: {treatment_site}, Additional context: {additional_context or 'None given'}"
    )
    log_run_usage(result, effective_model, operation="generate_beam_angles")
    beam_setup = result.output

    if pln.prop_stf is None:
        pln.prop_stf = {}

    pln.prop_stf["gantry_angles"] = beam_setup.gantry_angles
    pln.prop_stf["couch_angles"] = beam_setup.couch_angles

    return validate_pln(pln)

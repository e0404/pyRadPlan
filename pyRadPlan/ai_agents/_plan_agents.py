import json
from typing import Annotated, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import Agent

from pyRadPlan.cst import StructureSet
from pyRadPlan.plan import Plan, validate_pln
from ._settings import AiSettings, load_ai_env
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


BEAM_ANGLES_GEOMETRY_PROMPT = """
        Per-VOI geometry of the structure set is provided: center of mass, principal
        axes (unit vectors ordered by descending extent) and shape parameters, in the
        LPS patient coordinate system (+x = patient left, +y = posterior,
        +z = superior; lengths in mm). Beam angles relate to these coordinates as
        follows: at gantry 0 / couch 0 the source sits anterior of the isocenter
        (towards -y) and the beam travels along +y; the gantry rotates
        counter-clockwise around the z axis, so gantry 90 irradiates from the
        patient's left (+x); the couch rotates the beam around the y axis for
        non-coplanar setups. Use the geometry to make an informed choice: prefer
        entry directions with short paths to the target that do not pass through or
        exit into nearby OARs, and distribute beams around the target considering
        its shape and orientation.
        """


def beam_angles_system_prompt(radiation_mode: str, with_geometry: bool = False) -> str:
    """Return the system prompt used to suggest a beam setup for *radiation_mode*.

    Parameters
    ----------
    radiation_mode : str
        The radiation mode of the plan (e.g. "photons", "protons").
    with_geometry : bool, optional
        Whether per-VOI geometry of a structure set accompanies the request; adds
        an explanation of the coordinate system and how to use the geometry.
    """
    prompt = f"""
        You are a radiotherapy treatment planning assistant.
        Given a treatment site, suggest a typical beam setup (gantry and couch angles)
        used for photon IMRT or proton IMPT.
        The radiation mode is {radiation_mode}.
        You may respect the additional context provided by the user.
        """
    if with_geometry:
        prompt += BEAM_ANGLES_GEOMETRY_PROMPT
    return prompt


def _rounded(values, ndigits: int) -> list[float]:
    return [round(float(v), ndigits) for v in values]


def cst_geometry_summary(cst: StructureSet) -> dict:
    """Return a JSON-serialisable per-VOI geometry summary for an LLM data context.

    Carries the center of mass, principal axes and shape parameters of every VOI
    (world LPS coordinates, grid units — typically mm) so a model can reason
    about beam directions. The voxel masks themselves are never included.
    Empty VOIs list only their name and type.
    """
    vois = []
    for voi in cst.vois:
        entry: dict = {"name": voi.name, "type": voi.voi_type}
        shape = voi.shape_parameters
        if shape is not None:
            entry["center_of_mass_mm"] = _rounded(voi.center_of_mass, 1)
            entry["principal_axes"] = [_rounded(axis, 3) for axis in voi.principal_axes]
            entry["shape"] = {
                "volume_cc": round(shape["volume"] / 1000.0, 2),
                "bounding_box_size_mm": _rounded(shape["bounding_box_size"], 1),
                "equivalent_ellipsoid_diameters_mm": _rounded(
                    shape["equivalent_ellipsoid_diameters"], 1
                ),
                "elongation": round(shape["elongation"], 2),
                "flatness": round(shape["flatness"], 2),
            }
        vois.append(entry)
    return {
        "coordinate_system": "LPS, units mm: +x = patient left, +y = posterior, +z = superior",
        "vois": vois,
    }


def generate_beam_angles(
    pln: Plan,
    treatment_site: str,
    additional_context: Optional[str] = None,
    model: Optional[str] = None,
    cst: Optional[StructureSet] = None,
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
    cst : StructureSet, optional
        Structure set whose per-VOI geometry (centers of mass, principal axes,
        shape parameters) is summarized and sent along so the model can pick
        beam directions informed by the patient anatomy.

    Returns
    -------
    Plan
        The updated treatment plan with generated beam angles.
    """
    load_ai_env()
    effective_model = model or AiSettings().model

    prompt = beam_angles_system_prompt(pln.radiation_mode, with_geometry=cst is not None)

    agent = Agent(effective_model, output_type=BeamSetup, system_prompt=prompt)

    user_prompt = (
        f"Treatment site: {treatment_site}, "
        f"Additional context: {additional_context or 'None given'}"
    )
    if cst is not None:
        user_prompt += "\n\nStructure set geometry:\n" + json.dumps(
            cst_geometry_summary(cst), indent=2, default=str
        )

    result = agent.run_sync(user_prompt=user_prompt)
    log_run_usage(result, effective_model, operation="generate_beam_angles")
    beam_setup = result.output

    if pln.prop_stf is None:
        pln.prop_stf = {}

    pln.prop_stf["gantry_angles"] = beam_setup.gantry_angles
    pln.prop_stf["couch_angles"] = beam_setup.couch_angles

    return validate_pln(pln)

"""Module for pydantic-ai agent models."""

import os

# pydantic-ai pulls in fastmcp -> key_value.aio, which installs a beartype import hook.
# beartype's isinstance() probes on abc.ABC blow up once numpydantic's vendored nptyping
# is imported, flooding stderr with BeartypeClawDecorWarning. Opt out before the import.
os.environ.setdefault("PY_KEY_VALUE_DISABLE_BEARTYPE", "1")

from ._settings import AiSettings, load_ai_env
from ._models import available_models
from ._usage import pop_last_run_usage
from ._plan_agents import generate_beam_angles, beam_angles_system_prompt, cst_geometry_summary
from ._cst_agents import (
    generate_voi_objectives,
    cst_context_summary,
    objectives_system_prompt,
    OBJECTIVES_SYSTEM_PROMPT,
    OBJECTIVES_ADAPT_PROMPT,
)

__all__ = [
    "AiSettings",
    "load_ai_env",
    "available_models",
    "pop_last_run_usage",
    "generate_beam_angles",
    "beam_angles_system_prompt",
    "cst_geometry_summary",
    "generate_voi_objectives",
    "cst_context_summary",
    "objectives_system_prompt",
    "OBJECTIVES_SYSTEM_PROMPT",
    "OBJECTIVES_ADAPT_PROMPT",
]

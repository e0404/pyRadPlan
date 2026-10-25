"""Module for pydantic-ai agent models.

The agents, and with them pydantic-ai, are imported on first access, so the
settings, :func:`available_models` and :func:`pop_last_run_usage` stay cheap to
import, e.g. for the GUI deciding whether to offer its AI features.
"""

import os
from typing import TYPE_CHECKING

# pydantic-ai pulls in fastmcp -> key_value.aio, which installs a beartype import hook.
# beartype's isinstance() probes on abc.ABC blow up once numpydantic's vendored nptyping
# is imported, flooding stderr with BeartypeClawDecorWarning. Opt out before the import.
os.environ.setdefault("PY_KEY_VALUE_DISABLE_BEARTYPE", "1")

from ..._lazy import attach_lazy
from ._settings import AiSettings as AiSettings, load_ai_env as load_ai_env
from ._models import available_models as available_models
from ._usage import pop_last_run_usage as pop_last_run_usage

if TYPE_CHECKING:
    # Type checkers resolve the lazy names from here; at runtime __getattr__ does.
    from ._cst_agents import (
        OBJECTIVES_ADAPT_PROMPT as OBJECTIVES_ADAPT_PROMPT,
        OBJECTIVES_SYSTEM_PROMPT as OBJECTIVES_SYSTEM_PROMPT,
        cst_context_summary as cst_context_summary,
        generate_voi_objectives as generate_voi_objectives,
        objectives_system_prompt as objectives_system_prompt,
    )
    from ._plan_agents import (
        beam_angles_system_prompt as beam_angles_system_prompt,
        cst_geometry_summary as cst_geometry_summary,
        generate_beam_angles as generate_beam_angles,
    )

__getattr__, __dir__, __all__ = attach_lazy(
    __name__,
    {
        "._cst_agents": (
            "OBJECTIVES_ADAPT_PROMPT",
            "OBJECTIVES_SYSTEM_PROMPT",
            "cst_context_summary",
            "generate_voi_objectives",
            "objectives_system_prompt",
        ),
        "._plan_agents": (
            "beam_angles_system_prompt",
            "cst_geometry_summary",
            "generate_beam_angles",
        ),
    },
    eager=("AiSettings", "load_ai_env", "available_models", "pop_last_run_usage"),
)

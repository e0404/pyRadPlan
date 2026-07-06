"""GUI helpers for running pydantic-ai tasks."""

import importlib.util

from ._ai_task_dialog import AiTask, AiTaskDialog

#: Tooltip shown on AI features when the optional dependencies are missing.
AI_MISSING_TIP = "Install the optional 'ai' dependencies (pydantic-ai) to use this"


def ai_available() -> bool:
    """Whether the optional AI dependencies (pydantic-ai) are installed."""
    return importlib.util.find_spec("pydantic_ai") is not None


__all__ = ["AI_MISSING_TIP", "AiTask", "AiTaskDialog", "ai_available"]

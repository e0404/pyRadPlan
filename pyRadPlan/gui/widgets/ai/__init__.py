"""GUI helpers for running pydantic-ai tasks."""

import importlib.util
from contextlib import contextmanager
from typing import Iterator, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from ._ai_task_dialog import AiTask, AiTaskDialog

#: Tooltip shown on AI features when the optional dependencies are missing.
AI_MISSING_TIP = "Install the optional 'ai' dependencies (pydantic-ai) to use this"

#: Tooltip shown on AI features when no provider API key is configured.
AI_NO_MODEL_TIP = (
    "No AI model available: set a provider API key (e.g. ANTHROPIC_API_KEY,"
    " OPENAI_API_KEY) in the environment or a .env file"
)


def ai_available() -> bool:
    """Whether the optional AI dependencies (pydantic-ai) are installed."""
    return importlib.util.find_spec("pydantic_ai") is not None


def ai_disabled_reason() -> Optional[str]:
    """Why AI features cannot run right now, or ``None`` if they can.

    Checks that pydantic-ai is installed and that at least one model is usable
    (see :func:`pyRadPlan.ai.agents.available_models`); the returned string is
    suitable as a tooltip on disabled AI buttons.
    """
    if not ai_available():
        return AI_MISSING_TIP
    from pyRadPlan.ai.agents import available_models  # noqa: PLC0415

    if not available_models():
        return AI_NO_MODEL_TIP
    return None


@contextmanager
def busy_cursor() -> Iterator[None]:
    """Show a wait cursor while the block runs.

    Used around the first import of the agents, which loads pydantic-ai and can
    take a few seconds.
    """
    QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
    try:
        yield
    finally:
        QApplication.restoreOverrideCursor()


__all__ = [
    "AI_MISSING_TIP",
    "AI_NO_MODEL_TIP",
    "AiTask",
    "AiTaskDialog",
    "ai_available",
    "ai_disabled_reason",
    "busy_cursor",
]

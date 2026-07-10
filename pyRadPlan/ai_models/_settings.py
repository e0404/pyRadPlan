"""Pydantic settings for the ai_models module.

:class:`AiModelSettings` is defined in :mod:`pyRadPlan._settings` as part of the
global pyRadPlan configuration and re-exported here. It is exposed on the global
settings singleton as ``pyRadPlan.settings.ai_models``.
"""

from pyRadPlan._settings import AiModelSettings, get_settings

__all__ = ["AiModelSettings", "get_ai_model_settings"]


def get_ai_model_settings() -> AiModelSettings:
    """Return the AI-model sub-configuration of the global settings singleton.

    Returns
    -------
    AiModelSettings
        ``pyRadPlan.settings.ai_models`` — the process-wide AI model settings.
    """
    return get_settings().ai_models

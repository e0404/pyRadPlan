"""Pydantic settings for the ai_models module.

:class:`AiModelSettings` is defined in :mod:`pyRadPlan._settings` as part of the
global pyRadPlan configuration and re-exported here. Access the process-wide
instance via ``pyRadPlan.settings.ai_models`` (or
``pyRadPlan.get_settings().ai_models``).
"""

from pyRadPlan._settings import AiModelSettings

__all__ = ["AiModelSettings"]

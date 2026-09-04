"""Pydantic settings for the ai.modelhub module.

The AI configuration is unified in :class:`AiSettings`, defined in
:mod:`pyRadPlan._settings` as part of the global pyRadPlan configuration and
re-exported here. Access the process-wide instance via
``pyRadPlan.settings.ai`` (or ``pyRadPlan.get_settings().ai``).
"""

from pyRadPlan._settings import AiSettings

__all__ = ["AiSettings"]

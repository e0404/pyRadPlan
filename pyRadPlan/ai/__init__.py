"""AI functionality for pyRadPlan.

Two independent subpackages, each with its own optional dependency stack:

``pyRadPlan.ai.agents``
    LLM-powered treatment planning helpers built on *pydantic-ai*.

``pyRadPlan.ai.modelhub``
    Loading of trained models (and their preprocessors) from the HuggingFace
    Hub or a local directory. Requires a ``torch`` build matching your platform.

Nothing is re-exported here on purpose: importing :mod:`pyRadPlan.ai` must not
drag in either dependency stack. Import the subpackage you need explicitly.
"""

__all__: list[str] = []

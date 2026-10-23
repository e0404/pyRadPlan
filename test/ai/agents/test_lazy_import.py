"""The agents package loads pydantic-ai only when an agent is used."""

import subprocess
import sys

import pytest

pytest.importorskip("pydantic_ai")

_CHECK = """
import sys
import pyRadPlan.ai.agents as agents

agents.available_models()
agents.pop_last_run_usage()
assert "pydantic_ai" not in sys.modules, "helpers loaded pydantic_ai"
assert agents.generate_voi_objectives.__module__ == "pyRadPlan.ai.agents._cst_agents"
assert "pydantic_ai" in sys.modules
"""


def test_agents_import_pydantic_ai_on_first_use():
    # A fresh interpreter: other tests may already have imported pydantic_ai.
    subprocess.run([sys.executable, "-c", _CHECK], check=True)


def test_unknown_attribute_raises():
    import pyRadPlan.ai.agents as agents

    with pytest.raises(AttributeError, match="no_such_agent"):
        agents.no_such_agent  # noqa: B018

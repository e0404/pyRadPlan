# %% [markdown]
"""
# AI Agents for Radiotherapy Planning.

This example demonstrates the usage of pydantic-ai based `ai_agents` in `pyRadPlan`
to assist in treatment planning. The module uses a user-defined LLM to suggest
beam angles and optimization objectives based on the treatment site.

## Prerequisites

1. `pydantic-ai` and `pydantic-settings` installed.
2. An API key for your chosen AI provider set as an environment variable:
   - OpenAI:     `OPENAI_API_KEY`
   - Anthropic:  `ANTHROPIC_API_KEY`
   - Google:     `GOOGLE_API_KEY`
   - Mistral:    `MISTRAL_API_KEY`
   If a .env file is present, the environment will be populated with python-dotenv
3. Optionally, set the model via `PYRADPLAN_AI_MODEL` (default: `claude-sonnet-4-5`).

## Setup
"""

# %% tags=["active-ipynb"]
import logging
from dotenv import load_dotenv
from pyRadPlan import (
    IonPlan,
    load_tg119,
    ai_agents,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    DVHCollection,
)
from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
## These two lines are needed to allow nested event loops in Jupyter.
## Otherwise you will experience sync-errors when running the AI agents.
## Install it using `pip install nest_asyncio`.
# import nest_asyncio
# nest_asyncio.apply()

# %%
# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)

# %% [markdown]
"""
## Configuration

Set the model via the environment variable `PYRADPLAN_AI_MODEL`, or pass `model=`
explicitly to each call. Your provider API key must be set in the environment
(e.g. `OPENAI_API_KEY`). pydantic-ai picks it up automatically.
"""

# %%
# Set the model to use (can also be set via PYRADPLAN_AI_MODEL env var)
# os.environ["PYRADPLAN_AI_MODEL"] = "openai:gpt-5-mini"

# Inspect effective settings
settings = ai_agents.AiSettings()
print(f"Using model: {settings.model}")

# %% [markdown]
"""
## Load Data

We will use the TG119 phantom provided with `pyRadPlan`.
"""

# %%
ct, cst = load_tg119()
print("Loaded TG119 phantom.")
print("VOIs:", [v.name for v in cst.vois])

# %% [markdown]
"""
## Generate Beam Angles

We create a generic proton plan and ask the AI agent to suggest beam angles
for a prostate case.
"""

# %%
pln = IonPlan(radiation_mode="protons", machine="Generic", num_of_fractions=30, prescribed_dose=60)
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {"dose_grid": ct.grid}

print("Generating beam angles for prostate...")
pln = ai_agents.generate_beam_angles(pln, treatment_site="prostate")

print(f"Suggested Gantry Angles: {pln.prop_stf['gantry_angles']}")

# %% [markdown]
"""
## Generate Optimization Objectives

Ask the AI agent to suggest optimization objectives for the VOIs.
"""

# %%
print("Generating VOI objectives...")
cst = ai_agents.generate_voi_objectives(pln, cst, treatment_site="prostate")

for voi in cst.vois:
    if voi.objectives:
        print(f"\nVOI: {voi.name}")
        for obj in voi.objectives:
            print(obj)

# %% [markdown]
"""
## Run plan with AI-generated settings.

"""

# %%
stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)
fluence = fluence_optimization(ct, cst, stf, dij, pln)

result = dij.compute_result_ct_grid(fluence)
dvhs = DVHCollection.from_structure_set(cst, result["physical_dose"])

# %%
# TODO: agentic adaptation loop with information from DVH Collection

# %% [markdown]
"""
## Visualize Result
"""

# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    plot_slice(image_volume=ct, cst=cst, overlay=result["physical_dose"])

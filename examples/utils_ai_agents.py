# %% [markdown]
"""
# AI Agents for Radiotherapy Planning

This example demonstrates the usage of pydantics `ai_agents` in `pyRadPlan` to assist in treatment planning.
The module uses a user defined LLM to suggest beam angles and optimization objectives based on the treatment site and patient geometry.

## Prerequisites

To use this module, you need:
1.  `pydantic-ai` installed (`pip install pydantic-ai`).
2.  An API key for your chosen AI provider (e.g., OpenAI, Google Gemini, Anthropic).
3.  The API key set as an environment variable (e.g., `GOOGLE_API_KEY`, `OPENAI_API_KEY`) OR set via `ai_agents.API_KEY`.

## Setup
"""

# %%
import logging
from pyRadPlan import (
    IonPlan,
    load_tg119,
    ai_agents,
)

# Configure logging
logging.basicConfig(level=logging.INFO)

# %% [markdown]
"""
## Configuration

You must configure the `ai_agents` module with a model name.
The provider is inferred from the model name (e.g., "gemini-..." -> google, "gpt-..." -> openai).
You can also explicitly set the provider if needed.

Ensure your API key is available. You can set it in the environment variables or directly in the code (not recommended for shared scripts).
"""

# %%
# Example configuration for Google Gemini
# os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"

# You can also set the API key directly if not in environment variables:
# ai_agents.API_KEY = "YOUR_API_KEY_HERE"

# Set the default model to use
ai_agents.MODEL_NAME = "gemini-2.5-pro"
# ai_agents.PROVIDER = "google" # Optional: inferred from model name

print(f"Using model: {ai_agents.MODEL_NAME}")

# %% [markdown]
"""
## Load Data

We will use the TG119.mat provided with `pyRadPlan`.
"""

# %%
ct, cst = load_tg119()
print("Loaded TG119 phantom.")
print("VOIs:", [v.name for v in cst.vois])

# %% [markdown]
"""
## Generate Beam Angles

We create a generic proton plan and ask the AI agent to suggest beam angles for a prostate case.
"""

# %%
# Create a base plan
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {"dose_grid": ct.grid}

# Generate beam angles
print("Generating beam angles for prostate...")
pln = ai_agents.generate_beam_angles(pln, treatment_site="prostate")

print(f"Suggested Gantry Angles: {pln.prop_stf['gantry_angles']}")

# %% [markdown]
"""
## Generate Optimization Objectives

Now we ask the AI agent to suggest optimization objectives for our VOIs.
The agent analyzes the VOI names and types and suggests standard constraints.
"""

# %%
print("Generating VOI objectives...")
cst = ai_agents.generate_voi_objectives(cst, treatment_site="prostate")

# Display the generated objectives
for voi in cst.vois:
    if voi.objectives:
        print(f"\nVOI: {voi.name}")
        for obj in voi.objectives:
            print(f"  - {obj}")

# %% [markdown]
"""
## Advanced Usage: Additional Context & Model Switching

You can provide `additional_context` to guide the agent (e.g., specific clinical scenarios, sparing requirements).
You can also override the default model for a specific call.
"""

# %%
print("\nRegenerating objectives with additional context (Hip Prosthesis)...")

# We can pass a different model if needed, e.g., a more capable one for complex reasoning
cst = ai_agents.generate_voi_objectives(
    cst,
    treatment_site="prostate",
    additional_context="The patient has a hip prosthesis on the left side. Avoid beams entering through the prosthesis if possible.",
    model="gemini-1.5-pro",
)

for voi in cst.vois:
    if voi.objectives:
        print(f"\nVOI: {voi.name}")
        for obj in voi.objectives:
            print(f"  - {obj}")

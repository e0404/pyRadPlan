"""# Tool-using treatment planning agent (needs an LLM API key).

The agent decides itself which pyRadPlan tool to call next: inspect/map structures,
look up constraint tables (EQD2-converted), analyze target shape and feasibility,
create helper structures, set beams, check beam paths, set objectives, optimize,
localize hot/cold spots, analyze QIs, save/compare/restore plan versions. It stops when
the protocol is met or its budget is used up.

A head-and-neck case: `utils_ai_planning_agent_hn.py`.
"""

# %%
import logging

from pyRadPlan import PhotonPlan, load_tg119, plot_slice
from pyRadPlan.ai.agents import Goal, PlanningState, Protocol, run_planning_agent

logging.basicConfig(level=logging.WARNING)

MODEL = None  # settings.ai.agents_model (PYRADPLAN_AI_AGENTS_MODEL), or e.g. "anthropic:..."

# %% [markdown]
# ## A) Human-defined protocol (locked)
# Goals use the plan's dose convention (per fraction by default). Protocol names do not
# need to match the VOI names; the agent maps them.

# %%
ct, cst = load_tg119()
protocol = Protocol(
    name="TG119 C-shape (per fraction)",
    goals=[
        Goal(voi="PTV", metric="D95", op=">=", value=1.9),
        Goal(voi="PTV", metric="D5", op="<=", value=2.2),
        Goal(voi="Core", metric="max", op="<=", value=1.0),
        Goal(voi="Core", metric="D5", op="<=", value=0.8, hard=False),
    ],
)
state = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=protocol)
result = run_planning_agent(
    state, "Plan this C-shaped target with photon IMRT, max 9 coplanar beams.", model=MODEL
)

for step in state.history:  # what the agent did, step by step
    print(step)
print(result.output.model_dump_json(indent=2))
print(state.last_evaluation)
plot_slice(image_volume=ct, cst=state.cst, overlay=state.dose, overlay_unit="Gy")

# %% [markdown]
# ## B) Agent picks goals from a constraint table (unlocked, empty protocol)
# Template tables contain PLACEHOLDER values - not for clinical use.

# %%
ct, cst = load_tg119()
pln = PhotonPlan(num_of_fractions=20, prescribed_dose=60.0)
state = PlanningState(ct=ct, cst=cst, pln=pln, protocol=Protocol(locked=False))
result = run_planning_agent(
    state,
    "Prostate-like phantom, 60 Gy in 20 fractions. Use the RTOG template's goals; "
    "treat 'Core' as the rectum.",
    model=MODEL,
)
print(state.protocol.name, [g.label for g in state.protocol.goals])
print(result.output.model_dump_json(indent=2))

# %% Inspect the full conversation (tool calls and results)
# for m in result.all_messages(): print(m)

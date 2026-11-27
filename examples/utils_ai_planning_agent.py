# %% [markdown]
"""# Tool-Using Treatment Planning Agent on the TG119 Phantom."""
# %% [markdown]
# This example runs the planning agent of `pyRadPlan.ai.agents` on the TG119 phantom. The
# agent is a language model (LLM) that works through a plan with pyRadPlan's tools: it maps
# structure names, sets beams and objectives, optimizes, and checks the dose against
# clinical goals. After each tool call it sees a short result and decides what to do next,
# until the goals are met or its budget is used up.
#
# **Scope and limitations.** The planning agent is a research feature and **not for clinical
# use**. Its results depend on the model, and two runs of the same case can differ.
#
# A head-and-neck case with a review of the resulting plans is shown in
# `utils_ai_planning_agent_hn.py`.

# %% [markdown]
# ## Prerequisites
#
# 1. A language model that supports tool calls. Use either
#    - a cloud provider with its API key in the environment or a `.env` file
#      (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, ...), or
#    - a local [Ollama](https://ollama.com) server: set `OLLAMA_BASE_URL` (e.g.
#      `http://localhost:11434/v1`) and use a model such as `"ollama:gpt-oss:20b"`.
# 2. Optionally, select the model with `PYRADPLAN_AI_AGENTS_MODEL` or with `MODEL` below.
#
# **What is sent to the model provider.** Dose calculation and optimization run on your
# computer. The model receives text only: the task, the goals, the structure names, and
# summaries in numbers (structure volumes and positions, beam angles, objectives, dose
# metrics). The CT, the contours and the dose distributions are not sent, and neither is a
# patient name or ID. With a local Ollama server nothing is sent to a cloud provider.

# %%
import logging

from pyRadPlan import PhotonPlan, load_tg119, plot_slice
from pyRadPlan.ai.agents import (
    Goal,
    PlanningRunError,
    PlanningState,
    Protocol,
    load_ai_env,
    run_planning_agent,
)

logging.basicConfig(level=logging.WARNING)

load_ai_env()  # reads API keys and settings from a .env file, if there is one
MODEL = None  # None: settings.ai.agents_model; or e.g. "anthropic:..." or "ollama:..."

# %% [markdown]
# ## A) Planning for given goals
#
# A *goal* is one clinical requirement on the dose of a structure, e.g. "D95 of the PTV of at
# least 1.9 Gy". Hard goals must be met, soft goals (`hard=False`) should be met as well as
# possible. A *protocol* is the list of goals of a run. A protocol you define is locked: the
# agent plans for it and cannot change it.
#
# Doses use the plan's dose convention, which is per fraction by default. The structure
# names of the goals do not have to match the names in the patient; the agent maps them.

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

# %% [markdown]
# The `PlanningState` holds everything the agent works on: patient, plan, protocol, and later
# the beams, objectives, doses and all plans. The task is plain text, as you would brief a
# planner.
#
# The run is limited by `max_optimizations` (computation on your machine) and by
# `request_limit` (requests to the model, i.e. cost). A run that reaches the request limit
# or fails at the provider raises a `PlanningRunError`; the state keeps every plan made
# until then, and `continue_planning_agent(state, err)` would resume the run.

# %%
state = PlanningState(ct=ct, cst=cst, pln=PhotonPlan(), protocol=protocol)
try:
    result = run_planning_agent(
        state,
        "Plan this C-shaped target with photon IMRT, max 9 coplanar beams.",
        model=MODEL,
        max_optimizations=8,
        request_limit=40,
    )
except PlanningRunError as err:
    result = None
    print(f"Agent stopped: {err}")

# %% [markdown]
# ### What the agent did
# `state.history` lists the steps of the agent. The report is written by the model; the goal
# evaluation in `state.last_evaluation` is computed by pyRadPlan and is what the plan should
# be judged by. A margin of zero or more means the goal is met.

# %%
for step in state.history:
    print(step)
if result is not None:
    print(result.output.model_dump_json(indent=2))

if state.last_evaluation is not None:
    for goal in state.last_evaluation["goals"]:
        print(goal)
    plot_slice(image_volume=ct, cst=state.cst, overlay=state.dose, overlay_unit="Gy")

# %% [markdown]
# ## B) The agent picks the goals from a constraint table
#
# A *constraint table* holds goals per treatment site. With an empty, unlocked protocol the
# agent looks up the table, converts its doses to the fractionation of the plan (via EQD2
# with the alpha/beta of each structure) and applies the goals.
#
# The values of the bundled `demo` table are QUANTEC/RTOG-style: simplified, for
# demonstration only, not validated and not for clinical use.

# %%
ct, cst = load_tg119()
pln = PhotonPlan(num_of_fractions=20, prescribed_dose=60.0)
state = PlanningState(ct=ct, cst=cst, pln=pln, protocol=Protocol(locked=False))
try:
    result = run_planning_agent(
        state,
        "Prostate-like phantom, 60 Gy in 20 fractions. Use the demo table's prostate goals; "
        "treat 'Core' as the rectum.",
        model=MODEL,
    )
except PlanningRunError as err:
    result = None
    print(f"Agent stopped: {err}")

print(state.protocol.name, [g.label for g in state.protocol.goals])
if result is not None:
    print(result.output.model_dump_json(indent=2))

# %% [markdown]
# The full conversation with all tool calls and results is available from the result:

# %%
# for message in result.all_messages():
#     print(message)

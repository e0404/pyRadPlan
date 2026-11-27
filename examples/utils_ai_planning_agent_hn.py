# %% [markdown]
"""# Planning Agent on the matRad Head-and-Neck Phantom."""
# %% [markdown]
# This example lets the planning agent of `pyRadPlan.ai.agents` plan a simultaneous
# integrated boost (70 Gy to PTV70 and 63 Gy to PTV63 in 35 fractions) and then compares the
# plans it produced. The goals use clinical names (`SpinalCord`, `Parotid_L`, ...); the agent
# maps them to the phantom's matRad names (`SPINAL_CORD`, `PAROTID_LT`, ...).
#
# Afterwards the GUI shows the two best plans that trade off differently, next to a plot of
# how every goal developed over the optimizations.
#
# **Scope and limitations.** The planning agent is a research feature and **not for clinical
# use**. Its results depend on the model, and two runs of the same case can differ.
#
# The basics of the agent are introduced in `utils_ai_planning_agent.py`.

# %% [markdown]
# ## Prerequisites
#
# 1. The head-and-neck phantom of matRad (`HEAD_AND_NECK.mat` in `matRad/phantoms`). Set the
#    environment variable `MATRAD_PHANTOMS` to that folder.
# 2. A language model that supports tool calls. The example uses an Anthropic model, which
#    needs `ANTHROPIC_API_KEY` in the environment or a `.env` file. Other providers work as
#    well, and so does a local [Ollama](https://ollama.com) server (set `OLLAMA_BASE_URL`,
#    e.g. `http://localhost:11434/v1`, and `MODEL = "ollama:<model name>"`).
#
# **What is sent to the model provider.** Dose calculation and optimization run on your
# computer. The model receives text only: the task, the goals, the structure names, and
# summaries in numbers (structure volumes and positions, beam angles, objectives, dose
# metrics). The CT, the contours and the dose distributions are not sent, and neither is a
# patient name or ID. With a local Ollama server nothing is sent to a cloud provider.

# %%
import logging
import os
from pathlib import Path

from pyRadPlan import PhotonPlan, load_patient
from pyRadPlan.ai.agents import (
    Goal,
    PlanningRunError,
    PlanningState,
    Protocol,
    load_ai_env,
    run_planning_agent,
    select_distinct_plans,
)
from pyRadPlan.gui import GUI_AVAILABLE, launch_viewer

logging.basicConfig(level=logging.WARNING)

HN_PATH = Path(os.environ.get("MATRAD_PHANTOMS", "matRad/phantoms")) / "HEAD_AND_NECK.mat"

load_ai_env()  # reads API keys and settings from a .env file, if there is one
MODEL = "anthropic:claude-sonnet-5-5"

# %% [markdown]
# ## Patient, plan and protocol
# The phantom ships with matRad objectives, which are dropped so the agent starts from
# scratch.
#
# All doses in this example are total doses of the course: the plan uses
# `dose_convention="total"`, and the goals are written accordingly. A goal is hard (it must
# be met) unless it is created with `hard=False`.

# %%
ct, cst = load_patient(HN_PATH)
cst = cst.model_copy(update={"vois": [v.model_copy(update={"objectives": []}) for v in cst.vois]})
pln = PhotonPlan(num_of_fractions=35, prescribed_dose=70.0, dose_convention="total")

protocol = Protocol(
    name="HN SIB 70/63 Gy in 35 fx",
    goals=[
        Goal(voi="PTV70", metric="D95", op=">=", value=66.5),
        Goal(voi="PTV70", metric="D2", op="<=", value=74.9, hard=False),
        Goal(voi="PTV63", metric="D95", op=">=", value=59.85),
        Goal(voi="SpinalCord", metric="max", op="<=", value=45.0),
        Goal(voi="SpinalCord_PRV", metric="max", op="<=", value=50.0, hard=False),
        Goal(voi="Brainstem", metric="max", op="<=", value=54.0),
        Goal(voi="Brainstem_PRV", metric="max", op="<=", value=60.0, hard=False),
        Goal(voi="OpticChiasm", metric="max", op="<=", value=54.0),
        Goal(voi="Parotid_L", metric="mean", op="<=", value=26.0, hard=False),
        Goal(voi="Parotid_R", metric="mean", op="<=", value=26.0, hard=False),
        Goal(voi="Larynx", metric="mean", op="<=", value=44.0, hard=False),
    ],
)
state = PlanningState(ct=ct, cst=cst, pln=pln, protocol=protocol)

# %% [markdown]
# ## Run
# Dose influence calculations (at most `state.max_dij_calcs`, default 3) and optimizations
# (at most `max_optimizations`) run locally. `request_limit` caps the number of model
# requests and with it the cost.
#
# Every optimized plan is kept in `state.plans`, also when the run stops early: a run that
# reaches the request limit or fails at the provider raises a `PlanningRunError`, and
# `continue_planning_agent(state, err)` would resume it.

# %%
try:
    result = run_planning_agent(
        state,
        "Head-and-neck SIB: 70 Gy to PTV70 and 63 Gy to PTV63 in 35 fractions. "
        "Photon IMRT with at most 7 coplanar beams. Doses are total course doses. "
        "Explore two different trade-offs, e.g. best target coverage and best parotid "
        "sparing, and save each with plan_versions.",
        model=MODEL,
        max_optimizations=8,
        request_limit=40,
    )
    usage = result.usage
except PlanningRunError as err:
    result, usage = None, err.usage
    print(f"Agent stopped: {err}")

print(
    f"requests: {usage.requests}, tool calls: {usage.tool_calls}, input: {usage.input_tokens} "
    f"(cache read {usage.cache_read_tokens}, cache write {usage.cache_write_tokens}), "
    f"output: {usage.output_tokens}"
)
if result is not None:
    print(result.output.model_dump_json(indent=2))
for step in state.history:  # what the agent did, step by step
    print(step)

# %% [markdown]
# ## The two best plans that trade off differently
# Each plan is kept as a plan record with its dose, beams and goal evaluation.
# `select_distinct_plans` ranks the plans by (all hard goals passed, score). The score is
# the weighted mean of each goal's margin relative to its limit, clipped to
# [-100 %, +20 %]: higher is better, and it is zero or more when every goal passes. Plan 2
# must beat plan 1 by at least 5 % of a goal's limit on at least one goal, so it is not just
# a slightly worse copy of plan 1.

# %%
picks = select_distinct_plans(state.plans, n=2)
for number, p in enumerate(picks, start=1):
    ev = p.evaluation
    print(
        f"Plan {number}: plan {p.index}, score {ev['score']}, "
        f"hard goals passed: {ev['all_hard_passed']}, beams {p.beams['gantry_angles']}"
    )
if len(picks) == 2:
    a, b = (p.rel_margins() for p in picks)
    for goal in a:
        if abs(b.get(goal, a[goal]) - a[goal]) >= 0.05:
            print(f"  {goal}: plan 1 {100 * a[goal]:+.1f} %, plan 2 {100 * b[goal]:+.1f} %")
elif picks:
    print("No plan is better than plan 1 on any goal by >= 5 %; showing plan 1 only.")

# %% [markdown]
# ## GUI: progress plot and result viewer
# The plot shows every goal's margin per optimization (above the 0 line the goal is met) and
# the score. The viewer's quantity list holds one dose per selected plan. The main GUI shows
# the same in its agentic UI mode (File > UI Mode, or
# `pyRadPlan.settings.gui.ui_mode = "agentic"`).

# %%
if not picks:
    print("No optimized plan to show.")
elif GUI_AVAILABLE:
    from PySide6.QtWidgets import QApplication

    from pyRadPlan.gui.widgets.agent import GoalProgressPlot

    app = QApplication.instance() or QApplication([])
    plot = GoalProgressPlot()  # keep a reference while the GUI runs
    plot.set_plans(state.plans, picks)
    plot.resize(1000, 600)
    plot.show()
    doses = {
        f"{p.quantity} Plan {number} (plan {p.index}, score {p.evaluation['score']:g})": p.dose
        for number, p in enumerate(picks, start=1)
    }
    launch_viewer(ct, state.cst, doses)  # runs the event loop for both windows

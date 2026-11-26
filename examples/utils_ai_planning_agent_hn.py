"""# Planning agent on the matRad head-and-neck phantom (needs an Anthropic API key).

A simultaneous integrated boost (70 Gy to PTV70, 63 Gy to PTV63 in 35 fractions) with
a locked protocol. The goals use clinical names (``SpinalCord``, ``Parotid_L``, ...);
the agent maps them to the phantom's matRad names (``SPINAL_CORD``, ``PAROTID_LT``, ...).

Afterwards the GUI shows the two best plans that trade off differently, next to a plot
of how every goal developed over the optimizations.
"""

# %%
import logging
import os
from pathlib import Path

from pydantic_ai.exceptions import AgentRunError

from pyRadPlan import PhotonPlan, load_patient
from pyRadPlan.ai.agents import (
    Goal,
    PlanningState,
    Protocol,
    load_ai_env,
    run_planning_agent,
    select_distinct_plans,
)
from pyRadPlan.gui import GUI_AVAILABLE, launch_viewer

logging.basicConfig(level=logging.WARNING)

# The phantom ships with matRad (matRad/phantoms); MATRAD_PHANTOMS points to that folder
HN_PATH = Path(os.environ.get("MATRAD_PHANTOMS", "matRad/phantoms")) / "HEAD_AND_NECK.mat"

load_ai_env()  # ANTHROPIC_API_KEY from .env
MODEL = "anthropic:claude-sonnet-5-5"

# %% [markdown]
# ## Patient, plan and protocol
# The phantom ships with matRad objectives, which are dropped so the agent starts from
# scratch. Doses are total course doses (`dose_convention="total"`).

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
# Each dose influence calculation (at most `state.max_dij_calcs`, default 3) and
# optimization (at most `max_optimizations`) runs locally; `request_limit` caps the
# number of model requests and with it the cost. Every optimized plan is kept in
# `state.plans`, also when the run stops early (e.g. at the request limit).

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
except AgentRunError as e:  # e.g. request limit reached
    result = None
    print(f"Agent stopped: {e}")

if result is not None:
    u = result.usage
    print(
        f"requests: {u.requests}, tool calls: {u.tool_calls}, input: {u.input_tokens} "
        f"(cache read {u.cache_read_tokens}, cache write {u.cache_write_tokens}), "
        f"output: {u.output_tokens}"
    )
    print(result.output.model_dump_json(indent=2))
for step in state.history:  # what the agent did, step by step
    print(step)

# %% [markdown]
# ## The two best plans that trade off differently
# Plans are ranked by (all hard goals passed, score). The score is the weighted mean of
# each goal's margin relative to its limit, clipped to [-100 %, +20 %]: higher is
# better, >= 0 when every goal passes. Plan 2 must beat plan 1 by >= 5 % of a goal's
# limit on at least one goal, so it is not just a slightly worse copy of plan 1.

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
# The plot shows every goal's margin per optimization (above the 0 line it is met) and
# the score; the viewer's quantity list holds one dose per selected plan. The main GUI
# shows the same in its agentic UI mode (`pyRadPlan.settings.gui.ui_mode = "agentic"`).

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

# %% [markdown]
"""# Solver configuration and optimization utilities."""
# %% [markdown]
# This example shows how to configure the optimizer via `pln.prop_opt`,
# track the objective history, and work with objectives.

# %%
import logging
import matplotlib.pyplot as plt

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    load_tg119,
)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
ct, cst = load_tg119()

# %% [markdown]
# ## Configuring `pln.prop_opt`
#
# `pln.prop_opt` is a dictionary that controls the optimization.
# All keys are forwarded as attributes to the planning problem.
#
# | Key              | Type   | Default   | Description                                        |
# |------------------|--------|-----------|----------------------------------------------------|
# | `solver`         | str    | `"ipopt"` | `"scipy"` (L-BFGS-B) or `"ipopt"` (Interior Point) |
# | `display`        | bool   | `True`    | Show solver iteration output                       |
# | `max_iter`       | int    | `500`     | Maximum number of iterations                       |
# | `solver_options` | dict   | `{}`      | Solver-specific options (merged into solver defaults)|

# %%
pln = IonPlan(radiation_mode="protons", machine="Generic")

pln.prop_opt = {
    "solver": "ipopt",
    "display": True,  # set False to silence iteration output
    "max_iter": 200,
}

pln.prop_dose_calc = {"dose_grid": ct.grid}

stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# ## Available objectives
#
# Objectives are assigned per VOI via `cst.vois[i].objectives`.
# Each has a `priority` (weight) and objective-specific parameters.
#
# | Objective             | Key parameters           | Use case                         |
# |-----------------------|--------------------------|----------------------------------|
# | `SquaredDeviation`    | `d_ref`                  | Target dose prescription         |
# | `SquaredOverdosing`   | `d_max`                  | Upper dose limit (OAR/target)    |
# | `SquaredUnderdosing`  | `d_min`                  | Lower dose limit (target)        |
# | `MeanDose`            | `d_ref`                  | Mean dose objective              |
# | `EUD`                 | `eud_ref`, `k`- exponent | Equivalent uniform dose          |
# | `MinDVH`              | `d`, `v_min`             | DVH lower bound at volume %      |
# | `MaxDVH`              | `d`, `v_max`             | DVH upper bound at volume %      |
# | `DoseUniformity`      | *(none)*                 | Minimize dose variance           |
#
# The `priority` parameter controls the relative weighting of the objectives.
# Check `pyRadPlan.optimization.objectives` for more info

# %%
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [MeanDose(priority=1.0, d_ref=0.0)]  # Body

# %% [markdown]
# ## Tracking optimization info
#
# Pass an empty dict as ``opt_info`` to ``fluence_optimization``.
# After the solve it will contain ``obj_history`` (objective values per
# evaluation), ``num_iter`` (number of iterations), and ``result_info``
# (solver-specific result details like convergence status).

# %%
opt_info = {}
fluence = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

# %%
# Plot convergence
plt.figure()
plt.semilogy(opt_info["obj_history"])
plt.xlabel("Function evaluation")
plt.ylabel("Objective value")
plt.title("Optimization convergence")
plt.grid(True)
plt.show()

# %% [markdown]
# Note that the number of iterations may differ to the number set above.
# This is due to solver specific internal evaluations (e.g. line search ...)
#
# You can also inspect ``opt_info["num_iter"]`` and ``opt_info["result_info"]``
# for additional solver details.

# %% [markdown]
# ## Solver-specific options via `solver_options`
#
# Use `solver_options` to override solver-internal defaults.
# These are passed directly into the solver's `options` dict.
#
# **SciPy (L-BFGS-B):**
# ```python
# pln.prop_opt = {
#     "solver": "scipy",
#     "solver_options": {
#         "ftol": 1e-8,   # function value change tolerance (default 1e-5)
#         "gtol": 1e-8,   # gradient norm tolerance (default 1e-5)
#     },
# }
# ```
#
# **IPOPT:**
# ```python
# pln.prop_opt = {
#     "solver": "ipopt",
#     "solver_options": {
#         "tol": 1e-12,                    # overall convergence tolerance (default 1e-10)
#         "acceptable_obj_change_tol": 1e-6,  # acceptable objective change (default 1e-4)
#         "dual_inf_tol": 1e-6,            # dual infeasibility tolerance (default 1e-4)
#         "compl_inf_tol": 1e-6,           # complementarity tolerance (default 1e-4)
#         "acceptable_iter": 10,           # iterations within acceptable tol (default 5)
#     },
# }
# ```

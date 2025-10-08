# %% [markdown]
"""# Example for using LET as quantity during optimization."""
# %% [markdown]
# This example demonstrates how pyRadPlan handles additional quantities such as LET.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/proton_let.py

# %%
# Import necessary libraries
import logging

import numpy as np

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    load_tg119,
)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# Just like in the other examples, we load the ct and cst, create the Plan and Steeringinformation
# %%
# load TG119
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# When defining the objective functions and its parameters, one can add the respective quantity.
# Standard quantity is physical dose.
# %%
# OAR
cst.vois[0].objectives = [
    SquaredOverdosing(priority=10.0, d_max=1.0),
    SquaredOverdosing(priority=5.0, d_max=1.0, quantity="let_dose"),
]
# Target
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
# BODY
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]

# %% [markdown]
# We then calculate the optimized fluence and the corresponding result of the plan.
# Additionally we plot both the physical dose and the LET.
# %%
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result
result = dij.compute_result_ct_grid(fluence)

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
)
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["let"],
    view_slice=view_slice,
)

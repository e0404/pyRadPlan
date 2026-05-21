# %% [markdown]
"""Demonstrates the use of pyRadPlan for proton dose calculation using the FRED engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform proton dose calculations with TOPAS.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_proton.py
# %%
# Import necessary libraries
import logging


from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    calc_dose_forward,
    load_tg119,
    fluence_optimization,
)


logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()
# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_stf = {
    "gantry_angles": [90, 180],
    "couch_angles": [0, 0],
}
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

dij = calc_dose_influence(ct, cst, stf, pln)

fluence = fluence_optimization(ct, cst, stf, dij, pln)

result = dij.compute_result_ct_grid(fluence)
# %% [markdown]
# TOPAS recalc
pln.prop_dose_calc = {
    "engine": "TOPAS",  # necessary for the following to take effect
    "num_histories_direct": 1e7,
    "external_calculation": True,  # "E:\Code\pyRadPlan\pyRadPlan\data\TOPAS\2026-02-18",
}
result_mc = calc_dose_forward(ct, cst, stf, pln, fluence)

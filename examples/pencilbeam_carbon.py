# %% [markdown]
"""# Example for carbon dose calculation using pencilbeam engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform carbon ion dose calculations with biological effect modeling.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_carbon.py

# %%
# Import necessary libraries
import logging

import numpy as np

from pyRadPlan import (
    IonPlan,
    load_tg119,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_distributions,
)

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a carbon ion therapy plan with biological effect calculation.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="carbon", machine="Generic")
pln.prop_stf = {"bixel_width": 4}
pln.prop_dose_calc = {"calc_bio_dose": True}
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimize
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Result
result = dij.compute_result_ct_grid(fluence)

# %%
# Choose slices to visualize
view_slice = [int(np.round(ct.size[2] / 2))]

# Visualize the results
# Use plot_distributions to visualize the biological effect and physical dose
# use plot_slice() for single distributions
plot_distributions(
    ct=ct,
    cst=cst,
    overlays=[result["effect"], result["physical_dose"]],
    view_slice=view_slice,
    plane="axial",
    overlay_unit=["dimensionless", "Gy"],
    overlay_titles=["Biological Effect", "Physical Dose"],
    show_plot=True,
)

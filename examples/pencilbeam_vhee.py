# %% [markdown]
"""# Example for VHEE dose calculation using pencilbeam engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform dose calculations using the VHEE model.

# IMPORTANT: Please use the ipopt optimizer, because scipy might not converge or converge to fast.
# (Will be updated in the future)

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_vhee.py

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

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a VHEE therapy plan using the FermiEyges machine. </br>
# We choose "Generic" for FermiEyges or "Focused" for a focused VHEE machine. </br>
# We set the energy to 200 MeV, the bixel width to 5 mm and select 5 gantry angles with 72 degree spacing.
# %%
# Create a plan object - (VHEE | Generic) => FermiEyges
pln = IonPlan(radiation_mode="VHEE", machine="Generic")  # Alternative: Focused
pln.prop_opt = {"solver": "scipy"}  # better use IPOPT from ipyopt!

# Some specific Settings for VHEE
pln.prop_stf = {
    "energy": 200,  # set VHEE energy at [100, 150, 200] MeV
    "bixel_width": 5.0,
    "gantry_angles": [0, 72, 144, 216, 288],
    "couch_angles": [0, 0, 0, 0, 0],
}

# Setting the dose grid resolution
pln.prop_dose_calc = {"dose_grid": {"resolution": [5.0, 5.0, 5.0]}}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# Optimization
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [SquaredOverdosing(priority=10.0, d_max=2.0)]  # BODY

# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# Visualize the results
# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

# %%

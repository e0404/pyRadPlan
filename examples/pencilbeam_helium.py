# %% [markdown]
"""# Example for helium dose calculation using pencilbeam engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform helium dose calculations.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_helium.py

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


logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a helium therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="helium", machine="Generic")
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# Visualize the results
# %%
# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
plot_slice(
    image_volume=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)
plot_slice(
    image_volume=ct,
    cst=cst,
    overlay=result["rbe_x_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

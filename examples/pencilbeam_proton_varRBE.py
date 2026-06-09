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

# %% [markdown]
# Just like in the other examples, we load the ct and cst, create the Plan and Steeringinformation
# %%
# load TG119
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.bio_model = "MCN"

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# When defining the objective functions and its parameters, one can add the respective quantity.
# Standard quantity is physical dose.

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
    image_volume=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
)
plot_slice(
    image_volume=ct,
    cst=cst,
    overlay=result["rbe_x_dose"],
    view_slice=view_slice,
)

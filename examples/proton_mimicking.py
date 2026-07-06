# %% [markdown]
"""# Example for proton dose mimicking of an existing dose distribution."""
# %% [markdown]
# This example demonstrates how to use the perform dose mimicking on an existing dose distribution.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_proton.py

# %%
# Import necessary libraries
import logging

import numpy as np
import SimpleITK as sitk

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    load_tg119,
    xp_utils,
)

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    MeanDose,
    SquaredMimicking,
)


xp_utils.PREFER_GPU = False
xp_utils.PREFERRED_CPU_ARRAY_BACKEND = "numpy"
logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {"dose_grid": ct.grid}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %%
# Optimize a standard plan
# Optimization
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)

# %%
# Now let's mimic a reference dose distribution from the reference plan with noise
noise_stdev = 0.02
reference_dose = sitk.SpeckleNoise(result["physical_dose"], noise_stdev)

# Let's mimic the noisy dose distribution by using the SquaredMimicking objective.
cst.vois[1].objectives = []  # Target
cst.vois[0].objectives = []  # OAR
cst.vois[2].objectives = [SquaredMimicking(priority=1000.0, d_ref=reference_dose)]  # BODY

# Calculate optimized fluence
fluence_mimicked = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result on the CT grid
result_mimicked = dij.compute_result_ct_grid(fluence_mimicked)

# %% [markdown]
# Visualize the results
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result_mimicked)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[2] / 2))

    # Visualize
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result_mimicked["physical_dose"],
        view_slice=view_slice,
        plane="axial",
        overlay_unit="Gy",
    )

# %%

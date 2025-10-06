# %% [markdown]
"""# Example for proton dose calculation using the FRED engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform proton dose calculations using the FRED Monte Carlo engine.

# For installation instructions, please refer to https://www.fred-mc.org/

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/proton_MC_FRED.py

# %%
# Import necessary libraries
import logging
import numpy as np

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    calc_dose_forward,
    load_tg119,
    fluence_optimization,
    plot_slice,
)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the FRED engine.
# First, we define the parameters for the steering information (`prop_stf`).
# Next, we specify the settings for dose calculation (`prop_dose_calc`),
# including compatibility with different FRED versions through `dij_format_version`.
# Finally, we configure the optimization parameters (`prop_opt`), as demonstrated in previous examples.

# Use less histories and beams when experimenting. This reduces computation time significantly.

# %%
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_stf = {
    "gantry_angles": [0, 180],  # define gantry angles for n beams
    "couch_angles": [0, 0],
    "longitudinal_spot_spacing": 2.0,
    "iso_center": np.array([[-5.0, -5.0, -5.0], [-5.0, -5.0, -5.0]]),  # two beams
    "bixel_width": 10,
    "add_margin": 1,
}
pln.prop_dose_calc = {
    "engine": "FRED",  # Use the FRED engine for dose calculation
    "fred_version": "3.76.0",  # std: "3.70.0"
    "use_gpu": True,  # std: True | one has to set GPU = 1 using 'fred -config' manually!
    "calc_let": False,  # std: False
    "scorers": ["Dose"],  # std: ["Dose"]
    "room_material": "Air",  # vacuum | std: Air
    "print_output": False,
    "num_histories_per_beamlet": 2e3,  # use less histories for faster computation (e.g. 2e2) or less beams
    # "dij_format_version": "31", # 20/30 for fred 3.70.0 and 21/31 for fred 3.76.0
    # "save_input": "path/to/save/input",
    # "save_output": "path/to/save/output", # or True (for current dir)/False | std: False (using temp path)
    # "use_output": "path/to/use/output",  # skips output file generation if only 'use_output_file' is set and not 'save_output_file'
}

pln.prop_opt = {"solver": "scipy"}

# %% [markdown]
# Generate Steering Geometry ("stf")
# %%
stf = generate_stf(ct, cst, pln)

# %% [markdown]
# Run calc_dose_influence (dose influence matrix)
# %%
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# Define cst objectives and run fluence optimization
# %%
cst.vois[0].objectives = [SquaredOverdosing(priority=100.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=1000.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

fluence = fluence_optimization(ct, cst, stf, dij, pln)

# %% [markdown]
# Compute dose distribution and other quantities given in pln.prop_opt
# %%
result_opt = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# Visualize the results
# %%

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

plot_slice(
    ct=ct,
    cst=cst,
    overlay=result_opt["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

# %% [markdown]
# You can also run calc_dose_forward (direct dose calculation) without optimization.
# %%
# Set run_direct_dose_calc to True to perform direct dose calculation
run_direct_dose_calc = False
if run_direct_dose_calc:
    result_fwd = calc_dose_forward(ct, cst, stf, pln, np.ones(stf.total_number_of_bixels))

# %% [markdown]
"""# Example for photon dose calculation and sequencing."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform photon dose calculations and sequencing.
# We use the photon pencil beam SVD engine to create a plan for a TG119 phantom.
# The implementation is based on the Singular-value decomposition (SVD) method by Bortfeld, Schlegel & Rhein (1993)

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/proton_MC_FRED.py

# %%
# Import necessary libraries

import logging

import numpy as np
import SimpleITK as sitk
from copy import deepcopy

from pyRadPlan import (
    PhotonPlan,
    load_tg119,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    run_sequencing,
)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a photon therapy plan using the PhotonPencilBeamSVDEngine.
# %%
# 1) Create a plan object
pln = PhotonPlan(machine="Generic")

# 2) Define the parameters for the steering information
num_of_beams = 5
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
}

# 3) Specify the settings for dose calculation
pln.prop_dose_calc = {"dose_grid": {"resolution": [5.0, 5.0, 5.0]}}

# 4) Configure optimization parameters
pln.prop_opt = {"solver": "scipy"}

# 5) Configure sequencing
# Nondefined parameters in the regarding class will be ignored.
pln.prop_seq = {
    "run_sequencing": True,  # False if you want to skip the sequencing step
    "num_of_levels": 5,
    "sequencer": "xia",  # currently implemented: "xia" and "siochi"
    "mode": "reducing_levels",  # parameter for "xia" sequencer.
}


# %% [markdown]
# Just like in all the other examples, we first generate the stf, used for the calculation of the dij.
# %%
# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
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

# Compute dose distribution and other quantities given in pln.prop_opt
result = fluence_optimization(ct, cst, stf, dij, pln)

# %% [markdown]
# Now for the sequencing step, we use the `run_sequencing` function. <br>
# This will run the sequencing algorithm specified in the plan object. <br>
# The result will be stored in the `result` object, which contains the sequenced dose. <br>
# You can call the resulting information via "result.sequence".
# %%

result_sequenced = run_sequencing(deepcopy(result), pln, stf, dij=dij)

# %% [markdown]
# Visualize the results. (There is no plotting module for the sequenced data yet) <br>
# For plotting the individual doses, use the `plot_slice` function.
# %%

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Lets first visualize the original dose
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result_sequenced.physical_dose.values,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)
# %%
# And just out of curiosity, the difference between the original and sequenced dose
a = sitk.GetArrayFromImage(result.physical_dose.values)
b = sitk.GetArrayFromImage(result_sequenced.physical_dose.values)
diff = a - b
plot_slice(
    ct=ct,
    cst=cst,
    overlay=diff,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

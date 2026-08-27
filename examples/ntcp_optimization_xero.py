# %% [markdown]
"""# Example for proton dose calculation using pencilbeam engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to perform proton dose calculations.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_proton.py

# %%
# Import necessary libraries
import logging
from pyRadPlan import (
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    xp_utils,
)

from pyRadPlan.gui.apps import launch_viewer
from pyRadPlan.io import load_patient
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    OutcomeCNN,
)
import numpy as np
import os

from pyRadPlan.plan._plans import PhotonPlan

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
xp_utils.PREFER_GPU = True
xp_utils.PREFERRED_CPU_ARRAY_BACKEND = "numpy"
xp_utils.PREFERRED_GPU_ARRAY_BACKEND = "cupy"
logging.basicConfig(level=logging.INFO)
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_patient(r"path/to/hn_patient.mat")

cst.vois[0] = cst.vois[12]  # PAROTID_LT
cst.vois[1] = cst.vois[13]  # PAROTID_RT
cst.vois[2] = cst.vois[14]  # PTV63
cst.vois[3] = cst.vois[15]  # PTV70
cst.vois = cst.vois[:4]
cst.create_body_seg()  # adds BODY/SKIN as vois[4]

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = PhotonPlan(radiation_mode="photons", machine="Generic")
pln.prop_opt = {"solver": "ipopt", "max_iter": 200, "display": True}
pln.prop_dose_calc = {"dose_grid": {"resolution": [5.0, 5.0, 5.0]}}
num_of_beams = 8
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
}
stf = generate_stf(ct, cst, pln)
# Optimization
# Configure CNN-based NTCP objective for an OAR

# Keep handle to the CNN objective to access NTCP history after optimization.
# `model` accepts a HuggingFace name, a local model directory, or a preloaded
# (model, preprocessor) tuple. A matching HF repo must exist for the name below.
ntcp_obj = OutcomeCNN(
    priority=150000.0,
    model="pyRadPlan-outcome-ORPDenseNet-Xerostomia",
    mode="NTCP",
    warmup_iterations=5,
    ct=ct,  # needed as model input
    cst=cst,  # needed as model input
    cst_masks=[
        "Parotid_LT",
        "Parotid_RT",
    ],  # channel order matters unless the model collapses masks
    debug_plots="debug_grads",
)

cst.vois[0].objectives = [  # PAROTID_LT
    SquaredOverdosing(priority=100.0, d_max=25.0),
]
cst.vois[1].objectives = [  # PAROTID_RT
    SquaredOverdosing(priority=100.0, d_max=25.0),
]
cst.vois[2].objectives = [SquaredDeviation(priority=1000.0, d_ref=63.0)]  # PTV63
cst.vois[3].objectives = [SquaredDeviation(priority=1000.0, d_ref=70.0)]  # PTV70
cst.vois[4].objectives = [SquaredOverdosing(priority=800.0, d_max=30.0)]  # SKIN/BODY


# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# The CNN reads the dose over the whole patient, which SKIN/BODY already covers -- the
# objective restricts to its VOI and scatters back onto the dose grid internally.
cst.vois[4].objectives.append(ntcp_obj)
opt_info = {}
# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

# Save NTCP history from active optimization
ntcp_history_active = list(ntcp_obj.ntcp_history)
ntcp_obj.reset_history()

# Native optimization: switch CNN NTCP objective to monitor-only mode (logs but zero gradients)
ntcp_obj.monitor_only = True

opt_info_native = {}
fluence_native = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info_native)

# Save NTCP history from the native (monitor-only) optimization
ntcp_history_native = list(ntcp_obj.ntcp_history)


# # Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)
result_native = dij.compute_result_ct_grid(fluence_native)
# %% [markdown]
# Native plots: dose, native dose, signed difference; NTCP curve; DVHs
# %%


result_collective = result.copy()
result_collective["physical_dose_native"] = result_native["physical_dose"]
result_collective["physical_dose_diff"] = result["physical_dose"] - result_native["physical_dose"]

# %% [markdown]
# NTCP history: CNN-guided (active gradients) vs native (monitor-only) optimization.
# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(range(1, len(ntcp_history_active) + 1), ntcp_history_active, label="CNN-guided")
ax.plot(range(1, len(ntcp_history_native) + 1), ntcp_history_native, label="native (monitor)")
ax.set_xlabel("Objective evaluation")
ax.set_ylabel("NTCP")
ax.set_title("NTCP history during optimization")
ax.legend()
plt.show()

launch_viewer(ct, cst, result_collective)

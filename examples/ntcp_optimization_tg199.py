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
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    load_tg119,
    xp_utils,
)

from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    OutcomeCNN,
)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
xp_utils.PREFER_GPU = True
xp_utils.PREFERRED_CPU_ARRAY_BACKEND = "numpy"
xp_utils.PREFERRED_GPU_ARRAY_BACKEND = "cupy"
logging.basicConfig(level=logging.INFO)
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "ipopt", "max_iter": 200, "display": True}
pln.prop_dose_calc = {"dose_grid": ct.grid}
pln.prop_stf = {
    "gantry_angles": [0.0, 120.0, 240.0],
    "couch_angles": [0.0, 0.0, 0.0],
}
stf = generate_stf(ct, cst, pln)
# Optimization
# Configure CNN-based NTCP objective for an OAR

# Create the CNN objective (channels order for the model: [ct, dose, mask])
# Keep handle to the CNN objective to access NTCP history after optimization
ntcp_obj = OutcomeCNN(
    priority=10000.0,
    model="pyRadPlan-outcome-ORPDenseNet-tg119",  # HF name, local dir, or (model, preprocessor)
    mode="NTCP",
    warmup_iterations=5,
    ct=ct,
    cst=cst,
    cst_masks=["Core"],  # channel order matters unless the model collapses masks
    # debug_plots="debug_grads",
)

cst.vois[0].objectives = [
    SquaredOverdosing(priority=10.0, d_max=30.0),
]
# Keep remaining objectives as in the baseline example
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=70.0)]  # Target
cst.vois[2].objectives = [
    # MeanDose(priority=100.0, d_ref=0.0),
    SquaredOverdosing(priority=50.0, d_max=20.0),
]  # BODY


# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

cst.add_global_voi()
cst.vois[3].objectives = [ntcp_obj]
opt_info = {}
# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

# Save NTCP history from active optimization
ntcp_history_active = list(ntcp_obj.ntcp_history)
ntcp_obj.reset_history()

# Native optimization: switch CNN NTCP objective to monitor-only mode (logs but zero gradients)
ntcp_obj.monitor_only = True
cst.vois[3].objectives = [ntcp_obj]

opt_info_native = {}
fluence_native = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info_native)


# # Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)
result_native = dij.compute_result_ct_grid(fluence_native)
# %% [markdown]
# Native plots: dose, native dose, signed difference; NTCP curve; DVHs
# %%

diff = result_native["physical_dose"] - result["physical_dose"]
plot_slice(
    ct=ct,
    cst=cst,
    overlay=diff,
)

# %% [markdown]
"""# Example for CNN-based NTCP optimization on the TG119 phantom."""
# %% [markdown]
# This example demonstrates how to optimize a proton plan with a CNN-predicted NTCP for the Core.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/ntcp_optimization_tg119.py

# %%
# Import necessary libraries
import logging

import SimpleITK as sitk

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    load_tg119,
    settings,
)
from pyRadPlan.cst import HelperVOI
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    OutcomeCNN,
)

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
settings.xp.prefer_gpu = True
settings.xp.preferred_cpu_array_backend = "numpy"
settings.xp.preferred_gpu_array_backend = "cupy"
logging.basicConfig(level=logging.INFO)


# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
# The CNN was trained on physical dose, so keep the objectives on physical dose.
pln.prop_opt = {
    "solver": "scipy",
    "max_iter": 200,
    "display": True,
    "convert_dose_objectives": False,
}
pln.prop_dose_calc = {"dose_grid": ct.grid}
pln.prop_stf = {
    "gantry_angles": [0.0, 120.0, 240.0],
    "couch_angles": [0.0, 0.0, 0.0],
}
stf = generate_stf(ct, cst, pln)

# %%
# Configure the CNN-based NTCP objective for the Core.
# Keep a handle to it to access the NTCP history after optimization.
ntcp_obj = OutcomeCNN(
    priority=10000.0,
    model="pyRadPlan-outcome-ORPDenseNet-tg119",  # HF name, local dir, or (model, preprocessor)
    warmup_iterations=5,
    ct=ct,
    cst=cst,
    cst_masks=["Core"],  # channel order matters unless the model collapses masks
)

cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=30.0)]  # Core
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=70.0)]  # Target
cst.vois[2].objectives = [SquaredOverdosing(priority=50.0, d_max=20.0)]  # BODY


# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# The CNN sees the dose of the VOI it is attached to. A helper VOI over the whole grid,
# excluded from overlap priorities, hands it the full dose including Core and Target.
cnn_mask = sitk.Image(ct.cube_hu.GetSize(), sitk.sitkUInt8) + 1
cnn_mask.CopyInformation(ct.cube_hu)
cst.vois = cst.vois + [
    HelperVOI(
        name="NTCP_CNN",
        ct_image=ct,
        mask=cnn_mask,
        apply_overlap=None,
        objectives=[ntcp_obj],
    )
]

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
# Signed difference between the native and the CNN-guided dose.
# %%

diff = result_native["physical_dose"] - result["physical_dose"]
plot_slice(
    ct=ct,
    cst=cst,
    overlay=diff,
)

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

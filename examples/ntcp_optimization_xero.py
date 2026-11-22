# %% [markdown]
"""# Example for CNN-based xerostomia NTCP optimization on the matRad head-and-neck phantom."""
# %% [markdown]
# This example demonstrates how to optimize a photon plan with a CNN-predicted xerostomia NTCP
# for both parotids.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/ntcp_optimization_xero.py

# %%
# Import necessary libraries
import logging

import SimpleITK as sitk

from pyRadPlan import (
    PhotonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    settings,
)
from pyRadPlan.cst import HelperVOI
from pyRadPlan.gui.apps import launch_viewer
from pyRadPlan.io import load_patient
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    OutcomeModel,
)

# HEAD_AND_NECK.mat ships with matRad (matRad/phantoms), not with pyRadPlan.
HN_PHANTOM = r"path/to/matRad/default_hn_phantom.mat"

settings.xp.prefer_gpu = True
settings.xp.preferred_cpu_array_backend = "numpy"
settings.xp.preferred_gpu_array_backend = "cupy"
logging.basicConfig(level=logging.INFO)


# %%
# Load the head-and-neck phantom and keep the structures used below
ct, cst = load_patient(HN_PHANTOM)

cst.vois[0] = cst.vois[12]  # PAROTID_LT
cst.vois[1] = cst.vois[13]  # PAROTID_RT
cst.vois[2] = cst.vois[14]  # PTV63
cst.vois[3] = cst.vois[15]  # PTV70
cst.vois = cst.vois[:4]
cst.create_body_seg()  # adds BODY/SKIN as vois[4]

# %% [markdown]
# In this section, we create a photon IMRT plan.
# %%
# Create a plan object
pln = PhotonPlan(radiation_mode="photons", machine="Generic")
pln.prop_opt = {"solver": "ipax", "max_iter": 200, "display": True}
pln.prop_dose_calc = {"dose_grid": {"resolution": [5.0, 5.0, 5.0]}}

pln.prop_stf = {
    "gantry_angles": [0, 60, 120, 180, 240, 300],
    "couch_angles": [0, 0, 0, 0, 0],
}
stf = generate_stf(ct, cst, pln)

# %%
# Configure the CNN-based xerostomia NTCP objective.
# `model` accepts a HuggingFace name, a local model directory, or a preloaded
# (model, preprocessor) tuple.
ntcp_obj = OutcomeModel(
    priority=300000.0,
    model=r"pyRadPlan-outcome-ORPDenseNet-Xerostomia",
    warmup_iterations=20,
    cst_masks=["PAROTID_LT", "PAROTID_RT"],
    # Only parotid dose drives the NTCP; the dose elsewhere is frozen when the warmup ends.
    gradient_vois=["PAROTID_LT", "PAROTID_RT"],
)

cst.vois[0].objectives = [SquaredOverdosing(priority=100.0, d_max=25.0)]  # PAROTID_LT
cst.vois[1].objectives = [SquaredOverdosing(priority=100.0, d_max=25.0)]  # PAROTID_RT
cst.vois[2].objectives = [SquaredDeviation(priority=1000.0, d_ref=63.0)]  # PTV63
cst.vois[3].objectives = [SquaredDeviation(priority=1000.0, d_ref=70.0)]  # PTV70
cst.vois[4].objectives = [SquaredOverdosing(priority=800.0, d_max=30.0)]  # SKIN/BODY


# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# The CNN sees the dose of the VOI it is attached to. A helper VOI over the whole grid,
# excluded from overlap priorities, hands it the full dose including parotids and PTVs.
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

ntcp_history_active = opt_info["objective_info"][0]["history"]

# Native optimization: the NTCP objective only monitors (zero value and gradient)
ntcp_obj.monitor_only = True

opt_info_native = {}
fluence_native = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info_native)

ntcp_history_native = opt_info_native["objective_info"][0]["history"]


# # Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)
result_native = dij.compute_result_ct_grid(fluence_native)
# %% [markdown]
# CNN-guided dose, native dose and their difference in the viewer; NTCP curve.
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

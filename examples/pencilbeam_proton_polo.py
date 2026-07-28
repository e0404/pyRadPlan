# %% [markdown]
"""# Example for using POLO (Probability of Lesion Origin) as quantity during optimization."""
# %% [markdown]
# This example demonstrates how pyRadPlan handles the Probability of Lesion Origin (POLO) model
# during proton fluence optimization to reduce radiation-induced contrast-enhancing brain lesions (CEBLs).

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_proton_polo.py
# ```

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

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.optimization.objectives import (
    SquaredDeviation,
    SquaredOverdosing,
    MeanDose,
    NTCPPOLO,
)
from pyRadPlan.quantities import POLO

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# Just like in the other examples, we load the CT and CST, create the Plan and SteeringInformation.
# %%
# Load TG119 surrogate phantom
ct, cst = load_tg119()

# Create a proton plan object
pln = IonPlan(
    radiation_mode="protons",
    machine="Generic",
)
pln.prop_opt = {"solver": "scipy", "convert_dose_objectives": False}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# To use the POLO model, we attach a binary clinical risk mask ($R_i \in \{0, 1\}$) to the influence matrix.
# In clinical practice, this mask indicates voxels within $\le 4\text{ mm}$ distance to the ventricular system ($R_i = 1$).
# For our TG119 surrogate phantom, we designate the central OAR ('Core') as the clinical risk structure ($R_i = 1$).
# %%
import SimpleITK as sitk
from pyRadPlan.core.resample import resample_image

oar_mask = cst.vois[0].mask
risk_mask_img = resample_image(
    oar_mask,
    target_grid=dij.dose_grid,
    interpolator=sitk.sitkNearestNeighbor,
    extrapolate=False,
)

# Attach flattened binary risk mask to dij (auto-detected by POLO quantity during optimization)
dij.binary_risk_mask = sitk.GetArrayViewFromImage(risk_mask_img).flatten()

# %% [markdown]
# When defining the objective functions and parameters, we add the POLO objective (e.g., `NTCPPOLO` or `LogSumExpConvPOLO`).
# This penalizes distal high-LET deposition in high-dose regions overlapping our clinical risk mask.
# %%
# OAR ('Core' - simulating the ventricular risk structure)
cst.vois[0].objectives = [
    SquaredOverdosing(priority=10.0, d_max=15.0),
]
# Target ('OuterTarget' - prescribed to 50.0 Gy cumulative course dose)
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=50.0)]
# BODY
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=33.0),
    NTCPPOLO(priority=100.0),  # Evaluate POLO NTCP over the entire normal brain (BODY)
]

# %% [markdown]
# We calculate the optimized fluence and compute the resulting physical dose and POLO probability map.
# %%
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result
result = dij.compute_result_ct_grid(fluence)

# Explicitly compute total NTCP for reporting
polo_qty = POLO(dij, dij.binary_risk_mask)
polo_flat = polo_qty.compute(fluence).flat[0]
total_ntcp = 1.0 - np.prod(1.0 - np.clip(polo_flat, 0.0, 1.0 - 1e-12))
logging.info(f"Optimized Plan Total POLO NTCP: {total_ntcp:.4%}")

# %% [markdown]
# Finally, we visualize the results using either the built-in PyRadPlan GUI or Matplotlib slice plots.
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[2] / 2))

    # Visualize Physical Dose and POLO probability
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result["physical_dose"],
        view_slice=view_slice,
    )
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result["polo"],
        view_slice=view_slice,
    )

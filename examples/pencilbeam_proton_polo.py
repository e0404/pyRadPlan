# %% [markdown]
"""# Example for using POLO (Probability of Lesion Origin) NTCP Model during optimization."""
# %% [markdown]
# This example demonstrates how pyRadPlan handles Normal Tissue Complication Probability (NTCP) modeling
# using the Probability of Lesion Origin (POLO) model during proton fluence optimization to reduce
# radiation-induced contrast-enhancing brain lesions (CEBLs).
#
# References:
# - Bahn, E., Bauer, J., Harrabi, S. B., et al. (2020). Late contrast enhancing brain lesions in proton-treated patients with low-grade glioma. IJROBP, 107(3), 571–578. DOI: 10.1016/j.ijrobp.2020.03.013
# - Ortkamp, T., Sallem, H., Harrabi, S., et al. (2026). Direct optimization of the probability of lesion origin in proton treatment planning for low-grade glioma patients. Med Phys, 53, e70395. DOI: 10.1002/mp.70395

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
# When defining objective functions, we add the POLO objective (`NTCPPOLO`).
# For our TG119 surrogate phantom, we pass `risk_voi_name="Core"` to designate the central OAR ('Core') as the periventricular risk structure.
# %%
# OAR ('Core' - simulating the ventricular risk structure)
cst.vois[0].objectives = [
    SquaredOverdosing(priority=10.0, d_max=15.0),
]
# Target ('OuterTarget' - prescribed to 50.0 Gy cumulative course dose)
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=50.0)]

# BODY (Evaluate general max dose over standard BODY)
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=33.0),
]

# Create 'BODY_minus_Target' VOI to prevent POLO from penalizing dose inside the tumor
import SimpleITK as sitk
from pyRadPlan.cst import VOI

body_mask = next(v.mask for v in cst.vois if v.name == "BODY")
target_mask = next(v.mask for v in cst.vois if v.name == "OuterTarget")

# Subtract target mask from body mask
body_minus_target_mask = sitk.MaskNegated(body_mask, target_mask)

bmt_voi = VOI(
    name="BODY_minus_Target",
    voi_type="OAR",
    mask=body_minus_target_mask,
    grid=cst.vois[2].grid,
    objectives=[NTCPPOLO(priority=100.0, risk_voi_name="Core")],
)
cst.vois.append(bmt_voi)

# %% [markdown]
# We calculate the optimized fluence and compute the resulting physical dose and POLO probability map.
# %%
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute standard physical result (without POLO quantity)
result = dij.compute_result_ct_grid(fluence)

# Attach risk mask to polo_qty for result calculation (it evaluates POLO quantity on all voxels)
polo_qty = POLO(dij, cst=cst, risk_voi_name="Core")
result["polo_full"] = polo_qty.compute_ct_grid(fluence)

# Manually zero out voxels outside of BODY_minus_Target
# so the exported 3D image accurately reflects the optimized VOI.
result["polo_masked"] = sitk.Mask(result["polo_full"], sitk.Cast(bmt_voi.mask, sitk.sitkUInt8))

# Compute total NTCP for reporting (masked to the evaluation VOI)
total_ntcp = polo_qty.compute_total_ntcp(fluence, mask=bmt_voi.mask)
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
        overlay=result["polo_full"],
        view_slice=view_slice,
        overlay_cmap="inferno",
    )
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result["polo_masked"],
        view_slice=view_slice,
        overlay_cmap="inferno",
    )

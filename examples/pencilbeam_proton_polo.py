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
import SimpleITK as sitk

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
from pyRadPlan.cst import VOI

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# Just like in the other examples, we load the CT and CST, create the Plan and SteeringInformation.
#
# POLO was fitted on the dose of the whole treatment course, so the plan states its number of
# fractions, and `dose_convention="total"` lets us write the objectives in course dose as well.
# The dose influence matrix itself always holds the dose of one fraction. 50 Gy in 30 fractions
# (about 1.7 Gy each) is close to the conventional 1.8-2 Gy fractions the model was fitted on; it
# has no correction for fraction size, so hypofractionated plans lie outside its domain.
# %%
# Load TG119 surrogate phantom
ct, cst = load_tg119()

# Create a proton plan object
pln = IonPlan(
    radiation_mode="protons",
    machine="Generic",
    num_of_fractions=30,
    dose_convention="total",
)
pln.prop_opt = {"solver": "scipy", "convert_dose_objectives": False}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# When defining objective functions, we add the POLO objective (`NTCPPOLO`).
# For our TG119 surrogate phantom, we pass `risk_voi_name="Core"` to designate the central OAR
# ('Core') as the periventricular risk structure.
#
# POLO is evaluated on its own auxiliary structure: every voxel of BODY except the target. It
# therefore overlaps both 'BODY' and 'Core'. Overlap resolution normally assigns each voxel to a
# single VOI by priority, which would strip this structure down to nothing -- 'OuterTarget',
# 'Core' and 'BODY' all outrank it -- leaving the objective with no voxels to optimize.
# `overlap_handling="ignore"` exempts it from that: it keeps its full mask, and 'BODY' and 'Core'
# keep theirs.
# %%
vois = {voi.name: voi for voi in cst.vois}

# OAR ('Core' - simulating the ventricular risk structure)
vois["Core"].objectives = [SquaredOverdosing(priority=10.0, d_max=15.0)]

# Target ('OuterTarget' - prescribed to 50.0 Gy over the course)
vois["OuterTarget"].objectives = [SquaredDeviation(priority=100.0, d_ref=50.0)]

# BODY (general dose objectives over the standard BODY contour)
vois["BODY"].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=33.0),
]

# POLO evaluation region: every voxel of BODY except the target, including 'Core'.
#
# This matches the domain the model was fitted on: Bahn et al. estimated it over voxels "within
# the brain and outside the GTV" that received at least 2 Gy -- a purely anatomical/dose
# criterion, with no reference to the risk factor. The 4-mm periventricular fringe (distance to
# the ventricular system <= 4 mm) falls inside that domain, so we include it too: 'Core' carries
# no minimum-dose requirement here, so the optimizer can genuinely spare it if warranted.
polo_voi = VOI(
    name="POLO VOI",
    voi_type="OAR",
    mask=sitk.MaskNegated(vois["BODY"].mask, vois["OuterTarget"].mask),
    grid=vois["BODY"].grid,
    overlap_handling="ignore",
    objectives=[NTCPPOLO(priority=1.0e3, risk_voi_name="Core")],
)
cst.vois.append(polo_voi)

# %% [markdown]
# We optimize the fluence. The result holds the physical dose, reported as course dose because of
# `dose_convention="total"`, and the optimized fluence. The POLO probability map is added to it.
# %%
result = fluence_optimization(ct, cst, stf, dij, pln)
fluence = result.fluence

# Like every pyRadPlan quantity, POLO is computed over the whole dose grid. Only the POLO
# segment is exported, though: the model is defined for brain tissue, so values inside the
# target -- or out in air -- are not meaningful and should not be displayed. Because
# 'POLO VOI' opts out of overlap resolution, its mask is exactly the one the objective saw.
polo_qty = POLO(dij, cst=cst, risk_voi_name="Core", num_of_fractions=pln.num_of_fractions)
polo_full = polo_qty.compute_ct_grid(fluence)
result.add_quantity("polo", sitk.Mask(polo_full, sitk.Cast(polo_voi.mask, sitk.sitkUInt8)))

total_ntcp = polo_qty.compute_total_ntcp(fluence, mask=polo_voi.mask)
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
        overlay=result.physical_dose,
        view_slice=view_slice,
    )

    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result.polo,
        view_slice=view_slice,
        overlay_cmap="inferno",
    )

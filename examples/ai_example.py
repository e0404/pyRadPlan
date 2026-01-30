"""example script to demonstrate the usage of the AI dose engine in pyRadPlan for protons."""

"""torch is not natively installed with pyRadPlan, please install it manually if you want to use this example."""

# Necessary imports
import logging
import numpy as np

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    plot_slice,
    fluence_optimization,
    xp_utils,
    load_tg119,
)

logging.basicConfig(level=logging.INFO)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose


xp_utils.PREFER_GPU = True
xp_utils.PREFERRED_CPU_ARRAY_BACKEND = "numpy"
xp_utils.PREFERRED_GPU_ARRAY_BACKEND = "torch"

# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {
    "dose_grid": {"resolution": [2.0, 2.0, 2.0]},
    "engine": "AI",
    "model": "ConvLSTM",
}
pln.prop_stf = {
    "gantry_angles": [90, 270],
    "couch_angles": [0, 0],
}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimization
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY
fluence = fluence_optimization(ct, cst, stf, dij, pln)
result = dij.compute_result_ct_grid(fluence)

view_slice = int(np.round(ct.size[2] / 2))

plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

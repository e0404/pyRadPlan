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
    settings,
    load_tg119,
    fluence_optimization,
)

logging.basicConfig(level=logging.INFO)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose


settings.xp.prefer_gpu = True
settings.xp.preferred_cpu_array_backend = "numpy"
settings.xp.preferred_gpu_array_backend = "torch"

# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {
    "dose_grid": {"resolution": [2.0, 2.0, 2.0]},
    "engine": "AIBeamlet",
    "model": "pyRadPlan-dosecalc-Det-proton-lung",
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
    image_volume=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

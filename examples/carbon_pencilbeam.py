import logging

import numpy as np

from pyRadPlan import (
    IonPlan,
    load_tg119,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
)

logging.basicConfig(level=logging.INFO)

# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="carbon", machine="Generic")
pln.prop_stf = {"bixel_width": 4}
pln.prop_dose_calc = {"calc_bio_dose": True}
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimize -> Result
result = fluence_optimization(ct, cst, stf, dij, pln)

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result.physical_dose.cube,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result.physical_dose.cube,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="dimensionless",
)

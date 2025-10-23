# %% [markdown]
"""# Example for proton dose calculation using pencilbeam engine."""

# %%
# Necessary imports
from pyRadPlan import (
    calc_dose_influence,
    fluence_optimization,
    generate_stf,
    IonPlan,
    load_tg119,
    plot_slice,
)

#  Read patient from provided TG119.mat file and validate data
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan()

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Fluence Optimization (objectives loaded from "cst")
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Result
result = dij.compute_result_ct_grid(fluence)

# Plot
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    overlay_unit="Gy",
)

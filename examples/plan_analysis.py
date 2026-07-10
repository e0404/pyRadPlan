"""Plan analysis example with QI and DVH."""

# %% [markdown]
# # Plan analysis example
#
# This notebook-style script shows how to compute and visualize Quality Indicators (QI)
# and Dose-Volume Histograms (DVHs) for a simple proton plan.

# %%
import logging

import numpy as np

from pyRadPlan import (
    IonPlan,
    calc_dose_influence,
    fluence_optimization,
    generate_stf,
    load_tg119,
    plot_slice,
    settings,
)
from pyRadPlan.analysis import DVHCollection, QICollection, DX, VX, Mean
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

# Prefer GPU when available, fall back to numpy on CPU
settings.xp.prefer_gpu = True
settings.xp.preferred_cpu_array_backend = "numpy"
logging.basicConfig(level=logging.INFO)


# %%
ct, cst = load_tg119()
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)

cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

fluence = fluence_optimization(ct, cst, stf, dij, pln)
# Result
result = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# ## Quick single QIs

# %%
D2 = DX.compute_from(quantity=result["physical_dose"], mask=cst.vois[1].mask, ref_vol=2)
V3Gy = VX.compute_from(quantity=result["physical_dose"], mask=cst.vois[1].mask, ref_dose=3.0)
mean = Mean.compute_from(quantity=result["physical_dose"], mask=cst.vois[1].mask)

print(f"{D2.metric}: {D2.value:.2f} {D2.unit:~}")
print(f"{V3Gy.metric}: {V3Gy.value:.2f} {V3Gy.unit:~}")
print(f"{mean.metric}: {mean.value:.2f} {mean.unit:~}")

# %% [markdown]
# ## QI collection for multiple structures and metrics

# %%
qi_collection = QICollection.from_structure_set(
    cst=cst,
    dose=result["physical_dose"],
    ref_doses=[2],
    ref_vols=[50],
)

# print max dose to target
target_name = cst.vois[1].name
qi = qi_collection[target_name]["max"]
print(f"{target_name} {qi.metric}: {qi.value:.2f} {qi.unit:~}")
# %%
# Plot as table
qi_collection.plot(metrics=["mean", "max", "D50", "V2Gy"])

# %% [markdown]
# ## DVH example

# %%
dvh_collection = DVHCollection.from_structure_set(cst=cst, dose=result["physical_dose"])

# Plot DVHs for target and OAR
target_name = cst.vois[1].name
oar_name = cst.vois[0].name
dvh_collection.plot(structures=[target_name, oar_name], line_width=2, plot_legend=True)

# %% [markdown]
# ## Optional: slice visualization

# %%
view_slice = int(np.round(ct.size[2] / 2))
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result["physical_dose"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)
# %%

# %%
# Import necessary libraries
import logging


from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    load_tg119,
)
import SimpleITK as sitk
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

# %% [markdown]
# Just like in the other examples, we load the ct and cst, create the Plan and Steeringinformation
# %%
# load TG119
ct, cst = load_tg119()

# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.bio_model = "none"
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# When defining the objective functions and its parameters, one can add the respective quantity.
# Standard quantity is physical dose.

# %% [markdown]
# We then calculate the optimized fluence and the corresponding result of the plan.
# Additionally we plot both the physical dose and the LET.
# %%
fluence = fluence_optimization(ct, cst, stf, dij, pln)


result = dij.compute_result_ct_grid(fluence)

profile = {}
bio_models = ["none", "constant_rbe", "WED", "MCN", "CAR", "LSM"]
profile["none"] = sitk.GetArrayFromImage(result["physical_dose"])[
    int(ct.size[0] / 2), :, int(ct.size[2] / 2)
]

# %%
for model in bio_models[1:]:
    pln.bio_model = model
    dij = calc_dose_influence(ct, cst, stf, pln)  # alpha and beta parameters need to be calculated
    result = dij.compute_result_ct_grid(fluence)
    profile[model] = sitk.GetArrayFromImage(result["rbe_x_dose"])[
        int(ct.size[0] / 2), :, int(ct.size[2] / 2)
    ]

# %%
plt.figure(figsize=(10, 6))
for model in bio_models:
    plt.plot(profile[model], label=model)
plt.legend()
# %%

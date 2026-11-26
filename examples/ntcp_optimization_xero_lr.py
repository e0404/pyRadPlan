# %% [markdown]
"""# Xerostomia NTCP optimization with a scikit-learn outcome model."""
# %% [markdown]
# This example steers a photon plan with a logistic-regression NTCP model trained in
# pyanno4rt. Unlike the CNN in `ntcp_optimization_xero.py`, the model does not see dose
# volumes: it consumes eight scalar features -- the mean dose in each parotid plus six
# dose-independent geometry descriptors. Only the coefficients of the regression are
# read from the model, so value and gradient are analytic.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/ntcp_optimization_xero_lr.py

# %%
# Import necessary libraries
import logging

import matplotlib.pyplot as plt
import numpy as np
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
    OutcomeLR,
    SquaredDeviation,
    SquaredOverdosing,
)

settings.xp.prefer_gpu = True
settings.xp.preferred_cpu_array_backend = "numpy"
settings.xp.preferred_gpu_array_backend = "cupy"
logging.basicConfig(level=logging.INFO)

# The matRad head-and-neck phantom and the pyanno4rt snapshot of the trained model.
HN_PHANTOM = r"path/to/matRad/phantoms/HEAD_AND_NECK.mat"
PYANNO4RT_MODEL = r"path/to/pyanno4rt/Head-and-neck-linear-lr"

# %%
ct, cst = load_patient(HN_PHANTOM)

cst.vois[0] = cst.vois[12]  # PAROTID_LT
cst.vois[1] = cst.vois[13]  # PAROTID_RT
cst.vois[2] = cst.vois[14]  # PTV63
cst.vois[3] = cst.vois[15]  # PTV70
cst.vois = cst.vois[:4]
cst.create_body_seg()  # adds BODY/SKIN as vois[4]

# %% [markdown]
# Create the photon plan, mirroring the reference pyanno4rt run. Objective doses are
# given for the whole course (30 fractions), like in pyanno4rt.
# %%
pln = PhotonPlan(
    radiation_mode="photons",
    machine="Generic",
    num_of_fractions=30,
    dose_convention="total",
)
pln.prop_opt = {"solver": "scipy", "max_iter": 200, "display": True}
pln.prop_dose_calc = {"dose_grid": {"resolution": [5.0, 5.0, 5.0]}}
num_of_beams = 8
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
}
stf = generate_stf(ct, cst, pln)

# %% [markdown]
# Configure the NTCP objective. `model` points at the pyanno4rt snapshot folder: its
# `input.json` holds the feature definitions, `lrNTCP/` the pickled regression. Segment
# names are matched to the VOIs case-insensitively (use `segment_map` if they differ).
#
# The model was trained on total-course dose, so the objective multiplies the
# per-fraction dose of the optimizer by `num_of_fractions`.
#
# `transform=True` (off by default) keeps the gradient from vanishing once the predicted
# probability saturates: above 50% the sigmoid is replaced by its tangent. `priority` is *not* directly
# comparable to pyanno4rt's `weight`: pyanno4rt leaves the fraction factor out of the
# gradient, this objective includes it.
# %%
ntcp_obj = OutcomeLR(
    priority=100.0,
    model=PYANNO4RT_MODEL,
    transform=True,
    num_of_fractions=pln.num_of_fractions,
)
print("Model features:", ntcp_obj.feature_names)

# %%
# Conventional objectives, matching the weights of the reference plan.
cst.vois[0].objectives = [SquaredOverdosing(priority=100.0, d_max=25.0)]  # PAROTID_LT
cst.vois[1].objectives = [SquaredOverdosing(priority=100.0, d_max=25.0)]  # PAROTID_RT
cst.vois[2].objectives = [SquaredDeviation(priority=1000.0, d_ref=63.0)]  # PTV63
cst.vois[3].objectives = [SquaredDeviation(priority=1000.0, d_ref=70.0)]  # PTV70
cst.vois[4].objectives = [SquaredOverdosing(priority=800.0, d_max=30.0)]  # SKIN/BODY

# %%
# Calculate the dose influence matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# The objective only reads a mean dose per parotid, so it is attached to a helper VOI
# spanning both glands -- no global VOI needed. The VOI has to cover *both*, because the
# optimizer hands the objective only the voxels inside it and everything outside is
# zero-filled; on a single gland the other one's mean dose would silently come out as 0.
#
# `apply_overlap=None` keeps the union out of the overlap-priority pass, which would
# otherwise carve the overlapping PTVs out of the glands and shrink the feature support.
# %%
cst.vois.append(
    HelperVOI(
        name="PAROTIDS",
        ct_image=cst.ct_image,
        mask=sitk.Or(cst.vois[0].mask, cst.vois[1].mask),
        objectives=[ntcp_obj],
        apply_overlap=None,
    )
)

opt_info = {}
result = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info)

ntcp_history_active = opt_info["objective_info"][0]["history"]

# %% [markdown]
# Reference run: the same problem with the NTCP objective in monitor-only mode, so it
# reports the probability without contributing gradients. The objective is the same
# instance the helper VOI already holds, so only the flag changes.
# %%
ntcp_obj.monitor_only = True

opt_info_native = {}
result_native = fluence_optimization(ct, cst, stf, dij, pln, opt_info=opt_info_native)
ntcp_history_native = opt_info_native["objective_info"][0]["history"]

print(f"NTCP model-guided: {ntcp_history_active[-1]:.2%}")
print(f"NTCP native:       {ntcp_history_native[-1]:.2%}")

# %%
# The results hold the total-course dose on the CT grid


def mean_dose(dose_image, voi):
    """Mean total-course dose inside a VOI."""
    dose = sitk.GetArrayViewFromImage(dose_image)
    return float(dose[sitk.GetArrayViewFromImage(voi.mask) > 0].mean())


for voi in cst.vois[:4]:
    print(
        f"Mean dose {voi.name:<12} model-guided: {mean_dose(result.physical_dose.values, voi):6.2f} Gy"
        f"   native: {mean_dose(result_native.physical_dose.values, voi):6.2f} Gy"
    )

result_collective = result.model_copy()
result_collective.add_quantity("physical_dose_native", result_native.physical_dose)
result_collective.add_quantity(
    "physical_dose_diff", result.physical_dose.values - result_native.physical_dose.values
)

# %% [markdown]
# NTCP history: model-guided (active gradients) vs. native (monitor-only) optimization.
# %%
fig, ax = plt.subplots()
ax.plot(range(1, len(ntcp_history_active) + 1), ntcp_history_active, label="LR-guided")
ax.plot(range(1, len(ntcp_history_native) + 1), ntcp_history_native, label="native (monitor)")
ax.set_xlabel("Objective evaluation")
ax.set_ylabel("NTCP")
ax.set_title("Xerostomia NTCP during optimization")
ax.legend()
plt.show()

launch_viewer(ct, cst, result_collective)

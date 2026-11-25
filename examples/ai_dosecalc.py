# %% [markdown]
"""# Example for AI-based proton dose calculation."""
# %% [markdown]
# This example demonstrates how to compute dose with a trained AI model instead of an
# analytical pencil-beam kernel.

# The `AIBeamlet` engine predicts the dose cuboid of every beamlet with a neural network
# loaded through `pyRadPlan.ai.modelhub`. It plugs into the same pipeline as the analytical
# engines, so only the engine key and the model id change.

# **Scope and limitations.** AI dose engines in pyRadPlan are a research feature and are
# **not intended for clinical use**. A model is only valid for the anatomy, radiation mode and
# machine it was trained on; applying it to a different site, beam arrangement or machine gives
# no guarantee of accuracy, and no model uncertainty is validated here. Verify predicted dose
# against an established engine before drawing conclusions from it.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/ai_dosecalc.py

# %% [markdown]
# ## Prerequisites
#
# 1. A `torch` build matching your platform/CUDA (the Hub client itself ships with pyRadPlan).
# 2. The model repository below, either downloaded already or reachable on the HuggingFace Hub.
#    Models are addressed as `<org>/<model>`; `pyRadPlan.ai.modelhub.list_local_models()` shows
#    what is on disk.
# 3. Loading a model runs Python shipped inside the repository. A model folder you placed
#    yourself runs as-is; a downloaded one only once you trust it, with
#    `pyRadPlan.ai.modelhub.trust_model("<org>/<model>")` (done below) or for every download
#    with `settings.ai.modelhub_trust_remote_code = True`
#    (`PYRADPLAN_AI_MODELHUB_TRUST_REMOTE_CODE=1`).

# %%
# Import necessary libraries
import logging

import numpy as np

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    load_tg119,
    plot_slice,
    settings,
)
from pyRadPlan.ai.modelhub import trust_model
from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# The engine's arrays and the model share a device, so torch avoids conversions per beamlet.
settings.xp.prefer_gpu = True
settings.xp.preferred_gpu_array_backend = "torch"

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan with two opposing beams. The AI engine is
# selected like any other, with the model id as an additional parameter; the model must match
# the radiation mode and the anatomy it was trained on, and is downloaded on first use.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}  # better use IPOPT from ipyopt!
pln.prop_stf = {
    "gantry_angles": [90, 270],
    "couch_angles": [0, 0],
}
pln.prop_dose_calc = {
    "dose_grid": {"resolution": [2.0, 2.0, 2.0]},
    "engine": "AIBeamlet",
    "model": "DKFZ-RadOpt/pyRadPlan-dosecalc-Det-proton-lung",
}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# %% [markdown]
# The engine runs the code shipped with the model. `trust_model` downloads the model without
# running anything and records a hash of its code; inspect that code before you trust it. If the
# code changes upstream later, the engine refuses the model until you trust it again.
# %%
model_dir, _ = trust_model("DKFZ-RadOpt/pyRadPlan-dosecalc-Det-proton-lung")
print(f"Trusted the model code in {model_dir}")

# %% [markdown]
# ## Dose calculation with the AI engine
# %%
dij = calc_dose_influence(ct, cst, stf, pln)

# Which quantities the model produces is declared by the model itself
print(dij.quantities)

# %% [markdown]
# ## Optimization
#
# From here on nothing is specific to the AI engine: the dose influence matrix is used like
# any other.
# %%
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

# Optimize the fluence; the result holds the dose on the CT grid
result = fluence_optimization(ct, cst, stf, dij, pln)

# %% [markdown]
# Visualize the results
# %%

if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[2] / 2))
    plot_slice(ct, cst, result.physical_dose, view_slice)

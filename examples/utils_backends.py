# %% [markdown]
"""# Example for interactions with the array backends."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to use and interact
# with the underlying array api backends. These are povided by the 'xp_utils' module.

# If you find bugs or that your desired backend is not working properly,
# feel free to open an issue and reach out to us on github.
# Improvements are also very welcome :)

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/utils_backends.py
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
    settings,
    xp_utils,
)

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

# %% [markdown]
# ## 1. Exploring and Setting Backends
# The pyRadPlan Array API backend (`xp_utils`) allows using multiple computational
# backends seamlessly.
# Natively, every library which complies to the array API standard should be supported.
# In pyRadPlan however, we recommend using either NumPy (out of the box) or CuPy / PyTorch (GPU) for low grid sizes, especially for electrons.

# On the main webpage for cupy and pytorch you can find instructions on how to install it for your system.

# The following lines of code check your system for available backends.
# %%
print("--- Available Computational Backends ---")
print("NumPy available: True (Default CPU)")
print(
    f"PyTorch available: {xp_utils.pytorch_available()} (GPU: {xp_utils.pytorch_gpu_available()})"
)
print(f"CuPy available: {xp_utils.cupy_available()}")
print(f"JAX available: {xp_utils.jax_available()} (GPU: {xp_utils.jax_gpu_available()})")

# The preferred backends are the "xp" sub-configuration of the global pyRadPlan
# settings. They can be configured via environment variables / a `.env` file
# (PYRADPLAN_XP_PREFER_GPU, PYRADPLAN_XP_PREFERRED_CPU_ARRAY_BACKEND,
# PYRADPLAN_XP_PREFERRED_GPU_ARRAY_BACKEND) or at runtime as below.

# Let's start by calculating the dose influence matrix (dij) on the CPU
print("\nConfiguring CPU Backend (NumPy) for Dose Calculation...")
settings.xp.prefer_gpu = False
settings.xp.preferred_cpu_array_backend = "numpy"
logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_opt = {"solver": "scipy"}
pln.prop_dose_calc = {"dose_grid": {"resolution": (2.0, 2.0, 2.0)}}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimization
cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

# %% [markdown]
# ## 2. Changing the Backend before Optimization
# If you wanted to switch to a GPU backend mid-code you can do so.

# Here we switch the backends before optimization:
# %%
print("\n--- Switching to a GPU Backend for Optimization ---")
settings.xp.prefer_gpu = True

# `preferred_gpu_array_backend = None` selects the best available GPU backend automatically
print(f"Accelerated GPU backend selected: {xp_utils.choose_array_api_namespace().__name__}")

# or you can force a specific backend:
# settings.xp.preferred_gpu_array_backend = "torch"  # or "cupy"

# Calculate optimized fluence
fluence = fluence_optimization(ct, cst, stf, dij, pln)

# Compute the result on the CT grid
result = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# Visualize the results
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[2] / 2))

    # Visualize
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result["physical_dose"],
        view_slice=view_slice,
        plane="axial",
        overlay_unit="Gy",
    )

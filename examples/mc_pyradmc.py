# %% [markdown]
"""# Example for photon dose calculation using the pyRadMC Monte Carlo engine."""
# %% [markdown]
# This example demonstrates photon dose calculation with pyRadMC, a GPU-accelerated
# (NVIDIA warp) photon/electron Monte Carlo engine for beamlet-resolved dose influence
# calculation.
#
# pyRadMC is an optional dependency. Until it is published, install it locally in
# editable mode:
#
# ```bash
# pip install -e path/to/pyRadMC[warp]
# ```
#
# Notes:
# - Dij columns are scaled by a placeholder monitor-unit calibration
#   (`num_particles_per_mu`); a machine-specific output calibration is not modelled yet.
# - The primary photon spectrum and the beam-limiting device geometry (jaws/MLC) come
#   from the machine's Monte Carlo beam model (`PhotonLINAC.mc_beam_model`; the
#   'Generic' machine defaults to a Siemens-PRIMUS-like description with an analytic
#   Ali & Rogers 6 MV spectrum). `spectrum_beam` overrides the spectrum by name.
# - Dij calculation uses the plain divergent beamlet source (no jaw/MLC model).
#   Forward dose can alternatively run through the treatment-head pre-solve
#   (`forward_mode='head_presolve'`), which opens the machine's jaws/MLC conformally
#   to the stf rays and adds collimator transmission and scatter, but ignores
#   per-bixel fluence modulation.
# - The default `cross_section_model` 'tabulated' downloads the EPICS data libraries
#   (~120 MB) into `~/.cache/pyRadMC/epics` on first use.

# %%
# Import necessary libraries
import logging
import numpy as np

from pyRadPlan import (
    PhotonPlan,
    generate_stf,
    calc_dose_influence,
    load_tg119,
    fluence_optimization,
    plot_slice,
)

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %%
pln = PhotonPlan(radiation_mode="photons", machine="Generic")
pln.prop_stf = {
    "gantry_angles": [0, 90, 180, 270],
    "couch_angles": [0, 0, 0, 0],
    "bixel_width": 10.0,
}
pln.prop_dose_calc = {
    "engine": "pyRadMC",
    "backend": "warp",  # 'warp' (GPU if available, else CPU) or 'ref' (slow CPU oracle)
    "num_histories_per_beamlet": 1e4,  # increase for smoother dose
    "seed": 20260715,
    "dose_grid": {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}},
    # "spectrum_beam": "varian-6mv",  # named-spectrum override of the machine model
    # "forward_mode": "head_presolve",  # forward dose via the treatment-head pre-solve
}

pln.prop_opt = {"solver": "scipy"}

# %% [markdown]
# Generate Steering Geometry ("stf")
# %%
stf = generate_stf(ct, cst, pln)

# %% [markdown]
# Run calc_dose_influence (dose influence matrix)
# %%
dij = calc_dose_influence(ct, cst, stf, pln)

# %% [markdown]
# Define cst objectives and run fluence optimization
# %%
cst.vois[0].objectives = [SquaredOverdosing(priority=100.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=1000.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

fluence = fluence_optimization(ct, cst, stf, dij, pln)

# %% [markdown]
# Compute dose distribution
# %%
result_opt = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# Visualize the results
# %%
if GUI_AVAILABLE:
    launch_viewer(ct, cst, result_opt)
else:
    view_slice = int(np.round(ct.size[2] / 2))
    plot_slice(ct, cst, result_opt["physical_dose"], view_slice)

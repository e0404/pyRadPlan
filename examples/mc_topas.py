# %% [markdown]
"""# Example for proton dose recalculation using the TOPAS engine."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to recalculate an optimized
# proton plan with TOPAS.
#
# For installation instructions, please refer to
# https://opentopas.readthedocs.io/en/latest/index.html
#
# TOPAS is not driven by pyRadPlan directly. The workflow is therefore split into two
# passes over ``calc_dose_forward``:
#
# 1. ``external_calculation=True`` writes the TOPAS input files into ``simu_dir`` and
#    returns a *zero* dose cube. You then run TOPAS on those files yourself.
# 2. ``external_calculation="<folder with the TOPAS output>"`` reads the scored
#    ``*.bin`` cubes back in and assembles the dose.
#
# Both passes must use the *same* weights, ``num_histories_direct`` and ``num_runs``,
# because the normalization of the scored cubes depends on all three.
#
# To display this script in a Jupyter Notebook, you need to install jupytext via pip and
# run the following command. This will create a .ipynb file in the same directory:
#
# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/mc_topas.py
# ```

# %%
# Import necessary libraries
import logging
from pathlib import Path

import numpy as np

from pyRadPlan import (
    IonPlan,
    generate_stf,
    calc_dose_influence,
    calc_dose_forward,
    load_tg119,
    fluence_optimization,
    plot_slice,
)
from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE

logging.basicConfig(level=logging.INFO)

# Folder that holds the TOPAS input files and, later, the TOPAS output.
topas_dir = Path("topas_recalc").resolve()

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a proton therapy plan using the ParticleHongPencilBeamEngine.
# %%
# Create a plan object
pln = IonPlan(radiation_mode="protons", machine="Generic")
pln.prop_stf = {
    "gantry_angles": [90, 180],
    "couch_angles": [0, 0],
}
pln.prop_opt = {"solver": "scipy"}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

dij = calc_dose_influence(ct, cst, stf, pln)

# Objectives are taken from the TG119 cst
fluence = fluence_optimization(ct, cst, stf, dij, pln)

result_pb = dij.compute_result_ct_grid(fluence)

# %% [markdown]
# ## TOPAS recalculation, pass 1: write the input files
#
# The returned dose is all zeros here; only the input files matter. Run TOPAS on every
# ``pyRadPlan_plan_field*_run*.txt`` in ``topas_dir`` and keep the scored
# ``score_*_physicalDose.bin`` files in that same folder.
# %%
topas_settings = {
    "engine": "TOPAS",  # necessary for the following to take effect
    "num_histories_direct": 1e7,
    "num_runs": 5,  # statistically independent runs per beam, used for the variance
    "simu_dir": topas_dir,
}

pln.prop_dose_calc = {**topas_settings, "external_calculation": True}
calc_dose_forward(ct, cst, stf, pln, fluence)

# %% [markdown]
# ## TOPAS recalculation, pass 2: read the results back
#
# This only works once TOPAS has actually run. Note that pass 1 overwrote the beamlet
# weights in ``stf`` with history counts, so ``fluence`` has to be passed again to get
# the scaling right.
# %%
topas_output_available = any(topas_dir.glob("score_*_physicalDose.bin"))

if topas_output_available:
    pln.prop_dose_calc = {**topas_settings, "external_calculation": str(topas_dir)}
    result_mc = calc_dose_forward(ct, cst, stf, pln, fluence)
else:
    result_mc = None
    logging.warning("No TOPAS output found in %s - run TOPAS first.", topas_dir)

# %% [markdown]
# ## Visualize both dose distributions
#
# The viewer takes a dict of named quantities and lets you switch between them. Keeping
# the ``physical_dose`` prefix makes it pick up the correct label and unit.
# %%
result = {"physical_dose (pencil beam)": result_pb["physical_dose"]}
if result_mc is not None:
    result["physical_dose (TOPAS)"] = result_mc["physical_dose"]
    result["physical_dose_var (TOPAS)"] = result_mc["physical_dose_var"]

if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[2] / 2))
    for quantity in result.values():
        plot_slice(ct, cst, quantity, view_slice, overlay_unit="Gy")

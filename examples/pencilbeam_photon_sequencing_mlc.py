# %% [markdown]
"""# Example for photon sequencing with a physical MLC restriction."""
# %% [markdown]
# This example extends `pencilbeam_photon_sequencing.py` by equipping the machine
# with a physical multi-leaf collimator (MLC): a fixed leaf width and leaf count.
# The photon sequencer then computes shapes on the machine's real leaf grid
# instead of directly on the (finer) dose-calculation bixel grid, resampling the
# fluence maps onto the leaf grid wherever they differ.
#
# To display this script in a Jupyter Notebook, install jupytext and run:
# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_photon_sequencing_mlc.py
# ```

# %%
# Import necessary libraries

import logging

import numpy as np
from copy import deepcopy

from pyRadPlan import (
    PhotonPlan,
    load_tg119,
    generate_stf,
    calc_dose_influence,
    calc_dose_forward,
    fluence_optimization,
    plot_aperture_info,
    plot_dose_comparison,
    run_sequencing,
)
from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.machines import MLC, load_from_name
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# Load the "Generic" photon machine and equip it with a physical MLC: 10 mm
# leaf width and 40 leaf pairs. The dose-calc bixel width used below (5 mm) is
# finer than this leaf width, so the sequencer will resample the fluence maps
# onto the coarser leaf grid before computing shapes.
# %%
machine = load_from_name("photons", "Generic")
machine.mlc = MLC(device_type="MLCX", leaf_width=10.0, number_of_elements=40)

# %%
# 1) Create a plan object, using the MLC-equipped machine
pln = PhotonPlan()
pln.machine = machine

# 2) Define the parameters for the steering information
num_of_beams = 5
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
}

# 3) Specify the settings for dose calculation
pln.prop_dose_calc = {
    "dose_grid": {"resolution": [5.0, 5.0, 5.0]},
    "enable_dij_sampling": False,
    "force_uniform_fluence": True,
}

# 4) Configure optimization parameters
pln.prop_opt = {"solver": "scipy"}

# 5) Configure sequencing
# Note: num_of_mlc_leaf_pairs and the leaf grid are now taken from
# pln.machine.mlc instead of a sequencer default.
pln.prop_seq = {
    "run_sequencing": True,
    "num_of_levels": 5,
    "sequencer": "xia",  # xia/siochi
    "mode": "reducing_levels",
}

# %% [markdown]
# Generate stf, dij, and run fluence optimization exactly as in the base example.
# %%
stf = generate_stf(ct, cst, pln)
dij = calc_dose_influence(ct, cst, stf, pln)

cst.vois[0].objectives = [SquaredOverdosing(priority=100.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=1000.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

result = fluence_optimization(ct, cst, stf, dij, pln)

# %% [markdown]
# Run sequencing. Each shape's MLC now reports the machine's true 10 mm leaf
# width, and `full_leaf_positions` gives the full-device view (all 40 leaf
# pairs), with leaves outside the treated field closed at the center (0, 0).
# %%
result_sequenced = run_sequencing(deepcopy(result), pln, stf, dij=dij)

first_beam = result_sequenced.sequence.beam[0]
first_mlc = first_beam.mlc_sequence[0]
print(f"Leaf width used for sequencing: {first_mlc.leaf_width} mm")
print(f"Number of physical leaf pairs: {first_mlc.number_of_elements}")

full_positions = first_mlc.full_leaf_positions(first_beam.central_leaf_pair)
print(f"Full-device leaf positions shape: {full_positions.shape}")
print(f"Closed (unused) leaf pairs: {np.sum(np.all(full_positions == 0, axis=1))}")

# %% [markdown]
# `launch_viewer` opens the interactive Qt result viewer on the sequenced
# dose. Because `result_sequenced.sequence` is populated, its "Show
# Sequence" button is enabled: it opens a second, non-modal window with one
# tab per beam, each stepping through that beam's delivered segments with
# play/pause/speed/loop controls (dwell time proportional to each segment's
# relative weight).
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result_sequenced)

# %% [markdown]
# `plot_aperture_info` gives the full per-beam gallery: one subplot per
# delivered segment, shaded by that segment's relative weight (light grey =
# low, blue = high). Passing `unsequenced_result` also adds a fluence
# comparison row (the pre-sequencing target fluence next to what the
# sequenced shapes actually deliver) and, when `image_volume`/`cst` are also
# given, a per-beam dose comparison row (optimized vs. delivered vs.
# difference, for that beam's own dose contribution). One figure is opened
# per beam.
# %%
view_slice = int(np.round(ct.size[2] / 2))
plot_aperture_info(
    result_sequenced,
    stf=stf,
    unsequenced_result=result,
    image_volume=ct,
    cst=cst,
    view_slice=view_slice,
    show_plot=True,
)

# %% [markdown]
# `plot_dose_comparison` shows the same before/after story for the total
# (all-beams) dose: the optimized target dose, the dose the sequenced
# shapes/MLC positions actually deliver, and their difference.
# %%
plot_dose_comparison(
    result,
    result_sequenced,
    title_before="Optimized",
    title_after="Sequenced",
    quantity_before="physical_dose",
    quantity_after="physical_dose_sequenced",
    image_volume=ct,
    cst=cst,
    view_slice=view_slice,
)


# %% [markdown]
# ## Forward-planning verification
#
# To verify the sequencing result independently of the optimizer/dij path,
# re-deliver it through forward, field-based dose calculation.
# %%
pln_forward = PhotonPlan()
pln_forward.machine = machine
pln_forward.prop_stf = {
    "gantry_angles": pln.prop_stf["gantry_angles"],
    "couch_angles": pln.prop_stf["couch_angles"],
    "generator": "photonSingleBixel",
    "beam_sequences": result_sequenced.sequence.beam,
    "resolution": 0.5,
    "energy": 6.0,
}
pln_forward.prop_dose_calc = {
    "dose_grid": {"resolution": [5.0, 5.0, 5.0]},
    "enable_dij_sampling": False,
    "force_uniform_fluence": True,
}

stf_forward = generate_stf(ct, cst, pln_forward)
result_forward = calc_dose_forward(ct, cst, stf_forward, pln_forward)

# %% [markdown]
# Compare the dose from this independent forward/field-based recomputation
# against the dose already stored on `result_sequenced` (computed via the
# dij/fluence path during sequencing). They should closely agree.
# %%
plot_dose_comparison(
    result_sequenced,
    result_forward,
    title_before="Sequenced (dij)",
    title_after="Forward (field-based)",
    quantity_before="physical_dose_sequenced",
    quantity_after="physical_dose",
    beam_indices="all",
    stf=stf_forward,
    image_volume=ct,
    cst=cst,
)

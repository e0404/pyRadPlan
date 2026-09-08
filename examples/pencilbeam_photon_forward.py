# %% [markdown]
"""# Example for photon forward dose calculation with configured beam-limiting devices."""
# %% [markdown]
# This example demonstrates a field-based forward photon dose calculation:
#
# 1. The available beam-limiting devices (BLDs) are obtained from the machine's
#    Monte Carlo beam model (`PhotonLINAC.mc_beam_model`) - for the 'Generic'
#    (Siemens-PRIMUS-like) machine, that is an X-travelling MLC and a Y jaw pair.
# 2. The devices are then *configured* (jaw positions and MLC leaf positions).
#    Note that all positions - leaf boundaries/widths as well as x/y openings -
#    are always defined **at the isocenter plane** (IEC coordinates); projection
#    to the physical device planes is handled by the engines.
# 3. Forward dose is calculated with the pencil-beam engine, and - optionally,
#    if pyRadMC is installed - with the pyRadMC Monte Carlo engine through the
#    treatment-head pre-solve, which models the actual jaw/MLC transmission and
#    collimator scatter from the same configured devices.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip
# and run the following command (creates an .ipynb file in the same directory):

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_photon_forward.py

# %%
# Import necessary libraries
import logging
from importlib.util import find_spec

import numpy as np

import matplotlib.pyplot as plt

from pyRadPlan import (
    PhotonPlan,
    generate_stf,
    calc_dose_forward,
    plot_slice,
    load_tg119,
    settings,
)

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.machines import create_bld, load_from_name
from pyRadPlan.stf import FieldShapeAsBLD, FieldShapeComposite

from pyRadPlan.machines.photons._calculate_machine_scale import calculate_machine_scale

logging.basicConfig(level=logging.INFO)

# The field-based pencil-beam path currently requires the numpy backend
settings.xp.prefer_gpu = False

# %% [markdown]
# ## Obtain the available beam-limiting devices from the machine
#
# The machine's MC beam model describes the physical devices: leaf boundaries
# (at the isocenter plane), travel axis, and the jaw pairs. We derive the
# configurable BLD objects from it instead of hardcoding the geometry.

# %%
machine = load_from_name("photons", "Generic")
beam_model = machine.mc_beam_model

print(f"Machine '{machine.name}' provides:")
for jaw_geometry in beam_model.jaws:
    print(f"  - {jaw_geometry.axis.upper()} jaw pair (focused: {jaw_geometry.focused})")
if beam_model.mlc is not None:
    print(
        f"  - MLC{beam_model.mlc.travel_axis.upper()} with "
        f"{beam_model.mlc.num_leaf_pairs} leaf pairs ({beam_model.mlc.leaf_end_design})"
    )

# The pyRadPlan BLD transmission model requires uniform leaf widths, so we
# configure the central uniform section of the machine's MLC (the PRIMUS has
# 27 central 10 mm leaves flanked by one wide 65 mm leaf per side).
machine_edges = beam_model.mlc.leaf_edges  # mm at the isocenter plane
widths = np.diff(machine_edges)
uniform = np.isclose(widths, np.median(widths))
central_edges = machine_edges[:-1][uniform]  # leaf start boundaries
leaf_width = float(np.median(widths))

# Configure an L-shaped opening on 10 central leaves (positions in mm at iso)
n_configured = 10
first = int(np.searchsorted(central_edges, -n_configured / 2 * leaf_width))
positions = np.zeros((len(central_edges), 2))  # all other leaves closed
positions[first : first + n_configured] = [
    [0, 0],  # closed
    [-20, 20],
    [-20, 10],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [0, 0],  # closed
]

mlc = create_bld(
    {
        "device_type": "MLC",
        "device_orientation": beam_model.mlc.travel_axis.upper(),
        "leaf_position_boundaries": central_edges,
        "leaf_positions": positions,
        "leaf_width": leaf_width,
        "leaf_leakage": 0.1,
    }
)

# Configure the machine's jaw pair (openings at the isocenter plane)
jaw = create_bld(
    {
        "device_type": "JAW",
        "device_orientation": beam_model.jaws[0].axis.upper(),
        "positions": [-20, 10],
        "field_width": 2 * np.abs(central_edges).max() + leaf_width,
        "leakage": 0.1,
    }
)

# %% [markdown]
# Visualize the transmission masks of the configured devices (IEC coordinates).
# %%
resolution = 1.0
mask_mlc = mlc.calculate_transmission_mask(resolution)
mask_jaw = jaw.calculate_transmission_mask(resolution)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
half_size = (mask_mlc.shape[0] - 1) / 2 * resolution
extent = [-half_size, half_size, -half_size, half_size]
axes[0].imshow(mask_jaw, cmap="gray", origin="lower", extent=extent)
axes[0].set_title("Jaw Mask (IEC DICOM)")
axes[0].set_xlabel("X (mm)")
axes[0].set_ylabel("Y (mm)")
axes[1].imshow(mask_mlc, cmap="gray", origin="lower", extent=extent)
axes[1].set_title("MLC Mask (IEC DICOM)")
axes[1].set_xlabel("X (mm)")
axes[1].set_ylabel("Y (mm)")
plt.tight_layout()
plt.show()

# %%
# pyRadPlan internally defines "field shapes" to represent the shape of the beamlets.
mlc_shape = FieldShapeAsBLD(energy=6.0, bld=mlc, resolution=resolution)
jaw_shape = FieldShapeAsBLD(energy=6.0, bld=jaw, resolution=resolution)
combined_shape = FieldShapeComposite(energy=6.0, shapes=[mlc_shape, jaw_shape])

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(combined_shape.mask, cmap="gray", origin="lower", extent=extent)
ax.set_title("Combined Field Shape (LPS BEV)")
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
plt.tight_layout()
plt.show()

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# ## Forward dose with the pencil-beam engine
#
# The plan uses the `PhotonPencilBeamSVDEngine`.
# %%
# Create a plan object
pln = PhotonPlan(machine="Generic")
num_of_beams = 1
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
    "generator": "photonSingleBixel",
    "field_based": True,
    "blds": [mlc, jaw],
    "resolution": 0.5,
    "energy": 6,
}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Machine Calibration: Calculate the conversion factor between MU and weights
ct_dim = [64, 64, 64]
ct_resolution = [2.0, 2.0, 2.0]
num_of_ct_scen = 1

weights = calculate_machine_scale("Generic", ct_dim, ct_resolution, num_of_ct_scen)

# Apply weights to the rays in the stf
for beam in stf.beams:
    for ray in beam.rays:
        for beamlet in ray.beamlets:
            beamlet.weight *= weights

# Forward dose calculation, which directly returns the result on the CT grid
result_pb = calc_dose_forward(ct, cst, stf, pln)

# %% [markdown]
# ## Optional: forward dose with the pyRadMC Monte Carlo engine
#
# If pyRadMC is installed, the same configured devices are used to build the
# actual treatment-head geometry: the engine reads the jaw and MLC settings
# from the stf field shapes, projects them from the isocenter plane onto the
# machine's device planes, and runs the treatment-head pre-solve (collimator
# transmission and scatter) before transporting through the patient.
#
# Notes:
#
# - The absolute dose scaling of the MC path relies on the placeholder MU
#   calibration (`num_particles_per_mu`), so compare distributions, not
#   absolute values.
# - The two engines currently disagree on the *orientation* of BLD-based
#   field shapes: the MC path maps the devices IEC-literally
#   (BEV x = -IEC X, BEV z = +IEC Y, i.e. an MLCX travels along the patient
#   left-right axis), while the pencil-beam path applies the field mask
#   transposed (`np.rot90` in the SVDPB engine, marked TODO there). Until
#   this is adjudicated, asymmetric fields render mirrored/transposed
#   between the two engines.
# %%
result_mc = None
if find_spec("pyradmc") is not None:
    pln_mc = PhotonPlan(machine="Generic")
    pln_mc.prop_stf = pln.prop_stf
    pln_mc.prop_dose_calc = {
        "engine": "pyRadMC",
        "backend": "warp",  # GPU if available, else CPU
        "forward_mode": "head_presolve",  # jaw/MLC transmission + collimator scatter
        "num_histories_direct": 5e6,
        "seed": 20260716,
        "dose_grid": {"resolution": {"x": 3.0, "y": 3.0, "z": 3.0}},
    }
    result_mc = calc_dose_forward(ct, cst, stf, pln_mc)
else:
    print("pyRadMC not installed - skipping the Monte Carlo forward calculation.")

# %% [markdown]
# Visualize the results
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result_mc if result_mc is not None else result_pb)
else:
    view_slice = int(np.round(ct.size[1] / 2))
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result_pb["physical_dose"],
        view_slice=view_slice,
        plane="coronal",
        overlay_unit="Gy",
    )
    if result_mc is not None:
        plot_slice(
            image_volume=ct,
            cst=cst,
            overlay=result_mc["physical_dose"],
            view_slice=view_slice,
            plane="coronal",
            overlay_unit="Gy",
        )

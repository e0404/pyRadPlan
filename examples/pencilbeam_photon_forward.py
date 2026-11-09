# %% [markdown]
"""# Example for forward photon dose calculation of a field-based plan."""
# %% [markdown]
# This example demonstrates how to use the pyRadPlan library to define beam limiting devices
# (MLC and jaws), turn them into field shapes, and run a forward photon dose calculation. The
# single beam delivers a short step-and-shoot sequence: three MLC-defined segments with
# decreasing weight, each combined with the same static jaw.

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/pencilbeam_photon_forward.py

# %%
# Import necessary libraries
import logging

import numpy as np

import matplotlib.pyplot as plt

from pyRadPlan import (
    PhotonPlan,
    generate_stf,
    calc_dose_forward,
    plot_slice,
    load_tg119,
)

from pyRadPlan.gui import launch_viewer, GUI_AVAILABLE
from pyRadPlan.machines import create_bld
from pyRadPlan.stf import FieldShapeAsBLD, FieldShapeComposite
from pyRadPlan.result._beam_sequence import BeamSequence

from pyRadPlan.machines.photons._calculate_machine_scale import calculate_machine_scale

logging.basicConfig(level=logging.INFO)


resolution = 1.0
leaf_width = 5

# A short step-and-shoot sequence for this beam: the L-shape above, then two
# progressively smaller segments (leaves closing in further each time).
positions_shape_1 = [  # L-form for testing direction
    [0, 0],
    [-20, 20],
    [-20, 10],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [0, 0],
]
positions_shape_2 = [
    [0, 0],
    [-20, 15],
    [-20, 5],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-20, 0],
    [-15, 0],
    [-10, 0],
    [0, 0],
]
positions_shape_3 = [
    [0, 0],
    [-15, 10],
    [-15, 0],
    [-10, 0],
    [-10, 0],
    [-10, 0],
    [-5, 0],
    [0, 0],
    [0, 0],
    [0, 0],
]
number_of_elements = len(positions_shape_1)
boundaries = np.arange(
    -int(number_of_elements / 2) * leaf_width, int(number_of_elements / 2) * leaf_width, leaf_width
)

mlc_information = {
    "device_type": "MLC",
    "device_orientation": "X",
    "leaf_position_boundaries": boundaries,
    "leaf_positions": positions_shape_1,
    "leaf_width": leaf_width,
    "leaf_leakage": 0.1,
}
mlc_shape_1 = create_bld(mlc_information)

mlc_shapes = [
    mlc_shape_1,
    create_bld({**mlc_information, "leaf_positions": positions_shape_2}),
    create_bld({**mlc_information, "leaf_positions": positions_shape_3}),
]
shapes_weight = [1.0, 0.6, 0.3]

jaw_info = {
    "device_type": "JAW",
    "device_orientation": "X",
    "positions": [-20, 10],
    "field_width": 50,
    "leakage": 0.1,
}
jaw_x = create_bld(jaw_info)

# %%
# pyRadPlan internally defines "field shapes" to represent the shape of the beamlets.
# Each segment of the sequence gets its own MLC field shape, composed with the
# same jaw field shape.
jaw_shape = FieldShapeAsBLD(energy=6.0, bld=jaw_x, resolution=resolution)
mlc_field_shapes = [
    FieldShapeAsBLD(energy=6.0, bld=mlc, resolution=resolution) for mlc in mlc_shapes
]
combined_shapes = [
    FieldShapeComposite(energy=6.0, shapes=[mlc_shape, jaw_shape])
    for mlc_shape in mlc_field_shapes
]

# %%
# Plot masks of shape 1 in IEC and LPS BEV view
mask_mlc_shape_1 = mlc_shape_1.calculate_transmission_mask(resolution)
mask_jaw_x = jaw_x.calculate_transmission_mask(resolution)
mask_shape_1 = mask_mlc_shape_1 * mask_jaw_x

half_size = jaw_info["field_width"]
extent = [-half_size, half_size, -half_size, half_size]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(mask_jaw_x, cmap="viridis", origin="lower", extent=extent)
axes[0, 0].set_title("Jaw Mask (IEC DICOM)")
axes[0, 0].set_xlabel("X (mm)")
axes[0, 0].set_ylabel("Y (mm)")

axes[0, 1].imshow(mask_mlc_shape_1, cmap="viridis", origin="lower", extent=extent)
axes[0, 1].set_title("MLC Mask (IEC DICOM)")
axes[0, 1].set_xlabel("X (mm)")
axes[0, 1].set_ylabel("Y (mm)")

axes[0, 2].imshow(mask_shape_1, cmap="viridis", origin="lower", extent=extent)
axes[0, 2].set_title("Combined Mask (IEC DICOM)")
axes[0, 2].set_xlabel("X (mm)")
axes[0, 2].set_ylabel("Y (mm)")

axes[1, 0].imshow(jaw_shape.mask, cmap="viridis", origin="lower", extent=extent)
axes[1, 0].set_title("Jaw Field Shape (LPS BEV)")
axes[1, 0].set_xlabel("X (mm)")
axes[1, 0].set_ylabel("Y (mm)")

axes[1, 1].imshow(mlc_field_shapes[0].mask, cmap="viridis", origin="lower", extent=extent)
axes[1, 1].set_title("MLC Field Shape (LPS BEV)")
axes[1, 1].set_xlabel("X (mm)")
axes[1, 1].set_ylabel("Y (mm)")

im5 = axes[1, 2].imshow(combined_shapes[0].mask, cmap="viridis", origin="lower", extent=extent)
axes[1, 2].set_title("Combined Field Shape (LPS BEV)")
axes[1, 2].set_xlabel("X (mm)")
axes[1, 2].set_ylabel("Y (mm)")

cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
fig.colorbar(im5, cax=cbar_ax, label="Transmission")
plt.show()


# Now plot the combined field shape of every segment to verify their geometry
# in LPS BEV coordinates
w_max = max(shapes_weight)
fig, axes = plt.subplots(1, len(combined_shapes), figsize=(5 * len(combined_shapes), 5))
for m, (ax, combined_shape, weight) in enumerate(zip(axes, combined_shapes, shapes_weight)):
    im = ax.imshow(
        combined_shape.mask * weight,
        cmap="viridis",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=w_max,
    )
    ax.set_title(f"Segment {m + 1} (weight={weight:.2g})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")

cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
fig.colorbar(im, cax=cbar_ax, label="Weight")
plt.show()

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# %% [markdown]
# In this section, we create a photon therapy plan using the PhotonPencilBeamSVDEngine.
# %%
# Create a plan object
pln = PhotonPlan(machine="Generic")
num_of_beams = 1

# A BeamSequence describes the field shape(s) delivered by one beam: here, the
# short step-and-shoot sequence defined above (three MLC segments, decreasing
# weight), each combined with the same static jaw.
beam_sequence = BeamSequence(
    mlc_sequence=mlc_shapes,
    shapes_weight=shapes_weight,
    extra_blds=[[jaw_x] for _ in mlc_shapes],
)

pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),  # np.array([0, 270]),
    "generator": "photonSingleBixel",
    "beam_sequences": [beam_sequence],
    "resolution": 0.5,
    "energy": 6,
}

pln.prop_dose_calc = {
    "dose_grid": {"resolution": [5.0, 5.0, 5.0]},
    "enable_dij_sampling": False,
    "force_uniform_fluence": True,
}

# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Machine Calibration: Calculate the conversion factor between MU and weights
machine = "Generic"
ct_dim = [64, 64, 64]
ct_resolution = [2.0, 2.0, 2.0]
num_of_ct_scen = 1

weights = calculate_machine_scale(machine, ct_dim, ct_resolution, num_of_ct_scen)

# Apply weights to the rays in the stf
for beam in stf.beams:
    for ray in beam.rays:
        for beamlet in ray.beamlets:
            beamlet.weight *= weights

# Forward dose calculation, which directly returns the result on the CT grid
result = calc_dose_forward(ct, cst, stf, pln)


# %% [markdown]
# Visualize the results
# %%
if GUI_AVAILABLE:
    # Use the GUI if [gui] dependencies are installed
    launch_viewer(ct, cst, result)
else:
    # Choose a slice to visualize
    view_slice = int(np.round(ct.size[1] / 2))

    # Visualize
    plot_slice(
        image_volume=ct,
        cst=cst,
        overlay=result.physical_dose,
        view_slice=view_slice,
        plane="coronal",
        overlay_unit="Gy",
    )

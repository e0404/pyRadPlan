# %% [markdown]
"""# Example for photon dose calculation with the collapsed cone engine."""
# %% [markdown]
# This example compares the collapsed cone convolution/superposition engine ("CCC") with the
# SVD pencil beam engine ("SVDPB") on the TG119 phantom. The collapsed cone engine transports
# the TERMA of each bixel with a 6 MV energy deposition kernel scaled by the electron density.
# Its transport is written with Array API operations and runs on the array backend selected in
# ``pyRadPlan.settings.xp`` (NumPy on the CPU, CuPy or PyTorch on a GPU).

# To display this script in a Jupyter Notebook, you need to install jupytext via pip and run the following command.
# This will create a .ipynb file in the same directory:

# ```bash
# pip install jupytext
# jupytext --to notebook path/to/this/file/collapsed_cone_photon.py

# %%
# Import necessary libraries
import logging

import numpy as np
import SimpleITK as sitk

from pyRadPlan import (
    PhotonPlan,
    generate_stf,
    calc_dose_forward,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    load_tg119,
)
from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing, MeanDose

logging.basicConfig(level=logging.INFO)

# %%
# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

pln = PhotonPlan(machine="Generic")
num_of_beams = 5
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
    "bixel_width": 10.0,
}
stf = generate_stf(ct, cst, pln)

# %% [markdown]
# Forward dose of all bixels with unit weight. With the beam axis tilt of the cones (default)
# the collapsed cone engine transports the weighted TERMA of each beam once. To keep the example
# short it uses a 5 mm dose grid, 10 mm bixels and 6 x 6 instead of the default 12 x 8 cones.
# %%
dose_grid = {"resolution": {"x": 5.0, "y": 5.0, "z": 5.0}}
ccc_options = {"num_polar_angles": 6, "num_azimuthal_angles": 6, "scatter_cutoff": 100.0}
doses = {}
for engine in ("SVDPB", "CCC"):
    pln.prop_dose_calc = {"engine": engine, "dose_grid": dose_grid}
    if engine == "CCC":
        pln.prop_dose_calc.update(ccc_options)
    doses[engine] = calc_dose_forward(ct, cst, stf, pln).physical_dose.values

svd = sitk.GetArrayFromImage(doses["SVDPB"])
ccc = sitk.GetArrayFromImage(doses["CCC"])
high = svd > 0.5 * svd.max()
print(f"Median CCC / SVDPB dose above 50 % of the maximum: {np.median(ccc[high] / svd[high]):.3f}")

view_slice = int(np.round(ct.size[2] / 2))
plot_slice(
    image_volume=ct,
    cst=cst,
    overlay=doses["CCC"] - doses["SVDPB"],
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

# %% [markdown]
# The engine options are set through ``pln.prop_dose_calc`` like for any other engine, e.g.
# the number of cones, the lateral transport margin and the dij sampling, as above. The dose
# influence matrix is computed bixel by bixel and can be optimized as usual; here with only
# 4 x 4 cones, which keeps the example fast.
# %%
pln.prop_dose_calc = {
    "engine": "CCC",
    "dose_grid": dose_grid,
    **ccc_options,
    "num_polar_angles": 4,
    "num_azimuthal_angles": 4,
}
dij = calc_dose_influence(ct, cst, stf, pln)

cst.vois[0].objectives = [SquaredOverdosing(priority=10.0, d_max=1.0)]  # OAR
cst.vois[1].objectives = [SquaredDeviation(priority=100.0, d_ref=3.0)]  # Target
cst.vois[2].objectives = [
    MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(priority=10.0, d_max=2.0),
]  # BODY

result = fluence_optimization(ct, cst, stf, dij, pln)

plot_slice(
    image_volume=ct,
    cst=cst,
    overlay=result.physical_dose,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

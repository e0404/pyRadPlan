"""Example of photon sequencing with pyRadPlan."""

import logging

import numpy as np

from pyRadPlan import (
    PhotonPlan,
    load_tg119,
    generate_stf,
    calc_dose_influence,
    fluence_optimization,
    plot_slice,
    run_sequencing,
)

from pyRadPlan.optimization.objectives import SquaredDeviation, SquaredOverdosing

logging.basicConfig(level=logging.INFO)

# Load TG119 (provided within pyRadPlan)
ct, cst = load_tg119()

# Create a plan object
pln = PhotonPlan(
    machine="Generic",
    prop_seq={
        "run_sequencing": True,
        "num_of_levels": 5,
        "sequencer": "xia",
        "mode": "reducing_levels",
    },
)
num_of_beams = 2
pln.prop_stf = {
    "gantry_angles": np.linspace(0, 180, num_of_beams, endpoint=False),
    "couch_angles": np.zeros((num_of_beams,)),
}
# Generate Steering Geometry ("stf")
stf = generate_stf(ct, cst, pln)

# Calculate Dose Influence Matrix ("dij")
dij = calc_dose_influence(ct, cst, stf, pln)

# Optimize
cst.vois[1].objectives = [
    SquaredDeviation(penalty=1000.0, priority=1000.0, d_ref=50.0 / 30)
]  # Target
cst.vois[0].objectives = [SquaredOverdosing(penalty=300.0, priority=300.0, d_max=25.0 / 30)]  # OAR
cst.vois[2].objectives = [
    # MeanDose(priority=1.0, d_ref=0.0),
    SquaredOverdosing(penalty=100.0, priority=100.0, d_max=30.0 / 30),
]  # BODY

# Optimize -> Result
result = fluence_optimization(ct, cst, stf, dij, pln)

result_seq = run_sequencing(result, pln, stf, dij=dij)

# Choose a slice to visualize
view_slice = int(np.round(ct.size[2] / 2))

# Visualize
plot_slice(
    ct=ct,
    cst=cst,
    overlay=result.physical_dose.distribution,
    view_slice=view_slice,
    plane="axial",
    overlay_unit="Gy",
)

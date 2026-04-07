import numpy as np

from pyRadPlan import (
    StructureSet,
    PhotonPlan,
    generate_stf,
    calc_dose_forward,
)

from pyRadPlan.ct import create_ct
from pyRadPlan.cst import create_voi


def calculate_machine_scale(machine, ct_dim, ct_resolution, num_of_ct_scen):
    # Build water phantom
    ct, cst = build_water_phantom(ct_dim, ct_resolution)  # , num_of_ct_scen)

    # Machine
    if machine == "Generic":
        dose_at_isocenter_per_mu = 1.0  # Gy per 100 MU
    else:
        raise ValueError(f"Machine '{machine}' not recognized.")

    # TODO: correct?
    mask = np.ones((int(ct.y[-1] - ct.y[0]), int(ct.x[-1] - ct.x[0])))

    # Create a plan object
    pln = PhotonPlan(machine="Generic")
    num_of_beams = 1
    pln.prop_stf = {
        "gantry_angles": np.zeros(num_of_beams),
        "couch_angles": np.zeros(num_of_beams),
        "generator": "photonSingleBixel",
        "field_based": True,
        "field_shape": np.rot90(mask, k=-1),
        "resolution": 0.5,
    }

    # Generate Steering Geometry ("stf")
    stf = generate_stf(ct, cst, pln)

    # Ensure single bixel with weight 1.0
    stf.beams[0].rays[0].beamlets[0].weight = 1.0

    # Set isocenter & SAD
    stf.beams[0].iso_center = np.array([0.0, 0.0, 0.0])
    stf.beams[0].sad = 1000.0

    # Calculate Dose Influence Matrix ("dij")
    dij = calc_dose_forward(ct, cst, stf, pln)

    # Get peak dose
    peak_dose = central_axis_peak(dij, ct, stf)

    # Calculate weight to normalize dose at isocenter to dose_at_isocenter_per_mu Gy per MU
    return dose_at_isocenter_per_mu / peak_dose if peak_dose > 0 else 1.0


def central_axis_peak(dij, ct, stf):
    # TODO: adjust to central axis
    return np.max(dij["physical_dose"])


def build_water_phantom(ct_dim, ct_resolution):  # , num_of_ct_scen):
    box = np.zeros([ct_dim[1], ct_dim[0], ct_dim[2]])

    # Center the phantom around (0,0,0)
    size_mm = np.array(ct_dim) * np.array(ct_resolution)
    origin = -0.5 * size_mm

    ct_dict = {
        "cube_hu": box,
        "origin": tuple(origin),
        "direction": (1, 0, 0, 0, 1, 0, 0, 0, 1),
        "resolution": {"x": ct_resolution[0], "y": ct_resolution[1], "z": ct_resolution[2]},
    }
    ct = create_ct(ct_dict)

    # Create a single VOI that covers the entire phantom (for both target and body)
    box_mask = np.ones_like(box, dtype=np.uint8)
    voi_target = create_voi(voi_type="TARGET", name="phantom", ct_image=ct, mask=box_mask)
    voi_body = create_voi(voi_type="OAR", name="phantom", ct_image=ct, mask=box_mask)

    cst = StructureSet(vois=[voi_target, voi_body], ct_image=ct)

    return ct, cst

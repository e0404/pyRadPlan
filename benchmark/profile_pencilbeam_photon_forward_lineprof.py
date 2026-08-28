"""Line-profiler harness for the photon forward pencilbeam example.

This script keeps the workflow in functions so line_profiler can instrument it,
and it touches the selected backend once at startup to include backend setup
costs in the profile.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from line_profiler import LineProfiler

from pyRadPlan import (
    PhotonPlan,
    calc_dose_forward,
    generate_stf,
    load_tg119,
    plot_slice,
    settings,
    xp_utils,
)
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract
from pyRadPlan.dose.engines._base import DoseEngineBase
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine
from pyRadPlan.machines import create_bld
from pyRadPlan.machines.photons._calculate_machine_scale import central_axis_peak
from pyRadPlan.stf import FieldShapeAsBLD, FieldShapeComposite

logger = logging.getLogger(__name__)


def configure_backend(prefer_gpu: bool = True, backend_name: str = "jax") -> Any:
    """Configure preferred backend and touch it once to include init overhead."""
    settings.xp.prefer_gpu = prefer_gpu
    settings.xp.preferred_cpu_array_backend = backend_name
    settings.xp.preferred_gpu_array_backend = backend_name

    xp = xp_utils.choose_array_api_namespace()
    _ = xp_utils.choose_device(xp)

    # Touch backend once so runtime includes lazy backend/device setup costs.
    arr = xp.asarray([1.0, 2.0, 3.0])
    if hasattr(arr, "block_until_ready"):
        arr.block_until_ready()

    return xp


def build_blds_and_masks(resolution: float = 1.0, include_plots: bool = False):
    """Build field-shaping devices and optional diagnostic plots."""
    leaf_width = 5
    positions = [
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
    number_of_elements = len(positions)
    boundaries = np.arange(
        -int(number_of_elements / 2) * leaf_width,
        int(number_of_elements / 2) * leaf_width,
        leaf_width,
    )

    mlc_information = {
        "device_type": "MLC",
        "device_orientation": "X",
        "leaf_position_boundaries": boundaries,
        "leaf_positions": positions,
        "leaf_width": leaf_width,
        "leaf_leakage": 0.1,
    }
    mlc_x = create_bld(mlc_information)
    mask_mlc_x = mlc_x.calculate_transmission_mask(resolution)

    jaw_info = {
        "device_type": "JAW",
        "device_orientation": "X",
        "positions": [-20, 10],
        "field_width": 70,
        "leakage": 0.1,
    }
    jaw_info["field_width"] = mask_mlc_x.shape[0] * resolution
    jaw_x_matching_field_width = create_bld(jaw_info)
    mask_jaw_x = jaw_x_matching_field_width.calculate_transmission_mask(resolution)

    mask = mask_mlc_x * mask_jaw_x

    half_size = jaw_info["field_width"]
    extent = [-half_size, half_size, -half_size, half_size]

    if include_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(mask_jaw_x, cmap="gray", origin="lower", extent=extent)
        axes[0].set_title("Jaw Mask (IEC DICOM)")
        axes[0].set_xlabel("X (mm)")
        axes[0].set_ylabel("Y (mm)")

        axes[1].imshow(mask_mlc_x, cmap="gray", origin="lower", extent=extent)
        axes[1].set_title("MLC Mask (IEC DICOM)")
        axes[1].set_xlabel("X (mm)")
        axes[1].set_ylabel("Y (mm)")

        axes[2].imshow(mask, cmap="gray", origin="lower", extent=extent)
        axes[2].set_title("Combined Mask (IEC DICOM)")
        axes[2].set_xlabel("X (mm)")
        axes[2].set_ylabel("Y (mm)")

        plt.tight_layout()
        plt.show()

    return mlc_x, jaw_x_matching_field_width, extent


def build_field_shapes(
    mlc_x,
    jaw_x_matching_field_width,
    extent,
    resolution: float = 1.0,
    include_plots: bool = False,
):
    """Build field shapes and optional diagnostic plots."""
    mlc_shape = FieldShapeAsBLD(energy=6.0, bld=mlc_x, resolution=resolution)
    jaw_shape = FieldShapeAsBLD(energy=6.0, bld=jaw_x_matching_field_width, resolution=resolution)
    combined_shape = FieldShapeComposite(energy=6.0, shapes=[mlc_shape, jaw_shape])

    if include_plots:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(jaw_shape.mask, cmap="gray", origin="lower", extent=extent)
        axes[0].set_title("Jaw Field Shape (LPS BEV)")
        axes[0].set_xlabel("X (mm)")
        axes[0].set_ylabel("Y (mm)")

        axes[1].imshow(mlc_shape.mask, cmap="gray", origin="lower", extent=extent)
        axes[1].set_title("MLC Field Shape (LPS BEV)")
        axes[1].set_xlabel("X (mm)")
        axes[1].set_ylabel("Y (mm)")

        axes[2].imshow(combined_shape.mask, cmap="gray", origin="lower", extent=extent)
        axes[2].set_title("Combined Field Shape (LPS BEV)")
        axes[2].set_xlabel("X (mm)")
        axes[2].set_ylabel("Y (mm)")

        plt.tight_layout()
        plt.show()


def build_plan(mlc_x, jaw_x_matching_field_width) -> PhotonPlan:
    """Construct the photon plan used in the original example."""
    pln = PhotonPlan(machine="Generic")
    num_of_beams = 1
    pln.prop_stf = {
        "gantry_angles": np.linspace(0, 360, num_of_beams, endpoint=False),
        "couch_angles": np.zeros((num_of_beams,)),
        "generator": "photonSingleBixel",
        "field_based": True,
        "blds": [mlc_x, jaw_x_matching_field_width],
        "resolution": 0.5,
        "energy": 6,
    }
    return pln


def run_forward_example(include_plots: bool = False):
    """Run the complete forward-dose flow in one profiled function."""
    configure_backend(prefer_gpu=True, backend_name="jax")

    resolution = 1.0
    mlc_x, jaw_x_matching_field_width, extent = build_blds_and_masks(
        resolution=resolution,
        include_plots=include_plots,
    )
    build_field_shapes(
        mlc_x,
        jaw_x_matching_field_width,
        extent,
        resolution=resolution,
        include_plots=include_plots,
    )

    ct, cst = load_tg119()
    pln = build_plan(mlc_x, jaw_x_matching_field_width)
    stf = generate_stf(ct, cst, pln)

    # Absolute machine scaling (calculate_machine_scale) is skipped; not needed for profiling
    weights = 1.0

    for beam in stf.beams:
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                beamlet.weight *= weights

    dij = calc_dose_forward(ct, cst, stf, pln)

    # Ensure async backends finish before function return to keep timings meaningful.
    dose = dij["physical_dose"]
    if hasattr(dose, "block_until_ready"):
        dose.block_until_ready()

    if include_plots:
        view_slice = int(np.round(ct.size[1] / 2))
        plot_slice(
            image_volume=ct,
            cst=cst,
            overlay=dij["physical_dose"],
            view_slice=view_slice,
            plane="coronal",
            overlay_unit="Gy",
            save_filename="pencilbeam_photon_forward_dose.png",
        )

    return dij


def build_profiler() -> LineProfiler:
    """Create a line profiler with top-level and hotspot functions."""
    lp = LineProfiler()

    lp.add_function(central_axis_peak)
    lp.add_function(calc_dose_forward)
    lp.add_function(DoseEngineBase.calc_dose_forward)
    lp.add_function(PhotonPencilBeamSVDEngine._calc_dose)
    lp.add_function(PencilBeamEngineAbstract._init_dose_calc)

    return lp


def main():
    """Parse arguments and run the profiled forward-dose flow."""
    parser = argparse.ArgumentParser(
        description="Line profile the pencilbeam photon forward flow."
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable plotting (disabled by default for cleaner profiling).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of profiled runs (use 2 to compare cold vs warm behavior).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # logger.info("Starting unprofiled warmup run")
    # run_forward_example(include_plots=args.plots)

    lp = build_profiler()
    wrapped = lp(run_forward_example)

    for run_idx in range(args.runs):
        logger.info("Starting profiled run %d/%d", run_idx + 1, args.runs)
        wrapped(include_plots=args.plots)

    lp.print_stats()


if __name__ == "__main__":
    main()

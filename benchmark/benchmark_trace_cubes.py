"""Benchmarks for RayTracerSiddon.trace_cubes with varying image resolutions."""

import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.geometry import lps
from pyRadPlan.raytracer import RayTracerSiddon
from pyRadPlan.stf._beam import Beam
from pyRadPlan.stf._ray import Ray
from pyRadPlan.stf._beamlet import Beamlet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cube(size, spacing, origin=None):
    """Create a uniform-density SimpleITK image (density = 1.0 ~ water)."""
    arr = np.ones(size[::-1], dtype=np.float32)  # sitk expects (z, y, x)
    image = sitk.GetImageFromArray(arr)
    image.SetSpacing([float(s) for s in spacing])
    if origin is None:
        # Centre the cube around the origin
        origin = [-0.5 * (n - 1) * s for n, s in zip(size, spacing)]
    image.SetOrigin(origin)
    return image


def _make_beam(iso_center, n_rays=1, sad=1000.0, gantry_angle=0.0):
    """Build a minimal Beam object suitable for trace_cubes."""
    # Spread rays in a small grid around the isocenter in BEV (x, y=0, z)
    side = int(np.ceil(np.sqrt(n_rays)))
    offsets = np.linspace(-10, 10, side)
    rays = []
    count = 0
    for x in offsets:
        for z in offsets:
            if count >= n_rays:
                break
            ray_pos_bev = np.array([x, 0.0, z])
            ray_pos = ray_pos_bev.copy()  # simplified – gantry 0 means BEV ≈ LPS
            rays.append(
                Ray(
                    rayPos_bev=ray_pos_bev,
                    ray_pos=ray_pos,
                    beamlets=[Beamlet(energy=100.0)],
                )
            )
            count += 1
        if count >= n_rays:
            break

    # Explicit consistent source points so the benchmark is comparable across
    # pyRadPlan versions with and without the Beam source-point derivation
    source_point_bev = np.array([0.0, -sad, 0.0])
    rot_mat = lps.get_beam_rotation_matrix(gantry_angle, 0.0)
    return Beam(
        gantry_angle=gantry_angle,
        couch_angle=0.0,
        iso_center=np.array(iso_center, dtype=np.float64),
        rays=rays,
        SAD=sad,
        source_point_bev=source_point_bev,
        source_point=rot_mat @ source_point_bev,
    )


# ---------------------------------------------------------------------------
# Parametrised image configurations
# ---------------------------------------------------------------------------

# (label, size_xyz, spacing_xyz) – from small/coarse to large/fine
IMAGE_CONFIGS = [
    ("small_coarse__50x50x50_sp5", (50, 50, 50), (5.0, 5.0, 5.0)),
    ("medium_coarse__100x100x50_sp3", (200, 200, 80), (3.0, 3.0, 3.0)),
    ("medium_fine__100x100x50_sp1", (200, 200, 80), (1.0, 1.0, 2.0)),
    ("large_coarse__200x200x100_sp3", (512, 512, 128), (3.0, 3.0, 3.0)),
    ("large_fine__200x200x100_sp1", (512, 512, 128), (1.0, 1.0, 2.0)),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=IMAGE_CONFIGS, ids=[c[0] for c in IMAGE_CONFIGS])
def raytracer_and_beam(request):
    """Yield a (RayTracerSiddon, Beam) pair for each image configuration."""
    label, size, spacing = request.param
    cube = _make_cube(size, spacing)
    iso_center = cube.TransformContinuousIndexToPhysicalPoint([s / 2.0 for s in size])
    rt = RayTracerSiddon(cube)
    rt.lateral_cut_off = 50.0
    beam = _make_beam(iso_center, n_rays=4)
    return rt, beam


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


def test_trace_cubes(benchmark, raytracer_and_beam):
    """Benchmark trace_cubes for different image resolutions."""
    rt, beam = raytracer_and_beam
    # Use few rounds/iterations – raytracing can be expensive for large images.
    benchmark.pedantic(
        rt.trace_cubes,
        args=(beam,),
        rounds=3,
        iterations=1,
        warmup_rounds=1,
    )

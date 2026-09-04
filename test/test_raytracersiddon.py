import warnings

import array_api_strict as xps
import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan import settings
from pyRadPlan.core.xp_utils import from_numpy
from pyRadPlan.geometry import lps
from pyRadPlan.raytracer import RayTracerBase, RayTracerSiddon
from pyRadPlan.stf._beam import Beam
from pyRadPlan.stf._beamlet import Beamlet
from pyRadPlan.stf._ray import Ray


@pytest.fixture
def sample_cube():
    cubes = np.random.rand(*(3, 5, 7))  # 3x3x3 cube of ones
    image = sitk.GetImageFromArray(cubes)
    image.SetSpacing([1, 1, 1])
    return image


def test_raytracer_init_siddon(sample_cube):
    raytracer = RayTracerSiddon(sample_cube)
    assert isinstance(raytracer, RayTracerBase)
    assert isinstance(raytracer, RayTracerSiddon)
    assert isinstance(raytracer.lateral_cut_off, float)
    assert len(raytracer.cubes) == 1

    cube_dim = sample_cube.GetSize()

    assert len(raytracer._x_planes) == cube_dim[0] + 1
    assert len(raytracer._y_planes) == cube_dim[1] + 1
    assert len(raytracer._z_planes) == cube_dim[2] + 1


def test_raytracer_trace_single_ray(sample_cube):
    raytracer = RayTracerSiddon(sample_cube)

    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])

    source_points = np.array([0, -5, 0]).astype(float)
    target_points = np.array([0, 5, 0]).astype(float)

    alpha, l, rho, d12, ix = raytracer.trace_ray(isocenter, source_points, target_points)

    assert len(rho) == len(raytracer.cubes)

    # The ray should go through the middle of the cube in y
    cube_np = sitk.GetArrayViewFromImage(sample_cube)
    rho_expected = cube_np.ravel(order="F")[ix]
    assert np.allclose(rho[0], rho_expected)
    assert np.isclose(d12, np.sqrt(np.sum((target_points - source_points) ** 2)))
    assert d12.ndim != 0
    assert np.allclose(l, sample_cube.GetSpacing()[1] * np.ones_like(l))  # Spacing is one


def test_raytracer_trace_multiple_rays(sample_cube):
    raytracer = RayTracerSiddon(sample_cube)

    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])

    source_points = np.array([[0, -5, 0], [0, -5, 0]]).astype(float)
    target_points = np.array([[0, 5, 0], [2, 5, 0]]).astype(float)

    alpha, l, rho, d12, ix = raytracer.trace_rays(isocenter, source_points, target_points)

    assert len(rho) == len(raytracer.cubes)

    cube_np = sitk.GetArrayViewFromImage(sample_cube)
    rho_expected = cube_np.ravel(order="F")[ix].reshape(rho[0].shape)
    rho_expected[ix < 0] = 0.0
    # rho[0][np.isnan(rho[0])] = 0.0
    assert np.allclose(rho[0], rho_expected)


def test_raytracer_candidate_mx_array_api(sample_cube):
    raytracer = RayTracerSiddon(sample_cube)

    ray_spacing = xps.min(from_numpy(xps, raytracer._resolution)) / xps.sqrt(
        xps.asarray(2.0, dtype=xps.float32)
    )

    spacing_range = ray_spacing * xps.arange(
        xps.floor(-500.0 / ray_spacing), xps.ceil(500.0 / ray_spacing) + 1, dtype=xps.float32
    )

    lookup_pos = xps.zeros((100, 3), dtype=xps.float32)
    lookup_pos[:, 0] = xps.asarray(np.random.uniform(-500.0, 500.0, 100), dtype=xps.float32)
    lookup_pos[:, 2] = xps.asarray(np.random.uniform(-500.0, 500.0, 100), dtype=xps.float32)

    candidate_mx = raytracer._get_candidate_ray_matrix(spacing_range, lookup_pos)


def test_raytracer_trace_multiple_cubes(sample_cube):
    raytracer = RayTracerSiddon([sample_cube, sample_cube])

    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])

    source_points = np.array([[0, -5, 0], [0, -5, 0]]).astype(float)
    target_points = np.array([[0, 5, 0], [2, 5, 0]]).astype(float)

    alpha, l, rho, d12, ix = raytracer.trace_rays(isocenter, source_points, target_points)

    assert len(rho) == len(raytracer.cubes)

    cube_np = sitk.GetArrayViewFromImage(sample_cube)
    rho_expected = cube_np.ravel(order="F")[ix].reshape(rho[0].shape)
    rho_expected[ix < 0] = 0.0
    # rho[0][np.isnan(rho[0])] = 0.0
    # rho[1][np.isnan(rho[1])] = 0.0
    assert np.allclose(rho[0], rho_expected)
    assert np.allclose(rho[1], rho_expected)


def test_raytracer_ray_does_not_hit(sample_cube):
    raytracer = RayTracerSiddon(sample_cube)

    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])

    source_point = np.array([100, -5, 100]).astype(float)
    target_point = np.array([100, 5, 100]).astype(float)

    alpha, l, rho, d12, ix = raytracer.trace_ray(isocenter, source_point, target_point)

    assert alpha.size == 0
    assert l.size == 0
    assert rho[0].size == 0
    assert np.isclose(d12, np.sqrt(np.sum((target_point - source_point) ** 2)))
    assert ix.size == 0


def test_raytracer_trace_rays_jax(sample_cube, monkeypatch):
    pytest.importorskip("jax")
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", "jax")

    raytracer = RayTracerSiddon(sample_cube)
    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])
    source_points = np.array([[0.0, -5.0, 0.0]])
    target_points = np.array([[0.0, 5.0, 0.0], [2.0, 5.0, 0.0]])

    _, _, rho, _, ix = raytracer.trace_rays(isocenter, source_points, target_points)

    valid = ix >= 0
    expected = np.full(ix.shape, np.nan, dtype=raytracer.precision)
    cube_linear = sitk.GetArrayViewFromImage(sample_cube).ravel(order="F")
    expected[valid] = cube_linear[ix[valid]]
    np.testing.assert_allclose(rho[0], expected, equal_nan=True)


@pytest.mark.parametrize("backend", ["array_api_strict", "torch"])
def test_readonly_sitk_buffer_is_copied_for_backend(sample_cube, monkeypatch, backend):
    if backend == "torch":
        pytest.importorskip("torch")

    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", backend)
    cube_before = sitk.GetArrayViewFromImage(sample_cube).copy()
    raytracer = RayTracerSiddon(sample_cube)
    isocenter = sample_cube.TransformIndexToPhysicalPoint([3, 2, 1])

    with warnings.catch_warnings(record=True) as caught:
        _, _, rho, _, ix = raytracer.trace_rays(
            isocenter,
            np.array([[0.0, -5.0, 0.0]]),
            np.array([[0.0, 5.0, 0.0], [2.0, 5.0, 0.0]]),
        )

    valid = ix >= 0
    expected = np.full(ix.shape, np.nan, dtype=raytracer.precision)
    expected[valid] = cube_before.ravel(order="F")[ix[valid]]
    np.testing.assert_allclose(rho[0], expected, equal_nan=True)
    np.testing.assert_array_equal(sitk.GetArrayViewFromImage(sample_cube), cube_before)
    assert not any("not writable" in str(warning.message).lower() for warning in caught)


def test_boundary_planes_do_not_create_duplicate_voxel_segments(monkeypatch):
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", "numpy")

    size = (20, 3, 3)
    spacing = (0.7, 1.1, 1.3)
    origin = (0.1, 0.2, 0.3)
    cube = sitk.GetImageFromArray(np.ones(size[::-1], dtype=np.float32))
    cube.SetSpacing(spacing)
    cube.SetOrigin(origin)
    raytracer = RayTracerSiddon(cube)
    yz = np.array([origin[1] + spacing[1], origin[2] + spacing[2]])

    _, lengths, _, _, ix = raytracer.trace_rays(
        np.zeros(3),
        np.array([[-100.0, yz[0], yz[1]]]),
        np.array([[100.0, yz[0], yz[1]]]),
    )

    valid = ix[0] >= 0
    valid_ix = ix[0, valid]
    assert valid_ix.size == size[0]
    assert np.all(np.diff(valid_ix) != 0)
    assert np.isclose(np.sum(lengths[0, valid]), size[0] * spacing[0])


@pytest.mark.parametrize(("gantry", "couch"), [(0.0, 0.0), (45.0, 0.0), (30.0, 45.0)])
def test_trace_cubes_traverses_full_cube_at_oblique_angles(monkeypatch, gantry, couch):
    """The ray matrix extent must reach past the cube for any beam orientation.

    Regression test: a mis-derived BEV extent placed the ray targets inside the
    cube for oblique angles, truncating traversal on the far side.
    """
    monkeypatch.setattr(settings.xp, "prefer_gpu", False)
    monkeypatch.setattr(settings.xp, "preferred_cpu_array_backend", "numpy")

    # Anisotropic on purpose: with a cubic, isotropically spaced cube an index-layout
    # mix-up maps corners onto corners and would go unnoticed
    size = (24, 16, 10)
    spacing = (1.5, 2.5, 3.5)
    cube = sitk.GetImageFromArray(np.ones(size[::-1], dtype=np.float32))
    cube.SetSpacing(spacing)
    cube.SetOrigin([-0.5 * (n - 1) * s for n, s in zip(size, spacing)])

    raytracer = RayTracerSiddon(cube)
    raytracer.lateral_cut_off = 100.0  # covers the whole cube laterally

    beam = Beam(
        gantry_angle=gantry,
        couch_angle=couch,
        iso_center=np.zeros(3),
        rays=[Ray(ray_pos_bev=np.zeros(3), ray_pos=np.zeros(3), beamlets=[Beamlet(energy=100.0)])],
        SAD=10000.0,
    )
    # source points are derived consistently from SAD and the angles by the Beam model
    np.testing.assert_allclose(beam.source_point_bev, [0.0, -10000.0, 0.0])
    rot_mat = lps.get_beam_rotation_matrix(gantry, couch)
    np.testing.assert_allclose(beam.source_point, rot_mat @ beam.source_point_bev)

    rad_depth = sitk.GetArrayFromImage(raytracer.trace_cubes(beam)[0])
    finite = np.isfinite(rad_depth)

    # Unit density and full lateral coverage: every voxel must receive a depth, and the
    # deepest voxel must lie about one full chord through the cube behind the entry face
    assert finite.mean() > 0.99
    assert np.nanmax(rad_depth) > 0.9 * min(n * s for n, s in zip(size, spacing))

import pytest
import numpy as np
import SimpleITK as sitk

from pyRadPlan.raytracer import RayTracerBase, RayTracerSiddon


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

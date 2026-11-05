"""Photon depth-constant reuse and component arithmetic across Array API backends."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch"])
def engine(request):
    """Minimal photon engine on each backend, without unrelated patient setup."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy unavailable")
    module = pytest.importorskip(request.param)
    result = object.__new__(PhotonPencilBeamSVDEngine)
    result.xp = array_api_compat.array_namespace(module.asarray([0.0]))
    result.device = array_api_compat.device(module.asarray([0.0]))
    result.enable_dij_sampling = False
    result.use_custom_primary_photon_fluence = False
    return result


@pytest.mark.parametrize("depth_dtype", ["float32", "float64"])
@pytest.mark.parametrize("kernel_dtype", ["float32", "float64"])
@pytest.mark.parametrize("components,voxels", [(1, 7), (3, 7), (5, 0)])
def test_bixel_component_precision(
    engine, monkeypatch, depth_dtype, kernel_dtype, components, voxels
):
    """Preserve component rounding before summation, including mixed precision and empty rays."""
    xp = engine.xp
    dtype = getattr(xp, depth_dtype)
    kernel = SimpleNamespace(m=0.0037, kernel_betas=np.linspace(0.0123, 0.0987, components))
    depths = xp.asarray(np.linspace(0, 127.3, voxels), dtype=dtype)
    queries = xp.zeros((voxels, 2), dtype=dtype)
    fields = xp.asarray(
        (np.arange(voxels * components).reshape(voxels, components) / 17.123456789 + 0.123456789)
        * np.where(np.arange(components) % 2, -1.0, 1.0),
        dtype=getattr(xp, kernel_dtype),
    )

    def interpolate(actual):
        assert actual is queries
        return fields

    ray = dict(
        kernel=kernel,
        kernel_depth_parameters={},
        rad_depths=depths,
        kernel_interpolators=interpolate,
        iso_lat_dists=queries,
        geo_depths=xp.asarray(np.linspace(850, 1150, voxels), dtype=dtype),
        sad=1000.0,
        ix=xp.arange(voxels, dtype=xp.int64),
        beamlets=[dict(weight=0.75)],
    )
    # Independent scalar-component calculation, with the assignment cast made explicit.
    m = xp.asarray(kernel.m, dtype=dtype)
    betas = xp.asarray(kernel.kernel_betas, dtype=dtype)
    products = []
    for c in range(components):
        component = betas[c] / (betas[c] - m) * (xp.exp(-m * depths) - xp.exp(-betas[c] * depths))
        products.append(xp.astype(component * fields[:, c], dtype))
    expected = xp.sum(xp.stack(products), axis=0) * ((ray["sad"] / ray["geo_depths"]) ** 2)
    monkeypatch.setattr(
        "pyRadPlan.dose.engines._svdpb.to_namespace",
        lambda *args, **kwargs: pytest.fail("Unexpected conversion in bixel evaluation"),
    )
    for _ in range(2):
        actual = engine._compute_bixel(ray, 0)
        np.testing.assert_array_equal(to_numpy(actual["physical_dose"]), to_numpy(expected))
        assert actual["physical_dose"].dtype == expected.dtype
        assert actual["ix"] is ray["ix"]
        assert actual["weight"] == 0.75
    assert len(ray["kernel_depth_parameters"]) == 1


def test_constant_cache_precision_and_beam_isolation(engine):
    """Reuse constants across rays, separate scenario precision and a different beam kernel."""
    xp = engine.xp
    kernel = SimpleNamespace(m=0.003, kernel_betas=np.array([0.01, 0.02, 0.03]))
    ray = dict(kernel=kernel, kernel_depth_parameters={})
    depths = xp.asarray([1, 2], dtype=xp.float32)
    first = engine._get_kernel_depth_parameters(ray, depths)
    other_ray = ray.copy()
    assert engine._get_kernel_depth_parameters(other_ray, depths) is first
    second = engine._get_kernel_depth_parameters(other_ray, xp.astype(depths, xp.float64))
    assert second is not first
    for values in first:
        assert values.dtype == xp.float32
        assert array_api_compat.device(values) == array_api_compat.device(depths)
    for values in second:
        assert values.dtype == xp.float64
    assert len(ray["kernel_depth_parameters"]) == 2
    other_beam = dict(
        kernel=SimpleNamespace(m=0.006, kernel_betas=kernel.kernel_betas),
        kernel_depth_parameters={},
    )
    changed = engine._get_kernel_depth_parameters(other_beam, depths)
    assert changed is not first
    assert not np.array_equal(to_numpy(changed[2]), to_numpy(first[2]))
    np.testing.assert_array_equal(kernel.kernel_betas, [0.01, 0.02, 0.03])


def test_initialized_rays_share_beam_cache(engine, monkeypatch):
    """All initialized rays carry the same beam-owned depth-constant cache."""
    monkeypatch.setattr(PencilBeamEngineAbstract, "_init_ray", lambda self, beam, j: {})
    beam = dict(
        kernel=object(),
        kernel_depth_parameters={},
        field_based_dose_calc=False,
        kernel_interpolators=object(),
    )
    first = engine._init_ray(beam, 0)
    second = engine._init_ray(beam, 1)
    assert first["kernel_depth_parameters"] is second["kernel_depth_parameters"]
    assert first["kernel_depth_parameters"] is beam["kernel_depth_parameters"]

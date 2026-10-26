"""Shared depth preparation for both tracing grids and optional image retention."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest
import SimpleITK as sitk

from pyRadPlan.core import Grid
from pyRadPlan.core.resample import resample_image
from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine


@pytest.mark.parametrize("backend", ["numpy", "cupy", "array_api_strict"])
@pytest.mark.parametrize("trace_on_dose_grid", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_depth_selection_and_retained_images(monkeypatch, backend, trace_on_dose_grid, keep):
    """Keep values on device unless images are requested, preserving selected depths."""
    module = pytest.importorskip(backend)
    if backend == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    xp = array_api_compat.get_namespace(module.asarray([0.0]))
    values = np.random.default_rng(21).uniform(size=(9, 8, 7)).astype(np.float32)
    values.ravel()[::13] = np.nan
    source = sitk.GetImageFromArray(values)
    source.SetSpacing((1.3, 2.1, 1.7))
    source.SetOrigin((-3.0, 8.0, -2.0))
    grid = Grid.from_sitk_image(source)
    target = grid if trace_on_dose_grid else grid.resample(dict(x=2.1, y=2.9, z=3.1))
    expected = (
        source
        if trace_on_dose_grid
        else resample_image(source, interpolator=sitk.sitkLinear, target_grid=target)
    )
    indices = np.arange(target.num_voxels)[::3]
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.xp = xp
    engine.device = array_api_compat.device(module.asarray([0.0]))
    engine.dose_grid = target
    engine._vdose_grid = xp.asarray(indices)
    engine._vox_world_coords = xp.empty((0, 3), dtype=xp.float32)
    engine._raytracer = SimpleNamespace(cubes=[source])
    engine._depth_resampler = None
    engine._rad_depth_cubes = []
    engine.trace_on_dose_grid = trace_on_dose_grid
    engine.keep_rad_depth_cubes = keep
    if not keep:
        monkeypatch.setattr(
            "pyRadPlan.dose.engines._base_pencilbeam.to_numpy",
            lambda *args: pytest.fail("Unexpected depth download"),
        )
    result = engine._sample_radiological_depths([xp.asarray(values)])[0]
    old = sitk.GetArrayFromImage(expected)
    np.testing.assert_allclose(to_numpy(result), old.ravel()[indices], rtol=2e-6, atol=2e-6)
    assert array_api_compat.device(result) == engine.device
    assert array_api_compat.array_namespace(result) == xp
    if keep:
        image = engine._rad_depth_cubes[0]
        assert image.GetSpacing() == expected.GetSpacing()
        assert image.GetOrigin() == expected.GetOrigin()
        assert image.GetDirection() == expected.GetDirection()
        np.testing.assert_allclose(sitk.GetArrayFromImage(image), old, rtol=2e-6, atol=2e-6)
    else:
        assert engine._rad_depth_cubes == []

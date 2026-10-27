"""Regression coverage for prepared photon convolution interpolators."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available
from pyRadPlan.core.xp_utils.compat import _fft2, _ifft2, interpnd
from pyRadPlan.core.xp_utils.helpers import to_namespace
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine


@pytest.fixture(params=["numpy", "cupy", "array_api_strict"])
def interpolation_backend(request):
    """Exercise both prepared backends and the Array API fallback."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    module = pytest.importorskip(request.param)
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    return xp


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_kernel_interpolators_reuse_and_fluence_isolation(interpolation_backend, dtype_name):
    """Repeated queries preserve values; another field cannot overwrite the first."""
    xp = interpolation_backend
    dtype = getattr(xp, dtype_name)
    engine = SimpleNamespace(xp=xp, device=None)
    grid = xp.asarray([-2, -1, 0, 1, 2], dtype=dtype)
    kernels = xp.asarray(np.arange(18).reshape(2, 3, 3) / 20, dtype=dtype)
    fluence = xp.asarray([[0.0, 1.0], [0.5, 0.25]], dtype=dtype)
    info = {
        "kernel": SimpleNamespace(num_kernel_components=2),
        "kernel_conv_grid": grid,
        "kernel_conv_size": 5,
        "kernel_mxs": kernels,
    }
    queries = xp.asarray([[-2, -2], [2, 2], [0.25, -0.75], [-1, 1]], dtype=dtype)
    expected = []
    for c in range(kernels.shape[0]):
        kernel = kernels[c, ...]
        values = xp.real(_ifft2(_fft2(fluence, s=(5, 5)) * _fft2(kernel, s=(5, 5))))
        expected.append(interpnd(queries, (grid, grid), values, bounds_error=True))

    interpolators = PhotonPencilBeamSVDEngine._get_kernel_interpolators(engine, info, fluence)
    other = PhotonPencilBeamSVDEngine._get_kernel_interpolators(engine, info, 2 * fluence)
    for _ in range(2):
        original = to_namespace(np, interpolators(queries))
        changed = to_namespace(np, other(queries))
        reference = np.column_stack([to_namespace(np, v) for v in expected])
        np.testing.assert_allclose(original, reference, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(changed, 2 * reference, rtol=1e-6, atol=1e-8)

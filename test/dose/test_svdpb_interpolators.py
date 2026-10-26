"""Regression coverage for prepared photon convolution interpolators."""

from types import SimpleNamespace
from unittest.mock import patch

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

    # Count actual backend construction, not timing, to guard against rebuilding
    # interpolators inside a returned callable on every query.
    if array_api_compat.is_numpy_namespace(xp):
        from scipy import interpolate  # noqa: PLC0415
    elif array_api_compat.is_cupy_namespace(xp):
        from cupyx.scipy import interpolate  # noqa: PLC0415
    else:
        interpolate = None

    def check():
        interpolators = PhotonPencilBeamSVDEngine._get_kernel_interpolators(engine, info, fluence)
        other = PhotonPencilBeamSVDEngine._get_kernel_interpolators(engine, info, 2 * fluence)
        for _ in range(2):
            for original, changed, reference in zip(interpolators, other, expected):
                np.testing.assert_array_equal(
                    to_namespace(np, original(queries)), to_namespace(np, reference)
                )
                np.testing.assert_allclose(
                    to_namespace(np, changed(queries)),
                    2 * to_namespace(np, reference),
                    rtol=1e-6,
                )
                assert original(xp.empty((0, 2), dtype=dtype)).shape == (0,)
                for point in [[-2.1, 0], [0, 2.1]]:
                    with pytest.raises(ValueError):
                        original(xp.asarray([point], dtype=dtype))

    if interpolate is None:
        check()
    else:
        with patch.object(
            interpolate, "RegularGridInterpolator", wraps=interpolate.RegularGridInterpolator
        ) as constructor:
            check()
        assert constructor.call_count == 4  # Two components for each of two fluences.

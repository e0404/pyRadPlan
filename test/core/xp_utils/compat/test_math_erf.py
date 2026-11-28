"""The error function matches SciPy on every backend and in the portable fallback."""

import array_api_compat
import numpy as np
import pytest
from scipy.special import erf as scipy_erf

from pyRadPlan.core.xp_utils import erf, to_numpy
from pyRadPlan.core.xp_utils.compat import _erf_portable


@pytest.fixture(params=["numpy", "torch", "array_api_strict"])
def xp(request):
    backend = pytest.importorskip(request.param)
    return array_api_compat.array_namespace(backend.asarray([0.0]))


def _grid():
    """Dense grid up to |x| = 10 with the interval boundaries of the approximation."""
    x = np.concatenate(
        [
            np.linspace(-10, 10, 200001),
            np.geomspace(1e-300, 1e-3, 1000),
            [0.46875, np.nextafter(0.46875, 1), 4.0, np.nextafter(4.0, 5), 6.0, 0.0],
        ]
    )
    return np.concatenate([x, -x])


def test_portable_erf_float64_accuracy():
    """Cody's approximation is accurate to a few ulp in double precision."""
    x = _grid()
    ref = scipy_erf(x)
    np.testing.assert_allclose(_erf_portable(x), ref, rtol=2e-15, atol=1e-300)


def test_portable_erf_float32_accuracy():
    """In single precision the error stays at the level of the float32 resolution."""
    x = _grid().astype(np.float32)
    out = _erf_portable(x)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, scipy_erf(x.astype(np.float64)), rtol=1e-6, atol=1e-7)


def test_portable_erf_special_values():
    """Infinite inputs saturate and NaN propagates."""
    out = _erf_portable(np.array([np.inf, -np.inf, np.nan]))
    np.testing.assert_array_equal(out[:2], [1.0, -1.0])
    assert np.isnan(out[2])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_erf_matches_scipy_on_backend(xp, dtype):
    """The dispatched erf keeps namespace and dtype and matches SciPy."""
    x_np = np.linspace(-7, 7, 2001)
    x = xp.asarray(x_np, dtype=getattr(xp, dtype))
    out = erf(x)
    assert array_api_compat.array_namespace(out) is array_api_compat.array_namespace(x)
    assert out.dtype == x.dtype
    tol = 1e-6 if dtype == "float32" else 2e-15
    np.testing.assert_allclose(
        to_numpy(out), scipy_erf(to_numpy(x).astype(np.float64)), rtol=tol, atol=tol
    )

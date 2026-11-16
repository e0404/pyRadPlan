"""Benchmark interpnd with different backends."""

import numpy as np
import pytest

from pyRadPlan.core.xp_utils.compat import interpnd
from pyRadPlan.core.xp_utils import ArrayBackendGridInterpolator


try:
    import jax
    import jax.numpy as jnp
except Exception:
    jax = None
    jnp = None

try:
    import cupy as cp
except Exception:
    cp = None

try:
    import torch
except Exception:
    torch = None


# %% Helpers
def _case_size(ndim: int) -> tuple[int, int]:
    if ndim == 2:
        return 256, 10_000
    if ndim == 3:
        return 64, 10_000
    if ndim == 4:
        return 24, 5_000
    raise ValueError(ndim)


def _make_case(xp, ndim: int):
    n_grid, n_query = _case_size(ndim)

    axes_np = tuple(np.linspace(0.0, 1.0, n_grid) for _ in range(ndim))

    mesh = np.meshgrid(*axes_np, indexing="ij")
    values_np = sum((i + 1) * np.sin(2 * np.pi * m) for i, m in enumerate(mesh))

    rng = np.random.default_rng(42)
    xq_np = rng.uniform(0.0, 1.0, size=(n_query, ndim))

    x = tuple(xp.asarray(axis, dtype=xp.float64) for axis in axes_np)
    y = xp.asarray(values_np, dtype=xp.float64)
    xq = xp.asarray(xq_np, dtype=xp.float64)

    return xq, x, y


def _block_until_ready(result):
    if jax is not None and hasattr(result, "block_until_ready"):
        result.block_until_ready()

    if cp is not None:
        try:
            cp.cuda.Stream.null.synchronize()
        except Exception:
            pass

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass

    return result


# %% Tests
@pytest.mark.parametrize("backend", ["numpy", "cupy"])
@pytest.mark.parametrize("prefer_native", [True, False], ids=["native", "generic"])
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
@pytest.mark.parametrize("operation", ["construct", "evaluate", "oneshot"])
@pytest.mark.parametrize("single_query", [False, True], ids=["batch", "single"])
def test_interpnd_lifecycle(benchmark, backend, prefer_native, ndim, operation, single_query):  # noqa: PLR0913
    """Separate setup, repeated queries, and one-shot cost."""
    if not prefer_native and ndim == 4:
        pytest.skip("Generic interpolation is 2D/3D")
    if backend == "cupy" and (cp is None or not cp.cuda.is_available()):
        pytest.skip("CuPy device unavailable")
    xp = np if backend == "numpy" else cp
    xq, x, y = _make_case(xp, ndim)
    if single_query:
        xq = xq[:1, :]

    def construct():
        return ArrayBackendGridInterpolator(x, y, prefer_native=prefer_native)

    interpolator = construct()
    expected = interpnd(xq, x, y)
    actual = interpolator(xq)
    np.testing.assert_allclose(
        actual if backend == "numpy" else actual.get(),
        expected if backend == "numpy" else expected.get(),
        rtol=1e-12,
        atol=1e-12,
    )
    operations = {
        "construct": construct,
        "evaluate": lambda: interpolator(xq),
        "oneshot": lambda: interpnd(xq, x, y, prefer_native=prefer_native),
    }

    def measured():
        result = operations[operation]()
        if backend == "cupy":
            cp.cuda.get_current_stream().synchronize()
        return result

    measured()  # Warm up; synchronize only the backend under measurement.
    benchmark(measured)


@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_numpy(benchmark, ndim):
    """Measure one-shot interpolation with NumPy."""
    xq, x, y = _make_case(np, ndim)

    benchmark(lambda: interpnd(xq, x, y))


@pytest.mark.skipif(jax is None, reason="JAX not installed")
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_jax(benchmark, ndim):
    """Measure one-shot interpolation with JAX after compilation."""
    xq, x, y = _make_case(jnp, ndim)

    # Warmup: compile is not benchmarked
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(cp is None, reason="CuPy is not available")
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_cupy(benchmark, ndim):
    """Measure synchronized one-shot interpolation with CuPy."""
    xq, x, y = _make_case(cp, ndim)

    # warmup
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_interpnd_torch_cpu(benchmark, ndim):
    """Measure one-shot interpolation with CPU tensors."""
    xq, x, y = _make_case(torch, ndim)

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="PyTorch CUDA is not available",
)
@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_interpnd_torch_cuda(benchmark, ndim):
    """Measure synchronized one-shot interpolation with CUDA tensors."""
    xq_np, x_np, y_np = _make_case(np, ndim)

    x = tuple(torch.asarray(axis, dtype=torch.float64, device="cuda") for axis in x_np)
    y = torch.asarray(y_np, dtype=torch.float64, device="cuda")
    xq = torch.asarray(xq_np, dtype=torch.float64, device="cuda")

    # warmup
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))

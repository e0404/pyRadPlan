"""Benchmark interp1d with different backends."""

import numpy as np
import pytest

from pyRadPlan.core.xp_utils.compat import interp1d


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
def _make_case(xp, y_ndim: int):
    n_grid = 4096
    n_query = 100_000
    n_series = 32

    x_np = np.linspace(0.0, 1.0, n_grid)

    rng = np.random.default_rng(42)
    xq_np = rng.uniform(0.0, 1.0, size=n_query)

    if y_ndim == 1:
        y_np = np.sin(2 * np.pi * x_np)
    elif y_ndim == 2:
        y_np = np.stack(
            [np.sin(2 * np.pi * x_np * (i + 1)) + 0.1 * i * x_np for i in range(n_series)],
            axis=0,
        )
    else:
        raise ValueError(y_ndim)

    xq = xp.asarray(xq_np, dtype=xp.float64)
    x = xp.asarray(x_np, dtype=xp.float64)
    y = xp.asarray(y_np, dtype=xp.float64)

    return xq, x, y


def _make_sequence_case(xp):
    n_grid = 4096
    n_query = 100_000
    n_series = 8

    x_np = np.linspace(0.0, 1.0, n_grid)

    rng = np.random.default_rng(42)
    xq_np = rng.uniform(0.0, 1.0, size=n_query)

    ys_np = [np.sin(2 * np.pi * x_np * (i + 1)) + 0.1 * i * x_np for i in range(n_series)]

    xq = xp.asarray(xq_np, dtype=xp.float64)
    x = xp.asarray(x_np, dtype=xp.float64)
    y = [xp.asarray(v, dtype=xp.float64) for v in ys_np]

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
@pytest.mark.parametrize("y_ndim", [1, 2], ids=["y-1d", "y-2d"])
def test_interp1d_numpy(benchmark, y_ndim):
    xq, x, y = _make_case(np, y_ndim)

    benchmark(lambda: interp1d(xq, x, y))


@pytest.mark.skipif(jax is None, reason="JAX not installed")
@pytest.mark.parametrize("y_ndim", [1, 2], ids=["y-1d", "y-2d"])
def test_interp1d_jax(benchmark, y_ndim):
    xq, x, y = _make_case(jnp, y_ndim)

    # Warmup: compile is not benchmarked
    _block_until_ready(interp1d(xq, x, y))

    benchmark(lambda: _block_until_ready(interp1d(xq, x, y)))


@pytest.mark.skipif(cp is None, reason="CuPy is not available")
@pytest.mark.parametrize("y_ndim", [1, 2], ids=["y-1d", "y-2d"])
def test_interp1d_cupy(benchmark, y_ndim):
    xq, x, y = _make_case(cp, y_ndim)

    # Warmup
    _block_until_ready(interp1d(xq, x, y))

    benchmark(lambda: _block_until_ready(interp1d(xq, x, y)))


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@pytest.mark.parametrize("y_ndim", [1, 2], ids=["y-1d", "y-2d"])
def test_interp1d_torch_cpu(benchmark, y_ndim):
    xq, x, y = _make_case(torch, y_ndim)

    # Warmup: torch.compile is not benchmarked
    _block_until_ready(interp1d(xq, x, y))

    benchmark(lambda: _block_until_ready(interp1d(xq, x, y)))


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="PyTorch CUDA is not available",
)
@pytest.mark.parametrize("y_ndim", [1, 2], ids=["y-1d", "y-2d"])
def test_interp1d_torch_cuda(benchmark, y_ndim):
    xq, x, y = _make_case(torch, y_ndim)

    xq = xq.cuda()
    x = x.cuda()
    y = y.cuda()

    # Warmup: torch.compile / CUDA dispatch is not benchmarked
    _block_until_ready(interp1d(xq, x, y))

    benchmark(lambda: _block_until_ready(interp1d(xq, x, y)))


@pytest.mark.parametrize("stack", [False, True], ids=["list", "list-stack"])
def test_interp1d_numpy_sequence(benchmark, stack):
    xq, x, y = _make_sequence_case(np)

    benchmark(lambda: interp1d(xq, x, y, stack=stack))

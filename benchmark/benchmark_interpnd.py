"""Benchmark interpnd with different backends."""

import numpy as np
import pytest

from pyRadPlan.core.xp_utils.compat import interpnd


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
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_numpy(benchmark, ndim):
    xq, x, y = _make_case(np, ndim)

    benchmark(lambda: interpnd(xq, x, y))


@pytest.mark.skipif(jax is None, reason="JAX not installed")
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_jax(benchmark, ndim):
    xq, x, y = _make_case(jnp, ndim)

    # Warmup: compile is not benchmarked
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(cp is None, reason="CuPy is not available")
@pytest.mark.parametrize("ndim", [2, 3, 4], ids=["2d", "3d", "4d"])
def test_interpnd_cupy(benchmark, ndim):
    xq, x, y = _make_case(cp, ndim)

    # warmup
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(torch is None, reason="PyTorch is not available")
@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_interpnd_torch_cpu(benchmark, ndim):
    xq, x, y = _make_case(torch, ndim)

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))


@pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(),
    reason="PyTorch CUDA is not available",
)
@pytest.mark.parametrize("ndim", [2, 3], ids=["2d", "3d"])
def test_interpnd_torch_cuda(benchmark, ndim):
    xq_np, x_np, y_np = _make_case(np, ndim)

    x = tuple(torch.asarray(axis, dtype=torch.float64, device="cuda") for axis in x_np)
    y = torch.asarray(y_np, dtype=torch.float64, device="cuda")
    xq = torch.asarray(xq_np, dtype=torch.float64, device="cuda")

    # warmup
    _block_until_ready(interpnd(xq, x, y))

    benchmark(lambda: _block_until_ready(interpnd(xq, x, y)))

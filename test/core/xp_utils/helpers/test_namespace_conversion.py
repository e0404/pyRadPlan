import pytest
import importlib
import array_api_compat
import array_api_strict
from array_api_compat import numpy as np
import scipy.sparse as sp
from pyRadPlan.core.xp_utils import to_namespace, from_numpy

# Check for optional dependencies
try:
    import cupy as cp
    import cupyx.scipy.sparse as csp

    has_cupy = True
except ImportError:
    cp = None
    csp = None
    has_cupy = False

try:
    import torch

    has_torch = True
except ImportError:
    torch = None
    has_torch = False

try:
    import jax
    import jax.numpy as jnp
    from jax.experimental import sparse as jsparse

    has_jax = True
except ImportError:
    jax = None
    jnp = None
    jsparse = None
    has_jax = False


def _jax_gpu_available():
    if not has_jax:
        return False
    try:
        return len(jax.devices("gpu")) > 0
    except Exception:
        return False


def _cupy_gpu_available():
    if not has_cupy:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


@pytest.fixture
def numpy_array():
    return np.array([[1, 2, 3], [4, 5, 6]])


@pytest.fixture
def array_api_array(numpy_array):
    return array_api_strict.asarray(numpy_array)


def test_to_namespace_numpy_to_array_api(numpy_array):
    result = to_namespace(array_api_strict, numpy_array)
    assert array_api_compat.is_array_api_strict_namespace(result.__array_namespace__())
    assert np.array_equal(result, numpy_array)


def test_to_namespace_array_api_to_numpy(array_api_array):
    result = to_namespace(np, array_api_array)
    assert array_api_compat.is_numpy_array(result)
    assert np.array_equal(result, array_api_array)


def test_to_namespace_no_conversion(array_api_array):
    result = to_namespace(array_api_strict, array_api_array)
    assert result is array_api_array  # No conversion should occur


# TODO: In the future we should be working with arrays everywhere (?)


# --- Torch Tests ---
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
def test_numpy_to_torch(numpy_array):
    result = to_namespace(torch, numpy_array, device="cpu")
    assert isinstance(result, torch.Tensor)
    assert not result.is_cuda
    assert np.array_equal(result.numpy(), numpy_array)


@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
def test_torch_to_numpy(numpy_array):
    t_arr = torch.from_numpy(numpy_array)
    result = to_namespace(np, t_arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
def test_torch_device_argument(numpy_array):
    # Test explicit CPU
    result = to_namespace(torch, numpy_array, device="cpu")
    assert result.device.type == "cpu"

    if torch.cuda.is_available():
        # Test explicit GPU
        result_gpu = to_namespace(torch, numpy_array, device="gpu")
        assert result_gpu.is_cuda

        # Test move back to CPU
        result_cpu = to_namespace(torch, result_gpu, device="cpu")
        assert not result_cpu.is_cuda


# --- CuPy Tests ---
@pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
def test_numpy_to_cupy(numpy_array):
    result = to_namespace(cp, numpy_array)
    assert isinstance(result, cp.ndarray)
    assert cp.asnumpy(result).tolist() == numpy_array.tolist()


@pytest.mark.skipif(not has_cupy, reason="CuPy not installed")
def test_cupy_to_numpy(numpy_array):
    c_arr = cp.asarray(numpy_array)
    result = to_namespace(np, c_arr)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, numpy_array)


# --- Cross-GPU Tests (Torch <-> CuPy) ---
@pytest.mark.skipif(
    not (has_cupy and has_torch and torch.cuda.is_available()),
    reason="CuPy and PyTorch with CUDA required",
)
def test_cupy_to_torch_gpu(numpy_array):
    c_arr = cp.asarray(numpy_array)
    result = to_namespace(torch, c_arr)  # Should default to GPU
    assert isinstance(result, torch.Tensor)
    assert result.is_cuda
    assert np.array_equal(result.cpu().numpy(), numpy_array)


@pytest.mark.skipif(
    not (has_cupy and has_torch and torch.cuda.is_available()),
    reason="CuPy and PyTorch with CUDA required",
)
def test_torch_gpu_to_cupy(numpy_array):
    t_arr = torch.from_numpy(numpy_array).cuda()
    result = to_namespace(cp, t_arr)
    assert isinstance(result, cp.ndarray)
    assert cp.asnumpy(result).tolist() == numpy_array.tolist()


# --- Sparse Tests ---
@pytest.fixture
def scipy_sparse_matrix():
    return sp.coo_matrix([[0.0, 1.0], [1.0, 0.0]])


def test_scipy_sparse_to_numpy(scipy_sparse_matrix):
    # Should remain sparse if keep_sparse_compat=True (default)
    result = to_namespace(np, scipy_sparse_matrix)
    assert sp.issparse(result)

    # Should become dense if keep_sparse_compat=False
    result_dense = to_namespace(np, scipy_sparse_matrix, keep_sparse_compat=False)
    assert isinstance(result_dense, np.ndarray)


@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
def test_scipy_sparse_to_torch(scipy_sparse_matrix):
    result = to_namespace(torch, scipy_sparse_matrix, device="cpu")
    assert result.is_sparse
    assert not result.is_cuda


@pytest.mark.skipif(not has_jax, reason="JAX not installed")
@pytest.mark.parametrize(
    ("fmt", "jax_type"),
    (("coo", "COO"), ("csr", "CSR"), ("csc", "CSC")),
)
def test_scipy_jax_sparse_roundtrip_preserves_format(fmt, jax_type):
    source = getattr(sp, f"{fmt}_array")([[0.0, 1.0], [2.0, 0.0]])

    jax_sparse = to_namespace(jnp, source)
    result = to_namespace(np, jax_sparse, device="cpu")

    assert isinstance(jax_sparse, getattr(jsparse, jax_type))
    assert result.format == fmt
    assert np.array_equal(result.toarray(), source.toarray())


@pytest.mark.skipif(not (has_jax and has_torch), reason="JAX and PyTorch required")
@pytest.mark.parametrize(
    ("fmt", "jax_type", "torch_layout"),
    (
        ("coo", "COO", "sparse_coo"),
        ("csr", "CSR", "sparse_csr"),
        ("csc", "CSC", "sparse_csc"),
    ),
)
def test_torch_jax_sparse_cpu_roundtrip_preserves_format(fmt, jax_type, torch_layout):
    source = getattr(sp, f"{fmt}_array")([[0.0, 1.0], [2.0, 0.0]])
    torch_sparse = to_namespace(torch, source, device="cpu")

    jax_sparse = to_namespace(jnp, torch_sparse, device="cpu")
    result = to_namespace(torch, jax_sparse, device="cpu")

    assert isinstance(jax_sparse, getattr(jsparse, jax_type))
    assert result.layout == getattr(torch, torch_layout)
    assert np.array_equal(result.to_dense().numpy(), source.toarray())


@pytest.mark.skipif(
    not (has_jax and has_cupy and _jax_gpu_available() and _cupy_gpu_available()),
    reason="JAX and CuPy GPU backends required",
)
@pytest.mark.parametrize(
    ("fmt", "jax_type"),
    (("coo", "COO"), ("csr", "CSR"), ("csc", "CSC")),
)
def test_cupy_jax_sparse_gpu_roundtrip_preserves_format(fmt, jax_type):
    source_scipy = getattr(sp, f"{fmt}_array")([[0.0, 1.0], [2.0, 0.0]])
    source = getattr(csp, f"{fmt}_matrix")(source_scipy)

    jax_sparse = to_namespace(jnp, source, device="gpu")
    result = to_namespace(cp, jax_sparse, device="gpu")

    assert isinstance(jax_sparse, getattr(jsparse, jax_type))
    assert result.getformat() == fmt
    assert np.array_equal(result.get().toarray(), source_scipy.toarray())


@pytest.mark.skipif(
    not (has_cupy and has_torch and torch.cuda.is_available()),
    reason="CuPy and PyTorch with CUDA required",
)
def test_cupy_sparse_to_torch_gpu(scipy_sparse_matrix):
    c_sp = csp.coo_matrix(scipy_sparse_matrix)
    result = to_namespace(torch, c_sp, device="gpu")
    assert result.is_sparse
    assert result.is_cuda


# --- from_numpy with device ---
@pytest.mark.skipif(not has_torch, reason="PyTorch not installed")
def test_from_numpy_torch_cpu(numpy_array):
    import array_api_compat.torch as xp

    result = from_numpy(xp, numpy_array, device="cpu")
    assert isinstance(result, torch.Tensor)
    assert result.device.type == "cpu"


@pytest.mark.skipif(
    not (has_torch and torch.cuda.is_available()), reason="PyTorch GPU not available"
)
def test_from_numpy_torch_gpu(numpy_array):
    import array_api_compat.torch as xp

    result = from_numpy(xp, numpy_array, device="cuda:0")
    assert isinstance(result, torch.Tensor)
    assert result.is_cuda
    assert result.device.index == 0


# --- to_namespace multi-GPU (device index) ---
@pytest.mark.skipif(
    not (has_torch and torch.cuda.is_available() and torch.cuda.device_count() > 1),
    reason="Multiple CUDA GPUs required",
)
def test_to_namespace_torch_specific_gpu(numpy_array):
    result = to_namespace(torch, numpy_array, device="cuda:1")
    assert result.is_cuda
    assert result.device.index == 1


@pytest.mark.skipif(
    not (has_torch and torch.cuda.is_available()), reason="PyTorch with CUDA required"
)
def test_to_namespace_default_device_keeps_source_and_honors_prefer_gpu(numpy_array):
    """Without an explicit device, arrays stay where they are.

    The GPU is only used as the fallback default when ``settings.xp.prefer_gpu`` allows it.
    """
    from pyRadPlan import settings

    prefer_gpu = settings.xp.prefer_gpu
    try:
        settings.xp.prefer_gpu = False
        assert to_namespace(torch, numpy_array).device.type == "cpu"
        assert to_namespace(torch, torch.ones(3)).device.type == "cpu"
        assert to_namespace(torch, torch.ones(3, device="cuda")).device.type == "cuda"

        settings.xp.prefer_gpu = True
        # Source device is still preserved; the setting only governs the fallback default
        assert to_namespace(torch, numpy_array).device.type == "cpu"
        assert to_namespace(torch, torch.ones(3, device="cuda")).device.type == "cuda"
    finally:
        settings.xp.prefer_gpu = prefer_gpu

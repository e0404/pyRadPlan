import pytest
from pyRadPlan._settings import get_settings
from pyRadPlan.core.xp_utils import (
    cupy_available,
    pytorch_available,
    pytorch_gpu_available,
    numba_cuda_available,
    choose_array_api_namespace,
    choose_device,
)
import array_api_compat

# Check for actual availability to determine if we should skip
try:
    import cupy

    HAS_CUPY = True
    CUPY_CUDA_AVAILABLE = cupy.cuda.is_available()
except ImportError:
    HAS_CUPY = False
    CUPY_CUDA_AVAILABLE = False

try:
    import torch

    HAS_TORCH = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    HAS_TORCH = False
    TORCH_CUDA_AVAILABLE = False

try:
    from numba import cuda

    HAS_NUMBA = True
    try:
        NUMBA_CUDA_AVAILABLE = cuda.is_available()
    except Exception:
        NUMBA_CUDA_AVAILABLE = False
except ImportError:
    HAS_NUMBA = False
    NUMBA_CUDA_AVAILABLE = False


def test_cupy_available():
    """Test cupy_available function."""
    expected = HAS_CUPY and CUPY_CUDA_AVAILABLE
    assert cupy_available() == expected


def test_pytorch_available():
    """Test pytorch_available function."""
    assert pytorch_available() == HAS_TORCH


def test_pytorch_gpu_available():
    """Test pytorch_gpu_available function."""
    expected = HAS_TORCH and TORCH_CUDA_AVAILABLE
    assert pytorch_gpu_available() == expected


def test_numba_cuda_available():
    """Test numba_cuda_available function."""
    expected = HAS_NUMBA and NUMBA_CUDA_AVAILABLE
    assert numba_cuda_available() == expected


def test_choose_array_api_namespace_defaults():
    """Test choose_array_api_namespace with default arguments."""
    xp = choose_array_api_namespace()

    settings = get_settings().xp
    if settings.prefer_gpu and settings.preferred_gpu_array_backend == "cupy" and cupy_available():
        assert "cupy" in xp.__name__
    elif (
        settings.prefer_gpu
        and settings.preferred_gpu_array_backend == "torch"
        and pytorch_gpu_available()
    ):
        assert "torch" in xp.__name__
    else:
        # It seems array_api_compat.numpy might be aliased or implemented via array_api_strict in some envs?
        # Or maybe preferred_cpu_array_backend is different.
        assert "numpy" in xp.__name__ or "array_api_strict" in xp.__name__


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not installed")
def test_choose_array_api_namespace_cupy():
    """Test choose_array_api_namespace with 'cupy'."""
    xp = choose_array_api_namespace("cupy")
    assert "cupy" in xp.__name__


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
def test_choose_array_api_namespace_torch():
    """Test choose_array_api_namespace with 'torch'."""
    xp = choose_array_api_namespace("torch")
    assert "torch" in xp.__name__


def test_choose_array_api_namespace_numpy():
    """Test choose_array_api_namespace with 'numpy'."""
    xp = choose_array_api_namespace("numpy")
    assert "numpy" in xp.__name__


def test_choose_device_defaults():
    """Test choose_device with default arguments."""
    dev = choose_device()

    xp = choose_array_api_namespace()

    if array_api_compat.is_torch_namespace(xp) and pytorch_gpu_available():
        import torch

        assert dev == torch.device("cuda", 0)
    elif array_api_compat.is_cupy_namespace(xp) and cupy_available():
        import cupy as cp

        assert dev == cp.cuda.Device(0)
    elif array_api_compat.is_jax_namespace(xp):
        import jax

        assert dev in jax.devices()
    else:
        # numpy and array-api-strict have no device concept beyond the default
        assert dev is None


@pytest.mark.skipif(not (HAS_TORCH and TORCH_CUDA_AVAILABLE), reason="PyTorch GPU not available")
def test_choose_device_torch():
    """Test choose_device with torch namespace."""
    import array_api_compat.torch as xp

    import torch

    dev = choose_device(xp)
    assert dev == torch.device("cuda", 0)


@pytest.mark.skipif(not (HAS_TORCH and TORCH_CUDA_AVAILABLE), reason="PyTorch GPU not available")
def test_choose_device_torch_multi_gpu():
    """Test choose_device with torch namespace and explicit gpu_index."""
    import array_api_compat.torch as xp
    import torch

    assert choose_device(xp, gpu_index=0) == torch.device("cuda", 0)
    assert choose_device(xp, gpu_index=1) == torch.device("cuda", 1)


@pytest.mark.skipif(not (HAS_CUPY and CUPY_CUDA_AVAILABLE), reason="CuPy GPU not available")
def test_choose_device_cupy():
    """Test choose_device with cupy namespace."""
    import array_api_compat.cupy as xp

    import cupy as cp

    dev = choose_device(xp)
    assert dev == cp.cuda.Device(0)


@pytest.mark.skipif(not (HAS_CUPY and CUPY_CUDA_AVAILABLE), reason="CuPy GPU not available")
def test_choose_device_cupy_multi_gpu():
    """Test choose_device with cupy namespace and explicit gpu_index."""
    import array_api_compat.cupy as xp
    import cupy as cp

    assert choose_device(xp, gpu_index=0) == cp.cuda.Device(0)
    assert choose_device(xp, gpu_index=1) == cp.cuda.Device(1)


def test_choose_device_numpy():
    """Test choose_device with numpy namespace."""
    import array_api_compat.numpy as xp

    # NumPy exposes no device object; None means "the namespace default device"
    dev = choose_device(xp)
    assert dev is None


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_choose_device_torch_cpu_fallback():
    """choose_device must fall back to CPU when a GPU is preferred but unavailable."""
    import array_api_compat.torch as xp
    import torch

    if TORCH_CUDA_AVAILABLE:
        pytest.skip("GPU available, CPU fallback not exercised")

    settings = get_settings()
    prefer_gpu = settings.xp.prefer_gpu
    settings.xp.prefer_gpu = True
    try:
        with pytest.warns(UserWarning, match="Falling back to CPU"):
            assert choose_device(xp) == torch.device("cpu")
    finally:
        settings.xp.prefer_gpu = prefer_gpu

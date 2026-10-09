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

    if settings.prefer_gpu and settings.preferred_gpu_array_backend is not None:
        # An explicitly configured GPU backend is taken as-is (test/conftest.py pins it to
        # array_api_strict so the suite exercises this branch without a GPU)
        expected = settings.preferred_gpu_array_backend
    elif settings.prefer_gpu and (cupy_available() or pytorch_gpu_available()):
        # Auto-detected in the order cupy, torch, jax
        expected = "cupy" if cupy_available() else "torch"
    else:
        # Whatever CPU backend the settings ask for -- numpy by default, but the backend CI
        # jobs override it via PYRADPLAN_XP_PREFERRED_CPU_ARRAY_BACKEND
        expected = settings.preferred_cpu_array_backend

    assert expected in xp.__name__


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
    prefer_gpu = get_settings().xp.prefer_gpu

    if array_api_compat.is_torch_namespace(xp):
        import torch

        # torch runs on either device, so which one is picked follows prefer_gpu
        if prefer_gpu and pytorch_gpu_available():
            assert dev == torch.device("cuda", 0)
        else:
            assert dev == torch.device("cpu")
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

    # choose_device only reaches for a GPU when prefer_gpu is set
    if get_settings().xp.prefer_gpu:
        assert dev == torch.device("cuda", 0)
    else:
        assert dev == torch.device("cpu")


@pytest.mark.skipif(not (HAS_TORCH and TORCH_CUDA_AVAILABLE), reason="PyTorch GPU not available")
def test_choose_device_torch_multi_gpu():
    """Test choose_device with torch namespace and explicit gpu_index."""
    import array_api_compat.torch as xp
    import torch

    if not get_settings().xp.prefer_gpu:
        pytest.skip("gpu_index only takes effect when prefer_gpu is set")

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

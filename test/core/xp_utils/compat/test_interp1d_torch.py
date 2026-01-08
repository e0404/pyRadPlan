import pytest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from pyRadPlan.core.xp_utils.compat import interp1d


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
class TestInterp1dTorch:
    """Tests specifically for PyTorch backend in interp1d, covering optimizations."""

    def test_basic_interp(self):
        x = torch.linspace(0, 10, 11)
        y = torch.sin(x)
        xq = torch.linspace(0, 10, 101)

        res = interp1d(xq, x, y)

        expected = np.interp(xq.numpy(), x.numpy(), y.numpy())
        np.testing.assert_allclose(res.numpy(), expected, atol=1e-5)

    def test_list_optimization(self):
        """Test the list input optimization (should use vectorized JIT kernel)."""
        x = torch.linspace(0, 10, 11)
        # Create a list of y arrays
        ys = [torch.sin(x) * i for i in range(1, 4)]
        xq = torch.linspace(0, 10, 101)

        # Test default stack=False behavior (should return list)
        res_list = interp1d(xq, x, ys, stack=False)
        assert isinstance(res_list, list)
        assert len(res_list) == 3

        for i, res in enumerate(res_list):
            expected = np.interp(xq.numpy(), x.numpy(), ys[i].numpy())
            np.testing.assert_allclose(res.numpy(), expected, atol=1e-5)

    def test_dict_optimization(self):
        """Test the dict input optimization (should use vectorized JIT kernel)."""
        x = torch.linspace(0, 10, 11)
        # Create a dict of y arrays
        ys = {f"k{i}": torch.sin(x) * i for i in range(1, 4)}
        xq = torch.linspace(0, 10, 101)

        res_dict = interp1d(xq, x, ys)
        assert isinstance(res_dict, dict)
        assert len(res_dict) == 3

        for k, res in res_dict.items():
            expected = np.interp(xq.numpy(), x.numpy(), ys[k].numpy())
            np.testing.assert_allclose(res.numpy(), expected, atol=1e-5)

    def test_stack_true(self):
        """Test stack=True behavior with list input."""
        x = torch.linspace(0, 10, 11)
        ys = [torch.sin(x) * i for i in range(1, 4)]
        xq = torch.linspace(0, 10, 101)

        # Test stack=True (should return stacked tensor)
        res_stacked = interp1d(xq, x, ys, stack=True)
        assert isinstance(res_stacked, torch.Tensor)
        assert res_stacked.shape == (3, 101)

        for i in range(3):
            expected = np.interp(xq.numpy(), x.numpy(), ys[i].numpy())
            np.testing.assert_allclose(res_stacked[i].numpy(), expected, atol=1e-5)

    def test_stack_true_dict(self):
        """Test stack=True behavior with dict input."""
        x = torch.linspace(0, 10, 11)
        ys = {f"k{i}": torch.sin(x) * i for i in range(1, 4)}
        xq = torch.linspace(0, 10, 101)

        res_stacked = interp1d(xq, x, ys, stack=True)
        assert isinstance(res_stacked, torch.Tensor)
        assert res_stacked.shape == (3, 101)

        keys = list(ys.keys())
        for i, k in enumerate(keys):
            expected = np.interp(xq.numpy(), x.numpy(), ys[k].numpy())
            np.testing.assert_allclose(res_stacked[i].numpy(), expected, atol=1e-5)

    def test_clipping(self):
        """Test that out-of-bounds query points are clamped to the range of x."""
        x = torch.tensor([0.0, 1.0, 2.0])
        y = torch.tensor([0.0, 1.0, 0.0])
        xq = torch.tensor([-1.0, 0.5, 3.0])

        res = interp1d(xq, x, y)

        # xq clipped becomes [0.0, 0.5, 2.0]
        # interp at 0.0 is 0.0
        # interp at 0.5 is 0.5
        # interp at 2.0 is 0.0
        expected = np.array([0.0, 0.5, 0.0])

        np.testing.assert_allclose(res.numpy(), expected, atol=1e-5)

    def test_dtype_consistency(self):
        """Test that implementation handles float32/float64 correctly."""
        for dtype in [torch.float32, torch.float64]:
            x = torch.linspace(0, 10, 11, dtype=dtype)
            y = torch.sin(x)
            xq = torch.linspace(0, 10, 101, dtype=dtype)

            res = interp1d(xq, x, y)
            assert res.dtype == dtype
            assert res.shape == xq.shape

    def test_gpu_device(self):
        """Test if it runs on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        x = torch.linspace(0, 10, 11, device=device)
        y = torch.sin(x)
        xq = torch.linspace(0, 10, 101, device=device)

        res = interp1d(xq, x, y)
        assert res.device.type == "cuda"

        expected = np.interp(xq.cpu().numpy(), x.cpu().numpy(), y.cpu().numpy())
        np.testing.assert_allclose(res.cpu().numpy(), expected, atol=1e-5)

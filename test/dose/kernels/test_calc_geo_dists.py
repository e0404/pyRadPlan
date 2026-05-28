"""Tests for _calc_geo_dists kernels."""

import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, pytorch_gpu_available


def _reference_calc_geo_dists(coords, rot_mat, source_point, lateral_cutoff_sq_over_sad_sq):
    """
    NumPy reference implementation for validating GPU kernels.

    Parameters
    ----------
    coords : np.ndarray
        (N, 3) array of coordinates.
    rot_mat : np.ndarray
        (3, 3) rotation matrix.
    source_point : np.ndarray
        (3,) source point.
    lateral_cutoff_sq_over_sad_sq : float
        Squared lateral cutoff divided by squared SAD.

    Returns
    -------
    out_coords_temp : np.ndarray
        (N, 3) rotated coordinates.
    out_lat_dists : np.ndarray
        (N, 2) lateral distances (x, z).
    out_rad_dist_sq : np.ndarray
        (N,) squared radial distances.
    out_mask : np.ndarray
        (N,) boolean mask.
    """
    # Matrix multiplication: coords @ rot_mat
    out_coords_temp = coords @ rot_mat

    # Lateral distances
    lx = out_coords_temp[:, 0] + source_point[0]
    lz = out_coords_temp[:, 2] + source_point[2]
    out_lat_dists = np.stack((lx, lz), axis=1)

    # Radial distance squared
    out_rad_dist_sq = lx * lx + lz * lz

    # Mask
    ry = out_coords_temp[:, 1]
    limit = lateral_cutoff_sq_over_sad_sq * ry * ry
    out_mask = out_rad_dist_sq <= limit

    return out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask


@pytest.fixture
def sample_data():
    """Generate sample test data."""
    np.random.seed(42)
    n_points = 1000

    # Random coordinates
    coords = np.random.randn(n_points, 3).astype(np.float64) * 100

    # Random rotation matrix (orthonormal)
    q, _ = np.linalg.qr(np.random.randn(3, 3))
    rot_mat = q.astype(np.float64)

    # Source point
    source_point = np.array([0.0, -1000.0, 0.0], dtype=np.float64)

    # Lateral cutoff parameters
    lateral_cutoff = 50.0
    sad = 1000.0
    lateral_cutoff_sq_over_sad_sq = (lateral_cutoff / sad) ** 2

    return {
        "coords": coords,
        "rot_mat": rot_mat,
        "source_point": source_point,
        "lateral_cutoff_sq_over_sad_sq": lateral_cutoff_sq_over_sad_sq,
    }


@pytest.fixture
def identity_rotation_data():
    """Test data with identity rotation for easier verification."""
    n_points = 100
    coords = np.random.randn(n_points, 3).astype(np.float64) * 50
    rot_mat = np.eye(3, dtype=np.float64)
    source_point = np.array([0.0, -500.0, 0.0], dtype=np.float64)
    lateral_cutoff_sq_over_sad_sq = (30.0 / 500.0) ** 2

    return {
        "coords": coords,
        "rot_mat": rot_mat,
        "source_point": source_point,
        "lateral_cutoff_sq_over_sad_sq": lateral_cutoff_sq_over_sad_sq,
    }


# =============================================================================
# CuPy Kernel Tests
# =============================================================================


@pytest.mark.skipif(not cupy_available(), reason="CuPy is not available")
class TestCalcGeoDIstsCuPyKernel:
    """Tests for the CuPy ElementwiseKernel implementation."""

    def test_cupy_kernel_basic(self, sample_data):
        """Test CuPy kernel produces correct results."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_kernel

        # Convert to CuPy arrays
        coords = cp.asarray(sample_data["coords"])
        rot_mat = cp.asarray(sample_data["rot_mat"])
        source_point = cp.asarray(sample_data["source_point"])
        lateral_cutoff_sq_over_sad_sq = sample_data["lateral_cutoff_sq_over_sad_sq"]

        # Run CuPy kernel
        num_elements = coords.shape[0]
        (
            out_coords_x,
            out_coords_y,
            out_coords_z,
            out_lat_x,
            out_lat_z,
            out_rad_dist_sq,
            out_mask,
        ) = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            lateral_cutoff_sq_over_sad_sq,
            size=num_elements,
        )

        # Stack outputs
        out_coords_temp = cp.stack((out_coords_x, out_coords_y, out_coords_z), axis=1)
        out_lat_dists = cp.stack((out_lat_x, out_lat_z), axis=1)

        # Get reference results
        ref_coords, ref_lat_dists, ref_rad_dist_sq, ref_mask = _reference_calc_geo_dists(
            sample_data["coords"],
            sample_data["rot_mat"],
            sample_data["source_point"],
            lateral_cutoff_sq_over_sad_sq,
        )

        # Compare results
        np.testing.assert_allclose(cp.asnumpy(out_coords_temp), ref_coords, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            cp.asnumpy(out_lat_dists), ref_lat_dists, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            cp.asnumpy(out_rad_dist_sq), ref_rad_dist_sq, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(cp.asnumpy(out_mask), ref_mask)

    def test_cupy_kernel_identity_rotation(self, identity_rotation_data):
        """Test CuPy kernel with identity rotation matrix."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_kernel

        coords = cp.asarray(identity_rotation_data["coords"])
        rot_mat = cp.asarray(identity_rotation_data["rot_mat"])
        source_point = cp.asarray(identity_rotation_data["source_point"])
        lateral_cutoff_sq_over_sad_sq = identity_rotation_data["lateral_cutoff_sq_over_sad_sq"]

        num_elements = coords.shape[0]
        out_coords_x, out_coords_y, out_coords_z, _, _, _, _ = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            lateral_cutoff_sq_over_sad_sq,
            size=num_elements,
        )

        out_coords_temp = cp.stack((out_coords_x, out_coords_y, out_coords_z), axis=1)

        # With identity rotation, output coords should equal input coords
        np.testing.assert_allclose(
            cp.asnumpy(out_coords_temp),
            identity_rotation_data["coords"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_cupy_kernel_empty_input(self):
        """Test CuPy kernel with empty input."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_kernel

        coords = cp.empty((0, 3), dtype=np.float64)
        rot_mat = cp.eye(3, dtype=np.float64)
        source_point = cp.zeros(3, dtype=np.float64)

        (
            out_coords_x,
            out_coords_y,
            out_coords_z,
            out_lat_x,
            out_lat_z,
            out_rad_dist_sq,
            out_mask,
        ) = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            0.01,
            size=0,
        )

        assert out_coords_x.shape[0] == 0
        assert out_mask.shape[0] == 0

    def test_cupy_kernel_single_point(self):
        """Test CuPy kernel with a single point."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_kernel

        coords = cp.array([[10.0, 20.0, 30.0]], dtype=np.float64)
        rot_mat = cp.eye(3, dtype=np.float64)
        source_point = cp.array([0.0, -100.0, 0.0], dtype=np.float64)
        lateral_cutoff_sq_over_sad_sq = (50.0 / 100.0) ** 2

        (
            out_coords_x,
            out_coords_y,
            out_coords_z,
            out_lat_x,
            out_lat_z,
            out_rad_dist_sq,
            out_mask,
        ) = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            lateral_cutoff_sq_over_sad_sq,
            size=1,
        )

        # With identity rotation: out_coords = coords
        assert float(out_coords_x[0]) == pytest.approx(10.0)
        assert float(out_coords_y[0]) == pytest.approx(20.0)
        assert float(out_coords_z[0]) == pytest.approx(30.0)

        # lat_x = rx + source_point[0] = 10 + 0 = 10
        # lat_z = rz + source_point[2] = 30 + 0 = 30
        assert float(out_lat_x[0]) == pytest.approx(10.0)
        assert float(out_lat_z[0]) == pytest.approx(30.0)

        # rad_dist_sq = 10^2 + 30^2 = 100 + 900 = 1000
        assert float(out_rad_dist_sq[0]) == pytest.approx(1000.0)

    def test_cupy_kernel_float32(self, sample_data):
        """Test CuPy kernel with float32 data."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_kernel

        coords = cp.asarray(sample_data["coords"].astype(np.float32))
        rot_mat = cp.asarray(sample_data["rot_mat"].astype(np.float32))
        source_point = cp.asarray(sample_data["source_point"].astype(np.float32))

        num_elements = coords.shape[0]
        out_coords_x, _, _, _, _, _, _ = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            sample_data["lateral_cutoff_sq_over_sad_sq"],
            size=num_elements,
        )

        # Check output dtype matches input
        assert out_coords_x.dtype == np.float32


# =============================================================================
# CuPy RawKernel Tests
# =============================================================================


@pytest.mark.skipif(not cupy_available(), reason="CuPy is not available")
class TestCalcGeoDIstsCuPyRawKernel:
    """Tests for the CuPy RawKernel implementation (Numba-like calling convention)."""

    def test_cupy_raw_kernel_basic(self, sample_data):
        """Test CuPy RawKernel produces correct results."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_raw_kernel

        # Convert to CuPy arrays
        coords = cp.asarray(sample_data["coords"], dtype=cp.float64)
        rot_mat = cp.asarray(sample_data["rot_mat"], dtype=cp.float64)
        source_point = cp.asarray(sample_data["source_point"], dtype=cp.float64)
        lateral_cutoff_sq_over_sad_sq = sample_data["lateral_cutoff_sq_over_sad_sq"]

        num_elements = coords.shape[0]

        # Pre-allocate output arrays (Numba-style)
        out_coords_temp = cp.empty_like(coords)
        out_lat_dists = cp.empty((num_elements, 2), dtype=cp.float64)
        out_rad_dist_sq = cp.empty((num_elements,), dtype=cp.float64)
        out_mask = cp.empty((num_elements,), dtype=bool)

        # Launch kernel (Numba-like calling convention)
        threadsperblock = 256
        blockspergrid = (num_elements + threadsperblock - 1) // threadsperblock

        _calc_geo_dists_cupy_raw_kernel(
            (blockspergrid,),
            (threadsperblock,),
            (
                coords,
                rot_mat,
                source_point,
                lateral_cutoff_sq_over_sad_sq,
                out_coords_temp,
                out_lat_dists,
                out_rad_dist_sq,
                out_mask,
                num_elements,
            ),
        )

        # Get reference results
        ref_coords, ref_lat_dists, ref_rad_dist_sq, ref_mask = _reference_calc_geo_dists(
            sample_data["coords"],
            sample_data["rot_mat"],
            sample_data["source_point"],
            lateral_cutoff_sq_over_sad_sq,
        )

        # Compare results
        np.testing.assert_allclose(cp.asnumpy(out_coords_temp), ref_coords, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(
            cp.asnumpy(out_lat_dists), ref_lat_dists, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            cp.asnumpy(out_rad_dist_sq), ref_rad_dist_sq, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(cp.asnumpy(out_mask), ref_mask)

    def test_cupy_raw_kernel_identity_rotation(self, identity_rotation_data):
        """Test CuPy RawKernel with identity rotation matrix."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_raw_kernel

        coords = cp.asarray(identity_rotation_data["coords"], dtype=cp.float64)
        rot_mat = cp.asarray(identity_rotation_data["rot_mat"], dtype=cp.float64)
        source_point = cp.asarray(identity_rotation_data["source_point"], dtype=cp.float64)
        lateral_cutoff_sq_over_sad_sq = identity_rotation_data["lateral_cutoff_sq_over_sad_sq"]

        num_elements = coords.shape[0]

        out_coords_temp = cp.empty_like(coords)
        out_lat_dists = cp.empty((num_elements, 2), dtype=cp.float64)
        out_rad_dist_sq = cp.empty((num_elements,), dtype=cp.float64)
        out_mask = cp.empty((num_elements,), dtype=bool)

        threadsperblock = 256
        blockspergrid = (num_elements + threadsperblock - 1) // threadsperblock

        _calc_geo_dists_cupy_raw_kernel(
            (blockspergrid,),
            (threadsperblock,),
            (
                coords,
                rot_mat,
                source_point,
                lateral_cutoff_sq_over_sad_sq,
                out_coords_temp,
                out_lat_dists,
                out_rad_dist_sq,
                out_mask,
                num_elements,
            ),
        )

        # With identity rotation, output coords should equal input coords
        np.testing.assert_allclose(
            cp.asnumpy(out_coords_temp),
            identity_rotation_data["coords"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_cupy_raw_kernel_single_point(self):
        """Test CuPy RawKernel with a single point."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_cupy_raw_kernel

        coords = cp.array([[10.0, 20.0, 30.0]], dtype=cp.float64)
        rot_mat = cp.eye(3, dtype=cp.float64)
        source_point = cp.array([0.0, -100.0, 0.0], dtype=cp.float64)
        lateral_cutoff_sq_over_sad_sq = (50.0 / 100.0) ** 2

        num_elements = 1
        out_coords_temp = cp.empty_like(coords)
        out_lat_dists = cp.empty((num_elements, 2), dtype=cp.float64)
        out_rad_dist_sq = cp.empty((num_elements,), dtype=cp.float64)
        out_mask = cp.empty((num_elements,), dtype=bool)

        _calc_geo_dists_cupy_raw_kernel(
            (1,),
            (1,),
            (
                coords,
                rot_mat,
                source_point,
                lateral_cutoff_sq_over_sad_sq,
                out_coords_temp,
                out_lat_dists,
                out_rad_dist_sq,
                out_mask,
                num_elements,
            ),
        )

        # With identity rotation: out_coords = coords
        np.testing.assert_allclose(cp.asnumpy(out_coords_temp), [[10.0, 20.0, 30.0]], rtol=1e-10)

        # lat_dists = [10, 30]
        np.testing.assert_allclose(cp.asnumpy(out_lat_dists), [[10.0, 30.0]], rtol=1e-10)

        # rad_dist_sq = 10^2 + 30^2 = 1000
        assert float(out_rad_dist_sq[0]) == pytest.approx(1000.0)

    def test_elementwise_vs_raw_kernel_consistency(self, sample_data):
        """Test that ElementwiseKernel and RawKernel produce the same results."""
        import cupy as cp

        from pyRadPlan.dose.engines.kernels import (
            _calc_geo_dists_cupy_kernel,
            _calc_geo_dists_cupy_raw_kernel,
        )

        coords = cp.asarray(sample_data["coords"], dtype=cp.float64)
        rot_mat = cp.asarray(sample_data["rot_mat"], dtype=cp.float64)
        source_point = cp.asarray(sample_data["source_point"], dtype=cp.float64)
        lateral_cutoff_sq_over_sad_sq = sample_data["lateral_cutoff_sq_over_sad_sq"]
        num_elements = coords.shape[0]

        # ElementwiseKernel
        (
            ew_coords_x,
            ew_coords_y,
            ew_coords_z,
            ew_lat_x,
            ew_lat_z,
            ew_rad_dist_sq,
            ew_mask,
        ) = _calc_geo_dists_cupy_kernel(
            coords.ravel(),
            rot_mat.ravel(),
            source_point,
            lateral_cutoff_sq_over_sad_sq,
            size=num_elements,
        )
        ew_coords = cp.stack((ew_coords_x, ew_coords_y, ew_coords_z), axis=1)
        ew_lat_dists = cp.stack((ew_lat_x, ew_lat_z), axis=1)

        # RawKernel
        raw_coords = cp.empty_like(coords)
        raw_lat_dists = cp.empty((num_elements, 2), dtype=cp.float64)
        raw_rad_dist_sq = cp.empty((num_elements,), dtype=cp.float64)
        raw_mask = cp.empty((num_elements,), dtype=bool)

        threadsperblock = 256
        blockspergrid = (num_elements + threadsperblock - 1) // threadsperblock

        _calc_geo_dists_cupy_raw_kernel(
            (blockspergrid,),
            (threadsperblock,),
            (
                coords,
                rot_mat,
                source_point,
                lateral_cutoff_sq_over_sad_sq,
                raw_coords,
                raw_lat_dists,
                raw_rad_dist_sq,
                raw_mask,
                num_elements,
            ),
        )

        # Compare both implementations
        np.testing.assert_allclose(cp.asnumpy(ew_coords), cp.asnumpy(raw_coords), rtol=1e-14)
        np.testing.assert_allclose(cp.asnumpy(ew_lat_dists), cp.asnumpy(raw_lat_dists), rtol=1e-14)
        np.testing.assert_allclose(
            cp.asnumpy(ew_rad_dist_sq), cp.asnumpy(raw_rad_dist_sq), rtol=1e-14
        )
        np.testing.assert_array_equal(cp.asnumpy(ew_mask), cp.asnumpy(raw_mask))


# =============================================================================
# PyTorch Kernel Tests
# =============================================================================


@pytest.mark.skipif(not pytorch_gpu_available(), reason="PyTorch with GPU is not available")
class TestCalcGeoDIstsTorchKernel:
    """Tests for the PyTorch JIT kernel implementation."""

    def test_torch_kernel_basic(self, sample_data):
        """Test PyTorch kernel produces correct results."""
        import torch

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_torch_kernel

        # Convert to PyTorch tensors on GPU
        coords = torch.tensor(sample_data["coords"], device="cuda", dtype=torch.float64)
        rot_mat = torch.tensor(sample_data["rot_mat"], device="cuda", dtype=torch.float64)
        source_point = torch.tensor(
            sample_data["source_point"], device="cuda", dtype=torch.float64
        )
        lateral_cutoff_sq_over_sad_sq = sample_data["lateral_cutoff_sq_over_sad_sq"]

        # Run PyTorch kernel
        out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = _calc_geo_dists_torch_kernel(
            coords, rot_mat, source_point, lateral_cutoff_sq_over_sad_sq
        )

        # Get reference results
        ref_coords, ref_lat_dists, ref_rad_dist_sq, ref_mask = _reference_calc_geo_dists(
            sample_data["coords"],
            sample_data["rot_mat"],
            sample_data["source_point"],
            lateral_cutoff_sq_over_sad_sq,
        )

        # Compare results
        np.testing.assert_allclose(
            out_coords_temp.cpu().numpy(), ref_coords, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            out_lat_dists.cpu().numpy(), ref_lat_dists, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            out_rad_dist_sq.cpu().numpy(), ref_rad_dist_sq, rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(out_mask.cpu().numpy(), ref_mask)

    def test_torch_kernel_identity_rotation(self, identity_rotation_data):
        """Test PyTorch kernel with identity rotation matrix."""
        import torch

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_torch_kernel

        coords = torch.tensor(identity_rotation_data["coords"], device="cuda", dtype=torch.float64)
        rot_mat = torch.tensor(
            identity_rotation_data["rot_mat"], device="cuda", dtype=torch.float64
        )
        source_point = torch.tensor(
            identity_rotation_data["source_point"], device="cuda", dtype=torch.float64
        )
        lateral_cutoff_sq_over_sad_sq = identity_rotation_data["lateral_cutoff_sq_over_sad_sq"]

        out_coords_temp, _, _, _ = _calc_geo_dists_torch_kernel(
            coords, rot_mat, source_point, lateral_cutoff_sq_over_sad_sq
        )

        # With identity rotation, output coords should equal input coords
        np.testing.assert_allclose(
            out_coords_temp.cpu().numpy(),
            identity_rotation_data["coords"],
            rtol=1e-10,
            atol=1e-10,
        )

    def test_torch_kernel_single_point(self):
        """Test PyTorch kernel with a single point."""
        import torch

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_torch_kernel

        coords = torch.tensor([[10.0, 20.0, 30.0]], device="cuda", dtype=torch.float64)
        rot_mat = torch.eye(3, device="cuda", dtype=torch.float64)
        source_point = torch.tensor([0.0, -100.0, 0.0], device="cuda", dtype=torch.float64)
        lateral_cutoff_sq_over_sad_sq = (50.0 / 100.0) ** 2

        out_coords_temp, out_lat_dists, out_rad_dist_sq, out_mask = _calc_geo_dists_torch_kernel(
            coords, rot_mat, source_point, lateral_cutoff_sq_over_sad_sq
        )

        # With identity rotation: out_coords = coords
        np.testing.assert_allclose(out_coords_temp.cpu().numpy(), [[10.0, 20.0, 30.0]], rtol=1e-10)

        # lat_dists = [10, 30]
        np.testing.assert_allclose(out_lat_dists.cpu().numpy(), [[10.0, 30.0]], rtol=1e-10)

        # rad_dist_sq = 10^2 + 30^2 = 1000
        assert float(out_rad_dist_sq[0]) == pytest.approx(1000.0)

    def test_torch_kernel_float32(self, sample_data):
        """Test PyTorch kernel with float32 data."""
        import torch

        from pyRadPlan.dose.engines.kernels import _calc_geo_dists_torch_kernel

        coords = torch.tensor(sample_data["coords"], device="cuda", dtype=torch.float32)
        rot_mat = torch.tensor(sample_data["rot_mat"], device="cuda", dtype=torch.float32)
        source_point = torch.tensor(
            sample_data["source_point"], device="cuda", dtype=torch.float32
        )

        out_coords_temp, _, _, _ = _calc_geo_dists_torch_kernel(
            coords, rot_mat, source_point, sample_data["lateral_cutoff_sq_over_sad_sq"]
        )

        # Check output dtype matches input
        assert out_coords_temp.dtype == torch.float32


# =============================================================================
# Cross-validation Tests (when both are available)
# =============================================================================


@pytest.mark.skipif(
    not (cupy_available() and pytorch_gpu_available()),
    reason="Both CuPy and PyTorch with GPU are required",
)
class TestCalcGeoDistsCrossValidation:
    """Cross-validation tests between CuPy and PyTorch implementations."""

    def test_cupy_torch_consistency(self, sample_data):
        """Test that CuPy and PyTorch kernels produce the same results."""
        import cupy as cp
        import torch

        from pyRadPlan.dose.engines.kernels import (
            _calc_geo_dists_cupy_kernel,
            _calc_geo_dists_torch_kernel,
        )

        lateral_cutoff_sq_over_sad_sq = sample_data["lateral_cutoff_sq_over_sad_sq"]

        # CuPy
        coords_cp = cp.asarray(sample_data["coords"])
        rot_mat_cp = cp.asarray(sample_data["rot_mat"])
        source_point_cp = cp.asarray(sample_data["source_point"])

        num_elements = coords_cp.shape[0]
        (
            cp_coords_x,
            cp_coords_y,
            cp_coords_z,
            cp_lat_x,
            cp_lat_z,
            cp_rad_dist_sq,
            cp_mask,
        ) = _calc_geo_dists_cupy_kernel(
            coords_cp.ravel(),
            rot_mat_cp.ravel(),
            source_point_cp,
            lateral_cutoff_sq_over_sad_sq,
            size=num_elements,
        )
        cp_coords = cp.stack((cp_coords_x, cp_coords_y, cp_coords_z), axis=1)
        cp_lat_dists = cp.stack((cp_lat_x, cp_lat_z), axis=1)

        # PyTorch
        coords_torch = torch.tensor(sample_data["coords"], device="cuda", dtype=torch.float64)
        rot_mat_torch = torch.tensor(sample_data["rot_mat"], device="cuda", dtype=torch.float64)
        source_point_torch = torch.tensor(
            sample_data["source_point"], device="cuda", dtype=torch.float64
        )

        torch_coords, torch_lat_dists, torch_rad_dist_sq, torch_mask = (
            _calc_geo_dists_torch_kernel(
                coords_torch, rot_mat_torch, source_point_torch, lateral_cutoff_sq_over_sad_sq
            )
        )

        # Compare
        np.testing.assert_allclose(
            cp.asnumpy(cp_coords), torch_coords.cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            cp.asnumpy(cp_lat_dists), torch_lat_dists.cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_allclose(
            cp.asnumpy(cp_rad_dist_sq), torch_rad_dist_sq.cpu().numpy(), rtol=1e-10, atol=1e-10
        )
        np.testing.assert_array_equal(cp.asnumpy(cp_mask), torch_mask.cpu().numpy())

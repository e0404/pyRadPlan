"""Tests for OutcomeCNN objective and OutcomeCnnPreprocessor."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import SimpleITK as sitk

try:
    import torch

    _HAS_TORCH = True
except ImportError:
    torch = None
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="PyTorch not installed")

if _HAS_TORCH:
    # Limit torch threads to avoid Windows + CUDA-built torch crashes in pytest
    torch.set_num_threads(1)

    from pyRadPlan.optimization.objectives.model.preprocessor._outcome_cnn_preprocessor import (
        OutcomeCnnPreprocessor,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal JSON config matching the densenet_example structure
DUMMY_CONFIG = {
    "model_name": "DummyModel",
    "model_preprocessing": {
        "modality": {
            "dose": {"extract": False, "normalization_value": 70.0},
            "ct": {"window": [-1000, 1000]},
            "mask": {"collaps": True},
        },
        "input_dimensions": [8, 8, 4],
        "input_spacing": [3.0, 3.0, 6.0],
        "center": "target",
    },
    "model_input": {
        "type_order": ["dose", "ct", "mask"],
    },
    "model_params": {},
}


def _make_sitk_image(size=(20, 20, 10), spacing=(2.0, 2.0, 4.0), origin=(0.0, 0.0, 0.0)):
    """Create a SimpleITK image filled with a constant HU value."""
    arr = np.zeros(size[::-1], dtype=np.float32)  # (Z, Y, X)
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


def _make_mask_image(size=(20, 20, 10), spacing=(2.0, 2.0, 4.0), origin=(0.0, 0.0, 0.0)):
    """Create a binary mask with a small sphere in the centre."""
    arr = np.zeros(size[::-1], dtype=np.uint8)
    cz, cy, cx = arr.shape[0] // 2, arr.shape[1] // 2, arr.shape[2] // 2
    arr[cz - 1 : cz + 2, cy - 1 : cy + 2, cx - 1 : cx + 2] = 1
    img = sitk.GetImageFromArray(arr)
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    return img


class FakeVOI:
    """Minimal VOI-like object for testing."""

    def __init__(self, name, mask):
        self.name = name
        self.mask = mask


class FakeGrid:
    """Minimal Grid-like object for testing."""

    def __init__(self, dimensions, origin, resolution_vector, direction):
        self.dimensions = dimensions
        self.origin = np.array(origin)
        self.resolution_vector = np.array(resolution_vector)
        self.direction = np.array(direction).reshape(3, 3)


class FakeCT:
    def __init__(self, cube_hu, grid):
        self.cube_hu = cube_hu
        self.grid = grid


class FakeCST:
    def __init__(self, vois, target_com):
        self.vois = vois
        self._target_com = target_com

    def target_center_of_mass(self):
        return self._target_com


@pytest.fixture
def ct_and_cst():
    """Create matching fake CT and CST objects."""
    size = (20, 20, 10)
    spacing = (2.0, 2.0, 4.0)
    origin = (0.0, 0.0, 0.0)

    ct_img = _make_sitk_image(size, spacing, origin)
    mask_a = _make_mask_image(size, spacing, origin)
    mask_b = _make_mask_image(size, spacing, origin)

    grid = FakeGrid(
        dimensions=size,
        origin=origin,
        resolution_vector=spacing,
        direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    ct = FakeCT(ct_img, grid)

    vois = [
        FakeVOI("parotid_lt", mask_a),
        FakeVOI("parotid_rt", mask_b),
        FakeVOI("TARGET", _make_mask_image(size, spacing, origin)),
    ]
    # Target COM near the centre of the image
    target_com = np.array(
        [
            origin[0] + spacing[0] * size[0] / 2,
            origin[1] + spacing[1] * size[1] / 2,
            origin[2] + spacing[2] * size[2] / 2,
        ]
    )
    cst = FakeCST(vois, target_com)

    return ct, cst


@pytest.fixture
def preprocessor(ct_and_cst):
    """Return a prepared preprocessor."""
    ct, cst = ct_and_cst
    pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
    pp.prepare(ct, cst, cst_masks=["parotid_lt", "parotid_rt"])
    return pp


if _HAS_TORCH:

    class DummyCNN(torch.nn.Module):
        """Trivial CNN that accepts a single concatenated input tensor."""

        def __init__(self, in_channels=3):
            super().__init__()
            self.pool = torch.nn.AdaptiveAvgPool3d(1)
            self.fc = torch.nn.Linear(in_channels, 1)

        def forward(self, x):
            # x: (B, C, Z, Y, X)
            pooled = self.pool(x).view(x.size(0), -1)
            return self.fc(pooled)


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------


@requires_torch
class TestOutcomeCnnPreprocessor:
    def test_init_parses_config(self):
        pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
        assert pp.input_dimensions == (8, 8, 4)
        assert pp.input_spacing == (3.0, 3.0, 6.0)
        assert pp.type_order == ["dose", "ct", "mask"]
        assert pp.dose_config["normalization_value"] == 70.0

    def test_physical_extent_default(self):
        """When input_physical_extent is absent, it equals spacing * dim."""
        pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
        expected = (8 * 3.0, 8 * 3.0, 4 * 6.0)
        assert pp.input_physical_extent == expected

    def test_prepare_creates_tensors(self, ct_and_cst):
        ct, cst = ct_and_cst
        pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
        pp.prepare(ct, cst, cst_masks=["parotid_lt", "parotid_rt"])

        assert pp._ct_tensor is not None
        assert pp._mask_tensor is not None
        assert pp._grid_sample_coords is not None

    def test_ct_tensor_shape(self, preprocessor):
        # (1, 1, Z, Y, X) — Z=4, Y=8, X=8 from input_dimensions
        assert preprocessor._ct_tensor.shape == (1, 1, 4, 8, 8)

    def test_ct_normalized_to_01(self, preprocessor):
        """CT values should be in [0, 1] after window normalization."""
        ct_vals = preprocessor._ct_tensor
        assert ct_vals.min() >= 0.0
        assert ct_vals.max() <= 1.0

    def test_mask_tensor_shape_collapsed(self, preprocessor):
        # collaps=True → 1 channel even though 2 masks provided
        assert preprocessor._mask_tensor.shape == (1, 1, 4, 8, 8)

    def test_mask_no_collapse(self, ct_and_cst):
        """When collaps=False, each mask is a separate channel."""
        config = {
            **DUMMY_CONFIG,
            "model_preprocessing": {
                **DUMMY_CONFIG["model_preprocessing"],
                "modality": {
                    **DUMMY_CONFIG["model_preprocessing"]["modality"],
                    "mask": {"collaps": False},
                },
            },
        }
        ct, cst = ct_and_cst
        pp = OutcomeCnnPreprocessor(config=config, device=torch.device("cpu"))
        pp.prepare(ct, cst, cst_masks=["parotid_lt", "parotid_rt"])
        assert pp._mask_tensor.shape == (1, 2, 4, 8, 8)

    def test_grid_sample_coords_shape(self, preprocessor):
        # (1, mZ, mY, mX, 3) with mZ=4, mY=8, mX=8
        assert preprocessor._grid_sample_coords.shape == (1, 4, 8, 8, 3)

    def test_build_input_no_grad(self, preprocessor):
        dose_size = preprocessor._dose_grid["size"]
        n_voxels = dose_size[0] * dose_size[1] * dose_size[2]
        dose = np.random.rand(n_voxels).astype(np.float32)

        tensor = preprocessor.build_input_tensor(dose, requires_grad=False)
        assert isinstance(tensor, torch.Tensor)
        # 3 channels: dose(1) + ct(1) + mask(1)
        assert tensor.shape == (1, 3, 4, 8, 8)

    def test_build_input_with_grad(self, preprocessor):
        dose_size = preprocessor._dose_grid["size"]
        n_voxels = dose_size[0] * dose_size[1] * dose_size[2]
        dose = np.random.rand(n_voxels).astype(np.float32)

        result = preprocessor.build_input_tensor(dose, requires_grad=True)
        assert isinstance(result, tuple)
        input_tensor, dose_leaf = result
        assert input_tensor.shape == (1, 3, 4, 8, 8)
        assert dose_leaf.requires_grad is True

    def test_dose_normalization(self, preprocessor):
        """Dose channel should be divided by normalization_value (70)."""
        dose_size = preprocessor._dose_grid["size"]
        n_voxels = dose_size[0] * dose_size[1] * dose_size[2]
        dose = np.full(n_voxels, 70.0, dtype=np.float32)

        tensor = preprocessor.build_input_tensor(dose, requires_grad=False)
        # After normalisation by 70, non-zero resampled dose should be ~1.0
        dose_channel = tensor[0, 0]  # (Z, Y, X)
        # At least some values should be close to 1.0 (those inside the grid)
        assert dose_channel.max().item() <= 1.1
        assert dose_channel.max().item() > 0.5  # not all zeros

    def test_dose_extraction(self, ct_and_cst):
        """When extract=True, dose is masked element-wise."""
        config = {
            **DUMMY_CONFIG,
            "model_preprocessing": {
                **DUMMY_CONFIG["model_preprocessing"],
                "modality": {
                    **DUMMY_CONFIG["model_preprocessing"]["modality"],
                    "dose": {"extract": True, "normalization_value": 70.0},
                },
            },
        }
        ct, cst = ct_and_cst
        pp = OutcomeCnnPreprocessor(config=config, device=torch.device("cpu"))
        pp.prepare(ct, cst, cst_masks=["parotid_lt"])

        n = pp._dose_grid["size"][0] * pp._dose_grid["size"][1] * pp._dose_grid["size"][2]
        dose = np.full(n, 70.0, dtype=np.float32)
        tensor = pp.build_input_tensor(dose)

        dose_channel = tensor[0, 0]
        mask_channel = tensor[0, 2]  # mask is third in type_order

        # Where mask is zero, dose should be zero
        zero_mask = mask_channel == 0
        assert torch.all(dose_channel[zero_mask] == 0.0)

    def test_gradient_flows_through_grid_sample(self, preprocessor):
        """Backward pass through grid_sample gives non-zero gradient on dose_leaf."""
        dose_size = preprocessor._dose_grid["size"]
        n_voxels = dose_size[0] * dose_size[1] * dose_size[2]
        dose = np.random.rand(n_voxels).astype(np.float32) * 70.0

        input_tensor, dose_leaf = preprocessor.build_input_tensor(dose, requires_grad=True)

        # Simulate model: just sum the dose channel
        loss = input_tensor[:, 0].sum()
        loss.backward()

        assert dose_leaf.grad is not None
        assert dose_leaf.grad.shape == dose_leaf.shape
        # At least some gradients should be non-zero
        assert dose_leaf.grad.abs().sum().item() > 0

    def test_set_dose_grid(self, preprocessor):
        """set_dose_grid recomputes coordinates for a different grid."""
        new_grid = FakeGrid(
            dimensions=(10, 10, 5),
            origin=(1.0, 1.0, 1.0),
            resolution_vector=(3.0, 3.0, 6.0),
            direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        )
        preprocessor.set_dose_grid(new_grid)
        assert preprocessor._dose_grid["size"] == (10, 10, 5)
        assert preprocessor._grid_sample_coords is not None

    def test_3d_dose_input(self, preprocessor):
        """build_input_tensor should accept 3D (X, Y, Z) dose arrays."""
        dx, dy, dz = preprocessor._dose_grid["size"]
        dose_3d = np.random.rand(dx, dy, dz).astype(np.float32)
        tensor = preprocessor.build_input_tensor(dose_3d)
        assert tensor.shape == (1, 3, 4, 8, 8)

    def test_unknown_voi_raises(self, ct_and_cst):
        ct, cst = ct_and_cst
        pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
        with pytest.raises(ValueError, match="not found"):
            pp.prepare(ct, cst, cst_masks=["nonexistent_voi"])

    def test_not_prepared_raises(self):
        pp = OutcomeCnnPreprocessor(config=DUMMY_CONFIG, device=torch.device("cpu"))
        dose = np.zeros(100, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not prepared"):
            pp.build_input_tensor(dose)


# ---------------------------------------------------------------------------
# OutcomeCNN objective tests
# ---------------------------------------------------------------------------


@requires_torch
class TestOutcomeCNN:
    """Tests for the OutcomeCNN objective.

    These tests patch the model importer to avoid needing real model weights.
    """

    @pytest.fixture
    def outcome_obj(self, ct_and_cst):
        ct, cst = ct_and_cst
        dummy_model = DummyCNN(in_channels=3)
        dummy_model.eval()

        with patch("pyRadPlan.optimization.objectives.model._cnn.load_model") as mock_load:
            mock_load.return_value = (dummy_model, DUMMY_CONFIG)
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj = OutcomeCNN(
                priority=100.0,
                model="dummy",
                mode="NTCP",
                warmup_iterations=2,
                ct=ct,
                cst=cst,
                stf=None,
                cst_masks=["parotid_lt", "parotid_rt"],
            )
        return obj

    def test_construction(self, outcome_obj):
        assert outcome_obj.model == "dummy"
        assert outcome_obj.mode == "NTCP"
        assert outcome_obj.warmup_iterations == 2
        assert outcome_obj.nn_model is not None
        assert outcome_obj.preprocessor is not None

    def test_compute_objective_returns_float(self, outcome_obj):
        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50
        val = outcome_obj.compute_objective(dose)
        assert isinstance(val, float)
        assert 0.0 <= val <= 1.0

    def test_compute_objective_monitor_only(self, ct_and_cst):
        ct, cst = ct_and_cst
        dummy_model = DummyCNN(in_channels=3)
        dummy_model.eval()

        with patch("pyRadPlan.optimization.objectives.model._cnn.load_model") as mock_load:
            mock_load.return_value = (dummy_model, DUMMY_CONFIG)
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj = OutcomeCNN(
                priority=100.0,
                model="dummy",
                mode="NTCP",
                monitor_only=True,
                ct=ct,
                cst=cst,
                cst_masks=["parotid_lt"],
            )

        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50
        assert obj.compute_objective(dose) == 0.0

    def test_compute_gradient_shape(self, outcome_obj):
        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50

        # First 2 calls are warmup → zeros
        grad1 = outcome_obj.compute_gradient(dose)
        assert grad1.shape == (n,)
        assert np.all(grad1 == 0.0)

        grad2 = outcome_obj.compute_gradient(dose)
        assert np.all(grad2 == 0.0)

        # Third call: warmup over → real gradient
        grad3 = outcome_obj.compute_gradient(dose)
        assert grad3.shape == (n,)
        assert grad3.dtype == np.float32

    def test_warmup_then_active_gradient(self, outcome_obj):
        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50

        # Exhaust warmup
        for _ in range(2):
            outcome_obj.compute_gradient(dose)

        # Now gradient should be non-trivial
        grad = outcome_obj.compute_gradient(dose)
        # At least some gradient values should be non-zero
        assert np.any(grad != 0.0)

    def test_ntcp_history(self, outcome_obj):
        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50

        for _ in range(3):
            outcome_obj.compute_gradient(dose)

        hist = outcome_obj.get_ntcp_history()
        assert len(hist) == 3
        assert all(isinstance(v, float) for v in hist)

    def test_reset_history(self, outcome_obj):
        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50
        outcome_obj.compute_gradient(dose)

        outcome_obj.reset_history()
        assert outcome_obj.ntcp_history == []
        assert outcome_obj.iteration_count == 0

    def test_tcp_mode_negates(self, ct_and_cst):
        """TCP mode should negate objective and gradient."""
        ct, cst = ct_and_cst
        dummy_model = DummyCNN(in_channels=3)
        dummy_model.eval()

        with patch("pyRadPlan.optimization.objectives.model._cnn.load_model") as mock_load:
            mock_load.return_value = (dummy_model, DUMMY_CONFIG)
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj_ntcp = OutcomeCNN(
                priority=1.0,
                model="dummy",
                mode="NTCP",
                ct=ct,
                cst=cst,
                cst_masks=["parotid_lt"],
            )
            obj_tcp = OutcomeCNN(
                priority=1.0,
                model="dummy",
                mode="TCP",
                ct=ct,
                cst=cst,
                cst_masks=["parotid_lt"],
            )

        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50

        val_ntcp = obj_ntcp.compute_objective(dose)
        val_tcp = obj_tcp.compute_objective(dose)
        assert val_tcp == pytest.approx(-val_ntcp, abs=1e-6)

    def test_monitor_only_gradient_is_zero(self, ct_and_cst):
        ct, cst = ct_and_cst
        dummy_model = DummyCNN(in_channels=3)
        dummy_model.eval()

        with patch("pyRadPlan.optimization.objectives.model._cnn.load_model") as mock_load:
            mock_load.return_value = (dummy_model, DUMMY_CONFIG)
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj = OutcomeCNN(
                priority=1.0,
                model="dummy",
                mode="NTCP",
                monitor_only=True,
                ct=ct,
                cst=cst,
                cst_masks=["parotid_lt"],
            )

        n = 20 * 20 * 10
        dose = np.random.rand(n).astype(np.float32) * 50
        grad = obj.compute_gradient(dose)
        assert np.all(grad == 0.0)
        # But history should still be logged
        assert len(obj.ntcp_history) == 1

    def test_set_dose_grid(self, outcome_obj):
        new_grid = FakeGrid(
            dimensions=(10, 10, 5),
            origin=(0.0, 0.0, 0.0),
            resolution_vector=(3.0, 3.0, 6.0),
            direction=[1, 0, 0, 0, 1, 0, 0, 0, 1],
        )
        outcome_obj.set_dose_grid(new_grid)
        assert outcome_obj.preprocessor._dose_grid["size"] == (10, 10, 5)

    def test_not_initialised_raises(self):
        """Constructing without ct/cst then calling compute should raise."""
        with patch("pyRadPlan.optimization.objectives.model._cnn.load_model"):
            from pyRadPlan.optimization.objectives.model._cnn import OutcomeCNN

            obj = OutcomeCNN(model="dummy", mode="NTCP")

        with pytest.raises(RuntimeError, match="not initialised"):
            obj.compute_objective(np.zeros(100, dtype=np.float32))

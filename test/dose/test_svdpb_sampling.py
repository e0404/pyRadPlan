"""Sampling policy, random draw order and canonical rows across backends."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine


@pytest.mark.parametrize("backend", ["numpy", "cupy", "array_api_strict"])
@pytest.mark.parametrize("sample_type", ["radius", "dose"])
@pytest.mark.parametrize("bool_indices", [True, False])
def test_sample_policy(backend, sample_type, bool_indices):
    """Compare against a direct per-group reference including clipped negative depths."""
    if backend == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    module = pytest.importorskip(backend)
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    rng = np.random.default_rng(4321)
    dose = rng.random(200).astype(np.float32)
    depth = rng.uniform(-3, 80, 200).astype(np.float32)
    radius_sq = rng.uniform(0, 900, 200).astype(np.float32)
    rows = np.arange(0, 400, 2)
    mask = np.zeros(400, dtype=bool)
    mask[rows] = True
    config = dict(type=sample_type, delta_rad_depth=5, lat_cut_off=10, rel_dose_threshold=0.5)
    core = radius_sq < 10**2 if sample_type == "radius" else dose > 0.5 * dose.max()
    groups = (np.maximum(depth, 0) / 5).astype(np.int64)
    tail = np.where(core, 0, dose)
    thresholds = np.asarray([tail[groups == g].max() for g in groups], dtype=np.float32)
    random = np.random.default_rng(77).random(200).astype(np.float32)
    keep = core | ((random * thresholds <= tail) & (thresholds > 0))
    expected_dose = np.where(core, dose, thresholds)[keep]
    engine = SimpleNamespace(dij_sampling=config, _rng=np.random.default_rng(77))
    ix, values = PhotonPencilBeamSVDEngine._sample_dij(
        engine,
        xp.asarray(mask if bool_indices else rows),
        xp.asarray(dose),
        xp.asarray(depth),
        xp.asarray(radius_sq),
    )
    np.testing.assert_array_equal(to_numpy(ix), rows[keep])
    np.testing.assert_array_equal(to_numpy(values), expected_dose)
    # Exactly one random number per input voxel, including core voxels.
    reference_rng = np.random.default_rng(77)
    reference_rng.random(200)
    assert engine._rng.random() == reference_rng.random()

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


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch"])
def sampling_engine(request):
    """Minimal photon engine for sampling edge cases."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy unavailable")
    module = pytest.importorskip(request.param)
    engine = object.__new__(PhotonPencilBeamSVDEngine)
    engine.xp = array_api_compat.array_namespace(module.asarray([0.0]))
    engine.dij_sampling = dict(
        type="radius", delta_rad_depth=5.0, lat_cut_off=10.0, rel_dose_threshold=0.5
    )
    engine._rng = np.random.default_rng(77)
    return engine


@pytest.mark.parametrize("case", ["empty", "core", "zero_tail", "tail"])
def test_sampling_rng_on_early_exits(sampling_engine, case):
    """Early exits and tail sampling consume random numbers as before."""
    engine = sampling_engine
    xp = engine.xp
    n = 0 if case == "empty" else 12
    dose = xp.zeros(n, dtype=xp.float32) if case == "zero_tail" else xp.ones(n, dtype=xp.float32)
    radius = (
        xp.zeros(n, dtype=xp.float32) if case == "core" else xp.full((n,), 900.0, dtype=xp.float32)
    )
    rows = xp.arange(n, dtype=xp.int64)
    depths = xp.asarray(np.linspace(-1, 25, n), dtype=xp.float32)
    selected, sampled = engine._sample_dij(rows, dose, depths, radius)
    reference_rng = np.random.default_rng(77)
    if case not in ("empty", "core"):
        reference_rng.random(n)
    assert engine._rng.random() == reference_rng.random()
    expected_rows = np.array([], dtype=np.int64) if case == "zero_tail" else np.arange(n)
    np.testing.assert_array_equal(to_numpy(selected), expected_rows)
    np.testing.assert_array_equal(to_numpy(sampled), np.ones(expected_rows.shape, np.float32))

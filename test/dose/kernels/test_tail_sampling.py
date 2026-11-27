"""Tail sampling kernels: backend parity and unbiased weights."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.dose.engines.kernels._tail_sampling_kernel import (
    GOLDEN_RATIO_FRACTION,
    _select_tail_samples,
    _thin_tail_candidates,
    select_tail_samples,
    thin_tail_candidates,
)

MIN_PROBABILITY = 1e-2
QUERY_SHIFT = 1.5
GRID = np.array([0.0, 20.0, 60.0, 100.0, 140.0])
CORE_SQ = np.array([9.0, 16.0, 36.0, 64.0, 25.0])
SLOPE = np.array([0.08, 0.05, 0.02, 0.01, 0.03])
FAC = np.array([1.0, 1.02, 1.05, 1.1, 1.03])
NO_SHIFT = (np.array([1.0]), np.array([0.0]))


def _candidates(num=20_000, seed=3):
    rng = np.random.default_rng(seed)
    num_voxels = 3 * num
    offsets = rng.random(num_voxels, dtype=np.float32)
    ix = np.sort(rng.choice(num_voxels, num, replace=False))
    rad_depths = rng.uniform(-5.0, 150.0, num).astype(np.float32)
    radial_dist_sq = rng.uniform(0.0, 600.0, num).astype(np.float32)
    return offsets, ix, rad_depths, radial_dist_sq


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch"])
def xp(request):
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy device unavailable")
    module = pytest.importorskip(request.param)
    return array_api_compat.array_namespace(module.asarray([0.0]))


def _thin(xp, candidates, shifts=NO_SHIFT, rotation=0.3):
    return thin_tail_candidates(
        *(xp.asarray(value) for value in candidates),
        xp.asarray(GRID),
        xp.asarray(CORE_SQ),
        xp.asarray(SLOPE),
        xp.asarray(shifts[0]),
        xp.asarray(shifts[1]),
        QUERY_SHIFT,
        rotation,
        MIN_PROBABILITY,
    )


def _select(xp, rad_depths, radial_dist_sq, numbers, envelope, core_scale=1.0, rotation=0.0):
    return select_tail_samples(
        xp.asarray(rad_depths),
        xp.asarray(radial_dist_sq),
        xp.asarray(numbers),
        xp.asarray(envelope),
        xp.asarray(GRID),
        xp.asarray(core_scale * CORE_SQ),
        xp.asarray(SLOPE),
        xp.asarray(FAC),
        QUERY_SHIFT,
        rotation,
        MIN_PROBABILITY,
        2.0,
        120.0,
    )


@pytest.mark.parametrize(
    "shifts", [NO_SHIFT, (np.array([0.95, 1.0, 1.05]), np.array([-3.0, 0.0, 3.0]))]
)
def test_thinning_matches_generic_reference(xp, shifts):
    candidates = _candidates()
    expected = _thin_tail_candidates.generic(
        *candidates, GRID, CORE_SQ, SLOPE, *shifts, QUERY_SHIFT, 0.3, MIN_PROBABILITY
    )
    keep, numbers, envelope = (to_numpy(value) for value in _thin(xp, candidates, shifts))
    assert 0 < keep.size < candidates[1].size
    # Voxels at the acceptance threshold may differ between float32 and float64 arithmetic
    common, in_actual, in_expected = np.intersect1d(keep, expected[0], return_indices=True)
    assert common.size >= max(keep.size, expected[0].size) - 5
    np.testing.assert_allclose(envelope[in_actual], expected[2][in_expected], rtol=1e-4)
    np.testing.assert_allclose(numbers[in_actual], expected[1][in_expected], rtol=1e-4, atol=1e-6)
    assert numbers.dtype == envelope.dtype == np.float32
    assert np.all((numbers >= 0) & (numbers <= 1))


def test_selection_matches_generic_reference(xp):
    offsets, ix, rad_depths, radial_dist_sq = _candidates()
    keep, numbers, envelope = _thin_tail_candidates.generic(
        offsets, ix, rad_depths, radial_dist_sq, GRID, CORE_SQ, SLOPE, *NO_SHIFT,
        QUERY_SHIFT, 0.3, MIN_PROBABILITY,
    )  # fmt: skip
    args = (rad_depths[keep], radial_dist_sq[keep], numbers, envelope)
    expected = _select_tail_samples.generic(
        *args, GRID, 0.6 * CORE_SQ, SLOPE, FAC, QUERY_SHIFT, 0.7, MIN_PROBABILITY, 2.0, 120.0
    )
    indices, weights = (
        to_numpy(value) for value in _select(xp, *args, core_scale=0.6, rotation=0.7)
    )
    assert 0 < indices.size < keep.size
    common, in_actual, in_expected = np.intersect1d(indices, expected[0], return_indices=True)
    assert common.size >= max(indices.size, expected[0].size) - 5
    np.testing.assert_allclose(weights[in_actual], expected[1][in_expected], rtol=1e-4)
    assert weights.dtype == np.float32
    assert np.all(np.diff(indices) > 0)


def test_empty_ray(xp):
    empty = (
        np.zeros(10, dtype=np.float32),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.float32),
        np.zeros(0, dtype=np.float32),
    )
    keep, numbers, envelope = _thin(xp, empty)
    assert keep.shape[0] == numbers.shape[0] == envelope.shape[0] == 0
    indices, weights = _select(xp, empty[2], empty[3], numbers, envelope)
    assert indices.shape[0] == weights.shape[0] == 0


@pytest.mark.parametrize("core_scale", [1.0, 0.5])
def test_weights_are_unbiased(core_scale):
    """Summed over many voxels, the weights restore the tail factor of every eligible voxel."""
    xp = np
    num = 400_000
    candidates = _candidates(num, seed=11)
    _, _, rad_depths, radial_dist_sq = candidates
    keep, numbers, envelope = _thin(xp, candidates)
    assert keep.size < 0.5 * num

    query = rad_depths.astype(np.float64) + QUERY_SHIFT
    core_sq = np.interp(query, GRID, core_scale * CORE_SQ)
    probability = np.exp((core_sq - radial_dist_sq) * np.interp(query, GRID, SLOPE))
    eligible = (
        (probability >= MIN_PROBABILITY)
        & (rad_depths >= 2.0)
        & (rad_depths <= 120.0)
        & (query >= GRID[0])
        & (query <= GRID[-1])
    )
    in_core = eligible & (probability >= 1.0)
    expected = np.where(in_core, 1.0, np.interp(query, GRID, FAC))[eligible].sum()

    for bixel in range(3):
        indices, weights = _select(
            xp,
            rad_depths[keep],
            radial_dist_sq[keep],
            numbers,
            envelope,
            core_scale=core_scale,
            rotation=(bixel * GOLDEN_RATIO_FRACTION) % 1.0,
        )
        selected = keep[indices]
        assert np.all(eligible[selected])
        # Every core voxel is computed, with unit weight
        assert np.isin(np.flatnonzero(in_core), selected).all()
        np.testing.assert_array_equal(weights[in_core[selected]], 1.0)
        assert weights.sum() == pytest.approx(expected, rel=0.02)


def test_range_shifts_extend_the_envelope():
    candidates = _candidates()
    keep, _, envelope = _thin(np, candidates)
    shifts = (np.array([1.0, 1.1]), np.array([0.0, 5.0]))
    keep_shifted, _, envelope_shifted = _thin(np, candidates, shifts)
    assert np.isin(keep, keep_shifted).all()
    assert keep_shifted.size > keep.size
    assert np.all(envelope_shifted[np.isin(keep_shifted, keep)] >= envelope)

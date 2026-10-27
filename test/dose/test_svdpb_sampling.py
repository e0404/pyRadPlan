"""Sampling policy, random draw order and canonical rows across backends."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import cupy_available, to_numpy
from pyRadPlan.dose.engines._svdpb import PhotonPencilBeamSVDEngine
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract


@pytest.mark.parametrize("backend", ["numpy", "cupy", "array_api_strict"])
@pytest.mark.parametrize("sample_type", ["radius", "dose"])
@pytest.mark.parametrize("bool_indices", [True, False])
@pytest.mark.parametrize("prepared", [False, True])
def test_sample_policy(backend, sample_type, bool_indices, prepared):
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
        num_clusters=int(groups.max()) + 5 if prepared else None,
    )
    np.testing.assert_array_equal(to_numpy(ix), rows[keep])
    np.testing.assert_array_equal(to_numpy(values), expected_dose)
    # Exactly one random number per input voxel, including core voxels.
    reference_rng = np.random.default_rng(77)
    reference_rng.random(200)
    assert engine._rng.random() == reference_rng.random()


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch"])
def sampling_engine(request):
    """Minimal photon engine for range-scenario bounds and sampling edge cases."""
    if request.param == "cupy" and not cupy_available():
        pytest.skip("CuPy unavailable")
    module = pytest.importorskip(request.param)
    engine = object.__new__(PhotonPencilBeamSVDEngine)
    engine.xp = array_api_compat.array_namespace(module.asarray([0.0]))
    engine.device = array_api_compat.device(module.asarray([0.0]))
    engine.enable_dij_sampling = True
    engine.dij_sampling = dict(
        type="radius", delta_rad_depth=5.0, lat_cut_off=10.0, rel_dose_threshold=0.5
    )
    engine._rng = np.random.default_rng(77)
    engine.mult_scen = SimpleNamespace(
        rel_range_shift=np.array([0.0, 0.035, -0.03, -1.0, -1.5]),
        abs_range_shift=np.array([0.0, -2.7, 4.999999, 0.0, 17.3]),
    )
    return engine


@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("width", [5.0, 0.3])
def test_sampling_bounds_match_all_ct_scenarios(sampling_engine, monkeypatch, precision, width):
    """Endpoint bounds cover shifted bins exactly, including dtype-sensitive boundaries."""
    engine = sampling_engine
    xp = engine.xp
    dtype = getattr(xp, precision)
    engine.dij_sampling["delta_rad_depth"] = width
    depths = [
        xp.asarray([-7, 0, 4.999999, 5, 5.000001, 20.123, 1e8, np.nan, np.inf], dtype=dtype),
        xp.asarray([1, 9.999999, 10, 50.0123], dtype=xp.float64),
    ]
    valid = [xp.asarray([True] * 6 + [False] * 3), xp.asarray([True] * 4)]
    transfers = []

    def download(value):
        transfers.append(value.shape)
        return to_numpy(value)

    monkeypatch.setattr("pyRadPlan.dose.engines._svdpb.to_numpy", download)
    counts = engine._prepare_sampling_cluster_counts(dict(rad_depths=depths, valid_coords=valid))
    assert transfers == [(5,)]
    for scenario, (relative, absolute) in enumerate(
        zip(engine.mult_scen.rel_range_shift, engine.mult_scen.abs_range_shift)
    ):
        expected = 1
        for depth, mask in zip(depths, valid):
            values = xp.take(depth, xp.nonzero(mask)[0], axis=0)
            shifted = (1 + xp.asarray(relative, device=engine.device)) * values + xp.asarray(
                absolute, device=engine.device
            )
            groups = xp.astype(xp.clip(shifted, 0.0) / width, xp.int64)
            expected = max(expected, int(xp.max(groups)) + 1)
        assert counts[scenario] == expected
        assert isinstance(counts[scenario], int)


@pytest.mark.parametrize("empty", [True, False])
def test_sampling_empty_beam_bounds(sampling_engine, empty):
    """Empty or entirely invalid beam supports allocate only one unused cluster."""
    engine = sampling_engine
    xp = engine.xp
    depths = xp.asarray([] if empty else [np.nan, np.inf], dtype=xp.float32)
    counts = engine._prepare_sampling_cluster_counts(
        dict(rad_depths=[depths], valid_coords=[xp.zeros(depths.shape, dtype=xp.bool)])
    )
    assert counts == [1] * 5


@pytest.mark.parametrize("width", [0.0, -1.0, np.inf, np.nan])
def test_sampling_bound_rejects_invalid_width(sampling_engine, width):
    """A prepared cluster bound requires finite positive bin width."""
    sampling_engine.dij_sampling["delta_rad_depth"] = width
    with pytest.raises(ValueError, match="finite and positive"):
        sampling_engine._prepare_sampling_cluster_counts({})


@pytest.mark.parametrize("case", ["empty", "core", "zero_tail", "tail"])
def test_sampling_bound_rng_and_no_count_reduction(sampling_engine, monkeypatch, case):
    """Prepared counts preserve early exits and RNG consumption without a per-ray max."""
    engine = sampling_engine
    xp = engine.xp
    n = 0 if case == "empty" else 12
    dose = xp.zeros(n, dtype=xp.float32) if case == "zero_tail" else xp.ones(n, dtype=xp.float32)
    radius = (
        xp.zeros(n, dtype=xp.float32) if case == "core" else xp.full((n,), 900.0, dtype=xp.float32)
    )
    rows = xp.arange(n, dtype=xp.int64)
    depths = xp.asarray(np.linspace(-1, 25, n), dtype=xp.float32)
    with monkeypatch.context() as patch:
        patch.setattr(xp, "max", lambda *args, **kwargs: pytest.fail("Per-ray cluster reduction"))
        selected, sampled = engine._sample_dij(rows, dose, depths, radius, num_clusters=10)
    reference_rng = np.random.default_rng(77)
    if case not in ("empty", "core"):
        reference_rng.random(n)
    assert engine._rng.random() == reference_rng.random()
    expected_rows = np.array([], dtype=np.int64) if case == "zero_tail" else np.arange(n)
    np.testing.assert_array_equal(to_numpy(selected), expected_rows)
    np.testing.assert_array_equal(to_numpy(sampled), np.ones(expected_rows.shape, np.float32))


def test_sampling_count_follows_scenario_number(sampling_engine, monkeypatch):
    """Attach counts using the same scenario-number mapping as range adjustments."""
    engine = sampling_engine
    engine.mult_scen.scen_num = lambda index: {7: 1, 12: 0}[index]
    monkeypatch.setattr(
        PencilBeamEngineAbstract,
        "_extract_single_scenario_ray",
        lambda self, ray, index: ray.copy(),
    )
    ray = dict(sampling_cluster_counts=[11, 42])
    assert engine._extract_single_scenario_ray(ray, 7)["sampling_num_clusters"] == 42
    assert engine._extract_single_scenario_ray(ray, 12)["sampling_num_clusters"] == 11
    assert "sampling_num_clusters" not in ray
    engine.enable_dij_sampling = False
    assert engine._extract_single_scenario_ray({}, 7) == {}


def test_sampling_bounds_cached_after_beam_setup(sampling_engine, monkeypatch):
    """Prepare once per beam, share between rays, and reset for another beam/shift."""
    engine = sampling_engine
    engine.use_custom_primary_photon_fluence = False
    monkeypatch.setattr(PencilBeamEngineAbstract, "_init_ray", lambda self, beam, j: {})
    calls = []

    def prepare(beam):
        calls.append(beam)
        return [17]

    monkeypatch.setattr(engine, "_prepare_sampling_cluster_counts", prepare)
    beam = dict(
        kernel=object(),
        kernel_depth_parameters={},
        field_based_dose_calc=False,
        kernel_interpolators=object(),
    )
    first = engine._init_ray(beam, 0)
    second = engine._init_ray(beam, 1)
    assert first["sampling_cluster_counts"] is second["sampling_cluster_counts"]
    assert len(calls) == 1
    another = {key: value for key, value in beam.items() if key != "sampling_cluster_counts"}
    engine._init_ray(another, 0)
    assert len(calls) == 2

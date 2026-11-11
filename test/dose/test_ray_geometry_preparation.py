"""Ordered geometry selection across backends, scenarios and empty supports."""

from types import MethodType, SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.dose.engines._base_pencilbeam import PencilBeamEngineAbstract


@pytest.fixture(params=["numpy", "cupy", "array_api_strict", "torch"])
def engine(request):
    """Minimal engine on the requested backend."""
    module = pytest.importorskip(request.param)
    if request.param == "cupy" and not module.cuda.is_available():
        pytest.skip("CUDA unavailable")
    xp = array_api_compat.array_namespace(module.asarray([0.0]))
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.xp = xp
    engine.device = array_api_compat.device(module.asarray([0.0]))
    engine._eps_ijk = None
    engine._cuda_kernel_used = False
    engine._effective_lateral_cutoff = 5.0
    engine._get_lateral_distance_from_dose_cutoff_on_ray = MethodType(
        PencilBeamEngineAbstract._get_lateral_distance_from_dose_cutoff_on_ray, engine
    )
    return engine


@pytest.mark.parametrize("scenarios", [1, 3])
@pytest.mark.parametrize("support", ["mixed", "empty", "all", "miss"])
def test_ray_selection_and_cache(engine, monkeypatch, scenarios, support):
    """Preserve row order and scenario/depth alignment without full per-ray masks."""
    xp = engine.xp
    # Exact boundary points (+/-5), invalid holes, and points outside the cone.
    coords = np.array([[x, 1024.0, 0.0] for x in (-9, -5, -3, 0, 2, 5, 8)], np.float32)
    masks = np.array(
        [
            [False, True, True, True, False, True, True],
            [True, False, True, False, True, True, False],
            [False] * 7,
        ]
    )[:scenarios]
    if support == "empty":
        masks[:] = False
    elif support == "all":
        masks[:] = True
    if support == "miss":
        coords[:, 0] += 100.0
    depths = np.arange(7, dtype=np.float32) + 10
    rows = np.array([2, 7, 12, 33, 45, 81, 103])
    engine._vdose_grid = xp.asarray(rows)
    beam = {
        "bev_coords": xp.asarray(coords),
        "valid_coords": [xp.asarray(m) for m in masks],
        "valid_coords_all": xp.asarray(masks.any(axis=0)),
        "rad_depths": [xp.asarray(np.where(m, depths + i, np.nan)) for i, m in enumerate(masks)],
        "geo_depths": [xp.asarray(depths * 2)] * scenarios,
    }
    beam["beam"] = {
        "source_point_bev": xp.asarray([0.0, -1024.0, 0.0], dtype=xp.float32),
        "rays": [{"target_point_bev": xp.asarray([0.0, 1024.0, 0.0], dtype=xp.float32)}],
    }
    calls = []
    prepare = engine._prepare_beam_geometry

    def counted(info):
        calls.append(info)
        return prepare(info)

    monkeypatch.setattr(engine, "_prepare_beam_geometry", counted)
    for _ in range(2):
        ray = {
            "ray_index": 0,
            "source_point_bev": xp.asarray([0.0, -1024.0, 0.0], dtype=xp.float32),
            "target_point_bev": xp.asarray([0.0, 1024.0, 0.0], dtype=xp.float32),
            "sad": 1024.0,
        }
        engine._get_ray_geometry_from_beam(ray, beam)
        for scenario, mask in enumerate(masks):
            selected = mask & (coords[:, 0] ** 2 <= 25)
            expected = {
                "ix": rows[selected],
                "radial_dist_sq": coords[selected, 0] ** 2,
                "lat_dists": coords[selected][:, [0, 2]],
                "iso_lat_dists": coords[selected][:, [0, 2]],
                "rad_depths": (depths + scenario)[selected],
                "geo_depths": (depths * 2)[selected],
            }
            for name, values in expected.items():
                actual = ray[name][scenario]
                np.testing.assert_allclose(to_numpy(actual), values, rtol=1e-7, atol=1e-6)
                assert array_api_compat.device(actual) == engine.device
                assert array_api_compat.array_namespace(actual) == xp
    assert len(calls) == 1
    # A new beam/shift must not reuse the prior beam's cache.
    new_beam = {key: value for key, value in beam.items() if key != "_ray_geometry"}
    engine._get_ray_geometry_from_beam(ray, new_beam)
    assert len(calls) == 2


@pytest.mark.parametrize("angle", [0.0, 0.08])
@pytest.mark.parametrize("empty", [False, True])
def test_compatibility_mask_and_rotated_distances(engine, angle, empty):
    """Check the public mask interface against an independent rotation construction."""
    xp = engine.xp
    rng = np.random.default_rng(173)
    coords = rng.uniform([-25, 900, -25], [25, 1100, 25], size=(80, 3))
    valid = np.arange(80) % 3 != 0
    if empty:
        valid[:] = False
    target = np.array([2000 * np.sin(angle), 2000 * np.cos(angle) - 1000, 0.0])
    rotation = np.array(
        [[np.cos(angle), np.sin(angle), 0], [-np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    rotated = coords @ rotation
    lateral = rotated[:, [0, 2]]
    radial = np.sum(lateral**2, axis=1)
    mask = valid & (radial <= (30 / 1000) ** 2 * rotated[:, 1] ** 2)
    result = engine.calc_geo_dists(
        xp.asarray(coords),
        xp.asarray([0.0, -1000.0, 0.0], dtype=xp.float64),
        xp.asarray(target),
        1000.0,
        xp.asarray(valid),
        30.0,
    )
    np.testing.assert_array_equal(to_numpy(result[0]), mask)
    np.testing.assert_allclose(to_numpy(result[1]), radial[mask], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(to_numpy(result[2]), lateral[mask], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        to_numpy(result[3]), lateral[mask] / rotated[mask, 1, None] * 1000, rtol=1e-12, atol=1e-10
    )


def test_numpy_geometry_does_not_probe_gpu_libraries(monkeypatch):
    """CPU geometry does not invoke availability checks for unrelated backends."""
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine._eps_ijk = None
    engine._cuda_kernel_used = False
    for name in ("cupy_available", "pytorch_gpu_available"):
        monkeypatch.setattr(
            "pyRadPlan.dose.engines._base_pencilbeam." + name,
            lambda: pytest.fail("Unexpected GPU capability probe"),
        )
    engine.calc_geo_dists(
        np.array([[0.0, 1000.0, 0.0]]),
        np.array([0.0, -1000.0, 0.0]),
        np.array([0.0, 1000.0, 0.0]),
        1000,
        np.array([True]),
        5,
    )


@pytest.mark.parametrize("rays", [1, 7])
def test_ssd_metadata_transferred_in_batches(engine, monkeypatch, rays):
    """Transfer SSD values and validity flags once each, independent of ray count."""
    xp = engine.xp
    beam = {
        "num_of_rays": rays,
        "sad": 1000.0,
        "iso_center": xp.zeros(3, dtype=xp.float32),
        "source_point": xp.asarray([0.0, -1000.0, 0.0], dtype=xp.float32),
        "rays": [
            {
                "ray_pos_bev": xp.asarray([float(i), 0.0, 0.0], dtype=xp.float32),
                "target_point": xp.asarray([float(i), 1000.0, 0.0], dtype=xp.float32),
            }
            for i in range(rays)
        ],
    }
    alpha = xp.asarray(np.tile([0.1, 0.3, 0.8], (rays, 1)), dtype=xp.float32)
    rho = xp.asarray(np.tile([0.0, 0.5, 1.0], (rays, 1)), dtype=xp.float32)
    d12 = xp.asarray(np.arange(rays) + 2000, dtype=xp.float32)
    engine._raytracer = SimpleNamespace(trace_rays=lambda *args: (alpha, None, [rho], d12, None))
    transfers = []

    def download(value):
        transfers.append(value.shape)
        return to_numpy(value)

    monkeypatch.setattr("pyRadPlan.dose.engines._base_pencilbeam.to_numpy", download)
    engine._compute_ssd({"beam": beam}, None)
    assert transfers == [(rays,), (rays,)]
    np.testing.assert_array_equal(
        [ray["SSD"][0] for ray in beam["rays"]], to_numpy(d12 * xp.asarray(0.3, dtype=xp.float32))
    )
    assert all(isinstance(ray["SSD"][0], float) for ray in beam["rays"])


def test_empty_coordinate_array(engine):
    """The public interface accepts a genuinely empty selected voxel set."""
    xp = engine.xp
    mask, radial, lateral, projected = engine.calc_geo_dists(
        xp.empty((0, 3), dtype=xp.float32),
        xp.asarray([0.0, -1000.0, 0.0], dtype=xp.float32),
        xp.asarray([0.0, 1000.0, 0.0], dtype=xp.float32),
        1000.0,
        xp.empty(0, dtype=xp.bool),
        5.0,
    )
    assert mask.shape == radial.shape == (0,)
    assert lateral.shape == projected.shape == (0, 2)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_batched_ray_rotations(engine, monkeypatch, dtype):
    """Known rotations retain alignment tolerance and transfer only one flag vector."""
    xp = engine.xp
    precision = getattr(xp, dtype)
    # Both sides of the legacy directional tolerance, plus rotations in both planes.
    angles = np.array([0.0, 0.5e-7, 2e-7, -0.08, 0.15])
    targets = np.stack((2000 * np.sin(angles), 2000 * np.cos(angles) - 1000, np.zeros(5)), axis=1)
    targets = np.concatenate((targets, [[0, 1000 * np.cos(0.1) - 1000, 1000 * np.sin(0.1)]]))
    source = xp.asarray([0, -1000, 0], dtype=precision)
    transfers = []

    def download(array):
        transfers.append(array.shape)
        return to_numpy(array)

    monkeypatch.setattr("pyRadPlan.dose.engines._base_pencilbeam.to_numpy", download)
    prepared = engine._prepare_ray_transforms(source, xp.asarray(targets, dtype=precision))
    assert transfers == [(6,)]
    np.testing.assert_array_equal(prepared["aligned"], [True, True, False, False, False, False])
    matrices = to_numpy(prepared["rotations"])
    for i, angle in enumerate(angles):
        if i < 2:
            np.testing.assert_array_equal(matrices[i], np.eye(3))
        else:
            expected = [
                [np.cos(angle), np.sin(angle), 0],
                [-np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
            np.testing.assert_allclose(matrices[i], expected, rtol=2e-6, atol=2e-7)
    expected = [[1, 0, 0], [0, np.cos(0.1), -np.sin(0.1)], [0, np.sin(0.1), np.cos(0.1)]]
    np.testing.assert_allclose(matrices[-1], expected, rtol=2e-6, atol=2e-7)
    assert prepared["rotations"].dtype == precision
    assert array_api_compat.device(prepared["rotations"]) == engine.device
    # Evaluation must consume prepared state without rebuilding or downloading flags.
    monkeypatch.setattr(
        engine, "_prepare_ray_transforms", lambda *args: pytest.fail("Rebuilt transform")
    )
    coords = xp.asarray([[0, 1000, 0], [20, 1000, 5]], dtype=precision)
    for i in range(6):
        engine._calc_geo_dists_on_valid_coords(
            coords,
            source,
            xp.asarray(targets[i], dtype=precision),
            1000,
            30,
            prepared_transform=(source, prepared["rotations"][i, ...], prepared["aligned"][i]),
        )
    assert transfers == [(6,)]


def test_empty_ray_transform_batch(engine):
    """An empty beam has empty transform and metadata arrays on the right device."""
    xp = engine.xp
    prepared = engine._prepare_ray_transforms(
        xp.asarray([0, -1000, 0], dtype=xp.float32), xp.empty((0, 3), dtype=xp.float32)
    )
    assert prepared["rotations"].shape == (0, 3, 3)
    assert prepared["aligned"].shape == (0,)


def test_oblique_source_transforms(engine):
    """General source positions agree with an independent scalar Rodrigues formula."""
    xp = engine.xp
    source = np.array([20.0, -1000.0, -30.0])
    targets = np.array([[-40.0, 1000.0, 10.0], [50.0, 900.0, -60.0], -source])
    source_backend = xp.asarray(source)
    targets_backend = xp.asarray(targets)
    prepared = engine._prepare_ray_transforms(source_backend, targets_backend)
    actual = to_numpy(prepared["rotations"])
    for i, target in enumerate(targets):
        a = -source / np.linalg.norm(source)
        direction = target - source
        b = direction / np.linalg.norm(direction)
        if np.all(np.abs(a - b) < 1e-7):
            expected = np.eye(3)
        else:
            x, y, z = np.cross(a, b)
            skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
            expected = (
                np.eye(3) + skew + skew @ skew * (1 - np.dot(a, b)) / (x * x + y * y + z * z)
            )
        np.testing.assert_allclose(actual[i], expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_array_equal(to_numpy(source_backend), source)
    np.testing.assert_array_equal(to_numpy(targets_backend), targets)

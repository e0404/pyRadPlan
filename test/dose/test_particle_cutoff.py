"""Batched cutoff quadrature preserves the scalar sampling and selection rules."""

from types import SimpleNamespace

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import to_namespace, to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine
from pyRadPlan.dose.engines._particle_cutoff import CutoffQuadrature, evaluate_cutoff


@pytest.fixture(params=["numpy", "cupy", "array_api_strict"])
def xp(request):
    """Provide native and strict numerical backends."""
    backend = pytest.importorskip(request.param)
    if request.param == "cupy":
        try:
            backend.zeros(1)
        except backend.cuda.runtime.CUDARuntimeError:
            pytest.skip("CUDA device unavailable")
    return array_api_compat.array_namespace(backend.zeros(1))


def _inputs(model):
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.lateral_model = "multi" if model == "multi_1d" else model
    engine._calc_let = engine._use_let_kernel = engine._calc_bio_dose = False
    engine._machine = SimpleNamespace(get_energy_index=lambda *_: 0)
    kernel = dict(
        energy=100.0,
        offset=0.7,
        depths=np.array([0.0, 3.0, 8.0]),
        idd=np.array([2.0, 3.0, 1.0]),
        sigma=np.array([1.0, 2.0, 3.0]),
        sigma_1=np.array([1.0, 2.0, 3.0]),
        sigma_2=np.array([3.0, 4.0, 5.0]),
        weight=np.array([0.1, 0.2, 0.3]),
        sigma_x=np.array([0.8, 1.2, 2.0]),
        sigma_y=np.array([2.0, 3.0, 4.0]),
        sigma_multi=np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]),
        weight_multi=np.array([0.1, 0.2, 0.3]),
        lateral_cut_off=dict(comp_fac=1.0),
    )
    if model == "multi":
        kernel["sigma_multi"] = np.vstack((kernel["sigma_multi"], [5.0, 6.0, 7.0]))
        kernel["weight_multi"] = np.array([[0.1, 0.2, 0.3], [0.05, 0.05, 0.05]])
    edges = np.concatenate(([0.0], np.geomspace(0.05, 50, 128)))
    radii = (edges[1:] + edges[:-1]) * 0.5
    dr = np.diff(edges)
    theta = np.arange(16) * (2 * np.pi / 16)
    if model == "singleXY":
        radii = np.repeat(radii, theta.size)
        dr = np.repeat(dr, theta.size)
        theta = np.tile(theta, len(edges) - 1)
        lat = np.column_stack((radii * np.cos(theta), radii * np.sin(theta)))
        angular = np.full_like(radii, 2 * np.pi / 16)
    else:
        lat = np.tile((radii / np.sqrt(2))[:, None], (1, 2))
        angular = np.asarray(2 * np.pi)
    grid = CutoffQuadrature(
        radii,
        dr,
        angular,
        radii**2,
        lat,
        edges,
        16 if model == "singleXY" else 1,
        engine.lateral_model,
    )
    return engine, kernel, grid


def _scalar_reference(engine, kernel, grid, depth, sigma_sq):
    """Independent scalar Gaussian formulas on the fixed radial samples."""
    params = {
        name: np.array([np.interp(depth, kernel["depths"], row) for row in np.atleast_2d(v)])
        for name, v in kernel.items()
        if name in (*engine._lateral_kernel_names(), "idd")
    }
    if engine.lateral_model == "singleXY":
        sx, sy = params["sigma_x"][0] ** 2 + sigma_sq, params["sigma_y"][0] ** 2 + sigma_sq
        lateral = np.exp(
            -(grid.lat_dists[:, 0] ** 2 / (2 * sx) + grid.lat_dists[:, 1] ** 2 / (2 * sy))
        ) / (2 * np.pi * np.sqrt(sx * sy))
    else:
        if engine.lateral_model == "single":
            sigmas, weights = params["sigma"], np.array([1.0])
        elif engine.lateral_model == "double":
            sigmas = np.concatenate((params["sigma_1"], params["sigma_2"]))
            weights = np.array([1 - params["weight"][0], params["weight"][0]])
        else:
            sigmas = params["sigma_multi"]
            weights = np.concatenate(([1 - params["weight_multi"].sum()], params["weight_multi"]))
        variances = sigmas**2 + sigma_sq
        lateral = sum(
            w * np.exp(-grid.radial_dist_sq / (2 * v)) / (2 * np.pi * v)
            for w, v in zip(weights, variances)
        )
    return lateral * (1.6021766208e-02 * params["idd"][0])


@pytest.mark.parametrize("model", ["single", "double", "multi", "multi_1d", "singleXY"])
@pytest.mark.parametrize("method", ["integral", "relative"])
@pytest.mark.parametrize("max_samples", [1, 100_000])
def test_cutoff_matches_scalar_reference(xp, model, method, max_samples):
    """Batching preserves radii, compensation and numerical integration error."""
    engine, kernel, host_grid = _inputs(model)
    depths = np.array([0.0, 1.2, 4.5, 8.0])
    idd = 1.6021766208e-02 * np.interp(depths, kernel["depths"], kernel["idd"])
    cutoff = 0.95
    expected = []
    for depth, reference_idd in zip(depths, idd):
        dose = _scalar_reference(engine, kernel, host_grid, depth, 0.4)
        area = np.cumsum(host_grid.radii * dose * host_grid.dr * host_grid.angular_weights)
        condition = (
            area >= reference_idd * cutoff
            if method == "integral"
            else dose <= (1 - cutoff) * dose.max()
        )
        ix = condition.argmax()
        comp = cutoff**-1 if method == "integral" else (area[ix] / area[-1]) ** -1
        expected.append([host_grid.radii[ix], comp, abs(area[-1] / reference_idd - 1) * 100])
    backend_kernel = {
        key: to_namespace(xp, value) if isinstance(value, np.ndarray) else value
        for key, value in kernel.items()
    }
    grid = host_grid.to_namespace(xp)
    batches = []

    def profile(batch, quadrature):
        batches.append(batch.shape[0])
        return engine._cutoff_dose_profile(batch, backend_kernel, 0.4, quadrature)

    actual = evaluate_cutoff(
        xp.asarray(depths),
        xp.asarray(idd),
        grid,
        profile,
        cutoff=cutoff,
        method=method,
        max_samples=max_samples,
    )
    np.testing.assert_allclose(to_numpy(actual), expected, rtol=2e-12, atol=2e-12)
    assert max(batches) <= max(1, max_samples // grid.radii.shape[0])
    assert sum(batches) == len(depths)


def test_cutoff_full_and_empty_do_not_evaluate(xp):
    """Skip physical evaluation when no cutoff work is needed."""
    _, _, grid = _inputs("single")
    grid = grid.to_namespace(xp)

    def unexpected(*_):
        pytest.fail("Dose must not be evaluated for full cutoff or empty depths")

    full = evaluate_cutoff(
        xp.asarray([1.0]), xp.asarray([1.0]), grid, unexpected, cutoff=1, method="integral"
    )
    np.testing.assert_array_equal(to_numpy(full), [[np.inf, 1.0, 0.0]])
    empty = xp.asarray([], dtype=xp.float64)
    result = evaluate_cutoff(empty, empty, grid, unexpected, cutoff=0.99, method="integral")
    assert result.shape == (0, 3)


@pytest.mark.parametrize("bad", [np.nan, -1.0])
def test_cutoff_rejects_invalid_dose(xp, bad):
    """Retain physical-dose validation in every batch."""
    _, _, grid = _inputs("single")
    grid = grid.to_namespace(xp)
    with pytest.raises(ValueError, match="particle dose"):
        evaluate_cutoff(
            xp.asarray([1.0]),
            xp.asarray([1.0]),
            grid,
            lambda *_: xp.full((1, grid.radii.shape[0]), bad),
            cutoff=0.99,
            method="integral",
        )


def test_hong_subclass_uses_custom_physics():
    """An engine override must not silently use the Hong-only batching shortcut."""

    class CustomEngine(ParticleHongPencilBeamEngine):
        def _calc_particle_bixel(self, bixel):
            bixel["physical_dose"] = np.full_like(bixel["rad_depths"], 7.0)

    engine, kernel, grid = _inputs("single")
    custom = object.__new__(CustomEngine)
    custom._machine = engine._machine
    result = custom._cutoff_dose_profile(np.array([1.0, 2.0]), kernel, 0.4, grid)
    np.testing.assert_array_equal(result, np.full((2, grid.radii.size), 7.0))


def test_default_angular_grid_keeps_radius_major_order():
    """Angular samples retain the original flattened Fortran mesh order."""
    isotropic = CutoffQuadrature.create("single")
    angular = CutoffQuadrature.create("singleXY")
    np.testing.assert_array_equal(angular.radii, np.repeat(isotropic.radii, 360))
    np.testing.assert_array_equal(angular.dr, np.repeat(isotropic.dr, 360))
    assert angular.lat_dists.shape == (1200 * 360, 2)


@pytest.mark.parametrize("model", ["single", "singleXY"])
def test_native_grid_matches_historical_sampling(xp, model):
    """Default sampling agrees with independently built host geometry on all backends."""
    grid = CutoffQuadrature.create(model, xp=xp, device=array_api_compat.device(xp.zeros(1)))
    edges = np.concatenate(([0], np.logspace(-1, 3, 1200)))
    radii = 0.5 * (edges[:-1] + edges[1:])
    dr = np.diff(edges)
    if model == "singleXY":
        angles = np.linspace(0, 2 * np.pi, 361)[:-1]
        radii = np.repeat(radii, 360)
        dr = np.repeat(dr, 360)
        theta = np.tile(angles, 1200)
        lat = radii[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))
        angular = np.full(radii.shape, angles[1] - angles[0])
    else:
        lat = np.tile(np.sqrt(radii**2 / 2)[:, None], (1, 2))
        angular = np.asarray(2 * np.pi)
    for name, expected in dict(
        radial_edges=edges, radii=radii, dr=dr, angular_weights=angular, lat_dists=lat
    ).items():
        actual = getattr(grid, name)
        assert array_api_compat.array_namespace(actual) == xp
        np.testing.assert_allclose(to_numpy(actual), expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("n_angles", [1, 4, 17])
def test_custom_sampling_preserves_definition_and_integrates_annuli(xp, n_angles):
    """Changing edges/angles updates every dependent array and retains the definition."""
    edges = xp.asarray([0.0, 1.0, 3.0, 4.0], dtype=xp.float64)
    grid = CutoffQuadrature.create("singleXY", radial_edges=edges, n_angles=n_angles)
    assert array_api_compat.array_namespace(grid.radii) == xp
    assert grid.n_angles == n_angles
    assert grid.lateral_model == "singleXY"
    edges[1] = 0.5
    np.testing.assert_array_equal(to_numpy(grid.radial_edges), [0, 1, 3, 4])
    np.testing.assert_array_equal(to_numpy(grid.radii), np.repeat([0.5, 2, 3.5], n_angles))
    np.testing.assert_array_equal(to_numpy(grid.dr), np.repeat([1, 2, 1], n_angles))
    np.testing.assert_allclose(to_numpy(grid.radial_dist_sq), to_numpy(grid.radii) ** 2)
    np.testing.assert_allclose(
        to_numpy(xp.sum(grid.lat_dists**2, axis=1)), to_numpy(grid.radial_dist_sq), rtol=1e-14
    )
    area = xp.sum(grid.radii * grid.dr * grid.angular_weights)
    assert float(area) == pytest.approx(np.pi * 4**2)
    moved = grid.to_namespace(array_api_compat.array_namespace(np.empty(0)))
    assert moved.n_angles == n_angles
    assert moved.lateral_model == "singleXY"
    np.testing.assert_array_equal(moved.radial_edges, [0, 1, 3, 4])


@pytest.mark.parametrize(
    "edges", [[0], [[0, 1]], [1, 2], [0, 0, 1], [0, 2, 1], [0, np.inf], [0, np.nan]]
)
def test_invalid_sampling_edges(xp, edges):
    """Reject malformed radial definitions before any integration."""
    with pytest.raises(ValueError, match="radial_edges"):
        CutoffQuadrature.create("single", xp=xp, radial_edges=edges)


@pytest.mark.parametrize("n_angles", [0, -1, 1.5, True])
def test_invalid_angle_count(n_angles):
    """Angular resolution is a positive integer."""
    with pytest.raises(ValueError, match="n_angles"):
        CutoffQuadrature.create("singleXY", n_angles=n_angles)


def test_invalid_quadrature_model():
    """Unresolved or misspelled models should not silently get isotropic sampling."""
    with pytest.raises(ValueError, match="lateral model"):
        CutoffQuadrature.create("unknown")


def test_quadrature_respects_explicit_and_inferred_device():
    """Native creation and custom boundaries stay on a non-default logical device."""
    xp = pytest.importorskip("array_api_strict")
    device = xp.Device("device1")
    edges = xp.asarray([0.0, 1.0, 2.0], device=device)
    for grid in (
        CutoffQuadrature.create("singleXY", xp=xp, device=device, n_angles=4),
        CutoffQuadrature.create("singleXY", radial_edges=edges, n_angles=4),
        CutoffQuadrature.create(
            "singleXY", xp=xp, device=device, radial_edges=[0, 1, 2], n_angles=4
        ),
    ):
        for name in ("radial_edges", "radii", "dr", "lat_dists", "angular_weights"):
            assert array_api_compat.device(getattr(grid, name)) == device


def test_host_edges_move_to_requested_backend(xp):
    """Explicit placement overrides the custom input's namespace and keeps its definition."""
    edges = np.array([0.0, 1.0, 3.0])
    grid = CutoffQuadrature.create("single", xp=xp, radial_edges=edges)
    assert array_api_compat.array_namespace(grid.radial_edges) == xp
    assert grid.n_angles == 1
    edges[1] = 0.5
    np.testing.assert_array_equal(to_numpy(grid.radial_edges), [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(to_numpy(grid.radii), [0.5, 2.0])


def test_hong_subclass_uses_custom_interpolation():
    """A custom kernel interpolator remains authoritative during cutoff setup."""

    class CustomInterpolation(ParticleHongPencilBeamEngine):
        def _interpolate_kernels_in_depth(self, bixel):
            result = super()._interpolate_kernels_in_depth(bixel)
            result["idd"] *= 2
            return result

    engine, kernel, grid = _inputs("single")
    custom = object.__new__(CustomInterpolation)
    custom.__dict__.update(engine.__dict__)
    depths = np.array([1.0, 2.0])
    expected = engine._cutoff_dose_profile(depths, kernel, 0.4, grid) * 2
    actual = custom._cutoff_dose_profile(depths, kernel, 0.4, grid)
    np.testing.assert_allclose(actual, expected, rtol=1e-14)


def test_no_threshold_crossing_preserves_first_sample_rule(xp):
    """Keep the existing argmax outcome when every threshold comparison is false."""
    _, _, grid = _inputs("single")
    grid = grid.to_namespace(xp)
    result = evaluate_cutoff(
        xp.asarray([1.0]),
        xp.asarray([1e12]),
        grid,
        lambda *_: xp.ones((1, grid.radii.shape[0])),
        cutoff=0.99,
        method="integral",
    )
    assert float(to_numpy(result)[0, 0]) == float(to_numpy(grid.radii)[0])


@pytest.mark.parametrize("kwargs", [{"method": "invalid"}, {"cutoff": 0}, {"max_samples": 0}])
def test_invalid_cutoff_settings_fail(kwargs):
    """Reject invalid numerical settings before evaluating a profile."""
    _, _, grid = _inputs("single")
    options = dict(cutoff=0.99, method="integral")
    options.update(kwargs)
    with pytest.raises(ValueError):
        evaluate_cutoff(np.array([1.0]), np.array([1.0]), grid, lambda *_: None, **options)


def test_cutoff_placement_configuration(xp):
    """The plan option selects placement without changing the numerical implementation."""
    from pyRadPlan import IonPlan  # noqa: PLC0415

    engine = ParticleHongPencilBeamEngine(IonPlan())
    engine.xp = xp
    engine.device = array_api_compat.device(xp.asarray([1.0]))
    assert engine._cutoff_execution() == (xp, engine.device)
    engine.cutoff_force_numpy = True
    namespace, device = engine._cutoff_execution()
    assert array_api_compat.is_numpy_namespace(namespace)
    assert device is None
    config = engine.config_model()
    assert config.model_fields["cutoff_force_numpy"].default is False
    assert "cutoff_backend" not in config.model_fields
    forced = ParticleHongPencilBeamEngine(IonPlan(prop_dose_calc={"cutoff_force_numpy": True}))
    assert forced.cutoff_force_numpy is True
    with pytest.raises(ValueError):
        config(cutoff_force_numpy="invalid")


def test_cutoff_rejects_mismatched_shapes(xp):
    """Do not silently broadcast depth or physical-dose tables with wrong shapes."""
    _, _, grid = _inputs("single")
    grid = grid.to_namespace(xp)
    with pytest.raises(ValueError, match="matching shapes"):
        evaluate_cutoff(
            xp.asarray([1.0, 2.0]),
            xp.asarray([1.0]),
            grid,
            lambda *_: None,
            cutoff=0.99,
            method="integral",
        )
    with pytest.raises(ValueError, match="unexpected shape"):
        evaluate_cutoff(
            xp.asarray([1.0]),
            xp.asarray([1.0]),
            grid,
            lambda *_: xp.ones((1, 1)),
            cutoff=0.99,
            method="integral",
        )

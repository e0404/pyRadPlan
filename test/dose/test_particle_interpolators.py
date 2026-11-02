"""Prepared particle fields preserve offsets, biology and cache lifetimes."""

from types import SimpleNamespace
from unittest.mock import Mock

import array_api_compat
import numpy as np
import pytest
from scipy.interpolate import interp1d as scipy_interp1d

from pyRadPlan.core.xp_utils import interp1d, to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine


def engine_stub():
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine._kernel_cache = {}
    engine._depth_interpolator_cache = {}
    engine._focus_interpolator_cache = {}
    engine._calc_bio_dose = False
    engine._use_let_kernel = True
    engine.lateral_model = "single"
    return engine


@pytest.mark.parametrize("backend", ["numpy", "torch", "array_api_strict"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_translated_depths_and_cache(backend, dtype):
    backend = pytest.importorskip(backend)
    xp = array_api_compat.array_namespace(backend.asarray([0.0]))
    engine = engine_stub()
    kernel = {
        "depths": xp.asarray([0.0, 0.1, 0.8, 2.0], dtype=getattr(xp, dtype)),
        "offset": 0.03217,
        "idd": xp.asarray([1.0, 2.0, 3.0, 0.2], dtype=getattr(xp, dtype)),
        "sigma": xp.asarray([0.1, 0.2, 0.4, 0.7], dtype=getattr(xp, dtype)),
        "let": xp.asarray([2.0, 4.0, 5.0, 0.0], dtype=getattr(xp, dtype)),
    }
    prepare = Mock(wraps=engine._prepare_depth_interpolator)
    engine._prepare_depth_interpolator = prepare
    for offset in (0.0, 0.134, -0.23, 8.5):
        grid = kernel["depths"] + kernel["offset"] - offset
        q = xp.concat([grid, grid + 0.01, grid - 0.01])
        bixel = dict(kernel=kernel, rad_depths=q, rad_depth_offset=offset)
        expected = interp1d(
            q,
            grid,
            {
                "idd": kernel["idd"] * 1.6021766208e-02,
                "sigma": kernel["sigma"],
                "let": kernel["let"],
            },
        )
        actual = engine._interpolate_kernels_in_depth(bixel)
        for key in expected:
            np.testing.assert_array_equal(to_numpy(actual[key]), to_numpy(expected[key]))
    assert prepare.call_count == 1
    engine._kernel_cache[0] = kernel
    engine._init_geometry_group()
    assert not engine._kernel_cache
    assert not engine._depth_interpolator_cache
    engine._interpolate_kernels_in_depth(bixel)
    assert prepare.call_count == 2


def test_biological_fields_are_separate_from_cutoff_cache():
    engine = engine_stub()
    engine._calc_bio_dose = True
    engine._validated_bio_kernel_energies = set()
    engine._bio_evaluator = SimpleNamespace(
        kernel_quantities=Mock(return_value={"alpha": np.arange(6.0).reshape(2, 3)}),
        validate_kernel_quantities=Mock(),
    )
    kernel = dict(
        depths=np.arange(3.0),
        offset=0.1,
        idd=np.ones(3),
        sigma=np.ones(3),
        let=np.ones(3),
        energy=100.0,
    )
    bixel = dict(kernel=kernel, rad_depths=np.array([0.7, 1.9]), rad_depth_offset=0.03)
    assert "alpha" not in engine._interpolate_kernels_in_depth(bixel)
    bixel["v_alpha_x"] = np.ones(2)
    for _ in range(2):
        actual = engine._interpolate_kernels_in_depth(bixel)
        assert actual["alpha"].shape == (2, 2)
        expected = interp1d(
            bixel["rad_depths"],
            kernel["depths"] + 0.1 - 0.03,
            engine._bio_evaluator.kernel_quantities.return_value,
        )
        np.testing.assert_array_equal(actual["alpha"], expected["alpha"])
    assert len(engine._depth_interpolator_cache) == 2
    engine._bio_evaluator.kernel_quantities.assert_called_once()
    engine._bio_evaluator.validate_kernel_quantities.assert_called_once()


def test_focus_cache_retains_extrapolation():
    engine = engine_stub()
    focus = SimpleNamespace(dist=np.array([100.0, 200.0, 300.0]), sigma=np.array([3.0, 2.0, 4.0]))
    engine._machine = SimpleNamespace(foci={100.0: [focus]})
    reference = scipy_interp1d(focus.dist, focus.sigma, fill_value="extrapolate")
    for ssd in (50.0, 150.0, 300.0, 350.0):
        np.testing.assert_array_equal(
            engine._calc_sigma_ini_focus_ix(100.0, 0, ssd), reference(ssd)
        )
    assert len(engine._focus_interpolator_cache) == 1


@pytest.mark.parametrize("trace_on_dose_grid", [False, True])
def test_reused_engine_rebuilds_prepared_state(test_data_protons, trace_on_dose_grid):
    pln, ct, cst, stf, _, _ = test_data_protons
    engine = ParticleHongPencilBeamEngine(pln)
    engine.console_progress = False
    engine.trace_on_dose_grid = trace_on_dose_grid
    first = engine.calc_dose_influence(ct, cst, stf)
    assert engine._depth_interpolator_cache
    assert engine._focus_interpolator_cache
    engine._depth_interpolator_cache["stale"] = object()
    for key in engine._focus_interpolator_cache:
        engine._focus_interpolator_cache[key] = lambda _: np.nan
    second = engine.calc_dose_influence(ct, cst, stf)
    assert "stale" not in engine._depth_interpolator_cache
    for quantity in first.quantities:
        a = getattr(first, quantity).flat[0]
        b = getattr(second, quantity).flat[0]
        assert (a != b).nnz == 0

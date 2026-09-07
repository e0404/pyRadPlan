"""Differential tests for the dispatchable ray tracer kernels."""

import numpy as np
import pytest

from pyRadPlan.raytracer import _kernels

try:
    import numba  # noqa: F401

    has_numba = True
except ImportError:
    has_numba = False

try:
    import jax
    import jax.numpy as jnp

    has_jax = True
except ImportError:
    jnp = None
    has_jax = False


RNG = np.random.default_rng(42)


def _indices_inputs():
    n, p = 50, 40
    source = RNG.normal(0.0, 100.0, (n, 3)).astype(np.float32)
    ray = RNG.normal(0.0, 200.0, (n, 3)).astype(np.float32)
    ray[0, :] = 0.0  # degenerate ray
    alphas = np.sort(RNG.random((n, p), dtype=np.float32), axis=1)
    alphas[RNG.random((n, p)) < 0.2] = np.nan
    origin = np.array([-100.0, -100.0, -50.0])
    resolution = np.array([2.0, 2.0, 3.0])
    cube_dim = np.array([100, 100, 40], dtype=np.int64)
    return source, ray, alphas, origin, resolution, cube_dim


def _selection_inputs():
    r, s = 60, 80
    nx, ny, nz = 40, 30, 20
    num_voxels = nx * ny * nz
    ix = RNG.integers(-5, num_voxels + 5, (r, s))
    index_to_bev = RNG.normal(0.0, 1.5, (3, 3)).astype(np.float32)
    # keep the BEV y coordinate positive and away from zero
    index_to_bev[:, 1] = np.abs(index_to_bev[:, 1]) + 1.0
    bev_offset = np.array([5.0, 50.0, -3.0], dtype=np.float32)
    ray_x = RNG.normal(0.0, 30.0, r).astype(np.float32)
    ray_z = RNG.normal(0.0, 30.0, r).astype(np.float32)
    return ix, index_to_bev, bev_offset, ray_x, ray_z, 500.0, 0.7, num_voxels, ny, nz


def _selection_boundary_margin(args):
    """Distance of each segment's selection test to a decision boundary, in float64."""
    ix, m, b, ray_x, ray_z, scale_num, ray_selection, num_voxels, ny, nz = args
    valid = (ix >= 0) & (ix < num_voxels)
    v = np.where(valid, ix, 0)
    tmp = v // nz
    coords = np.stack(
        [
            (tmp // ny).astype(np.float64),
            (tmp % ny).astype(np.float64),
            (v % nz).astype(np.float64),
        ],
        axis=-1,
    ) @ m.astype(np.float64) + b.astype(np.float64)
    scale = scale_num / coords[..., 1]
    x_dist = coords[..., 0] * scale - ray_x[:, None].astype(np.float64)
    z_dist = coords[..., 2] * scale - ray_z[:, None].astype(np.float64)
    return np.minimum(
        np.abs(np.abs(x_dist) - ray_selection), np.abs(np.abs(z_dist) - ray_selection)
    )


@pytest.mark.skipif(not has_numba, reason="numba not installed")
def test_indices_from_alpha_numba_matches_generic():
    args = _indices_inputs()
    assert _kernels.compute_indices_from_alpha._impls.get("numpy") is not None
    val_ref, ijk_ref = _kernels.compute_indices_from_alpha.generic(*args)
    val_out, ijk_out = _kernels.compute_indices_from_alpha(*args)
    np.testing.assert_array_equal(val_out, val_ref)
    np.testing.assert_array_equal(ijk_out, ijk_ref)


@pytest.mark.skipif(not has_numba, reason="numba not installed")
def test_select_segments_numba_matches_generic():
    args = _selection_inputs()
    assert _kernels.select_rad_depth_segments._impls.get("numpy") is not None
    ref = _kernels.select_rad_depth_segments.generic(*args)
    out = _kernels.select_rad_depth_segments(*args)
    # the fused float32 arithmetic may resolve boundary ties differently than the
    # generic GEMM; any disagreement must sit at a decision boundary
    mismatch = out != ref
    if np.any(mismatch):
        margin = _selection_boundary_margin(args)
        assert np.all(margin[mismatch] < 1e-3)
    assert np.count_nonzero(mismatch) <= 0.001 * mismatch.size


@pytest.mark.skipif(not has_numba, reason="numba not installed")
def test_select_segments_numba_falls_back_for_float64():
    args = list(_selection_inputs())
    args[1] = args[1].astype(np.float64)
    args[2] = args[2].astype(np.float64)
    ref = _kernels.select_rad_depth_segments.generic(*args)
    out = _kernels.select_rad_depth_segments(*args)
    np.testing.assert_array_equal(out, ref)


@pytest.mark.skipif(not has_jax, reason="jax not installed")
def test_jit_backends_setting_gates_jit(monkeypatch):
    from pyRadPlan._settings import get_settings
    from pyRadPlan.core.xp_utils.jittable import jittable

    @jittable(backends=("jax",))
    def double(x):
        return x * 2

    arr = jnp.arange(3)

    monkeypatch.setattr(get_settings().xp, "jit_backends", "")
    np.testing.assert_array_equal(np.asarray(double(arr)), [0, 2, 4])
    assert double._jitted == {}

    monkeypatch.setattr(get_settings().xp, "jit_backends", "jax")
    np.testing.assert_array_equal(np.asarray(double(arr)), [0, 2, 4])
    assert "jax" in double._jitted


def test_jit_backends_setting_gates_registered_impls(monkeypatch):
    from pyRadPlan._settings import get_settings
    from pyRadPlan.core.xp_utils.jittable import jittable

    @jittable(backends=())
    def double(x):
        return x * 2

    calls = []

    @double.register("numpy")
    def _double_numpy(x):
        calls.append(1)
        return x * 2

    arr = np.arange(3)

    monkeypatch.setattr(get_settings().xp, "jit_backends", "")
    np.testing.assert_array_equal(double(arr), [0, 2, 4])
    assert calls == []

    monkeypatch.setattr(get_settings().xp, "jit_backends", "numpy")
    np.testing.assert_array_equal(double(arr), [0, 2, 4])
    assert calls == [1]


@pytest.mark.skipif(not has_jax, reason="jax not installed")
def test_jax_jit_matches_generic():
    source, ray, alphas, origin, resolution, cube_dim = _indices_inputs()
    jargs = tuple(jnp.asarray(a) for a in (source, ray, alphas, origin, resolution, cube_dim))
    val_ref, ijk_ref = _kernels.compute_indices_from_alpha.generic(*jargs)
    val_out, ijk_out = _kernels.compute_indices_from_alpha(*jargs)
    np.testing.assert_array_equal(np.asarray(val_out), np.asarray(val_ref))
    np.testing.assert_array_equal(np.asarray(ijk_out), np.asarray(ijk_ref))

    limits = jnp.asarray(np.sort(RNG.random((20, 2), dtype=np.float32), axis=1))
    ax, ay, az = (
        jnp.asarray(np.sort(RNG.random((20, 15), dtype=np.float32), axis=1)) for _ in range(3)
    )
    merged_ref = _kernels.merge_sorted_unique.generic(limits, ax, ay, az)
    merged_out = _kernels.merge_sorted_unique(limits, ax, ay, az)
    np.testing.assert_array_equal(np.asarray(merged_out), np.asarray(merged_ref))

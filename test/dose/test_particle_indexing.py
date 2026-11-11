"""Particle support selection preserves row identity across all ray quantities."""

import array_api_compat
import numpy as np
import pytest

from pyRadPlan.core.xp_utils import to_numpy
from pyRadPlan.dose.engines import ParticleHongPencilBeamEngine


@pytest.fixture(params=["numpy", "cupy", "strict2023", "strict2025"])
def xp(request):
    """Exercise native arrays and strict versions without boolean indexing."""
    if request.param.startswith("strict"):
        strict = pytest.importorskip("array_api_strict")
        # Table-cutoff interpolation already requires clip, introduced in 2023.12.
        version = request.param.removeprefix("strict") + ".12"
        with strict.ArrayAPIStrictFlags(api_version=version, boolean_indexing=False):
            yield strict
    else:
        backend = pytest.importorskip(request.param)
        if request.param == "cupy":
            try:
                backend.zeros(1)
            except backend.cuda.runtime.CUDARuntimeError:
                pytest.skip("CUDA device unavailable")
        yield array_api_compat.array_namespace(backend.zeros(1))


@pytest.mark.parametrize("selection", ["empty", "none", "all", "mixed"])
@pytest.mark.parametrize("optional_fields", [False, True])
@pytest.mark.parametrize("cutoff", ["full", "constant", "table"])
def test_particle_support_selection(xp, selection, optional_fields, cutoff):
    """Select physical rows, preserving unsorted voxel IDs and field shapes/dtypes."""
    engine = object.__new__(ParticleHongPencilBeamEngine)
    engine.dosimetric_lateral_cutoff = 1.0 if cutoff == "full" else 0.995
    engine._calc_bio_dose = optional_fields
    n = 0 if selection == "empty" else 5
    depths = np.array([2, 6, 1, 3, 0], dtype=np.float32)[:n]
    radial = np.array([1, 1, 9, 4, 1], dtype=np.float32)[:n]
    if selection == "none":
        depths[:] = 99
    elif selection == "all":
        depths[:] = 2
        radial[:] = 1
    source = {
        "ix": np.array([29, 8, 17, 42, 3], dtype=np.int64)[:n],
        "rad_depths": depths,
        "radial_dist_sq": radial,
    }
    if optional_fields:
        source.update(
            lat_dists=np.arange(n * 2, dtype=np.float32).reshape(n, 2),
            v_alpha_x=np.arange(n, dtype=np.float64).reshape(n, 1),
            v_beta_x=np.arange(n, dtype=np.float32),
            v_tissue_class_ix=np.array([2, 0, 1, 2, 1], dtype=np.int32)[:n],
        )
    ray = {name: xp.asarray(value) for name, value in source.items()}
    cut_off = [2.0, 2.0] if cutoff == "table" else [2.0]
    bixel = {
        "rad_depth_offset": 0.5,
        "kernel": {
            "offset": 1.5,
            "depths": xp.asarray([0.0, 3.0]),
            "lateral_cut_off": {
                "depths": xp.asarray([0.0, 3.0]),
                "cut_off": xp.asarray(cut_off),
            },
        },
    }
    if selection in ("empty", "none"):
        selected = []
    elif selection == "all":
        selected = list(range(n))
    else:
        selected = {"full": [0, 2, 3, 4], "constant": [0, 3, 4], "table": [0, 3]}[cutoff]

    result = engine._get_bixel_indices_on_ray(bixel, ray)

    assert result is bixel
    for name, original in source.items():
        actual = to_numpy(result[name])
        np.testing.assert_array_equal(actual, original[selected])
        assert actual.dtype == original.dtype
        assert array_api_compat.device(result[name]) == array_api_compat.device(ray[name])
        np.testing.assert_array_equal(to_numpy(ray[name]), original)
    if not optional_fields:
        assert "lat_dists" not in result
        assert "v_alpha_x" not in result

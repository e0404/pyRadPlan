import array_api_compat
import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models import ExactClassLookup, TissueParameterLookup, make_tissue_lookup
from pyRadPlan.core.xp_utils import get_device_info


@pytest.fixture
def lookup():
    return ExactClassLookup([0.1, 0.5], [0.05, 0.05])


def test_make_tissue_lookup(lookup):
    made = make_tissue_lookup("exact", [0.1, 0.5], [0.05, 0.05])
    assert isinstance(made, ExactClassLookup)
    assert isinstance(made, TissueParameterLookup)
    with pytest.raises(ValueError, match="Unknown tissue lookup"):
        make_tissue_lookup("nearest", [0.1], [0.05])


def test_exact_class_index(lookup):
    v_alpha_x = np.asarray([[0.1, 0.1], [0.5, 0.5], [0.0, 0.5]])
    v_beta_x = np.asarray([[0.05, 0.05], [0.05, 0.05], [0.0, 0.05]])
    assert np.array_equal(lookup.class_index(v_alpha_x, v_beta_x), [[0, 0], [1, 1], [0, 1]])

    ix_1d = lookup.class_index(xp.asarray([0.5, 0.1]), xp.asarray([0.05, 0.05]))
    assert np.array_equal(np.asarray(ix_1d), [1, 0])


def test_exact_validate(lookup):
    lookup.validate({"alpha_x": np.asarray([[0.1], [0.0]]), "beta_x": np.asarray([[0.05], [0.0]])})
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.validate({"alpha_x": np.asarray([[0.3]]), "beta_x": np.asarray([[0.05]])})


def test_exact_gather(lookup):
    alpha_x = xp.asarray([0.1, 0.5, 0.1, 0.0])
    beta_x = xp.asarray([0.05, 0.05, 0.05, 0.0])
    kernels = {
        "alpha": xp.asarray([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        "beta": xp.asarray([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]]),
        "let": xp.asarray([9.0, 9.0, 9.0, 9.0]),  # not requested, must be ignored
    }
    rows = lookup.gather(alpha_x, beta_x, kernels, ["alpha", "beta"])
    assert set(rows) == {"alpha", "beta"}
    assert np.allclose(np.asarray(rows["alpha"]), [1.0, 2.0, 1.0, 1.0])
    assert np.allclose(np.asarray(rows["beta"]), [4.0, 5.0, 4.0, 4.0])


def test_exact_gather_still_validates_before_the_grid_was_checked(lookup):
    """Without a validated dose grid, an unknown tissue is caught in the gather itself."""
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.gather(
            xp.asarray([0.3]),
            xp.asarray([0.05]),
            {"alpha": xp.asarray([[1.0], [2.0]])},
            ["alpha"],
        )


def test_exact_gather_skips_the_check_once_the_grid_is_validated(lookup):
    """Per-bixel parameters are a subset of the validated grid, so the check is redundant."""
    lookup.validate(
        {"alpha_x": np.asarray([[0.1], [0.5]]), "beta_x": np.asarray([[0.05], [0.05]])}
    )
    rows = lookup.gather(
        xp.asarray([0.5, 0.1]),
        xp.asarray([0.05, 0.05]),
        {"alpha": xp.asarray([[1.0, 1.0], [2.0, 2.0]])},
        ["alpha"],
    )
    assert np.allclose(np.asarray(rows["alpha"]), [2.0, 1.0])


def test_exact_validate_failure_does_not_disable_the_gather_check(lookup):
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.validate({"alpha_x": np.asarray([[0.3]]), "beta_x": np.asarray([[0.05]])})
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.gather(
            xp.asarray([0.3]),
            xp.asarray([0.05]),
            {"alpha": xp.asarray([[1.0], [2.0]])},
            ["alpha"],
        )


def test_exact_validate_reports_missing_voxel_parameters(lookup):
    with pytest.raises(ValueError, match="needs the per-voxel"):
        lookup.validate({"alpha_x": np.asarray([[0.1]])})


def test_reference_arrays_are_converted_once_per_namespace_device_and_dtype(lookup):
    alpha_x = xp.asarray([0.1, 0.5])
    beta_x = xp.asarray([0.05, 0.05])
    lookup.class_index(alpha_x, beta_x)
    first = lookup._reference(alpha_x)
    lookup.class_index(alpha_x, beta_x)
    assert lookup._reference(alpha_x)[1] is first[1]

    # a different namespace gets its own entry rather than reusing the cached arrays
    lookup.class_index(np.asarray([0.1, 0.5]), np.asarray([0.05, 0.05]))
    assert len(lookup._reference_cache) == 2


def test_lookup_rejects_inconsistent_class_declarations():
    with pytest.raises(ValueError, match="2 reference alpha_x but 1 reference beta_x"):
        ExactClassLookup([0.1, 0.5], [0.05])
    with pytest.raises(ValueError, match="at least one reference tissue class"):
        ExactClassLookup([], [])


def test_reference_cache_key_is_hashable(lookup):
    """Not every backend device object is hashable (``cupy.cuda.Device`` is not)."""
    lookup.class_index(xp.asarray([0.1]), xp.asarray([0.05]))

    (key,) = lookup._reference_cache
    namespace, _device, _dtype = key
    assert hash(key)
    assert namespace is array_api_compat.array_namespace(xp.asarray([0.1]))


def test_reference_cache_distinguishes_logical_devices(lookup):
    """Every array_api_strict device shares the DLPack tuple (1, 0), so it cannot be the key."""
    devices = [xp.Device("CPU_DEVICE"), xp.Device("device1")]
    assert len({get_device_info(xp.asarray([0.1], device=d)) for d in devices}) == 1

    for device in devices:
        class_ix = lookup.class_index(
            xp.asarray([0.1, 0.5], device=device), xp.asarray([0.05, 0.05], device=device)
        )
        assert array_api_compat.device(class_ix) == device

    assert len(lookup._reference_cache) == len(devices)


def test_gather_works_on_a_non_default_logical_device(lookup):
    """A cached reference from another device must not leak into this one."""
    lookup.class_index(xp.asarray([0.1]), xp.asarray([0.05]))  # warm the CPU_DEVICE entry

    device = xp.Device("device1")
    rows = lookup.gather(
        xp.asarray([0.5, 0.1], device=device),
        xp.asarray([0.05, 0.05], device=device),
        {"alpha": xp.asarray([[1.0, 1.0], [2.0, 2.0]], device=device)},
        ["alpha"],
    )

    assert array_api_compat.device(rows["alpha"]) == device
    on_host = xp.asarray(rows["alpha"], device=xp.Device("CPU_DEVICE"))
    assert np.allclose(np.asarray(on_host), [2.0, 1.0])

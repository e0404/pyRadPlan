import array_api_compat
import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.biology.models import ExactClassLookup, TissueParameterLookup, make_tissue_lookup
from pyRadPlan.biology.tissues import (
    TissueProperties,
    LEMTissueProperties,
    MKMTissueProperties,
)
from pyRadPlan.core.xp_utils import get_device_info


@pytest.fixture
def tissue_classes():
    return [TissueProperties(alpha_x=0.1, beta_x=0.05), TissueProperties(alpha_x=0.5, beta_x=0.05)]


@pytest.fixture
def lookup(tissue_classes):
    return ExactClassLookup(tissue_classes, [0, 1])


def test_make_tissue_lookup(tissue_classes):
    made = make_tissue_lookup("exact", tissue_classes, np.asarray(tissue_classes))
    assert isinstance(made, ExactClassLookup)
    assert isinstance(made, TissueParameterLookup)
    with pytest.raises(ValueError, match="Unknown tissue lookup"):
        make_tissue_lookup("nearest", tissue_classes, np.asarray(tissue_classes))


def test_make_tissue_lookup_rejects_unmatched_cst_tissue(tissue_classes):
    with pytest.raises(ValueError, match="No matching tissue class"):
        make_tissue_lookup(
            "exact", tissue_classes, np.asarray([TissueProperties(alpha_x=0.3, beta_x=0.05)])
        )


def test_make_tissue_lookup_matches_lem_on_model_parameters():
    """LEM tissues are identified by their reference parameters *and* cell geometry."""
    machine = [LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)]
    cst = np.asarray([LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)])
    assert isinstance(make_tissue_lookup("exact", machine, cst), ExactClassLookup)

    deviating = np.asarray([LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=6.0, r_nucleus=5.0)])
    with pytest.raises(ValueError, match="No matching tissue class"):
        make_tissue_lookup("exact", machine, deviating)


def test_make_tissue_lookup_matches_mkm_on_geometry_only():
    """MKM zs kernels only depend on the domain/nucleus geometry."""
    machine = [MKMTissueProperties(alpha_x=float("nan"), r_nucleus=3.9, r_domain=0.32)]
    cst = np.asarray(
        [MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.172, beta_0=0.0615)]
    )
    assert isinstance(make_tissue_lookup("exact", machine, cst), ExactClassLookup)


def test_make_tissue_lookup_does_not_mix_lem_and_mkm():
    machine = [LEMTissueProperties(alpha_x=0.1, beta_x=0.05, d_t=30.0, r_nucleus=5.0)]
    cst = np.asarray([MKMTissueProperties(alpha_x=0.1, beta_x=0.05, r_nucleus=5.0)])
    with pytest.raises(ValueError, match="No matching tissue class"):
        make_tissue_lookup("exact", machine, cst)


def test_lookup_rejects_inconsistent_class_declarations(tissue_classes):
    with pytest.raises(ValueError, match="2 reference classes but 1 class indices"):
        ExactClassLookup(tissue_classes, [0])
    with pytest.raises(ValueError, match="at least one reference tissue class"):
        ExactClassLookup([], [])


def test_exact_class_index(lookup):
    v_tissue_class_ix = np.asarray([[0, 0], [1, 1], [0, 1]])
    assert np.array_equal(lookup.class_index(v_tissue_class_ix), [[0, 0], [1, 1], [0, 1]])

    ix_1d = lookup.class_index(xp.asarray([1, 0]))
    assert np.array_equal(np.asarray(ix_1d), [1, 0])


def test_exact_validate(lookup):
    lookup.validate({"tissue_class_ix": np.asarray([[1], [0]])})
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.validate({"tissue_class_ix": np.asarray([[3]])})


def test_exact_gather(lookup):
    context = {
        "alpha_x": xp.asarray([0.1, 0.5, 0.1, 0.0]),
        "beta_x": xp.asarray([0.05, 0.05, 0.05, 0.0]),
        "tissue_class_ix": xp.asarray([0, 1, 0, 0]),
        # interpolated kernels: (n_classes, n_voxels)
        "alpha": xp.asarray([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]),
        "beta": xp.asarray([[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]]),
        "let": xp.asarray([9.0, 9.0, 9.0, 9.0]),  # not requested, must be ignored
    }
    rows = lookup.gather(context, ["alpha", "beta"])
    assert set(rows) == {"alpha", "beta"}
    assert np.allclose(np.asarray(rows["alpha"]), [1.0, 2.0, 1.0, 1.0])
    assert np.allclose(np.asarray(rows["beta"]), [4.0, 5.0, 4.0, 4.0])


def test_exact_gather_takes_one_dimensional_kernels_per_voxel(lookup):
    """A kernel that is not tabulated per tissue class is used as-is."""
    context = {
        "alpha_x": xp.asarray([0.1, 0.5]),
        "tissue_class_ix": xp.asarray([0, 1]),
        "zs": xp.asarray([7.0, 8.0]),
    }
    rows = lookup.gather(context, ["zs"])
    assert np.allclose(np.asarray(rows["zs"]), [7.0, 8.0])


def test_exact_gather_still_validates_before_the_grid_was_checked(lookup):
    """Without a validated dose grid, an unknown tissue is caught in the gather itself."""
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.gather(
            {
                "alpha_x": xp.asarray([0.3]),
                "tissue_class_ix": xp.asarray([3]),
                "alpha": xp.asarray([[1.0], [2.0]]),
            },
            ["alpha"],
        )


def test_exact_gather_skips_the_check_once_the_grid_is_validated(lookup):
    """Per-bixel indices are a subset of the validated grid, so the check is redundant."""
    lookup.validate({"tissue_class_ix": np.asarray([[1], [0]])})
    rows = lookup.gather(
        {
            "alpha_x": xp.asarray([0.5, 0.1]),
            "tissue_class_ix": xp.asarray([1, 0]),
            "alpha": xp.asarray([[1.0, 1.0], [2.0, 2.0]]),
        },
        ["alpha"],
    )
    assert np.allclose(np.asarray(rows["alpha"]), [2.0, 1.0])


def test_exact_validate_failure_does_not_disable_the_gather_check(lookup):
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.validate({"tissue_class_ix": np.asarray([[3]])})
    with pytest.raises(ValueError, match="No matching tissue class"):
        lookup.gather(
            {
                "alpha_x": xp.asarray([0.3]),
                "tissue_class_ix": xp.asarray([3]),
                "alpha": xp.asarray([[1.0], [2.0]]),
            },
            ["alpha"],
        )


def test_exact_validate_reports_missing_voxel_parameters(lookup):
    with pytest.raises(ValueError, match="needs the per-voxel"):
        lookup.validate({"alpha_x": np.asarray([[0.1]])})


def test_gather_tissue_parameters_expands_the_class_parameters_onto_the_voxels():
    """MKM reads alpha_0 / beta_0 of a voxel's tissue class alongside its zs kernels."""
    classes = [
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05),
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.2, beta_0=0.06),
    ]
    lookup = ExactClassLookup(classes, [0, 1])
    context = {
        "alpha_x": xp.asarray([0.1, 0.2, 0.1]),
        "tissue_class_ix": xp.asarray([0, 1, 1]),
    }

    rows = lookup.gather_tissue_parameters(context, ["alpha_0", "beta_0"])

    assert np.allclose(np.asarray(rows["alpha_0"]), [0.1, 0.2, 0.2])
    assert np.allclose(np.asarray(rows["beta_0"]), [0.05, 0.06, 0.06])
    assert lookup.gather_tissue_parameters(context, []) == {}


def _count_host_conversions(monkeypatch):
    """Count ``asarray`` calls on host data (a per-bixel host→device copy on a GPU)."""
    calls = []
    asarray = xp.asarray

    def counting(obj, *args, **kwargs):
        if isinstance(obj, (list, tuple, np.ndarray)):
            calls.append(obj)
        return asarray(obj, *args, **kwargs)

    monkeypatch.setattr(xp, "asarray", counting)
    return calls


def _gather_tissue_parameters_per_bixel(lookup, context, fields):
    """Gather as before the cache, converting the class parameters on every call."""
    class_ix = lookup.class_index(context["tissue_class_ix"])
    device = array_api_compat.device(class_ix)
    return {
        name: xp.take(
            xp.asarray(
                [float(getattr(tissue, name)) for tissue in lookup.tissue_classes],
                dtype=context["alpha_x"].dtype,
                device=device,
            ),
            class_ix,
            axis=0,
        )
        for name in fields
    }


def _mkm_bixels(n_bixels, device=None):
    rng = np.random.default_rng(0)
    return [
        {
            "alpha_x": xp.asarray(rng.random(5), device=device),
            "tissue_class_ix": xp.asarray(rng.integers(0, 2, 5), device=device),
        }
        for _ in range(n_bixels)
    ]


def test_gather_tissue_parameters_converts_the_classes_once_per_calculation(monkeypatch):
    """Per bixel only device arrays are indexed; the class parameters are copied once."""
    classes = [
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05),
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.2, beta_0=0.06),
    ]
    lookup = ExactClassLookup(classes, [0, 1])
    bixels = _mkm_bixels(20)
    calls = _count_host_conversions(monkeypatch)

    lookup.gather_tissue_parameters(bixels[0], ["alpha_0", "beta_0"])
    first = len(calls)
    for context in bixels[1:]:
        lookup.gather_tissue_parameters(context, ["alpha_0", "beta_0"])

    # class indices, kernel rows, alpha_0 and beta_0
    assert first == 4
    assert len(calls) == first


def test_gather_tissue_parameters_matches_the_per_bixel_conversion():
    """The cached class parameters give identical results on every device."""
    classes = [
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.1, beta_0=0.05),
        MKMTissueProperties(r_nucleus=3.9, r_domain=0.32, alpha_0=0.2, beta_0=0.06),
    ]
    lookup = ExactClassLookup(classes, [0, 1])
    for device in (None, xp.Device("device1")):
        for context in _mkm_bixels(5, device=device):
            cached = lookup.gather_tissue_parameters(context, ["alpha_0", "beta_0"])
            expected = _gather_tissue_parameters_per_bixel(lookup, context, ["alpha_0", "beta_0"])
            for name, values in expected.items():
                assert array_api_compat.device(cached[name]) == array_api_compat.device(values)
                assert cached[name].dtype == values.dtype
                assert bool(xp.all(cached[name] == values))
    # one entry per parameter and device
    assert len(lookup._parameter_cache) == 4


def test_reference_arrays_are_converted_once_per_namespace_device_and_dtype(lookup):
    v_tissue_class_ix = xp.asarray([0, 1])
    lookup.class_index(v_tissue_class_ix)
    _, first_classes, first_rows = lookup._reference(v_tissue_class_ix)
    lookup.class_index(v_tissue_class_ix)
    _, again_classes, again_rows = lookup._reference(v_tissue_class_ix)
    assert again_classes is first_classes
    assert again_rows is first_rows

    # a different namespace gets its own entry rather than reusing the cached arrays
    lookup.class_index(np.asarray([0, 1]))
    assert len(lookup._reference_cache) == 2


def test_reference_cache_key_is_hashable(lookup):
    """Not every backend device object is hashable (``cupy.cuda.Device`` is not)."""
    lookup.class_index(xp.asarray([0]))

    (key,) = lookup._reference_cache
    namespace, _device, _dtype = key
    assert hash(key)
    assert namespace is array_api_compat.array_namespace(xp.asarray([0]))


def test_reference_cache_distinguishes_logical_devices(lookup):
    """Every array_api_strict device shares the DLPack tuple (1, 0), so it cannot be the key."""
    devices = [xp.Device("CPU_DEVICE"), xp.Device("device1")]
    assert len({get_device_info(xp.asarray([0.1], device=d)) for d in devices}) == 1

    for device in devices:
        class_ix = lookup.class_index(xp.asarray([0, 1], device=device))
        assert array_api_compat.device(class_ix) == device

    assert len(lookup._reference_cache) == len(devices)


def test_gather_works_on_a_non_default_logical_device(lookup):
    """A cached reference from another device must not leak into this one."""
    lookup.class_index(xp.asarray([0]))  # warm the CPU_DEVICE entry

    device = xp.Device("device1")
    rows = lookup.gather(
        {
            "alpha_x": xp.asarray([0.5, 0.1], device=device),
            "tissue_class_ix": xp.asarray([1, 0], device=device),
            "alpha": xp.asarray([[1.0, 1.0], [2.0, 2.0]], device=device),
        },
        ["alpha"],
    )

    assert array_api_compat.device(rows["alpha"]) == device
    on_host = xp.asarray(rows["alpha"], device=xp.Device("CPU_DEVICE"))
    assert np.allclose(np.asarray(on_host), [2.0, 1.0])

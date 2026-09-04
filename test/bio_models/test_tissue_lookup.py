import numpy as np
import pytest
import array_api_strict as xp

from pyRadPlan.bio_models import ExactClassLookup, TissueParameterLookup, make_tissue_lookup


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

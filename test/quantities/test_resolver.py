import pytest

import numpy as np

from pyRadPlan.dij import Dij
from pyRadPlan.quantities import (
    AlphaDose,
    Dose,
    Effect,
    LETxDose,
    QuantityResolver,
    RBExDose,
    SqrtBetaDose,
)


@pytest.fixture
def base_dij_dict():
    return {
        "ct_grid": {
            "resolution": {"x": 1.5, "y": 1.5, "z": 1.5},
            "dimensions": (10, 10, 10),
            "num_of_voxels": 1000,
        },
        "dose_grid": {
            "resolution": {"x": 3.0, "y": 3.0, "z": 3.0},
            "dimensions": (5, 5, 5),
            "num_of_voxels": 125,
        },
        "num_of_beams": 1,
        "total_num_of_bixels": 10,
        "bixel_num": np.arange(10),
        "ray_num": np.arange(10),
        "beam_num": np.zeros((10,), dtype=np.int64),
        "alphax": np.ones(125, dtype=np.float32),
        "betax": np.ones(125, dtype=np.float32),
    }


def _fill(container, value):
    container.flat[0] = value


@pytest.fixture
def full_dij(base_dij_dict):
    """Dij with all influence matrices: physical_dose, let_dose, alpha_dose, sqrt_beta_dose."""
    mat = lambda: np.ones((125, 10), dtype=np.float32)  # noqa: E731
    base_dij_dict["physical_dose"] = np.empty((1, 1, 1), dtype=object)
    base_dij_dict["let_dose"] = np.empty((1, 1, 1), dtype=object)
    base_dij_dict["alpha_dose"] = np.empty((1, 1, 1), dtype=object)
    base_dij_dict["sqrt_beta_dose"] = np.empty((1, 1, 1), dtype=object)
    _fill(base_dij_dict["physical_dose"], mat())
    _fill(base_dij_dict["let_dose"], mat())
    _fill(base_dij_dict["alpha_dose"], mat())
    _fill(base_dij_dict["sqrt_beta_dose"], mat())
    return Dij.model_validate(base_dij_dict)


@pytest.fixture
def dij_no_alpha_dose(base_dij_dict):
    """Dij without alpha_dose — forces AlphaDose into its indirect fallback path."""
    mat = lambda: np.ones((125, 10), dtype=np.float32)  # noqa: E731
    base_dij_dict["physical_dose"] = np.empty((1, 1, 1), dtype=object)
    _fill(base_dij_dict["physical_dose"], mat())
    return Dij.model_validate(base_dij_dict)


def test_resolver_shares_instances_between_roots(full_dij):
    """alpha_dose requested both directly and via effect must yield a single instance."""
    resolver = QuantityResolver(full_dij)
    resolver.resolve(["effect", "alpha_dose"])

    effect = resolver.instances["effect"]
    alpha_root = resolver.instances["alpha_dose"]
    assert effect.dependencies["alpha_dose"] is alpha_root


def test_resolver_idempotent_get(full_dij):
    resolver = QuantityResolver(full_dij)
    a = resolver.get("physical_dose")
    b = resolver.get("physical_dose")
    assert a is b


def test_resolver_picks_direct_mode_when_matrix_present(full_dij):
    resolver = QuantityResolver(full_dij)
    alpha = resolver.get("alpha_dose")
    assert isinstance(alpha, AlphaDose)
    assert alpha.mode == "direct"


def test_resolver_falls_back_to_indirect_when_matrix_missing(dij_no_alpha_dose):
    resolver = QuantityResolver(dij_no_alpha_dose)
    alpha = resolver.get("alpha_dose")
    assert isinstance(alpha, AlphaDose)
    assert alpha.mode == "indirect"
    assert "physical_dose" in alpha.dependencies
    assert isinstance(alpha.dependencies["physical_dose"], Dose)


def test_resolver_unknown_identifier_raises(full_dij):
    resolver = QuantityResolver(full_dij)
    with pytest.raises(ValueError, match="Unknown quantity identifier"):
        resolver.get("not_a_real_quantity")


def test_resolver_resolves_transitive_dependencies(full_dij):
    """Requesting rbe_x_dose pulls in effect, alpha_dose, sqrt_beta_dose transitively."""
    resolver = QuantityResolver(full_dij)
    resolver.resolve(["rbe_x_dose"])
    keys = set(resolver.instances)
    assert keys == {"rbe_x_dose", "effect", "alpha_dose", "sqrt_beta_dose"}
    assert isinstance(resolver.instances["rbe_x_dose"], RBExDose)
    assert isinstance(resolver.instances["effect"], Effect)
    assert isinstance(resolver.instances["alpha_dose"], AlphaDose)
    assert isinstance(resolver.instances["sqrt_beta_dose"], SqrtBetaDose)


def test_resolver_dij_is_namespace_converted_only_once(full_dij):
    resolver = QuantityResolver(full_dij)
    resolver.resolve(["effect", "alpha_dose", "sqrt_beta_dose"])
    # All instances share the resolver's converted dij.
    converted = resolver.dij
    for inst in resolver.instances.values():
        assert inst._dij is converted


def test_let_dose_root_works(full_dij):
    resolver = QuantityResolver(full_dij)
    let = resolver.get("let_dose")
    assert isinstance(let, LETxDose)
    assert let.mode == "direct"

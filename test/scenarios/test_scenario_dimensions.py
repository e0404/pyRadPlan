"""Uncertainty dimensions and joint distributions."""

import numpy as np
import pytest
from pydantic import ValidationError

from pyRadPlan.scenarios import (
    CTPhaseUncertainty,
    IndependentGaussian,
    MultivariateGaussian,
    RangeUncertainty,
    SetupUncertainty,
    UncertaintyDimension,
    available_dimensions,
    available_distributions,
    create_dimension,
    create_distribution,
)
from pyRadPlan.scenarios._dimensions import register_dimension


def test_setup_dimension_defaults():
    d = SetupUncertainty()
    assert d.name == "setup" and d.kind == "continuous"
    assert d.keys == ("setup.x", "setup.y", "setup.z")
    assert d.affects == {"geometry"}
    assert np.array_equal(d.std(), [2.25, 2.25, 2.25])
    assert np.array_equal(d.mean(), np.zeros(3))
    assert np.array_equal(d.nominal(), np.zeros(3))
    with pytest.raises(TypeError):
        d.support()
    with pytest.raises(ValidationError):
        SetupUncertainty(sd=(-1, 1, 1))


def test_range_dimension_percent_to_fraction():
    d = RangeUncertainty(abs_sd=2.0, rel_sd=5.0)
    assert d.keys == ("range.abs", "range.rel")
    assert np.allclose(d.std(), [2.0, 0.05])
    assert d.affects == {"range"}
    assert d.combined is True


def test_ct_phase_dimension():
    d = CTPhaseUncertainty(phases=[0, 2], probabilities=[0.3, 0.7])
    assert d.kind == "discrete"
    assert np.array_equal(d.support(), [[0], [2]])
    assert np.array_equal(d.pmf(), [0.3, 0.7])
    assert np.array_equal(d.nominal(), [0])
    assert d.pairs() == [(0, 0.3), (2, 0.7)]
    assert d.affects == {"geometry", "voxel_params"}
    with pytest.raises(TypeError):
        d.std()
    with pytest.raises(ValidationError):
        CTPhaseUncertainty(phases=[0, 0], probabilities=[0.5, 0.5])
    with pytest.raises(ValidationError):
        CTPhaseUncertainty(phases=[0], probabilities=[0.5, 0.5])
    with pytest.raises(ValidationError):
        CTPhaseUncertainty(phases=[0], probabilities=[1.5])
    assert CTPhaseUncertainty.from_pairs([(1, 0.5)]).pairs() == [(1, 0.5)]
    assert CTPhaseUncertainty.from_pairs(np.array([[0, 0.25], [1, 0.75]])).phases == [0, 1]


class _FakeCT:
    num_of_ct_scen = 2


def test_ct_phase_validate_against_ct():
    CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5]).validate_against_ct(_FakeCT())
    with pytest.raises(ValueError, match="only has 2"):
        CTPhaseUncertainty(phases=[0, 2], probabilities=[0.5, 0.5]).validate_against_ct(_FakeCT())
    d = CTPhaseUncertainty.from_ct(_FakeCT())
    assert d.phases == [0, 1] and np.allclose(d.probabilities, [0.5, 0.5])
    SetupUncertainty().validate_against_ct(_FakeCT())  # no-op


def test_coerce_and_registry():
    d = SetupUncertainty()
    assert np.array_equal(d.coerce([1, 2, 3]), [1, 2, 3])
    with pytest.raises(ValueError):
        d.coerce([1, 2])
    assert {"setup", "range", "ct_phase"} <= set(available_dimensions())
    assert create_dimension("setup") == SetupUncertainty()
    assert create_dimension({"dimension": "range", "abs_sd": 2}) == RangeUncertainty(abs_sd=2)
    assert create_dimension(d) is d
    with pytest.raises(ValueError):
        create_dimension({"abs_sd": 2})
    with pytest.raises(ValueError):
        create_dimension("bio")
    with pytest.raises(TypeError):
        create_dimension(3)
    # dumps carry the dimension name so sets can be rebuilt
    assert d.model_dump()["dimension"] == "setup"
    assert create_dimension(d.model_dump()) == d


def test_subclass_with_unknown_effect_is_rejected():
    with pytest.raises(TypeError):

        class Bad(UncertaintyDimension):  # noqa: D401
            name = "bad"
            components = ("a",)
            kind = "continuous"
            units = ("",)
            affects = frozenset({"colour"})


def test_custom_dimension_registers_and_round_trips():
    @register_dimension
    class _HUScale(UncertaintyDimension):
        name = "hu_scale_test"
        components = ("factor",)
        kind = "continuous"
        units = ("",)
        affects = frozenset({"range"})
        sd: float = 0.03

        def mean(self):
            return np.zeros(1)

        def std(self):
            return np.asarray([self.sd])

    try:
        d = create_dimension({"dimension": "hu_scale_test", "sd": 0.05})
        assert isinstance(d, _HUScale) and d.sd == 0.05
    finally:
        available_dimensions()  # registry copy; remove the test entry from the real one
        from pyRadPlan.scenarios import _dimensions

        _dimensions._DIMENSION_REGISTRY.pop("hu_scale_test", None)


# ----------------------------------------------------------------- distributions


def _dims():
    return (CTPhaseUncertainty(), SetupUncertainty(), RangeUncertainty())


def test_independent_gaussian_matches_matrad_formula():
    dims = _dims()
    dist = IndependentGaussian(dims)
    assert dist.keys == ("setup.x", "setup.y", "setup.z", "range.abs", "range.rel")
    assert dist.ndim == 5
    sigma = np.array([2.25, 2.25, 2.25, 1.0, 0.035])
    pts = np.array([[0, 0, 0, 0, 0], [2.25, 0, 0, 1, 0.035], [1, -1, 0.5, -0.5, 0.01]])
    z = pts / sigma
    expected = (2 * np.pi) ** (-2.5) * np.exp(-0.5 * np.sum(z**2, axis=1)) / np.prod(sigma)
    assert np.allclose(dist.pdf(pts), expected)
    assert np.allclose(dist.std(), sigma)
    assert np.allclose(dist.mean(), np.zeros(5))
    samples = dist.sample(1000, np.random.default_rng(0))
    assert samples.shape == (1000, 5)
    assert np.allclose(samples.std(axis=0), sigma, rtol=0.15)
    assert dist.to_dict() == {"distribution": "independent_gaussian"}


def test_independent_gaussian_zero_sd_component_is_deterministic():
    dims = (SetupUncertainty(sd=(0, 2.25, 2.25)),)
    dist = IndependentGaussian(dims)
    p = dist.pdf([[0, 0, 0], [1, 0, 0]])
    assert p[0] > 0 and p[1] == 0
    assert np.allclose(p[0], (2 * np.pi) ** (-1) / 2.25**2)


def test_independent_gaussian_discrete_only():
    dist = IndependentGaussian((CTPhaseUncertainty(),))
    assert dist.ndim == 0
    assert np.array_equal(dist.pdf(np.zeros((3, 0))), np.ones(3))


def test_multivariate_gaussian():
    dims = (SetupUncertainty(), RangeUncertainty())
    ind = IndependentGaussian(dims)
    mv = MultivariateGaussian(dims)  # diagonal default == independent
    pts = np.array([[0.5, -0.5, 1, 0.2, 0.01], [0, 0, 0, 0, 0]])
    assert np.allclose(mv.pdf(pts), ind.pdf(pts))
    cov = np.diag(ind.std() ** 2)
    cov[0, 1] = cov[1, 0] = 2.0
    corr = MultivariateGaussian(dims, cov=cov)
    assert not np.allclose(corr.pdf(pts), ind.pdf(pts))
    assert np.allclose(corr.std(), ind.std())
    assert corr.sample(5, np.random.default_rng(1)).shape == (5, 5)
    assert corr.to_dict()["distribution"] == "multivariate_gaussian"
    with pytest.raises(ValueError):
        MultivariateGaussian(dims, cov=np.eye(3))
    with pytest.raises(ValueError):
        MultivariateGaussian(dims, cov=cov, mean=[0, 0])


def test_create_distribution():
    dims = _dims()
    assert isinstance(create_distribution(None, dims), IndependentGaussian)
    assert isinstance(create_distribution("multivariate_gaussian", dims), MultivariateGaussian)
    d = create_distribution({"distribution": "multivariate_gaussian", "cov": np.eye(5)}, dims)
    assert np.array_equal(d.cov, np.eye(5))
    assert create_distribution(d, dims) is d
    with pytest.raises(ValueError):
        create_distribution(d, (SetupUncertainty(),))  # keys mismatch
    with pytest.raises(ValueError):
        create_distribution("copula", dims)
    with pytest.raises(ValueError):
        create_distribution({"cov": np.eye(5)}, dims)
    with pytest.raises(TypeError):
        create_distribution(5, dims)
    assert {"independent_gaussian", "multivariate_gaussian"} <= set(available_distributions())
    with pytest.raises(ValueError):
        IndependentGaussian(dims).pdf(np.zeros((2, 3)))


def test_dimension_registry_rejects_a_second_class_under_a_name():
    class Clash(SetupUncertainty):
        pass

    with pytest.raises(ValueError, match="already registered"):
        register_dimension(Clash)


def test_dimension_interfaces_by_kind():
    phase = CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5])
    with pytest.raises(TypeError, match="no mean"):
        phase.mean()
    with pytest.raises(TypeError, match="no mass function"):
        SetupUncertainty().pmf()
    assert RangeUncertainty().label([1.0, 0.035]) == "range +1.0 mm / +3.5 %"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"phases": [], "probabilities": []}, "At least one"),
        ({"phases": [-1], "probabilities": [1.0]}, "non-negative"),
    ],
)
def test_ct_phase_validation_errors(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        CTPhaseUncertainty(**kwargs)


def test_ct_phase_from_pairs_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(n, 2\)"):
        CTPhaseUncertainty.from_pairs([[0, 0.5, 0.5]])


def test_distribution_edge_cases():
    from pyRadPlan.scenarios import MultivariateGaussian

    dims = (SetupUncertainty(), RangeUncertainty())
    gaussian = IndependentGaussian(dims)
    assert gaussian.validate_points([0, 0, 0, 0, 0]).shape == (1, 5)
    assert repr(gaussian) == f"IndependentGaussian(keys={gaussian.keys})"

    discrete_only = IndependentGaussian(
        [CTPhaseUncertainty(phases=[0, 1], probabilities=[0.5, 0.5])]
    )
    assert discrete_only.ndim == 0
    assert discrete_only.mean().shape == (0,)
    assert discrete_only.std().shape == (0,)

    with pytest.raises(ValueError, match="symmetric"):
        MultivariateGaussian(dims, cov=np.triu(np.ones((5, 5))))
    shifted = MultivariateGaussian(dims, mean=np.arange(5.0))
    assert np.array_equal(shifted.mean(), np.arange(5.0))

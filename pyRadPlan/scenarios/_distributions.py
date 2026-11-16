"""
Joint probability distributions over the continuous components of uncertainty dimensions.

A distribution is built from a sequence of dimensions and operates on points laid out as
``(n, d)`` arrays whose columns are the continuous component keys in dimension order
(:attr:`JointDistribution.keys`). Discrete dimensions are handled by their own mass function in
:class:`~pyRadPlan.scenarios.ScenarioSet`, so a distribution never sees them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Optional, Sequence

import numpy as np

from ._dimensions import UncertaintyDimension

_DISTRIBUTION_REGISTRY: dict[str, type["JointDistribution"]] = {}


def register_distribution(cls: type["JointDistribution"]) -> type["JointDistribution"]:
    """Register a distribution class under its :attr:`name` (class decorator)."""
    _DISTRIBUTION_REGISTRY[cls.name] = cls
    return cls


def available_distributions() -> list[str]:
    """Names of the registered joint distributions."""
    return sorted(_DISTRIBUTION_REGISTRY)


def create_distribution(
    spec: Any, dimensions: Sequence[UncertaintyDimension]
) -> "JointDistribution":
    """
    Create a joint distribution for ``dimensions``.

    Parameters
    ----------
    spec : JointDistribution or str or dict or None
        Instance (returned as is, after checking it matches the dimensions' keys), registered
        name, or ``{"distribution": name, **parameters}``. ``None`` selects
        :class:`IndependentGaussian`.
    dimensions : sequence of UncertaintyDimension
        Dimensions whose continuous components the distribution covers.
    """
    if spec is None:
        spec = IndependentGaussian.name
    if isinstance(spec, JointDistribution):
        expected = tuple(k for d in dimensions if d.kind == "continuous" for k in d.keys)
        if spec.keys != expected:
            raise ValueError(
                f"Distribution covers components {spec.keys}, dimensions require {expected}"
            )
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("distribution", None)
        if name is None:
            raise ValueError("Distribution dictionaries need a 'distribution' key")
    else:
        raise TypeError(f"Cannot create a distribution from {type(spec).__name__}")
    try:
        cls = _DISTRIBUTION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown distribution {name!r}. Known: {available_distributions()}"
        ) from exc
    return cls(dimensions, **params)


class JointDistribution(ABC):
    """
    Joint distribution of the continuous components of a set of dimensions.

    Parameters
    ----------
    dimensions : sequence of UncertaintyDimension
        Discrete dimensions are ignored; the continuous ones define :attr:`keys` and the
        column order of all points.
    """

    name: ClassVar[str]

    def __init__(self, dimensions: Sequence[UncertaintyDimension]):
        self._dimensions = tuple(d for d in dimensions if d.kind == "continuous")
        self._keys = tuple(k for d in self._dimensions for k in d.keys)

    @property
    def dimensions(self) -> tuple[UncertaintyDimension, ...]:
        """The continuous dimensions covered."""
        return self._dimensions

    @property
    def keys(self) -> tuple[str, ...]:
        """Component keys in column order."""
        return self._keys

    @property
    def ndim(self) -> int:
        """Number of continuous components."""
        return len(self._keys)

    def validate_points(self, points: Any) -> np.ndarray:
        """Coerce ``points`` to an ``(n, ndim)`` float array."""
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1) if self.ndim != 1 else arr.reshape(-1, 1)
        if arr.ndim != 2 or arr.shape[1] != self.ndim:
            raise ValueError(f"Expected points of shape (n, {self.ndim}), got {arr.shape}")
        return arr

    def mean(self) -> np.ndarray:
        """Mean vector ``(ndim,)``."""
        if self.ndim == 0:
            return np.zeros(0)
        return np.concatenate([d.mean() for d in self._dimensions])

    @abstractmethod
    def std(self) -> np.ndarray:
        """Marginal standard deviation per component ``(ndim,)``."""

    @abstractmethod
    def pdf(self, points: Any) -> np.ndarray:
        """Probability density at each of the ``(n, ndim)`` points, returned as ``(n,)``."""

    @abstractmethod
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw ``n`` samples as an ``(n, ndim)`` array."""

    def to_dict(self) -> dict[str, Any]:
        """Serializable description (``{"distribution": name, ...parameters}``)."""
        return {"distribution": self.name}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(keys={self.keys})"


@register_distribution
class IndependentGaussian(JointDistribution):
    """
    Independent normal components with the dimensions' means and standard deviations.

    Reproduces matRad's scenario probability (diagonal covariance of setup SDs, absolute and
    relative range SD). Components with zero standard deviation are deterministic: the density
    factor is one on the mean and zero elsewhere.
    """

    name: ClassVar[str] = "independent_gaussian"

    def std(self) -> np.ndarray:
        if self.ndim == 0:
            return np.zeros(0)
        return np.concatenate([d.std() for d in self._dimensions])

    def pdf(self, points: Any) -> np.ndarray:
        x = self.validate_points(points)
        if self.ndim == 0:
            return np.ones(x.shape[0])
        mu, sigma = self.mean(), self.std()
        random = sigma > 0
        z = (x[:, random] - mu[random]) / sigma[random]
        dens = (2 * np.pi) ** (-random.sum() / 2) * np.exp(-0.5 * np.sum(z**2, axis=1))
        dens = dens / np.prod(sigma[random])
        fixed_ok = np.all(np.isclose(x[:, ~random], mu[~random]), axis=1)
        return np.where(fixed_ok, dens, 0.0)

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.normal(self.mean(), self.std(), size=(int(n), self.ndim))


@register_distribution
class MultivariateGaussian(JointDistribution):
    """
    Correlated normal components with a full covariance matrix.

    Parameters
    ----------
    dimensions : sequence of UncertaintyDimension
    cov : array_like, optional
        ``(ndim, ndim)`` covariance in the dimensions' component order. Defaults to the
        diagonal built from the dimensions' standard deviations.
    mean : array_like, optional
        ``(ndim,)`` mean; defaults to the dimensions' means.
    """

    name: ClassVar[str] = "multivariate_gaussian"

    def __init__(
        self,
        dimensions: Sequence[UncertaintyDimension],
        cov: Optional[Any] = None,
        mean: Optional[Any] = None,
    ):
        super().__init__(dimensions)
        if cov is None:
            cov = np.diag(np.concatenate([d.std() for d in self._dimensions]) ** 2)
        self._cov = np.asarray(cov, dtype=float)
        if self._cov.shape != (self.ndim, self.ndim):
            raise ValueError(f"cov must have shape ({self.ndim}, {self.ndim})")
        if not np.allclose(self._cov, self._cov.T):
            raise ValueError("cov must be symmetric")
        self._mean = super().mean() if mean is None else np.asarray(mean, dtype=float)
        if self._mean.shape != (self.ndim,):
            raise ValueError(f"mean must have shape ({self.ndim},)")

    @property
    def cov(self) -> np.ndarray:
        """Covariance matrix."""
        return self._cov

    def mean(self) -> np.ndarray:
        return self._mean

    def std(self) -> np.ndarray:
        return np.sqrt(np.diag(self._cov))

    def pdf(self, points: Any) -> np.ndarray:
        from scipy.stats import multivariate_normal  # noqa: PLC0415

        x = self.validate_points(points)
        if self.ndim == 0:
            return np.ones(x.shape[0])
        dist = multivariate_normal(mean=self._mean, cov=self._cov, allow_singular=True)
        return np.atleast_1d(dist.pdf(x))

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        rng = np.random.default_rng() if rng is None else rng
        return rng.multivariate_normal(self._mean, self._cov, size=int(n))

    def to_dict(self) -> dict[str, Any]:
        return {"distribution": self.name, "cov": self._cov.tolist(), "mean": self._mean.tolist()}

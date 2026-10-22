"""
Uncertainty dimensions.

A dimension describes one source of uncertainty: its named components, whether it is
continuous or discrete, the parameters of its marginal distribution and which parts of a
calculation it affects. Dimensions are the vocabulary a :class:`~pyRadPlan.scenarios.ScenarioSet`
is expressed in; consumers query a set by dimension name and never assume a fixed axis order.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, Literal, Optional

import numpy as np
from pydantic import Field, SerializationInfo, SerializerFunctionWrapHandler, model_serializer
from pydantic import model_validator
from typing_extensions import Self

from pyRadPlan.core import PyRadPlanBaseModel

#: Effect categories a dimension can declare in :attr:`UncertaintyDimension.affects`.
AFFECTS_GEOMETRY = "geometry"
AFFECTS_RANGE = "range"
AFFECTS_VOXEL_PARAMS = "voxel_params"
VALID_EFFECTS = frozenset({AFFECTS_GEOMETRY, AFFECTS_RANGE, AFFECTS_VOXEL_PARAMS})

_DIMENSION_REGISTRY: dict[str, type["UncertaintyDimension"]] = {}


def register_dimension(cls: type["UncertaintyDimension"]) -> type["UncertaintyDimension"]:
    """Register a concrete dimension class under its :attr:`name` (class decorator)."""
    if cls.name in _DIMENSION_REGISTRY and _DIMENSION_REGISTRY[cls.name] is not cls:
        raise ValueError(f"An uncertainty dimension named {cls.name!r} is already registered")
    _DIMENSION_REGISTRY[cls.name] = cls
    return cls


def available_dimensions() -> dict[str, type["UncertaintyDimension"]]:
    """Return the registered dimension classes by name."""
    return dict(_DIMENSION_REGISTRY)


def create_dimension(spec: Any) -> "UncertaintyDimension":
    """
    Create a dimension from an instance, a name or a ``{"dimension": name, ...}`` dict.

    Parameters
    ----------
    spec : UncertaintyDimension or str or dict
        Dimension instance (returned as is), registered name (default parameters) or a
        dictionary with a ``"dimension"`` key and the parameters of that dimension.
    """
    if isinstance(spec, UncertaintyDimension):
        return spec
    if isinstance(spec, str):
        name, params = spec, {}
    elif isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("dimension", None)
        if name is None:
            raise ValueError("Dimension dictionaries need a 'dimension' key naming the dimension")
    else:
        raise TypeError(f"Cannot create an uncertainty dimension from {type(spec).__name__}")
    try:
        cls = _DIMENSION_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown uncertainty dimension {name!r}. Known: {sorted(_DIMENSION_REGISTRY)}"
        ) from exc
    return cls.model_validate(params)


class UncertaintyDimension(PyRadPlanBaseModel, ABC):
    """
    One source of uncertainty with named components.

    Subclasses set the class variables and, depending on :attr:`kind`, implement either the
    continuous interface (:meth:`mean`, :meth:`std`) or the discrete one (:meth:`support`,
    :meth:`pmf`). Values of a realization are always a 1-D array with one entry per component,
    in the units given by :attr:`units`.

    Attributes
    ----------
    name : str
        Dimension name used as key in scenario sets (``"setup"``, ``"range"``, ``"ct_phase"``).
    components : tuple of str
        Component names; the full component keys are ``"<name>.<component>"``.
    kind : {"continuous", "discrete"}
        Whether realizations are sampled from a density or from a finite support.
    units : tuple of str
        Unit per component (``""`` for dimensionless).
    affects : frozenset of str
        Which parts of a calculation a realization changes: ``"geometry"`` (ray tracing /
        isocenter), ``"range"`` (radiological depth) and/or ``"voxel_params"`` (per-voxel
        parameters such as alpha/beta or tissue class). Consumers use this to group scenarios
        that share expensive inputs.
    """

    name: ClassVar[str]
    components: ClassVar[tuple[str, ...]]
    kind: ClassVar[Literal["continuous", "discrete"]]
    units: ClassVar[tuple[str, ...]]
    affects: ClassVar[frozenset[str]]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "affects" in cls.__dict__ and not cls.affects <= VALID_EFFECTS:
            raise TypeError(
                f"{cls.__name__}.affects contains unknown effects "
                f"{set(cls.affects) - VALID_EFFECTS}; valid are {sorted(VALID_EFFECTS)}"
            )

    @property
    def num_components(self) -> int:
        """Number of components."""
        return len(self.components)

    @property
    def keys(self) -> tuple[str, ...]:
        """Component keys ``"<name>.<component>"``."""
        return tuple(f"{self.name}.{c}" for c in self.components)

    # Continuous interface
    def mean(self) -> np.ndarray:
        """Return the mean of each component (continuous dimensions)."""
        raise TypeError(f"Dimension {self.name!r} is discrete and has no mean")

    def std(self) -> np.ndarray:
        """Return the standard deviation of each component (continuous dimensions)."""
        raise TypeError(f"Dimension {self.name!r} is discrete and has no standard deviation")

    # Discrete interface
    def support(self) -> np.ndarray:
        """Possible states as ``(num_states, num_components)`` array (discrete dimensions)."""
        raise TypeError(f"Dimension {self.name!r} is continuous and has no finite support")

    def pmf(self) -> np.ndarray:
        """Probability mass of each state in :meth:`support` (discrete dimensions)."""
        raise TypeError(f"Dimension {self.name!r} is continuous and has no mass function")

    def nominal(self) -> np.ndarray:
        """Nominal value: the mean (continuous) or the first state of the support (discrete)."""
        if self.kind == "continuous":
            return np.asarray(self.mean(), dtype=float)
        return np.asarray(self.support()[0], dtype=float)

    def validate_against_ct(self, ct: Any) -> None:
        """Raise ``ValueError`` if the dimension is inconsistent with ``ct`` (default: no-op)."""

    def coerce(self, value: Any) -> np.ndarray:
        """Coerce ``value`` to a 1-D float array with one entry per component."""
        arr = np.atleast_1d(np.asarray(value, dtype=float))
        if arr.shape != (self.num_components,):
            raise ValueError(
                f"Dimension {self.name!r} expects {self.num_components} component(s), "
                f"got shape {arr.shape}"
            )
        return arr

    @model_serializer(mode="wrap")
    def _serialize_with_name(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ):
        out = handler(self)
        if isinstance(out, dict):
            out = {"dimension": self.name, **out}
        return out


@register_dimension
class SetupUncertainty(UncertaintyDimension):
    """
    Patient setup (isocenter shift) uncertainty in the three spatial directions.

    Realizations are isocenter shifts in mm added to every beam's isocenter.
    """

    name: ClassVar[str] = "setup"
    components: ClassVar[tuple[str, ...]] = ("x", "y", "z")
    kind: ClassVar[Literal["continuous", "discrete"]] = "continuous"
    units: ClassVar[tuple[str, ...]] = ("mm", "mm", "mm")
    affects: ClassVar[frozenset[str]] = frozenset({AFFECTS_GEOMETRY})

    sd: tuple[float, float, float] = Field(
        default=(2.25, 2.25, 2.25),
        description="Standard deviation of the setup shift per direction in mm.",
    )
    shift_mean: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Systematic offset of the setup shift per direction in mm.",
    )

    @model_validator(mode="after")
    def _check_sd(self) -> Self:
        if any(s < 0 for s in self.sd):
            raise ValueError("Setup standard deviations must be non-negative")
        return self

    def mean(self) -> np.ndarray:
        return np.asarray(self.shift_mean, dtype=float)

    def std(self) -> np.ndarray:
        return np.asarray(self.sd, dtype=float)


@register_dimension
class RangeUncertainty(UncertaintyDimension):
    """
    Particle range uncertainty as an absolute (mm) and a relative (fraction) component.

    A realization ``(abs, rel)`` scales radiological depths as ``(1 + rel) * d + abs``. The
    relative standard deviation is configured in percent (matRad convention) but realizations
    carry the fraction.
    """

    name: ClassVar[str] = "range"
    components: ClassVar[tuple[str, ...]] = ("abs", "rel")
    kind: ClassVar[Literal["continuous", "discrete"]] = "continuous"
    units: ClassVar[tuple[str, ...]] = ("mm", "")
    affects: ClassVar[frozenset[str]] = frozenset({AFFECTS_RANGE})

    abs_sd: float = Field(default=1.0, ge=0, description="Absolute range SD in mm.")
    rel_sd: float = Field(default=3.5, ge=0, description="Relative range SD in percent.")
    combined: bool = Field(
        default=True,
        description="Whether gridded models move the absolute and relative component together "
        "(one grid axis) or independently (product grid). matRad: combineRange.",
    )

    def mean(self) -> np.ndarray:
        return np.zeros(2)

    def std(self) -> np.ndarray:
        return np.asarray([self.abs_sd, self.rel_sd / 100.0], dtype=float)


@register_dimension
class CTPhaseUncertainty(UncertaintyDimension):
    """
    Discrete anatomy uncertainty: which CT phase (scenario cube) of the CT is used.

    The realization is the 0-based phase index. Probabilities are used as given (matRad
    ``ctScenProb``); scenario weights normalize them.
    """

    name: ClassVar[str] = "ct_phase"
    components: ClassVar[tuple[str, ...]] = ("index",)
    kind: ClassVar[Literal["continuous", "discrete"]] = "discrete"
    units: ClassVar[tuple[str, ...]] = ("",)
    affects: ClassVar[frozenset[str]] = frozenset({AFFECTS_GEOMETRY, AFFECTS_VOXEL_PARAMS})

    phases: list[int] = Field(default=[0], description="0-based CT phase indices used.")
    probabilities: list[float] = Field(
        default=[1.0], description="Probability of each phase in ``phases``."
    )

    @model_validator(mode="after")
    def _check(self) -> Self:
        if len(self.phases) == 0:
            raise ValueError("At least one CT phase is required")
        if len(self.phases) != len(self.probabilities):
            raise ValueError("phases and probabilities must have the same length")
        if any(p < 0 for p in self.phases):
            raise ValueError("CT phase indices must be non-negative")
        if len(set(self.phases)) != len(self.phases):
            raise ValueError("CT phase indices must be unique")
        if any(not 0 <= p <= 1 for p in self.probabilities):
            raise ValueError("CT phase probabilities must be within [0, 1]")
        return self

    @classmethod
    def from_ct(cls, ct: Any, probabilities: Optional[list[float]] = None) -> Self:
        """All phases of ``ct`` with equal (or the given) probabilities."""
        num = int(getattr(ct, "num_of_ct_scen", 1))
        if probabilities is None:
            probabilities = [1.0 / num] * num
        return cls(phases=list(range(num)), probabilities=list(probabilities))

    @classmethod
    def from_pairs(cls, pairs: Any) -> Self:
        """Build from matRad-style ``[(phase, probability), ...]`` pairs (0-based)."""
        arr = np.atleast_2d(np.asarray(pairs, dtype=float))
        if arr.shape[1] != 2:
            raise ValueError("Expected (n, 2) pairs of (phase index, probability)")
        return cls(phases=[int(round(p)) for p in arr[:, 0]], probabilities=list(arr[:, 1]))

    def pairs(self) -> list[tuple[int, float]]:
        """``[(phase, probability), ...]`` pairs (0-based)."""
        return [(int(p), float(w)) for p, w in zip(self.phases, self.probabilities)]

    def support(self) -> np.ndarray:
        return np.asarray(self.phases, dtype=float).reshape(-1, 1)

    def pmf(self) -> np.ndarray:
        return np.asarray(self.probabilities, dtype=float)

    def validate_against_ct(self, ct: Any) -> None:
        num = int(getattr(ct, "num_of_ct_scen", 1))
        bad = [p for p in self.phases if p >= num]
        if bad:
            raise ValueError(
                f"CT phases {bad} requested, but the CT only has {num} phase(s) (0-based indices)"
            )

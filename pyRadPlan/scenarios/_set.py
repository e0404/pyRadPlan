"""
Scenario sets: concrete realizations of uncertainty dimensions.

A :class:`ScenarioSet` is an immutable list of scenarios, each a realization of every dimension
of the set with a probability density, an aggregation weight and a *flat index* that addresses
the matching slot of a :class:`~pyRadPlan.dij.Dij` container. The flat index is the only stable
handle. Dimensional access (``isel`` / ``sel`` / ``grid_view``) is a derived view provided by
:class:`ScenarioLayout`, which can be re-tied at will with :meth:`ScenarioSet.with_axes`.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

import numpy as np
from pydantic import Field, PrivateAttr, field_serializer, field_validator, model_validator
from pydantic import SerializationInfo, SerializeAsAny
from typing_extensions import Self

from pyRadPlan.core import PyRadPlanBaseModel

from ._dimensions import AFFECTS_VOXEL_PARAMS, UncertaintyDimension, create_dimension
from ._distributions import create_distribution

#: Axis tying matRad uses: CT phase, setup shift (x, y, z tied) and range (abs, rel tied).
MATRAD_AXES: tuple[tuple[str, ...], ...] = (("ct_phase",), ("setup",), ("range",))
#: The dimensions matRad's ``(ct, shift, range)`` cell layout can express.
MATRAD_DIMENSIONS: tuple[str, ...] = tuple(axis[0] for axis in MATRAD_AXES)

#: Tag of the nominal (unperturbed) scenario.
TAG_NOMINAL = "nominal"

AxisSpec = Union[str, Sequence[str]]


class ScenarioBasis(PyRadPlanBaseModel):
    """
    Linear combination defining a derived scenario.

    A scenario carrying a basis is not computed by a dose engine; its quantities are the sum of
    ``coefficient * quantity[index]`` over the referenced *computed* scenarios. A basis with a
    single unit coefficient shares the referenced scenario's influence matrices (e.g. a
    biological variant of the same geometry).
    """

    coefficients: dict[int, float] = Field(description="Flat scenario index -> coefficient.")

    @field_validator("coefficients", mode="before")
    @classmethod
    def _coerce(cls, v: Any) -> dict[int, float]:
        if not isinstance(v, dict):
            raise ValueError("coefficients must be a mapping of scenario index to coefficient")
        out = {int(k): float(c) for k, c in v.items()}
        if not out:
            raise ValueError("A scenario basis needs at least one coefficient")
        if any(k < 0 for k in out):
            raise ValueError("Scenario indices must be non-negative")
        return out

    @property
    def is_alias(self) -> bool:
        """Whether the basis is a single unit coefficient, i.e. shares one scenario's slot."""
        return len(self.coefficients) == 1 and next(iter(self.coefficients.values())) == 1.0

    def shifted(self, offset: int) -> "ScenarioBasis":
        """Basis with all referenced indices shifted by ``offset``."""
        return ScenarioBasis(coefficients={k + offset: c for k, c in self.coefficients.items()})


class Scenario(PyRadPlanBaseModel):
    """
    One realization of all dimensions of a scenario set.

    Attributes
    ----------
    index : int
        Flat index: slot of this scenario in the containers of a Dij built for the set.
    values : dict[str, np.ndarray]
        Dimension name -> 1-D array with one entry per component.
    probability : float
        Joint density / mass of the realization under the set's distribution.
    weight : float
        Weight for aggregating over scenarios (normalized by the generating model).
    tags : frozenset[str]
        Free-form markers, e.g. ``{"nominal"}``.
    basis : ScenarioBasis, optional
        Set for derived scenarios that are not computed but combined from others.
    """

    index: int = Field(ge=0)
    values: dict[str, Any]
    probability: float = Field(ge=0)
    weight: float = Field(ge=0)
    tags: frozenset[str] = frozenset()
    basis: Optional[ScenarioBasis] = None
    component_names: dict[str, tuple[str, ...]] = Field(
        default_factory=dict, description="Dimension name -> component names (for value())."
    )

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, v: Any) -> dict[str, np.ndarray]:
        if not isinstance(v, dict):
            raise ValueError("values must map dimension names to component arrays")
        return {str(k): np.atleast_1d(np.asarray(val, dtype=float)) for k, val in v.items()}

    @field_serializer("values")
    def _serialize_values(self, v: dict[str, np.ndarray], info: SerializationInfo):
        if info.mode == "json":
            return {k: np.asarray(val).tolist() for k, val in v.items()}
        return v

    @property
    def is_computed(self) -> bool:
        """Whether a dose engine computes this scenario (no basis)."""
        return self.basis is None

    @property
    def is_nominal(self) -> bool:
        """Whether the scenario carries the nominal tag."""
        return TAG_NOMINAL in self.tags

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Names of the dimensions realized."""
        return tuple(self.values)

    def __getitem__(self, dimension: str) -> np.ndarray:
        """Component values of ``dimension``."""
        return self.values[dimension]

    def value(self, key: str) -> float:
        """Value of a single component key such as ``"setup.x"``."""
        dim, _, comp = key.partition(".")
        if not comp:
            arr = self.values[dim]
            if arr.size != 1:
                raise KeyError(f"{key!r} has {arr.size} components; use '<dim>.<component>'")
            return float(arr[0])
        comps = self.component_names.get(dim)
        if comps is None or comp not in comps:
            raise KeyError(f"Unknown component key {key!r}")
        return float(self.values[dim][comps.index(comp)])

    @property
    def ct_phase(self) -> int:
        """CT phase index of the realization (0 when the set has no ct_phase dimension)."""
        arr = self.values.get("ct_phase")
        return 0 if arr is None else int(round(float(arr[0])))

    def __repr__(self) -> str:
        vals = ", ".join(f"{k}={np.round(v, 6).tolist()}" for k, v in self.values.items())
        extra = f", tags={sorted(self.tags)}" if self.tags else ""
        extra += ", derived" if self.basis is not None else ""
        return f"Scenario(index={self.index}, {vals}, weight={self.weight:.4g}{extra})"

    __str__ = __repr__


class ScenarioLayout(PyRadPlanBaseModel):
    """
    Grid view of a scenario set for a given axis tying.

    Each axis ties a tuple of component keys; its coordinates are the distinct value tuples of
    those keys in first-appearance order with the nominal tuple first. ``position`` holds the
    grid subscripts of every scenario (matRad ``linearMask``) and :attr:`mask` marks occupied
    grid cells (matRad ``scenMask``). Positions are not stable when a set is extended; use the
    flat scenario index to persist references.
    """

    axes: tuple[tuple[str, ...], ...]
    axis_names: tuple[str, ...]
    coords: tuple[Any, ...]
    position: Any
    num_scenarios: int

    @property
    def ndim(self) -> int:
        """Number of axes."""
        return len(self.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape (number of coordinates per axis)."""
        return tuple(int(c.shape[0]) for c in self.coords)

    @property
    def mask(self) -> np.ndarray:
        """Boolean array of :attr:`shape`, True where a scenario exists."""
        mask = np.zeros(self.shape, dtype=bool)
        if self.num_scenarios:
            mask[tuple(self.position.T)] = True
        return mask

    @property
    def index_grid(self) -> np.ndarray:
        """Int array of :attr:`shape` with the flat scenario index per cell, ``-1`` off-mask."""
        grid = -np.ones(self.shape, dtype=int)
        if self.num_scenarios:
            grid[tuple(self.position.T)] = np.arange(self.num_scenarios)
        return grid

    @property
    def is_dense(self) -> bool:
        """Whether every grid cell holds a scenario."""
        return bool(self.mask.all())

    def axis(self, name: str) -> int:
        """Axis number for an axis name, dimension name or component key."""
        if name in self.axis_names:
            return self.axis_names.index(name)
        candidates = [
            k
            for k, axis in enumerate(self.axes)
            if name in axis
            or name.replace("_", ".") in axis
            or all(key.split(".")[0] == name for key in axis)
        ]
        if len(candidates) != 1:
            raise KeyError(f"{name!r} does not identify one axis of {self.axis_names}")
        return candidates[0]

    def linear_index(self, positions: Optional[dict[str, int]] = None, **kwargs: int) -> int:
        """Flat scenario index at a full grid position (matRad ``sub2scenIx`` + ``scenNum``)."""
        pos = self._positions(positions, kwargs)
        if len(pos) != self.ndim:
            raise KeyError(f"All {self.ndim} axes need a position, got {sorted(pos)}")
        sub = tuple(pos[k] for k in range(self.ndim))
        idx = int(self.index_grid[sub])
        if idx < 0:
            raise KeyError(f"No scenario at grid position {sub}")
        return idx

    def indices_at(self, positions: Optional[dict[str, int]] = None, **kwargs: int) -> np.ndarray:
        """Flat indices of all scenarios matching the given (partial) grid position."""
        pos = self._positions(positions, kwargs)
        hit = np.ones(self.num_scenarios, dtype=bool)
        for k, p in pos.items():
            hit &= self.position[:, k] == p
        return np.flatnonzero(hit)

    def grid_view(self, container: Any) -> np.ndarray:
        """
        Arrange a flat per-scenario container on the grid.

        Returns an object array of :attr:`shape` holding the container entries (same objects,
        no copy) at their grid cells and ``None`` elsewhere.
        """
        items = list(container.flat) if isinstance(container, np.ndarray) else list(container)
        if len(items) != self.num_scenarios:
            raise ValueError(
                f"Container has {len(items)} entries, layout has {self.num_scenarios} scenarios"
            )
        view = np.empty(self.shape, dtype=object)
        for i, sub in enumerate(self.position):
            view[tuple(sub)] = items[i]
        return view

    def _positions(self, positions: Optional[dict[str, int]], kwargs: dict[str, int]):
        merged = dict(positions or {})
        merged.update(kwargs)
        out: dict[int, int] = {}
        for name, p in merged.items():
            k = self.axis(name)
            if not 0 <= int(p) < self.shape[k]:
                raise IndexError(f"Position {p} out of range for axis {self.axis_names[k]}")
            out[k] = int(p)
        return out


class ScenarioSet(PyRadPlanBaseModel):
    """
    Immutable collection of scenario realizations over a set of uncertainty dimensions.

    Build one from a :class:`~pyRadPlan.scenarios.ScenarioModel` (``model.scenarios()``) or from
    explicit points with :meth:`from_points`. Dose engines, quantities and the Dij consume the
    set; they address scenarios by flat index and group them by dimension values.

    Attributes
    ----------
    dimensions : tuple of UncertaintyDimension
        Dimensions realized, in canonical order.
    points : dict[str, np.ndarray]
        Dimension name -> ``(num_scenarios, num_components)`` realizations.
    probabilities, weights : np.ndarray
        Per-scenario probability density and aggregation weight.
    tags : tuple of frozenset
        Per-scenario tags.
    bases : tuple of ScenarioBasis or None
        Per-scenario basis for derived scenarios (``None`` = computed).
    axes : tuple of tuple of str
        Default axis tying for the :attr:`layout` view.
    distribution : str, optional
        Name of the distribution the probabilities stem from (informative).
    """

    dimensions: tuple[SerializeAsAny[UncertaintyDimension], ...]
    points: dict[str, Any]
    probabilities: Any
    weights: Any
    tags: tuple[frozenset[str], ...] = ()
    bases: tuple[Optional[ScenarioBasis], ...] = ()
    axes: tuple[tuple[str, ...], ...] = ()
    distribution: Optional[str] = None

    _scenario_cache: dict[int, Scenario] = PrivateAttr(default_factory=dict)
    _layout_cache: dict[tuple, ScenarioLayout] = PrivateAttr(default_factory=dict)
    _group_cache: dict[tuple, list] = PrivateAttr(default_factory=dict)

    # ------------------------------------------------------------------ validation
    @field_validator("dimensions", mode="before")
    @classmethod
    def _coerce_dimensions(cls, v: Any) -> tuple[UncertaintyDimension, ...]:
        if isinstance(v, (UncertaintyDimension, str, dict)):
            v = [v]
        return tuple(create_dimension(d) for d in v)

    @field_validator("points", mode="before")
    @classmethod
    def _coerce_points(cls, v: Any) -> dict[str, np.ndarray]:
        if not isinstance(v, dict):
            raise ValueError("points must map dimension names to (n, components) arrays")
        return {str(k): np.asarray(val, dtype=float) for k, val in v.items()}

    @field_validator("probabilities", "weights", mode="before")
    @classmethod
    def _coerce_vectors(cls, v: Any) -> np.ndarray:
        arr = np.atleast_1d(np.asarray(v, dtype=float))
        if arr.ndim != 1:
            raise ValueError("probabilities and weights must be 1-D")
        if np.any(arr < 0) or not np.all(np.isfinite(arr)):
            raise ValueError("probabilities and weights must be finite and non-negative")
        return arr

    @field_validator("tags", mode="before")
    @classmethod
    def _coerce_tags(cls, v: Any) -> tuple[frozenset[str], ...]:
        if v is None:
            return ()
        return tuple(frozenset(str(t) for t in (tags or ())) for tags in v)

    @field_validator("bases", mode="before")
    @classmethod
    def _coerce_bases(cls, v: Any) -> tuple[Optional[ScenarioBasis], ...]:
        if v is None:
            return ()
        out = []
        for b in v:
            if b is None or isinstance(b, ScenarioBasis):
                out.append(b)
            else:
                out.append(ScenarioBasis.model_validate(b))
        return tuple(out)

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        names = [d.name for d in self.dimensions]
        if len(set(names)) != len(names):
            raise ValueError(f"Dimension names must be unique, got {names}")
        if set(self.points) != set(names):
            raise ValueError(
                f"points must cover exactly the dimensions {names}, got {sorted(self.points)}"
            )
        n = None
        for dim in self.dimensions:
            arr = self.points[dim.name]
            if arr.ndim == 1:
                arr = arr.reshape(-1, dim.num_components)
            if arr.ndim != 2 or arr.shape[1] != dim.num_components:
                raise ValueError(
                    f"points[{dim.name!r}] must have shape (n, {dim.num_components}), "
                    f"got {arr.shape}"
                )
            arr = np.ascontiguousarray(arr)
            arr.setflags(write=False)
            self.points[dim.name] = arr
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise ValueError("All dimensions must have the same number of scenarios")
        n = 0 if n is None else n
        if self.probabilities.shape != (n,) or self.weights.shape != (n,):
            raise ValueError(f"probabilities and weights must have shape ({n},)")
        self.probabilities.setflags(write=False)
        self.weights.setflags(write=False)
        if not self.tags:
            object.__setattr__(self, "tags", tuple(frozenset() for _ in range(n)))
        elif len(self.tags) != n:
            raise ValueError(f"tags must have one entry per scenario ({n})")
        if not self.bases:
            object.__setattr__(self, "bases", tuple(None for _ in range(n)))
        elif len(self.bases) != n:
            raise ValueError(f"bases must have one entry per scenario ({n})")
        for i, basis in enumerate(self.bases):
            if basis is None:
                continue
            for ref in basis.coefficients:
                if ref >= n or ref == i:
                    raise ValueError(f"Basis of scenario {i} references invalid scenario {ref}")
                if self.bases[ref] is not None:
                    raise ValueError(
                        f"Basis of scenario {i} references derived scenario {ref}; bases may "
                        "only reference computed scenarios"
                    )
        object.__setattr__(self, "axes", self._normalize_axes(self.axes))
        return self

    @field_serializer("points")
    def _serialize_points(self, v: dict[str, np.ndarray], info: SerializationInfo):
        if info.mode == "json":
            return {k: np.asarray(val).tolist() for k, val in v.items()}
        return v

    @field_serializer("probabilities", "weights")
    def _serialize_vectors(self, v: np.ndarray, info: SerializationInfo):
        if info.mode == "json":
            return np.asarray(v).tolist()
        return v

    @field_serializer("tags")
    def _serialize_tags(self, v: tuple[frozenset[str], ...], info: SerializationInfo):
        if info.mode == "json":
            return [sorted(t) for t in v]
        return v

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_points(  # noqa: PLR0913
        cls,
        dimensions: Sequence[Any],
        points: dict[str, Any],
        *,
        distribution: Any = None,
        probabilities: Optional[Any] = None,
        weights: Optional[Any] = None,
        tags: Optional[Sequence[Any]] = None,
        bases: Optional[Sequence[Any]] = None,
        axes: Sequence[AxisSpec] = (),
    ) -> "ScenarioSet":
        """
        Build a set from explicit realizations.

        Parameters
        ----------
        dimensions : sequence of UncertaintyDimension (or names / dicts)
            Dimensions of the set. Dimensions missing from ``points`` are filled with their
            nominal value.
        points : dict[str, array_like]
            Dimension name -> ``(n, num_components)`` realizations (``(n,)`` accepted for
            one-component dimensions).
        distribution : JointDistribution or str or dict, optional
            Distribution to evaluate probabilities with (default independent Gaussian). Ignored
            when ``probabilities`` are given.
        probabilities : array_like, optional
            Explicit per-scenario densities.
        weights : array_like, optional
            Explicit aggregation weights; default: probabilities normalized to one.
        tags, bases : sequences, optional
            Per-scenario tags and bases.
        axes : sequence of axis specs, optional
            Default layout tying; unmentioned dimensions become one axis each.
        """
        dims = tuple(create_dimension(d) for d in dimensions)
        if not dims:
            raise ValueError("A scenario set needs at least one dimension")
        names = {d.name for d in dims}
        unknown = set(points) - names
        if unknown:
            raise ValueError(f"points contain unknown dimensions {sorted(unknown)}")
        n = None
        coerced: dict[str, np.ndarray] = {}
        for dim in dims:
            if dim.name in points:
                arr = np.asarray(points[dim.name], dtype=float)
                if arr.ndim == 1 and dim.num_components == 1:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.ndim != 2 or arr.shape[1] != dim.num_components:
                    raise ValueError(
                        f"points[{dim.name!r}] must have shape (n, {dim.num_components}), "
                        f"got {arr.shape}"
                    )
                coerced[dim.name] = arr
                n = arr.shape[0] if n is None else n
        if n is None:
            raise ValueError("points must contain at least one dimension")
        for dim in dims:
            if dim.name not in coerced:
                coerced[dim.name] = np.tile(dim.nominal(), (n, 1))

        if probabilities is None:
            dist = create_distribution(distribution, dims)
            continuous = [coerced[d.name] for d in dims if d.kind == "continuous"]
            prob = dist.pdf(np.hstack(continuous)) if continuous else np.ones(n)
            for dim in dims:
                if dim.kind == "discrete":
                    prob = prob * _discrete_mass(dim, coerced[dim.name])
            dist_name: Optional[str] = dist.name
        else:
            prob = np.asarray(probabilities, dtype=float)
            dist_name = None
        if weights is None:
            total = float(np.sum(prob))
            w = prob / total if total > 0 else np.full(n, 1.0 / n)
        else:
            w = np.asarray(weights, dtype=float)

        return cls(
            dimensions=dims,
            points=coerced,
            probabilities=prob,
            weights=w,
            tags=tags,
            bases=bases,
            axes=cls._normalize_axes_for(dims, axes),
            distribution=dist_name,
        )

    # ------------------------------------------------------------------ basic access
    def __len__(self) -> int:
        return int(self.probabilities.shape[0])

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __getitem__(self, index: int) -> Scenario:
        if isinstance(index, (np.integer, int)):
            idx = int(index)
            if idx < 0:
                idx += len(self)
            if not 0 <= idx < len(self):
                raise IndexError(f"Scenario index {index} out of range for {len(self)} scenarios")
            return self._scenario(idx)
        raise TypeError("ScenarioSet indices must be integers; use subset() for several")

    def _scenario(self, idx: int) -> Scenario:
        cached = self._scenario_cache.get(idx)
        if cached is None:
            cached = Scenario(
                index=idx,
                values={d.name: self.points[d.name][idx] for d in self.dimensions},
                probability=float(self.probabilities[idx]),
                weight=float(self.weights[idx]),
                tags=self.tags[idx],
                basis=self.bases[idx],
                component_names={d.name: d.components for d in self.dimensions},
            )
            self._scenario_cache[idx] = cached
        return cached

    @property
    def scenarios(self) -> tuple[Scenario, ...]:
        """All scenarios as objects."""
        return tuple(self[i] for i in range(len(self)))

    @property
    def num_scenarios(self) -> int:
        """Number of scenarios (computed and derived)."""
        return len(self)

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Names of the dimensions in canonical order."""
        return tuple(d.name for d in self.dimensions)

    @property
    def keys(self) -> tuple[str, ...]:
        """All component keys in canonical order."""
        return tuple(k for d in self.dimensions for k in d.keys)

    def dimension(self, name: str) -> UncertaintyDimension:
        """Return the dimension called ``name``."""
        for d in self.dimensions:
            if d.name == name:
                return d
        raise KeyError(f"Set has no dimension {name!r}; dimensions are {self.dimension_names}")

    def has_dimension(self, name: str) -> bool:
        """Whether the set realizes dimension ``name``."""
        return name in self.dimension_names

    def dimensions_affecting(self, what: str) -> tuple[str, ...]:
        """Names of the dimensions whose :attr:`affects` contains ``what``."""
        return tuple(d.name for d in self.dimensions if what in d.affects)

    def require_supported(self, supported: Sequence[str]) -> None:
        """Raise ``ValueError`` if the set realizes a dimension not in ``supported``."""
        missing = [n for n in self.dimension_names if n not in set(supported)]
        if missing:
            raise ValueError(
                f"Scenario set realizes uncertainty dimension(s) {missing} which are not "
                f"supported here (supported: {list(supported)})"
            )

    def values(self, dimension: str) -> np.ndarray:
        """``(num_scenarios, num_components)`` realizations of ``dimension``."""
        return self.points[self.dimension(dimension).name]

    def component_values(self, key: str) -> np.ndarray:
        """``(num_scenarios,)`` realizations of one component key."""
        dim, comp = self._split_key(key)
        return self.points[dim.name][:, dim.components.index(comp)]

    def ct_phase(self, index: int) -> int:
        """CT phase of scenario ``index`` (0 if the set has no ``ct_phase`` dimension)."""
        if not self.has_dimension("ct_phase"):
            return 0
        return int(round(float(self.points["ct_phase"][index, 0])))

    def ct_phases(self) -> np.ndarray:
        """CT phase per scenario."""
        if not self.has_dimension("ct_phase"):
            return np.zeros(len(self), dtype=int)
        return np.rint(self.points["ct_phase"][:, 0]).astype(int)

    def extent(self, dimension: str) -> np.ndarray:
        """Maximum absolute realization per component over the computed scenarios."""
        dim = self.dimension(dimension)
        if dim.kind != "continuous":
            raise TypeError(f"extent is only defined for continuous dimensions, not {dimension!r}")
        computed = self.computed_indices()
        if computed.size == 0:
            return np.zeros(dim.num_components)
        return np.max(np.abs(self.points[dim.name][computed]), axis=0)

    # ------------------------------------------------------------------ status queries
    def computed_indices(self) -> np.ndarray:
        """Flat indices of the computed (non-derived) scenarios."""
        return np.asarray([i for i, b in enumerate(self.bases) if b is None], dtype=int)

    def derived_indices(self) -> np.ndarray:
        """Flat indices of the derived scenarios."""
        return np.asarray([i for i, b in enumerate(self.bases) if b is not None], dtype=int)

    @property
    def num_computed(self) -> int:
        """Number of computed scenarios."""
        return int(self.computed_indices().size)

    def computed(self) -> "ScenarioSet":
        """Subset of the computed scenarios (reindexed)."""
        return self.subset(self.computed_indices())

    def has_nominal(self) -> bool:
        """Whether :meth:`nominal_index` succeeds."""
        try:
            self.nominal_index()
        except ValueError:
            return False
        return True

    def nominal_index(self) -> int:
        """
        Flat index of the nominal scenario.

        The first scenario tagged ``"nominal"``; otherwise the first computed scenario whose
        values equal every dimension's nominal value. Raises ``ValueError`` if there is none.
        """
        for i, tags in enumerate(self.tags):
            if TAG_NOMINAL in tags:
                return i
        hit = np.ones(len(self), dtype=bool)
        for d in self.dimensions:
            hit &= np.all(np.isclose(self.points[d.name], d.nominal()), axis=1)
        for i in np.flatnonzero(hit):
            if self.bases[i] is None:
                return int(i)
        raise ValueError("Scenario set has no nominal scenario")

    def alias_of(self, index: int) -> Optional[int]:
        """Index of the scenario whose matrices scenario ``index`` shares, if it is an alias."""
        basis = self.bases[index]
        if basis is None or not basis.is_alias:
            return None
        return next(iter(basis.coefficients))

    def tagged(self, tag: str) -> np.ndarray:
        """Flat indices of the scenarios carrying ``tag``."""
        return np.asarray([i for i, t in enumerate(self.tags) if tag in t], dtype=int)

    # ------------------------------------------------------------------ grouping
    def group_by(self, *keys: str) -> list[tuple[dict[str, Any], list[int]]]:
        """
        Group scenarios by shared values of dimensions and/or component keys.

        Parameters
        ----------
        *keys : str
            Dimension names (all components) or component keys such as ``"setup.x"``. With no
            keys, all scenarios form one group.

        Returns
        -------
        list of (values, indices)
            Groups in first-appearance order; ``values`` maps each requested key to its shared
            value (array for a dimension, float for a component) and ``indices`` lists the flat
            scenario indices of the group.
        """
        cache_key = tuple(keys)
        if cache_key in self._group_cache:
            return [(dict(v), list(ix)) for v, ix in self._group_cache[cache_key]]
        columns = []
        for key in keys:
            if "." in key:
                columns.append(self.component_values(key).reshape(-1, 1))
            else:
                columns.append(self.values(key))
        table = np.hstack(columns) if columns else np.zeros((len(self), 0))
        groups: dict[tuple, list[int]] = {}
        for i, row in enumerate(table):
            groups.setdefault(tuple(row.tolist()), []).append(i)
        out = []
        for row, indices in groups.items():
            vals: dict[str, Any] = {}
            pos = 0
            for key in keys:
                if "." in key:
                    vals[key] = float(row[pos])
                    pos += 1
                else:
                    c = self.dimension(key).num_components
                    vals[key] = np.asarray(row[pos : pos + c])
                    pos += c
            out.append((vals, indices))
        self._group_cache[cache_key] = out
        return [(dict(v), list(ix)) for v, ix in out]

    def voxel_param_sets(self) -> list[tuple[dict[str, Any], list[int]]]:
        """
        Group scenarios sharing their per-voxel parameters.

        One group per unique combination of the dimensions affecting ``"voxel_params"``; the
        group position is the column of the Dij's per-voxel parameter arrays (``alphax``,
        ``betax``, ``tissue_class_ix``) those scenarios use.
        """
        return self.group_by(*self.dimensions_affecting(AFFECTS_VOXEL_PARAMS))

    @property
    def num_voxel_param_sets(self) -> int:
        """Number of per-voxel parameter columns the set requires."""
        return len(self.voxel_param_sets())

    def voxel_param_set(self, index: int) -> int:
        """Per-voxel parameter column of scenario ``index``."""
        return int(self.voxel_param_set_indices()[index])

    def voxel_param_set_indices(self) -> np.ndarray:
        """Per-voxel parameter column of every scenario."""
        cols = np.zeros(len(self), dtype=int)
        for k, (_, indices) in enumerate(self.voxel_param_sets()):
            cols[indices] = k
        return cols

    # ------------------------------------------------------------------ set algebra
    def subset(self, indices: Any) -> "ScenarioSet":
        """
        Return a new set with the selected scenarios, reindexed in the given order.

        Raises ``ValueError`` if a derived scenario is kept while a scenario its basis
        references is dropped.
        """
        idx = np.atleast_1d(np.asarray(indices, dtype=int))
        if idx.size and (idx.min() < 0 or idx.max() >= len(self)):
            raise IndexError("subset indices out of range")
        remap = {int(old): new for new, old in enumerate(idx)}
        bases = []
        for old in idx:
            basis = self.bases[old]
            if basis is not None:
                try:
                    basis = ScenarioBasis(
                        coefficients={remap[k]: c for k, c in basis.coefficients.items()}
                    )
                except KeyError as exc:
                    raise ValueError(
                        f"Derived scenario {old} references scenario {exc.args[0]} which is "
                        "not part of the subset"
                    ) from exc
            bases.append(basis)
        return ScenarioSet(
            dimensions=self.dimensions,
            points={k: v[idx] for k, v in self.points.items()},
            probabilities=self.probabilities[idx],
            weights=self.weights[idx],
            tags=tuple(self.tags[i] for i in idx),
            bases=tuple(bases),
            axes=self.axes,
            distribution=self.distribution,
        )

    def extend(self, other: "ScenarioSet") -> "ScenarioSet":
        """
        Append the scenarios of ``other`` (same dimensions) with indices shifted by ``len(self)``.

        Weights are concatenated as they are; renormalize explicitly if needed.
        """
        if other.dimensions != self.dimensions:
            raise ValueError("Can only extend with a set over identical dimensions")
        n = len(self)
        return ScenarioSet(
            dimensions=self.dimensions,
            points={k: np.vstack([v, other.points[k]]) for k, v in self.points.items()},
            probabilities=np.concatenate([self.probabilities, other.probabilities]),
            weights=np.concatenate([self.weights, other.weights]),
            tags=self.tags + other.tags,
            bases=self.bases + tuple(b.shifted(n) if b is not None else None for b in other.bases),
            axes=self.axes,
            distribution=self.distribution if self.distribution == other.distribution else None,
        )

    def with_weights(self, weights: Any) -> "ScenarioSet":
        """Copy with new aggregation weights."""
        return self.model_copy(update={"weights": self._coerce_vectors(weights)})

    def normalized(self) -> "ScenarioSet":
        """Copy whose weights sum to one."""
        total = float(np.sum(self.weights))
        if total <= 0:
            raise ValueError("Cannot normalize weights that sum to zero")
        return self.with_weights(self.weights / total)

    # ------------------------------------------------------------------ layout view
    @property
    def layout(self) -> ScenarioLayout:
        """Grid view for the default :attr:`axes`."""
        return self._layout_for(self.axes)

    def with_axes(self, *axes: AxisSpec) -> "ScenarioSet":
        """Copy whose default layout ties the given axes (see :meth:`from_points`)."""
        new = self.model_copy(update={"axes": self._normalize_axes(axes)})
        return new

    def layout_for(self, *axes: AxisSpec) -> ScenarioLayout:
        """Grid view for an explicit axis tying without changing the set."""
        return self._layout_for(self._normalize_axes(axes))

    def isel(
        self, positions: Optional[dict[str, int]] = None, **kwargs: int
    ) -> tuple[Scenario, ...]:
        """Scenarios at the given grid positions of the default layout (partial selection ok)."""
        return tuple(self[i] for i in self.layout.indices_at(positions, **kwargs))

    def sel(
        self, values: Optional[dict[str, Any]] = None, *, atol: float = 1e-9, **kwargs: Any
    ) -> tuple[Scenario, ...]:
        """
        Scenarios whose dimension / component values match the given ones.

        Keys are dimension names (value per component) or component keys (``"setup.x"`` as a
        dict key or ``setup_x`` as a keyword). Matching is ``isclose`` with ``atol``.
        """
        merged = dict(values or {})
        merged.update(kwargs)
        hit = np.ones(len(self), dtype=bool)
        for key, val in merged.items():
            key_dot = key if "." in key or key in self.dimension_names else key.replace("_", ".")
            if key_dot in self.dimension_names:
                dim = self.dimension(key_dot)
                hit &= np.all(
                    np.isclose(self.points[dim.name], dim.coerce(val), atol=atol), axis=1
                )
            else:
                hit &= np.isclose(self.component_values(key_dot), float(val), atol=atol)
        return tuple(self[i] for i in np.flatnonzero(hit))

    def linear_index(self, positions: Optional[dict[str, int]] = None, **kwargs: int) -> int:
        """Flat index at a full grid position of the default layout."""
        return self.layout.linear_index(positions, **kwargs)

    def grid_view(self, container: Any) -> np.ndarray:
        """Arrange a flat per-scenario container on the default layout grid."""
        return self.layout.grid_view(container)

    def require_matrad_compatible(self, *, allow_alias: bool = True) -> None:
        """
        Raise ``ValueError`` if the set cannot be written as matRad ``(ct, shift, range)`` cells.

        Only the ``ct_phase``, ``setup`` and ``range`` dimensions are expressible, and derived
        scenarios have no matrices of their own (aliases share one and are allowed unless
        ``allow_alias`` is False).
        """
        unsupported = [d for d in self.dimension_names if d not in MATRAD_DIMENSIONS]
        if unsupported:
            raise ValueError(
                f"Scenario dimension(s) {unsupported} cannot be expressed in matRad's "
                f"(ct_phase, setup, range) cell layout"
            )
        derived = [
            i
            for i, b in enumerate(self.bases)
            if b is not None and not (allow_alias and b.is_alias)
        ]
        if derived:
            raise ValueError(
                f"Derived scenarios {derived} have no influence matrices and cannot be "
                "exported to matRad"
            )

    def matrad_layout(self) -> tuple[tuple[int, int, int], np.ndarray]:
        """
        Return the matRad cell shape ``(ct, shift, range)`` and per-scenario subscripts.

        Dimensions the set lacks have extent one and subscript zero. Raises ``ValueError``
        for dimensions matRad cannot express.
        """
        self.require_matrad_compatible(allow_alias=True)
        present = [d for d in MATRAD_DIMENSIONS if self.has_dimension(d)]
        layout = self.layout_for(*((d,) for d in present))
        shape = [1, 1, 1]
        position = np.zeros((len(self), 3), dtype=int)
        for k, dim in enumerate(present):
            column = MATRAD_DIMENSIONS.index(dim)
            shape[column] = layout.shape[k]
            position[:, column] = layout.position[:, k]
        return (shape[0], shape[1], shape[2]), position

    def to_xarray(self):
        """Flat scenario indices as an ``xarray.DataArray`` over the default layout (optional)."""
        try:
            import xarray as xr  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("to_xarray requires the optional xarray package") from exc
        layout = self.layout
        coords = {
            name: [tuple(np.round(c, 12).tolist()) for c in coord]
            for name, coord in zip(layout.axis_names, layout.coords)
        }
        return xr.DataArray(layout.index_grid, dims=layout.axis_names, coords=coords)

    # ------------------------------------------------------------------ presentation
    def to_table(self) -> list[list[Any]]:
        """Rows ``[index, *component values..., probability, weight, tags]``."""
        rows = []
        for s in self:
            vals = [float(v) for d in self.dimensions for v in s.values[d.name]]
            rows.append([s.index, *vals, s.probability, s.weight, ",".join(sorted(s.tags))])
        return rows

    def __str__(self) -> str:
        header = ["#", *self.keys, "prob", "weight", "tags"]
        lines = ["\t".join(header)]
        for row in self.to_table():
            cells = [str(row[0])] + [f"{v:.4g}" for v in row[1:-1]] + [row[-1]]
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ScenarioSet(num_scenarios={len(self)}, dimensions={self.dimension_names}, "
            f"axes={self.axes})"
        )

    # ------------------------------------------------------------------ helpers
    def _split_key(self, key: str) -> tuple[UncertaintyDimension, str]:
        name, _, comp = key.partition(".")
        dim = self.dimension(name)
        if not comp:
            if dim.num_components != 1:
                raise KeyError(f"{key!r} is a dimension with several components")
            comp = dim.components[0]
        if comp not in dim.components:
            raise KeyError(f"Dimension {name!r} has no component {comp!r}")
        return dim, comp

    def _normalize_axes(self, axes: Sequence[AxisSpec]) -> tuple[tuple[str, ...], ...]:
        return self._normalize_axes_for(self.dimensions, axes)

    @staticmethod
    def _normalize_axes_for(
        dimensions: Sequence[UncertaintyDimension], axes: Sequence[AxisSpec]
    ) -> tuple[tuple[str, ...], ...]:
        by_name = {d.name: d for d in dimensions}
        all_keys = [k for d in dimensions for k in d.keys]
        out: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for spec in axes:
            entries = [spec] if isinstance(spec, str) else list(spec)
            keys: list[str] = []
            for entry in entries:
                if entry in by_name:
                    keys.extend(by_name[entry].keys)
                elif entry in all_keys:
                    keys.append(entry)
                else:
                    raise ValueError(
                        f"Axis entry {entry!r} is neither a dimension {sorted(by_name)} nor a "
                        f"component key {all_keys}"
                    )
            dup = seen.intersection(keys)
            if dup or len(set(keys)) != len(keys):
                raise ValueError(f"Component keys {sorted(dup) or keys} tied to several axes")
            seen.update(keys)
            out.append(tuple(keys))
        for d in dimensions:
            remaining = tuple(k for k in d.keys if k not in seen)
            if remaining:
                out.append(remaining)
                seen.update(remaining)
        return tuple(out)

    def _layout_for(self, axes: tuple[tuple[str, ...], ...]) -> ScenarioLayout:
        if axes in self._layout_cache:
            return self._layout_cache[axes]
        n = len(self)
        coords, position, names = [], np.zeros((n, len(axes)), dtype=int), []
        nominal_by_key = {k: v for d in self.dimensions for k, v in zip(d.keys, d.nominal())}
        for k, axis in enumerate(axes):
            table = np.column_stack([self.component_values(key) for key in axis])
            nominal = np.asarray([nominal_by_key[key] for key in axis])
            order: dict[tuple, int] = {}
            rows = []
            has_nominal = n > 0 and bool(np.any(np.all(np.isclose(table, nominal), axis=1)))
            if has_nominal:
                order[tuple(nominal.tolist())] = 0
                rows.append(nominal)
            for row in table:
                key = tuple(row.tolist())
                if has_nominal and np.all(np.isclose(row, nominal)):
                    key = tuple(nominal.tolist())
                if key not in order:
                    order[key] = len(rows)
                    rows.append(row)
            lookup = {tuple(nominal.tolist()): 0} if has_nominal else {}
            lookup.update(order)
            for i, row in enumerate(table):
                key = tuple(row.tolist())
                if has_nominal and np.all(np.isclose(row, nominal)):
                    key = tuple(nominal.tolist())
                position[i, k] = lookup[key]
            coords.append(np.asarray(rows, dtype=float).reshape(len(rows), len(axis)))
            names.append(self._axis_name(axis))
        layout = ScenarioLayout(
            axes=axes,
            axis_names=tuple(names),
            coords=tuple(coords),
            position=position,
            num_scenarios=n,
        )
        self._layout_cache[axes] = layout
        return layout

    def _axis_name(self, axis: tuple[str, ...]) -> str:
        dims = {key.split(".")[0] for key in axis}
        if len(dims) == 1:
            dim = self.dimension(next(iter(dims)))
            if tuple(axis) == dim.keys:
                return dim.name
            if len(axis) == 1:
                return axis[0].replace(".", "_")
        return "+".join(axis)


def _discrete_mass(dim: UncertaintyDimension, values: np.ndarray) -> np.ndarray:
    """Probability mass of each row of ``values`` under the discrete ``dim``."""
    support, pmf = dim.support(), dim.pmf()
    out = np.zeros(values.shape[0])
    for i, row in enumerate(values):
        hit = np.flatnonzero(np.all(np.isclose(support, row), axis=1))
        if hit.size == 0:
            raise ValueError(f"Value {row} is not in the support of dimension {dim.name!r}")
        out[i] = pmf[hit[0]]
    return out

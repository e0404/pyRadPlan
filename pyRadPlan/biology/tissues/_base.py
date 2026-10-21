import logging
import math
from numbers import Integral

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing import Any, ClassVar, Iterable, Literal, Mapping, Optional, Sequence, List, Tuple

from ._reference import (
    DEFAULT_REFERENCE_RADIATION,
    get_reference_radiation,
    saturation_corrected_specific_energy,
    saturation_specific_energy,
)

# Versions of the Local Effect Model, in the Roman numerals they are published under.
LEM_VERSIONS: tuple[str, ...] = ("I", "II", "III", "IV")

logger = logging.getLogger(__name__)

#: Generic reference photon LQ parameters a tissue gets when nothing else defines them.
#: They are those the bundled generic base data is tabulated for (the "LEMax01" classes
#: of the carbon and oxygen machines), see :func:`default_reference_parameters`.
DEFAULT_ALPHA_X: float = 0.1
DEFAULT_BETA_X: float = 0.05

#: LQ parameters of the reference radiation (photons) the RBE is taken relative to, carried
#: by every tissue. For LEM they are model parameters as well; for MKM they are derived from
#: the model parameters, see :attr:`TissueProperties.reference_fields`.
REFERENCE_FIELDS: tuple[str, ...] = ("alpha_x", "beta_x")

#: Where a reference parameter came from, the values of
#: :attr:`TissueProperties.reference_sources`, from the most to the least specific:
#:
#: - ``"explicit"``: given by the user (or an input file) as they are.
#: - ``"published"``: given as published, e.g. the measured X-ray LQ parameters a cell line
#:   preset gives along with its model parameters (the tissue's `cell_line_source` names the
#:   preset), see :meth:`TissueProperties.from_preset`.
#: - ``"estimated"``: derived from the model's own parameters, e.g. by the MKM conversion
#:   :meth:`MKMTissueProperties.estimate_reference_parameters`.
#: - ``"base_data"``: taken over from a tissue class of base data, see
#:   :func:`~pyRadPlan.biology.tissues.apply_tissue_class`.
#: - ``"default"``: the generic :func:`default_reference_parameters`.
#:
#: Explicit and published values are kept as they are; a model deriving its reference
#: parameters (MKM) derives the others again from its model parameters.
REFERENCE_SOURCES: tuple[str, ...] = ("explicit", "published", "estimated", "base_data", "default")

#: Reference sources whose values a model deriving its reference parameters keeps.
FIXED_REFERENCE_SOURCES: tuple[str, ...] = ("explicit", "published")

# Validation context key building a tissue class of base data, see from_base_data.
_BASE_DATA_CONTEXT = "tissue_class_of_base_data"


def default_reference_parameters() -> dict[str, float]:
    """
    Return the reference photon LQ parameters a tissue gets when nothing defines them.

    The single place these defaults are defined: ``alpha_x = 0.1`` / ``beta_x = 0.05``,
    the values of the generic classes of the bundled base data (:data:`DEFAULT_ALPHA_X`,
    :data:`DEFAULT_BETA_X`).

    Returns
    -------
    dict[str, float]
        ``{"alpha_x": ..., "beta_x": ...}``.
    """
    return {"alpha_x": DEFAULT_ALPHA_X, "beta_x": DEFAULT_BETA_X}


def weakest_reference_source(sources: Iterable[str]) -> str:
    """
    Least specific of several reference sources, in the order of :data:`REFERENCE_SOURCES`.

    Summarizes a pair of reference parameters that came from different places, e.g. an
    explicit `alpha_x` with a default `beta_x` is ``"default"``.
    """
    known = [REFERENCE_SOURCES.index(s) for s in sources if s in REFERENCE_SOURCES]
    return REFERENCE_SOURCES[max(known)] if known else "explicit"


def _check_reference_source(value: Any) -> str:
    if value not in REFERENCE_SOURCES:
        raise ValueError(f"reference_source must be one of {REFERENCE_SOURCES}, not {value!r}")
    return value


def _reference_source_mapping(value: Any) -> dict[str, str]:
    """
    Per-parameter reference sources given in any supported form, validated.

    A mapping by parameter name (a record array read from a ``.mat`` file included), or a
    single source applying to both parameters (the form files written before the sources
    were tracked per parameter use); None gives none.
    """
    if value is None:
        return {}
    if isinstance(value, str):
        return dict.fromkeys(REFERENCE_FIELDS, _check_reference_source(value))
    names = getattr(getattr(value, "dtype", None), "names", None)
    if names:
        value = {name: value[name] for name in names}
    if not isinstance(value, Mapping):
        raise ValueError(f"reference_sources must be a mapping, not {value!r}")
    unknown = [name for name in value if name not in REFERENCE_FIELDS]
    if unknown:
        raise ValueError(f"reference_sources names unknown parameters {unknown}")
    return {name: _check_reference_source(str(source)) for name, source in value.items()}


def _unset_reference(value: Any) -> bool:
    """Whether a reference parameter value leaves the parameter unset (None or NaN)."""
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _parse_lem_version(name: str) -> Optional[str]:
    """
    Return the LEM version encoded in a name, None if it encodes none.

    Accepts the Roman numerals the versions are published under as well as plain
    numbers, each with or without a leading model name, e.g. "LEM_II", "LEM 2", "III"
    and "4" all denote the same version.
    """
    token = name.strip().upper()
    if token.startswith("LEM"):
        token = token[3:].lstrip(" _-")
    if token in LEM_VERSIONS:
        return token
    if token.isdigit() and 1 <= int(token) <= len(LEM_VERSIONS):
        return LEM_VERSIONS[int(token) - 1]
    return None


def _undeclared(value: Any) -> bool:
    """Whether a parameter value is NaN, i.e. not declared."""
    return isinstance(value, float) and math.isnan(value)


class TissueProperties(BaseModel):
    """
    Radiobiological properties of a tissue (linear-quadratic model).

    A tissue carries two kinds of parameters. *Model parameters*
    (:attr:`model_parameter_fields`) parameterize one RBE model and are specific to its
    provenance, e.g. `d_t` / `r_nucleus` of LEM or `r_nucleus`, `r_domain`, `alpha_0` /
    `beta_0` of MKM. *Reference parameters* (:attr:`reference_fields`: `alpha_x` /
    `beta_x`) are the LQ parameters of the reference radiation the RBE is taken relative to.
    This base class only carries the reference parameters and doubles as the reference-only
    tissue type. Subclasses add the parameters of one model: `LEMTissueProperties` (Local
    Effect Model), which takes the reference parameters as model inputs as well, and
    `MKMTissueProperties` (Microdosimetric Kinetic Model), which derives them from its model
    parameters (:attr:`derives_reference`).

    The reference parameters `alpha_x` / `beta_x` are never unset. One not given, or
    given as NaN or None (e.g. by an old file), is resolved in this order: the model's own
    derivation from its parameters (MKM, see
    :meth:`MKMTissueProperties.estimate_reference_parameters`), else the generic
    :func:`default_reference_parameters`. Values a tissue class of base data declares are
    taken over by :func:`~pyRadPlan.biology.tissues.apply_tissue_class`.
    `reference_sources` records where each value came from (:data:`REFERENCE_SOURCES`),
    `reference_source` summarizes the two. A model deriving its reference parameters keeps
    each derived from its current model parameters unless it is explicit or published, so
    overriding `alpha_x` alone keeps deriving `beta_x`. Tissue classes of base data are the
    exception: built by :meth:`from_base_data`, they keep a NaN reference parameter as "not
    declared", which base-data matching does not compare.

    `cell_line` names the cell line the parameters describe and `cell_line_source` where
    their values come from (set by :meth:`from_preset`, see
    :data:`~pyRadPlan.biology.tissues.CELL_LINE_PRESETS`).

    Equality is NaN-aware: two tissues of the same class are equal if all their fields
    are, with unset (NaN) parameters counting as equal; the :attr:`metadata_fields`
    (`reference_sources`, `cell_line_source`) are not compared, nor matched. Equality is
    strict about the provenance (`model`, the LEM `version`); matching a tissue to a tissue
    class of base data (:meth:`matches`) is not, since the provenance labels the parameters
    rather than being one of them. Instances are mutable and hence unhashable; assignments
    are validated, so assigning `alpha_x` or `beta_x` makes the source of that parameter
    ``"explicit"``. JSON dumps write NaN as the ``NaN`` constant, so they validate back.
    """

    model_config = ConfigDict(
        populate_by_name=True, ser_json_inf_nan="constants", validate_assignment=True
    )

    cell_line: Optional[str] = Field(None, description="Name of the cell line", alias="cell_Line")
    cell_line_source: Optional[str] = Field(
        None,
        description=(
            "Where the parameter values of the cell line come from (publication, toolkit, "
            "dataset, ...), e.g. as recorded by a cell line preset; 'Unverified' or "
            "'Unpublished' if unconfirmed"
        ),
        alias="cellLineSource",
        repr=False,
    )
    model: str = Field("Generic", description="Radiobiological model used for this tissue")
    alpha_x: float = Field(
        DEFAULT_ALPHA_X,
        description="alpha parameter of the reference radiation (linear-quadratic model)",
        alias="alphaX",
    )
    beta_x: float = Field(
        DEFAULT_BETA_X,
        description="beta parameter of the reference radiation (linear-quadratic model)",
        alias="betaX",
    )
    reference_sources: dict[str, str] = Field(
        default_factory=lambda: dict.fromkeys(REFERENCE_FIELDS, "explicit"),
        description=(
            "Where each of alpha_x / beta_x came from: 'explicit', 'published', 'estimated', "
            "'base_data' or 'default' (see REFERENCE_SOURCES)"
        ),
        alias="referenceSources",
        repr=False,
    )

    #: Fields describing a tissue without being part of its parameters, ignored by `==`.
    metadata_fields: ClassVar[tuple[str, ...]] = ("reference_sources", "cell_line_source")

    #: Parameters of the RBE model these properties belong to, as its provenance defines
    #: them; empty for the reference-only tissue.
    model_parameter_fields: ClassVar[tuple[str, ...]] = ()

    #: LQ parameters of the reference radiation the RBE is taken relative to.
    reference_fields: ClassVar[tuple[str, ...]] = REFERENCE_FIELDS

    #: Whether the model derives the reference parameters from its model parameters (MKM)
    #: rather than taking them as given.
    derives_reference: ClassVar[bool] = False

    _MODEL_ALIASES: ClassVar[dict[str, str]] = {
        "LEM": "LEM_I",
    }

    #: Fields identifying a tissue class of this model, compared by `matches` when no
    #: `fields` are given. These are the parameters the model's base data is tabulated
    #: for; the remaining ones describe the tissue itself (e.g. `alpha_0` / `beta_0`,
    #: which MKM takes per voxel). Subclasses set the parameters of their own model.
    identifying_fields: ClassVar[tuple[str, ...]] = (
        "alpha_x",
        "beta_x",
    )

    #: Fields labelling the provenance of the parameters (the model, e.g. MKM vs mMKM, and
    #: the LEM version). A soft label: `matches` does not compare them unless asked to, so
    #: a tissue labelled with another provenance still matches a class of base data by its
    #: parameters; see :attr:`provenance`.
    provenance_fields: ClassVar[tuple[str, ...]] = ("model",)

    @field_validator("reference_sources", mode="before")
    @classmethod
    def _known_reference_sources(cls, value: Any) -> Any:
        return {**dict.fromkeys(REFERENCE_FIELDS, "explicit"), **_reference_source_mapping(value)}

    @property
    def reference_source(self) -> str:
        """
        Summary of :attr:`reference_sources`: the least specific source of the two values.

        E.g. ``"default"`` for an explicit `alpha_x` next to a default `beta_x`, see
        :func:`weakest_reference_source`. Assigning a source applies it to both values (a
        derived source derives them again), as reading a tissue written before the sources
        were tracked per parameter does.
        """
        return weakest_reference_source(self.reference_sources.values())

    @reference_source.setter
    def reference_source(self, value: str) -> None:
        data = {**self.__dict__, "reference_sources": _reference_source_mapping(value)}
        resolved = self._resolved_reference(data, assigned=None, base_data=False)
        for field in (*REFERENCE_FIELDS, "reference_sources"):
            self.__dict__[field] = resolved[field]

    @model_validator(mode="before")
    @classmethod
    def _resolve_reference_parameters(cls, data: Any, info: ValidationInfo) -> Any:
        """
        Fill unset reference parameters and record their sources in `reference_sources`.

        Given `reference_sources` (a mapping by parameter, or a single `reference_source`
        for both, as files written before they were tracked per parameter have it; given
        besides a mapping, e.g. merged into a dump, the single source overrides it) describe
        the given values (e.g. from a dump) and are kept; given values without one are
        ``"explicit"``. Assigning `alpha_x` or `beta_x` makes that parameter explicit. A
        model deriving its reference parameters (:attr:`derives_reference`) derives every
        one that is neither explicit nor published again from the given model parameters,
        so a dump of a derived value stays consistent with its model parameters. In the
        validation context of :meth:`from_base_data`, given values are kept even if NaN and
        are ``"base_data"``.
        """
        if not isinstance(data, dict):
            return data
        if info.field_name is not None and info.field_name not in REFERENCE_FIELDS:
            return data
        base_data = bool(info.context and info.context.get(_BASE_DATA_CONTEXT))
        return cls._resolved_reference(dict(data), assigned=info.field_name, base_data=base_data)

    @classmethod
    def _resolved_reference(cls, data: dict, assigned: Optional[str], base_data: bool) -> dict:
        """
        Resolve the reference parameters in `data`, see :meth:`_resolve_reference_parameters`.

        `assigned` names a reference parameter just assigned, which becomes explicit.
        """
        legacy = data.pop("referenceSource", None)
        legacy = data.pop("reference_source", legacy)
        given = data.pop("referenceSources", None)
        given = data.pop("reference_sources", given)
        given_sources = {**_reference_source_mapping(given), **_reference_source_mapping(legacy)}
        if assigned is not None:
            given_sources[assigned] = "explicit"

        sources: dict[str, str] = {}
        missing = []
        for name in REFERENCE_FIELDS:
            alias = cls.model_fields[name].alias
            if alias in data:
                value = data.pop(alias)
                data.setdefault(name, value)
            if name not in data or (not base_data and _unset_reference(data[name])):
                data.pop(name, None)
                missing.append(name)
            else:
                sources[name] = given_sources.get(name, "base_data" if base_data else "explicit")

        defaults = default_reference_parameters()
        for name in missing:
            data[name] = defaults[name]
            sources[name] = "default"

        rederive = cls.derives_reference and not base_data
        derive = [
            name
            for name in REFERENCE_FIELDS
            if (name in missing or rederive) and sources[name] not in FIXED_REFERENCE_SOURCES
        ]
        if derive:
            derived = cls.derive_reference(data)
            for name in derive:
                if name in derived:
                    data[name] = derived[name]
                    sources[name] = "estimated"
                elif sources[name] == "estimated":
                    data[name] = defaults[name]
                    sources[name] = "default"
            cls._log_derived(data, {name: derived[name] for name in missing if name in derived})
        data["reference_sources"] = sources
        return data

    @model_validator(mode="after")
    def _resolve_assigned_reference_parameter(self, info: ValidationInfo) -> Any:
        """
        Resolve the reference parameters after an assignment, as on construction.

        A reference parameter assigned NaN is resolved anew. Assigning another parameter of a
        model deriving its reference parameters derives those again that are neither
        explicit nor published.
        """
        name = info.field_name
        if name is None:
            return self
        if name in REFERENCE_FIELDS:
            if not _unset_reference(getattr(self, name)):
                return self
            data = {**self.__dict__, name: None}
        elif self.derives_reference and any(
            source not in FIXED_REFERENCE_SOURCES for source in self.reference_sources.values()
        ):
            data = dict(self.__dict__)
        else:
            return self
        resolved = self._resolved_reference(data, assigned=None, base_data=False)
        for field in (*REFERENCE_FIELDS, "reference_sources"):
            self.__dict__[field] = resolved[field]
        return self

    @classmethod
    def derive_reference(cls, parameters: Mapping[str, Any]) -> dict[str, float]:
        """
        Derive the reference parameters from model parameters, as this model does.

        Parameters
        ----------
        parameters : Mapping[str, Any]
            Tissue parameters by field name or alias, e.g. the values entered in an editor;
            parameters not given take the field default.

        Returns
        -------
        dict[str, float]
            The reference parameters derivable from what `parameters` gives. Empty for a
            model taking them as given (:attr:`derives_reference` false); MKM overrides
            this.
        """
        return {}

    @classmethod
    def _log_derived(cls, data: Mapping[str, Any], derived: Mapping[str, float]) -> None:
        """Report reference parameters derived because they were not given."""

    @classmethod
    def from_base_data(
        cls, data: Optional[Mapping[str, Any]] = None, **overrides
    ) -> "TissueProperties":
        """
        Build a tissue class of base data, e.g. one entry of a machine's tissueParameters.

        Like :meth:`from_dict`, but a parameter the base data gives as NaN is kept as NaN
        ("not declared"), the reference parameters included, so that matching a structure's
        tissue to the class does not compare it (``ignore_undeclared``). The reference
        parameters the class does give are recorded as ``reference_source="base_data"``;
        absent ones are resolved as for any tissue. Only machine and table loaders build
        tissues this way; a structure's tissue never keeps NaN reference parameters.
        """
        return cls._build(dict(data or {}), overrides, base_data=True)

    @property
    def display_name(self) -> str:
        """Name of the model these properties belong to, as shown to the user."""
        return self.model

    @classmethod
    def _resolve_model(cls, model_name: str) -> str:
        """Map a model name through _MODEL_ALIASES if applicable."""
        return cls._MODEL_ALIASES.get(model_name, model_name)

    def _same_model_as(self, other: "TissueProperties") -> bool:
        """
        Whether `other` has a compatible parameter class, i.e. is comparable at all.

        Types are compatible as long as one derives from the other, so the reference-only
        `TissueProperties` is comparable to any model. The provenance (model name, LEM
        version) is not considered: variants of a model sharing a parameter class (MKM and
        mMKM, the LEM versions) are compared by their parameters.
        """
        return isinstance(self, type(other)) or isinstance(other, type(self))

    def same_provenance(self, other: "TissueProperties") -> bool:
        """Whether `other` is labelled with the same provenance, e.g. both ``"LEM_IV"``."""
        return self.provenance == other.provenance

    def _compare_fields_with(self, other: "TissueProperties") -> tuple[str, ...]:
        """
        Return this model's identifying fields that `other` also carries.

        Falls back to the reference parameters if `other` has none of them (e.g. an MKM
        tissue compared against a reference-only one), so a comparison is never vacuous.
        """
        other_fields = type(other).model_fields
        common = tuple(f for f in self.identifying_fields if f in other_fields)
        return common or TissueProperties.identifying_fields

    @staticmethod
    def _values_equal(a, b) -> bool:
        """Equality that treats NaN == NaN as a match (unlike normal float semantics)."""
        if isinstance(a, float) and isinstance(b, float):
            if math.isnan(a) and math.isnan(b):
                return True
            return a == b
        return a == b

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, BaseModel):
            return NotImplemented
        if type(self) is not type(other):
            return False
        return all(
            self._values_equal(getattr(self, name), getattr(other, name))
            for name in type(self).model_fields
            if name not in self.metadata_fields
        )

    __hash__ = None

    def matches(
        self,
        other: "TissueProperties",
        fields: Optional[Sequence[str]] = None,
        resolve_model_aliases: bool = True,
        ignore_undeclared: bool = False,
    ) -> bool:
        """
        Compare this TissueProperties instance to another on a chosen set of fields.

        Parameters
        ----------
        other : TissueProperties
            The instance to compare against.
        fields : Sequence[str], optional
            Names of fields to compare (Python attribute names, e.g. "alpha_x", not
            aliases). Defaults to this model's :attr:`identifying_fields`, narrowed to
            those `other` also carries; the provenance labels (:attr:`provenance_fields`)
            are only compared if named here.
        resolve_model_aliases : bool
            If True (default), compare `model` after mapping both sides through
            `_MODEL_ALIASES` (e.g. "LEM" and "LEM_I" are treated as equal), if `fields`
            names it.
        ignore_undeclared : bool
            If True, a numeric field `other` leaves NaN (not declared) is not compared, so
            any value of this tissue matches it. Meant for matching a tissue against a
            tissue class of base data that does not document every parameter (e.g. LEM IV
            classes without ``Dt`` / ``rNucleus``). At least one compared field must still
            be declared and equal.

        Returns
        -------
        bool
            False if the two have incompatible parameter classes (e.g. LEM vs MKM);
            otherwise True if all compared fields are equal (NaN treated as equal to NaN).
            Types are compatible as long as one derives from the other, so the plain
            `TissueProperties` (reference-only) type is compatible with any subclass. The
            provenance is not compared by default: an MKM tissue matches an mMKM class and
            a LEM I tissue a LEM IV class if their parameters do.

        Raises
        ------
        ValueError
            If `fields` contains a name that isn't a field on both models.
        """
        if not self._same_model_as(other):
            return False

        compare_fields = fields if fields is not None else self._compare_fields_with(other)

        unknown = [
            f
            for f in compare_fields
            if f not in type(self).model_fields or f not in type(other).model_fields
        ]
        if unknown:
            raise ValueError(f"Unknown field(s): {unknown}")

        compared = 0
        for field_name in compare_fields:
            a = getattr(self, field_name)
            b = getattr(other, field_name)

            if field_name == "model" and resolve_model_aliases:
                a = self._resolve_model(a)
                b = self._resolve_model(b)

            if ignore_undeclared and _undeclared(b):
                continue
            if not self._values_equal(a, b):
                return False
            compared += 1
        return compared > 0 or not ignore_undeclared

    def find_match(
        self,
        candidates: Sequence["TissueProperties"],
        fields: Optional[Sequence[str]] = None,
        resolve_model_aliases: bool = True,
        ignore_undeclared: bool = False,
    ) -> Optional["TissueProperties"]:
        """
        Return the candidate this tissue matches, None if it matches none.

        Of several matching candidates, the first one of the same provenance (see
        :meth:`same_provenance`) is preferred, else the first one. The parameters are passed
        through to :meth:`matches`.
        """
        matching = [
            candidate
            for candidate in candidates
            if self.matches(
                candidate,
                fields=fields,
                resolve_model_aliases=resolve_model_aliases,
                ignore_undeclared=ignore_undeclared,
            )
        ]
        return next(
            (candidate for candidate in matching if self.same_provenance(candidate)),
            matching[0] if matching else None,
        )

    @classmethod
    def lookup(
        cls,
        source: Sequence["TissueProperties"],
        candidates: Sequence["TissueProperties"],
        fields: Optional[Sequence[str]] = None,
        resolve_model_aliases: bool = True,
        ignore_undeclared: bool = False,
    ) -> List[Tuple["TissueProperties", Optional["TissueProperties"]]]:
        """
        Match each tissue in `source` to a matching tissue in `candidates`.

        Each tissue gets the first matching candidate of its own provenance, else the first
        matching one, see :meth:`find_match`.

        Parameters
        ----------
        source : Sequence[TissueProperties]
            Tissues to find matches for (e.g. tissue_classes_cst).
        candidates : Sequence[TissueProperties]
            Pool of tissues to match against (e.g. tissue_classes_machine).
        fields : Sequence[str], optional
            Fields to compare. Defaults to the identifying fields of each source
            tissue's model, see `matches`.
        resolve_model_aliases, ignore_undeclared : bool
            Passed through to `matches`.

        Returns
        -------
        List[Tuple[TissueProperties, Optional[TissueProperties]]]
            (source_tissue, matched_candidate_or_None) pairs, in source order.
        """
        return [
            (
                src_tp,
                src_tp.find_match(
                    candidates,
                    fields=fields,
                    resolve_model_aliases=resolve_model_aliases,
                    ignore_undeclared=ignore_undeclared,
                ),
            )
            for src_tp in source
        ]

    @classmethod
    def class_for_model(cls, model_name: str) -> type["TissueProperties"]:
        """
        Tissue class implementing the named model, plain `TissueProperties` if none does.

        Besides the canonical model names, every name encoding a LEM version selects
        :class:`LEMTissueProperties`, which takes the version over into its own field;
        this covers the names base data is tabulated under ("LEMI", "LEM 2").
        """
        from ._provenance import get_provenance  # noqa: PLC0415 - registry imports this module

        return get_provenance(model_name).tissue_type

    @property
    def provenance(self) -> str:
        """
        Canonical provenance these parameters belong to, e.g. ``"LEM_II"`` or ``"MKM"``.

        See :func:`~pyRadPlan.biology.tissues.tissue_provenance`.
        """
        from ._provenance import tissue_provenance  # noqa: PLC0415 - registry imports this module

        return tissue_provenance(self)

    @classmethod
    def from_dict(cls, data: dict, **overrides) -> "TissueProperties":
        """
        Create an instance from a plain dict, with optional field overrides.

        Called on the base `TissueProperties` class, the `model` entry in `data` (e.g.
        ``"LEM_II"``, ``"MKM"`` or ``"mMKM"``, optionally with a separate ``version``) selects
        the provenance and with it the tissue class to build, see
        :func:`~pyRadPlan.biology.tissues.get_provenance`; called on a specific subclass,
        that subclass is used directly. A version carried by the model name is taken over
        into the `version` field, unless `data` gives one explicitly.

        Field names may be given as their camelCase aliases (``alphaX``, ``rNucleus``),
        so metadata read from ``.mat`` base data can be passed through as it is stored.
        Entries no tissue class knows (e.g. ``ref_Kernel``) are ignored. Unset (NaN)
        reference parameters are resolved as described in the class docstring; base data
        keeping them unset is built by :meth:`from_base_data` instead.
        """
        return cls._build(dict(data), overrides, base_data=False)

    @classmethod
    def _build(cls, data: dict, overrides: Mapping[str, Any], base_data: bool):
        params = {**data, **overrides}
        target_cls = cls
        if cls is TissueProperties:
            from ._provenance import get_provenance  # noqa: PLC0415 - registry imports this module

            provenance = get_provenance(params.get("model"), params.get("version"))
            target_cls = provenance.tissue_type
            for key, value in provenance.parameters.items():
                if key == "version" and params.get("version") not in (None, ""):
                    continue
                params[key] = value
            if params.get("model") in (None, ""):
                params.pop("model", None)
        context = {_BASE_DATA_CONTEXT: True} if base_data else None
        try:
            return target_cls.model_validate(params, context=context)
        except TypeError as e:
            raise ValueError(f"Invalid data for TissueProperties: {e}") from e

    def adopt_provenance(self, target: Any) -> "TissueProperties":
        """
        Convert the tissue to another provenance, keeping the parameters both share.

        Shorthand for ``adopt_provenance(self, target).tissue``, see
        :func:`~pyRadPlan.biology.tissues.adopt_provenance` for the parameters carried over
        and a report of them.
        """
        from ._conversion import adopt_provenance  # noqa: PLC0415 - imports this module

        return adopt_provenance(self, target).tissue

    @classmethod
    def from_preset(
        cls, cell_line: str, model: str, source: Optional[str] = None, **overrides
    ) -> "TissueProperties":
        """
        Build the tissue of a cell line preset, see :func:`get_cell_line_preset`.

        The tissue records the cell line and the preset's source (`cell_line_source`), which
        identifies the preset among those of its cell line and provenance. Reference
        parameters the preset gives are ``"published"`` (see :data:`REFERENCE_SOURCES`).
        Called on a subclass, that subclass is built instead of the preset's own.

        Parameters
        ----------
        cell_line : str
            Cell line, case-insensitive, e.g. ``"HSG"``.
        model : str
            Provenance as :func:`~pyRadPlan.biology.tissues.get_provenance` accepts it,
            e.g. ``"MKM"``, ``"LEM_II"`` or ``"LEM"`` (LEM I).
        source : str, optional
            Which of several presets of the cell line and provenance, by its label (e.g.
            ``"Elsässer 2008"``) or full citation (a tissue's `cell_line_source`); the
            default preset if None.
        **overrides
            Field values replacing those of the preset.
        """
        from ._presets import get_cell_line_preset  # noqa: PLC0415 - presets import this module

        preset = get_cell_line_preset(cell_line, model, source)
        target = None if cls is TissueProperties else cls
        return preset.tissue(tissue_type=target, **overrides)


class LEMTissueProperties(TissueProperties):
    """
    Tissue parameters for the Local Effect Model (LEM) RBE model.

    Every version is described by this one class, through its `version` field. Like the
    model name, the version labels the provenance of the parameters (see
    :attr:`~TissueProperties.provenance_fields`): matching a tissue to a tissue class of
    base data compares the parameters only, so a LEM I tissue matches a LEM IV class with
    the same parameters, and dose calculations warn about the differing label. A version
    carried by the model name ("LEM_II", "LEM 2") is taken over into `version`, leaving
    `model` as the plain ``"LEM"``.

    LEM takes the photon LQ parameters `alpha_x` / `beta_x` as model inputs, so for LEM they
    are model parameters and the reference parameters at once.
    """

    model: str = Field("LEM", description="Radiobiological model used for this tissue")
    version: Literal["I", "II", "III", "IV"] = Field(
        "I", description="Version of the Local Effect Model (LEM I to LEM IV)"
    )
    d_t: float = Field(float("nan"), description="Dose threshold", alias="Dt")
    r_nucleus: float = Field(float("nan"), description="Cell nucleus radius", alias="rNucleus")

    # LEM base data tabulates alpha/beta per reference LQ parameters and cell geometry.
    identifying_fields: ClassVar[tuple[str, ...]] = (
        "alpha_x",
        "beta_x",
        "d_t",
        "r_nucleus",
    )
    provenance_fields: ClassVar[tuple[str, ...]] = ("model", "version")
    model_parameter_fields: ClassVar[tuple[str, ...]] = (
        "alpha_x",
        "beta_x",
        "d_t",
        "r_nucleus",
    )

    @model_validator(mode="before")
    @classmethod
    def _split_version_from_model(cls, data: Any) -> Any:
        """Move a version carried by the model name into `version`, if it has none yet."""
        if not isinstance(data, dict):
            return data
        model_name = data.get("model")
        if not isinstance(model_name, str):
            return data
        version = _parse_lem_version(model_name)
        if version is None:
            return data
        data = {**data, "model": "LEM"}
        data.setdefault("version", version)
        return data

    @field_validator("version", mode="before")
    @classmethod
    def _canonical_version(cls, value: Any) -> Any:
        """Accept a version given as a number (2, "2") besides its numeral ("II")."""
        if isinstance(value, str):
            return _parse_lem_version(value) or value
        if isinstance(value, Integral) and 1 <= value <= len(LEM_VERSIONS):
            return LEM_VERSIONS[int(value) - 1]
        return value

    @property
    def display_name(self) -> str:
        """Name of the model including its version, e.g. ``"LEM II"``."""
        return f"{self.model} {self.version}"


def _finite(value: Any) -> Optional[float]:
    """Return a finite number as float, None for anything else (NaN, None, bools, strings)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) if math.isfinite(value) else None
    return None


class MKMTissueProperties(TissueProperties):
    """
    Tissue parameters for the Microdosimetric Kinetic Model (MKM) and its modified form.

    The model parameters are the nucleus and domain radius `r_nucleus` / `r_domain` and the
    LQ coefficients `alpha_0` / `beta_0`; `model` labels them as MKM or mMKM. The reference
    parameters `alpha_x` / `beta_x` are not MKM inputs: they follow from the model
    parameters by the conversion :meth:`estimate_reference_parameters`,

    - ``alpha_x = alpha_0 + beta_0 * z_1D,ref`` and ``beta_x = beta_0``,

    with ``z_1D,ref`` the dose-mean specific energy the reference radiation
    (`reference_radiation`, see :func:`~pyRadPlan.biology.tissues.get_reference_radiation`)
    deposits in a domain, saturation-corrected (``z*_1D,ref``) for mMKM. Unless given
    explicitly, they are kept derived ("estimated") from the current model parameters, also
    when those are assigned later; given explicitly, they are an override and kept as they
    are.
    """

    model: str = Field("MKM", description="Radiobiological model used for this tissue")
    r_nucleus: float = Field(float("nan"), description="Cell nucleus radius", alias="rNucleus")
    r_domain: float = Field(float("nan"), description="Domain radius", alias="rDomain")
    alpha_0: float = Field(float("nan"), description="Baseline alpha parameter", alias="alpha0")
    beta_0: float = Field(float("nan"), description="Baseline beta parameter", alias="beta0")
    reference_radiation: str = Field(
        DEFAULT_REFERENCE_RADIATION,
        description=(
            "Registered reference radiation alpha_x / beta_x are derived for, see "
            "get_reference_radiation"
        ),
        alias="referenceRadiation",
    )

    # MKM zs kernels depend only on the domain/nucleus geometry; the reference LQ
    # parameters and alpha_0/beta_0 come from the tissue itself, not from the base data.
    identifying_fields: ClassVar[tuple[str, ...]] = (
        "r_nucleus",
        "r_domain",
    )
    model_parameter_fields: ClassVar[tuple[str, ...]] = (
        "r_nucleus",
        "r_domain",
        "alpha_0",
        "beta_0",
    )
    derives_reference: ClassVar[bool] = True

    #: Provenances evaluating the saturation-corrected specific energy z*.
    SATURATION_CORRECTED_MODELS: ClassVar[tuple[str, ...]] = ("mMKM",)

    #: How each reference LQ parameter is derived from the model parameters.
    REFERENCE_FORMULAS: ClassVar[dict[str, str]] = {
        "alpha_x": "alpha_0 + beta_0 * z_1D,ref",
        "beta_x": "beta_0",
    }

    @field_validator("reference_radiation")
    @classmethod
    def _registered_reference_radiation(cls, value: str) -> str:
        try:
            get_reference_radiation(value)
        except KeyError as exc:
            raise ValueError(exc.args[0]) from None
        return value

    @classmethod
    def saturation_corrected(cls, model: Any) -> bool:
        """Whether the provenance `model` (e.g. ``"mMKM"``) uses the saturation-corrected z*."""
        from ._provenance import get_provenance  # noqa: PLC0415 - registry imports this module

        return get_provenance(model).name in cls.SATURATION_CORRECTED_MODELS

    @classmethod
    def reference_specific_energy(
        cls,
        r_domain: float,
        r_nucleus: float = float("nan"),
        beta_0: float = float("nan"),
        model: str = "MKM",
        reference_radiation: Optional[str] = None,
    ) -> float:
        """
        Dose-mean specific energy [Gy] the reference radiation deposits in a domain.

        For MKM the plain ``z_1D,ref`` (Kase et al., Radiat. Res. 166 (2006) 629, Eq. 3):
        with the dose-mean lineal energy ``y_D`` of the reference radiation,
        ``z_1D = y_D / (rho * pi * r_domain**2)`` for unit density (0.1602 converts
        keV/µm³ to Gy). For mMKM the saturation-corrected
        ``z*_1D,ref = z_0**2 * (1 - exp(-z_1D**2 / z_0**2)) / z_1D`` with
        ``z_0 = r_nucleus**2 / (r_domain * sqrt(beta_0 * (r_domain**2 + r_nucleus**2)))``
        (Inaniwa et al., Phys. Med. Biol. 55 (2010) 6721, Eqs. 4-5), see
        :func:`~pyRadPlan.biology.tissues.saturation_corrected_specific_energy`.

        Parameters
        ----------
        r_domain : float
            Domain radius [µm].
        r_nucleus : float
            Nucleus radius [µm], needed for mMKM only.
        beta_0 : float
            beta_0 [Gy⁻²], needed for mMKM only.
        model : str
            Provenance, ``"MKM"`` or ``"mMKM"``.
        reference_radiation : str, optional
            Registered reference radiation; the default one if None.

        Returns
        -------
        float
            The specific energy, NaN if the parameters it needs are unset or invalid.
        """
        z_1d = get_reference_radiation(reference_radiation).specific_energy(
            _finite(r_domain) or math.nan
        )
        if not cls.saturation_corrected(model) or math.isnan(z_1d):
            return z_1d
        r_d, r_n, b_0 = _finite(r_domain), _finite(r_nucleus), _finite(beta_0)
        if not (r_d and r_d > 0 and r_n and r_n > 0 and b_0 is not None and b_0 >= 0):
            return math.nan
        return saturation_corrected_specific_energy(
            z_1d, saturation_specific_energy(r_d, r_n, b_0)
        )

    @classmethod
    def estimate_reference_parameters(
        cls,
        alpha_0: Any = float("nan"),
        beta_0: Any = float("nan"),
        r_domain: Any = float("nan"),
        r_nucleus: Any = float("nan"),
        model: str = "MKM",
        reference_radiation: Optional[str] = None,
    ) -> dict[str, float]:
        """
        Derive the reference LQ parameters from the MKM model parameters.

        In the MKM, ``alpha = alpha_0 + beta_0 * z_1D`` and ``beta = beta_0`` for any
        radiation (Hawkins, Radiat. Res. 140 (1994) 366, Eq. 26; Kase et al., Radiat. Res.
        166 (2006) 629, Eq. 2; Kase et al., Phys. Med. Biol. 53 (2008) 37, Eq. 4). Evaluated
        for the reference radiation this gives ``alpha_x = alpha_0 + beta_0 * z_1D,ref`` and
        ``beta_x = beta_0``, with ``z_1D,ref`` saturation-corrected for mMKM, see
        :meth:`reference_specific_energy`.

        Returns
        -------
        dict[str, float]
            Only the parameters actually derivable from the finite arguments given:
            `alpha_x` needs `alpha_0`, `beta_0` and `r_domain` (and `r_nucleus` for mMKM),
            `beta_x` only `beta_0`. Empty if neither is.
        """
        a_0, b_0 = _finite(alpha_0), _finite(beta_0)
        estimated: dict[str, float] = {}
        if a_0 is not None and b_0 is not None:
            z_ref = cls.reference_specific_energy(
                r_domain, r_nucleus, b_0, model=model, reference_radiation=reference_radiation
            )
            if math.isfinite(z_ref):
                estimated["alpha_x"] = a_0 + b_0 * z_ref
        if b_0 is not None:
            estimated["beta_x"] = b_0
        return estimated

    def derived_reference_parameters(self) -> dict[str, float]:
        """
        Return the reference parameters this tissue's model parameters convert to.

        See :meth:`estimate_reference_parameters`; independent of the values `alpha_x` /
        `beta_x` currently hold, e.g. an explicit override.
        """
        return self.estimate_reference_parameters(
            alpha_0=self.alpha_0,
            beta_0=self.beta_0,
            r_domain=self.r_domain,
            r_nucleus=self.r_nucleus,
            model=self.model,
            reference_radiation=self.reference_radiation,
        )

    @classmethod
    def derive_reference(cls, parameters: Mapping[str, Any]) -> dict[str, float]:
        """
        Derive the reference parameters from MKM model parameters given by name.

        See :meth:`estimate_reference_parameters`; also reads `model` (MKM or mMKM) and
        `reference_radiation`. Empty for an unknown reference radiation.
        """

        def given(name: str) -> Any:
            alias = cls.model_fields[name].alias
            return parameters.get(name, parameters.get(alias, cls.model_fields[name].default))

        try:
            return cls.estimate_reference_parameters(
                alpha_0=given("alpha_0"),
                beta_0=given("beta_0"),
                r_domain=given("r_domain"),
                r_nucleus=given("r_nucleus"),
                model=given("model"),
                reference_radiation=given("reference_radiation"),
            )
        except KeyError:
            return {}

    @classmethod
    def _log_derived(cls, data: Mapping[str, Any], derived: Mapping[str, float]) -> None:
        tissue = data.get("cell_line") or data.get("cell_Line") or "unnamed"
        for name, value in derived.items():
            logger.info(
                "MKM tissue '%s': %s not given, deriving %s = %.4g.",
                tissue,
                name,
                cls.REFERENCE_FORMULAS[name],
                value,
            )

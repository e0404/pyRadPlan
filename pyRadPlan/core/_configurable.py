"""Configuration introspection for algorithm classes.

Provides the :class:`ConfigurableAlgorithm` mixin, which collects the public
class-level annotations of an algorithm (dose engine, stf generator, planning
problem) into a dynamically generated pydantic model. Algorithms stay plain
Python classes; the generated model is a companion used for validation,
(de-)serialization of configuration dictionaries (e.g. ``pln.prop_dose_calc``)
and dynamic GUI generation.
"""

import inspect
import logging
import warnings
from copy import deepcopy
from typing import (
    Any,
    ClassVar,
    Optional,
    Type,
    Union,
    get_origin,
    get_type_hints,
)
from collections.abc import Mapping

import annotated_types
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from .datamodel import PyRadPlanBaseModel

logger = logging.getLogger(__name__)

_CONFIG_MODEL_ATTR = "__config_model__"


class AlgorithmParameterMetadata:
    """
    Optional metadata to attach to algorithm configuration parameters.

    Intended for use within ``Annotated`` type hints of configuration
    parameters of a :class:`ConfigurableAlgorithm`. It does not influence
    validation, but provides hints for dynamically generated configuration
    interfaces (e.g. GUI widgets).

    Parameters
    ----------
    configurable : bool, optional, default=True
        Whether the parameter should be exposed in configuration interfaces.
        Set to False for programmatic inputs (e.g. data cubes) that should
        validate but not appear in a GUI.
    kind : str, optional, default='numeric'
        Semantic type of the parameter (e.g. 'numeric', 'data').
    advanced : bool, optional, default=False
        Whether the parameter is an advanced setting (GUIs may fold it away).
    label : str, optional
        Human-readable label. If None, interfaces derive one from the name.
    """

    def __init__(
        self,
        configurable: bool = True,
        kind: str = "numeric",
        advanced: bool = False,
        label: Optional[str] = None,
    ):
        self.configurable = configurable
        self.kind = kind
        self.advanced = advanced
        self.label = label

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"


class AlgorithmConfig(PyRadPlanBaseModel):
    """
    Base class for dynamically generated algorithm configuration models.

    Inherits camelCase aliasing, assignment validation and arbitrary type
    support from :class:`PyRadPlanBaseModel`. Extra keys are ignored, since
    unknown configuration entries are handled (with warnings) by
    :meth:`ConfigurableAlgorithm.apply_config`.
    """

    model_config = ConfigDict(extra="ignore")


def _is_classvar(annotation: Any) -> bool:
    """Check if an annotation (possibly a string) designates a ClassVar."""
    if annotation is ClassVar or get_origin(annotation) is ClassVar:
        return True
    return isinstance(annotation, str) and annotation.replace(" ", "").startswith(
        ("ClassVar[", "ClassVar", "typing.ClassVar")
    )


def _wrap_mutable_default(default: Any) -> Any:
    """Wrap mutable defaults in a deepcopy factory to avoid shared state."""
    if isinstance(default, (list, dict, set, bytearray, np.ndarray)):
        return Field(default_factory=lambda d=default: deepcopy(d))
    return default


class ConfigurableAlgorithm:
    """
    Mixin collecting class-level annotations into a pydantic config model.

    Every public (non-underscore), non-``ClassVar`` annotation declared on a
    subclass (or any of its bases that are themselves ConfigurableAlgorithm
    subclasses) is considered a configuration parameter. Class-level
    assignments provide the defaults.

    The generated model is exposed via :meth:`config_model` and used by
    :meth:`apply_config` to validate configuration dictionaries before
    assigning them to the instance.
    """

    @classmethod
    def config_model(cls) -> Type[AlgorithmConfig]:
        """
        Get the pydantic model describing this algorithm's configuration.

        The model is generated lazily on first access and cached per class
        (subclasses get their own model including inherited parameters).

        Returns
        -------
        Type[AlgorithmConfig]
            A pydantic model class named ``<ClassName>Config`` with one field
            per configuration parameter.
        """
        if _CONFIG_MODEL_ATTR in cls.__dict__:
            return cls.__dict__[_CONFIG_MODEL_ATTR]

        try:
            hints = get_type_hints(cls, include_extras=True)
        except (NameError, TypeError) as exc:
            warnings.warn(
                f"Could not resolve all annotations of {cls.__name__}: {exc}. "
                "Configuration parameters will not be type-validated."
            )
            hints = {}

        field_defs = {
            name: cls._safe_field_definition(
                name, cls._field_definition(name, hints.get(name, Any))
            )
            for name in cls._collect_parameter_names(hints)
        }

        model = create_model(
            f"{cls.__name__}Config",
            __base__=AlgorithmConfig,
            __doc__=f"Configuration parameters of {cls.__name__}.",
            **field_defs,
        )
        setattr(cls, _CONFIG_MODEL_ATTR, model)
        return model

    @classmethod
    def _collect_parameter_names(cls, hints: dict) -> list[str]:
        """Collect public, non-ClassVar annotation names along the MRO."""
        names: list[str] = []
        for klass in reversed(cls.__mro__):
            if klass is ConfigurableAlgorithm or not (
                isinstance(klass, type) and issubclass(klass, ConfigurableAlgorithm)
            ):
                continue
            for name, raw_annotation in inspect.get_annotations(klass).items():
                if name.startswith("_") or name in names:
                    continue
                if _is_classvar(hints.get(name, raw_annotation)):
                    continue
                names.append(name)
        return names

    @classmethod
    def _field_definition(cls, name: str, annotation: Any) -> tuple:
        """Build the (annotation, default) tuple for a parameter."""
        default = getattr(cls, name, PydanticUndefined)
        if isinstance(default, FieldInfo):
            return (annotation, default)
        if inspect.isdatadescriptor(default) or inspect.isroutine(default):
            # Property-backed parameter: the default is managed by the
            # class itself (e.g. in __init__ through the setter)
            default = PydanticUndefined
        if default is PydanticUndefined or default is NotImplemented:
            # Parameters without class-level default (e.g. still set in
            # __init__) become optional with default None
            return (Optional[annotation], None)
        return (annotation, _wrap_mutable_default(default))

    @classmethod
    def _safe_field_definition(cls, name: str, definition: tuple) -> tuple:
        """Probe schema generation, degrading unsupported types to Any."""
        try:
            create_model("_ProbeModel", __base__=AlgorithmConfig, **{name: definition})
            return definition
        except Exception as exc:  # noqa: BLE001 - schema generation must never block
            logger.debug(
                "Configuration field %s.%s cannot be schematized (%s); falling back to Any",
                cls.__name__,
                name,
                exc,
            )
            fallback = definition[1]
            if isinstance(fallback, FieldInfo):
                fallback = fallback.get_default(call_default_factory=True)
                if fallback is PydanticUndefined:
                    fallback = None
            return (Optional[Any], _wrap_mutable_default(fallback))

    def _init_config_defaults(self) -> None:
        """
        Initialize instance attributes with the declared parameter defaults.

        Attributes already present in the instance ``__dict__`` are left
        untouched, so subclasses may assign values before calling this.
        """
        for name, info in self.config_model().model_fields.items():
            if name in self.__dict__:
                continue
            if inspect.isdatadescriptor(getattr(type(self), name, None)):
                # don't overwrite property-backed parameters with model defaults
                continue
            setattr(self, name, info.get_default(call_default_factory=True))

    def get_config(self, validate: bool = False) -> AlgorithmConfig:
        """
        Get the current configuration values as a model instance.

        Parameters
        ----------
        validate : bool, optional, default=False
            If True, the current values are validated against the model.

        Returns
        -------
        AlgorithmConfig
            Instance of :meth:`config_model` holding the current values.
        """
        model = self.config_model()
        data = {name: getattr(self, name, None) for name in model.model_fields}
        if validate:
            return model.model_validate(data)
        return model.model_construct(**data)

    def apply_config(
        self,
        values: Union[Mapping[str, Any], BaseModel],
        *,
        warn_on_overwrite: bool = False,
        overwrite_source: str = "config",
        strict: bool = False,
    ) -> None:
        """
        Validate configuration values and assign them to the instance.

        Keys may be given in snake_case or camelCase. Known parameters are
        validated through the configuration model. Unknown keys and values
        failing validation emit a warning but are assigned as-is, unless
        ``strict`` is set.

        Parameters
        ----------
        values : Mapping[str, Any] or BaseModel
            Configuration values. For a model instance, only explicitly set
            fields are applied.
        warn_on_overwrite : bool, optional, default=False
            Log a warning for each assigned parameter.
        overwrite_source : str, optional, default='config'
            Source description used in overwrite log messages.
        strict : bool, optional, default=False
            If True, raise a ValidationError instead of falling back to raw
            assignment when a value fails validation.
        """
        if isinstance(values, BaseModel):
            values = {name: getattr(values, name) for name in values.model_fields_set}

        model = self.config_model()
        alias_map = {info.alias: name for name, info in model.model_fields.items() if info.alias}

        known: dict[str, Any] = {}
        unknown: dict[str, Any] = {}
        for key, value in values.items():
            name = key if key in model.model_fields else alias_map.get(key)
            if name is None:
                unknown[key] = value
            else:
                known[name] = value

        for key in unknown:
            if not hasattr(self, key):
                warnings.warn(f'Property "{key}" not found in {type(self).__name__}!')
        self._assign_entries(unknown, warn_on_overwrite, overwrite_source)

        if not known:
            return

        failed: dict[str, Any] = {}
        try:
            validated = model.model_validate(known)
        except ValidationError as exc:
            if strict:
                raise
            failed = self._pop_failed_entries(known, exc)
            validated = model.model_validate(known) if known else None

        self._assign_entries(
            {key: getattr(validated, key) for key in known}, warn_on_overwrite, overwrite_source
        )
        self._assign_entries(failed, warn_on_overwrite, overwrite_source)

    def _assign_entries(
        self, entries: Mapping[str, Any], warn_on_overwrite: bool, overwrite_source: str
    ) -> None:
        """Assign entries to the instance, optionally logging each overwrite."""
        for key, value in entries.items():
            if warn_on_overwrite:
                logger.warning("Property overwritten from %s: %s", overwrite_source, key)
            setattr(self, key, value)

    def _pop_failed_entries(self, known: dict[str, Any], exc: ValidationError) -> dict[str, Any]:
        """Remove entries that failed validation from `known`, with a warning each."""
        failed: dict[str, Any] = {}
        failed_keys = {err["loc"][0] for err in exc.errors() if err["loc"]}
        for key in failed_keys:
            if key in known:
                failed[key] = known.pop(key)
                warnings.warn(
                    f"Configuration value for '{key}' of {type(self).__name__} failed "
                    "validation; assigning raw value."
                )
        return failed


def field_constraints(info: FieldInfo) -> dict[str, Any]:
    """
    Extract constraints and metadata from a configuration model field.

    Provides a Qt-independent description of a field for dynamic widget
    builders.

    Parameters
    ----------
    info : FieldInfo
        Field info from ``config_model().model_fields``.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'description' and, where present, numeric bounds
        ('ge', 'gt', 'le', 'lt', 'multiple_of') and 'param_meta'
        (:class:`AlgorithmParameterMetadata`).
    """
    out: dict[str, Any] = {"description": info.description}
    for meta in info.metadata:
        if isinstance(meta, annotated_types.Ge):
            out["ge"] = meta.ge
        elif isinstance(meta, annotated_types.Gt):
            out["gt"] = meta.gt
        elif isinstance(meta, annotated_types.Le):
            out["le"] = meta.le
        elif isinstance(meta, annotated_types.Lt):
            out["lt"] = meta.lt
        elif isinstance(meta, annotated_types.MultipleOf):
            out["multiple_of"] = meta.multiple_of
        elif isinstance(meta, AlgorithmParameterMetadata):
            out["param_meta"] = meta
    return out

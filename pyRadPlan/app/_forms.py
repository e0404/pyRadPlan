"""Field descriptors and form schemas generated from configuration models.

The JSON counterpart of ``pyRadPlan.gui.widgets._config_form``: a pydantic model (an
algorithm's ``config_model()``, a scenario model, a settings section) becomes a list of
``FieldDescriptor`` dicts (see ``web/PROTOCOL.md``, v1) that a frontend renders generically,
and the values it sends back are validated with the same model.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import os
import types
from collections.abc import Mapping
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Iterable,
    Literal,
    Optional,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

import numpy as np
from pydantic import BaseModel, ValidationError

from pyRadPlan.core._configurable import field_constraints, is_nested_instance, replace_nested

_UNION_TYPES = (Union, types.UnionType)


class FormValidationError(ValueError):
    """Values a form rejects, with one entry per offending field in :attr:`details`."""

    code = "invalid"
    protocol_type = "ValidationError"

    def __init__(self, message: str, details: list[dict[str, str]]) -> None:
        super().__init__(message)
        self.details = details


def validation_details(exc: ValidationError, prefix: str = "") -> list[dict[str, str]]:
    """``[{field, message}]`` of a pydantic validation error, fields as dot paths."""
    details = []
    for error in exc.errors():
        path = ".".join(str(part) for part in error.get("loc", ()))
        details.append({"field": f"{prefix}{path}", "message": error.get("msg", "")})
    return details


def jsonable(value: Any) -> Any:
    """Return a JSON-able rendering of a configuration value (unknown objects as their ``str``)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, enum.Enum):
        return jsonable(value.value)
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(v) for v in value]
    if isinstance(value, BaseModel):
        return jsonable(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    return str(value)


def _comparable(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return repr(float(value))
    if isinstance(value, np.ndarray):
        return _comparable(value.tolist())
    if isinstance(value, Mapping):
        return {str(k): _comparable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_comparable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_comparable(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
    if isinstance(value, BaseModel):
        return _comparable(value.model_dump())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _comparable(dataclasses.asdict(value))
    return jsonable(value)


def comparison_key(value: Any) -> str:
    """Return a string telling configuration values apart, for dirty tracking.

    Equal values give equal keys; tuples and lists, numpy scalars and Python numbers,
    enums and their values compare alike, non-finite floats stay distinct from ``None``.
    The plan forms of both frontends and the session's keep-or-rebuild decisions use it.
    """
    return json.dumps(_comparable(value), sort_keys=True)


def unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Strip Optional[...] from an annotation, returning (inner, was_optional)."""
    if get_origin(annotation) in _UNION_TYPES:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
        return annotation, type(None) in get_args(annotation)
    return annotation, False


_unwrap_optional = unwrap_optional


def _choice(value: Any, label: Optional[str] = None) -> dict[str, Any]:
    return {"value": jsonable(value), "label": str(value) if label is None else label}


def _number_item_type(annotation: Any) -> Optional[tuple[str, Optional[int]]]:
    origin = get_origin(annotation)
    if origin not in (list, tuple):
        return None
    args = [a for a in get_args(annotation) if a is not Ellipsis]
    if not args or any(a not in (int, float) for a in args):
        return None
    length = len(args) if origin is tuple and Ellipsis not in get_args(annotation) else None
    return ("int" if all(a is int for a in args) else "float"), length


def _type_descriptor(annotation: Any, value: Any) -> dict[str, Any]:
    """Return the ``type`` (and its type-specific keys) of a field."""
    if annotation is bool:
        return {"type": "bool"}
    if get_origin(annotation) is Literal:
        return {"type": "choice", "choices": [_choice(c) for c in get_args(annotation)]}
    if get_origin(annotation) in _UNION_TYPES:
        members = get_args(annotation)
        literals = [c for m in members if get_origin(m) is Literal for c in get_args(m)]
        if literals:
            choices = [_choice(c) for c in literals]
            if bool in members:
                choices += [_choice(True, "true"), _choice(False, "false")]
            descriptor = {"type": "choice", "choices": choices}
            if str in members:
                descriptor["editable_choice"] = True
            return descriptor
    if annotation is int and not isinstance(value, bool):
        return {"type": "int"}
    if annotation is float:
        return {"type": "float"}
    if annotation is str:
        return {"type": "str"}
    numbers = _number_item_type(annotation)
    if numbers is not None:
        descriptor = {"type": "number_list", "item_type": numbers[0]}
        if numbers[1] is not None:
            descriptor["length"] = numbers[1]
        return descriptor
    return {"type": "json"}


def _bounds(constraints: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for key, target in (
        ("ge", "min"),
        ("le", "max"),
        ("gt", "exclusive_min"),
        ("lt", "exclusive_max"),
    ):
        if constraints.get(key) is not None:
            out[target] = jsonable(constraints[key])
    return out


def field_label(name: str, meta: Any = None) -> str:
    """Label of a form field: its ``param_meta`` label (without a colon), else its name."""
    if meta is not None and getattr(meta, "label", None):
        return str(meta.label).rstrip(":")
    return name.replace("_", " ").capitalize()


_label = field_label


def nested_fields(obj: Any) -> list[tuple[str, Any, dict]]:
    """Editable fields of a dataclass or pydantic model: ``(name, annotation, constraints)``."""
    if isinstance(obj, BaseModel):
        return [
            (name, info.annotation, field_constraints(info))
            for name, info in type(obj).model_fields.items()
        ]
    try:
        hints = get_type_hints(type(obj))
    except Exception:  # noqa: BLE001 - unresolvable hints only cost the dedicated editors
        hints = {}
    return [(f.name, hints.get(f.name, Any), {}) for f in dataclasses.fields(obj) if f.init]


_nested_fields = nested_fields


def _field_descriptor(
    name: str,
    label: str,
    annotation: Any,
    value: Any,
    constraints: Mapping[str, Any],
    default: Any,
) -> dict[str, Any]:
    inner, optional = _unwrap_optional(annotation)
    descriptor: dict[str, Any] = {"name": name, "label": label}
    descriptor.update(_type_descriptor(inner, value))
    descriptor["default"] = jsonable(default)
    if optional:
        descriptor["nullable"] = True
    descriptor.update(_bounds(constraints))
    if constraints.get("description"):
        descriptor["description"] = constraints["description"]
    meta = constraints.get("param_meta")
    if meta is not None and getattr(meta, "advanced", False):
        descriptor["advanced"] = True
    return descriptor


def _group_descriptor(
    name: str, label: str, obj: Any, hidden: Iterable[str], path: tuple[str, ...] = ()
) -> dict[str, Any]:
    hidden = frozenset(hidden)
    fields = []
    for field_name, hint, constraints in _nested_fields(obj):
        field_path = (*path, field_name)
        if ".".join(field_path) in hidden:
            continue
        value = getattr(obj, field_name)
        key = ".".join((name, *field_path))
        sub_label = field_name.replace("_", " ").capitalize()
        if is_nested_instance(value):
            fields.append(_group_descriptor(name, sub_label, value, hidden, field_path))
            fields[-1]["name"] = key
        else:
            fields.append(_field_descriptor(key, sub_label, hint, value, constraints, value))
    return {"name": ".".join((name, *path)), "label": label, "type": "group", "fields": fields}


def valid_entries(
    model_cls: type[BaseModel],
    values: Mapping[str, Any],
    on_invalid: Optional[Callable[[ValidationError], None]] = None,
) -> dict[str, Any]:
    """Validate entries of *values* by field name, dropping invalid ones.

    *on_invalid* is called with the validation error of the complete mapping when some
    entries are dropped.
    """
    try:
        validated = model_cls.model_validate(dict(values))
    except ValidationError as exc:
        if on_invalid is not None:
            on_invalid(exc)
        entries = {}
        for key, value in values.items():
            try:
                validated = model_cls.model_validate({key: value})
            except ValidationError:
                continue
            entries.update({n: getattr(validated, n) for n in validated.model_fields_set})
        return entries
    return {name: getattr(validated, name) for name in validated.model_fields_set}


_valid_entries = valid_entries


def _field_default(info: Any) -> Any:
    try:
        return info.get_default(call_default_factory=True)
    except Exception:  # noqa: BLE001 - a default that cannot be built is shown as unset
        return None


def build_form(
    model_cls: type[BaseModel],
    *,
    form_id: str,
    title: str,
    values: Optional[Mapping[str, Any]] = None,
    defaults: Optional[Mapping[str, Any]] = None,
    exclude: Iterable[str] = (),
    notes: Iterable[str] = (),
) -> dict[str, Any]:
    """Describe *model_cls* as a ``FormSchema`` with the current *values*.

    Mirrors ``ConfigFormWidget``: *values* are the explicitly set entries (reported back in
    ``explicit``), *defaults* replace the model defaults for display only, nested
    dataclass / model fields become groups whose value is the dict of their overrides,
    ``AlgorithmParameterMetadata(configurable=False)`` fields and ``hidden_entries`` are
    left out. Invalid initial entries are dropped, as the Qt form does.
    """
    exclude = set(exclude)
    values = dict(values or {})
    nested_overrides: dict[str, dict] = {}
    for name, value in list(values.items()):
        info = model_cls.model_fields.get(name)
        if info is None or not isinstance(value, Mapping):
            continue
        default = (defaults or {}).get(name, _field_default(info))
        if is_nested_instance(default):
            nested_overrides[name] = deepcopy(dict(values.pop(name)))
    display_defaults = _valid_entries(model_cls, dict(defaults or {}))
    initial = _valid_entries(model_cls, {k: v for k, v in values.items() if k not in exclude})
    explicit = set(initial) | set(nested_overrides)
    draft = model_cls.model_construct(**{**display_defaults, **initial})

    fields = []
    form_values: dict[str, Any] = {}
    for name, info in model_cls.model_fields.items():
        if name in exclude:
            continue
        constraints = field_constraints(info)
        meta = constraints.get("param_meta")
        if meta is not None and not meta.configurable:
            continue
        value = getattr(draft, name, None)
        label = _label(name, meta)
        if is_nested_instance(value):
            hidden = meta.hidden_entries if meta is not None else ()
            overrides = nested_overrides.get(name, {})
            try:
                shown = replace_nested(value, overrides) if overrides else value
            except (TypeError, ValueError):
                shown, overrides = value, {}
            descriptor = _group_descriptor(name, label, shown, hidden)
            if constraints.get("description"):
                descriptor["description"] = constraints["description"]
            fields.append(descriptor)
            form_values[name] = jsonable(overrides)
            continue
        default = display_defaults.get(name, _field_default(info))
        fields.append(_field_descriptor(name, label, info.annotation, value, constraints, default))
        form_values[name] = jsonable(value)
    return {
        "id": form_id,
        "title": title,
        "fields": fields,
        "values": form_values,
        "explicit": sorted(n for n in explicit if n not in exclude),
        "notes": list(notes),
    }


def _coerce_choice(model_cls: type[BaseModel], name: str, value: Any) -> Any:
    """Turn the text ``"true"`` / ``"false"`` of a bool-or-literal choice into a bool."""
    info = model_cls.model_fields.get(name)
    if info is None or not isinstance(value, str) or value not in ("true", "false"):
        return value
    inner, _ = _unwrap_optional(info.annotation)
    if get_origin(inner) in _UNION_TYPES and bool in get_args(inner):
        return value == "true"
    return value


def validate_values(
    model_cls: type[BaseModel],
    values: Mapping[str, Any],
    *,
    exclude: Iterable[str] = (),
    base: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate the values a form sends back and return them as the plan stores them.

    Plain fields are returned as validated by *model_cls*; nested (dataclass / model)
    fields stay the dict of their overrides, checked by validating the merged tree, as
    ``ConfigFormWidget.values()`` reports them. *base* gives the values nested overrides
    apply to (default: the model defaults).

    Raises
    ------
    FormValidationError
        With one ``{field, message}`` entry per rejected field.
    """
    exclude = set(exclude)
    unknown = [n for n in values if n not in model_cls.model_fields or n in exclude]
    if unknown:
        raise FormValidationError(
            f"Unknown field(s): {sorted(unknown)}",
            [{"field": n, "message": "unknown field"} for n in unknown],
        )
    plain: dict[str, Any] = {}
    nested: dict[str, dict] = {}
    for name, value in values.items():
        info = model_cls.model_fields[name]
        default = (base or {}).get(name, _field_default(info))
        if isinstance(value, Mapping) and is_nested_instance(default):
            nested[name] = dict(value)
        else:
            plain[name] = _coerce_choice(model_cls, name, value)
    try:
        validated = model_cls.model_validate(plain)
    except ValidationError as exc:
        raise FormValidationError(str(exc), validation_details(exc)) from None
    result = {name: getattr(validated, name) for name in plain}
    for name, overrides in nested.items():
        default = (base or {}).get(name, _field_default(model_cls.model_fields[name]))
        try:
            replace_nested(default, overrides)
            merged = merged_mapping(default, overrides)
            model_cls.model_validate({name: merged})
        except ValidationError as exc:
            raise FormValidationError(str(exc), validation_details(exc)) from None
        except (TypeError, ValueError) as exc:
            raise FormValidationError(str(exc), [{"field": name, "message": str(exc)}]) from None
        result[name] = overrides
    return result


def merged_mapping(base: Any, overrides: Mapping[str, Any]) -> dict:
    """Return *base* (a dataclass or pydantic model) as a mapping with *overrides* applied.

    Built for re-validation (``model_validate({name: merged})``): fields keep their values,
    overridden nested ones become mappings; dataclass fields without ``init`` are left out.
    Invalid overrides never have to be serialized.
    """
    if isinstance(base, BaseModel):
        merged = {name: getattr(base, name) for name in type(base).model_fields}
    else:
        merged = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and is_nested_instance(current):
            merged[key] = merged_mapping(current, value)
        else:
            merged[key] = value
    return merged

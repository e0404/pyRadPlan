"""Objectives editor (protocol v1: ``objectives.*``), the rules of the Qt objectives table."""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np

from ._forms import FormValidationError, jsonable, validation_details

#: Objective added to a new row per VOI type (Qt ``_DEFAULT_OBJECTIVE_BY_TYPE``).
DEFAULT_BY_TYPE = {"TARGET": "Squared Deviation", "*": "Squared Overdosing"}


def _parameter_meta(info: Any) -> Any:
    from pyRadPlan.optimization.objectives._objective import ParameterMetadata  # noqa: PLC0415

    return next((m for m in info.metadata if isinstance(m, ParameterMetadata)), None)


def _image_label(value: Any) -> str:
    if isinstance(value, tuple) and value:
        shape = "x".join(str(n) for n in np.shape(value[0]))
        return f"Custom array ({shape})"
    get_size = getattr(value, "GetSize", None)
    if callable(get_size):
        size = "x".join(str(n) for n in get_size())
        try:
            import SimpleITK as sitk  # noqa: PLC0415

            array = sitk.GetArrayViewFromImage(value)
            if array.size and np.all(array == array.flat[0]):
                return f"Custom image ({size}, value {float(array.flat[0]):g})"
        except Exception:  # noqa: BLE001 - the summary is cosmetic
            pass
        return f"Custom image ({size})"
    return "Custom value"


def reference_values(result: Any, ct: Any) -> list[tuple[str, str, Any]]:
    """Return ``(key, label, value)`` of the result entries usable as an image reference.

    Images are used as they are; 3-D arrays (imported matRad cubes) are paired with the CT
    grid, so they need a CT. Weights and per-beam lists are skipped.
    """
    from pyRadPlan.quantities._display_names import result_key_labels  # noqa: PLC0415

    if not isinstance(result, Mapping):
        return []
    values = {}
    for key, value in result.items():
        if callable(getattr(value, "GetSize", None)):
            values[key] = value
        elif isinstance(value, np.ndarray) and value.ndim == 3 and ct is not None:
            values[key] = (value, ct.grid)
    labels = result_key_labels(values)
    return [(key, labels.get(key, key), value) for key, value in values.items()]


def reference_images(result: Any, ct: Any) -> list[dict[str, Any]]:
    """Return the result entries a Squared Mimicking reference can use, as choices."""
    return [
        {"value": {"$result": key}, "label": label}
        for key, label, _ in reference_values(result, ct)
    ]


def _encode_parameter(value: Any, result: Any) -> Any:
    if isinstance(result, Mapping):
        for key, entry in result.items():
            if entry is value or (isinstance(value, tuple) and value and value[0] is entry):
                return {"$result": key}
    if callable(getattr(value, "GetSize", None)) or isinstance(value, tuple):
        return {"$custom": _image_label(value)}
    return jsonable(value)


def objective_data(objective: Any, index: int, result: Any = None) -> dict[str, Any]:
    """``ObjectiveData`` of an objective."""
    return {
        "index": index,
        "name": objective.name,
        "priority": float(objective.priority),
        "quantity": objective.quantity,
        "robustness": objective.robustness,
        "parameters": {
            name: _encode_parameter(getattr(objective, name), result)
            for name in objective._parameter_names()
        },
    }


def _parameter_descriptor(name: str, info: Any, references: list[dict]) -> dict[str, Any]:
    from pyRadPlan.core._configurable import field_constraints  # noqa: PLC0415

    meta = _parameter_meta(info)
    kind = getattr(meta, "kind", None)
    constraints = field_constraints(info)
    entry: dict[str, Any] = {"name": name, "label": name, "default": None}
    try:
        entry["default"] = jsonable(info.get_default(call_default_factory=True))
    except Exception:  # noqa: BLE001
        entry["default"] = None
    if isinstance(kind, (list, tuple)):
        entry["type"] = "choice"
        entry["choices"] = [{"value": c, "label": str(c)} for c in kind]
    elif kind == "image_reference":
        entry["type"] = "image_ref"
        entry["choices"] = references
        entry["default"] = None
    else:
        entry["type"] = "float"
        if kind == "reference":
            entry["unit_from"] = "quantity"
        elif kind == "relative_volume":
            entry["unit"] = "%"
    for key, target in (("ge", "min"), ("le", "max"), ("gt", "exclusive_min"),
                        ("lt", "exclusive_max")):  # fmt: skip
        if constraints.get(key) is not None:
            entry[target] = jsonable(constraints[key])
    if constraints.get("description"):
        entry["description"] = constraints["description"]
    return entry


def _robustness_choices(names: list[str]) -> list[dict[str, Any]]:
    from pyRadPlan.quantities._display_names import robustness_display_name  # noqa: PLC0415

    return [{"value": n, "label": robustness_display_name(n), "identifier": n} for n in names]


def catalog(result: Any = None, ct: Any = None) -> dict[str, Any]:
    """``objectives.catalog``."""
    from pyRadPlan.optimization.objectives import get_available_objectives  # noqa: PLC0415
    from pyRadPlan.optimization.robustness import available_robustness  # noqa: PLC0415
    from pyRadPlan.quantities import get_available_quantities  # noqa: PLC0415

    references = reference_images(result, ct)
    objectives = []
    for name, cls in get_available_objectives().items():
        objectives.append(
            {
                "name": name,
                "scenario_aware": bool(getattr(cls, "scenario_aware", False)),
                "parameters": [
                    _parameter_descriptor(p, cls.model_fields[p], references)
                    for p in cls._parameter_names()
                ],
                "robustness": _robustness_choices(cls.available_robustness()),
            }
        )
    quantities = [
        {"value": identifier, "label": quantity.name, "identifier": identifier,
         "unit": f"{quantity.unit:~P}" if getattr(quantity, "unit", None) else ""}
        for identifier, quantity in get_available_quantities().items()
    ]  # fmt: skip
    names = ["none", *[n for n in available_robustness() if n not in ("none", "scenario_stack")]]
    if "scenario_stack" in available_robustness():
        names.append("scenario_stack")
    return {
        "objectives": objectives,
        "quantities": quantities,
        "robustness": _robustness_choices(names),
        "default_by_type": dict(DEFAULT_BY_TYPE),
        "reference_images": references,
    }


def _voi(cst: Any, structure: str, voi_index: Optional[int] = None) -> Any:
    vois = list(getattr(cst, "vois", None) or [])
    if voi_index is not None:
        # Names need not be unique; an in-process frontend also gives the position.
        if 0 <= int(voi_index) < len(vois) and vois[int(voi_index)].name == structure:
            return vois[int(voi_index)]
        raise KeyError(f"No structure {structure!r} at position {voi_index}")
    for voi in vois:
        if voi.name == structure:
            return voi
    raise KeyError(f"Unknown structure {structure!r}")


def objectives_of(voi: Any) -> list:
    """Return a VOI's objectives as objective instances, skipping empty and invalid entries.

    Entries are indexed in this list by the edit functions.
    """
    from pyRadPlan.optimization.objectives import get_objective  # noqa: PLC0415

    objectives = []
    for entry in getattr(voi, "objectives", None) or []:
        if entry is None or (isinstance(entry, (list, tuple)) and not entry):
            continue
        try:
            objectives.append(get_objective(entry))
        except (KeyError, ValueError, TypeError):
            continue
    return objectives


def _resolve_parameter(value: Any, result: Any, ct: Any, current: Any) -> Any:
    if isinstance(value, Mapping) and "$result" in value:
        key = value["$result"]
        if not isinstance(result, Mapping) or key not in result:
            raise ValueError(f"No result entry {key!r} to use as reference")
        entry = result[key]
        if isinstance(entry, np.ndarray):
            return (entry, ct.grid)
        return entry
    if isinstance(value, Mapping) and "$custom" in value:
        return current
    return value


def _validated(objective: Any, changes: Mapping[str, Any], prefix: str) -> Any:
    from pydantic import ValidationError  # noqa: PLC0415

    data = {name: getattr(objective, name) for name in type(objective).model_fields}
    data.update(changes)
    try:
        return type(objective).model_validate(data)
    except ValidationError as exc:
        raise FormValidationError(str(exc), validation_details(exc, prefix)) from None


def _write(ws: Any, voi: Any, objectives: list) -> None:
    from pydantic import ValidationError  # noqa: PLC0415

    from pyRadPlan.cst import validate_cst  # noqa: PLC0415

    cst = ws.cst
    previous = list(voi.objectives or [])
    try:
        voi.objectives = list(objectives)
        validate_cst(cst)
    except ValidationError as exc:
        voi.objectives = previous
        raise FormValidationError(str(exc), validation_details(exc, "objectives.")) from None
    except (ValueError, TypeError):
        voi.objectives = previous
        raise
    ws.cst = cst


def _shared_quantity(cst: Any) -> Optional[str]:
    quantities = {objective.quantity for voi in cst.vois for objective in objectives_of(voi)}
    return next(iter(quantities)) if len(quantities) == 1 else None


def add(ws: Any, structure: str, voi_index: Optional[int] = None) -> None:
    """Append the default objective of the VOI type (Qt "+ Add objective")."""
    from pyRadPlan.optimization.objectives import (  # noqa: PLC0415
        get_available_objectives,
        get_objective,
    )

    voi = _voi(ws.cst, structure, voi_index)
    name = DEFAULT_BY_TYPE.get(str(voi.voi_type).upper(), DEFAULT_BY_TYPE["*"])
    available = list(get_available_objectives())
    if name not in available and available:
        name = available[0]
    objective = get_objective(name)
    # Follow the quantity the objectives agree on, so a new objective does not silently
    # reintroduce a mix after "Change all quantities".
    quantity = _shared_quantity(ws.cst)
    if quantity is not None:
        objective = _validated(objective, {"quantity": quantity}, "")
    _write(ws, voi, [*objectives_of(voi), objective])


def _at(voi: Any, index: int) -> tuple[list, Any]:
    objectives = objectives_of(voi)
    if not 0 <= int(index) < len(objectives):
        raise IndexError(f"{voi.name} has no objective {index}")
    return objectives, objectives[int(index)]


def update(
    ws: Any,
    structure: str,
    index: int,
    changes: Mapping[str, Any],
    voi_index: Optional[int] = None,
) -> None:
    """Change priority, quantity, robustness or parameters of one objective.

    A parameter value is a number, ``{"$result": key}`` for a result entry used as an image
    reference, ``{"$custom": ...}`` to keep the current one, or (in-process) the value
    itself.
    """
    voi = _voi(ws.cst, structure, voi_index)
    objectives, objective = _at(voi, index)
    allowed = {"priority", "quantity", "robustness", "parameters"}
    unknown = set(changes) - allowed
    if unknown:
        raise ValueError(f"Unknown objective change(s): {sorted(unknown)}")
    flat = {k: v for k, v in changes.items() if k != "parameters"}
    parameters = dict(changes.get("parameters") or {})
    names = set(objective._parameter_names())
    bad = set(parameters) - names
    if bad:
        raise ValueError(f"{objective.name} has no parameter(s) {sorted(bad)}")
    for name, value in parameters.items():
        flat[name] = _resolve_parameter(value, ws.result, ws.ct, getattr(objective, name))
    objectives[int(index)] = _validated(objective, flat, "parameters.")
    _write(ws, voi, objectives)


def change_type(
    ws: Any, structure: str, index: int, name: str, voi_index: Optional[int] = None
) -> None:
    """Replace an objective by another type, keeping priority and an allowed robustness."""
    from pyRadPlan.optimization.objectives import get_objective  # noqa: PLC0415

    voi = _voi(ws.cst, structure, voi_index)
    objectives, old = _at(voi, index)
    new = get_objective(name)
    changes: dict[str, Any] = {"priority": old.priority}
    if old.robustness in type(new).available_robustness():
        changes["robustness"] = old.robustness
    objectives[int(index)] = _validated(new, changes, "")
    _write(ws, voi, objectives)


def remove(ws: Any, structure: str, index: int, voi_index: Optional[int] = None) -> None:
    """Delete one objective."""
    voi = _voi(ws.cst, structure, voi_index)
    objectives, _ = _at(voi, index)
    del objectives[int(index)]
    _write(ws, voi, objectives)


def set_quantity_all(ws: Any, quantity: str) -> None:
    """Set the quantity of every objective of every VOI in one write."""
    from pydantic import ValidationError  # noqa: PLC0415

    from pyRadPlan.cst import validate_cst  # noqa: PLC0415

    cst = ws.cst
    if cst is None:
        raise LookupError("No structure set loaded")
    previous = {voi.name: list(voi.objectives or []) for voi in cst.vois}
    try:
        for voi in cst.vois:
            objectives = objectives_of(voi)
            if any(o.quantity != quantity for o in objectives):
                voi.objectives = [
                    _validated(o, {"quantity": quantity}, "quantity") for o in objectives
                ]
        validate_cst(cst)
    except (ValidationError, FormValidationError, ValueError, TypeError):
        for voi in cst.vois:
            voi.objectives = previous[voi.name]
        raise
    ws.cst = cst

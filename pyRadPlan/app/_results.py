"""Result catalog: component keys, quantity groups, snapshot tags (Qt key grammar)."""

from __future__ import annotations

import re
from typing import Any, Mapping

import numpy as np

from ._stages import SCENARIO_LABELS_KEY

#: Snapshot names that would be mistaken for the per-beam / per-scenario result suffixes.
RESERVED_TAGS = ("beam", "scenario")

#: Colorbar / save-dialog label and unit of each known result key (Qt ``QUANTITY_META``).
QUANTITY_META: dict[str, tuple[str, str]] = {
    "physical_dose": ("Dose", "Gy"),
    "let": ("LET", "keV/μm"),
    "effect": ("Effect", ""),
    "rbe_x_dose": ("RBE-weighted Dose", "Gy (RBE)"),
    "alpha_dose": ("Alpha Dose", "Gy"),
    "sqrt_beta_dose": ("Sqrt(Beta) Dose", "Gy½"),
    "let_dose": ("LET·Dose", "Gy·keV/μm"),
}

_SUFFIXES = ("_beam", "_scenario")


def _is_volume(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.ndim == 3
    get_dimension = getattr(value, "GetDimension", None)
    return callable(get_dimension) and get_dimension() == 3


def result_components(result: Any) -> dict[str, Any]:
    """Every displayable volume of a result by component key.

    Plain 3-D entries keep their key; entry ``i`` of a list (per-beam / per-scenario
    results) becomes ``"<key> <i>"``, skipping ``None`` entries but keeping positions,
    as the Qt viewer flattens them.
    """
    if not isinstance(result, Mapping):
        return {}
    components: dict[str, Any] = {}
    for key, value in result.items():
        if isinstance(value, (list, tuple)) and key != SCENARIO_LABELS_KEY:
            for i, item in enumerate(value):
                if item is not None and _is_volume(item):
                    components[f"{key} {i}"] = item
        elif _is_volume(value):
            components[key] = value
    return components


def resolve_result_key(result: Any, key: str) -> Any:
    """Return the volume a component key names.

    Raises
    ------
    KeyError
        If the result has no such volume.
    """
    components = result_components(result)
    if key not in components:
        raise KeyError(f"No volume quantity {key!r} in the result")
    return components[key]


def scenario_labels(result: Any, dij: Any) -> list[str]:
    """Return the result's own scenario labels, else those of the dose influence."""
    labels = result.get(SCENARIO_LABELS_KEY) if isinstance(result, Mapping) else None
    if labels is not None:
        return [str(label) for label in labels]
    if hasattr(dij, "scenario_labels"):
        try:
            return [str(label) for label in dij.scenario_labels()]
        except Exception:  # noqa: BLE001 - labels are cosmetic
            return []
    return []


def _meta(base: str) -> tuple[str, str]:
    for suffix in _SUFFIXES:
        if base not in QUANTITY_META and base.endswith(suffix):
            base = base.removesuffix(suffix)
    return QUANTITY_META.get(base, ("Quantity", ""))


def default_tag(pln: Any) -> str:
    """Return the default snapshot name, ``"<radiation_mode>_<n>fx"`` (Qt)."""
    try:
        return f"{pln.radiation_mode}_{pln.num_of_fractions}fx"
    except AttributeError:
        return "result"


def sanitize_tag(tag: str) -> str:
    """Snapshot name as the Qt workflow stores it; rejects empty and reserved names."""
    clean = re.sub(r"\W+", "_", str(tag).strip()).strip("_")
    if not clean:
        raise ValueError("A snapshot needs a name")
    if clean in RESERVED_TAGS:
        raise ValueError(
            f"'{clean}' names per-{clean} results and cannot be used as a snapshot name."
        )
    return clean


def is_tagged(key: str, tags: list[str]) -> bool:
    """Whether *key* is a snapshot key of one of *tags*."""
    return any(key.endswith(f"_{tag}") for tag in tags)


def prune_tags(result: Any, tags: list[str]) -> list[str]:
    """Drop tags whose keys no longer exist in the result (Qt ``_prune_saved_tags``)."""
    keys = list(result) if isinstance(result, Mapping) else []
    return [tag for tag in tags if any(key.endswith(f"_{tag}") for key in keys)]


def snapshot(result: Mapping[str, Any], tag: str, tags: list[str]) -> dict[str, Any]:
    """Copy every key that is not itself a snapshot to ``"<key>_<tag>"``."""
    new = dict(result)
    for key in list(new):
        if not is_tagged(key, tags):
            new[f"{key}_{tag}"] = new[key]
    return new


def result_catalog(result: Any, dij: Any, pln: Any, tags: list[str]) -> dict[str, Any]:
    """Return the ``results`` part of SessionState v1."""
    from pyRadPlan.quantities._display_names import group_result_keys  # noqa: PLC0415

    from ._views import default_quantity  # noqa: PLC0415

    keys = list(result_components(result))
    groups = []
    for group in group_result_keys(keys):
        meta_label, unit = _meta(group.base)
        groups.append(
            {
                "base": group.base,
                "tag": group.tag,
                "label": group.label,
                "meta_label": meta_label,
                "unit": unit,
                "components": [
                    {"key": key, "label": label, "kind": kind or "other"}
                    for (label, key), kind in zip(group.components, group.kinds)
                ],
            }
        )
    return {
        "groups": groups,
        "scenario_labels": scenario_labels(result, dij),
        "default": default_quantity(keys),
        "tags": list(tags),
        "default_tag": default_tag(pln),
    }


def carry_forward(previous: Mapping[str, Any], result: dict, tags: list[str]) -> dict:
    """Keep snapshots and imported doses of a previous result (Qt optimize)."""
    for key, value in previous.items():
        if is_tagged(key, tags) or key.startswith("import_"):
            result.setdefault(key, value)
    return result

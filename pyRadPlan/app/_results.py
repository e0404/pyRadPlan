"""Result catalog: component keys and quantity groups of the named results."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from ._workspace import DEFAULT_RESULT_NAME, IMPORTED_RESULT_NAME, unique_result_name

_CONTAINERS = ("quantities", "beam_quantities", "scenario_quantities")
_LEGACY_IMPORT_PREFIX = "import_"
_LEGACY_TAGS_KEY = "saved_result_tags"

#: Colorbar / save-dialog label and unit of each known result quantity; per-beam and
#: per-scenario keys resolve to their quantity's (see :func:`quantity_meta`).
QUANTITY_META: dict[str, tuple[str, str]] = {
    "physical_dose": ("Dose", "Gy"),
    "let": ("LET", "keV/μm"),
    "effect": ("Effect", ""),
    "rbe_x_dose": ("RBE-weighted Dose", "Gy (RBE)"),
    "alpha_dose": ("Alpha Dose", "Gy"),
    "sqrt_beta_dose": ("Sqrt(Beta) Dose", "Gy½"),
    "let_dose": ("LET·Dose", "Gy·keV/μm"),
}

#: Suffixes :func:`quantity_meta` strips, in this order: a per-beam list, a per-scenario
#: list, and a per-beam list of each scenario (``<quantity>_beam_scenario``).
QUANTITY_SUFFIXES: tuple[str, ...] = ("_beam", "_scenario", "_beam")


def quantity_meta(
    key: str,
    default: tuple[str, str] = ("Quantity", ""),
    suffixes: tuple[str, ...] = QUANTITY_SUFFIXES,
) -> tuple[str, str]:
    """Label and unit of a result key from :data:`QUANTITY_META`.

    The result name of a ``"<result>/<key>"`` key is ignored. Each of *suffixes* is
    stripped in turn while the key is no known quantity, so ``"physical_dose_beam"`` is
    labelled as ``"physical_dose"``. Pass ``suffixes=("_beam",)`` to resolve per-beam keys
    only.
    """
    from pyRadPlan.quantities._display_names import split_result_key  # noqa: PLC0415

    base = split_result_key(key)[1]
    for suffix in suffixes:
        if base not in QUANTITY_META and base.endswith(suffix):
            base = base.removesuffix(suffix)
    return QUANTITY_META.get(base, default)


def result_keys_meta(
    keys: Iterable[str], default: tuple[str, str] = ("Quantity", "")
) -> dict[str, tuple[str, str]]:
    """Label and unit of each result key, also of list entries and of other results.

    Unlike :func:`quantity_meta`, the keys are grouped as the result catalog groups them
    (``group_result_keys``), so ``"<result>/rbe_x_dose_beam 0"`` carries the label and
    unit of ``"rbe_x_dose"``. Keys of unknown quantities get *default*.
    """
    from pyRadPlan.quantities._display_names import group_result_keys  # noqa: PLC0415

    keys = list(keys)
    meta: dict[str, tuple[str, str]] = {}
    for group in group_result_keys(keys):
        group_meta = quantity_meta(group.base, default)
        for _, member in group.components:
            meta[member] = group_meta
    return {key: meta.get(key, quantity_meta(key, default)) for key in keys}


def _is_volume(value: Any) -> bool:
    get_dimension = getattr(value, "GetDimension", None)
    return callable(get_dimension) and get_dimension() == 3


def result_entries(results: Any) -> list[tuple[str, Any]]:
    """``(tag, result)`` pairs of *results*.

    Accepts the entries of :meth:`Workspace.result_entries` (the active result under the
    tag ``""``, every other result under its name), a single result (taken as the active
    one) or ``None``.
    """
    if results is None:
        return []
    if hasattr(results, "quantities"):
        return [("", results)]
    return list(results)


def full_quantities(results: Any) -> dict[str, Any]:
    """Return the whole-volume quantities (no beam or scenario lists) of all results by key."""
    from pyRadPlan.quantities._display_names import result_key  # noqa: PLC0415

    return {
        result_key(tag, name): distribution.values
        for tag, result in result_entries(results)
        for name, distribution in result.quantities.items()
        if _is_volume(distribution.values)
    }


def result_components(results: Any) -> dict[str, Any]:
    """Every displayable volume of all results by component key.

    The keys are ``"[<result>/]<quantity>[_beam|_scenario][ <i>]"``: the quantities of the
    active result keep their name, those of another result are prefixed with its name.
    Entry ``i`` of a per-beam or per-scenario list becomes ``"<key> <i>"``; scenarios that
    were not computed are skipped, keeping the positions of the others.
    """
    from pyRadPlan.quantities._display_names import result_key  # noqa: PLC0415

    components: dict[str, Any] = {}
    for tag, result in result_entries(results):
        for name, distribution in result.quantities.items():
            if _is_volume(distribution.values):
                components[result_key(tag, name)] = distribution.values
        for suffix, container in (
            ("_beam", result.beam_quantities),
            ("_scenario", result.scenario_quantities),
        ):
            for name, items in container.items():
                for i, item in enumerate(items):
                    if item is not None and _is_volume(item.values):
                        components[result_key(tag, f"{name}{suffix} {i}")] = item.values
    return components


def resolve_result_key(results: Any, key: str) -> Any:
    """Return the volume a component key names.

    Raises
    ------
    KeyError
        If the results have no such volume.
    """
    components = result_components(results)
    if key not in components:
        raise KeyError(f"No volume quantity {key!r} in the results")
    return components[key]


def scenario_labels(result: Any, dij: Any = None) -> list[str]:
    """Return the scenario labels of *result*, else those of the dose influence *dij*."""
    labels = getattr(result, "scenario_labels", None)
    if labels is not None:
        return [str(label) for label in labels]
    if hasattr(dij, "scenario_labels"):
        try:
            return [str(label) for label in dij.scenario_labels()]
        except Exception:  # noqa: BLE001 - labels are cosmetic
            return []
    return []


def default_tag(pln: Any) -> str:
    """Return the default snapshot name, ``"<radiation_mode>_<n>fx"``."""
    try:
        return f"{pln.radiation_mode}_{pln.num_of_fractions}fx"
    except AttributeError:
        return "result"


def sanitize_tag(tag: str) -> str:
    """Return *tag* as a result name; rejects names that are empty after sanitizing."""
    clean = re.sub(r"\W+", "_", str(tag).strip()).strip("_")
    if not clean:
        raise ValueError("A snapshot needs a name")
    return clean


def result_catalog(
    results: Any = None,
    dij: Any = None,
    pln: Any = None,
    names: Sequence[str] = (),
    active: str = DEFAULT_RESULT_NAME,
) -> dict[str, Any]:
    """Return the ``results`` part of the session state.

    Parameters
    ----------
    results
        The entries of :meth:`Workspace.result_entries`.
    dij
        The dose influence; it names the scenarios of an active result without labels.
    pln
        The plan, for the default snapshot name.
    names
        All result names, in workspace order.
    active
        Name of the active result.

    Returns
    -------
    dict
        ``groups``: one entry per quantity of each result (``tag`` is the result name,
        ``""`` for the active result) with its ``components`` and the ``scenario_labels``
        of its result; ``scenario_labels`` of the active result; the ``default`` key to
        show; the ``names`` of all results, the ``active`` one and the ``tags`` of the
        others; ``default_tag``, the proposed snapshot name.
    """
    from pyRadPlan.quantities._display_names import group_result_keys  # noqa: PLC0415

    from ._views import default_quantity  # noqa: PLC0415

    entries = result_entries(results)
    labels = {tag: scenario_labels(result, None if tag else dij) for tag, result in entries}
    keys = list(result_components(entries))
    groups = []
    for group in group_result_keys(keys):
        meta_label, unit = quantity_meta(group.base)
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
                "scenario_labels": labels.get(group.tag, []),
            }
        )
    return {
        "groups": groups,
        "scenario_labels": labels[""] if "" in labels else scenario_labels(None, dij),
        "default": default_quantity(keys),
        "names": list(names),
        "active": active,
        "tags": [name for name in names if name != active],
        "default_tag": default_tag(pln),
    }


def is_legacy_result(data: Mapping[str, Any]) -> bool:
    """Whether the ``result`` of loaded *data* is a result dict of a session up to 0.9."""
    result = data.get("result")
    if not isinstance(result, Mapping):
        return False
    return bool(data.get(_LEGACY_TAGS_KEY)) or any(
        isinstance(key, str) and key.startswith(_LEGACY_IMPORT_PREFIX) for key in result
    )


def _split_legacy_result(result: Mapping[str, Any], tags: Iterable[str]) -> dict[str, Any]:
    """Split a result that holds snapshots and imported doses under mangled names.

    Sessions saved up to version 0.9 keep everything in one dict: a snapshot as a copy of
    every entry under ``"<key>_<tag>"`` (also the weights and the scenario labels) and
    imported doses as ``"import_<name>"``.
    """
    tags = sorted({tag for tag in tags if isinstance(tag, str) and tag}, key=len, reverse=True)
    kept: dict[str, Any] = {}
    split: dict[str, dict[str, Any]] = {}
    for key, value in result.items():
        tag = next((t for t in tags if key.endswith(f"_{t}") and len(key) > len(t) + 1), None)
        if tag is not None:
            split.setdefault(tag, {})[key[: -len(tag) - 1]] = value
        elif key.startswith(_LEGACY_IMPORT_PREFIX):
            name = unique_result_name(key[len(_LEGACY_IMPORT_PREFIX) :], [*split, *tags])
            split[name] = {"physical_dose": value}
        else:
            kept[key] = value

    from pyRadPlan.result import validate_result  # noqa: PLC0415

    results: dict[str, Any] = {}
    if any(key != "scenario_labels" for key in kept):
        results[DEFAULT_RESULT_NAME] = validate_result(kept)
    for name, part in split.items():
        results[unique_result_name(name, results)] = validate_result(part)
    return results


def loaded_results(data: Mapping[str, Any]) -> tuple[dict[str, Any], Optional[str]]:
    """Results of a loaded file as ``(results by name, name of the active one)``.

    A session file brings its named results. Any other plan result becomes the active one
    (named :data:`DEFAULT_RESULT_NAME`); a dose without a plan result is kept as the result
    :data:`IMPORTED_RESULT_NAME`. A session saved up to version 0.9, which holds snapshots
    and imported doses inside one result, is split into separate results.
    """
    from pyRadPlan.result import validate_result  # noqa: PLC0415

    results = data.get("results")
    if isinstance(results, Mapping) and results:
        return dict(results), data.get("active_result")
    result = data.get("result")
    if result is not None:
        if is_legacy_result(data):
            return _split_legacy_result(result, data.get(_LEGACY_TAGS_KEY) or ()), None
        return {DEFAULT_RESULT_NAME: validate_result(result)}, None
    if data.get("dose") is not None:
        return {IMPORTED_RESULT_NAME: {"physical_dose": data["dose"]}}, None
    return {}, None

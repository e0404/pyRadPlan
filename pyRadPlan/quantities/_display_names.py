"""Display names of quantities, result keys, biological models and robustness operators.

Moved out of ``pyRadPlan.gui.widgets._display_names`` so that GUI-independent frontends
(``pyRadPlan.app``) can use them; the GUI module re-exports every name defined here.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from pyRadPlan.optimization.robustness import get_robustness_operator_class
from pyRadPlan.quantities import get_available_quantities

#: Component label of the total (all beams) entry of a result group.
FULL_COMPONENT = "full"


#: Flattened list entries of the viewer (``"<key> <i>"``).
_INDEXED_KEY = re.compile(r"^(?P<stem>.+) (?P<index>\d+)$")


_BEAM_SUFFIX = "_beam"


_SCENARIO_SUFFIX = "_scenario"


#: Result entry naming the scenarios of the result's ``"<key>_scenario"`` lists.
SCENARIO_LABELS_KEY = "scenario_labels"


#: Component kinds of a :class:`ResultGroup` besides :data:`FULL_COMPONENT`.
BEAM_COMPONENT = "beam"


SCENARIO_COMPONENT = "scenario"


#: Component suffixes of result keys (``"<identifier><suffix>"``) and their component kind.
_COMPONENT_SUFFIXES = {_BEAM_SUFFIX: BEAM_COMPONENT, _SCENARIO_SUFFIX: SCENARIO_COMPONENT}


def _split_component_suffix(stem: str) -> tuple[str, str]:
    """Split ``"<base>_beam"`` / ``"<base>_scenario"`` into base and kind (``""`` if none)."""
    for suffix, kind in _COMPONENT_SUFFIXES.items():
        if stem.endswith(suffix) and len(stem) > len(suffix):
            return stem[: -len(suffix)], kind
    return stem, ""


def quantity_display_name(key: str) -> str:
    """Return the display name of a quantity identifier or result key.

    Registered quantity identifiers map to :attr:`RTQuantity.name`, including
    per-beam and per-scenario result keys (``"<identifier>_beam"``,
    ``"<identifier>_scenario"`` and their flattened ``"<identifier>_beam <i>"``
    entries).  Any other key (``alpha``, imported doses, matRad names) is
    returned unchanged; use :func:`group_result_keys` to also resolve snapshot
    tags.
    """
    stem, sep, index = key.partition(" ")
    base, kind = _split_component_suffix(stem)
    quantity = get_available_quantities().get(base)
    if quantity is None:
        return key
    name = f"{quantity.name} {kind}" if kind else quantity.name
    return f"{name}{sep}{index}"


def bio_model_display_name(model_cls: Any) -> str:
    """Return the display name of a biological model class (or instance)."""
    return getattr(model_cls, "name", None) or model_cls.model


def robustness_display_name(name: str) -> str:
    """Return the label of a robustness operator name; unknown names are returned unchanged."""
    try:
        operator_cls = get_robustness_operator_class(name)
    except (KeyError, ValueError):
        return name
    return operator_cls.label or operator_cls.name


@dataclass(frozen=True)
class ResultGroup:
    """Result entries belonging to one quantity of one (saved) result.

    Attributes
    ----------
    base:
        Quantity identifier or, for unregistered entries, the key stem.
    tag:
        Snapshot tag of a saved result; empty for the current result.
    label:
        Display label, e.g. ``"RBExDose"`` or ``"dose (protons_30fx)"``.
    components:
        ``(label, key)`` pairs: :data:`FULL_COMPONENT` for the total, then
        ``"beam <i>"`` for each per-beam entry and ``"scenario <i>"`` for each
        per-scenario entry (``<i>`` is the flat scenario index).
    kinds:
        Component kind per entry of :attr:`components`: :data:`FULL_COMPONENT`,
        :data:`BEAM_COMPONENT`, :data:`SCENARIO_COMPONENT` or ``""`` for other
        indexed entries.  Empty for groups built without kinds.
    """

    base: str
    tag: str
    label: str
    components: tuple[tuple[str, str], ...]
    kinds: tuple[str, ...] = ()

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for _, key in self.components)

    def keys_of(self, kind: str) -> tuple[str, ...]:
        """Keys of the components of *kind* (e.g. :data:`SCENARIO_COMPONENT`)."""
        return tuple(key for (_, key), k in zip(self.components, self.kinds) if k == kind)

    @property
    def identifier(self) -> str:
        """Key of the total entry, or the group stem when there is none."""
        for label, key in self.components:
            if label == FULL_COMPONENT:
                return key
        return f"{self.base}_{self.tag}" if self.tag else self.base


def _split_index(key: str) -> tuple[str, int | None]:
    match = _INDEXED_KEY.match(key)
    if match is None:
        return key, None
    return match["stem"], int(match["index"])


def _snapshot_tags(stems: set[str]) -> list[str]:
    """Deduce snapshot tags (``"<key>_<tag>"``) from the key suffixes.

    A candidate is the remainder after a registered quantity identifier (and
    an optional ``_beam`` / ``_scenario``); it counts as a tag only when at
    least two keys carry it, which tells a saved result (every key is copied)
    apart from a single suffixed entry such as ``physical_dose_var``.
    """
    identifiers = sorted(get_available_quantities(), key=len, reverse=True)
    canonical = {
        stem
        for name in identifiers
        for stem in (name, *(name + suffix for suffix in _COMPONENT_SUFFIXES))
    }
    # A tag such as "dose" must not count the live physical_dose/rbe_x_dose
    # entries as evidence of a snapshot, or rename those entries later.
    stems = stems - canonical
    candidates: set[str] = set()
    for stem in stems:
        for identifier in identifiers:
            if stem == identifier:
                break
            if stem.startswith(identifier + "_"):
                rest = stem[len(identifier) + 1 :]
                for kind in _COMPONENT_SUFFIXES.values():
                    if rest.startswith(kind + "_"):
                        rest = rest[len(kind) + 1 :]
                        break
                if rest and rest not in _COMPONENT_SUFFIXES.values():
                    candidates.add(rest)
                break
    counts = Counter(tag for tag in candidates for stem in stems if stem.endswith("_" + tag))
    return sorted((tag for tag, n in counts.items() if n >= 2), key=len, reverse=True)


def group_result_keys(keys: Iterable[str]) -> list[ResultGroup]:
    """Group result keys by quantity and saved result, deduced from their suffixes.

    ``"<base>[_beam|_scenario][_<tag>][ <i>]"`` keys are grouped by
    ``(base, tag)``: the per-beam and per-scenario entries join the total of
    their quantity (beams first, then scenarios), and snapshot tags (see the
    workflow's *Save Result*) are split off.  Groups keep the order of first
    appearance.
    """
    keys = list(keys)
    tags = _snapshot_tags({_split_index(key)[0] for key in keys})
    quantities = get_available_quantities()
    ranks = {FULL_COMPONENT: 0, BEAM_COMPONENT: 1, "": 1, SCENARIO_COMPONENT: 2}

    entries: dict[tuple[str, str], list[tuple[tuple[int, int], str, str, str]]] = {}
    for key in keys:
        stem, index = _split_index(key)
        tag = ""
        if _split_component_suffix(stem)[0] not in quantities:
            tag = next(
                (t for t in tags if stem.endswith("_" + t) and len(stem) > len(t) + 1),
                "",
            )
        if tag:
            stem = stem[: -len(tag) - 1]
        stem, kind = _split_component_suffix(stem)
        if kind:
            label = kind if index is None else f"{kind} {index}"
        elif index is not None:
            label = f"[{index}]"
        else:
            label = kind = FULL_COMPONENT
        order = (ranks[kind], -1 if index is None else index)
        entries.setdefault((stem, tag), []).append((order, label, key, kind))

    groups = []
    for (base, tag), items in entries.items():
        label = quantity_display_name(base)
        if tag:
            label = f"{label} ({tag})"
        items.sort(key=lambda e: e[0])
        components = tuple((lbl, key) for _, lbl, key, _ in items)
        kinds = tuple(kind for *_, kind in items)
        groups.append(ResultGroup(base, tag, label, components, kinds))
    return groups


def result_key_labels(keys: Iterable[str]) -> dict[str, str]:
    """Map each result key to a display label, e.g. ``"dose (tag), beam 0"``."""
    labels = {}
    for group in group_result_keys(keys):
        for component, key in group.components:
            labels[key] = (
                group.label if component == FULL_COMPONENT else f"{group.label}, {component}"
            )
    return labels

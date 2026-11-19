"""Structures and tissue editing (protocol v1: ``cst.*``, ``tissue.*``).

Follows the Qt tissue dialog (``gui/widgets/plan/_tissue_dialog.py``) and the VOI metadata
dialog (``gui/widgets/result/vois_widget.py``); the helpers live in
:mod:`pyRadPlan.biology._editing`. Drafts are plain dicts the frontend edits; nothing is
written to the workspace before :func:`apply_tissues` / :func:`update_structure`.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from ._forms import jsonable
from ._planform import _machine_of, draft_context, issue_dict


def _value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    return jsonable(value)


def draft_of(tissue: Any) -> dict[str, Any]:
    """``TissueDraft`` of tissue properties."""
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        reference_radiation_fields,
        tissue_edit_fields,
        tissue_model_label,
    )

    cls = type(tissue)
    names = tissue_edit_fields(cls) + reference_radiation_fields(cls)
    return {
        "model": tissue_model_label(tissue),
        "values": {name: _value(getattr(tissue, name, float("nan"))) for name in names},
        "sources": dict(getattr(tissue, "reference_sources", {}) or {}),
    }


def tissue_of(draft: Any, current: Any = None) -> Any:
    """Tissue properties of a draft, keeping the cell line of *current* while it matches.

    Tissue properties (from an in-process frontend) are returned as they are.
    """
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        build_tissue_properties,
        model_label_name,
        tissue_class,
    )
    from pyRadPlan.biology.tissues import REFERENCE_FIELDS, TissueProperties  # noqa: PLC0415

    if isinstance(draft, TissueProperties):
        return draft
    label = draft.get("model") or "Generic"
    cls = tissue_class(label)
    values = {
        name: (float("nan") if value is None else value)
        for name, value in dict(draft.get("values") or {}).items()
    }
    sources = {
        name: source
        for name, source in dict(draft.get("sources") or {}).items()
        if name in REFERENCE_FIELDS
    }
    return build_tissue_properties(
        cls, values, current, sources or None, model_label_name(label, current)
    )


def _derive(draft: dict[str, Any], force: bool) -> None:
    """Derive α_x / β_x of an MKM draft from its model parameters (Qt live derivation)."""
    from pyRadPlan.biology._editing import derives_reference_params, tissue_class  # noqa: PLC0415
    from pyRadPlan.biology.tissues import (  # noqa: PLC0415
        DEFAULT_ALPHA_X,
        DEFAULT_BETA_X,
        FIXED_REFERENCE_SOURCES,
        REFERENCE_FIELDS,
    )

    cls = tissue_class(draft["model"])
    if not derives_reference_params(cls):
        return
    sources = draft.setdefault("sources", {})
    names = [
        n
        for n in REFERENCE_FIELDS
        if force or sources.get(n, "explicit") not in FIXED_REFERENCE_SOURCES
    ]
    if not names:
        return
    values = {k: (float("nan") if v is None else v) for k, v in dict(draft["values"]).items()}
    values["model"] = draft["model"]
    derived = cls.derive_reference(values)
    defaults = {"alpha_x": DEFAULT_ALPHA_X, "beta_x": DEFAULT_BETA_X}
    for name in names:
        if name in derived:
            draft["values"][name] = float(derived[name])
            sources[name] = "estimated"
        elif sources.get(name) == "estimated":
            draft["values"][name] = defaults[name]
            sources[name] = "default"


def _context(
    pln: Any,
    bio_model: Any = None,
    machine: Any = None,
    engine: Any = None,
    draft: Optional[Mapping[str, Any]] = None,
):
    from pyRadPlan.biology._editing import bio_model_quietly, dose_engine_quietly  # noqa: PLC0415

    if draft:
        # The unapplied plan form, resolved as plan.tissue_check does (Qt evaluates the dialog
        # against the plan widget's current selection).
        resolved = draft_context(pln, draft)
        if resolved is not None:
            mode, machine_d, engine_d, spec_d = resolved
            return (
                _machine_of(mode, machine if machine is not None else machine_d),
                bio_model_quietly(bio_model if bio_model is not None else spec_d, mode),
                dose_engine_quietly(mode, engine if engine is not None else engine_d),
            )
    mode = getattr(pln, "radiation_mode", "photons")
    machine = machine if machine is not None else getattr(pln, "machine", None)
    prop = getattr(pln, "prop_dose_calc", None)
    engine_name = (
        engine if engine is not None else (prop.get("engine") if isinstance(prop, dict) else None)
    )
    spec = bio_model if bio_model is not None else getattr(pln, "bio_model", None)
    return (
        _machine_of(mode, machine),
        bio_model_quietly(spec, mode),
        dose_engine_quietly(mode, engine_name),
    )


def tissue_schema(pln: Any, cst: Any, **context: Any) -> dict[str, Any]:
    """``tissue.schema``: models with their fields, reference radiations, picks."""
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        TISSUE_MODELS,
        available_tissue_classes,
        cell_line_presets,
        param_field,
        param_label,
        param_unit,
        picker_title,
        ranked_tissue_picks,
        reference_radiation_fields,
        reference_radiation_label,
        target_provenance,
        tissue_edit_fields,
    )
    from pyRadPlan.biology.tissues import registered_reference_radiations  # noqa: PLC0415

    machine, model, engine = _context(pln, **context)
    models = []
    for label, cls in TISSUE_MODELS:
        fields = []
        for name in tissue_edit_fields(cls) + reference_radiation_fields(cls):
            info = cls.model_fields[name]
            default = info.get_default(call_default_factory=True)
            entry: dict[str, Any] = {
                "name": name,
                "label": param_label(name),
                "default": _value(default),
                "description": param_field(name).description or "",
            }
            if param_unit(name):
                entry["unit"] = param_unit(name)
            if name == "reference_radiation":
                entry["type"] = "choice"
                entry["choices"] = [
                    {"value": r.name, "label": reference_radiation_label(r)}
                    for r in registered_reference_radiations()
                ]
            elif isinstance(default, str):
                from pyRadPlan.biology._editing import param_choices  # noqa: PLC0415

                entry["type"] = "choice"
                entry["choices"] = [
                    {"value": str(c), "label": str(c)} for c in param_choices(name)
                ]
            else:
                entry["type"] = "float"
                entry["nullable"] = isinstance(default, float) and math.isnan(default)
            fields.append(entry)
        models.append({"label": label, "fields": fields})
    classes = available_tissue_classes(machine, model, engine)
    presets = cell_line_presets()
    provenance = target_provenance(machine, model, engine)
    picks = []
    for pick in ranked_tissue_picks(classes, presets, provenance):
        draft = draft_of(pick.tissue)
        picks.append(
            {
                "id": f"{pick.kind}:{pick.index}",
                "kind": "base_data" if pick.kind == "class" else "preset",
                "label": pick.label,
                "group": "Base data" if pick.kind == "class" else pick.provenance,
                "exact": bool(pick.exact),
                "preferred": bool(pick.preferred),
                "model": draft["model"],
                "values": draft["values"],
                "description": getattr(pick, "tooltip", "") or "",
            }
        )
    return {
        "models": models,
        "reference_radiations": [
            {"value": r.name, "label": reference_radiation_label(r)}
            for r in registered_reference_radiations()
        ],
        "picks": picks,
        "option_label": picker_title(machine, model, engine),
    }


def _comparison_dict(comparison: Any) -> Optional[dict]:
    if comparison is None:
        return None
    from pyRadPlan.biology._editing import class_label  # noqa: PLC0415

    tissue_class = getattr(comparison, "tissue_class", None)
    return {
        "closest_class": class_label(tissue_class) if tissue_class is not None else None,
        "matched": bool(getattr(comparison, "matched", False)),
        "provenance_fix": getattr(comparison, "fix_provenance", None),
        "parameters": [
            {
                "name": p.name,
                "declared": _value(p.declared),
                "value": _value(p.value),
                "matches": bool(p.matches),
            }
            for p in getattr(comparison, "parameters", ()) or ()
        ],
    }


def evaluate_tissues(
    pln: Any,
    cst: Any,
    rows: Sequence[Mapping[str, Any]],
    op: str = "none",
    pick: Optional[str] = None,
    targets: Optional[Sequence[str]] = None,
    edited: Optional[Mapping[str, Any]] = None,
    **context: Any,
) -> dict[str, Any]:
    """``tissue.evaluate``: apply a dialog operation to drafts and check them."""
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        adopt_preset,
        apply_base_data_class,
        available_tissue_classes,
        cell_line_presets,
        compare_tissues_quietly,
        derives_reference_params,
        provenance_fixes,
        relabel_issues,
        target_provenance,
        tissue_class,
        tissue_issues_quietly,
    )
    from pyRadPlan.biology.models import apply_provenance_fixes  # noqa: PLC0415
    from pyRadPlan.biology.tissues import REFERENCE_FIELDS  # noqa: PLC0415

    machine, model, engine = _context(pln, **context)
    names = [voi.name for voi in getattr(cst, "vois", [])]
    if len(set(names)) != len(names):
        raise ValueError("tissue.evaluate needs unique structure names")
    vois = {voi.name: voi for voi in getattr(cst, "vois", [])}
    drafts: dict[str, dict] = {}
    for row in rows:
        name = row["structure"]
        if name not in vois:
            raise KeyError(f"Unknown structure {name!r}")
        tissue = dict(row["tissue"])
        drafts[name] = {
            "model": tissue.get("model", "Generic"),
            "values": dict(tissue.get("values") or {}),
            "sources": dict(tissue.get("sources") or {}),
        }
    target_names = list(targets) if targets else list(drafts)
    notes: list[str] = []

    if edited and edited.get("structure") in drafts:
        draft = drafts[edited["structure"]]
        field = edited.get("field")
        if field in REFERENCE_FIELDS:
            draft["sources"][field] = "explicit"
        else:
            _derive(draft, force=False)
    if op == "derive":
        derivable = [n for n in target_names if derives_reference_params(
            tissue_class(drafts[n]["model"]))]  # fmt: skip
        for name in derivable:
            _derive(drafts[name], force=True)
        notes.append(
            "Nothing to derive from"
            if not derivable
            else f"Derived the reference parameters of {len(derivable)} structure(s)"
        )

    def tissues() -> dict[str, Any]:
        return {
            name: tissue_of(drafts[name], voi.tissue_properties)
            if name in drafts
            else voi.tissue_properties
            for name, voi in vois.items()
        }

    if op == "apply_pick" and pick:
        kind, _, index = pick.partition(":")
        classes = available_tissue_classes(machine, model, engine)
        presets = cell_line_presets()
        for name in target_names:
            if kind == "class":
                tissue, _ = apply_base_data_class(tissues()[name], classes[int(index)])
                note = "Applied the base data class"
            else:
                tissue, note = adopt_preset(
                    presets[int(index)][1], target_provenance(machine, model, engine)
                )
            drafts[name] = draft_of(tissue)
            notes.append(note)
    current = tissues()
    comparisons = compare_tissues_quietly(machine, model, list(current.items()), engine)
    by_name = dict(zip(current, comparisons)) if len(comparisons) == len(current) else {}
    if op == "match":
        matched = 0
        for name in target_names:
            comparison = by_name.get(name)
            if comparison is None or comparison.tissue_class is None:
                continue
            tissue, _ = apply_base_data_class(current[name], comparison.tissue_class)
            drafts[name] = draft_of(tissue)
            matched += 1
        notes.append(f"Matched {matched} structure(s) to their closest base data class")
    elif op == "relabel":
        issues = relabel_issues(by_name.get(name) for name in target_names)
        fixed = apply_provenance_fixes(dict(current), issues)
        changed = [n for n in target_names if fixed[n] is not current[n]]
        for name in changed:
            drafts[name] = draft_of(fixed[name])
        notes.append(f"Relabelled {len(changed)} structure(s) with the base data's model")
    if op in ("match", "relabel", "apply_pick"):
        current = tissues()
        comparisons = compare_tissues_quietly(machine, model, list(current.items()), engine)
        by_name = dict(zip(current, comparisons)) if len(comparisons) == len(current) else {}

    issues_by_name: dict[str, list] = {}
    for issue in tissue_issues_quietly(machine, model, dict(current), engine):
        issues_by_name.setdefault(getattr(issue, "structure", None), []).append(issue)
    relabel = provenance_fixes(relabel_issues(by_name.values()))
    return {
        "rows": [
            {
                "structure": name,
                "tissue": drafts[name],
                "issues": [issue_dict(i) for i in issues_by_name.get(name, [])],
                "comparison": _comparison_dict(by_name.get(name)),
            }
            for name in drafts
        ],
        "notes": notes,
        "can_match": any(getattr(c, "tissue_class", None) is not None for c in by_name.values()),
        "can_relabel": list(relabel),
        "can_derive": any(
            derives_reference_params(tissue_class(d["model"])) for d in drafts.values()
        ),
    }


def recreate_voi(voi: Any, **changes: Any) -> Any:
    """Return a copy of a VOI with *changes* (:func:`pyRadPlan.cst.recreate_voi`)."""
    from pyRadPlan.cst import recreate_voi as _recreate_voi  # noqa: PLC0415

    return _recreate_voi(voi, **changes)


def replace_tissues(cst: Any, tissues: Mapping[int, Any]) -> None:
    """Give the structures at the positions of *tissues* their new tissue properties.

    The structures are recreated (:func:`recreate_voi`) and assigned to *cst* at once.
    """
    vois = list(cst.vois)
    for index, tissue in tissues.items():
        vois[index] = recreate_voi(vois[index], tissue_properties=tissue)
    cst.vois = vois


def check_errors(pln: Any, cst: Any, replaced: Mapping[int, Any], **context: Any) -> list:
    """Errors of the structures' tissues after replacing those at the positions *replaced*."""
    from pyRadPlan.biology._editing import tissue_issues_quietly  # noqa: PLC0415

    machine, model, engine = _context(pln, **context)
    names = [voi.name for voi in cst.vois]
    tissues = {
        (f"{voi.name} (#{i})" if names.count(voi.name) > 1 else voi.name): replaced.get(
            i, voi.tissue_properties
        )
        for i, voi in enumerate(cst.vois)
    }
    issues = tissue_issues_quietly(machine, model, tissues, engine)
    return [issue for issue in issues if issue.is_error]


def structure_detail(voi: Any, index: int, result: Any = None) -> dict[str, Any]:
    """``StructureDetail`` of a VOI (objectives as ``ObjectiveData``; ``num_voxels`` 0)."""
    from pyRadPlan.biology._editing import tissue_summary  # noqa: PLC0415

    from ._objectives import objective_data  # noqa: PLC0415

    try:
        center = voi.center_of_mass
    except Exception:  # noqa: BLE001 - an empty or unreadable mask has no centre
        center = None
    objectives = [
        objective_data(objective, i, result)
        for i, objective in enumerate(getattr(voi, "objectives", None) or [])
    ]
    return {
        "name": voi.name,
        "index": index,
        "type": voi.voi_type,
        "color": [int(c) for c in voi.color],
        "visible": bool(voi.visible),
        "overlap_priority": int(voi.overlap_priority),
        "num_voxels": 0,
        "num_scenarios": int(voi.num_of_scenarios),
        "center_of_mass": None if center is None else [float(c) for c in center],
        "alpha_x": float(voi.alpha_x),
        "beta_x": float(voi.beta_x),
        "tissue": draft_of(voi.tissue_properties),
        "tissue_summary": tissue_summary(voi.tissue_properties),
        "objectives": objectives,
    }

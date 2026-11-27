"""What the model sees of the tool results.

The state keeps the full dicts (GUI, plan selection, tests); the tools send these shorter
views, one line per row instead of repeated JSON keys.
"""

import copy
from typing import Any, Optional, Sequence


def format_mix(plans: Sequence[int], weights: Sequence[float]) -> str:
    """'75 % plan 1 + 25 % plan 3' for a blend; plans without weight are left out."""
    return " + ".join(f"{100 * w:.0f} % plan {p}" for p, w in zip(plans, weights) if w)


def short_goal(label: str) -> str:
    """Goal label without its limit: 'PAROTID_LT mean <= 26' -> 'PAROTID_LT mean'."""
    return label.rsplit(" ", 2)[0]


def goal_line(r: dict) -> str:
    tag = "hard" if r["hard"] else "soft"
    verdict = "passed" if r["passed"] else "FAILED"
    return (
        f"{r['goal']} [{tag}]: {r['value']:g}, margin {r['margin']:+g} "
        f"({100 * r['rel_margin']:+.1f} %), {verdict}"
    )


def model_evaluation(ev: Optional[dict], plan: Optional[int] = None) -> Optional[dict]:
    if not ev:
        return ev
    out = {"plan": plan} if plan is not None else {}
    out.update(ev, goals=[goal_line(r) for r in ev["goals"]])
    out.pop("unresolved_structures", None)
    if out.get("unresolved_goals"):
        out["note"] = (
            "Unresolved goals are not evaluated: map their structures. Structures "
            "listed under empty_structures have no voxels."
        )
    else:
        out.pop("unresolved_goals", None)
    return out


def model_comparison(cmp: dict) -> dict:
    def line(g: dict) -> str:
        a, b = ("-" if p is None else "passed" if p else "FAILED" for p in g["passed"])
        if "value" not in g:
            return f"{g['goal']}: {a} -> {b}"
        va, vb = g["value"]
        return f"{g['goal']}: {va:g} -> {vb:g} ({g['delta']:+g}), {a} -> {b}"

    return {**cmp, "goals": [line(g) for g in cmp["goals"]]}


def model_beam_influence(out: dict) -> dict:
    def line(b: dict) -> str:
        ratios = ", ".join(
            f"{name} {'-' if r is None else f'{r:g}'}"
            for name, r in b["rel_mean_dose_vs_target"].items()
        )
        crossing = ", ".join(f"{name} {p:g} %" for name, p in b.get("crossing_pct", {}).items())
        mark = " [current]" if b.get("current") else ""
        return (
            f"gantry {b['gantry']:g}, couch {b['couch']:g}{mark}: "
            f"clean {b.get('clean_pct', 0):g} %; crossing {crossing or 'none'}; "
            f"rel. mean dose {ratios}"
        )

    return {**out, "beams": [line(b) for b in out["beams"]]}


def model_pareto(out: dict) -> dict:
    plans = out["plans"]

    def line(e: dict) -> str:
        vals = ", ".join(f"{short_goal(g)} {v:g}" for g, v in e["values"].items())
        best = f"; best on {', '.join(map(short_goal, e['best_on']))}" if e.get("best_on") else ""
        return (
            f"{format_mix(plans, e['weights'])} (weights {e['weights']}): score {e['score']:g}, "
            f"{e['num_failed']} failed; {vals}{best}"
        )

    view = {k: v for k, v in out.items() if k not in ("axes", "front", "compromise")}
    if "axes" in out:
        view["axes"] = [
            f"{a['goal']}: front {a['best']:g} (best) to {a['worst']:g}; plans "
            + ", ".join(f"{p}: {v:g}" for p, v in a["plans"].items())
            for a in out["axes"]
        ]
    view["front"] = [line(e) for e in out["front"]]
    if "compromise" in out:
        c = out["compromise"]
        view["compromise"] = {
            "blend": line(c),
            "weakest_axes": [
                f"{w['goal']}: {w['pct_of_range_from_best']:g} % of the front's range from its best"
                for w in c["weakest_axes"]
            ],
        }
    return view


def model_optimizer_report(rep: dict) -> dict:
    return {
        **rep,
        "contributions": [
            f"{c['voi']} {c['objective']}: {c['value']:g} ({c['share_pct']:g} %)"
            for c in rep["contributions"]
        ],
    }


# Objective fields the agent leaves at their defaults: hidden from the schema it is sent,
# and reset by with_hidden_defaults in case the model sends them anyway (validation runs
# on the full objective models)
_HIDDEN_OBJECTIVE_FIELDS = ("quantity", "robustness")


def with_hidden_defaults(objective: Any) -> Any:
    """Return *objective* with the fields hidden from the model at their defaults."""
    fields = type(objective).model_fields
    defaults = {
        name: fields[name].get_default(call_default_factory=True)
        for name in _HIDDEN_OBJECTIVE_FIELDS
        if name in fields
    }
    changed = {k: v for k, v in defaults.items() if getattr(objective, k) != v}
    return objective.model_copy(update=changed) if changed else objective


def slim_schema(schema: dict) -> dict:
    """Shrink a tool schema; objective unions in it lose about half their size."""
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("discriminator", None)  # the const "name" of each variant says the same
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for d in schema.get("$defs", {}).values():
        d.pop("title", None)
        props = d.get("properties", {})
        if "penalty" not in props or "name" not in props:
            continue
        for name in _HIDDEN_OBJECTIVE_FIELDS:
            props.pop(name, None)
        props["name"].pop("default", None)
        d["required"] = ["name", *(r for r in d.get("required", []) if r != "name")]
        props["penalty"] = {**props["penalty"], "description": "Relative weight."}
        if d.get("description"):
            d["description"] = d["description"].split("\n", 1)[0]
    walk(schema)
    return schema

"""JSON-able plan summary and the plan built from it."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Optional

import numpy as np

#: Fields of a ``PlanSummary`` (see ``web/PROTOCOL.md``).
PLAN_SUMMARY_FIELDS: tuple[str, ...] = (
    "radiation_mode",
    "machine",
    "num_of_fractions",
    "gantry_angles",
    "couch_angles",
    "bixel_width",
    "iso_center",
    "dose_grid_resolution",
    "engine",
    "bio_model",
)

#: The plan a session applies once patient data is loaded, the Qt plan form's defaults
#: (total-course doses, which keep objectives meaningful for fraction-size dependent
#: biological models; the plan model's own default is per fraction).
DEFAULT_PLAN: dict[str, Any] = {
    "radiation_mode": "photons",
    "machine": "Generic",
    "num_of_fractions": 30,
    "gantry_angles": [0.0],
    "couch_angles": [0.0],
    "bixel_width": 5.0,
    "iso_center": None,
    "dose_grid_resolution": [3.0, 3.0, 3.0],
    "engine": None,
    "dose_convention": "total",
}


def _float_list(value: Any) -> list[float]:
    return [float(v) for v in np.atleast_1d(np.asarray(value, dtype=float)).ravel()]


def _machine_name(machine: Any) -> Optional[str]:
    if machine is None or isinstance(machine, str):
        return machine
    if isinstance(machine, Mapping):
        return machine.get("name") or machine.get("meta", {}).get("machine")
    return getattr(machine, "name", None) or type(machine).__name__


def _dose_grid_resolution(prop_dose_calc: Mapping[str, Any]) -> Optional[list[float]]:
    dose_grid = prop_dose_calc.get("dose_grid")
    if isinstance(dose_grid, Mapping):
        resolution = dose_grid.get("resolution")
    else:
        resolution = getattr(dose_grid, "resolution", None)
    if isinstance(resolution, Mapping):
        try:
            return [float(resolution[axis]) for axis in ("x", "y", "z")]
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _bio_model_name(bio_model: Any) -> Optional[str]:
    if bio_model is None or isinstance(bio_model, str):
        return bio_model
    return getattr(bio_model, "model", None)


def plan_summary(pln: Any) -> Optional[dict[str, Any]]:
    """Return the ``PlanSummary`` of *pln* (``None`` without a plan).

    ``iso_center`` is ``None`` when the plan uses the target center of mass, a list of
    three coordinates for one isocenter and a list of lists for per-beam isocenters.
    """
    if pln is None:
        return None
    prop_stf = pln.prop_stf or {}
    prop_dose_calc = pln.prop_dose_calc or {}
    iso = prop_stf.get("iso_center")
    if iso is None or np.size(iso) == 0:
        iso_center = None
    else:
        iso_arr = np.atleast_2d(np.asarray(iso, dtype=float))
        unique = np.unique(iso_arr, axis=0)
        iso_center = unique[0].tolist() if len(unique) == 1 else iso_arr.tolist()
    gantry = prop_stf.get("gantry_angles")
    couch = prop_stf.get("couch_angles")
    bixel_width = prop_stf.get("bixel_width")
    engine = prop_dose_calc.get("engine") if isinstance(prop_dose_calc, Mapping) else None
    return {
        "radiation_mode": pln.radiation_mode,
        "machine": _machine_name(pln.machine),
        "num_of_fractions": int(pln.num_of_fractions),
        "gantry_angles": _float_list(gantry) if gantry is not None else [],
        "couch_angles": _float_list(couch) if couch is not None else [],
        "bixel_width": float(bixel_width) if bixel_width is not None else None,
        "iso_center": iso_center,
        "dose_grid_resolution": _dose_grid_resolution(prop_dose_calc),
        "engine": engine if isinstance(engine, str) else None,
        "bio_model": _bio_model_name(pln.bio_model),
    }


def plan_class(radiation_mode: str) -> type:
    """Return the plan class of a radiation mode (``PhotonPlan`` or ``IonPlan``)."""
    from pyRadPlan.plan import IonPlan, PhotonPlan  # noqa: PLC0415

    return PhotonPlan if radiation_mode == "photons" else IonPlan


_plan_class = plan_class


def _couch_angles(gantry: list[float], summary: Mapping, changes: Mapping) -> list[float]:
    given = changes.get("couch_angles")
    if given is None:
        previous = _float_list(summary["couch_angles"]) if summary.get("couch_angles") else []
        return previous if len(previous) == len(gantry) else [0.0] * len(gantry)
    couch = _float_list(given)
    if len(couch) == 1 and len(gantry) > 1:
        couch = couch * len(gantry)
    if len(couch) != len(gantry):
        raise ValueError(
            f"{len(couch)} couch angles for {len(gantry)} gantry angles; give one per beam"
        )
    return couch


def _prop_stf(prop_stf: dict, merged: Mapping, summary: Mapping, changes: Mapping) -> dict:
    gantry = _float_list(merged["gantry_angles"]) if merged.get("gantry_angles") else []
    prop_stf.update(gantry_angles=gantry, couch_angles=_couch_angles(gantry, summary, changes))
    if merged.get("bixel_width") is not None:
        prop_stf["bixel_width"] = float(merged["bixel_width"])
    iso_center = merged.get("iso_center")
    if iso_center is None or np.size(iso_center) == 0:
        prop_stf.pop("iso_center", None)
    else:
        iso = np.asarray(iso_center, dtype=float)
        if iso.size % 3 != 0:
            raise ValueError("iso_center requires three coordinates (x, y, z) per isocenter")
        prop_stf["iso_center"] = iso.tolist()
    return prop_stf


def _prop_dose_calc(prop_dose_calc: dict, merged: Mapping) -> dict:
    if merged.get("engine"):
        if prop_dose_calc.get("engine") != merged["engine"]:
            prop_dose_calc = {"engine": merged["engine"]}
    else:
        prop_dose_calc.pop("engine", None)
    resolution = merged.get("dose_grid_resolution")
    if resolution is None:
        prop_dose_calc.pop("dose_grid", None)
    else:
        values = _float_list(resolution)
        if len(values) != 3:
            raise ValueError("dose_grid_resolution requires three values (x, y, z) in mm")
        prop_dose_calc["dose_grid"] = {"resolution": dict(zip(("x", "y", "z"), values))}
    return prop_dose_calc


def _copy_field(name: str, value: Any) -> Any:
    solver = value.get("solver") if name == "prop_opt" and isinstance(value, Mapping) else None
    if solver is not None and not isinstance(solver, (str, Mapping)):
        # A live solver instance (and its thread state) is kept, not copied.
        return deepcopy(value, {id(solver): solver})
    return deepcopy(value)


def build_plan(
    current: Any, changes: Mapping[str, Any], overrides: Optional[Mapping[str, Any]] = None
) -> Any:
    """Build a validated plan from *current* (may be ``None``) and a partial summary.

    Fields not named in *changes* keep their value; plan settings outside the summary
    (prescription, scenarios, ``prop_opt``, ...) are carried over as the Qt plan form
    does. Changing the radiation mode starts from that mode's defaults for the machine,
    the biological model and the per-mode properties. A gantry list without couch
    angles gets couch 0 for every beam; a single couch angle applies to every beam.

    Raises
    ------
    ValueError
        For unknown fields or values the plan model rejects.

    Notes
    -----
    *overrides* carries the v1 settings resolved by :mod:`pyRadPlan.app._planform`:
    ``engine_props`` (engine-specific ``prop_dose_calc`` entries, replacing the current
    ones), and ready plan fields ``prop_opt`` and ``scenarios``. ``dose_convention`` may
    be given in *changes*.
    """
    overrides = dict(overrides or {})
    unknown = set(changes) - set(PLAN_SUMMARY_FIELDS) - {"dose_convention"}
    if unknown:
        raise ValueError(f"Unknown plan field(s): {sorted(unknown)}")

    summary = plan_summary(current) or dict(DEFAULT_PLAN)
    radiation_mode = changes.get("radiation_mode", summary["radiation_mode"])
    same_mode = current is not None and current.radiation_mode == radiation_mode
    merged = {**summary, **changes}
    if not same_mode and current is not None:
        for field, default in (("machine", "Generic"), ("bio_model", None), ("engine", None)):
            if field not in changes:
                merged[field] = default

    prop_stf = _prop_stf(deepcopy(current.prop_stf) if same_mode else {}, merged, summary, changes)
    prop_dose_calc = _prop_dose_calc(
        deepcopy(dict(current.prop_dose_calc or {})) if same_mode else {}, merged
    )
    if "engine_props" in overrides:
        kept = {k: v for k, v in prop_dose_calc.items() if k in ("engine", "dose_grid")}
        prop_dose_calc = {**dict(overrides["engine_props"]), **kept}

    plan_data: dict[str, Any] = {}
    if current is not None:
        plan_data = {
            name: _copy_field(name, getattr(current, name))
            for name in type(current).model_fields
            if name not in ("machine", "radiation_mode")
        }
        if not same_mode:
            for name in ("bio_model", "prop_seq", "prop_opt", "scenarios"):
                plan_data.pop(name, None)
    machine = merged.get("machine") or "Generic"
    if same_mode and "machine" not in changes and not isinstance(current.machine, str):
        machine = current.machine
    plan_data.update(
        radiation_mode=radiation_mode,
        machine=machine,
        num_of_fractions=int(merged["num_of_fractions"]),
        prop_stf=prop_stf,
        prop_dose_calc=prop_dose_calc,
    )
    if "bio_model" in changes or not same_mode:
        plan_data["bio_model"] = merged.get("bio_model")
    if changes.get("dose_convention") is not None:
        plan_data["dose_convention"] = changes["dose_convention"]
    for key in ("prop_opt", "scenarios"):
        if key in overrides:
            plan_data[key] = overrides[key]
    return _plan_class(radiation_mode)(**plan_data)

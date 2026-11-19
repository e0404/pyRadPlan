"""The Qt plan form applies through the session the plan it built itself before.

Drives :class:`PlanWidget` through form states and compares the plan *Apply* writes
(``PlanWidget._form_changes`` -> ``changes_from_form`` -> ``Session.set_plan``) with the
plan of :func:`reference_plan`, a frozen copy of the builder the form used before it
applied through the session (``PlanWidget._build_plan``).
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pytest

pytest.importorskip("PySide6")

from pyRadPlan.gui.widgets._config_form import ConfigFormWidget
from pyRadPlan.gui.widgets._display_names import current_identifier, select_identifier
from pyRadPlan.gui.widgets.plan import PlanWidget
from pyRadPlan.gui.widgets.plan._optimization_dialog import OptimizationConfigDialog
from pyRadPlan.gui.workspace import WorkspaceManager
from pyRadPlan.io import load_tg119
from pyRadPlan.optimization.solvers import get_available_solvers
from pyRadPlan.scenarios import get_scenario_model_class


skipif_no_ipopt = pytest.mark.skipif(
    "ipopt" not in get_available_solvers(), reason="IPOPT is not installed on this system."
)


@pytest.fixture(scope="module")
def tg119():
    return load_tg119()


# ----------------------------------------------------------------------
# Frozen reference: the plan form's own builder before the migration
# ----------------------------------------------------------------------


def _props_key(props: dict) -> str:
    import json

    return json.dumps(props, default=str, sort_keys=True)


def reference_plan(widget: PlanWidget) -> Any:
    """The plan ``PlanWidget._build_plan`` built from the form (kept as it was)."""
    from pyRadPlan.app._plan import plan_class
    from pyRadPlan.app._planform import scenario_overrides
    from pyRadPlan.gui.widgets._base import parse_number_list

    radiation_mode = widget._cmb_radiation.currentText()
    gantry, couch = widget._form_beam_angles()
    current = widget._ws.pln
    same_mode = current is not None and current.radiation_mode == radiation_mode
    prop_stf = deepcopy(current.prop_stf) if same_mode else {}
    prop_stf.update(
        gantry_angles=gantry, couch_angles=couch, bixel_width=float(widget._spn_bixel.value())
    )
    if not widget._chk_iso_auto.isChecked():
        iso = parse_number_list(widget._txt_iso.text())
        if len(iso) != 3:
            raise ValueError("iso center requires exactly three coordinates (x y z)")
        prop_stf["iso_center"] = iso
    elif widget._per_beam_iso_center is not None:
        if len(widget._per_beam_iso_center) != len(gantry):
            raise ValueError("per-beam iso centers do not match the beams")
        prop_stf["iso_center"] = deepcopy(widget._per_beam_iso_center)
    else:
        prop_stf.pop("iso_center", None)

    prop_dose_calc: dict = {}
    engine_name = current_identifier(widget._cmb_engine)
    if engine_name:
        prop_dose_calc["engine"] = engine_name
        prop_dose_calc.update(widget._engine_props.get(engine_name, {}))
    prop_dose_calc["dose_grid"] = {
        "resolution": {
            "x": float(widget._spn_res_x.value()),
            "y": float(widget._spn_res_y.value()),
            "z": float(widget._spn_res_z.value()),
        }
    }
    plan_data = (
        {
            name: deepcopy(getattr(current, name))
            for name in type(current).model_fields
            if name not in ("machine", "prop_opt")
        }
        if current is not None
        else {}
    )
    scenario_name = current_identifier(widget._cmb_scenario)
    scenario_props = deepcopy(widget._scenario_props.get(scenario_name, {}))
    scenario: Any = {"model": scenario_name, **scenario_props}
    if (
        current is not None
        and getattr(current.scenarios, "short_name", None) == scenario_name
        and (
            scenario_name == "explicit"
            or _props_key(scenario_overrides(current.scenarios)) == _props_key(scenario_props)
        )
    ):
        scenario = deepcopy(current.scenarios)

    problem_name = current_identifier(widget._cmb_problem)
    prop_opt = widget._copy_prop_opt(widget._problem_props.get(problem_name, {}))
    names_problem = (
        current is not None
        and isinstance(current.prop_opt, dict)
        and bool(current.prop_opt.get("problem") or current.prop_opt.get("opti_prob"))
    )
    if problem_name and (names_problem or problem_name != next(iter(widget._problems), None)):
        prop_opt = {"problem": problem_name, **prop_opt}
    plan_data.update(
        {
            "radiation_mode": radiation_mode,
            "machine": widget._form_machine(),
            "num_of_fractions": int(widget._spn_fractions.value()),
            "dose_convention": widget._cmb_dose_convention.currentData(),
            "bio_model": widget._selected_bio_model(),
            "scenarios": scenario,
            "prop_stf": prop_stf,
            "prop_seq": deepcopy(current.prop_seq) if same_mode else {},
            "prop_dose_calc": prop_dose_calc,
            "prop_opt": prop_opt,
        }
    )
    return plan_class(radiation_mode)(**plan_data)


# ----------------------------------------------------------------------
# Comparison
# ----------------------------------------------------------------------


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        nan = a.dtype.kind in "fc" and b.dtype.kind in "fc"
        return np.array_equal(a, b, equal_nan=nan)
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - objects without a usable equality
        return repr(a) == repr(b)


def plan_diff(a: Any, b: Any, path: str = "") -> list[str]:
    """Paths where two ``model_dump`` trees (or plans) differ."""
    if hasattr(a, "model_dump") and hasattr(b, "model_dump") and type(a) is type(b):
        if not path and not isinstance(a.machine, str) and a.machine is b.machine:
            # The plan's own base data, kept as the same object (hundreds of MB to dump).
            return plan_diff(a.model_dump(exclude={"machine"}), b.model_dump(exclude={"machine"}))
        return plan_diff(a.model_dump(), b.model_dump(), path)
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        return [f"{path or '.'}: {type(a).__name__} {a!r:.80} != {type(b).__name__} {b!r:.80}"]
    if isinstance(a, dict):
        diffs = []
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                side = "Qt" if key in a else "app"
                diffs.append(f"{path}.{key}: only in {side}")
                continue
            diffs.extend(plan_diff(a[key], b[key], f"{path}.{key}"))
        return diffs
    if isinstance(a, (list, tuple)) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in plan_diff(x, y, f"{path}[{i}]")]
    return [] if _same(a, b) else [f"{path or '.'}: {a!r:.80} != {b!r:.80}"]


# ----------------------------------------------------------------------
# Form states
# ----------------------------------------------------------------------


def _set_mode(widget: PlanWidget, mode: str) -> None:
    widget._cmb_radiation.setCurrentText(mode)


def _set_text(field: str, text: str) -> Callable[[PlanWidget], None]:
    return lambda w: getattr(w, field).setText(text)


def _edit_form(model_cls: type, initial: dict, edits: dict, exclude=None) -> dict:
    form = ConfigFormWidget(model_cls, initial=initial, exclude=exclude)
    for name, value in edits.items():
        form._set_value(name, value)
    return form.values()


def _engine_config(edits: dict) -> Callable[[PlanWidget], None]:
    def edit(widget: PlanWidget) -> None:
        name = current_identifier(widget._cmb_engine)
        cls = widget._engines[name]
        widget._engine_props[name] = _edit_form(
            cls.config_model(), widget._engine_props.get(name, {}), edits
        )

    return edit


def _select_engine(name: str) -> Callable[[PlanWidget], None]:
    return lambda w: select_identifier(w._cmb_engine, name)


def _scenario(name: str, edits: dict | None = None) -> Callable[[PlanWidget], None]:
    def edit(widget: PlanWidget) -> None:
        select_identifier(widget._cmb_scenario, name)
        if edits:
            widget._scenario_props[name] = _edit_form(
                get_scenario_model_class(name), widget._scenario_props.get(name, {}), edits
            )

    return edit


def _bio_option(position: int) -> Callable[[PlanWidget], None]:
    def edit(widget: PlanWidget) -> None:
        options = [i for i, o in enumerate(widget._bio_options) if o is not None and o.available]
        widget._cmb_bio_model.setCurrentIndex(options[position % len(options)])

    return edit


def _optimization(problem_edits: dict, solver: str | None = None, solver_edits=None):
    def edit(widget: PlanWidget) -> None:
        name = current_identifier(widget._cmb_problem)
        dialog = OptimizationConfigDialog(
            widget._problems[name], initial=widget._problem_props.get(name, {})
        )
        for field, value in problem_edits.items():
            dialog.problem_form._set_value(field, value)
        if solver is not None:
            dialog.select_solver(solver)
        for field, value in (solver_edits or {}).items():
            dialog.solver_form._set_value(field, value)
        widget._problem_props[name] = dialog.values()

    return edit


def _bio_params(edits: dict) -> Callable[[PlanWidget], None]:
    def edit(widget: PlanWidget) -> None:
        name = widget._bio_model_name()
        cls = widget._bio_model_classes[name]
        widget._bio_params[name] = _edit_form(
            cls.config_model(), widget._bio_params.get(name, {}), edits
        )
        widget._select_matching_option()

    return edit


def _machine(name: str) -> Callable[[PlanWidget], None]:
    def edit(widget: PlanWidget) -> None:
        widget._cmb_machine.setCurrentText(name)
        widget._on_machine_changed()

    return edit


def _plan(**fields: Any) -> Callable[[PlanWidget], None]:
    """Store a plan in the workspace, as a load would."""

    def edit(widget: PlanWidget) -> None:
        from pyRadPlan.plan import IonPlan, PhotonPlan

        cls = PhotonPlan if fields.get("radiation_mode", "photons") == "photons" else IonPlan
        widget._ws.pln = cls(**fields)

    return edit


def _loaded_machine(widget: PlanWidget) -> None:
    from pyRadPlan.machines import load_from_name

    machine = load_from_name("oxygen", "Generic").model_copy(update={"name": "Loaded"})
    _plan(radiation_mode="oxygen", machine=machine, num_of_fractions=5)(widget)


def _apply(widget: PlanWidget) -> None:
    widget._on_apply()


#: (id, steps); each step edits the form, ``_apply`` applies it as the plan.
STATES: list[tuple[str, list[Callable[[PlanWidget], None]]]] = [
    ("default", []),
    (
        "geometry",
        [
            _set_text("_txt_gantry", "0 72 144 216 288"),
            _set_text("_txt_couch", "0"),
            lambda w: w._spn_bixel.setValue(7.0),
            lambda w: w._spn_fractions.setValue(5),
            lambda w: w._spn_res_x.setValue(4.0),
        ],
    ),
    ("couch-per-beam", [_set_text("_txt_gantry", "0 90"), _set_text("_txt_couch", "10 20")]),
    ("iso-manual", [lambda w: w._chk_iso_auto.setChecked(False), _set_text("_txt_iso", "1 2 3")]),
    (
        "per-fraction",
        [
            lambda w: w._cmb_dose_convention.setCurrentIndex(0),
            lambda w: w._spn_fractions.setValue(3),
        ],
    ),
    ("photon-engine-config", [_engine_config({"enable_dij_sampling": False})]),
    ("scenario-wc", [_scenario("wcScen")]),
    ("scenario-wc-props", [_scenario("wcScen", {"wc_sigma": 2.0})]),
    ("scenario-random", [_scenario("rndScen")]),
    ("scenario-wc-applied", [_scenario("wcScen", {"wc_sigma": 2.0}), _apply]),
    ("optimization-problem", [_optimization({"p_norm": 10.0})]),
    ("optimization-solver", [_optimization({}, "scipy", {"max_iter": 20})]),
    ("optimization-applied", [_optimization({}, "scipy", {"max_iter": 20}), _apply]),
    ("protons", [lambda w: _set_mode(w, "protons")]),
    ("protons-applied", [lambda w: _set_mode(w, "protons"), _apply]),
    (
        "protons-geometry-applied",
        [lambda w: _set_mode(w, "protons"), _apply, _set_text("_txt_gantry", "90 270")],
    ),
    *[(f"protons-bio-{i}", [lambda w: _set_mode(w, "protons"), _bio_option(i)]) for i in range(4)],
    *[
        (f"protons-bio-{i}-applied", [lambda w: _set_mode(w, "protons"), _bio_option(i), _apply])
        for i in range(4)
    ],
    ("protons-engine-HongPB", [lambda w: _set_mode(w, "protons"), _select_engine("HongPB")]),
    (
        "protons-engine-config",
        [lambda w: _set_mode(w, "protons"), _engine_config({"enable_dij_sampling": False})],
    ),
    ("carbon", [lambda w: _set_mode(w, "carbon")]),
    ("carbon-applied-bio-1", [lambda w: _set_mode(w, "carbon"), _apply, _bio_option(1)]),
    ("helium", [lambda w: _set_mode(w, "helium")]),
    (
        "protons-to-photons",
        [lambda w: _set_mode(w, "protons"), _apply, lambda w: _set_mode(w, "photons")],
    ),
    (
        "protons-scenario-wc",
        [lambda w: _set_mode(w, "protons"), _apply, _scenario("wcScen", {"wc_sigma": 1.5})],
    ),
    ("scenario-importance", [_scenario("impScen")]),
    ("protons-constant-rbe-1.2", [lambda w: _set_mode(w, "protons"), _bio_params({"rbe": 1.2})]),
    (
        "protons-constant-rbe-1.2-applied",
        [lambda w: _set_mode(w, "protons"), _bio_params({"rbe": 1.2}), _apply],
    ),
    *[
        (
            f"{mode}-engine-{engine}",
            [lambda w, m=mode: _set_mode(w, m), _select_engine(engine), _engine_config(edits)],
        )
        for mode, engine, edits in (
            ("photons", "SVDPB", {"keep_rad_depth_cubes": True}),
            ("protons", "HongPB", {"geometric_lateral_cutoff": 40.0}),
            ("protons", "FRED", {"num_histories_per_beamlet": 1000}),
            ("protons", "TOPAS", {"num_histories_per_beamlet": 1000}),
            ("carbon", "FRED", {"num_histories_direct": 1000}),
            ("helium", "TOPAS", {}),
        )
    ],
    (
        "engine-config-applied-then-other-engine",
        [
            lambda w: _set_mode(w, "protons"),
            _engine_config({"keep_rad_depth_cubes": True}),
            _apply,
            _select_engine("FRED"),
        ],
    ),
    (
        "carbon-machine",
        [lambda w: _set_mode(w, "carbon"), _machine("MCF_doubleGauss_ridgeFilter_sigmaAveraged")],
    ),
    ("oxygen", [lambda w: _set_mode(w, "oxygen")]),
    ("VHEE-focused", [lambda w: _set_mode(w, "VHEE"), _machine("Focused")]),
    *[(f"solver-{solver}", [_optimization({}, solver)]) for solver in ("ipopt", "ipax", "scipy")],
    (
        "solver-ipopt-applied-then-scipy",
        [_optimization({}, "ipopt"), _apply, _optimization({}, "scipy")],
    ),
    (
        "per-beam-iso-loaded",
        [
            _plan(
                prop_stf={
                    "gantry_angles": [0.0, 180.0],
                    "couch_angles": [0.0, 0.0],
                    "iso_center": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                }
            ),
            lambda w: w._spn_fractions.setValue(4),
        ],
    ),
    ("machine-object-loaded", [_loaded_machine, lambda w: w._spn_fractions.setValue(7)]),
    (
        "machine-object-loaded-then-generic",
        [_loaded_machine, _machine("Generic")],
    ),
    (
        "loaded-plan-with-prescription",
        [
            _plan(radiation_mode="protons", num_of_fractions=10, prescribed_dose=60.0),
            _set_text("_txt_gantry", "45"),
        ],
    ),
]


def _applied(widget: PlanWidget) -> Any:
    """Apply the form (the real path) and return the plan it wrote."""
    widget._on_apply()
    assert not widget._status_is_error, widget._lbl_status.text()
    return widget._ws.pln


def apply_mismatches(widget: PlanWidget) -> list[str]:
    """Differences between the reference plan and the plan Apply writes.

    Known difference: a plan leaving the dose engine to its radiation mode keeps doing so
    while the form shows that default engine; the reference named it.
    """
    from pyRadPlan.dose.engines import get_engine_class

    before = widget._ws.pln
    reference = reference_plan(widget)
    diffs = plan_diff(reference, _applied(widget))
    default = get_engine_class(reference.radiation_mode, None).short_name
    implicit = (
        before is not None
        and before.radiation_mode == reference.radiation_mode
        and not (before.prop_dose_calc or {}).get("engine")
    )
    if implicit and reference.prop_dose_calc.get("engine") == default:
        diffs = [d for d in diffs if d != ".prop_dose_calc.engine: only in Qt"]
    return diffs


def form_mismatches(tg119, steps) -> list[str]:
    ct, cst = tg119
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    ws.set_many(ct=ct, cst=cst)
    for step in steps:
        step(widget)
    return apply_mismatches(widget)


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(steps, id=name, marks=[skipif_no_ipopt] if "ipopt" in name else [])
        for name, steps in STATES
    ],
)
def test_apply_writes_the_plan_the_form_built(qapp, tg119, steps):
    assert form_mismatches(tg119, steps) == []


def test_the_default_plan_is_the_profile_default(qapp, tg119):
    from pyRadPlan.app import build_plan

    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    default = build_plan(None, ws.session.default_plan)
    # The form names the mode's default dose engine; the profile leaves it to the mode.
    known = [".prop_dose_calc.engine: only in Qt"]
    assert plan_diff(reference_plan(widget), default) == known
    ws.set_many(ct=tg119[0], cst=tg119[1])
    assert widget._lbl_status.text() == "Default plan applied."
    assert plan_diff(ws.pln, default) == known


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [lambda w: _set_mode(w, "protons"), _bio_option(2), _scenario("wcScen")],
        [_set_text("_txt_gantry", "0 120 240"), _optimization({}, "scipy", {"max_iter": 5})],
    ],
    ids=["default", "protons", "beams-solver"],
)
def test_apply_without_a_plan_writes_the_form_plan(qapp, tg119, steps):
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    ws.ct = tg119[0]
    for step in steps:
        step(widget)
    assert ws.pln is None
    assert apply_mismatches(widget) == []


@pytest.mark.parametrize(
    "plan",
    [
        {
            "radiation_mode": "carbon",
            "bio_model": "alpha_beta_kernels",
            "prop_dose_calc": {"engine": "FRED"},
        },
        {
            "radiation_mode": "oxygen",
            "bio_model": {"model": "alpha_beta_kernels", "data_model": "mMKM"},
        },
    ],
    ids=["engine", "machine"],
)
def test_a_plan_model_the_form_cannot_offer_is_kept(qapp, tg119, plan):
    steps = [_plan(**plan), lambda w: w._spn_fractions.setValue(3)]
    ws = WorkspaceManager()
    widget = PlanWidget(workspace=ws)
    ws.set_many(ct=tg119[0], cst=tg119[1])
    for step in steps:
        step(widget)
    assert widget._unsupported_bio_model is not None
    assert "bio_model" not in widget._form_changes()
    assert apply_mismatches(widget) == []

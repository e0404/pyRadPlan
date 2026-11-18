"""Plan editor support (protocol v1): plan detail, options, algorithm forms, v1 plan.set.

The rules follow ``pyRadPlan.gui.widgets.plan.PlanWidget`` and its optimization dialog:
what the form lists for a radiation mode, how a bio-model option selects base data, how
``prop_opt`` is assembled from the problem and the solver, when a scenario model instance
is kept.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping, Optional

from ._forms import FormValidationError, build_form, jsonable, validate_values
from ._plan import DEFAULT_PLAN, plan_summary

#: Engine fields the plan sets elsewhere (scenario model, biological model, dose grid);
#: the Qt engine form shows them by accident (see ``web/docs/parity.md``).
ENGINE_FORM_EXCLUDE = frozenset(
    {"mult_scen", "bio_model", "dose_grid", "voxel_sub_ix", "select_voxels_in_scenarios"}
)
#: Problem fields the problem form leaves out (Qt ``_HIDDEN_PROBLEM_FIELDS``).
PROBLEM_FORM_EXCLUDE = frozenset({"solver", "convert_dose_objectives"})

RADIATION_MODES: tuple[str, ...] = ("photons", "protons", "helium", "carbon", "oxygen", "VHEE")


def _choice(value: Any, label: str, identifier: Optional[str] = None, **extra: Any) -> dict:
    choice = {"value": value, "label": label}
    if identifier is not None:
        choice["identifier"] = identifier
    choice.update({k: v for k, v in extra.items() if v is not None})
    return choice


def option_id(option: Any) -> str:
    """Stable identifier of a ``BioModelOption`` (formalism + base-data selection)."""
    from pyRadPlan.biology._editing import selection_parameters  # noqa: PLC0415

    return json.dumps(
        {"model": option.model, "parameters": jsonable(selection_parameters(option.parameters))},
        sort_keys=True,
    )


def _machine_of(radiation_mode: str, machine: Any) -> Any:
    from pyRadPlan.biology._editing import load_machine_quietly  # noqa: PLC0415

    if machine is None or isinstance(machine, str):
        return load_machine_quietly(radiation_mode, machine or "Generic")
    return machine


def bio_options(radiation_mode: str, machine: Any = None, engine: Optional[str] = None) -> list:
    """Return the biological model options of a radiation mode, machine and dose engine."""
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        bio_options_quietly,
        dose_engine_quietly,
    )

    return bio_options_quietly(
        radiation_mode,
        _machine_of(radiation_mode, machine),
        dose_engine_quietly(radiation_mode, engine),
    )


def bio_configurable(model: str, radiation_mode: str) -> bool:
    """Whether the bio model's "…" form has any field (Qt enables the button only then)."""
    try:
        form_cls, title, exclude, defaults = form_model("bio_model", model, radiation_mode)
        form = build_form(
            form_cls,
            form_id=f"bio_model:{model}",
            title=title,
            values={},
            defaults=defaults,
            exclude=exclude,
        )
    except Exception:  # noqa: BLE001 - an unknown model has nothing to configure
        return False
    return bool(form["fields"])


def _bio_choices(options: list, radiation_mode: str = "photons") -> list[dict]:
    from pyRadPlan.biology._editing import selection_parameters  # noqa: PLC0415

    configurable: dict[str, bool] = {}
    for option in options:
        if option.model not in configurable:
            configurable[option.model] = bio_configurable(option.model, radiation_mode)
    return [
        _choice(
            option_id(option),
            option.label or option.model,
            option.model,
            enabled=bool(option.available),
            reason=option.reason or None,
            description=option.description or None,
            model=option.model,
            parameters=jsonable(selection_parameters(option.parameters)),
            configurable=configurable[option.model],
        )
        for option in options
    ]


def _default_option(radiation_mode: str, options: list) -> Optional[str]:
    from pyRadPlan.plan import default_bio_models  # noqa: PLC0415

    wanted = default_bio_models.get(radiation_mode, "none")
    for model in (wanted, "none"):
        for option in options:
            if option.model == model and option.available:
                return option_id(option)
    return option_id(options[0]) if options else None


def _engine_name(pln: Any) -> Optional[str]:
    prop = pln.prop_dose_calc if pln is not None and isinstance(pln.prop_dose_calc, dict) else {}
    name = prop.get("engine")
    return name if isinstance(name, str) else None


def _solver_spec(spec: Any) -> tuple[Optional[str], dict[str, Any]]:
    """Solver name and configuration of a ``prop_opt["solver"]`` entry (Qt)."""
    from pyRadPlan.optimization.solvers import SolverBase  # noqa: PLC0415

    if spec is None:
        return None, {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, SolverBase):
        names = spec.config_model().model_fields
        return spec.short_name, {n: getattr(spec, n) for n in names if hasattr(spec, n)}
    if isinstance(spec, Mapping):
        values = dict(spec)
        return values.pop("name", None), values
    return None, {}


def _problems(radiation_mode: str) -> dict[str, type]:
    from pyRadPlan.optimization.problems import get_available_problems  # noqa: PLC0415

    return {
        name: cls
        for name, cls in get_available_problems().items()
        if radiation_mode in getattr(cls, "possible_radiation_modes", (radiation_mode,))
    }


def default_solver(problem_cls: type) -> Optional[str]:
    """Return the solver a problem picks when none is configured (Qt dialog)."""
    from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

    solvers = get_available_solvers()
    default = problem_cls.config_model().model_fields["solver"].default
    if isinstance(default, str) and default in solvers:
        return default
    return next(iter(solvers), default if isinstance(default, str) else None)


def _problem_of(pln: Any, radiation_mode: str) -> tuple[Optional[str], dict[str, Any]]:
    prop_opt = pln.prop_opt if pln is not None and isinstance(pln.prop_opt, dict) else {}
    problems = _problems(radiation_mode)
    requested = prop_opt.get("problem") or prop_opt.get("opti_prob")
    default = next(iter(problems), None)
    name = requested if requested in problems else default
    values = {k: v for k, v in prop_opt.items() if k not in ("problem", "opti_prob")}
    return name, values


def _scenario_overrides(model: Any) -> dict:
    overrides = model.model_dump(exclude_unset=True)
    overrides.pop("model", None)
    for key, value in list(overrides.items()):
        if isinstance(value, dict):
            nested = {k: v for k, v in value.items() if k != "dimension"}
            if nested:
                overrides[key] = nested
            else:
                del overrides[key]
    return overrides


def plan_detail(pln: Any, cst: Any) -> Optional[dict[str, Any]]:
    """``PlanDetail`` of a plan (``None`` without one)."""
    if pln is None:
        return None
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        SELECTION_PARAMETERS,
        matching_bio_option,
    )
    from pyRadPlan.stf import resolve_iso_centers  # noqa: PLC0415

    summary = plan_summary(pln)
    mode = pln.radiation_mode
    resolved = None
    try:
        iso = resolve_iso_centers((pln.prop_stf or {}).get("iso_center"), cst)
        resolved = [float(v) for v in iso[0]]
    except Exception:  # noqa: BLE001 - the isocenter preview is optional
        resolved = None
    prop_dose_calc = pln.prop_dose_calc if isinstance(pln.prop_dose_calc, dict) else {}
    model = pln.bio_model
    parameters = dict(getattr(model, "parameters", {}) or {})
    option = None
    try:
        options = bio_options(mode, pln.machine, _engine_name(pln))
        match = matching_bio_option(options, model, _machine_of(mode, pln.machine))
        option = option_id(match) if match is not None else None
    except Exception:  # noqa: BLE001 - without base data no option is selected
        option = None
    problem, problem_values = _problem_of(pln, mode)
    solver, solver_values = _solver_spec(problem_values.pop("solver", None))
    return {
        **summary,
        "dose_convention": pln.dose_convention,
        "prescribed_dose": float(pln.prescribed_dose),
        "machine_source": "name" if isinstance(pln.machine, str) else "plan",
        "resolved_iso_center": resolved,
        "engine_config": {
            "name": _engine_name(pln),
            "values": jsonable(
                {k: v for k, v in prop_dose_calc.items() if k not in ("engine", "dose_grid")}
            ),
        },
        "bio": {
            "model": getattr(model, "model", "none"),
            "option": option,
            "values": jsonable(
                {k: v for k, v in parameters.items() if k not in SELECTION_PARAMETERS}
            ),
        },
        "scenario": {
            "model": getattr(pln.scenarios, "short_name", None),
            "values": jsonable(_scenario_overrides(pln.scenarios)),
        },
        "problem": {"name": problem, "values": jsonable(problem_values)},
        "solver": {"name": solver, "values": jsonable(solver_values)},
    }


def _plan_fields(locked: set[str], default_plan: Mapping[str, Any]) -> list[dict]:
    def field(name, label, type_, **extra):
        entry = {"name": name, "label": label, "type": type_, **extra}
        if name in default_plan:
            entry.setdefault("default", jsonable(default_plan[name]))
        if name in locked:
            entry["editable"] = False
            entry["reason"] = "Fixed in this deployment"
        return entry

    off = {"editable": False, "reason": "not yet implemented", "advanced": True}
    return [
        field(
            "radiation_mode",
            "Radiation mode",
            "choice",
            choices=[{"value": m, "label": m} for m in RADIATION_MODES],
        ),  # fmt: skip
        field("machine", "Machine", "choice", editable_choice=True, advanced=True),
        field("engine", "Dose engine", "choice", nullable=True, advanced=True),
        field("num_of_fractions", "Fractions", "int", min=1, max=1000, advanced=True),
        field(
            "dose_convention",
            "Doses",
            "choice",
            advanced=True,
            choices=[
                {"value": "per_fraction", "label": "per fraction"},
                {"value": "total", "label": "total"},
            ],
        ),
        field("gantry_angles", "Gantry angles", "number_list", item_type="float", unit="deg"),
        field("couch_angles", "Couch angles", "number_list", item_type="float", unit="deg"),
        field(
            "bixel_width",
            "Bixel width / spot spacing",
            "float",
            min=0.1,
            max=100.0,
            decimals=2,
            unit="mm",
        ),  # fmt: skip
        field(
            "iso_center",
            "Iso center",
            "number_list",
            item_type="float",
            length=3,
            nullable=True,
            unit="mm",
        ),  # fmt: skip
        field("bio_model", "Biological model", "choice"),
        field("scenarios", "Scenario model", "choice", advanced=True),
        field(
            "dose_grid_resolution",
            "Dose grid res.",
            "number_list",
            item_type="float",
            length=3,
            min=0.1,
            max=50.0,
            decimals=2,
            unit="mm",
        ),  # fmt: skip
        field("problem", "Optimization problem", "choice", advanced=True),
        field(
            "optimized_quantity",
            "Optimized quantity",
            "choice",
            default="auto",
            choices=[{"value": "auto", "label": "auto"}],
            **off,
        ),  # fmt: skip
        field("run_sequencing", "Run sequencing", "bool", default=False, **off),
        field("sequencing_levels", "Levels", "int", default=7, min=1, max=100, **off),
        field("run_dao", "Run DAO", "bool", default=False, **off),
        field("conformal", "3D conformal", "bool", default=False, **off),
    ]


def plan_options(
    pln: Any,
    default_plan: Mapping[str, Any],
    locked: set[str],
    radiation_mode: Optional[str] = None,
    machine: Optional[str] = None,
    engine: Optional[str] = None,
) -> dict[str, Any]:
    """``PlanOptions`` for a radiation mode, machine and engine (defaults: the plan's)."""
    from pyRadPlan.dose.engines import get_available_engines  # noqa: PLC0415
    from pyRadPlan.machines import get_available_machines  # noqa: PLC0415
    from pyRadPlan.optimization.solvers import (  # noqa: PLC0415
        IPOPT_DISABLED_REASON,
        get_available_solvers,
    )
    from pyRadPlan.scenarios import (  # noqa: PLC0415
        available_scenario_models,
        get_scenario_model_class,
    )
    from pyRadPlan.stf import get_beam_angle_presets  # noqa: PLC0415

    from ._profiles import running_in_pyodide  # noqa: PLC0415

    plan_mode = pln.radiation_mode if pln is not None else default_plan.get("radiation_mode")
    mode = radiation_mode or plan_mode or "photons"
    same_mode = pln is not None and mode == pln.radiation_mode
    machine_value: Any = machine
    if machine_value is None:
        machine_value = pln.machine if same_mode else default_plan.get("machine", "Generic")
    engine_name = engine if engine is not None else (_engine_name(pln) if same_mode else None)

    machines = [_choice(name, name, name) for name in get_available_machines(mode).get(mode, [])]
    if same_mode and not isinstance(pln.machine, str) and pln.machine is not None:
        label = f"{getattr(pln.machine, 'name', 'machine')} (from plan)"
        machines.insert(0, _choice("__plan__", label, None))

    engines = []
    for name, cls in get_available_engines(mode).items():
        reason = None
        if running_in_pyodide() and name in ("FRED", "TOPAS"):
            reason = "Monte Carlo engines run external executables, not available in the browser"
        engines.append(
            _choice(
                name,
                getattr(cls, "name", name) or name,
                name,
                enabled=reason is None,
                reason=reason,
            )  # fmt: skip
        )

    options = bio_options(mode, machine_value, engine_name)

    scenario_models = []
    for name in available_scenario_models():
        cls = get_scenario_model_class(name)
        scenario_models.append(_choice(name, getattr(cls, "name", name) or name, name))
    if same_mode and getattr(pln.scenarios, "short_name", None) == "explicit":
        scenario_models.append(_choice("explicit", "Explicit Scenarios", "explicit"))

    problems = [
        _choice(name, getattr(cls, "name", name) or name, name)
        for name, cls in _problems(mode).items()
    ]
    solvers = []
    registered = get_available_solvers()
    for name, cls in registered.items():
        solvers.append(
            {
                **_choice(name, getattr(cls, "name", name) or name, name),
                "gpu_compatible": bool(getattr(cls, "gpu_compatible", False)),
            }
        )
    if "ipopt" not in registered:
        reason = IPOPT_DISABLED_REASON or "it is not installed in this environment"
        solvers.append(
            {
                **_choice(
                    "ipopt",
                    "Interior Point Optimizer",
                    "ipopt",
                    enabled=False,
                    reason=f"Solver 'ipopt' is unavailable: {reason}",
                ),  # fmt: skip
                "gpu_compatible": False,
            }
        )
    problem_classes = _problems(mode)
    first_problem = next(iter(problem_classes.values()), None)
    presets = [
        {
            "name": preset.name,
            "label": preset.name,
            "gantry_angles": [float(a) for a in preset.gantry_angles],
            "couch_angles": [float(a) for a in preset.couch_angles]
            or [0.0] * len(preset.gantry_angles),
            "description": preset.description,
            "typical": mode in preset.radiation_modes,
        }
        for preset in get_beam_angle_presets(mode)
    ]
    return {
        "radiation_mode": mode,
        "radiation_modes": [_choice(m, m) for m in RADIATION_MODES],
        "machines": machines,
        "engines": engines,
        "bio_models": _bio_choices(options, mode),
        "default_bio_model": _default_option(mode, options),
        "scenario_models": scenario_models,
        "problems": problems,
        "solvers": solvers,
        "default_solver": default_solver(first_problem) if first_problem else None,
        "beam_presets": presets,
        "fields": _plan_fields(locked, default_plan),
    }


def _engine_class(mode: str, name: Optional[str]) -> type:
    from pyRadPlan.dose.engines import get_engine_class  # noqa: PLC0415

    cls = get_engine_class(mode, name or None)
    if cls is None:
        raise ValueError(f"No dose engine {name!r} for {mode}")
    return cls


def _bio_class(name: str, mode: str) -> type:
    from pyRadPlan.biology.models import create_bio_model  # noqa: PLC0415
    from pyRadPlan.biology.models._factory import BIO_MODELS  # noqa: PLC0415

    cls = BIO_MODELS.get(name)
    return cls if cls is not None else type(create_bio_model(name, mode))


def _solver_defaults(solver_cls: type) -> dict[str, Any]:
    from pyRadPlan.core._configurable import field_constraints  # noqa: PLC0415

    try:
        solver = solver_cls()
    except Exception:  # noqa: BLE001 - the form then shows the model defaults
        return {}
    model = solver_cls.config_model()
    names = [
        n
        for n, info in model.model_fields.items()
        if getattr(field_constraints(info).get("param_meta"), "configurable", True)
    ]
    return {n: getattr(solver, n) for n in names if hasattr(solver, n)}


def form_model(kind: str, name: str, mode: str) -> tuple[type, str, frozenset, dict]:
    """Model class, title, excluded fields and display defaults of an algorithm form."""
    if kind == "engine":
        cls = _engine_class(mode, name)
        return cls.config_model(), f"Configure {cls.name}", ENGINE_FORM_EXCLUDE, {}
    if kind == "bio_model":
        from pyRadPlan.biology._editing import SELECTION_PARAMETERS  # noqa: PLC0415

        cls = _bio_class(name, mode)
        title = f"Configure {getattr(cls, 'name', None) or cls.model}"
        return cls.config_model(), title, frozenset(SELECTION_PARAMETERS), {}
    if kind == "scenario":
        from pyRadPlan.scenarios import get_scenario_model_class  # noqa: PLC0415

        cls = get_scenario_model_class(name)
        return cls, f"Configure {cls.name}", frozenset({"model"}), {}
    if kind == "problem":
        cls = _problems(mode).get(name)
        if cls is None:
            raise ValueError(f"No planning problem {name!r} for {mode}")
        exclude = PROBLEM_FORM_EXCLUDE | frozenset(getattr(cls, "solver_properties", ()))
        return cls.config_model(), f"Configure {cls.name}", exclude, {}
    if kind == "solver":
        from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

        cls = get_available_solvers().get(name)
        if cls is None:
            raise ValueError(f"Solver {name!r} is not available")
        return cls.config_model(), f"Configure {cls.name}", frozenset(), _solver_defaults(cls)
    raise ValueError(f"Unknown form kind {kind!r}")


def plan_form(
    pln: Any,
    kind: str,
    name: str,
    radiation_mode: Optional[str] = None,
    values: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """``FormSchema`` of an algorithm, filled with *values* or the plan's."""
    mode = radiation_mode or (pln.radiation_mode if pln is not None else "photons")
    notes: list[str] = []
    if kind == "solver":
        from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

        if name not in get_available_solvers():
            from pyRadPlan.optimization.solvers import IPOPT_DISABLED_REASON  # noqa: PLC0415

            reason = (IPOPT_DISABLED_REASON if name == "ipopt" else None) or (
                "it is not installed in this environment"
            )
            return {
                "id": f"solver:{name}",
                "title": f"Configure {name}",
                "fields": [],
                "values": {},
                "explicit": [],
                "notes": [f"Solver '{name}' is unavailable: {reason}. Its configuration is kept."],
            }
    model, title, exclude, defaults = form_model(kind, name, mode)
    if values is None:
        values = _plan_values(pln, kind, name, mode)
    return build_form(
        model,
        form_id=f"{kind}:{name}",
        title=title,
        values=values,
        defaults=defaults,
        exclude=exclude,
        notes=notes,
    )


def _plan_values(pln: Any, kind: str, name: str, mode: str) -> dict[str, Any]:
    if pln is None or pln.radiation_mode != mode:
        return {}
    if kind == "engine":
        if _engine_name(pln) not in (None, name):
            return {}
        prop = pln.prop_dose_calc if isinstance(pln.prop_dose_calc, dict) else {}
        return {k: v for k, v in prop.items() if k not in ("engine", "dose_grid")}
    if kind == "bio_model":
        model = pln.bio_model
        if getattr(model, "model", None) != name:
            return {}
        return dict(getattr(model, "parameters", {}) or {})
    if kind == "scenario":
        if getattr(pln.scenarios, "short_name", None) != name:
            return {}
        return _scenario_overrides(pln.scenarios)
    problem, values = _problem_of(pln, mode)
    if kind == "problem":
        values.pop("solver", None)
        return values if problem == name else {}
    if kind == "solver":
        solver, solver_values = _solver_spec(values.get("solver"))
        if solver not in (None, name):
            return {}
        from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

        cls = get_available_solvers().get(name)
        problem_cls = _problems(mode).get(problem) if problem else None
        fields = cls.config_model().model_fields if cls is not None else {}
        for prop in getattr(problem_cls, "solver_properties", ()):
            if values.get(prop) is not None and prop in fields:
                solver_values.setdefault(prop, values[prop])
        return solver_values
    return {}


def _check_form(kind: str, name: str, mode: str, values: Mapping[str, Any]) -> dict:
    model, _, exclude, defaults = form_model(kind, name, mode)
    try:
        return validate_values(model, values, exclude=exclude, base=defaults or None)
    except FormValidationError as exc:
        for entry in exc.details:
            entry["field"] = f"{kind}.{entry['field']}"
        raise


def resolve_changes(pln: Any, changes: Mapping[str, Any]) -> tuple[dict, dict]:
    """Split v1 ``plan.set`` params into ``build_plan`` changes and overrides.

    Raises
    ------
    FormValidationError, ValueError
        For values the algorithm models reject.
    """
    from pyRadPlan.biology._editing import SELECTION_PARAMETERS  # noqa: PLC0415

    changes = dict(changes)
    overrides: dict[str, Any] = {}
    summary = plan_summary(pln) or dict(DEFAULT_PLAN)
    mode = changes.get("radiation_mode", summary["radiation_mode"])
    same_mode = pln is not None and pln.radiation_mode == mode

    if changes.get("machine") == "__plan__":
        changes.pop("machine")

    engine = changes.get("engine")
    if engine is not None:
        from pyRadPlan.dose.engines import get_available_engines  # noqa: PLC0415

        if engine not in get_available_engines(mode):
            message = f"dose engine {engine!r} is not available for {mode}"
            raise FormValidationError(message, [{"field": "engine", "message": message}])

    engine_config = changes.pop("engine_config", None)
    if engine_config is not None:
        engine = changes.get("engine", summary.get("engine") if same_mode else None)
        cls = _engine_class(mode, engine)
        overrides["engine_props"] = _check_form(
            "engine", cls.short_name, mode, dict(engine_config.get("values") or {})
        )

    bio = changes.get("bio_model")
    if isinstance(bio, Mapping):
        model = bio.get("model") or "none"
        params: dict[str, Any] = {}
        option = bio.get("option")
        if option:
            params.update(json.loads(option).get("parameters", {}))
        values = dict(bio.get("values") or {})
        if values:
            checked = _check_form("bio_model", model, mode, values)
            params.update({k: v for k, v in checked.items() if k not in SELECTION_PARAMETERS})
        changes["bio_model"] = {"model": model, **params} if params else model

    scenarios = changes.pop("scenarios", None)
    if scenarios is not None:
        model = scenarios.get("model") if isinstance(scenarios, Mapping) else scenarios
        values = dict(scenarios.get("values") or {}) if isinstance(scenarios, Mapping) else {}
        current = pln.scenarios if pln is not None else None
        if (
            same_mode
            and current is not None
            and getattr(current, "short_name", None) == model
            and (
                model == "explicit"
                or json.dumps(jsonable(_scenario_overrides(current)), sort_keys=True)
                == json.dumps(jsonable(values), sort_keys=True)
            )
        ):
            overrides["scenarios"] = deepcopy(current)
        else:
            if values:
                _check_form("scenario", model, mode, values)
            overrides["scenarios"] = {"model": model, **values}

    problem = changes.pop("problem", None)
    solver = changes.pop("solver", None)
    if problem is not None or solver is not None:
        overrides["prop_opt"] = _prop_opt(pln, mode, same_mode, problem, solver)
    return changes, overrides


def _prop_opt(pln: Any, mode: str, same_mode: bool, problem: Any, solver: Any) -> dict:
    problems = _problems(mode)
    current_name, current_values = _problem_of(pln, mode) if same_mode else (None, {})
    prop_opt_named = bool(
        same_mode and isinstance(pln.prop_opt, dict) and pln.prop_opt.get("problem")
    )
    if isinstance(problem, Mapping):
        name = problem.get("name") or current_name or next(iter(problems), None)
        values = dict(problem.get("values") or {})
        problem_values = _check_form("problem", name, mode, values) if values else {}
        kept_solver = current_values.get("solver") if name == current_name else None
        forwarded = {
            k: current_values[k]
            for k in getattr(problems.get(name), "solver_properties", ())
            if k in current_values and name == current_name
        }
    else:
        name = current_name or next(iter(problems), None)
        problem_values = {k: v for k, v in current_values.items() if k != "solver"}
        kept_solver = current_values.get("solver")
        forwarded = {}
    if name is None or name not in problems:
        raise ValueError(f"No planning problem {name!r} for {mode}")
    problem_cls = problems[name]
    prop_opt: dict[str, Any] = {}
    if prop_opt_named or name != next(iter(problems), None):
        prop_opt["problem"] = name
    prop_opt.update(problem_values)
    prop_opt.update({k: v for k, v in forwarded.items() if k not in prop_opt})

    if solver is None:
        if kept_solver is not None:
            prop_opt["solver"] = kept_solver
        return prop_opt
    if isinstance(solver, str):
        solver_name, solver_values = solver, {}
    else:
        solver_name = solver.get("name")
        solver_values = dict(solver.get("values") or {})
    from pyRadPlan.optimization.solvers import get_available_solvers  # noqa: PLC0415

    solver_cls = get_available_solvers().get(solver_name)
    if solver_cls is None:
        if solver_name is None:
            return prop_opt
        prop_opt["solver"] = (
            kept_solver if _solver_spec(kept_solver)[0] == solver_name else (solver_name)
        )
        return prop_opt
    checked = _check_form("solver", solver_name, mode, solver_values) if solver_values else {}
    for prop in getattr(problem_cls, "solver_properties", ()):
        if prop in checked:
            prop_opt.pop(prop, None)
    if checked:
        prop_opt["solver"] = {"name": solver_name, **checked}
    elif solver_name != default_solver(problem_cls) or kept_solver is not None:
        prop_opt["solver"] = solver_name
    return prop_opt


def draft_context(pln: Any, draft: Optional[Mapping[str, Any]] = None) -> Optional[tuple]:
    """Resolve an unapplied plan form ``{radiation_mode, machine, engine, bio_model}``.

    Missing entries fall back to the plan. ``bio_model`` is a model name or
    ``{model, option, values}`` (the ``plan.set`` form). Returns ``(radiation_mode, machine,
    engine, bio_model_spec)`` or ``None`` without a radiation mode.
    """
    draft = dict(draft or {})
    mode = draft.get("radiation_mode") or (pln.radiation_mode if pln is not None else None)
    if mode is None:
        return None
    same_mode = pln is not None and mode == pln.radiation_mode
    machine = draft.get("machine", pln.machine if same_mode else "Generic")
    engine = draft.get("engine", _engine_name(pln) if same_mode else None)
    spec = draft.get("bio_model", pln.bio_model if same_mode else None)
    if isinstance(spec, Mapping) and "option" in spec:
        params = json.loads(spec["option"]).get("parameters", {}) if spec.get("option") else {}
        spec = {"model": spec.get("model", "none"), **params, **dict(spec.get("values") or {})}
    return mode, machine, engine, spec


def tissue_check(pln: Any, cst: Any, draft: Optional[Mapping[str, Any]] = None) -> dict:
    """Return the plan form's tissue marker for the plan or an unapplied draft."""
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        bio_model_quietly,
        dose_engine_quietly,
        tissue_issues_quietly,
    )

    context = draft_context(pln, draft)
    if context is None or cst is None:
        return {"marker": "", "issues": []}
    mode, machine, engine, spec = context
    model = bio_model_quietly(spec, mode)
    issues = tissue_issues_quietly(
        _machine_of(mode, machine), model, cst, dose_engine_quietly(mode, engine)
    )
    errors = [issue for issue in issues if issue.is_error]
    return {
        "marker": "error" if errors else ("warning" if issues else ""),
        "issues": [issue_dict(issue) for issue in issues],
    }


def issue_dict(issue: Any) -> dict[str, Any]:
    """Return the ``Issue`` (protocol) of a ``TissueIssue``."""
    from pyRadPlan.biology._editing import tissue_issue_text  # noqa: PLC0415

    return {
        "structure": getattr(issue, "structure", None),
        "kind": getattr(issue, "kind", ""),
        "severity": "error" if issue.is_error else "warning",
        "message": tissue_issue_text(issue),
        "closest_class": _class_label(getattr(issue, "suggestion", None)),
    }


def _class_label(tissue: Any) -> Optional[str]:
    if tissue is None:
        return None
    from pyRadPlan.biology._editing import class_label  # noqa: PLC0415

    return class_label(tissue)

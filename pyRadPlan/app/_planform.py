"""Plan editor support (protocol v1): plan detail, options, algorithm forms, v1 plan.set.

The rules follow ``pyRadPlan.gui.widgets.plan.PlanWidget`` and its optimization dialog:
what the form lists for a radiation mode, how a bio-model option selects base data, how
``prop_opt`` is assembled from the problem and the solver, when a scenario model instance
is kept.
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

from ._forms import FormValidationError, build_form, comparison_key, jsonable, validate_values
from ._plan import DEFAULT_PLAN, plan_summary
from ._stages import sequencing_settings

#: Engine fields the plan sets elsewhere (scenario model, biological model, dose grid);
#: the Qt engine form shows them by accident (see ``web/docs/parity.md``).
ENGINE_FORM_EXCLUDE = frozenset(
    {"mult_scen", "bio_model", "dose_grid", "voxel_sub_ix", "select_voxels_in_scenarios"}
)
#: Problem fields the problem form leaves out. The solver has its own selector; the legacy
#: conversion of physical-dose objectives is switched off by the GUI workflows, whose
#: objectives tables set every quantity, unless a plan explicitly asks for it.
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


#: Suffix of the model entry keeping a plan's model the machine cannot run.
UNSUPPORTED_SUFFIX = "(not supported by this machine)"


@dataclass
class BioSelection:
    """Biological model entries of the plan form and which one is selected.

    Attributes
    ----------
    options : list
        The ``BioModelOption`` entries, in order; unavailable ones are listed disabled.
    selected : BioModelOption or None
        The selected option; ``None`` while the plan's model is kept as the extra
        ``unsupported`` entry (or without any entry).
    unsupported : BiologicalModel or None
        The plan's model the context cannot run, kept as a last, disabled entry so that
        Apply keeps it.
    unsupported_label, unsupported_tooltip : str
        Text of that entry.
    replaced : bool
        The selection was replaced (or newly chosen): the form takes over the base-data
        parameters of the selected option. ``False`` keeps the model's parameters.
    note : tuple of str or None
        Status note and its reason when a selection the new context cannot run was
        replaced.
    model_classes : dict
        The biological model classes of the radiation mode, by ``model`` name.
    """

    options: list
    selected: Any = None
    unsupported: Any = None
    unsupported_label: str = ""
    unsupported_tooltip: str = ""
    replaced: bool = False
    note: Optional[tuple[str, str]] = None
    model_classes: dict = field(default_factory=dict)


def first_bio_option(
    options: list, name: str, radiation_mode: str, available_only: bool = False
) -> Any:
    """First option of a formalism (identifier or alias), available ones preferred.

    With *available_only*, ``None`` if the formalism has no available option.
    """
    from pyRadPlan.biology._editing import bio_model_quietly  # noqa: PLC0415

    entries = [o for o in options if o.model == name]
    if not entries:
        model = bio_model_quietly(name, radiation_mode)
        if model is not None and model.model != name:
            return first_bio_option(options, model.model, radiation_mode, available_only)
        return None
    return next((o for o in entries if o.available), None if available_only else entries[0])


def _spec_name(spec: Any) -> str:
    if spec is None:
        return ""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, Mapping):
        return str(spec.get("model") or "")
    return str(getattr(spec, "model", "") or "")


def _machine_issues(machine: Any, model: Any, engine: Any) -> tuple[list, bool]:
    """Issues of a model on a machine with an engine, and whether the machine alone runs it."""
    from pyRadPlan.biology._editing import tissue_issues_quietly  # noqa: PLC0415

    issues = [i for i in tissue_issues_quietly(machine, model, [], engine) if i.structure is None]
    on_machine = not any(i.structure is None for i in tissue_issues_quietly(machine, model, []))
    return issues, on_machine


def _unsupported_entry(
    bio_model: Any, radiation_mode: str, machine: Any, engine: Any
) -> Optional[tuple[Any, str, str]]:
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        bio_model_quietly,
        selection_parameters,
        tissue_issues_text,
    )
    from pyRadPlan.biology.tissues import provenance_label  # noqa: PLC0415
    from pyRadPlan.quantities._display_names import bio_model_display_name  # noqa: PLC0415

    model = bio_model_quietly(bio_model, radiation_mode)
    if model is None:
        return None
    suffix = []
    for key, value in selection_parameters(model.parameters).items():
        if key == "data_model":
            suffix.append(provenance_label(value))
        elif isinstance(value, dict):
            suffix.extend(str(variant) for variant in value.values())
        else:
            suffix.append(str(value))
    label = " · ".join([bio_model_display_name(model), *suffix])
    issues, on_machine = _machine_issues(machine, model, engine)
    if engine is not None and issues and on_machine:
        subject = f"the dose engine {engine.name}"
        unsupported = f"(not supported by {subject})"
    else:
        subject, unsupported = "the machine", UNSUPPORTED_SUFFIX
    reason = (
        tissue_issues_text(issues)
        if issues
        else "The machine offers no base data matching the configuration of this model."
    )
    tooltip = (
        f"Model of the plan, which {subject} cannot run:\n{reason}\n"
        "Apply keeps it; select another model to replace it."
    )
    return model, f"{label} {unsupported}", tooltip


def _replacement_note(
    previous: Any,
    previous_label: str,
    selected_label: str,
    radiation_mode: str,
    machine: Any,
    engine: Any,
    machine_name: str,
    model_classes: Mapping[str, type],
) -> tuple[str, str]:
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        bio_model_quietly,
        tissue_issues_text,
    )

    model = bio_model_quietly(previous, radiation_mode)
    reason = ""
    if model is None or model.model not in model_classes:
        context = f"for {radiation_mode}"
    else:
        issues, on_machine = _machine_issues(machine, model, engine)
        reason = tissue_issues_text(issues)
        if engine is not None and on_machine:
            context = f"with the dose engine {engine.name}"
        else:
            context = f"on machine '{machine_name}'"
    return (
        f"Biological model '{previous_label}' is not available {context}; "
        f"selected '{selected_label}'.",
        reason,
    )


def select_bio_model(
    radiation_mode: str,
    machine: Any = None,
    engine: Any = None,
    *,
    restore: Any = None,
    current: Any = None,
    current_name: Optional[str] = None,
    current_label: Optional[str] = None,
    previous_mode: Optional[str] = None,
    machine_name: str = "Generic",
) -> BioSelection:
    """List the biological model entries of the plan form and choose the selected one.

    The rules of the plan form (both frontends) after its radiation mode, machine or dose
    engine changed, or when it shows a plan.

    Parameters
    ----------
    radiation_mode : str
        The radiation mode listed.
    machine : BioMachineSummary or Machine, optional
        The machine (summary) the models are judged by; ``None`` if it cannot be loaded
        (the plain formalism list, still judged by the engine).
    engine : type, optional
        The dose engine class the calculation uses.
    restore : BiologicalModel, optional
        Model of a plan to select. Without an available entry matching it, an entry naming
        it as not supported by the machine (or the engine) is added and selected.
    current : optional
        The selection before the change (a model, a ``{"model", **parameters}``
        specification or a name); ``None`` when nothing was listed before.
    current_name : str, optional
        The ``model`` name of that selection (default: derived from *current*).
    current_label : str, optional
        Its label, for the replacement note (default: the name).
    previous_mode : str, optional
        The radiation mode the previous selection was made in.
    machine_name : str
        Name of the machine shown in the replacement note.

    Returns
    -------
    BioSelection
        Without *restore*, a selection the new context can run is kept (or another entry
        of its formalism); otherwise, and for a mode's default followed into a new mode,
        the default of the mode is selected: its first available entry, else ``none``.
    """
    from pyRadPlan.biology._editing import (  # noqa: PLC0415
        bio_options_quietly,
        matching_bio_option,
    )
    from pyRadPlan.biology.models import available_bio_models  # noqa: PLC0415
    from pyRadPlan.plan import default_bio_models  # noqa: PLC0415

    options = bio_options_quietly(radiation_mode, machine, engine)
    classes = {cls.model: cls for cls in available_bio_models(radiation_mode)}
    selection = BioSelection(options=options, model_classes=classes)
    if restore is not None:
        option = matching_bio_option(options, restore, machine)
        if option is not None and option.available:
            selection.selected = option
            return selection
        entry = _unsupported_entry(restore, radiation_mode, machine, engine)
        if entry is not None:
            selection.unsupported, selection.unsupported_label, selection.unsupported_tooltip = (
                entry
            )
        return selection

    had_entries = current is not None
    name = current_name if current_name is not None else _spec_name(current)
    explicit = had_entries and name not in ("", default_bio_models.get(previous_mode))
    # A mode's default follows into the new mode's default; anything else is kept if the
    # new context can run it.
    keep = had_entries and (explicit or previous_mode == radiation_mode)
    if keep and name in classes:
        option = matching_bio_option(options, current, machine)
        if option is not None and option.available:
            selection.selected = option
            return selection
    chosen = first_bio_option(options, name, radiation_mode, True) if keep else None
    if chosen is None:
        default = default_bio_models.get(radiation_mode, "none")
        chosen = first_bio_option(options, default, radiation_mode, True)
    if chosen is None:
        chosen = first_bio_option(options, "none", radiation_mode)
    if chosen is None and options:
        chosen = options[0]
    selection.selected = chosen
    selection.replaced = True
    if keep:
        selection.note = _replacement_note(
            current,
            current_label or name,
            chosen.label if chosen is not None else "",
            radiation_mode,
            machine,
            engine,
            machine_name,
            classes,
        )
    return selection


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


def plan_engine_config(pln: Any) -> tuple[Optional[str], dict[str, Any]]:
    """Return the engine a plan's form shows (its own, else the default) and its values.

    The values are the plan's ``prop_dose_calc`` entries besides ``engine`` and
    ``dose_grid`` (as stored, not copied).
    """
    prop = pln.prop_dose_calc if isinstance(pln.prop_dose_calc, dict) else {}
    engine = _engine_name(pln)
    if engine is None:
        try:
            engine = _engine_class(pln.radiation_mode, None).short_name
        except Exception:  # noqa: BLE001 - a mode without engines shows none
            engine = None
    return engine, {k: v for k, v in prop.items() if k not in ("engine", "dose_grid")}


def split_solver_spec(spec: Any) -> tuple[Optional[str], dict[str, Any]]:
    """Solver name and configuration of a ``prop_opt["solver"]`` entry."""
    from pyRadPlan.optimization.solvers import SolverBase  # noqa: PLC0415

    if spec is None:
        return None, {}
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, SolverBase):
        # Hidden parameters such as keyboard cancellation still belong to the
        # configuration and must survive replacing an instance with a dictionary.
        names = spec.config_model().model_fields
        return spec.short_name, {n: getattr(spec, n) for n in names if hasattr(spec, n)}
    if isinstance(spec, Mapping):
        values = dict(spec)
        return values.pop("name", None), values
    return None, {}


_solver_spec = split_solver_spec


def solver_change(spec: Any) -> Any:
    """``plan.set`` ``solver`` param of a ``prop_opt["solver"]`` entry (``None``: default)."""
    if spec is None or isinstance(spec, str):
        return spec
    name, values = split_solver_spec(spec)
    if name is None:
        return None
    return {"name": name, "values": values} if values else name


def optimization_change(problem: Optional[str], prop_opt: Mapping[str, Any]) -> dict:
    """``plan.set`` ``problem`` and ``solver`` params of a problem's ``prop_opt`` entries."""
    values = {k: v for k, v in prop_opt.items() if k not in ("problem", "opti_prob")}
    solver = values.pop("solver", None)
    return {"problem": {"name": problem, "values": values}, "solver": solver_change(solver)}


def bio_change(model: Optional[str], option: Any, parameters: Mapping[str, Any]) -> dict:
    """``plan.set`` ``bio_model`` param: formalism, base-data option and other parameters.

    *option* is the selected ``BioModelOption`` (``None``: none); its base-data selection
    replaces the selection parameters among *parameters*.
    """
    from pyRadPlan.biology._editing import SELECTION_PARAMETERS  # noqa: PLC0415

    return {
        "model": model or "none",
        "option": option_id(option) if option is not None else None,
        "values": {k: v for k, v in parameters.items() if k not in SELECTION_PARAMETERS},
    }


def _bio_spec(change: Any) -> Any:
    if isinstance(change, Mapping) and set(change) <= {"model", "option", "values"}:
        option = change.get("option")
        params = json.loads(option).get("parameters", {}) if option else {}
        return {
            "model": change.get("model") or "none",
            **params,
            **dict(change.get("values") or {}),
        }
    model = getattr(change, "model", None)
    if model is not None:
        return {"model": model, **dict(getattr(change, "parameters", {}) or {})}
    return {"model": change or "none"}


def plan_form_values(pln: Any) -> Optional[dict[str, Any]]:
    """Return the plan as complete ``plan.set`` params: what an unedited form sends.

    A plan without a dose engine shows the default engine of its radiation mode; a plan
    carrying its machine shows ``"__plan__"``.
    """
    if pln is None:
        return None
    summary = plan_summary(pln)
    mode = pln.radiation_mode
    engine, engine_values = plan_engine_config(pln)
    model = getattr(pln.scenarios, "short_name", None)
    problem, values = _problem_of(pln, mode)
    return {
        **{
            key: summary[key]
            for key in (
                "radiation_mode",
                "num_of_fractions",
                "gantry_angles",
                "couch_angles",
                "bixel_width",
                "iso_center",
                "dose_grid_resolution",
            )
        },
        "machine": summary["machine"] if isinstance(pln.machine, str) else "__plan__",
        "dose_convention": pln.dose_convention,
        "engine": engine,
        "engine_config": {"values": engine_form_values(engine_values)},
        "bio_model": pln.bio_model,
        "scenarios": {
            "model": model,
            "values": {} if model == "explicit" else _scenario_overrides(pln.scenarios),
        },
        **optimization_change(problem, values),
        "sequencing": sequencing_change(pln),
    }


def sequencing_change(pln: Any) -> dict[str, Any]:
    """``plan.set`` ``sequencing`` param of a plan: ``run``, ``sequencer`` and its ``values``.

    A plan that does not ask for sequencing has no sequencer here, as its form shows none.
    """
    run, name, values = sequencing_settings(pln)
    if not run:
        return {"run": False, "sequencer": None, "values": {}}
    return {"run": True, "sequencer": name, "values": deepcopy(values)}


def _prop_seq(pln: Any, mode: str, same_mode: bool, sequencing: Mapping[str, Any]) -> dict:
    from pyRadPlan.sequencing.sequencer import get_available_sequencers  # noqa: PLC0415

    def invalid(message: str) -> FormValidationError:
        return FormValidationError(message, [{"field": "sequencing", "message": message}])

    unknown = sorted(set(sequencing) - {"run", "sequencer", "values"})
    if unknown:
        raise invalid(f"sequencing takes run, sequencer and values, not {unknown}")
    sequencers = get_available_sequencers(mode)
    # Only the entries the form edits are replaced; others are kept.
    prop_seq = deepcopy(dict(pln.prop_seq or {})) if same_mode else {}
    edited = {"run_sequencing", "sequencer"}.union(
        *(cls.config_model().model_fields for cls in sequencers.values())
    )
    for key in edited:
        prop_seq.pop(key, None)
    if not sequencing.get("run"):
        return prop_seq
    name = sequencing.get("sequencer")
    if name not in sequencers:
        raise invalid(f"sequencer {name!r} is not available for {mode}")
    values = dict(sequencing.get("values") or {})
    if values:
        values = _check_form("sequencer", name, mode, values)
    prop_seq.update(run_sequencing=True, sequencer=name, **values)
    return prop_seq


#: ``plan.set`` params sent together: couch angles follow the gantry angles, an engine's
#: configuration its engine, the solver its problem.
_COUPLED = (("gantry_angles", "couch_angles"), ("engine", "engine_config"), ("problem", "solver"))


def changes_from_form(pln: Any, form: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ``plan.set`` params turning *pln* into the plan a form shows.

    *form* holds complete params (every field the form shows). Without a plan or for
    another radiation mode they are all sent; otherwise only those differing from the
    plan (:func:`plan_form_values`), with :data:`_COUPLED` params sent together. Plan
    settings the form shows as the plan has them but cannot represent (a live solver,
    ``prop_opt`` entries outside the problem form, the plan's own scenario model
    instance) therefore survive an Apply.
    """
    if pln is None or form.get("radiation_mode", pln.radiation_mode) != pln.radiation_mode:
        return dict(form)
    current = plan_form_values(pln)
    changed = set()
    for key, value in form.items():
        if key not in current:
            changed.add(key)
        elif key == "bio_model":
            if comparison_key(_bio_spec(value)) != comparison_key(_bio_spec(current[key])):
                changed.add(key)
        elif comparison_key(value) != comparison_key(current[key]):
            changed.add(key)
    for group in _COUPLED:
        if changed & set(group):
            changed.update(k for k in group if k in form)
    return {key: value for key, value in form.items() if key in changed}


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


def plan_problem(pln: Any) -> tuple[Optional[str], dict[str, Any]]:
    """Return the problem a plan's form shows and its ``prop_opt`` entries (with solver).

    A problem the plan names but the radiation mode cannot use falls back to the default,
    as ``get_problem_from_pln`` does; the entries are copied shallowly.
    """
    return _problem_of(pln, pln.radiation_mode)


def problem_classes(radiation_mode: str) -> dict[str, type]:
    """Planning problem classes that can optimize a radiation mode, by short name."""
    return _problems(radiation_mode)


def engine_class(radiation_mode: str, name: Optional[str]) -> type:
    """Dose engine class of a radiation mode (``None``: the default engine)."""
    return _engine_class(radiation_mode, name)


def _problem_of(pln: Any, radiation_mode: str) -> tuple[Optional[str], dict[str, Any]]:
    prop_opt = pln.prop_opt if pln is not None and isinstance(pln.prop_opt, dict) else {}
    problems = _problems(radiation_mode)
    requested = prop_opt.get("problem") or prop_opt.get("opti_prob")
    default = next(iter(problems), None)
    name = requested if requested in problems else default
    values = {k: v for k, v in prop_opt.items() if k not in ("problem", "opti_prob")}
    return name, values


def scenario_overrides(model: Any) -> dict:
    """Parameters set on a scenario model, nested per uncertainty dimension.

    These are the values of the Qt scenario configuration popup; dimensions without a
    parameter of their own are dropped.
    """
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


_scenario_overrides = scenario_overrides


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
    problem_cls = _problems(mode).get(problem) if problem else None
    if problem_cls is not None:
        # Entries the problem form does not show stay in the plan unless sent.
        shown = set(problem_cls.config_model().model_fields) - PROBLEM_FORM_EXCLUDE
        problem_values = {k: v for k, v in problem_values.items() if k in shown}
    return {
        **summary,
        "dose_convention": pln.dose_convention,
        "prescribed_dose": float(pln.prescribed_dose),
        "machine_source": "name" if isinstance(pln.machine, str) else "plan",
        "resolved_iso_center": resolved,
        "engine_config": {
            "name": _engine_name(pln),
            "values": jsonable(
                engine_form_values({k: v for k, v in prop_dose_calc.items() if k != "engine"})
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
        "sequencing": jsonable(sequencing_change(pln)),
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
        field("run_dao", "Run DAO", "bool", default=False, **off),
        field("conformal", "3D conformal", "bool", default=False, **off),
    ]


def plan_machine_label(machine: Any) -> str:
    """Label of the machine entry of the base data a plan carries itself."""
    return f"{getattr(machine, 'name', None) or 'machine'} (from plan)"


def machine_choices(radiation_mode: str, plan_machine: Any = None) -> list[dict]:
    """Machines of a radiation mode, led by *plan_machine* (``"__plan__"``) when it is one."""
    from pyRadPlan.machines import get_available_machines  # noqa: PLC0415

    names = get_available_machines(radiation_mode).get(radiation_mode, [])
    choices = [_choice(name, name, name) for name in names]
    if plan_machine is not None and not isinstance(plan_machine, str):
        if getattr(plan_machine, "radiation_mode", radiation_mode) == radiation_mode:
            choices.insert(0, _choice("__plan__", plan_machine_label(plan_machine), None))
    return choices


def engine_choices(radiation_mode: str) -> list[dict]:
    """Dose engines of a radiation mode; Monte Carlo engines disabled under Pyodide."""
    from pyRadPlan.dose.engines import get_available_engines  # noqa: PLC0415

    from ._profiles import running_in_pyodide  # noqa: PLC0415

    engines = []
    for name, cls in get_available_engines(radiation_mode).items():
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
    return engines


def sequencer_choices(radiation_mode: str) -> list[dict]:
    """Sequencers of a radiation mode (none for the modes without sequencing)."""
    from pyRadPlan.sequencing.sequencer import get_available_sequencers  # noqa: PLC0415

    return [
        _choice(name, getattr(cls, "name", name) or name, name)
        for name, cls in get_available_sequencers(radiation_mode).items()
    ]


def scenario_choices(explicit: bool = False) -> list[dict]:
    """Scenario models; the explicit one only for a plan holding an explicit scenario set."""
    from pyRadPlan.scenarios import (  # noqa: PLC0415
        available_scenario_models,
        get_scenario_model_class,
    )

    choices = []
    for name in available_scenario_models():
        cls = get_scenario_model_class(name)
        choices.append(_choice(name, getattr(cls, "name", name) or name, name))
    if explicit:
        choices.append(_choice("explicit", "Explicit Scenarios", "explicit"))
    return choices


def problem_choices(radiation_mode: str) -> list[dict]:
    """Planning problems that can optimize a radiation mode."""
    return [
        _choice(name, getattr(cls, "name", name) or name, name)
        for name, cls in _problems(radiation_mode).items()
    ]


def solver_choices() -> list[dict]:
    """List the registered solvers, plus ``ipopt`` disabled with the reason when missing."""
    from pyRadPlan.optimization.solvers import (  # noqa: PLC0415
        IPOPT_DISABLED_REASON,
        get_available_solvers,
    )

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
    return solvers


def beam_presets(radiation_mode: str) -> list:
    """Beam angle presets, those typical for the radiation mode first."""
    from pyRadPlan.stf import get_beam_angle_presets  # noqa: PLC0415

    return list(get_beam_angle_presets(radiation_mode))


def plan_options(
    pln: Any,
    default_plan: Mapping[str, Any],
    locked: set[str],
    radiation_mode: Optional[str] = None,
    machine: Optional[str] = None,
    engine: Optional[str] = None,
    current_bio: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """``PlanOptions`` for a radiation mode, machine and engine (defaults: the plan's).

    *current_bio* is the form's biological model selection before the change
    (``{radiation_mode, model, option, values?, label?}``); ``bio_selection`` is then the
    entry the form selects for the new context (:func:`select_bio_model`). Without it, the
    plan's model is restored for the plan's radiation mode, else the mode's default is
    selected.
    """
    from pyRadPlan.biology._editing import dose_engine_quietly  # noqa: PLC0415

    plan_mode = pln.radiation_mode if pln is not None else default_plan.get("radiation_mode")
    mode = radiation_mode or plan_mode or "photons"
    same_mode = pln is not None and mode == pln.radiation_mode
    machine_value: Any = machine
    if machine_value is None:
        machine_value = pln.machine if same_mode else default_plan.get("machine", "Generic")
    elif machine_value == "__plan__" and same_mode and not isinstance(pln.machine, str):
        machine_value = pln.machine
    engine_name = engine if engine is not None else (_engine_name(pln) if same_mode else None)
    machine_obj = _machine_of(mode, machine_value)
    engine_cls = dose_engine_quietly(mode, engine_name)
    selection = _form_bio_selection(
        pln, default_plan, mode, machine_value, machine_obj, engine_cls, current_bio
    )
    options = selection.options
    first_problem = next(iter(_problems(mode).values()), None)
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
        for preset in beam_presets(mode)
    ]
    explicit = same_mode and getattr(pln.scenarios, "short_name", None) == "explicit"
    return {
        "radiation_mode": mode,
        "radiation_modes": [_choice(m, m) for m in RADIATION_MODES],
        "machines": machine_choices(mode, pln.machine if same_mode else None),
        "engines": engine_choices(mode),
        "bio_models": _bio_choices(options, mode),
        "default_bio_model": _default_option(mode, options),
        "bio_selection": _bio_selection_dict(selection),
        "scenario_models": scenario_choices(explicit),
        "problems": problem_choices(mode),
        "solvers": solver_choices(),
        "sequencers": sequencer_choices(mode),
        "default_solver": default_solver(first_problem) if first_problem else None,
        "beam_presets": presets,
        "fields": _plan_fields(locked, default_plan),
    }


def _current_bio_spec(pln: Any, current: Mapping[str, Any]) -> Any:
    """Model specification of a ``current_bio`` selection (the plan's for its kept entry)."""
    model = current.get("model") or "none"
    values = dict(current.get("values") or {})
    option = current.get("option")
    if option:
        return {"model": model, **json.loads(option).get("parameters", {}), **values}
    if pln is not None and _spec_name(pln.bio_model) == model:
        return pln.bio_model
    return {"model": model, **values} if values else model


def _form_bio_selection(
    pln: Any,
    default_plan: Mapping[str, Any],
    mode: str,
    machine_value: Any,
    machine: Any,
    engine: Any,
    current: Optional[Mapping[str, Any]],
) -> BioSelection:
    if isinstance(machine_value, str):
        machine_name = machine_value or "Generic"
    else:
        machine_name = plan_machine_label(machine_value)
    if current is not None:
        return select_bio_model(
            mode,
            machine,
            engine,
            current=_current_bio_spec(pln, current),
            current_name=current.get("model") or "none",
            current_label=current.get("label"),
            previous_mode=current.get("radiation_mode"),
            machine_name=machine_name,
        )
    if pln is not None and pln.radiation_mode == mode:
        return select_bio_model(mode, machine, engine, restore=pln.bio_model)
    selection = select_bio_model(mode, machine, engine, machine_name=machine_name)
    preferred = default_plan.get("bio_model")
    if pln is None and isinstance(preferred, str) and default_plan.get("radiation_mode") == mode:
        option = first_bio_option(selection.options, preferred, mode)
        if option is not None:
            selection.selected = option
    return selection


def _bio_selection_dict(selection: BioSelection) -> dict[str, Any]:
    selected = selection.selected
    unsupported = selection.unsupported
    if selected is not None:
        model = selected.model
    else:
        model = getattr(unsupported, "model", None) or "none"
    return {
        "option": option_id(selected) if selected is not None else None,
        "model": model,
        "replaced": selection.replaced,
        "unsupported": (
            {
                "model": unsupported.model,
                "label": selection.unsupported_label,
                "description": selection.unsupported_tooltip,
            }
            if unsupported is not None
            else None
        ),
        "note": (
            {"text": selection.note[0], "reason": selection.note[1]}
            if selection.note is not None
            else None
        ),
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


def solver_defaults(
    solver_cls: type, on_error: Optional[Callable[[Exception], None]] = None
) -> dict[str, Any]:
    """Effective defaults of a solver, which most solvers only set in ``__init__``.

    Empty (the form then shows the model defaults) when the solver cannot be
    instantiated; *on_error* is called with the exception in that case.
    """
    from pyRadPlan.core._configurable import field_constraints  # noqa: PLC0415

    try:
        solver = solver_cls()
    except Exception as exc:  # noqa: BLE001 - the form then shows the model defaults
        if on_error is not None:
            on_error(exc)
        return {}
    model = solver_cls.config_model()
    names = [
        n
        for n, info in model.model_fields.items()
        if getattr(field_constraints(info).get("param_meta"), "configurable", True)
    ]
    return {n: getattr(solver, n) for n in names if hasattr(solver, n)}


_solver_defaults = solver_defaults


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
    if kind == "sequencer":
        from pyRadPlan.sequencing.sequencer import get_available_sequencers  # noqa: PLC0415

        cls = get_available_sequencers(mode).get(name)
        if cls is None:
            raise ValueError(f"No sequencer {name!r} for {mode}")
        return cls.config_model(), f"Configure {cls.name}", frozenset(), {}
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
    if kind == "sequencer":
        _, sequencer, sequencer_values = sequencing_settings(pln)
        return dict(sequencer_values) if sequencer == name else {}
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
        values = dict(engine_config.get("values") or {})
        # Entries the engine form does not show are kept unless sent (sent back as the
        # plan has them, they are kept as well).
        hidden = _engine_hidden(pln, mode, cls)
        for key in [k for k in values if k in hidden]:
            if comparison_key(values[key]) == comparison_key(hidden[key]):
                del values[key]
        overrides["engine_props"] = {
            **hidden,
            **_check_form("engine", cls.short_name, mode, values),
        }

    bio = changes.get("bio_model")
    if isinstance(bio, Mapping):
        unknown = sorted(set(bio) - {"model", "option", "values"})
        if unknown:
            # A plain specification ({"model", **parameters}) would lose its parameters here.
            message = f"bio_model takes model, option and values, not {unknown}"
            raise FormValidationError(message, [{"field": "bio_model", "message": message}])
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
    if scenarios is None and pln is not None and not same_mode:
        scenarios = _form_scenarios(pln.scenarios)
    if scenarios is not None:
        model = scenarios.get("model") if isinstance(scenarios, Mapping) else scenarios
        values = dict(scenarios.get("values") or {}) if isinstance(scenarios, Mapping) else {}
        current = pln.scenarios if pln is not None else None
        # The explicit model cannot be built from its name: its set is kept, also for
        # another radiation mode.
        if (
            current is not None
            and getattr(current, "short_name", None) == model
            and (
                model == "explicit"
                or (
                    same_mode
                    and comparison_key(_scenario_overrides(current)) == comparison_key(values)
                )
            )
        ):
            overrides["scenarios"] = deepcopy(current)
        else:
            if values:
                _check_form("scenario", model, mode, values)
            overrides["scenarios"] = {"model": model, **values}

    sequencing = changes.pop("sequencing", None)
    if sequencing is not None:
        overrides["prop_seq"] = _prop_seq(pln, mode, same_mode, sequencing)

    problem = changes.pop("problem", None)
    solver = changes.pop("solver", None)
    if problem is not None or solver is not None or (pln is not None and not same_mode):
        overrides["prop_opt"] = _prop_opt(pln, mode, same_mode, problem, solver)
    return changes, overrides


def engine_form_values(values: Mapping[str, Any]) -> dict[str, Any]:
    """Return the engine settings the engine form shows (without :data:`ENGINE_FORM_EXCLUDE`)."""
    return {k: v for k, v in values.items() if k not in ENGINE_FORM_EXCLUDE}


def _engine_hidden(pln: Any, mode: str, engine_cls: type) -> dict[str, Any]:
    """Return the settings of a plan's engine the engine form does not show, ``dose_grid`` aside."""
    if pln is None or pln.radiation_mode != mode or not isinstance(pln.prop_dose_calc, dict):
        return {}
    try:
        current = _engine_class(mode, _engine_name(pln))
    except Exception:  # noqa: BLE001 - a plan without a usable engine keeps nothing
        return {}
    if current.short_name != engine_cls.short_name:
        return {}
    return {
        k: v
        for k, v in pln.prop_dose_calc.items()
        if k in ENGINE_FORM_EXCLUDE and k != "dose_grid"
    }


def _form_scenarios(model: Any) -> Optional[dict[str, Any]]:
    """``plan.set`` ``scenarios`` param showing a scenario model (``None`` without one)."""
    name = getattr(model, "short_name", None)
    if name is None:
        return None
    return {"model": name, "values": {} if name == "explicit" else _scenario_overrides(model)}


def _carried_problem(pln: Any, mode: str) -> tuple[Optional[str], dict[str, Any]]:
    """Problem and ``prop_opt`` entries a plan keeps when its radiation mode changes.

    The plan's problem with its entries if the new mode offers it, else none: the new
    mode's default problem then starts from its defaults.
    """
    if pln is None:
        return None, {}
    name, values = _problem_of(pln, pln.radiation_mode)
    if name is None or name not in _problems(mode):
        return None, {}
    return name, values


def _prop_opt(pln: Any, mode: str, same_mode: bool, problem: Any, solver: Any) -> dict:
    problems = _problems(mode)
    if same_mode:
        current_name, current_values = _problem_of(pln, mode)
    else:
        current_name, current_values = _carried_problem(pln, mode)
    prop_opt_named = bool(
        current_name is not None and isinstance(pln.prop_opt, dict) and pln.prop_opt.get("problem")
    )
    if isinstance(problem, Mapping):
        name = problem.get("name") or current_name or next(iter(problems), None)
        values = dict(problem.get("values") or {})
        if name == current_name and name in problems:
            # Entries the form does not show, sent back as the plan has them, are kept.
            form_fields = set(problems[name].config_model().model_fields) - (
                PROBLEM_FORM_EXCLUDE | set(problems[name].solver_properties)
            )
            for key in [k for k in values if k not in form_fields and k in current_values]:
                if key != "solver" and comparison_key(values[key]) == comparison_key(
                    current_values[key]
                ):
                    del values[key]
        problem_values = _check_form("problem", name, mode, values) if values else {}
        kept_solver = current_values.get("solver") if name == current_name else None
        forwarded = {
            k: current_values[k]
            for k in getattr(problems.get(name), "solver_properties", ())
            if k in current_values and name == current_name
        }
        if name == current_name and name in problems:
            # Entries the problem form does not show are kept unless sent (Qt dialog).
            shown = set(problems[name].config_model().model_fields) - PROBLEM_FORM_EXCLUDE
            forwarded.update(
                {
                    k: v
                    for k, v in current_values.items()
                    if k != "solver" and k not in shown and k not in values
                }
            )
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

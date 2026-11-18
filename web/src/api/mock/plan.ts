/**
 * Mock handlers of package A (PROTOCOL.md v1: plan detail / options / forms, tissues,
 * objectives, optimization history). Per-session data lives in `MockSession.ext.plan`.
 */
import type {
  AppInfo,
  BackendEvent,
  FieldDescriptor,
  Issue,
  ObjectiveData,
  OptimizationHistory,
  PlanSummary,
  Profile,
  SessionOptions,
  SessionState,
  StructureDetail,
  TissueDraft,
  ValidationDetail,
  Vec3,
} from "../protocol";
import type {
  BioModelChoice,
  BioSelection,
  CurrentBio,
  FormKind,
  PlanDetail,
  PlanOptions,
  PlanSetParams,
  TissueComparison,
  TissueEvalRow,
  TissueEvaluateParams,
  TissueSource,
} from "../protocol/plan";
import { MethodError, precondition, type MockContext, type MockHandlers, type MockSession } from "./mockBackend";
import {
  BIO_OPTIONS,
  DEFAULT_BIO,
  DEFAULT_SOLVER,
  ENGINES,
  IPOPT_REASON,
  MACHINES,
  MC_REASON,
  OBJECTIVES,
  PROBLEMS,
  QUANTITIES,
  RADIATION_MODES,
  REFERENCE_RADIATIONS,
  SCENARIO_MODELS,
  SOLVERS,
  TISSUE_CLASSES,
  TISSUE_MODELS,
  beamPresetsFor,
  bioOptionsFor,
  defaultEngine,
  enginesFor,
  formSchema,
  planFields,
  tissuePicks,
} from "./planData";

// ---------------------------------------------------------------------------
// Session data
// ---------------------------------------------------------------------------

export interface MockStructure {
  name: string;
  type: string;
  color: Vec3;
  overlap_priority: number;
  num_voxels: number;
  center_of_mass: Vec3 | null;
  tissue: TissueDraft;
  objectives: Omit<ObjectiveData, "index">[];
}

interface PlanExt {
  profile: Profile;
  lockedFields: string[];
  doseConvention: "per_fraction" | "total";
  engineValues: Record<string, Record<string, unknown>>;
  bio: { option: string | null; values: Record<string, unknown> };
  scenario: { model: string; values: Record<string, unknown> };
  problem: { name: string; values: Record<string, unknown> };
  solver: { name: string | null; values: Record<string, unknown> };
  structures: MockStructure[];
  /** Phantom the structures were built from (rebuilt after a reload). */
  phantom: unknown;
  history: OptimizationHistory | null;
}

const WEB_GRID: Vec3 = [5, 5, 5];

function ext(s: MockSession): PlanExt {
  let e = s.ext.plan as PlanExt | undefined;
  if (!e) {
    e = {
      profile: "web",
      lockedFields: [],
      doseConvention: "total",
      engineValues: {},
      bio: { option: "none", values: {} },
      scenario: { model: "nomScen", values: {} },
      problem: { name: "nonlin_fluence", values: {} },
      solver: { name: null, values: {} },
      structures: [],
      phantom: null,
      history: null,
    };
    s.ext.plan = e;
  }
  if (s.phantom && e.phantom !== s.phantom) {
    e.phantom = s.phantom;
    e.structures = buildStructures(s);
  }
  if (!s.phantom) e.structures = [];
  return e;
}

const DEFAULT_OBJECTIVE: Record<string, string> = {
  TARGET: "Squared Deviation",
  OAR: "Squared Overdosing",
  EXTERNAL: "Squared Overdosing",
  HELPER: "Squared Overdosing",
};

function buildStructures(s: MockSession): MockStructure[] {
  const phantom = s.phantom;
  if (!phantom) return [];
  const [nz, ny, nx] = phantom.grid.shape;
  const g = phantom.grid;
  return phantom.structures.map((st, index) => {
    let n = 0;
    let cx = 0;
    let cy = 0;
    let cz = 0;
    let i = 0;
    const m = st.mask;
    for (let k = 0; k < nz; k++)
      for (let j = 0; j < ny; j++)
        for (let c = 0; c < nx; c++, i++)
          if (m[i]) {
            n++;
            cx += c;
            cy += j;
            cz += k;
          }
    const com: Vec3 | null = n
      ? [g.origin[0] + (cx / n) * g.spacing[0], g.origin[1] + (cy / n) * g.spacing[1], g.origin[2] + (cz / n) * g.spacing[2]]
      : null;
    const objective = (name: string, priority: number, parameters: Record<string, unknown>) => ({
      name,
      priority,
      quantity: "physical_dose",
      robustness: "none",
      parameters,
    });
    const objectives =
      st.type === "TARGET"
        ? [objective("Squared Deviation", 800, { d_ref: 60 })]
        : st.type === "OAR"
          ? [objective("Squared Overdosing", 300, { d_max: 30 }), ...(index === 0 ? [objective("Max DVH", 50, { d: 20, v_max: 40 })] : [])]
          : [objective("Squared Overdosing", 100, { d_max: 30 })];
    return {
      name: st.name,
      type: st.type,
      color: st.color,
      overlap_priority: st.type === "TARGET" ? 1 : st.type === "EXTERNAL" ? 3 : 2,
      num_voxels: n,
      center_of_mass: com,
      tissue: { model: "Generic", values: { alpha_x: 0.1, beta_x: 0.05 }, sources: {} },
      objectives,
    };
  });
}

/** Structures of a session as `StructureDetail` (for other packages' mock handlers too). */
export function mockStructures(s: MockSession): MockStructure[] {
  return ext(s).structures;
}

export function structureDetail(st: MockStructure, index: number): StructureDetail {
  const ax = Number(st.tissue.values.alpha_x ?? 0.1);
  const bx = Number(st.tissue.values.beta_x ?? 0.05);
  const params = Object.entries(st.tissue.values)
    .filter(([, v]) => v !== null && v !== undefined)
    .map(([k, v]) => `${k} = ${v}`)
    .join(", ");
  return {
    name: st.name,
    index,
    type: st.type,
    color: st.color,
    visible: true,
    overlap_priority: st.overlap_priority,
    num_voxels: st.num_voxels,
    num_scenarios: 1,
    center_of_mass: st.center_of_mass,
    alpha_x: ax,
    beta_x: bx,
    tissue: structuredClone(st.tissue),
    tissue_summary: `${st.tissue.model}: ${params}`,
    objectives: st.objectives.map((o, i) => ({ index: i, ...structuredClone(o) })),
  };
}

export function planStateExtension(s: MockSession, state: SessionState): Partial<SessionState> {
  const e = ext(s);
  const byName = new Map(e.structures.map((st) => [st.name, st]));
  const options: SessionOptions = {
    profile: e.profile,
    default_plan: { ...defaultPlanFor(e.profile) },
    locked_fields: [...e.lockedFields],
  };
  return {
    options,
    structures: state.structures.map((st) => {
      const m = byName.get(st.name);
      return m ? { ...st, overlap_priority: m.overlap_priority, num_objectives: m.objectives.length, num_voxels: m.num_voxels } : st;
    }),
  };
}

function defaultPlanFor(profile: Profile): PlanSummary {
  return {
    radiation_mode: "photons",
    machine: "Generic",
    num_of_fractions: 30,
    gantry_angles: [0],
    couch_angles: [0],
    bixel_width: 5,
    iso_center: null,
    dose_grid_resolution: profile === "web" ? [...WEB_GRID] : [3, 3, 3],
    engine: "SVDPB",
    bio_model: "none",
  };
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function validationError(details: ValidationDetail[]): MethodError {
  return new MethodError("ValidationError", details.map((d) => `${d.field}: ${d.message}`).join("; "), "invalid", details);
}

function flatFields(fields: FieldDescriptor[]): FieldDescriptor[] {
  return fields.flatMap((f) => (f.type === "group" ? flatFields(f.fields ?? []) : [f]));
}

function getPath(values: Record<string, unknown>, path: string): unknown {
  let node: unknown = values;
  for (const key of path.split(".")) {
    if (!node || typeof node !== "object") return undefined;
    node = (node as Record<string, unknown>)[key];
  }
  return node;
}

/** Bounds / type check of a value against its descriptor (null = valid). */
export function checkValue(f: FieldDescriptor, value: unknown): string | null {
  if (value === null || value === undefined) return f.nullable || value === undefined ? null : "Input should not be None";
  const num = (v: unknown): string | null => {
    if (typeof v !== "number" || !Number.isFinite(v)) return "Input should be a valid number";
    if ((f.type === "int" || f.item_type === "int") && !Number.isInteger(v)) return "Input should be a valid integer";
    if (f.min !== undefined && v < f.min) return `Input should be greater than or equal to ${f.min}`;
    if (f.max !== undefined && v > f.max) return `Input should be less than or equal to ${f.max}`;
    if (f.exclusive_min !== undefined && v <= f.exclusive_min) return `Input should be greater than ${f.exclusive_min}`;
    if (f.exclusive_max !== undefined && v >= f.exclusive_max) return `Input should be less than ${f.exclusive_max}`;
    return null;
  };
  switch (f.type) {
    case "int":
    case "float":
      return num(value);
    case "bool":
      return typeof value === "boolean" ? null : "Input should be a valid boolean";
    case "number_list": {
      if (!Array.isArray(value)) return "Input should be a valid list";
      if (f.length !== undefined && value.length !== f.length) return `List should have ${f.length} items`;
      for (const v of value) {
        const err = num(v);
        if (err) return err;
      }
      return null;
    }
    case "choice":
      if (f.editable_choice) return null;
      return f.choices?.some((c) => c.value === value) ? null : `Input should be ${f.choices?.map((c) => `'${c.value}'`).join(", ")}`;
    default:
      return null;
  }
}

function validateValues(fields: FieldDescriptor[], values: Record<string, unknown>, prefix: string): ValidationDetail[] {
  const out: ValidationDetail[] = [];
  for (const f of flatFields(fields)) {
    const value = getPath(values, f.name);
    if (value === undefined) continue;
    const message = checkValue(f, value);
    if (message) out.push({ field: `${prefix}${f.name}`, message });
  }
  return out;
}

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

function engineDef(id: string | null) {
  return ENGINES.find((e) => e.id === id);
}

function bioOptionOf(id: string | null) {
  return BIO_OPTIONS.find((o) => o.id === id);
}

function planDetail(s: MockSession): PlanDetail | null {
  if (!s.plan) return null;
  const e = ext(s);
  syncBio(s);
  const option = bioOptionOf(e.bio.option);
  const target = e.structures.find((st) => st.type === "TARGET");
  return {
    ...s.plan,
    dose_convention: e.doseConvention,
    prescribed_dose: 60,
    machine_source: "name",
    resolved_iso_center: s.plan.iso_center ? null : (target?.center_of_mass ?? null),
    engine_config: { name: s.plan.engine, values: structuredClone(e.engineValues[s.plan.engine ?? ""] ?? {}) },
    bio: { model: option?.model ?? s.plan.bio_model ?? "none", option: e.bio.option, values: structuredClone(e.bio.values) },
    scenario: structuredClone(e.scenario),
    problem: { name: e.problem.name, values: structuredClone(shownProblemValues(e.problem.name, e.problem.values, true)) },
    solver: structuredClone(e.solver),
  };
}

/** Problem settings the problem form (or, for `max_iter` / `display`, the solver form) shows, or (`shown: false`) the others. */
function shownProblemValues(name: string, values: Record<string, unknown>, shown: boolean): Record<string, unknown> {
  const fields = new Set([...flatFields(PROBLEMS.find((pr) => pr.id === name)?.fields ?? []).map((f) => f.name.split(".")[0]), "max_iter", "display"]);
  return Object.fromEntries(Object.entries(values).filter(([k]) => fields.has(k) === shown));
}

/** Follow bio model changes made by the core (e.g. `physical_only`). */
function syncBio(s: MockSession): void {
  const e = ext(s);
  const model = s.plan?.bio_model ?? "none";
  const option = bioOptionOf(e.bio.option);
  if (option?.model !== model) {
    const match = BIO_OPTIONS.find((o) => o.model === model && o.modes.includes(s.plan?.radiation_mode ?? ""));
    e.bio = { option: match?.id ?? null, values: match ? structuredClone(match.parameters) : {} };
  }
}

/**
 * Simplified `bio_selection` (the real rules are `pyRadPlan.app._planform.select_bio_model`):
 * the plan's model for its mode (an unsupported entry when disabled); after a change, a
 * selection the context can run is kept unless it was the previous mode's default, else the
 * mode's default, else `none`.
 */
function mockBioSelection(s: MockSession, mode: string, machine: string, list: BioModelChoice[], current: CurrentBio | undefined): BioSelection {
  const enabled = (o: BioModelChoice) => o.enabled !== false;
  const pick = (o: BioModelChoice | undefined, replaced: boolean, note: BioSelection["note"] = null): BioSelection => ({
    option: o ? String(o.value) : null,
    model: o?.model ?? "none",
    replaced,
    unsupported: null,
    note,
  });
  const fallback = () => list.find((o) => o.value === DEFAULT_BIO[mode] && enabled(o)) ?? list.find((o) => o.model === "none") ?? list[0];
  if (!current) {
    if (s.plan && s.plan.radiation_mode === mode) {
      const option = list.find((o) => o.value === ext(s).bio.option);
      if (option && enabled(option)) return pick(option, false);
      const model = s.plan.bio_model ?? "none";
      return {
        option: null,
        model,
        replaced: false,
        unsupported: { model, label: `${option?.label ?? model} (not supported by this machine)`, description: `Model of the plan, which the machine cannot run:\n${option?.reason ?? ""}\nApply keeps it; select another model to replace it.` },
        note: null,
      };
    }
    return pick(fallback(), true);
  }
  const keep = current.radiation_mode === mode || current.model !== (bioOptionOf(DEFAULT_BIO[current.radiation_mode] ?? "none")?.model ?? "none");
  const same = list.find((o) => o.value === current.option);
  if (keep && same && enabled(same)) return pick(same, false);
  const chosen = (keep ? list.find((o) => o.model === current.model && enabled(o)) : undefined) ?? fallback();
  if (!keep) return pick(chosen, true);
  const reason = same?.reason ?? "";
  const context = !same ? `for ${mode}` : /engine/i.test(reason) ? "with the dose engine" : `on machine '${machine}'`;
  return pick(chosen, true, { text: `Biological model '${current.label ?? current.model}' is not available ${context}; selected '${chosen?.label ?? "none"}'.`, reason });
}

function planOptions(s: MockSession, p: Record<string, unknown>): PlanOptions {
  const e = ext(s);
  const mode = String(p.radiation_mode ?? s.plan?.radiation_mode ?? "photons");
  if (!RADIATION_MODES.includes(mode)) throw new MethodError("ValueError", `Unknown radiation mode '${mode}'`);
  const machine = String(p.machine ?? (s.plan && s.plan.radiation_mode === mode ? s.plan.machine : "Generic"));
  const engine = (p.engine as string | undefined) ?? (s.plan && s.plan.radiation_mode === mode ? s.plan.engine : defaultEngine(mode));
  const scenarios = SCENARIO_MODELS.map((m) => ({ value: m.id, label: m.label, identifier: m.id }));
  if (e.scenario.model === "explicit") {
    scenarios.push({
      value: "explicit",
      label: "Explicit Scenarios",
      identifier: "explicit",
      description: "The scenario set the applied plan carries itself. It is kept unless another model is selected.",
    } as (typeof scenarios)[number]);
  }
  const bioModels = bioOptionsFor(mode, machine, engine ?? null);
  return {
    radiation_modes: RADIATION_MODES.map((m) => ({ value: m, label: m })),
    machines: (MACHINES[mode] ?? ["Generic"]).map((m) => ({ value: m, label: m })),
    engines: enginesFor(mode).map((en) => ({
      value: en.id,
      label: en.label,
      identifier: en.id,
      ...(en.mc ? { enabled: false, reason: MC_REASON } : {}),
    })),
    bio_models: bioModels,
    default_bio_model: DEFAULT_BIO[mode] ?? "none",
    bio_selection: mockBioSelection(s, mode, machine, bioModels, p.current_bio as CurrentBio | undefined),
    scenario_models: scenarios,
    problems: PROBLEMS.map((pr) => ({ value: pr.id, label: pr.label, identifier: pr.id })),
    solvers: SOLVERS.map(({ fields: _fields, ...c }) => c),
    default_solver: DEFAULT_SOLVER,
    beam_presets: beamPresetsFor(mode),
    fields: planFields(e.lockedFields.includes("dose_grid_resolution")),
  };
}

function planForm(s: MockSession, p: Record<string, unknown>) {
  const kind = p.kind as FormKind;
  const name = String(p.name ?? "");
  const e = ext(s);
  const given = p.values as Record<string, unknown> | undefined;
  switch (kind) {
    case "engine": {
      const en = engineDef(name);
      if (!en) throw new MethodError("ValueError", `Unknown dose engine '${name}'`);
      const notes = en.mc ? [`${en.label}: ${MC_REASON}.`] : undefined;
      return formSchema(`engine:${name}`, `Configure ${en.label}`, en.fields, structuredClone(given ?? e.engineValues[name] ?? {}), notes);
    }
    case "bio_model": {
      const option = BIO_OPTIONS.find((o) => o.model === name);
      if (!option) throw new MethodError("ValueError", `Unknown biological model '${name}'`);
      const values = given ?? (bioOptionOf(e.bio.option)?.model === name ? e.bio.values : option.parameters);
      return formSchema(`bio_model:${name}`, `Configure ${option.model === "none" ? option.label : option.model}`, option.fields, structuredClone(values));
    }
    case "scenario": {
      const model = SCENARIO_MODELS.find((m) => m.id === name);
      if (!model) throw new MethodError("ValueError", `Scenario model '${name}' cannot be configured`);
      const values = given ?? (e.scenario.model === name ? e.scenario.values : {});
      return formSchema(`scenario:${name}`, `Configure ${model.label}`, model.fields, structuredClone(values));
    }
    case "problem": {
      const problem = PROBLEMS.find((pr) => pr.id === name);
      if (!problem) throw new MethodError("ValueError", `Unknown planning problem '${name}'`);
      const values = { ...(given ?? (e.problem.name === name ? e.problem.values : {})) };
      for (const hidden of ["solver", "convert_dose_objectives", "max_iter", "display"]) delete values[hidden];
      return formSchema(`problem:${name}`, `Configure ${problem.label}`, problem.fields, structuredClone(values));
    }
    case "solver": {
      const solver = SOLVERS.find((sv) => sv.value === name);
      if (!solver || solver.enabled === false) {
        const reason = solver?.reason ?? (name === "ipopt" ? IPOPT_REASON : "it is not installed in this environment");
        return formSchema(`solver:${name}`, `Configure ${solver?.label ?? name}`, [], {}, [
          `Solver '${name}' is unavailable: ${reason}. Its configuration is kept; choose another solver to replace it.`,
        ]);
      }
      let values: Record<string, unknown>;
      if (given) values = { ...given };
      else {
        values = e.solver.name === name ? { ...e.solver.values } : {};
        for (const key of ["max_iter", "display"]) {
          if (e.problem.values[key] !== undefined && values[key] === undefined) values[key] = e.problem.values[key];
        }
      }
      return formSchema(`solver:${name}`, `Configure ${solver.label}`, solver.fields, structuredClone(values));
    }
    default:
      throw new MethodError("ValueError", `Unknown form kind '${String(kind)}'`);
  }
}

const PLAN_SET_FIELDS = new Set([
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
  "dose_convention",
  "engine_config",
  "scenarios",
  "problem",
  "solver",
]);

function planSet(ctx: MockContext, s: MockSession, p: PlanSetParams): SessionState {
  if (!s.phantom) throw new MethodError("LookupError", "No CT / structures loaded");
  if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
  const unknown = Object.keys(p).filter((k) => !PLAN_SET_FIELDS.has(k));
  if (unknown.length) throw new MethodError("ValueError", `Unknown plan field(s): ${unknown.join(", ")}`);
  const e = ext(s);
  const locked = Object.keys(p).filter((k) => e.lockedFields.includes(k));
  if (locked.length) throw precondition(`Plan field(s) locked by the profile: ${locked.join(", ")}`, "locked");
  const base = s.plan ?? defaultPlanFor(e.profile);
  const next: PlanSummary = { ...base };
  const details: ValidationDetail[] = [];
  const fields = planFields(false);
  const check = (name: string, value: unknown) => {
    const f = fields.find((x) => x.name === name);
    const message = f ? checkValue(f, value) : null;
    if (message) details.push({ field: name, message });
  };

  const modeChanged = p.radiation_mode !== undefined && p.radiation_mode !== base.radiation_mode;
  if (p.radiation_mode !== undefined) {
    if (!RADIATION_MODES.includes(p.radiation_mode)) throw new MethodError("ValueError", `Unknown radiation mode ${p.radiation_mode}`);
    next.radiation_mode = p.radiation_mode;
  }
  if (modeChanged) {
    next.machine = (MACHINES[next.radiation_mode] ?? ["Generic"])[0];
    next.engine = defaultEngine(next.radiation_mode);
    const bio = bioOptionOf(DEFAULT_BIO[next.radiation_mode] ?? "none");
    next.bio_model = bio?.model ?? "none";
    e.bio = { option: bio?.id ?? "none", values: structuredClone(bio?.parameters ?? {}) };
  }
  if (p.machine !== undefined) next.machine = p.machine;
  if (p.num_of_fractions !== undefined) {
    check("num_of_fractions", p.num_of_fractions);
    next.num_of_fractions = p.num_of_fractions;
  }
  if (p.bixel_width !== undefined) {
    check("bixel_width", p.bixel_width);
    next.bixel_width = p.bixel_width;
  }
  if (p.dose_grid_resolution !== undefined) {
    check("dose_grid_resolution", p.dose_grid_resolution);
    next.dose_grid_resolution = p.dose_grid_resolution;
  }
  if (p.iso_center !== undefined) next.iso_center = p.iso_center;
  if (p.gantry_angles !== undefined) {
    if (!p.gantry_angles.length) details.push({ field: "gantry_angles", message: "At least one gantry angle is required" });
    next.gantry_angles = p.gantry_angles;
    if (p.couch_angles === undefined) next.couch_angles = p.gantry_angles.map(() => 0);
  }
  if (p.couch_angles !== undefined) next.couch_angles = p.couch_angles;
  if (next.couch_angles.length === 1 && next.gantry_angles.length > 1) next.couch_angles = next.gantry_angles.map(() => next.couch_angles[0]);
  if (next.couch_angles.length !== next.gantry_angles.length) {
    throw new MethodError("ValueError", `${next.gantry_angles.length} gantry angles but ${next.couch_angles.length} couch angles`);
  }
  if (Array.isArray(next.iso_center) && next.iso_center.length && !Array.isArray(next.iso_center[0]) && next.iso_center.length !== 3) {
    details.push({ field: "iso_center", message: "iso center requires exactly three coordinates (x y z)" });
  }

  if (p.engine !== undefined) {
    const en = engineDef(p.engine);
    if (p.engine !== null && (!en || !en.modes.includes(next.radiation_mode))) {
      details.push({ field: "engine", message: `Dose engine '${p.engine}' is not available for ${next.radiation_mode}` });
    } else if (en?.mc) {
      throw precondition(`${en.label}: ${MC_REASON}`, "not_available");
    }
    next.engine = p.engine;
  }
  let engineValues: Record<string, unknown> | undefined;
  if (p.engine_config) {
    const en = engineDef(next.engine);
    engineValues = structuredClone(p.engine_config.values ?? {});
    if (en) details.push(...validateValues(en.fields, engineValues, "engine."));
  }

  let bio = e.bio;
  if (p.bio_model !== undefined && !modeChanged) bio = resolveBio(p.bio_model, next.radiation_mode, details);
  else if (p.bio_model !== undefined) bio = resolveBio(p.bio_model, next.radiation_mode, details);
  const bioOption = bioOptionOf(bio.option);
  if (bioOption) {
    if (!bioOption.modes.includes(next.radiation_mode)) {
      details.push({ field: "bio_model", message: `${bioOption.label} is not available for ${next.radiation_mode}` });
    } else details.push(...validateValues(bioOption.fields, bio.values, "bio_model."));
  }

  let scenario = e.scenario;
  if (p.scenarios) {
    const model = SCENARIO_MODELS.find((m) => m.id === p.scenarios!.model);
    if (!model && p.scenarios.model !== "explicit") details.push({ field: "scenarios.model", message: `Unknown scenario model '${p.scenarios.model}'` });
    if (model) details.push(...validateValues(model.fields, p.scenarios.values ?? {}, "scenario."));
    scenario = { model: p.scenarios.model, values: structuredClone(p.scenarios.values ?? {}) };
  }
  let problem = e.problem;
  if (p.problem) {
    const pr = PROBLEMS.find((x) => x.id === p.problem!.name);
    if (!pr) details.push({ field: "problem.name", message: `Unknown planning problem '${p.problem.name}'` });
    else details.push(...validateValues(pr.fields, p.problem.values ?? {}, "problem."));
    // Settings the form does not show are kept unless sent.
    const kept = p.problem.name === e.problem.name ? shownProblemValues(e.problem.name, e.problem.values, false) : {};
    problem = { name: p.problem.name, values: { ...structuredClone(kept), ...structuredClone(p.problem.values ?? {}) } };
  }
  let solver = e.solver;
  if (p.solver !== undefined) {
    const spec = typeof p.solver === "string" ? { name: p.solver, values: {} } : p.solver;
    const sv = SOLVERS.find((x) => x.value === spec.name);
    if (sv && sv.enabled !== false) details.push(...validateValues(sv.fields, spec.values ?? {}, "solver."));
    solver = { name: spec.name, values: structuredClone(spec.values ?? {}) };
  }

  if (details.length) throw validationError(details);

  if (p.dose_convention !== undefined) e.doseConvention = p.dose_convention;
  if (engineValues && next.engine) e.engineValues[next.engine] = engineValues;
  e.bio = bio;
  next.bio_model = bioOptionOf(bio.option)?.model ?? next.bio_model ?? "none";
  e.scenario = scenario;
  e.problem = problem;
  e.solver = solver;
  s.plan = next;
  ctx.touch(s, "plan");
  ctx.log(s, "info", `Plan updated: ${next.radiation_mode}, ${next.gantry_angles.length} beam(s)`);
  return ctx.publish(s);
}

function resolveBio(spec: PlanSetParams["bio_model"], mode: string, details: ValidationDetail[]): PlanExt["bio"] {
  if (spec === null || spec === undefined) return { option: "none", values: {} };
  if (typeof spec === "string") {
    const option = BIO_OPTIONS.find((o) => (o.id === spec || o.model === spec) && o.modes.includes(mode)) ?? bioOptionOf(spec);
    if (!option) {
      details.push({ field: "bio_model", message: `Unknown biological model '${spec}'` });
      return { option: null, values: {} };
    }
    return { option: option.id, values: structuredClone(option.parameters) };
  }
  const option = (spec.option ? bioOptionOf(spec.option) : undefined) ?? BIO_OPTIONS.find((o) => o.model === spec.model && o.modes.includes(mode));
  if (!option) {
    details.push({ field: "bio_model.model", message: `Unknown biological model '${spec.model}'` });
    return { option: null, values: {} };
  }
  return { option: option.id, values: structuredClone(spec.values ?? option.parameters) };
}

// ---------------------------------------------------------------------------
// Tissues
// ---------------------------------------------------------------------------

/** Tissue model the plan's biological model needs, with the base data provenance. */
function tissueNeed(bioId: string | null, machine: string): { model: "LEM" | "mMKM" | "Generic" | "MKM" | null; classes: typeof TISSUE_CLASSES } {
  const option = bioOptionOf(bioId);
  if (!option || !option.tissue) return { model: null, classes: [] };
  const classes = TISSUE_CLASSES.filter((c) => c.model === option.tissue && (machine === "HIT_APM" || c.kernel === "HIT"));
  return { model: option.tissue, classes };
}

const REQUIRED: Record<string, string[]> = {
  Generic: ["alpha_x", "beta_x"],
  LEM: ["d_t", "r_nucleus", "alpha_x", "beta_x"],
  MKM: ["r_domain", "r_nucleus", "alpha_0", "beta_0"],
  mMKM: ["r_domain", "r_nucleus", "alpha_0", "beta_0"],
};

function modelFields(model: string): string[] {
  return TISSUE_MODELS.find((m) => m.label === model)?.fields.map((f) => f.name) ?? [];
}

const LABEL: Record<string, string> = Object.fromEntries(
  TISSUE_MODELS.flatMap((m) => m.fields.map((f) => [f.name, f.label] as const)),
);

function tissueIssues(bioId: string | null, machine: string, rows: Array<{ structure: string; tissue: TissueDraft }>): Issue[] {
  const need = tissueNeed(bioId, machine);
  const issues: Issue[] = [];
  const option = bioOptionOf(bioId);
  if (option?.machines && !option.machines.includes(machine)) {
    issues.push({ structure: null, kind: "machine", severity: "error", message: `Machine '${machine}' offers no base data for ${option.label}` });
  }
  if (!need.model) return issues;
  for (const { structure, tissue } of rows) {
    const family = (m: string) => (m === "mMKM" ? "MKM" : m);
    if (need.model !== "Generic" && family(tissue.model) !== family(need.model)) {
      issues.push({
        structure,
        kind: "model",
        severity: "error",
        message: `${structure}: the tissue is a ${tissue.model} tissue; ${option?.model} needs ${need.model} parameters (${REQUIRED[need.model].map((f) => LABEL[f] ?? f).join(", ")})`,
        closest_class: need.classes[0]?.label.replace(/^✓ /, "") ?? null,
      });
      continue;
    }
    const missing = REQUIRED[need.model === "Generic" ? "Generic" : tissue.model].filter((f) => tissue.values[f] === null || tissue.values[f] === undefined);
    if (missing.length) {
      issues.push({ structure, kind: "missing", severity: "error", message: `${structure}: ${missing.map((f) => LABEL[f] ?? f).join(", ")} not set` });
      continue;
    }
    if (need.model === "mMKM" && tissue.model === "MKM") {
      issues.push({
        structure,
        kind: "provenance",
        severity: "warning",
        message: `${structure}: labelled MKM, matches the mMKM base data; evaluated with mMKM`,
        closest_class: need.classes[0]?.label.replace(/^✓ /, "") ?? null,
      });
    }
  }
  return issues;
}

const Z_REF: Record<string, number> = { xray_200kV: 0.0058, co60: 0.0021, mv6: 0.0019 };

function derive(tissue: TissueDraft, force: boolean): boolean {
  if (tissue.model !== "MKM" && tissue.model !== "mMKM") return false;
  const v = tissue.values;
  const rd = Number(v.r_domain);
  const a0 = v.alpha_0;
  const b0 = v.beta_0;
  if (v.r_domain === null || a0 === null || b0 === null || a0 === undefined || b0 === undefined || !(rd > 0)) return false;
  const z = (Z_REF[String(v.reference_radiation ?? "xray_200kV")] ?? Z_REF.xray_200kV) / (rd * rd) * (tissue.model === "mMKM" ? 0.94 : 1);
  const sources = (tissue.sources ??= {});
  let changed = false;
  for (const [field, value] of [
    ["alpha_x", Number(a0) + Number(b0) * z],
    ["beta_x", Number(b0)],
  ] as const) {
    if (!force && sources[field] === "explicit") continue;
    v[field] = Math.round(value * 1e6) / 1e6;
    sources[field] = "derived";
    changed = true;
  }
  return changed;
}

function compare(tissue: TissueDraft, classes: typeof TISSUE_CLASSES): TissueComparison | null {
  if (!classes.length) return null;
  const scored = classes.map((c) => {
    const parameters = Object.entries(c.values).map(([name, declared]) => ({
      name,
      declared,
      value: tissue.values[name] ?? null,
      matches: tissue.values[name] === declared || (typeof declared === "number" && Math.abs(Number(tissue.values[name]) - declared) < 1e-9),
    }));
    return { c, parameters, misses: parameters.filter((p) => !p.matches).length };
  });
  scored.sort((a, b) => a.misses - b.misses);
  const best = scored[0];
  const family = (m: string) => (m === "mMKM" ? "MKM" : m);
  const sameFamily = family(tissue.model) === family(best.c.model);
  return {
    closest_class: best.c.label.replace(/^✓ /, ""),
    matched: sameFamily && best.misses === 0,
    provenance_fix: sameFamily && tissue.model !== best.c.model,
    parameters: best.parameters,
  };
}

function sourcesOf(tissue: TissueDraft): Record<string, TissueSource> {
  const out: Record<string, TissueSource> = {};
  for (const name of modelFields(tissue.model)) {
    if (name === "reference_radiation" || name === "version") continue;
    const given = tissue.sources?.[name] as TissueSource | undefined;
    out[name] = given ?? (tissue.values[name] === null || tissue.values[name] === undefined ? "default" : "explicit");
  }
  return out;
}

function tissueEvaluate(s: MockSession, p: TissueEvaluateParams) {
  const e = ext(s);
  syncBio(s);
  const machine = s.plan?.machine ?? "Generic";
  const need = tissueNeed(e.bio.option, machine);
  const rows = p.rows.map((r) => ({ structure: r.structure, tissue: structuredClone(r.tissue) }));
  const targets = new Set(p.targets?.length ? p.targets : rows.map((r) => r.structure));
  const notes: string[] = [];
  for (const row of rows) {
    const t = row.tissue;
    t.sources ??= {};
    if (p.edited?.structure === row.structure && (p.edited.field === "alpha_x" || p.edited.field === "beta_x")) {
      t.sources[p.edited.field] = "explicit";
    }
    const inTargets = targets.has(row.structure);
    switch (p.op ?? "none") {
      case "derive":
        if (inTargets && derive(t, true)) notes.push(`Derived α_x / β_x of ${row.structure} from its model parameters`);
        continue;
      case "match": {
        const cmp = inTargets ? compare(t, need.classes) : null;
        const cls = cmp && need.classes.find((c) => c.label.replace(/^✓ /, "") === cmp.closest_class);
        if (cls) {
          row.tissue = { model: cls.model, values: { ...t.values, ...cls.values }, sources: { ...t.sources } };
          if (cls.model === "mMKM") {
            delete row.tissue.values.alpha_x;
            delete row.tissue.values.beta_x;
            row.tissue.values.alpha_0 = t.values.alpha_0 ?? 0.172;
            row.tissue.values.beta_0 = t.values.beta_0 ?? 0.0615;
          }
          notes.push(`Matched ${row.structure} to ${cmp!.closest_class}`);
        }
        break;
      }
      case "relabel": {
        const cmp = inTargets ? compare(t, need.classes) : null;
        if (cmp?.provenance_fix && need.model && need.model !== "Generic") {
          t.model = need.model as TissueDraft["model"];
          notes.push(`Relabelled ${row.structure} as ${need.model}`);
        }
        break;
      }
      case "apply_pick": {
        if (!inTargets) break;
        const pick = tissuePicks(need.model === "LEM" || need.model === "mMKM" ? need.model : null, machine).find((x) => x.id === p.pick);
        if (!pick) throw new MethodError("KeyError", `Unknown tissue pick ${p.pick}`);
        let model = pick.model;
        if (pick.kind === "preset" && need.model === "mMKM" && model === "MKM") {
          model = "mMKM";
          notes.push(`Adopted the MKM preset ${pick.label} as mMKM, keeping the parameters both share`);
        } else notes.push(pick.kind === "base_data" ? `Applied the base data class ${pick.label.replace(/^✓ /, "")}` : `Applied the cell line preset ${pick.label}`);
        row.tissue = {
          model,
          values: pick.kind === "preset" ? { ...structuredClone(pick.values) } : { ...t.values, ...pick.values },
          sources: {},
        };
        break;
      }
    }
    derive(row.tissue, false);
  }
  const count = (prefix: string) => notes.filter((n) => n.startsWith(prefix)).length;
  if (p.op === "match" && count("Matched")) notes.splice(0, notes.length, `Matched ${count("Matched")} structure(s) to their closest base data class`);
  if (p.op === "relabel" && count("Relabelled")) notes.splice(0, notes.length, `Relabelled ${count("Relabelled")} structure(s) with the base data's model`);
  const uniqueNotes = [...new Set(notes)];
  notes.splice(0, notes.length, ...uniqueNotes);
  const issues = tissueIssues(e.bio.option, machine, rows);
  const comparisons = rows.map((r) => (need.model && need.model !== "Generic" ? compare(r.tissue, need.classes) : null));
  const outRows: TissueEvalRow[] = rows.map((r, i) => ({
    structure: r.structure,
    tissue: { ...r.tissue, sources: sourcesOf(r.tissue) },
    issues: issues.filter((x) => x.structure === r.structure),
    comparison: comparisons[i],
  }));
  const relabel = need.model && comparisons.some((c) => c?.provenance_fix) ? [need.model] : [];
  return {
    rows: outRows,
    notes,
    can_match: comparisons.some((c) => c !== null),
    can_relabel: relabel,
    can_derive: rows.some((r) => r.tissue.model === "MKM" || r.tissue.model === "mMKM"),
  };
}

function tissueCheck(s: MockSession, draft: Record<string, unknown> | undefined) {
  const e = ext(s);
  syncBio(s);
  let bioId = e.bio.option;
  let machine = s.plan?.machine ?? "Generic";
  if (draft) {
    machine = String(draft.machine ?? machine);
    const spec = draft.bio_model as PlanSetParams["bio_model"];
    const details: ValidationDetail[] = [];
    bioId = resolveBio(spec, String(draft.radiation_mode ?? s.plan?.radiation_mode ?? "photons"), details).option;
  }
  const issues = tissueIssues(
    bioId,
    machine,
    e.structures.map((st) => ({ structure: st.name, tissue: st.tissue })),
  );
  const marker: "error" | "warning" | "" = issues.some((i) => i.severity === "error") ? "error" : issues.length ? "warning" : "";
  return { marker, issues };
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

function findStructure(s: MockSession, name: unknown): MockStructure {
  const st = ext(s).structures.find((x) => x.name === name);
  if (!st) throw new MethodError("KeyError", `Unknown structure ${JSON.stringify(name)}`);
  return st;
}

function objectiveType(name: string) {
  const t = OBJECTIVES.find((o) => o.name === name);
  if (!t) throw new MethodError("ValueError", `Unknown objective '${name}'`);
  return t;
}

function newObjective(name: string): Omit<ObjectiveData, "index"> {
  const t = objectiveType(name);
  const parameters: Record<string, unknown> = {};
  for (const f of t.parameters) parameters[f.name] = f.type === "image_ref" ? { $custom: "Custom image (1×1×1, value 0)" } : f.default;
  return { name, priority: 1, quantity: "physical_dose", robustness: String(t.robustness[0]?.value ?? "none"), parameters };
}

let cstRevision = 0;

/** New cst revision in the viewer mock's `revisions.workspace.cst` (negative: never one of its own). */
function bumpCst(s: MockSession): void {
  const viewer = s.ext.viewer as { revCst?: number } | undefined;
  if (viewer) viewer.revCst = --cstRevision;
}

/** The fields this module owns in a `StructureDetail` (merged over the viewer mock's). */
function pickOwn(d: StructureDetail): Partial<StructureDetail> {
  return { tissue: d.tissue, tissue_summary: d.tissue_summary, alpha_x: d.alpha_x, beta_x: d.beta_x, objectives: d.objectives };
}

function structureResult(ctx: MockContext, s: MockSession, st: MockStructure) {
  bumpCst(s);
  const index = ext(s).structures.indexOf(st);
  return { state: ctx.publish(s), structure: structureDetail(st, index) };
}

function writeCheck(ctx: MockContext): void {
  if (ctx.isBusy()) throw precondition("A computation is still running", "busy");
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

const FEATURES = {
  image_file_io: { available: false, reason: "Reading and writing image files needs SimpleITK, which the browser runtime only shims" },
  dicom_import: { available: false, reason: "DICOM import needs SimpleITK.ImageSeriesReader, which the browser runtime lacks" },
  dicom_export: { available: true, reason: null },
  ai: { available: false, reason: "AI features need pydantic-ai, which is not installed in the browser (Pyodide)" },
  ipopt: { available: false, reason: IPOPT_REASON },
  gpu: { available: false, reason: "The browser runtime computes with numpy on the CPU only" },
  numba: { available: false, reason: "numba is not available in Pyodide" },
  monte_carlo: { available: false, reason: MC_REASON },
  pause: {
    available: false,
    reason: "Pausing needs a second thread; in the browser pyRadPlan runs requests one at a time in a single worker",
  },
  open_folder: { available: false, reason: "The browser cannot open folders of the host" },
  host_paths: { available: false, reason: "The browser has no access to host paths" },
  base_data_persistence: { available: false, reason: "Base data added in the browser lives until the page is closed" },
};

export const planHandlers: MockHandlers = {
  async "app.info"(_p, ctx) {
    const info = (await ctx.fallback()) as AppInfo;
    const caps = info.capabilities;
    return {
      ...info,
      runtime: info.runtime ?? "pyodide",
      default_profile: info.default_profile ?? "web",
      quantities: info.quantities ?? QUANTITIES,
      solvers: ["ipax", "scipy"],
      engines: Object.fromEntries(RADIATION_MODES.map((m) => [m, enginesFor(m).map((e) => e.id)])),
      machines: MACHINES,
      capabilities: {
        features: { ...FEATURES, ...(caps?.features ?? {}) },
        formats: caps?.formats ?? { import: [], export: [] },
      },
    };
  },

  async "session.create"(p, ctx) {
    const result = (await ctx.fallback()) as { session: string };
    const s = ctx.sessionById(result.session);
    if (!s) return result;
    const e = ext(s);
    e.profile = (p.profile as Profile | undefined) ?? "web";
    const options = (p.options ?? {}) as { locked_fields?: string[] };
    e.lockedFields = [...(options.locked_fields ?? [])];
    return { ...result, options: planStateExtension(s, ctx.state(s)).options };
  },

  async "data.load_phantom"(_p, ctx) {
    return applyProfileDefaults(ctx, await ctx.fallback());
  },

  async "data.load_file"(_p, ctx) {
    return applyProfileDefaults(ctx, await ctx.fallback());
  },

  "plan.get"(p, ctx) {
    const s = ctx.session();
    if (!p.detail) return s.plan;
    return planDetail(s);
  },

  "plan.options"(p, ctx) {
    return planOptions(ctx.session(), p);
  },

  "plan.form"(p, ctx) {
    return planForm(ctx.session(), p);
  },

  "plan.set"(p, ctx) {
    return planSet(ctx, ctx.session(), p as PlanSetParams);
  },

  "plan.tissue_check"(p, ctx) {
    return tissueCheck(ctx.session(), p.draft as Record<string, unknown> | undefined);
  },

  async "cst.get"(p, ctx) {
    const s = ctx.session();
    if (!s.phantom) throw new MethodError("LookupError", "No structures loaded");
    const names = p.names as string[] | undefined;
    const own = ext(s).structures.map((st, i) => structureDetail(st, i));
    let base: StructureDetail[] | null = null;
    try {
      base = ((await ctx.fallback()) as { structures: StructureDetail[] }).structures;
    } catch {
      base = null;
    }
    const merged = (base ?? own).map((b) => {
      const mine = own.find((o) => o.name === b.name);
      return mine ? { ...b, ...pickOwn(mine) } : b;
    });
    return { structures: merged.filter((st) => !names || names.includes(st.name)) };
  },

  async "cst.update_structure"(p, ctx) {
    const s = ctx.session();
    const changes = (p.changes ?? {}) as { type?: string; color?: Vec3; tissue?: TissueDraft; overlap_priority?: number };
    const st = findStructure(s, p.name);
    if (changes.tissue && !p.apply_anyway) {
      const errors = tissueIssues(ext(s).bio.option, s.plan?.machine ?? "Generic", [{ structure: st.name, tissue: changes.tissue }]).filter(
        (i) => i.severity === "error",
      );
      if (errors.length) throw precondition("The selected biological model cannot be evaluated for this tissue", "tissue_errors", errors);
    }
    let result: { state: SessionState; structure: StructureDetail } | null = null;
    try {
      result = (await ctx.fallback()) as { state: SessionState; structure: StructureDetail };
    } catch (err) {
      if (!(err instanceof MethodError && err.type === "NotImplementedError")) throw err;
    }
    if (changes.type !== undefined) st.type = changes.type;
    if (changes.color !== undefined) st.color = changes.color;
    if (changes.overlap_priority !== undefined) st.overlap_priority = Number(changes.overlap_priority);
    if (changes.tissue !== undefined) st.tissue = structuredClone(changes.tissue);
    const index = ext(s).structures.indexOf(st);
    const mine = structureDetail(st, index);
    return { state: ctx.publish(s), structure: result ? { ...result.structure, ...pickOwn(mine) } : mine };
  },

  "tissue.schema"(_p, ctx) {
    const s = ctx.session();
    const e = ext(s);
    syncBio(s);
    const machine = s.plan?.machine ?? "Generic";
    const need = tissueNeed(e.bio.option, machine);
    const preferred = need.model === "LEM" || need.model === "mMKM" ? need.model : null;
    return {
      models: TISSUE_MODELS,
      reference_radiations: REFERENCE_RADIATIONS,
      picks: tissuePicks(preferred, machine),
      option_label: bioOptionOf(e.bio.option)?.label ?? "None (physical dose)",
    };
  },

  "tissue.evaluate"(p, ctx) {
    return tissueEvaluate(ctx.session(), p as unknown as TissueEvaluateParams);
  },

  "tissue.apply"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const e = ext(s);
    const rows = (p.rows ?? []) as Array<{ structure: string; tissue: TissueDraft }>;
    const issues = tissueIssues(e.bio.option, s.plan?.machine ?? "Generic", rows).filter((i) => i.severity === "error");
    if (issues.length && !p.apply_anyway) {
      throw precondition("The selected biological model cannot be evaluated for these tissues", "tissue_errors", issues);
    }
    for (const row of rows) findStructure(s, row.structure).tissue = structuredClone(row.tissue);
    bumpCst(s);
    ctx.log(s, "info", `Tissue parameters of ${rows.length} structure(s) updated`);
    return { state: ctx.publish(s) };
  },

  "objectives.catalog"(_p, ctx) {
    const s = ctx.session();
    return {
      objectives: OBJECTIVES,
      quantities: QUANTITIES.map((q) => ({ value: q.identifier, label: q.name, identifier: q.identifier })),
      robustness: OBJECTIVES[0].robustness,
      default_by_type: { TARGET: "Squared Deviation", "*": "Squared Overdosing" },
      reference_images: s.dose ? [{ value: "physical_dose", label: "dose", identifier: "physical_dose" }] : [],
    };
  },

  "objectives.add"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const st = findStructure(s, p.structure);
    const obj = newObjective(DEFAULT_OBJECTIVE[st.type] ?? "Squared Overdosing");
    const used = new Set(ext(s).structures.flatMap((x) => x.objectives.map((o) => o.quantity)));
    if (used.size === 1) obj.quantity = [...used][0];
    st.objectives.push(obj);
    return structureResult(ctx, s, st);
  },

  "objectives.update"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const st = findStructure(s, p.structure);
    const index = Number(p.index);
    const obj = st.objectives[index];
    if (!obj) throw new MethodError("IndexError", `Objective ${index} of ${st.name} does not exist`);
    const changes = (p.changes ?? {}) as { priority?: number; quantity?: string; robustness?: string; parameters?: Record<string, unknown> };
    const t = objectiveType(obj.name);
    const details: ValidationDetail[] = [];
    if (changes.priority !== undefined && !(typeof changes.priority === "number" && changes.priority >= 0 && changes.priority <= 1e9)) {
      details.push({ field: "priority", message: "Input should be greater than or equal to 0" });
    }
    if (changes.quantity !== undefined && !QUANTITIES.some((q) => q.identifier === changes.quantity)) {
      details.push({ field: "quantity", message: `Unknown quantity '${changes.quantity}'` });
    }
    if (changes.robustness !== undefined && !t.robustness.some((r) => r.value === changes.robustness)) {
      details.push({ field: "robustness", message: `Robustness '${changes.robustness}' is not available for ${obj.name}` });
    }
    for (const [name, value] of Object.entries(changes.parameters ?? {})) {
      const f = t.parameters.find((x) => x.name === name);
      if (!f) details.push({ field: `parameters.${name}`, message: `${obj.name} has no parameter ${name}` });
      else if (f.type === "image_ref") {
        const key = (value as { $result?: string } | null)?.$result;
        if (!key || (key !== "physical_dose" && !key.startsWith("physical_dose")) || !s.dose) {
          details.push({ field: `parameters.${name}`, message: `No result image ${JSON.stringify(key)}` });
        }
      } else {
        const message = checkValue(f, value);
        if (message) details.push({ field: `parameters.${name}`, message });
      }
    }
    if (details.length) throw validationError(details);
    if (changes.priority !== undefined) obj.priority = changes.priority;
    if (changes.quantity !== undefined) obj.quantity = changes.quantity;
    if (changes.robustness !== undefined) obj.robustness = changes.robustness;
    if (changes.parameters) Object.assign(obj.parameters, structuredClone(changes.parameters));
    return structureResult(ctx, s, st);
  },

  "objectives.change_type"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const st = findStructure(s, p.structure);
    const index = Number(p.index);
    const old = st.objectives[index];
    if (!old) throw new MethodError("IndexError", `Objective ${index} of ${st.name} does not exist`);
    const obj = newObjective(String(p.name));
    obj.priority = old.priority;
    if (objectiveType(obj.name).robustness.some((r) => r.value === old.robustness)) obj.robustness = old.robustness;
    st.objectives[index] = obj;
    return structureResult(ctx, s, st);
  },

  "objectives.remove"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const st = findStructure(s, p.structure);
    const index = Number(p.index);
    if (!st.objectives[index]) throw new MethodError("IndexError", `Objective ${index} of ${st.name} does not exist`);
    st.objectives.splice(index, 1);
    return structureResult(ctx, s, st);
  },

  "objectives.set_quantity_all"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const quantity = String(p.quantity);
    if (!QUANTITIES.some((q) => q.identifier === quantity)) throw validationError([{ field: "quantity", message: `Unknown quantity '${quantity}'` }]);
    for (const st of ext(s).structures) for (const o of st.objectives) o.quantity = quantity;
    bumpCst(s);
    return { state: ctx.publish(s) };
  },

  async "workflow.optimize"(p, ctx) {
    const s = ctx.session();
    const e = ext(s);
    if (p.max_iter === undefined) {
      const fromPlan = Number(e.solver.values.max_iter ?? e.problem.values.max_iter);
      (ctx.request as { params: unknown }).params = { ...p, max_iter: Number.isFinite(fromPlan) && fromPlan > 0 ? Math.min(fromPlan, 400) : 80 };
    }
    const solver = e.solver.name ?? DEFAULT_SOLVER;
    const history: OptimizationHistory = { label: `Optimization (${solver})`, points: [], finished: false, outcome: null };
    const unsubscribe = ctx.subscribe((ev: BackendEvent) => {
      if (ev.event === "progress" && ev.data && typeof ev.data.objective === "number") {
        history.points.push({ iteration: Number(ev.data.iteration ?? history.points.length + 1), objective: ev.data.objective, ...ev.data });
      }
    });
    if (!ctx.isBusy()) e.history = history;
    try {
      const result = (await withBioCheck(ctx, s, false, () => ctx.fallback())) as Record<string, unknown>;
      const { stopped, ...opt } = (result.optimization ?? {}) as Record<string, unknown>;
      history.outcome = stopped ? "stopped" : "done";
      return result.optimization ? { ...result, optimization: { ...opt, solver } } : result;
    } catch (err) {
      history.outcome = err instanceof Error && (err.name === "KeyboardInterrupt" || err.name === "ComputeCancelledError") ? "stopped" : "failed";
      throw err;
    } finally {
      history.finished = true;
      unsubscribe();
    }
  },

  async "workflow.calc_dose_influence"(p, ctx) {
    return withBioCheck(ctx, ctx.session(), Boolean(p.physical_only), () => ctx.fallback());
  },

  async "workflow.recalc"(p, ctx) {
    return withBioCheck(ctx, ctx.session(), Boolean(p.physical_only), () => ctx.fallback());
  },

  async "workflow.run_all"(p, ctx) {
    return withBioCheck(ctx, ctx.session(), Boolean(p.physical_only), () => ctx.fallback());
  },

  async "session.history"(_p, ctx) {
    let previous: { log?: unknown[] } = {};
    try {
      previous = ((await ctx.fallback()) as { log?: unknown[] }) ?? {};
    } catch {
      previous = {};
    }
    return { ...previous, log: previous.log ?? [], optimization: ext(ctx.session()).history };
  },

  "session.pause"() {
    throw precondition(FEATURES.pause.reason, "not_available");
  },

  "session.resume"() {
    throw precondition(FEATURES.pause.reason, "not_available");
  },
};

/** After a load: the session profile's default plan (the core applies the native one). */
function applyProfileDefaults(ctx: MockContext, result: unknown): unknown {
  const s = ctx.session();
  const e = ext(s);
  if (s.plan && e.profile === "web") {
    s.plan = { ...s.plan, dose_grid_resolution: [...WEB_GRID] };
    e.doseConvention = "total";
    e.bio = { option: "none", values: {} };
    e.engineValues = {};
    e.scenario = { model: "nomScen", values: {} };
    e.problem = { name: "nonlin_fluence", values: {} };
    e.solver = { name: null, values: {} };
    return { ...(result as object), ...ctx.publish(s) };
  }
  return result;
}

/**
 * The tissue check of the plan's biological model replaces the core's carbon rule:
 * fails with `bio_model_unsupported` while the structures' tissues have errors.
 */
async function withBioCheck<T>(ctx: MockContext, s: MockSession, physicalOnly: boolean, run: () => Promise<T>): Promise<T> {
  if (!s.plan || !s.phantom) return run();
  const check = tissueCheck(s, undefined);
  const saved = s.plan.bio_model;
  if (check.marker === "error" && saved !== "none") {
    if (!physicalOnly) {
      throw precondition(
        `The biological model ${bioOptionOf(ext(s).bio.option)?.label ?? saved} cannot be evaluated: ${check.issues
          .filter((i) => i.severity === "error")
          .map((i) => i.message)
          .join("; ")}`,
        "bio_model_unsupported",
      );
    }
    s.plan = { ...s.plan, bio_model: "none" };
    ext(s).bio = { option: "none", values: {} };
    ctx.touch(s, "plan");
    return run();
  }
  // Bypass the core's carbon rule while the tissues are fine.
  s.plan = { ...s.plan, bio_model: "none" };
  let result: T;
  try {
    result = await run();
  } finally {
    if (s.plan) s.plan = { ...s.plan, bio_model: saved };
  }
  if (result && typeof result === "object" && "has" in result) return { ...result, ...ctx.publish(s) };
  return result;
}

/**
 * Mock handlers of package A (PROTOCOL.md v1: plan detail / options / forms, tissues,
 * objectives, optimization history). Per-session data lives in `MockSession.ext.plan`; the
 * catalogues and forms are the recorded app responses of `./planData`.
 */
import type {
  BackendEvent,
  FieldDescriptor,
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
  TissueEvaluateParams,
  TissueEvaluateResult,
} from "../protocol/plan";
import { checkValue } from "../../components/forms/formModel";
import { BROWSER_REASONS } from "./io";
import { mockReferenceImages, mockResultKeys } from "./viewer";
import { MethodError, precondition, resolveStructure, type MockContext, type MockHandlers, type MockSession } from "./mockBackend";
import {
  DEFAULT_SOLVER,
  OBJECTIVES,
  PHANTOM_STRUCTURES,
  PROBLEMS,
  RADIATION_MODES,
  SCENARIO_MODELS,
  SOLVERS,
  TISSUE_SCHEMA,
  bioOptionById,
  bioOptionOfModel,
  defaultBioOption,
  defaultEngine,
  machinesFor,
  planOptionsFor,
  recordedForm,
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

function noBio(): PlanExt["bio"] {
  const option = bioOptionOfModel("photons", "none");
  return { option: option ? String(option.value) : null, values: {} };
}

function ext(s: MockSession): PlanExt {
  let e = s.ext.plan as PlanExt | undefined;
  if (!e) {
    e = {
      profile: "web",
      lockedFields: [],
      doseConvention: "total",
      engineValues: {},
      bio: noBio(),
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

/** Tissue, overlap priority and objectives the recorded phantom has; other structures get the catalogue's defaults. */
function buildStructures(s: MockSession): MockStructure[] {
  const phantom = s.phantom;
  if (!phantom) return [];
  const [nz, ny, nx] = phantom.grid.shape;
  const g = phantom.grid;
  const recorded = PHANTOM_STRUCTURES[phantom.name] ?? {};
  const fallbackTissue = Object.values(PHANTOM_STRUCTURES)[0] && Object.values(Object.values(PHANTOM_STRUCTURES)[0])[0].tissue;
  return phantom.structures.map((st) => {
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
    const known = recorded[st.name];
    return {
      name: st.name,
      type: st.type,
      color: st.color,
      overlap_priority: known?.overlap_priority ?? (st.type === "TARGET" ? 1 : st.type === "EXTERNAL" ? 3 : 2),
      num_voxels: n,
      center_of_mass: com,
      tissue: structuredClone(known?.tissue ?? fallbackTissue),
      objectives: known ? structuredClone(known.objectives) : [newObjective(OBJECTIVES.default_by_type[st.type] ?? OBJECTIVES.default_by_type["*"])],
    };
  });
}

/** Structures of a session (for other packages' mock handlers too). */
export function mockStructures(s: MockSession): MockStructure[] {
  return ext(s).structures;
}

export function structureDetail(st: MockStructure, index: number): StructureDetail {
  const ax = Number(st.tissue.values.alpha_x ?? 0);
  const bx = Number(st.tissue.values.beta_x ?? 0);
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
  const options: SessionOptions = {
    profile: e.profile,
    default_plan: { ...defaultPlanFor(e.profile) },
    locked_fields: [...e.lockedFields],
  };
  return {
    options,
    structures: state.structures.map((st) => {
      const m = e.structures[st.index];
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
    engine: defaultEngine("photons"),
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

const formFields = (kind: FormKind, name: string | null, mode: string): FieldDescriptor[] =>
  name === null ? [] : (recordedForm(kind, name, mode, {})?.fields ?? []);

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

function planDetail(s: MockSession): PlanDetail | null {
  if (!s.plan) return null;
  const e = ext(s);
  syncBio(s);
  const option = bioOptionById(e.bio.option);
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
  const fields = new Set([...flatFields(formFields("problem", name, "")).map((f) => f.name.split(".")[0]), "max_iter", "display"]);
  return Object.fromEntries(Object.entries(values).filter(([k]) => fields.has(k) === shown));
}

/** Follow bio model changes made by the core (e.g. `physical_only`). */
function syncBio(s: MockSession): void {
  const e = ext(s);
  const model = s.plan?.bio_model ?? "none";
  if (bioOptionById(e.bio.option)?.model !== model) {
    const match = bioOptionOfModel(s.plan?.radiation_mode ?? "photons", model);
    e.bio = { option: match ? String(match.value) : null, values: match ? structuredClone(match.parameters) : {} };
  }
}

/**
 * Simplified `bio_selection` (the real rules are `pyRadPlan.app._planform.select_bio_model`):
 * the plan's model for its mode (an unsupported entry when disabled); after a change, a
 * selection the context can run is kept unless it was the previous mode's default, else the
 * mode's default, else `none`.
 */
function mockBioSelection(s: MockSession, mode: string, machine: string, list: BioModelChoice[], defaultOption: string, current: CurrentBio | undefined): BioSelection {
  const enabled = (o: BioModelChoice) => o.enabled !== false;
  const pick = (o: BioModelChoice | undefined, replaced: boolean, note: BioSelection["note"] = null): BioSelection => ({
    option: o ? String(o.value) : null,
    model: o?.model ?? "none",
    replaced,
    unsupported: null,
    note,
  });
  const fallback = () => list.find((o) => o.value === defaultOption && enabled(o)) ?? list.find((o) => o.model === "none") ?? list[0];
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
  const keep = current.radiation_mode === mode || current.model !== (defaultBioOption(current.radiation_mode)?.model ?? "none");
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
  const samePlan = s.plan !== null && s.plan.radiation_mode === mode;
  const machine = String(p.machine ?? (samePlan ? s.plan!.machine : machinesFor(mode)[0]));
  const engine = (p.engine as string | undefined) ?? (samePlan ? s.plan!.engine : defaultEngine(mode));
  const options = planOptionsFor(mode, machine, engine ?? null, e.lockedFields.includes("dose_grid_resolution"));
  const scenarios = [...options.scenario_models];
  if (e.scenario.model === "explicit") {
    scenarios.push({
      value: "explicit",
      label: "Explicit Scenarios",
      identifier: "explicit",
      description: "The scenario set the applied plan carries itself. It is kept unless another model is selected.",
    } as (typeof scenarios)[number]);
  }
  return {
    ...options,
    scenario_models: scenarios,
    bio_selection: mockBioSelection(s, mode, machine, options.bio_models, options.default_bio_model, p.current_bio as CurrentBio | undefined),
  };
}

function planForm(s: MockSession, p: Record<string, unknown>) {
  const kind = p.kind as FormKind;
  const name = String(p.name ?? "");
  const mode = String(p.radiation_mode ?? s.plan?.radiation_mode ?? "photons");
  const e = ext(s);
  const given = p.values as Record<string, unknown> | undefined;
  let values: Record<string, unknown>;
  switch (kind) {
    case "engine":
      values = given ?? e.engineValues[name] ?? {};
      break;
    case "bio_model":
      values = given ?? (bioOptionById(e.bio.option)?.model === name ? e.bio.values : (bioOptionOfModel(mode, name)?.parameters ?? {}));
      break;
    case "scenario":
      values = given ?? (e.scenario.model === name ? e.scenario.values : {});
      break;
    case "problem": {
      values = { ...(given ?? (e.problem.name === name ? e.problem.values : {})) };
      for (const hidden of ["solver", "convert_dose_objectives", "max_iter", "display"]) delete values[hidden];
      break;
    }
    case "solver": {
      if (given) values = { ...given };
      else {
        values = e.solver.name === name ? { ...e.solver.values } : {};
        for (const key of ["max_iter", "display"]) {
          if (e.problem.values[key] !== undefined && values[key] === undefined) values[key] = e.problem.values[key];
        }
      }
      break;
    }
    default:
      throw new MethodError("ValueError", `Unknown form kind '${String(kind)}'`);
  }
  const form = recordedForm(kind, name, mode, structuredClone(values));
  if (!form) throw new MethodError("ValueError", `Unknown ${kind.replace("_", " ")} '${name}'`);
  return form;
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
  const fields = planOptionsFor(base.radiation_mode, base.machine, base.engine, false).fields;
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
    next.machine = machinesFor(next.radiation_mode)[0];
    next.engine = defaultEngine(next.radiation_mode);
    const bio = defaultBioOption(next.radiation_mode);
    next.bio_model = bio?.model ?? "none";
    e.bio = bio ? { option: String(bio.value), values: structuredClone(bio.parameters) } : noBio();
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

  const context = planOptionsFor(next.radiation_mode, next.machine, next.engine, false);
  if (p.engine !== undefined) {
    const en = context.engines.find((x) => x.value === p.engine);
    if (p.engine !== null && !en) {
      details.push({ field: "engine", message: `Dose engine '${p.engine}' is not available for ${next.radiation_mode}` });
    } else if (en?.enabled === false) {
      throw precondition(`${en.label}: ${en.reason ?? "not available"}`, "not_available");
    }
    next.engine = p.engine;
  }
  let engineValues: Record<string, unknown> | undefined;
  if (p.engine_config) {
    engineValues = structuredClone(p.engine_config.values ?? {});
    details.push(...validateValues(formFields("engine", next.engine, next.radiation_mode), engineValues, "engine."));
  }

  const bio = p.bio_model !== undefined ? resolveBio(p.bio_model, next.radiation_mode, details) : e.bio;
  const bioOption = bioOptionById(bio.option);
  if (bioOption) {
    if (!context.bio_models.some((o) => o.value === bioOption.value)) {
      details.push({ field: "bio_model", message: `${bioOption.label} is not available for ${next.radiation_mode}` });
    } else details.push(...validateValues(formFields("bio_model", bioOption.model, next.radiation_mode), bio.values, "bio_model."));
  }

  let scenario = e.scenario;
  if (p.scenarios) {
    const known = SCENARIO_MODELS.some((m) => m.value === p.scenarios!.model);
    if (!known && p.scenarios.model !== "explicit") details.push({ field: "scenarios.model", message: `Unknown scenario model '${p.scenarios.model}'` });
    if (known) details.push(...validateValues(formFields("scenario", p.scenarios.model, ""), p.scenarios.values ?? {}, "scenario."));
    scenario = { model: p.scenarios.model, values: structuredClone(p.scenarios.values ?? {}) };
  }
  let problem = e.problem;
  if (p.problem) {
    if (!PROBLEMS.some((x) => x.value === p.problem!.name)) details.push({ field: "problem.name", message: `Unknown planning problem '${p.problem.name}'` });
    else details.push(...validateValues(formFields("problem", p.problem.name, ""), p.problem.values ?? {}, "problem."));
    // Settings the form does not show are kept unless sent.
    const kept = p.problem.name === e.problem.name ? shownProblemValues(e.problem.name, e.problem.values, false) : {};
    problem = { name: p.problem.name, values: { ...structuredClone(kept), ...structuredClone(p.problem.values ?? {}) } };
  }
  let solver = e.solver;
  if (p.solver !== undefined) {
    const spec = typeof p.solver === "string" ? { name: p.solver, values: {} } : p.solver;
    const sv = SOLVERS.find((x) => x.value === spec.name);
    if (sv && sv.enabled !== false) details.push(...validateValues(formFields("solver", spec.name, ""), spec.values ?? {}, "solver."));
    solver = { name: spec.name, values: structuredClone(spec.values ?? {}) };
  }

  if (details.length) throw validationError(details);

  if (p.dose_convention !== undefined) e.doseConvention = p.dose_convention;
  if (engineValues && next.engine) e.engineValues[next.engine] = engineValues;
  e.bio = bio;
  next.bio_model = bioOptionById(bio.option)?.model ?? next.bio_model ?? "none";
  e.scenario = scenario;
  e.problem = problem;
  e.solver = solver;
  s.plan = next;
  ctx.touch(s, "plan");
  ctx.log(s, "info", `Plan updated: ${next.radiation_mode}, ${next.gantry_angles.length} beam(s)`);
  return ctx.publish(s);
}

function resolveBio(spec: PlanSetParams["bio_model"], mode: string, details: ValidationDetail[]): PlanExt["bio"] {
  if (spec === null || spec === undefined) return noBio();
  if (typeof spec === "string") {
    const option = bioOptionOfModel(mode, spec) ?? bioOptionById(spec);
    if (!option) {
      details.push({ field: "bio_model", message: `Unknown biological model '${spec}'` });
      return { option: null, values: {} };
    }
    return { option: String(option.value), values: structuredClone(option.parameters) };
  }
  const option = (spec.option ? bioOptionById(spec.option) : undefined) ?? bioOptionOfModel(mode, spec.model);
  if (!option) {
    details.push({ field: "bio_model.model", message: `Unknown biological model '${spec.model}'` });
    return { option: null, values: {} };
  }
  return { option: String(option.value), values: structuredClone(spec.values ?? option.parameters) };
}

// ---------------------------------------------------------------------------
// Tissues
// ---------------------------------------------------------------------------

/**
 * Rows come back as sent, without issues or derived values: how tissues are checked against
 * base data (and derived, matched, relabelled) is the app layer's, and the recorded phantom
 * has no issues with the bundled data. Only `apply_pick` is simulated (the pick's model and
 * values replace the tissue), and a row names the pick whose model and values it holds.
 */
function tissueEvaluate(p: TissueEvaluateParams): TissueEvaluateResult {
  const op = p.op ?? "none";
  const pick = op === "apply_pick" ? TISSUE_SCHEMA.picks.find((x) => x.id === p.pick) : undefined;
  const targets = p.targets?.length ? p.targets : p.rows.map((r) => r.structure);
  const sameValues = (a: Record<string, unknown>, b: Record<string, unknown>) => Object.keys(b).every((k) => (a[k] ?? null) === (b[k] ?? null));
  return {
    rows: p.rows.map((r) => {
      const applied = pick && targets.includes(r.structure);
      const tissue: TissueDraft = applied
        ? { model: pick.model, values: structuredClone(pick.values), sources: {}, cell_line: pick.cell_line ?? null, cell_line_source: pick.cell_line_source ?? null }
        : { ...structuredClone(r.tissue), sources: structuredClone(r.tissue.sources ?? {}) };
      const equal = TISSUE_SCHEMA.picks.filter((x) => x.model === tissue.model && sameValues(tissue.values, x.values));
      const held = (tissue.cell_line_source ? equal.find((x) => x.cell_line === tissue.cell_line && x.cell_line_source === tissue.cell_line_source) : undefined) ?? equal[0];
      return { structure: r.structure, tissue, issues: [], comparison: null, pick: (applied ? pick.id : held?.id) ?? null };
    }),
    notes: op === "none" || pick ? [] : [`Mock backend: the tissue operation "${op}" is not simulated`],
    can_match: false,
    can_relabel: [],
    can_derive: false,
  };
}

// ---------------------------------------------------------------------------
// Objectives
// ---------------------------------------------------------------------------

function findStructure(s: MockSession, ref: unknown): MockStructure {
  const list = ext(s).structures;
  return list[resolveStructure(list, ref)];
}

function objectiveType(name: string) {
  const t = OBJECTIVES.objectives.find((o) => o.name === name);
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

const quantityKnown = (quantity: string) => OBJECTIVES.quantities.some((q) => q.value === quantity);

export const planHandlers: MockHandlers = {
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

  // As in the browser: no model on disk and no hub client, so nothing to suggest.
  "plan.form_choices"(p) {
    return { id: `${String(p.kind)}:${String(p.name)}`, choices: {} };
  },

  "plan.set"(p, ctx) {
    return planSet(ctx, ctx.session(), p as PlanSetParams);
  },

  "plan.tissue_check"() {
    return { marker: "", issues: [] };
  },

  async "cst.get"(p, ctx) {
    const s = ctx.session();
    if (!s.phantom) throw new MethodError("LookupError", "No structures loaded");
    const list = ext(s).structures;
    const wanted = Array.isArray(p.names) ? new Set((p.names as unknown[]).map((n) => resolveStructure(list, n))) : null;
    const own = list.map((st, i) => structureDetail(st, i));
    let base: StructureDetail[] | null = null;
    try {
      base = ((await ctx.fallback()) as { structures: StructureDetail[] }).structures;
    } catch {
      base = null;
    }
    const merged = (base ?? own).map((b) => ({ ...b, ...pickOwn(own[b.index]) }));
    return { structures: merged.filter((st) => !wanted || wanted.has(st.index)) };
  },

  async "cst.update_structure"(p, ctx) {
    const s = ctx.session();
    const changes = (p.changes ?? {}) as { type?: string; color?: Vec3; tissue?: TissueDraft; overlap_priority?: number };
    const st = findStructure(s, p.name);
    // Written before the viewer mock publishes its state, whose sync reads the structures back.
    const before = structuredClone(st);
    if (changes.type !== undefined) st.type = changes.type;
    if (changes.color !== undefined) st.color = changes.color;
    if (changes.overlap_priority !== undefined) st.overlap_priority = Number(changes.overlap_priority);
    if (changes.tissue !== undefined) st.tissue = structuredClone(changes.tissue);
    let result: { state: SessionState; structure: StructureDetail } | null = null;
    try {
      result = (await ctx.fallback()) as { state: SessionState; structure: StructureDetail };
    } catch (err) {
      if (!(err instanceof MethodError && err.type === "NotImplementedError")) {
        Object.assign(st, before);
        throw err;
      }
    }
    const index = ext(s).structures.indexOf(st);
    const mine = structureDetail(st, index);
    return { state: ctx.publish(s), structure: result ? { ...result.structure, ...pickOwn(mine) } : mine };
  },

  "tissue.schema"() {
    return TISSUE_SCHEMA;
  },

  "tissue.evaluate"(p) {
    return tissueEvaluate(p as unknown as TissueEvaluateParams);
  },

  "tissue.apply"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const rows = (p.rows ?? []) as Array<{ structure: number | string; tissue: TissueDraft }>;
    const targets = rows.map((row) => findStructure(s, row.structure));
    rows.forEach((row, i) => (targets[i].tissue = structuredClone(row.tissue)));
    bumpCst(s);
    ctx.log(s, "info", `Tissue parameters of ${rows.length} structure(s) updated`);
    return { state: ctx.publish(s) };
  },

  "objectives.catalog"(_p, ctx) {
    const s = ctx.session();
    return {
      ...OBJECTIVES,
      reference_images: mockReferenceImages(s),
    };
  },

  "objectives.add"(p, ctx) {
    const s = ctx.session();
    writeCheck(ctx);
    const st = findStructure(s, p.structure);
    const obj = newObjective(OBJECTIVES.default_by_type[st.type] ?? OBJECTIVES.default_by_type["*"]);
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
    if (changes.quantity !== undefined && !quantityKnown(changes.quantity)) {
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
        if (!key || !mockResultKeys(s).includes(key)) {
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
    if (!quantityKnown(quantity)) throw validationError([{ field: "quantity", message: `Unknown quantity '${quantity}'` }]);
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
      const result = (await ctx.fallback()) as Record<string, unknown>;
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
    throw precondition(BROWSER_REASONS.pause, "not_available");
  },

  "session.resume"() {
    throw precondition(BROWSER_REASONS.pause, "not_available");
  },
};

/** After a load: the session profile's default plan (the core applies the native one). */
function applyProfileDefaults(ctx: MockContext, result: unknown): unknown {
  const s = ctx.session();
  const e = ext(s);
  if (s.plan && e.profile === "web") {
    s.plan = { ...s.plan, dose_grid_resolution: [...WEB_GRID] };
    e.doseConvention = "total";
    e.bio = noBio();
    e.engineValues = {};
    e.scenario = { model: "nomScen", values: {} };
    e.problem = { name: "nonlin_fluence", values: {} };
    e.solver = { name: null, values: {} };
    return { ...(result as object), ...ctx.publish(s) };
  }
  return result;
}

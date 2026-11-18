/**
 * Draft model of the plan form (Qt `PlanWidget`): the form's fields, the per-field
 * "modified" snapshot (`_capture_state`), applying the backend's bio-model selection
 * (`plan.options.bio_selection`, the rules of Qt `_refresh_bio_models`) and the `plan.set`
 * request of an Apply (sending only what changed so plan settings the form cannot show survive).
 */
import type { FieldDescriptor, PlanSummary, Vec3 } from "../../api/protocol";
import {
  PLAN_MACHINE,
  type BioModelChoice,
  type BioSelection,
  type CurrentBio,
  type DoseConvention,
  type PlanDetail,
  type PlanOptions,
  type PlanSetParams,
} from "../../api/protocol/plan";

export interface SolverSpec {
  name: string;
  values: Record<string, unknown>;
}

export interface ProblemProps {
  values: Record<string, unknown>;
  /** The solver entry of `prop_opt` (null: the problem's default solver). */
  solver: SolverSpec | null;
}

export interface PlanDraft {
  radiation_mode: string;
  machine: string;
  engine: string | null;
  engineProps: Record<string, Record<string, unknown>>;
  /** Selected bio-model option id; null keeps the plan's unsupported model. */
  bioOption: string | null;
  bioModel: string;
  bioParams: Record<string, Record<string, unknown>>;
  fractions: string;
  doseConvention: DoseConvention;
  gantry: string;
  couch: string;
  bixel: string;
  isoAuto: boolean;
  iso: string;
  /** The plan's per-beam isocenters, kept on Apply until the iso setting changes. */
  perBeamIso: Vec3[] | null;
  scenario: string;
  scenarioProps: Record<string, Record<string, unknown>>;
  problem: string;
  problemProps: Record<string, ProblemProps>;
  res: [string, string, string];
}

/** Model parameters choosing the base data a model reads (`SELECTION_PARAMETERS`). */
export const SELECTION_PARAMETERS = ["data_model", "kernel_set", "quantity_variants", "quantity_table_name"];

export const MODIFIED_NOTE = "Modified — not applied";

export function formatNumberList(values: readonly number[]): string {
  return values.map((v) => String(Number(v.toFixed(6)))).join(" ");
}

export function parseNumberList(text: string): number[] {
  const parts = text.trim().split(/[\s,;]+/).filter(Boolean);
  const values = parts.map(Number);
  const bad = parts.find((_, i) => !Number.isFinite(values[i]));
  if (bad !== undefined) throw new Error(`could not convert string to float: '${bad}'`);
  return values;
}

function safeParse(text: string): number[] | string {
  try {
    return parseNumberList(text);
  } catch {
    return text;
  }
}

const clone = <T,>(v: T): T => JSON.parse(JSON.stringify(v ?? null)) as T;

/** Form fields shown for a plan (`_sync_from_workspace`). */
export function draftFromPlan(plan: PlanSummary | null, detail: PlanDetail | null, fallbackGrid: Vec3 = [5, 5, 5]): PlanDraft {
  const iso = plan?.iso_center ?? null;
  const perBeam = Array.isArray(iso?.[0]) ? (iso as Vec3[]) : null;
  const uniquePerBeam = perBeam && perBeam.every((p) => JSON.stringify(p) === JSON.stringify(perBeam[0])) ? perBeam[0] : null;
  const single = perBeam ? uniquePerBeam : (iso as Vec3 | null);
  const engine = plan?.engine ?? null;
  const bioModel = detail?.bio.model ?? plan?.bio_model ?? "none";
  const problem = detail?.problem.name ?? "nonlin_fluence";
  return {
    radiation_mode: plan?.radiation_mode ?? "photons",
    machine: detail?.machine_source === "plan" ? PLAN_MACHINE : (plan?.machine ?? "Generic"),
    engine,
    engineProps: engine ? { [engine]: clone(detail?.engine_config.values ?? {}) } : {},
    bioOption: detail ? detail.bio.option : bioModel,
    bioModel,
    bioParams: { [bioModel]: clone(detail?.bio.values ?? {}) },
    fractions: String(plan?.num_of_fractions ?? 30),
    doseConvention: detail?.dose_convention ?? "total",
    gantry: formatNumberList(plan?.gantry_angles ?? [0]),
    couch: formatNumberList(plan?.couch_angles ?? [0]),
    bixel: (plan?.bixel_width ?? 5).toFixed(2),
    isoAuto: !single,
    iso: single ? formatNumberList(single) : perBeam ? "multiple iso centers" : "0 0 0",
    perBeamIso: perBeam && !uniquePerBeam ? perBeam : null,
    scenario: detail?.scenario.model ?? "nomScen",
    scenarioProps: { [detail?.scenario.model ?? "nomScen"]: clone(detail?.scenario.values ?? {}) },
    problem,
    problemProps: {
      [problem]: {
        values: clone(detail?.problem.values ?? {}),
        solver: detail?.solver.name ? { name: detail.solver.name, values: clone(detail.solver.values) } : null,
      },
    },
    res: (plan?.dose_grid_resolution ?? fallbackGrid).map((v) => v.toFixed(2)) as [string, string, string],
  };
}

const json = (v: unknown) => JSON.stringify(v ?? null, Object.keys(flatKeys(v)).sort());

function flatKeys(v: unknown, out: Record<string, true> = {}): Record<string, true> {
  if (v && typeof v === "object") {
    for (const [k, x] of Object.entries(v as Record<string, unknown>)) {
      out[k] = true;
      flatKeys(x, out);
    }
  }
  return out;
}

const nonSelection = (params: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(params).filter(([k]) => !SELECTION_PARAMETERS.includes(k)));
const selection = (params: Record<string, unknown>) =>
  Object.fromEntries(Object.entries(params).filter(([k]) => SELECTION_PARAMETERS.includes(k)));

export type FieldKey =
  | "radiation"
  | "machine"
  | "engine"
  | "engine_props"
  | "problem"
  | "problem_props"
  | "bio_model"
  | "bio_params"
  | "fractions"
  | "dose_convention"
  | "gantry"
  | "couch"
  | "bixel"
  | "iso"
  | "scenario"
  | "scenario_props"
  | "res_x"
  | "res_y"
  | "res_z";

/** Snapshot of the tracked values (Qt `_capture_state`), as comparable strings. */
export function captureState(d: PlanDraft): Record<FieldKey, string> {
  const bioParams = d.bioParams[d.bioModel] ?? {};
  const num = (t: string) => {
    const v = Number(t);
    return Number.isFinite(v) && t.trim() !== "" ? String(v) : `text:${t}`;
  };
  return {
    radiation: d.radiation_mode,
    machine: d.machine,
    engine: d.engine ?? "",
    engine_props: json(d.engineProps[d.engine ?? ""] ?? {}),
    problem: d.problem,
    problem_props: json(d.problemProps[d.problem] ?? { values: {}, solver: null }),
    bio_model: json([d.bioModel, d.bioOption === null, d.bioOption, selection(bioParams)]),
    bio_params: json(nonSelection(bioParams)),
    fractions: num(d.fractions),
    dose_convention: d.doseConvention,
    gantry: JSON.stringify(safeParse(d.gantry)),
    couch: JSON.stringify(safeParse(d.couch)),
    bixel: num(d.bixel),
    iso: JSON.stringify([d.isoAuto, d.perBeamIso !== null, d.isoAuto ? null : d.iso]),
    scenario: d.scenario,
    scenario_props: json(d.scenarioProps[d.scenario] ?? {}),
    res_x: num(d.res[0]),
    res_y: num(d.res[1]),
    res_z: num(d.res[2]),
  };
}

export function modifiedKeys(d: PlanDraft, clean: Record<FieldKey, string>): Set<FieldKey> {
  const now = captureState(d);
  return new Set((Object.keys(now) as FieldKey[]).filter((k) => now[k] !== clean[k]));
}

/** Gantry and couch angle per beam (`_form_beam_angles`); throws on an invalid set. */
export function formBeamAngles(d: Pick<PlanDraft, "gantry" | "couch">): { gantry: number[]; couch: number[] } {
  const gantry = parseNumberList(d.gantry);
  if (!gantry.length) throw new Error("at least one gantry angle is required");
  let couch = parseNumberList(d.couch);
  if (couch.length === 1) couch = gantry.map(() => couch[0]);
  else if (couch.length !== 0 && couch.length !== gantry.length) throw new Error("number of couch angles must be 1 or match gantry angles");
  if (!couch.length) couch = gantry.map(() => 0);
  return { gantry, couch };
}

/** Couch field after a gantry edit (`_sync_couch_angles`); null when unchanged. */
export function syncedCouch(gantryText: string, couchText: string): string | null {
  try {
    const n = parseNumberList(gantryText).length;
    const couch = parseNumberList(couchText);
    if (n === 0 || couch.length === n) return null;
    const pad = couch.length === 1 ? couch[0] : 0;
    return formatNumberList([...couch, ...Array(n).fill(pad)].slice(0, n));
  } catch {
    return null;
  }
}

/** `count` equally spaced gantry angles from 0° (`_on_num_beams_changed`). */
export function equidistantAngles(count: number): number[] {
  return Array.from({ length: count }, (_, i) => Number(((360 * i) / count).toFixed(2)));
}

export function fieldOf(options: PlanOptions | null, name: string): FieldDescriptor | undefined {
  return options?.fields.find((f) => f.name === name);
}

function numberIn(text: string, label: string, f: FieldDescriptor | undefined, fallback: { min: number; max: number }, integer = false): number {
  const v = Number(text);
  if (text.trim() === "" || !Number.isFinite(v)) throw new Error(`${label}: could not convert string to float: '${text}'`);
  if (integer && !Number.isInteger(v)) throw new Error(`${label}: must be an integer`);
  const min = f?.min ?? fallback.min;
  const max = f?.max ?? fallback.max;
  if (v < min || v > max) throw new Error(`${label}: must be between ${min} and ${max}`);
  return v;
}

/**
 * The `plan.set` params of an Apply: the fields that changed since the form was clean.
 * A changed radiation mode sends machine, engine and model too (the backend resets them to the
 * mode's defaults otherwise); the scenario model and problem settings the backend keeps itself
 * (v1 amendment 22). Throws a `ValueError`-like message for invalid entries.
 */
export function planSetParams(d: PlanDraft, clean: Record<FieldKey, string>, options: PlanOptions | null): PlanSetParams {
  const mod = modifiedKeys(d, clean);
  const out: PlanSetParams = {};
  const modeChanged = mod.has("radiation");
  if (modeChanged) out.radiation_mode = d.radiation_mode;
  if (modeChanged || mod.has("machine")) out.machine = d.machine || "Generic";
  if (modeChanged || mod.has("engine") || mod.has("engine_props")) {
    out.engine = d.engine;
    if (d.engine) out.engine_config = { values: clone(d.engineProps[d.engine] ?? {}) };
  }
  if ((modeChanged || mod.has("bio_model") || mod.has("bio_params")) && d.bioOption !== null) {
    out.bio_model = { model: d.bioModel, option: d.bioOption, values: clone(d.bioParams[d.bioModel] ?? {}) };
  }
  if (mod.has("fractions")) {
    out.num_of_fractions = numberIn(d.fractions, "num_of_fractions", fieldOf(options, "num_of_fractions"), { min: 1, max: 1000 }, true);
  }
  if (mod.has("dose_convention")) out.dose_convention = d.doseConvention;
  let beams: { gantry: number[]; couch: number[] } | null = null;
  if (mod.has("gantry") || mod.has("couch")) {
    beams = formBeamAngles(d);
    out.gantry_angles = beams.gantry;
    out.couch_angles = beams.couch;
  }
  if (mod.has("bixel")) out.bixel_width = numberIn(d.bixel, "bixel_width", fieldOf(options, "bixel_width"), { min: 0.1, max: 100 });
  if (!d.isoAuto && (mod.has("iso") || modeChanged)) {
    const iso = parseNumberList(d.iso);
    if (iso.length !== 3) throw new Error("iso center requires exactly three coordinates (x y z)");
    out.iso_center = iso as Vec3;
  } else if (d.isoAuto && d.perBeamIso && beams && d.perBeamIso.length !== beams.gantry.length) {
    throw new Error(
      `the plan's ${d.perBeamIso.length} per-beam iso centers do not match ${beams.gantry.length} beams; enter an iso center, or untick and re-tick Auto to use the target center of mass`,
    );
  } else if (d.isoAuto && !d.perBeamIso && mod.has("iso")) {
    out.iso_center = null;
  }
  if (mod.has("scenario") || mod.has("scenario_props")) {
    out.scenarios = { model: d.scenario, values: clone(d.scenarioProps[d.scenario] ?? {}) };
  }
  if (mod.has("problem") || mod.has("problem_props")) {
    const props = d.problemProps[d.problem] ?? { values: {}, solver: null };
    out.problem = { name: d.problem, values: clone(props.values) };
    if (props.solver) out.solver = Object.keys(props.solver.values).length ? clone(props.solver) : props.solver.name;
  }
  if (mod.has("res_x") || mod.has("res_y") || mod.has("res_z")) {
    const f = fieldOf(options, "dose_grid_resolution");
    out.dose_grid_resolution = d.res.map((t, i) => numberIn(t, `dose_grid_resolution.${"xyz"[i]}`, f, { min: 0.1, max: 50 })) as Vec3;
  }
  return out;
}

/** Plan field names of `plan.set` (and `ValidationError.details`) → form field key. */
export function fieldKeyOf(name: string): FieldKey | null {
  const head = name.split(".")[0];
  // v1 amendment 10: algorithm form fields come as "<kind>.<field>".
  if (name.includes(".")) {
    const form: Record<string, FieldKey> = {
      engine: "engine_props",
      engine_config: "engine_props",
      bio_model: "bio_params",
      scenario: "scenario_props",
      scenarios: "scenario_props",
      problem: "problem_props",
      solver: "problem_props",
    };
    if (form[head]) return form[head];
  }
  const map: Record<string, FieldKey> = {
    radiation_mode: "radiation",
    machine: "machine",
    engine: "engine",
    engine_config: "engine_props",
    bio_model: "bio_model",
    num_of_fractions: "fractions",
    dose_convention: "dose_convention",
    gantry_angles: "gantry",
    couch_angles: "couch",
    bixel_width: "bixel",
    iso_center: "iso",
    scenarios: "scenario_props",
    problem: "problem_props",
    solver: "problem_props",
    dose_grid_resolution: "res_x",
  };
  return map[head] ?? null;
}

// ---------------------------------------------------------------------------
// Option re-listing (radiation mode / machine / engine changes)
// ---------------------------------------------------------------------------

export function pickMachine(current: string, options: PlanOptions): string {
  const names = options.machines.map((c) => String(c.value));
  if (names.includes(current)) return current;
  if (names.includes("Generic")) return "Generic";
  return names[0] ?? (current || "Generic");
}

export function pickIdentifier(current: string | null, choices: PlanOptions["engines"]): string | null {
  if (current && choices.some((c) => c.value === current && c.enabled !== false)) return current;
  return String(choices.find((c) => c.enabled !== false)?.value ?? choices[0]?.value ?? "") || null;
}

type BioDraft = Pick<PlanDraft, "bioOption" | "bioModel" | "bioParams">;

/** A model's parameters with an option's base-data selection taken over (Qt `_apply_option_parameters`). */
export function optionParams(choice: BioModelChoice, params: Record<string, unknown> | undefined): Record<string, unknown> {
  return { ...nonSelection(params ?? {}), ...clone(choice.parameters) };
}

/**
 * The draft's biological model selection as `plan.options {current_bio}`: what the backend
 * needs to choose the entry after a radiation mode, machine or engine change.
 */
export function currentBio(d: BioDraft, mode: string, options: PlanOptions | null, unsupportedLabel?: string | null): CurrentBio {
  const label =
    d.bioOption === null
      ? (unsupportedLabel ?? d.bioModel).split(" (not supported")[0]
      : (options?.bio_models.find((o) => o.value === d.bioOption)?.label ?? d.bioModel);
  return { radiation_mode: mode, model: d.bioModel, option: d.bioOption, values: clone(nonSelection(d.bioParams[d.bioModel] ?? {})), label };
}

/** The draft's bio fields with the backend's `bio_selection` applied (unchanged without one). */
export function applyBioSelection(d: BioDraft, selection: BioSelection | undefined, options: PlanOptions): BioDraft {
  if (!selection) return { bioOption: d.bioOption, bioModel: d.bioModel, bioParams: d.bioParams };
  if (selection.unsupported) return { bioOption: null, bioModel: selection.unsupported.model, bioParams: d.bioParams };
  const choice = options.bio_models.find((o) => o.value === selection.option);
  const model = selection.model;
  const kept = d.bioParams[model];
  const params = choice && (selection.replaced || !kept) ? optionParams(choice, kept) : clone(kept ?? {});
  return { bioOption: selection.option, bioModel: model, bioParams: { ...d.bioParams, [model]: params } };
}

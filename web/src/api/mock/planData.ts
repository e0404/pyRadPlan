/**
 * Catalogues of the mock backend, read from the responses of `pyRadPlan.app` recorded in
 * `./fixtures`, which `scripts/gen_mock_fixtures.py` writes (`npm run fixtures`, run before every
 * script that needs them; they are not committed). The recordings describe the browser runtime.
 */
import type { Choice, FieldDescriptor, FormSchema, ObjectiveData, TissueDraft } from "../protocol";
import type { BioModelChoice, FormKind, ObjectivesCatalog, PlanOptions, TissueSchema } from "../protocol/plan";
import appInfo from "./fixtures/app_info.json";
import lockedGridFields from "./fixtures/plan_fields_locked_grid.json";
import forms from "./fixtures/plan_forms.json";
import optionsFixture from "./fixtures/plan_options.json";
import objectivesCatalog from "./fixtures/objectives_catalog.json";
import phantomStructures from "./fixtures/phantom_structures.json";
import tissueSchema from "./fixtures/tissue_schema.json";

type Context = Pick<PlanOptions, "machines" | "engines" | "bio_models" | "default_bio_model" | "beam_presets" | "sequencers">;
type Shared = Pick<PlanOptions, "radiation_modes" | "scenario_models" | "problems" | "solvers" | "default_solver" | "fields">;

const shared = optionsFixture.shared as unknown as Shared;
const contexts = optionsFixture.contexts as unknown as Record<string, Context>;

export const REGISTRY = appInfo as unknown as {
  phantoms: string[];
  machines: Record<string, string[]>;
  engines: Record<string, string[]>;
  solvers: string[];
  quantities: Array<{ identifier: string; name: string; meta_label: string; unit: string }>;
};

export const RADIATION_MODES = shared.radiation_modes.map((m) => String(m.value));
export const DEFAULT_SOLVER = shared.default_solver;
export const SCENARIO_MODELS: Choice[] = shared.scenario_models;
export const PROBLEMS: Choice[] = shared.problems;
export const SOLVERS = shared.solvers;
export const QUANTITIES = REGISTRY.quantities;
export const OBJECTIVES = objectivesCatalog as unknown as ObjectivesCatalog;
export const PHANTOM_STRUCTURES = phantomStructures as unknown as Record<
  string,
  Record<string, { overlap_priority: number; tissue: TissueDraft; objectives: Omit<ObjectiveData, "index">[] }>
>;
export const TISSUE_SCHEMA = tissueSchema as unknown as TissueSchema;

export const machinesFor = (mode: string): string[] => REGISTRY.machines[mode] ?? ["Generic"];

export function defaultEngine(mode: string): string | null {
  return REGISTRY.engines[mode]?.[0] ?? null;
}

const ofMode = (mode: string) => Object.entries(contexts).filter(([key]) => key.startsWith(`${mode}|`));

/** The recorded options of a radiation mode, machine and engine (the closest recorded ones when it was not). */
function context(mode: string, machine: string, engine: string | null): Context {
  const exact = contexts[`${mode}|${machine}|${engine}`];
  if (exact) return exact;
  const candidates = ofMode(mode);
  const found = candidates.find(([key]) => key.startsWith(`${mode}|${machine}|`)) ?? candidates[0] ?? Object.entries(contexts)[0];
  return found[1];
}

export function planOptionsFor(mode: string, machine: string, engine: string | null, lockedGrid: boolean): Omit<PlanOptions, "bio_selection"> {
  return {
    ...shared,
    ...context(mode, machine, engine),
    radiation_mode: mode,
    fields: lockedGrid ? (lockedGridFields as unknown as FieldDescriptor[]) : shared.fields,
  };
}

/** A biological model entry of any recorded context by its option id. */
export function bioOptionById(id: string | null): BioModelChoice | undefined {
  if (id === null) return undefined;
  for (const c of Object.values(contexts)) {
    const found = c.bio_models.find((o) => o.value === id);
    if (found) return found;
  }
  return undefined;
}

/** The mode's entry of a biological model, as offered with its first machine and default engine. */
export function bioOptionOfModel(mode: string, model: string): BioModelChoice | undefined {
  const primary = context(mode, machinesFor(mode)[0], defaultEngine(mode));
  return [primary, ...ofMode(mode).map(([, c]) => c)].flatMap((c) => c.bio_models).find((o) => o.model === model);
}

export function defaultBioOption(mode: string): BioModelChoice | undefined {
  const primary = context(mode, machinesFor(mode)[0], defaultEngine(mode));
  return primary.bio_models.find((o) => o.value === primary.default_bio_model);
}

/** The recorded form of an algorithm with `values` laid over its defaults (`undefined`: no such algorithm). */
export function recordedForm(kind: FormKind, name: string, mode: string, values: Record<string, unknown>): FormSchema | undefined {
  const key = kind === "engine" || kind === "bio_model" || kind === "sequencer" ? `${kind}:${mode}:${name}` : `${kind}:${name}`;
  const form = (forms as unknown as Record<string, FormSchema>)[key];
  if (!form) return undefined;
  const { notes, ...rest } = form;
  return { ...rest, values: { ...form.values, ...values }, explicit: Object.keys(values), ...(notes?.length ? { notes } : {}) };
}

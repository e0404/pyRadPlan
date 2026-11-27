/** Pure helpers of the objectives table (Qt `OptimizationWidget`). */
import type { Choice, FieldDescriptor, ObjectiveData, StructureDetail } from "../../api/protocol";
import type { ObjectiveType, ObjectivesCatalog } from "../../api/protocol/plan";
import type { ComboOption } from "../forms/Combo";

export interface ObjectiveRow {
  structure: StructureDetail;
  objective: ObjectiveData;
  /** First row of its VOI (shows the VOI label). */
  first: boolean;
}

/** One row per objective, grouped by VOI in structure order. */
export function objectiveRows(structures: readonly StructureDetail[]): ObjectiveRow[] {
  return structures.flatMap((structure) => structure.objectives.map((objective, i) => ({ structure, objective, first: i === 0 })));
}

/** The quantity every objective uses, or null for none or a mix (`_shared_quantity`). */
export function sharedQuantity(structures: readonly StructureDetail[]): string | null {
  const used = new Set(structures.flatMap((s) => s.objectives.map((o) => o.quantity)));
  return used.size === 1 ? [...used][0] : null;
}

/**
 * Type entries of a row. An objective the catalog does not list was created in code
 * (`Objective.selectable` is false): its own name is the only entry and the type is locked.
 */
export function typeOptions(types: readonly ObjectiveType[], name: string): { options: ComboOption[]; locked: boolean } {
  if (!types.some((t) => t.name === name)) return { options: [{ key: name, label: name }], locked: true };
  return { options: types.map((t) => ({ key: t.name, label: t.name })), locked: false };
}

/**
 * Robustness entries of an objective the catalog does not list (`Objective.available_robustness`):
 * the scenario-aware operator alone where it is in use, every other operator otherwise.
 */
export function unlistedRobustness(all: readonly Choice[], current: string): Choice[] {
  return all.filter((c) => (c.value === "scenario_stack") === (current === "scenario_stack"));
}

/**
 * Robustness entries of an objective type (*fallback* for a type the catalog does not list); a
 * stored value the type does not list is added while the type offers a choice (a scenario-aware
 * objective has a single fixed option).
 */
export function robustnessOptions(
  type: ObjectiveType | undefined,
  current: string,
  fallback: readonly Choice[] = [],
): { options: ComboOption[]; current: string | null } {
  const list = type?.robustness ?? fallback;
  const options: ComboOption[] = list.map((c) => ({ key: String(c.value), label: c.label, identifier: c.identifier ?? String(c.value) }));
  if (options.length > 1 && !options.some((o) => o.key === current)) options.push({ key: current, label: current, identifier: current });
  if (options.length <= 1) return { options, current: options[0]?.key ?? null };
  return { options, current };
}

/** Display unit of a parameter: its own, or the objective quantity's for dose references. */
export function parameterUnit(f: FieldDescriptor, quantity: string, unitOf: (quantity: string) => string): string | undefined {
  if (f.unit) return f.unit;
  if (f.unit_from === "quantity") return unitOf(quantity) || undefined;
  return undefined;
}

/**
 * Image-reference dropdown: the result images, with the current value matched by its result key;
 * a custom value (not held in the workspace) is kept as a summarized first entry.
 */
export function referenceChoices(catalog: ObjectivesCatalog, value: unknown): { options: ComboOption[]; current: string } {
  const options: ComboOption[] = catalog.reference_images.map((c) => ({
    key: String(c.value),
    label: c.label,
    identifier: c.identifier ?? String(c.value),
  }));
  const ref = value && typeof value === "object" ? (value as { $result?: string; $custom?: string }) : {};
  if (ref.$result && options.some((o) => o.key === ref.$result)) return { options, current: ref.$result };
  const label = ref.$custom ?? (ref.$result ? `${ref.$result} (not in the workspace)` : "Custom image");
  return { options: [{ key: "$custom", label }, ...options], current: "$custom" };
}

/**
 * VOI-reference dropdown (Qt `_build_voi_reference_combo`): the names of the structure set. A
 * value naming none of them (e.g. the objective's default) is kept as a marked first entry, so
 * the mismatch shows here instead of failing once the optimization starts.
 */
export function voiReferenceChoices(names: readonly string[], value: unknown): { options: ComboOption[]; current: string; missing: boolean } {
  const options: ComboOption[] = [...new Set(names)].map((name) => ({ key: name, label: name }));
  const current = typeof value === "string" ? value : String(value ?? "");
  if (options.some((o) => o.key === current)) return { options, current, missing: false };
  return { options: [{ key: current, label: `${current} (not in structures)` }, ...options], current, missing: true };
}

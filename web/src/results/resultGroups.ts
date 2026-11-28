/**
 * Component keys of the result collection, their grouping and display names, ported from
 * `pyRadPlan/quantities/_display_names.py` (`result_key`, `group_result_keys`) and
 * `_result_widget.py` (`QUANTITY_META`, `default_quantity_key`).
 * A v1 backend sends the groups in `state.results`; this is the fallback for v0 backends
 * (and what the mock uses to build `state.results`).
 */
import type { ComponentKind, ResultGroup } from "../api/protocol";

/** `QUANTITY_META`: colorbar / save-dialog label and unit per quantity identifier. */
export const QUANTITY_META: Record<string, [string, string]> = {
  physical_dose: ["Dose", "Gy"],
  let: ["LET", "keV/μm"],
  effect: ["Effect", ""],
  rbe_x_dose: ["RBE-weighted Dose", "Gy (RBE)"],
  alpha_dose: ["Alpha Dose", "Gy"],
  sqrt_beta_dose: ["Sqrt(Beta) Dose", "Gy½"],
  let_dose: ["LET·Dose", "Gy·keV/μm"],
};

/** `RTQuantity.name` per identifier (what the Qt selector and readout show). */
export const QUANTITY_NAMES: Record<string, string> = {
  physical_dose: "dose",
  rbe_x_dose: "RBExDose",
  let: "LETd",
  effect: "Effect",
  alpha_dose: "alphaDose",
  sqrt_beta_dose: "sqrtBetaDose",
  let_dose: "LETxDose",
};

const DEFAULT_KEYS = ["rbe_x_dose", "physical_dose", "Physical quantity", "physical_quantity"];

export function defaultQuantityKey(keys: readonly string[]): string | null {
  for (const candidate of DEFAULT_KEYS) if (keys.includes(candidate)) return candidate;
  return keys.length ? keys[0] : null;
}

const SUFFIXES: [string, ComponentKind][] = [
  ["_beam", "beam"],
  ["_scenario", "scenario"],
];

export function splitComponentSuffix(stem: string): [string, ComponentKind | ""] {
  for (const [suffix, kind] of SUFFIXES) {
    if (stem.endsWith(suffix) && stem.length > suffix.length) return [stem.slice(0, -suffix.length), kind];
  }
  return [stem, ""];
}

export function splitIndex(key: string): [string, number | null] {
  const m = /^(.*) (\d+)$/.exec(key);
  return m ? [m[1], Number(m[2])] : [key, null];
}

/** Separator between the result name and the entry key of a component key. */
export const RESULT_KEY_SEPARATOR = "/";

/** Component key of the entry `key` of the result `tag` ("" = the active result). */
export function resultKey(tag: string, key: string): string {
  return tag ? `${tag}${RESULT_KEY_SEPARATOR}${key}` : key;
}

/**
 * Split a component key into the result name ("" = the active result) and the entry key
 * (`pyRadPlan/quantities/_display_names.py:split_result_key`): the first `/` separates them.
 */
export function splitResultKey(key: string): [string, string] {
  const at = key.indexOf(RESULT_KEY_SEPARATOR);
  return at > 0 && at < key.length - 1 ? [key.slice(0, at), key.slice(at + 1)] : ["", key];
}

/** Suffix of the quantities recomputed from a sequenced fluence (`SEQUENCED_SUFFIX`). */
export const SEQUENCED_SUFFIX = "_sequenced";

/** `"<quantity>_sequenced"` -> the quantity and whether it is a sequenced one (`split_sequenced`). */
export function splitSequenced(identifier: string): [string, boolean] {
  return identifier.endsWith(SEQUENCED_SUFFIX) && identifier.length > SEQUENCED_SUFFIX.length
    ? [identifier.slice(0, -SEQUENCED_SUFFIX.length), true]
    : [identifier, false];
}

export function quantityName(base: string): string {
  const [quantity, sequenced] = splitSequenced(base);
  const name = QUANTITY_NAMES[quantity];
  if (name === undefined) return base;
  return sequenced ? `${name} (sequenced)` : name;
}

/** `QUANTITY_META` of an identifier; a sequenced quantity has the label and unit of its quantity. */
export function quantityMeta(base: string): [string, string] | undefined {
  return QUANTITY_META[base] ?? QUANTITY_META[splitSequenced(base)[0]];
}

/**
 * Group component keys `"[<result>/]<base>[_beam|_scenario][ <i>]"` by `(base, result)` like
 * `group_result_keys`. `scenarioLabels` gives the scenario labels per result name ("" = the
 * active result); groups of results it does not name get none.
 */
export function groupResultKeys(keys: readonly string[], scenarioLabels: Readonly<Record<string, readonly string[]>> = {}): ResultGroup[] {
  const ranks: Record<string, number> = { full: 0, beam: 1, other: 1, scenario: 2 };
  const entries = new Map<string, { base: string; tag: string; items: { order: [number, number]; label: string; key: string; kind: ComponentKind }[] }>();
  for (const key of keys) {
    const [tag, entry] = splitResultKey(key);
    const [stem, index] = splitIndex(entry);
    const [base, suffixKind] = splitComponentSuffix(stem);
    let kind: ComponentKind;
    let label: string;
    if (suffixKind) {
      kind = suffixKind;
      label = index === null ? suffixKind : `${suffixKind} ${index}`;
    } else if (index !== null) {
      kind = "other";
      label = `[${index}]`;
    } else {
      kind = "full";
      label = "full";
    }
    const id = `${base}\u0000${tag}`;
    if (!entries.has(id)) entries.set(id, { base, tag, items: [] });
    entries.get(id)!.items.push({ order: [ranks[kind], index ?? -1], label, key, kind });
  }
  const groups: ResultGroup[] = [];
  for (const { base, tag, items } of entries.values()) {
    items.sort((a, b) => a.order[0] - b.order[0] || a.order[1] - b.order[1]);
    const [meta, unit] = quantityMeta(base) ?? ["Quantity", ""];
    groups.push({
      base,
      tag,
      label: tag ? `${quantityName(base)} (${tag})` : quantityName(base),
      meta_label: meta,
      unit,
      components: items.map(({ label, key, kind }) => ({ key, label, kind })),
      scenario_labels: [...(scenarioLabels[tag] ?? [])],
    });
  }
  return groups;
}

/** Selector text of a component, with the scenario label (`"scenario 2: setup (...)"`). */
export function componentText(label: string, kind: ComponentKind, scenarioLabels: readonly string[]): string {
  if (kind !== "scenario") return label;
  const index = /(\d+)$/.exec(label);
  const i = index ? Number(index[1]) : -1;
  return i >= 0 && i < scenarioLabels.length && scenarioLabels[i] ? `${label}: ${scenarioLabels[i]}` : label;
}

export interface LocatedKey {
  group: ResultGroup;
  component: ResultGroup["components"][number];
}

export function locateKey(groups: readonly ResultGroup[], key: string | null | undefined): LocatedKey | null {
  if (!key) return null;
  for (const group of groups) {
    const component = group.components.find((c) => c.key === key);
    if (component) return { group, component };
  }
  return null;
}

/** `"dose, beam 0"` style label of a key (Qt `QuantitySelector.current_label`). */
export function keyLabel(groups: readonly ResultGroup[], key: string): string {
  const found = locateKey(groups, key);
  if (!found) return key;
  return found.component.kind === "full" ? found.group.label : `${found.group.label}, ${found.component.label}`;
}

/** Keys of full quantities (one readout line each in the Qt viewer). */
export function fullKeys(groups: readonly ResultGroup[]): string[] {
  return groups.flatMap((g) => g.components.filter((c) => c.kind === "full").map((c) => c.key));
}

/** Result name of a snapshot tag (`re.sub(r"\W+", "_", tag.strip()).strip("_")`). */
export function sanitizeTag(tag: string): string {
  return tag.trim().replace(/\W+/g, "_").replace(/^_+|_+$/g, "");
}

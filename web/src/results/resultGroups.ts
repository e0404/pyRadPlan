/**
 * Result key grouping and display names, ported from `gui/widgets/_display_names.py`
 * (`group_result_keys`) and `_result_widget.py` (`QUANTITY_META`, `default_quantity_key`).
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

/**
 * Snapshot tags (`pyRadPlan/quantities/_display_names.py:_snapshot_tags`): the remainder after
 * a quantity identifier (and an optional `_beam` / `_scenario`), counted as a tag only when at
 * least two stems carry it (Qt ambiguity 15).
 */
export function snapshotTags(allStems: Set<string>, quantities: Set<string>): string[] {
  const identifiers = [...quantities].sort((a, b) => b.length - a.length);
  const canonical = new Set(identifiers.flatMap((n) => [n, `${n}_beam`, `${n}_scenario`]));
  const stems = [...allStems].filter((s) => !canonical.has(s));
  const candidates = new Set<string>();
  for (const stem of stems) {
    for (const id of identifiers) {
      if (stem === id) break;
      if (stem.startsWith(id + "_")) {
        let rest = stem.slice(id.length + 1);
        for (const kind of ["beam", "scenario"]) {
          if (rest.startsWith(kind + "_")) {
            rest = rest.slice(kind.length + 1);
            break;
          }
        }
        if (rest && rest !== "beam" && rest !== "scenario") candidates.add(rest);
        break;
      }
    }
  }
  const counts = new Map<string, number>();
  for (const tag of candidates) for (const stem of stems) if (stem.endsWith("_" + tag)) counts.set(tag, (counts.get(tag) ?? 0) + 1);
  return [...counts.entries()]
    .filter(([, n]) => n >= 2)
    .map(([t]) => t)
    .sort((a, b) => b.length - a.length);
}

export function quantityName(base: string): string {
  return QUANTITY_NAMES[base] ?? base;
}

/** Group result keys by `(base, tag)` like `group_result_keys`. */
export function groupResultKeys(keys: readonly string[], known: readonly string[] = Object.keys(QUANTITY_META)): ResultGroup[] {
  const quantities = new Set(known);
  const tags = snapshotTags(new Set(keys.map((k) => splitIndex(k)[0])), quantities);
  const ranks: Record<string, number> = { full: 0, beam: 1, other: 1, scenario: 2 };
  const entries = new Map<string, { base: string; tag: string; items: { order: [number, number]; label: string; key: string; kind: ComponentKind }[] }>();
  for (const key of keys) {
    let [stem, index] = splitIndex(key);
    let tag = "";
    if (!quantities.has(splitComponentSuffix(stem)[0])) {
      tag = tags.find((t) => stem.endsWith("_" + t) && stem.length > t.length + 1) ?? "";
    }
    if (tag) stem = stem.slice(0, -tag.length - 1);
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
    const [meta, unit] = QUANTITY_META[base] ?? ["Quantity", ""];
    groups.push({
      base,
      tag,
      label: tag ? `${quantityName(base)} (${tag})` : quantityName(base),
      meta_label: meta,
      unit,
      components: items.map(({ label, key, kind }) => ({ key, label, kind })),
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

/** Snapshot tag rule of the Qt workflow widget (`_on_save_result`). */
export function sanitizeTag(tag: string): string {
  return tag.trim().replace(/\W+/g, "_").replace(/^_+|_+$/g, "");
}

export const RESERVED_TAGS = ["beam", "scenario"];

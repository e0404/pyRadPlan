/**
 * Logic of the generic configuration form (`ConfigForm.tsx`), mirroring
 * `pyRadPlan/gui/widgets/_config_form.py`: descriptor → editor kind, text ↔ value,
 * client-side validation (bounds, types) and the values reported back (only the names in
 * `explicit` or edited; groups as nested override dicts, like `ConfigFormWidget.values()`).
 */
import type { Choice, FieldDescriptor, FormSchema } from "../../api/protocol";

export type EditorKind = "checkbox" | "spin" | "optional-number" | "combo" | "editable-combo" | "text" | "list" | "json" | "section" | "color" | "image";

export interface FieldState {
  /** Editor text (unused by checkboxes and combos). */
  text: string;
  /** Last valid value (what `values()` reports). */
  value: unknown;
  error: string | null;
  edited: boolean;
}

export type FormFields = Record<string, FieldState>;

/** Leaf descriptors of a field list, groups expanded (depth first). */
export function leafFields(fields: readonly FieldDescriptor[]): FieldDescriptor[] {
  return fields.flatMap((f) => (f.type === "group" ? leafFields(f.fields ?? []) : [f]));
}

export function editorKind(f: FieldDescriptor): EditorKind {
  switch (f.type) {
    case "bool":
      return "checkbox";
    case "int":
    case "float":
      return f.nullable ? "optional-number" : "spin";
    case "choice":
      return f.editable_choice ? "editable-combo" : "combo";
    case "str":
    case "voi_ref":
      return "text";
    case "number_list":
      return "list";
    case "group":
      return "section";
    case "color":
      return "color";
    case "image_ref":
      return "image";
    default:
      return "json";
  }
}

export function getPath(values: unknown, path: readonly string[]): unknown {
  let node = values;
  for (const key of path) {
    if (!node || typeof node !== "object" || Array.isArray(node)) return undefined;
    node = (node as Record<string, unknown>)[key];
  }
  return node;
}

function setPath(target: Record<string, unknown>, path: readonly string[], value: unknown): void {
  let node = target;
  for (const key of path.slice(0, -1)) {
    const child = node[key];
    if (!child || typeof child !== "object" || Array.isArray(child)) node[key] = {};
    node = node[key] as Record<string, unknown>;
  }
  node[path[path.length - 1]] = value;
}

const clone = <T,>(v: T): T => (v === undefined ? v : (JSON.parse(JSON.stringify(v)) as T));

/** Value shown for a field: the schema's (nested override) value, else the descriptor default. */
export function initialValue(schema: FormSchema, f: FieldDescriptor): unknown {
  const path = f.name.split(".");
  const v = path.length === 1 ? schema.values[f.name] : getPath(schema.values, path);
  if (v !== undefined) return clone(v);
  return clone(f.default ?? null);
}

function fmtNumber(f: FieldDescriptor, v: number): string {
  if (f.decimals !== undefined && f.type !== "int") return v.toFixed(f.decimals);
  return String(v);
}

/** Editor text of a value. */
export function formatValue(f: FieldDescriptor, v: unknown): string {
  if (v === null || v === undefined) return "";
  switch (f.type) {
    case "int":
    case "float":
      return typeof v === "number" ? fmtNumber(f, v) : String(v);
    case "number_list":
      return Array.isArray(v) ? v.map((x) => (typeof x === "number" ? String(x) : String(x))).join(" ") : String(v);
    case "str":
      return String(v);
    case "choice": {
      const c = f.choices?.find((x) => x.value === v);
      return c ? c.label : String(v);
    }
    case "json":
      try {
        return JSON.stringify(v);
      } catch {
        return String(v);
      }
    default:
      return typeof v === "string" ? v : JSON.stringify(v);
  }
}

/** Bounds and type check of a value (null: valid). Messages follow pydantic's wording. */
export function checkValue(f: FieldDescriptor, value: unknown): string | null {
  if (value === null || value === undefined) return f.nullable ? null : "Input should not be None";
  const num = (v: unknown, integer: boolean): string | null => {
    if (typeof v !== "number" || !Number.isFinite(v)) return "Input should be a valid number";
    if (integer && !Number.isInteger(v)) return "Input should be a valid integer";
    if (f.min !== undefined && v < f.min) return `Input should be greater than or equal to ${f.min}`;
    if (f.max !== undefined && v > f.max) return `Input should be less than or equal to ${f.max}`;
    if (f.exclusive_min !== undefined && v <= f.exclusive_min) return `Input should be greater than ${f.exclusive_min}`;
    if (f.exclusive_max !== undefined && v >= f.exclusive_max) return `Input should be less than ${f.exclusive_max}`;
    return null;
  };
  switch (f.type) {
    case "int":
      return num(value, true);
    case "float":
      return num(value, false);
    case "bool":
      return typeof value === "boolean" ? null : "Input should be a valid boolean";
    case "number_list":
      if (!Array.isArray(value)) return "Input should be a valid list";
      if (f.length !== undefined && value.length !== f.length) {
        return `List should have ${f.length} item${f.length === 1 ? "" : "s"}, not ${value.length}`;
      }
      for (const v of value) {
        const err = num(v, f.item_type === "int");
        if (err) return err;
      }
      return null;
    case "voi_ref":
      return typeof value === "string" ? null : "Input should be a valid string";
    case "choice":
    case "image_ref":
      if (f.editable_choice || !f.choices) return null;
      return f.choices.some((c) => c.value === value) ? null : `Input should be ${f.choices.map((c) => `'${String(c.value)}'`).join(" or ")}`;
    default:
      return null;
  }
}

export type Parsed = { value: unknown } | { error: string };

/** Parse editor text into a value of the field (Qt editors' commit rules). */
export function parseText(f: FieldDescriptor, text: string): Parsed {
  const raw = text.trim();
  let value: unknown;
  switch (f.type) {
    case "int":
    case "float": {
      if (!raw) {
        if (f.nullable) return { value: null };
        return { error: "Input should be a valid number" };
      }
      const v = Number(raw.replace(",", "."));
      if (!Number.isFinite(v)) return { error: `Input should be a valid ${f.type === "int" ? "integer" : "number"}, unable to parse "${raw}"` };
      value = v;
      break;
    }
    case "number_list": {
      if (!raw) {
        value = f.nullable ? null : [];
        break;
      }
      const parts = raw.replace(/^[[(]|[\])]$/g, "").split(/[\s,;]+/).filter(Boolean);
      const values = parts.map(Number);
      const bad = parts.find((_, i) => !Number.isFinite(values[i]));
      if (bad !== undefined) return { error: `Not a list of numbers: "${bad}"` };
      value = values;
      break;
    }
    case "str":
      value = f.nullable && !text ? null : text;
      break;
    case "choice": {
      const c = f.choices?.find((x) => x.label === text || String(x.value) === text);
      value = c ? c.value : f.editable_choice ? text : text;
      break;
    }
    case "json":
    default:
      if (!raw) {
        value = null;
        break;
      }
      try {
        value = JSON.parse(raw);
      } catch {
        value = raw;
      }
  }
  const error = checkValue(f, value);
  return error ? { error } : { value };
}

/** Editor state of every leaf field of a schema. */
export function initFields(schema: FormSchema): FormFields {
  const out: FormFields = {};
  for (const f of leafFields(schema.fields)) {
    const value = initialValue(schema, f);
    out[f.name] = { text: formatValue(f, value), value, error: null, edited: false };
  }
  return out;
}

/** Apply a typed text to a field: the value follows while it is valid, else the error. */
export function editText(fields: FormFields, f: FieldDescriptor, text: string): FormFields {
  const parsed = parseText(f, text);
  const prev = fields[f.name];
  const next: FieldState =
    "error" in parsed ? { ...prev, text, error: parsed.error } : { text, value: parsed.value, error: null, edited: true };
  return { ...fields, [f.name]: next };
}

/** Set a field's value directly (checkbox, combo); validated. */
export function editValue(fields: FormFields, f: FieldDescriptor, value: unknown): FormFields {
  const error = checkValue(f, value);
  const prev = fields[f.name];
  const next: FieldState = error ? { ...prev, error } : { text: formatValue(f, value), value, error: null, edited: true };
  return { ...fields, [f.name]: next };
}

/**
 * The values reported back: names in `explicit` or edited; a group as the nested dict of its
 * overrides (initial overrides plus edits), reported when explicit or non-empty. Explicit names
 * without a descriptor (hidden entries) are passed through unchanged.
 */
export function formValues(schema: FormSchema, fields: FormFields): Record<string, unknown> {
  const explicit = new Set(schema.explicit);
  const out: Record<string, unknown> = {};
  const known = new Set(schema.fields.map((f) => f.name));
  for (const name of schema.explicit) {
    if (!known.has(name) && schema.values[name] !== undefined) out[name] = clone(schema.values[name]);
  }
  for (const f of schema.fields) {
    if (f.type === "group") {
      const given = schema.values[f.name];
      const overrides: Record<string, unknown> = given && typeof given === "object" && !Array.isArray(given) ? clone(given as Record<string, unknown>) : {};
      for (const leaf of leafFields(f.fields ?? [])) {
        const state = fields[leaf.name];
        if (state?.edited) setPath(overrides, leaf.name.split(".").slice(1), clone(state.value));
      }
      if (explicit.has(f.name) || Object.keys(overrides).length) out[f.name] = overrides;
      continue;
    }
    const state = fields[f.name];
    if (state && (explicit.has(f.name) || state.edited)) out[f.name] = clone(state.value);
  }
  return out;
}

/** First field with an invalid entry, as Qt's `validation_failed(name, message)`. */
export function firstError(schema: FormSchema, fields: FormFields): { name: string; label: string; message: string } | null {
  for (const f of leafFields(schema.fields)) {
    const error = fields[f.name]?.error;
    if (error) return { name: f.name, label: f.label, message: error };
  }
  return null;
}

/** Label shown in front of an editor (Qt appends a colon). */
export function fieldLabel(f: FieldDescriptor): string {
  return f.label.endsWith(":") ? f.label : `${f.label}:`;
}

/** Choices of a combo, with an entry for a current value the list does not hold. */
export function comboChoices(f: FieldDescriptor, value: unknown): Choice[] {
  const list = f.choices ?? [];
  if (value === undefined || list.some((c) => c.value === value)) return list;
  return [...list, { value: value as Choice["value"], label: value === null ? "None" : String(value) }];
}

/**
 * Suggested values of a field (`plan.form_choices`) not already among its choices, by label and
 * without duplicates. Suggestions never restrict what can be typed.
 */
export function suggestionsOf(f: FieldDescriptor, suggestions?: Record<string, Choice[]>): Choice[] {
  const seen = new Set((f.choices ?? []).map((c) => c.label));
  const out: Choice[] = [];
  for (const c of suggestions?.[f.name] ?? []) {
    if (seen.has(c.label)) continue;
    seen.add(c.label);
    out.push(c);
  }
  return out;
}

/** Spin step of a numeric field. */
export function stepOf(f: FieldDescriptor): number {
  if (f.step !== undefined) return f.step;
  if (f.type === "int" || f.item_type === "int") return 1;
  if (f.decimals !== undefined) return 10 ** -Math.min(f.decimals, 2);
  return 0.1;
}

/** One spin step from a value, clamped to the bounds. */
export function stepValue(f: FieldDescriptor, value: number, direction: 1 | -1): number {
  const step = stepOf(f);
  let v = value + direction * step;
  if (f.type === "int") v = Math.round(v);
  else v = Math.round(v / step) * step;
  v = Number(v.toPrecision(12));
  if (f.min !== undefined) v = Math.max(f.min, v);
  if (f.max !== undefined) v = Math.min(f.max, v);
  if (f.exclusive_min !== undefined && v <= f.exclusive_min) v = value;
  if (f.exclusive_max !== undefined && v >= f.exclusive_max) v = value;
  return v;
}

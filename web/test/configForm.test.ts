import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { FieldDescriptor, FormSchema } from "../src/api/protocol";
import { ConfigForm } from "../src/components/forms/ConfigForm";
import {
  checkValue,
  editText,
  editValue,
  editorKind,
  firstError,
  formValues,
  initFields,
  parseText,
  stepValue,
  suggestionsOf,
} from "../src/components/forms/formModel";

const fd = (name: string, type: FieldDescriptor["type"], extra: Partial<FieldDescriptor> = {}): FieldDescriptor => ({ name, label: name, type, ...extra });

const schema: FormSchema = {
  id: "engine:SVDPB",
  title: "Configure SVD Pencil Beam",
  fields: [
    fd("enabled", "bool", { default: true }),
    fd("cutoff", "float", { default: 400, exclusive_min: 0, unit: "mm" }),
    fd("seed", "int", { default: 0, min: 0 }),
    fd("penumbra", "float", { default: null, nullable: true }),
    fd("method", "choice", { default: "L-BFGS-B", editable_choice: true, choices: [{ value: "L-BFGS-B", label: "L-BFGS-B" }, { value: "SLSQP", label: "SLSQP" }] }),
    fd("calc_let", "choice", { default: "auto", choices: [{ value: "auto", label: "auto" }, { value: true, label: "true" }, { value: false, label: "false" }] }),
    fd("sd", "number_list", { default: [1, 1, 1], length: 3, item_type: "float", min: 0 }),
    fd("options", "json", { default: {} }),
    fd("name", "str", { default: "", nullable: true }),
    fd("sampling", "group", {
      fields: [
        fd("sampling.threshold", "float", { default: 0.01, min: 0, max: 1 }),
        fd("sampling.inner", "group", { fields: [fd("sampling.inner.kind", "choice", { default: "radius", choices: [{ value: "radius", label: "radius" }, { value: "dose", label: "dose" }] })] }),
      ],
    }),
  ],
  values: { seed: 3, sampling: { threshold: 0.02 }, hidden_entry: "kept" },
  explicit: ["seed", "sampling", "hidden_entry"],
};

const field = (name: string): FieldDescriptor => {
  const find = (list: FieldDescriptor[]): FieldDescriptor | undefined =>
    list.map((f) => (f.name === name ? f : f.fields ? find(f.fields) : undefined)).find(Boolean);
  return find(schema.fields)!;
};

describe("config form model", () => {
  it("maps descriptors to Qt editors", () => {
    expect(editorKind(field("enabled"))).toBe("checkbox");
    expect(editorKind(field("cutoff"))).toBe("spin");
    expect(editorKind(field("penumbra"))).toBe("optional-number");
    expect(editorKind(field("method"))).toBe("editable-combo");
    expect(editorKind(field("calc_let"))).toBe("combo");
    expect(editorKind(field("sd"))).toBe("list");
    expect(editorKind(field("options"))).toBe("json");
    expect(editorKind(field("name"))).toBe("text");
    expect(editorKind(field("sampling"))).toBe("section");
  });

  it("shows schema values over defaults, nested overrides by dot path", () => {
    const f = initFields(schema);
    expect(f.seed.value).toBe(3);
    expect(f.cutoff.text).toBe("400");
    expect(f["sampling.threshold"].value).toBe(0.02);
    expect(f["sampling.inner.kind"].value).toBe("radius");
    expect(f.penumbra.text).toBe("");
    expect(f.sd.text).toBe("1 1 1");
  });

  it("reports only explicit or edited values, groups as nested override dicts", () => {
    let f = initFields(schema);
    expect(formValues(schema, f)).toEqual({ seed: 3, sampling: { threshold: 0.02 }, hidden_entry: "kept" });
    f = editText(f, field("cutoff"), "350");
    f = editValue(f, field("sampling.inner.kind"), "dose");
    f = editValue(f, field("calc_let"), false);
    expect(formValues(schema, f)).toEqual({
      seed: 3,
      cutoff: 350,
      calc_let: false,
      sampling: { threshold: 0.02, inner: { kind: "dose" } },
      hidden_entry: "kept",
    });
  });

  it("validates bounds, types and list lengths live without committing invalid values", () => {
    let f = initFields(schema);
    f = editText(f, field("cutoff"), "-1");
    expect(f.cutoff.error).toBe("Input should be greater than 0");
    expect(f.cutoff.value).toBe(400);
    expect(firstError(schema, f)).toMatchObject({ name: "cutoff" });
    expect(formValues(schema, f).cutoff).toBeUndefined();
    f = editText(f, field("seed"), "1.5");
    expect(f.seed.error).toBe("Input should be a valid integer");
    f = editText(f, field("sd"), "1 2");
    expect(f.sd.error).toMatch(/3 items/);
    f = editText(f, field("sd"), "1 x 2");
    expect(f.sd.error).toMatch(/Not a list of numbers/);
    f = editText(f, field("sampling.threshold"), "2");
    expect(f["sampling.threshold"].error).toMatch(/less than or equal to 1/);
  });

  it("parses nullable, JSON and free-text entries like the Qt editors", () => {
    expect(parseText(field("penumbra"), "")).toEqual({ value: null });
    expect(parseText(field("cutoff"), "")).toEqual({ error: "Input should be a valid number" });
    expect(parseText(field("options"), '{"maxls": 20}')).toEqual({ value: { maxls: 20 } });
    expect(parseText(field("options"), "plain")).toEqual({ value: "plain" });
    expect(parseText(field("method"), "trust-krylov")).toEqual({ value: "trust-krylov" });
    expect(parseText(field("name"), "")).toEqual({ value: null });
    expect(checkValue(field("calc_let"), "maybe")).toMatch(/Input should be/);
    expect(checkValue(field("calc_let"), true)).toBeNull();
  });

  it("steps spin values within the bounds", () => {
    expect(stepValue(field("seed"), 0, -1)).toBe(0);
    expect(stepValue(field("seed"), 3, 1)).toBe(4);
    expect(stepValue(fd("x", "float", { max: 1, step: 0.1 }), 0.95, 1)).toBe(1);
  });
});

describe("suggested values (plan.form_choices)", () => {
  const model = fd("model", "str", { default: null, nullable: true });
  const aiSchema: FormSchema = { id: "engine:AIBeamlet", title: "Configure AI Beamlet Engine", fields: [model], values: { model: "org/mine" }, explicit: ["model"] };
  const suggestions = {
    model: [
      { value: "org/a", label: "org/a" },
      { value: "org/b", label: "org/b" },
      { value: "org/a", label: "org/a" },
    ],
  };
  const render = (s?: Record<string, { value: string; label: string }[]>) =>
    renderToStaticMarkup(createElement(ConfigForm, { schema: aiSchema, suggestions: s, testId: "f" }));

  it("are deduplicated and leave out the field's own choices", () => {
    expect(suggestionsOf(model, suggestions).map((c) => c.label)).toEqual(["org/a", "org/b"]);
    expect(suggestionsOf(field("method"), { method: [{ value: "SLSQP", label: "SLSQP" }, { value: "COBYLA", label: "COBYLA" }] }).map((c) => c.label)).toEqual(["COBYLA"]);
    expect(suggestionsOf(model, undefined)).toEqual([]);
  });

  it("render a text input with a datalist that keeps free text", () => {
    const html = render(suggestions);
    const list = /<input[^>]*list="([^"]+)"[^>]*>/.exec(html);
    expect(list).not.toBeNull();
    expect(html).toContain('value="org/mine"');
    expect(html).toContain(`<datalist id="${list![1]}" data-testid="f-model-suggestions"><option value="org/a"></option><option value="org/b"></option></datalist>`);
    // free text is still parsed as the string field it is
    expect(parseText(model, "other/model")).toEqual({ value: "other/model" });
  });

  it("render the plain text input while there are none", () => {
    const html = render({});
    expect(html).not.toContain("datalist");
    expect(html).not.toMatch(/<input[^>]*list=/);
    expect(render(undefined)).toBe(html);
  });
});

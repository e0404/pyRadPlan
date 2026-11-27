import { describe, expect, it } from "vitest";
import type { PlanSummary } from "../src/api/protocol";
import type { PlanDetail, PlanOptions } from "../src/api/protocol/plan";
import { objectiveRows, referenceChoices, robustnessOptions, sharedQuantity, typeOptions, unlistedRobustness, voiReferenceChoices } from "../src/components/objectives/objectivesModel";
import {
  applyBioSelection,
  captureState,
  currentBio,
  draftFromPlan,
  equidistantAngles,
  formBeamAngles,
  modifiedKeys,
  planSetParams,
  syncedCouch,
} from "../src/components/plan/planModel";
import { isUnappliedPick, pickComboOptions, tissueForModel } from "../src/components/plan/TissueDialog";
import { withModel, withoutIdentity } from "../src/components/plan/tissueDraft";
import { sameTissue } from "../src/components/vois/EditVoiDialog";
import type { TissueDraft } from "../src/api/protocol";
import { formatSummary, logAxis, pyExp } from "../src/components/optimization/StatusWindow";

const plan: PlanSummary = {
  radiation_mode: "protons",
  machine: "Generic",
  num_of_fractions: 30,
  gantry_angles: [0, 90],
  couch_angles: [0, 0],
  bixel_width: 5,
  iso_center: [[1, 2, 3], [4, 5, 6]],
  dose_grid_resolution: [5, 5, 5],
  engine: "HongPB",
  bio_model: "constRBE",
};

const detail: PlanDetail = {
  ...plan,
  dose_convention: "total",
  prescribed_dose: 60,
  machine_source: "name",
  resolved_iso_center: null,
  engine_config: { name: "HongPB", values: { lateral_model: "single" } },
  bio: { model: "constRBE", option: "constRBE", values: { rbe: 1.1 } },
  scenario: { model: "nomScen", values: {} },
  problem: { name: "nonlin_fluence", values: { max_iter: 50 } },
  solver: { name: "scipy", values: {} },
};

const options = (bio: PlanOptions["bio_models"], defaultBio: string): PlanOptions => ({
  radiation_modes: [],
  machines: [{ value: "Generic", label: "Generic" }],
  engines: [{ value: "HongPB", label: "Hong Particle Pencil-Beam" }],
  bio_models: bio,
  default_bio_model: defaultBio,
  scenario_models: [],
  problems: [{ value: "nonlin_fluence", label: "Fluence (nonlinear)" }],
  solvers: [],
  default_solver: "scipy",
  beam_presets: [],
  fields: [{ name: "num_of_fractions", label: "Fractions", type: "int", min: 1, max: 1000 }],
});

const bio = (value: string, model: string, enabled = true, reason?: string) => ({ value, label: value, model, parameters: {}, enabled, reason });

describe("plan form model", () => {
  it("builds the draft from the plan detail, keeping per-beam isocenters", () => {
    const d = draftFromPlan(plan, detail);
    expect(d.gantry).toBe("0 90");
    expect(d.isoAuto).toBe(true);
    expect(d.iso).toBe("multiple iso centers");
    expect(d.perBeamIso).toHaveLength(2);
    expect(d.engineProps.HongPB).toEqual({ lateral_model: "single" });
    expect(d.problemProps.nonlin_fluence.solver).toEqual({ name: "scipy", values: {} });
    expect(d.bixel).toBe("5.00");
  });

  it("sends only the changed fields on Apply", () => {
    const d = draftFromPlan(plan, detail);
    const clean = captureState(d);
    expect(planSetParams(d, clean, null)).toEqual({});
    const edited = { ...d, fractions: "20", engineProps: { HongPB: { lateral_model: "double" } } };
    expect([...modifiedKeys(edited, clean)].sort()).toEqual(["engine_props", "fractions"]);
    expect(planSetParams(edited, clean, null)).toEqual({ num_of_fractions: 20, engine: "HongPB", engine_config: { values: { lateral_model: "double" } } });
  });

  it("validates entries before sending (Qt _build_plan errors)", () => {
    const d = draftFromPlan(plan, detail);
    const clean = captureState(d);
    const opts = options([], "none");
    expect(() => planSetParams({ ...d, fractions: "2000" }, clean, opts)).toThrow(/between 1 and 1000/);
    expect(() => planSetParams({ ...d, couch: "0 0 0" }, clean, opts)).toThrow(/couch angles must be 1 or match/);
    expect(() => planSetParams({ ...d, gantry: "0 90 180", couch: "0" }, clean, opts)).toThrow(/per-beam iso centers do not match 3 beams/);
    expect(() => planSetParams({ ...d, isoAuto: false, perBeamIso: null, iso: "1 2" }, clean, opts)).toThrow(/three coordinates/);
  });

  it("sends machine, engine and model with a new radiation mode", () => {
    const d = draftFromPlan(plan, detail);
    const clean = captureState(d);
    const p = planSetParams({ ...d, radiation_mode: "carbon", engine: "HongPB", bioOption: "LEM:HIT", bioModel: "LEM", bioParams: { LEM: { kernel_set: "HIT" } } }, clean, null);
    expect(p).toMatchObject({ radiation_mode: "carbon", machine: "Generic", engine: "HongPB", bio_model: { model: "LEM", option: "LEM:HIT", values: { kernel_set: "HIT" } } });
  });

  it("sends problem and solver from the optimization dialog", () => {
    const d = draftFromPlan(plan, detail);
    const clean = captureState(d);
    const p = planSetParams({ ...d, problemProps: { nonlin_fluence: { values: { apply_overlap: false }, solver: { name: "ipax", values: { max_iter: 10 } } } } }, clean, null);
    expect(p).toEqual({ problem: { name: "nonlin_fluence", values: { apply_overlap: false } }, solver: { name: "ipax", values: { max_iter: 10 } } });
  });

  it("keeps the couch angles aligned with the gantry angles", () => {
    expect(syncedCouch("0 90 180", "10")).toBe("10 10 10");
    expect(syncedCouch("0 90 180", "10 20")).toBe("10 20 0");
    expect(syncedCouch("0 90", "10 20 30")).toBe("10 20");
    expect(syncedCouch("0 90", "1 2")).toBeNull();
    expect(equidistantAngles(5)).toEqual([0, 72, 144, 216, 288]);
    expect(formBeamAngles({ gantry: "0 90", couch: "" })).toEqual({ gantry: [0, 90], couch: [0, 0] });
  });

  it("applies the backend's bio-model selection and describes the current one", () => {
    const d = { ...draftFromPlan(plan, detail), bioParams: { constRBE: { rbe: 1.1 }, LEM: { kernel_set: "old", alpha: 2 } } };
    const protons = options([bio("none", "none"), bio("constRBE", "constRBE")], "constRBE");
    expect(currentBio(d, "protons", protons)).toEqual({ radiation_mode: "protons", model: "constRBE", option: "constRBE", values: { rbe: 1.1 }, label: "constRBE" });
    expect(currentBio({ ...d, bioOption: null }, "protons", protons, "MCN · x (not supported by this machine)").label).toBe("MCN · x");

    const carbon = options([bio("none", "none"), { ...bio("LEM:HIT", "LEM"), parameters: { kernel_set: "HIT" } }], "LEM:HIT");
    const replaced = applyBioSelection(d, { option: "LEM:HIT", model: "LEM", replaced: true, unsupported: null, note: null }, carbon);
    expect(replaced).toMatchObject({ bioOption: "LEM:HIT", bioModel: "LEM" });
    expect(replaced.bioParams.LEM).toEqual({ alpha: 2, kernel_set: "HIT" });
    const kept = applyBioSelection(d, { option: "constRBE", model: "constRBE", replaced: false, unsupported: null, note: null }, protons);
    expect(kept.bioParams.constRBE).toEqual({ rbe: 1.1 });
    const unsupported = applyBioSelection(d, { option: null, model: "MCN", replaced: false, unsupported: { model: "MCN", label: "MCN (not supported by this machine)", description: "" }, note: null }, protons);
    expect(unsupported).toMatchObject({ bioOption: null, bioModel: "MCN" });
    expect(applyBioSelection(d, undefined, protons).bioOption).toBe("constRBE");
  });

  it("sends back only the problem settings the plan detail lists (unshown entries stay in the plan)", () => {
    const d = draftFromPlan(plan, detail);
    const clean = captureState(d);
    const edited = { ...d, problemProps: { nonlin_fluence: { values: { ...d.problemProps.nonlin_fluence.values, p_norm: 3 }, solver: d.problemProps.nonlin_fluence.solver } } };
    const p = planSetParams(edited, clean, null);
    expect(p.problem).toEqual({ name: "nonlin_fluence", values: { max_iter: 50, p_norm: 3 } });
  });
});

describe("objectives model", () => {
  const structures = [
    {
      name: "Core",
      index: 0,
      type: "OAR",
      color: [0, 0, 0] as [number, number, number],
      visible: true,
      overlap_priority: 2,
      num_voxels: 10,
      num_scenarios: 1,
      center_of_mass: null,
      alpha_x: 0.1,
      beta_x: 0.05,
      tissue: { model: "Generic" as const, values: {} },
      tissue_summary: "",
      objectives: [
        { index: 0, name: "Squared Overdosing", priority: 300, quantity: "physical_dose", robustness: "none", parameters: { d_max: 30 } },
        { index: 1, name: "Max DVH", priority: 50, quantity: "physical_dose", robustness: "none", parameters: { d: 20, v_max: 40 } },
      ],
    },
    {
      name: "Target",
      index: 1,
      type: "TARGET",
      color: [0, 0, 0] as [number, number, number],
      visible: true,
      overlap_priority: 1,
      num_voxels: 10,
      num_scenarios: 1,
      center_of_mass: null,
      alpha_x: 0.1,
      beta_x: 0.05,
      tissue: { model: "Generic" as const, values: {} },
      tissue_summary: "",
      objectives: [{ index: 0, name: "Squared Deviation", priority: 800, quantity: "rbe_x_dose", robustness: "none", parameters: { d_ref: 60 } }],
    },
  ];

  it("groups rows by VOI with the label on the first row", () => {
    const rows = objectiveRows(structures);
    expect(rows.map((r) => [r.structure.name, r.objective.index, r.first])).toEqual([
      ["Core", 0, true],
      ["Core", 1, false],
      ["Target", 0, true],
    ]);
    expect(sharedQuantity(structures)).toBeNull();
    expect(sharedQuantity([structures[0]])).toBe("physical_dose");
  });

  it("offers the type's robustness, a single option disabled", () => {
    const type = { name: "x", scenario_aware: false, parameters: [], robustness: [{ value: "none", label: "None" }, { value: "expected_value", label: "Expected value" }] };
    expect(robustnessOptions(type, "composite_worst_case").options.map((o) => o.key)).toEqual(["none", "expected_value", "composite_worst_case"]);
    const single = { ...type, robustness: [{ value: "scenario_stack", label: "Over all scenarios" }] };
    expect(robustnessOptions(single, "none")).toMatchObject({ current: "scenario_stack" });
  });

  it("locks the type of an objective the catalog does not list and keeps its robustness editable", () => {
    const types = ["Squared Deviation", "Max DVH"].map((name) => ({ name, scenario_aware: false, parameters: [], robustness: [] }));
    expect(typeOptions(types, "Max DVH")).toEqual({ options: types.map((t) => ({ key: t.name, label: t.name })), locked: false });
    expect(typeOptions(types, "Outcome LR")).toEqual({ options: [{ key: "Outcome LR", label: "Outcome LR" }], locked: true });
    const all = ["none", "expected_value", "scenario_stack"].map((value) => ({ value, label: value }));
    expect(unlistedRobustness(all, "none").map((c) => c.value)).toEqual(["none", "expected_value"]);
    expect(unlistedRobustness(all, "scenario_stack").map((c) => c.value)).toEqual(["scenario_stack"]);
    expect(robustnessOptions(undefined, "expected_value", unlistedRobustness(all, "expected_value"))).toMatchObject({ current: "expected_value" });
    expect(robustnessOptions(undefined, "none").options).toEqual([]);
  });

  it("matches image references by result key and keeps custom values", () => {
    const catalog = { objectives: [], quantities: [], robustness: [], default_by_type: {}, reference_images: [{ value: "physical_dose", label: "dose" }] };
    expect(referenceChoices(catalog, { $result: "physical_dose" }).current).toBe("physical_dose");
    const custom = referenceChoices(catalog, { $custom: "Custom image (1×1×1, value 0)" });
    expect(custom.current).toBe("$custom");
    expect(custom.options[0].label).toBe("Custom image (1×1×1, value 0)");
  });

  it("lists the structures for a VOI reference and marks a name that is not one of them", () => {
    const names = ["BODY", "Core", "Core", "OuterTarget"];
    const known = voiReferenceChoices(names, "Core");
    expect(known).toMatchObject({ current: "Core", missing: false });
    expect(known.options.map((o) => o.key)).toEqual(["BODY", "Core", "OuterTarget"]);
    const unknown = voiReferenceChoices(names, "Ventricles_4mm");
    expect(unknown).toMatchObject({ current: "Ventricles_4mm", missing: true });
    expect(unknown.options[0]).toEqual({ key: "Ventricles_4mm", label: "Ventricles_4mm (not in structures)" });
    expect(unknown.options.slice(1)).toEqual(known.options);
  });
});

describe("status window", () => {
  it("formats the summary like Python", () => {
    expect(pyExp(12.3456, 3)).toBe("1.235e+01");
    expect(pyExp(-0.00034, 2, true)).toBe("-3.40e-04");
    expect(pyExp(0.5, 2, true)).toBe("+5.00e-01");
    expect(formatSummary(null, [])).toBe("Waiting for optimization…");
    expect(
      formatSummary({ iteration: 12, objective: 12.3 }, [
        { iteration: 11, objective: 15 },
        { iteration: 12, objective: 12.3 },
      ]),
    ).toBe("iter 12 · f=1.230e+01 · Δf=-1.80e-01");
  });

  it("puts log ticks on decades", () => {
    const axis = logAxis(3, 1500);
    expect(axis.min).toBe(0);
    expect(axis.max).toBe(4);
    expect(axis.ticks.filter((t) => t.major).map((t) => t.label)).toEqual(["1", "10", "100", "1000", "10000"]);
  });
});

describe("tissue picker", () => {
  it("treats only a selection differing from the corresponding pick as unapplied", () => {
    expect(isUnappliedPick(null, null)).toBe(false);
    expect(isUnappliedPick(null, "class:0")).toBe(false);
    expect(isUnappliedPick("class:0", "class:0")).toBe(false);
    expect(isUnappliedPick("preset:3", "class:0")).toBe(true);
    expect(isUnappliedPick("preset:3", null)).toBe(true);
  });

  it("groups the picks with headers and a separator after the preferred ones", () => {
    const pick = (id: string, group: string, preferred: boolean, exact = false) => ({
      id,
      kind: "preset" as const,
      label: id,
      group,
      exact,
      preferred,
      model: "LEM" as const,
      values: {},
      description: "",
    });
    const options = pickComboOptions({
      models: [],
      reference_radiations: [],
      option_label: "",
      picks: [pick("class:0", "Base data", true, true), pick("preset:0", "LEM I", true), pick("preset:1", "MKM", false)],
    });
    expect(options.map((o) => (o.separator ? "---" : o.header ? `# ${o.label}` : o.label))).toEqual(["# Base data", "class:0", "# LEM I", "preset:0", "---", "# MKM", "preset:1"]);
    expect(options.find((o) => o.key === "class:0")?.bold).toBe(true);
  });
});

describe("tissue draft identity", () => {
  const schema = {
    models: [
      { label: "Generic" as const, fields: [{ name: "alpha_x" }, { name: "beta_x" }] },
      { label: "MKM" as const, fields: [{ name: "r_domain" }, { name: "alpha_x" }] },
      { label: "mMKM" as const, fields: [{ name: "r_domain" }, { name: "alpha_x" }] },
    ],
    reference_radiations: [],
    picks: [],
    option_label: "",
  } as unknown as Parameters<typeof tissueForModel>[0] & object;
  const hsg: TissueDraft = { model: "MKM", values: { r_domain: 0.2, alpha_x: 0.3 }, cell_line: "HSG", cell_line_source: "Inaniwa 2010" };

  it("is sent with the draft, also when null, and absent for drafts without it", () => {
    expect(tissueForModel(schema, hsg)).toMatchObject({ cell_line: "HSG", cell_line_source: "Inaniwa 2010" });
    expect(tissueForModel(schema, { ...hsg, cell_line: null, cell_line_source: null })).toMatchObject({ cell_line: null, cell_line_source: null });
    const plain = tissueForModel(schema, { model: "MKM", values: { r_domain: 0.2 } });
    expect("cell_line" in plain).toBe(false);
  });

  it("is dropped when the parameters or the parameter class change, kept for MKM <-> mMKM", () => {
    expect(withoutIdentity(hsg)).toMatchObject({ cell_line: null, cell_line_source: null });
    expect(withoutIdentity({ model: "MKM", values: {} })).toEqual({ model: "MKM", values: {} });
    expect(withModel(schema, hsg, "mMKM")).toMatchObject({ model: "mMKM", cell_line: "HSG" });
    expect(withModel(schema, hsg, "Generic")).toMatchObject({ model: "Generic", cell_line: null });
  });

  it("makes a preset of equal values but another cell line a change", () => {
    expect(sameTissue(hsg, { ...hsg })).toBe(true);
    expect(sameTissue(hsg, { ...hsg, cell_line: "CHO" })).toBe(false);
    expect(sameTissue(hsg, { model: "MKM", values: hsg.values })).toBe(false);
    expect(sameTissue({ model: "MKM", values: {} }, { model: "MKM", values: {}, cell_line: null, cell_line_source: null })).toBe(true);
  });
});

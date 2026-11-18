import { describe, expect, it } from "vitest";
import type { PlanSummary } from "../src/api/protocol";
import type { PlanDetail, PlanOptions } from "../src/api/protocol/plan";
import { objectiveRows, referenceChoices, robustnessOptions, sharedQuantity } from "../src/components/objectives/objectivesModel";
import {
  captureState,
  draftFromPlan,
  equidistantAngles,
  formBeamAngles,
  modifiedKeys,
  pickBioModel,
  planSetParams,
  syncedCouch,
} from "../src/components/plan/planModel";
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

  it("re-lists the biological model like Qt (default follows the mode, unavailable explicit choice replaced with a note)", () => {
    const d = draftFromPlan(plan, detail);
    const protons = options([bio("none", "none"), bio("constRBE", "constRBE"), bio("MCN", "MCN")], "constRBE");
    const carbon = options([bio("none", "none"), bio("LEM:HIT", "LEM")], "LEM:HIT");
    const followed = pickBioModel(d, { mode: "protons", options: protons }, "carbon", carbon, "Generic", "HongPB");
    expect(followed).toMatchObject({ option: "LEM:HIT", model: "LEM", note: null });

    const explicit = { ...d, bioOption: "MCN", bioModel: "MCN" };
    const carbonMcnOff = options([bio("none", "none"), bio("LEM:HIT", "LEM"), bio("MCN", "MCN", false, "Machine 'Generic' offers no MCN base data")], "LEM:HIT");
    const replaced = pickBioModel(explicit, { mode: "protons", options: protons }, "carbon", carbonMcnOff, "Generic", "HongPB");
    expect(replaced.option).toBe("LEM:HIT");
    expect(replaced.note?.text).toBe("Biological model 'MCN' is not available on machine 'Generic'; selected 'LEM:HIT'.");
    expect(replaced.note?.reason).toMatch(/offers no MCN/);

    const kept = pickBioModel(explicit, { mode: "protons", options: protons }, "protons", protons, "Generic", "HongPB");
    expect(kept).toMatchObject({ option: "MCN", note: null });
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

  it("matches image references by result key and keeps custom values", () => {
    const catalog = { objectives: [], quantities: [], robustness: [], default_by_type: {}, reference_images: [{ value: "physical_dose", label: "dose" }] };
    expect(referenceChoices(catalog, { $result: "physical_dose" }).current).toBe("physical_dose");
    const custom = referenceChoices(catalog, { $custom: "Custom image (1×1×1, value 0)" });
    expect(custom.current).toBe("$custom");
    expect(custom.options[0].label).toBe("Custom image (1×1×1, value 0)");
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

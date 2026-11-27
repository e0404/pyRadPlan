import { describe, expect, it } from "vitest";
import { BackendError } from "../src/api/backend";
import { MockBackend } from "../src/api/mock/mockBackend";
import type { BackendEvent } from "../src/api/protocol";
import type { PlanDetail } from "../src/api/protocol/plan";

async function loaded() {
  const backend = new MockBackend({ speed: 0 });
  await backend.ready();
  const { session } = await backend.request("session.create", {});
  const events: BackendEvent[] = [];
  backend.subscribe((e) => events.push(e));
  await backend.request("data.load_phantom", { name: "TG119" }, session);
  const req = <M extends Parameters<typeof backend.request>[0]>(method: M, params: Parameters<typeof backend.request<M>>[1]) =>
    backend.request(method, params, session);
  return { backend, session, events, req };
}

async function failure(p: Promise<unknown>): Promise<BackendError> {
  const err = await p.catch((e: unknown) => e);
  expect(err).toBeInstanceOf(BackendError);
  return err as BackendError;
}

describe("mock backend, package A methods", () => {
  it("serves plan detail, options and capabilities for the web profile", async () => {
    const { backend, req } = await loaded();
    const info = await backend.request("app.info", {});
    expect(info.runtime).toBe("pyodide");
    expect(info.capabilities?.features.pause?.available).toBe(false);
    expect(info.capabilities?.features.ai?.reason).toMatch(/pydantic/);
    expect(info.methods).toContain("objectives.update");

    const detail = (await req("plan.get", { detail: true })) as PlanDetail;
    expect(detail.dose_grid_resolution).toEqual([5, 5, 5]);
    expect(detail.dose_convention).toBe("total");
    expect(detail.resolved_iso_center).toHaveLength(3);
    const v0 = await req("plan.get", {});
    expect(v0).not.toHaveProperty("dose_convention");

    const photons = await req("plan.options", {});
    expect(photons.engines.map((e) => e.value)).toEqual(["SVDPB"]);
    expect(photons.solvers.find((s) => s.value === "ipopt")).toMatchObject({ enabled: false });
    expect(photons.fields.find((f) => f.name === "num_of_fractions")).toMatchObject({ min: 1, max: 1000, advanced: true });
    const carbon = await req("plan.options", { radiation_mode: "carbon", machine: "Generic", engine: "HongPB" });
    expect(carbon.engines.find((e) => e.value === "TOPAS")).toMatchObject({ enabled: false });
    expect(carbon.bio_models.find((b) => b.model === "LSM")).toMatchObject({ enabled: false });
    expect(JSON.parse(carbon.default_bio_model).model).toBe("alpha_beta_kernels");
  });

  it("chooses the bio-model entry itself (bio_selection), the frontend only applies it", async () => {
    const { req } = await loaded();
    const restored = await req("plan.options", {});
    const none = String(restored.bio_models.find((b) => b.model === "none")!.value);
    expect(restored.bio_selection).toMatchObject({ option: none, replaced: false, unsupported: null, note: null });
    const kernels = String((await req("plan.options", { radiation_mode: "carbon" })).bio_models.find((b) => b.model === "alpha_beta_kernels")!.value);
    const followed = await req("plan.options", { radiation_mode: "carbon", current_bio: { radiation_mode: "photons", model: "none", option: none } });
    expect(followed.bio_selection).toMatchObject({ option: kernels, model: "alpha_beta_kernels", replaced: true, note: null });
    const kept = await req("plan.options", { radiation_mode: "carbon", current_bio: { radiation_mode: "carbon", model: "alpha_beta_kernels", option: kernels } });
    expect(kept.bio_selection).toMatchObject({ option: kernels, replaced: false });
    const lsm = String((await req("plan.options", { radiation_mode: "carbon" })).bio_models.find((b) => b.model === "LSM")!.value);
    const replaced = await req("plan.options", { radiation_mode: "carbon", current_bio: { radiation_mode: "carbon", model: "LSM", option: lsm, label: "LSM" } });
    expect(replaced.bio_selection?.option).toBe(kernels);
    expect(replaced.bio_selection?.note?.text).toMatch(/^Biological model 'LSM' is not available with the dose engine; selected '.+'\.$/);
  });

  it("keeps scenario model and problem settings on a mode change and problem entries the form does not show", async () => {
    const { backend, session, req } = await loaded();
    await req("plan.set", { scenarios: { model: "wcScen", values: {} }, problem: { name: "nonlin_fluence", values: { convert_dose_objectives: false, max_iter: 40 } } });
    const detail = (await req("plan.get", { detail: true })) as PlanDetail;
    expect(detail.problem.values).toEqual({ max_iter: 40 });
    await req("plan.set", { problem: { name: "nonlin_fluence", values: { ...detail.problem.values, max_iter: 60 } } });
    await req("plan.set", { radiation_mode: "protons" });
    const after = (await req("plan.get", { detail: true })) as PlanDetail;
    expect(after.scenario.model).toBe("wcScen");
    expect(after.problem.values).toEqual({ max_iter: 60 });
    const internal = (backend as unknown as { sessions: Map<string, { ext: { plan: { problem: { values: Record<string, unknown> } } } }> }).sessions;
    expect(internal.get(session)?.ext.plan.problem.values).toEqual({ convert_dose_objectives: false, max_iter: 60 });
  });

  it("builds forms with nested groups and forwards max_iter into the solver form", async () => {
    const { req } = await loaded();
    const engine = await req("plan.form", { kind: "engine", name: "SVDPB" });
    expect(engine.title).toBe("Configure SVD Pencil Beam");
    expect(engine.fields.find((f) => f.name === "kernel_cutoff")).toMatchObject({ type: "float", exclusive_min: 0 });
    expect(engine.explicit).toEqual([]);
    expect((await req("plan.form", { kind: "scenario", name: "wcScen" })).fields.find((f) => f.name === "setup")?.type).toBe("group");
    expect(await failure(req("plan.form", { kind: "engine", name: "NoSuchEngine" }))).toMatchObject({ type: "ValueError" });
    await req("plan.set", { problem: { name: "nonlin_fluence", values: { max_iter: 42 } } });
    const solver = await req("plan.form", { kind: "solver", name: "scipy" });
    expect(solver.values.max_iter).toBe(42);
    const problem = await req("plan.form", { kind: "problem", name: "nonlin_fluence" });
    expect(problem.values).not.toHaveProperty("max_iter");
    const ipopt = await req("plan.form", { kind: "solver", name: "ipopt" });
    expect(ipopt.notes?.[0]).toMatch(/unavailable/);
  });

  it("applies v1 plan fields and rejects invalid values with field details", async () => {
    const { req } = await loaded();
    const state = await req("plan.set", {
      engine: "SVDPB",
      engine_config: { values: { kernel_cutoff: 300 } },
      dose_convention: "per_fraction",
      scenarios: { model: "wcScen", values: { setup: { sd: [3, 3, 3] } } },
      solver: { name: "scipy", values: { max_iter: 20 } },
    });
    expect(state.stages.plan).toBe("done");
    const detail = (await req("plan.get", { detail: true })) as PlanDetail;
    expect(detail.engine_config.values).toEqual({ kernel_cutoff: 300 });
    expect(detail.scenario.model).toBe("wcScen");
    expect(detail.solver).toEqual({ name: "scipy", values: { max_iter: 20 } });
    const err = await failure(req("plan.set", { engine_config: { values: { kernel_cutoff: -1 } }, num_of_fractions: 0 }));
    expect(err.type).toBe("ValidationError");
    expect(err.code).toBe("invalid");
    expect(err.details).toEqual(
      expect.arrayContaining([
        { field: "num_of_fractions", message: "Input should be greater than or equal to 1" },
        { field: "engine.kernel_cutoff", message: "Input should be greater than 0" },
      ]),
    );
    expect((await failure(req("plan.set", { engine: "TOPAS" }))).code).toBe("invalid");
    await req("plan.set", { radiation_mode: "carbon" });
    expect((await failure(req("plan.set", { engine: "TOPAS" }))).code).toBe("not_available");
  });

  it("serves the recorded tissue schema and evaluates rows without simulating the base data checks", async () => {
    const { req } = await loaded();
    await req("plan.set", { radiation_mode: "carbon" });
    expect(await req("plan.tissue_check", {})).toEqual({ marker: "", issues: [] });

    const schema = await req("tissue.schema", {});
    expect(schema.models.map((m) => m.label)).toEqual(["Generic", "LEM", "MKM", "mMKM"]);
    expect(schema.picks[0]).toMatchObject({ kind: "base_data" });

    const structures = (await req("cst.get", {})).structures;
    const rows = structures.map((s) => ({ structure: s.name, tissue: s.tissue }));
    const live = await req("tissue.evaluate", { rows });
    expect(live.rows.map((r) => r.structure)).toEqual(structures.map((s) => s.name));
    expect(live.rows.every((r) => r.issues.length === 0 && r.comparison === null)).toBe(true);
    expect(live.notes).toEqual([]);
    expect((await req("tissue.evaluate", { rows, op: "derive" })).notes[0]).toMatch(/not simulated/);

    const pick = schema.picks[0];
    const applied = await req("tissue.evaluate", { rows, op: "apply_pick", pick: pick.id, targets: [rows[0].structure] });
    expect(applied.rows[0]).toMatchObject({ pick: pick.id, tissue: { model: pick.model, values: pick.values } });
    expect(applied.rows[1].tissue.values).toEqual(rows[1].tissue.values);
    expect(applied.notes).toEqual([]);
    expect(applied.rows[0].tissue).toMatchObject({ cell_line: pick.cell_line ?? null, cell_line_source: pick.cell_line_source ?? null });
    const preset = schema.picks.find((p) => p.kind === "preset" && p.cell_line);
    if (preset) {
      const named = await req("tissue.evaluate", { rows, op: "apply_pick", pick: preset.id, targets: [rows[0].structure] });
      expect(named.rows[0].tissue).toMatchObject({ cell_line: preset.cell_line, cell_line_source: preset.cell_line_source });
      await req("tissue.apply", { rows: [{ structure: 0, tissue: named.rows[0].tissue }] });
      const stored = (await req("cst.get", { names: [0] })).structures[0].tissue;
      expect(stored.cell_line).toBe(preset.cell_line);
    }
    const perturbed = { ...applied.rows[0].tissue, values: { ...applied.rows[0].tissue.values, alpha_x: 0.987 } };
    expect((await req("tissue.evaluate", { rows: [{ structure: rows[0].structure, tissue: perturbed }] })).rows[0].pick).toBeNull();

    const edited = { model: "Generic" as const, values: { alpha_x: 0.3, beta_x: 0.05 } };
    await req("tissue.apply", { rows: [{ structure: 0, tissue: edited }] });
    const after = (await req("cst.get", { names: [0] })).structures[0];
    expect(after.tissue.values).toEqual(edited.values);
    const missing = await failure(req("tissue.apply", { rows: [{ structure: 7, tissue: edited }] }));
    expect(missing.type).toBe("KeyError");
  });

  it("edits objectives (add / update / change type / remove / set all) and validates parameters", async () => {
    const { req } = await loaded();
    const catalog = await req("objectives.catalog", {});
    expect(catalog.objectives.find((o) => o.name === "Mean Dose")?.parameters.find((p) => p.name === "f_diff")?.type).toBe("choice");
    expect(catalog.objectives.find((o) => o.name === "Squared Mimicking")?.parameters[0].type).toBe("image_ref");
    expect(catalog.reference_images).toEqual([]);

    let res = await req("objectives.add", { structure: 1 });
    const added = res.structure.objectives[res.structure.objectives.length - 1];
    expect(added).toMatchObject({ name: "Squared Deviation", quantity: "physical_dose" });
    expect(res.state.structures.find((s) => s.name === "OuterTarget")?.num_objectives).toBe(2);

    res = await req("objectives.update", { structure: 1, index: added.index, changes: { priority: 250, parameters: { d_ref: 55 } } });
    expect(res.structure.objectives[added.index]).toMatchObject({ priority: 250, parameters: { d_ref: 55 } });
    const bad = await failure(req("objectives.update", { structure: 1, index: added.index, changes: { parameters: { d_ref: -3 } } }));
    expect(bad.details).toEqual([{ field: "parameters.d_ref", message: "Input should be greater than or equal to 0" }]);

    await req("objectives.update", { structure: 1, index: added.index, changes: { robustness: "composite_worst_case" } });
    res = await req("objectives.change_type", { structure: 1, index: added.index, name: "Min DVH" });
    expect(res.structure.objectives[added.index]).toMatchObject({ name: "Min DVH", priority: 250, robustness: "composite_worst_case", parameters: { d: 30, v_min: 95 } });
    res = await req("objectives.change_type", { structure: 1, index: added.index, name: "Variance" });
    expect(res.structure.objectives[added.index].robustness).toBe("scenario_stack");

    await req("objectives.set_quantity_all", { quantity: "rbe_x_dose" });
    const all = (await req("cst.get", {})).structures.flatMap((s) => s.objectives.map((o) => o.quantity));
    expect(new Set(all)).toEqual(new Set(["rbe_x_dose"]));

    res = await req("objectives.remove", { structure: 1, index: added.index });
    expect(res.structure.objectives).toHaveLength(1);
  });

  it("describes the risk structure of a POLO objective as a VOI reference and stores a name", async () => {
    const { req } = await loaded();
    const catalog = await req("objectives.catalog", {});
    const risk = catalog.objectives.find((o) => o.name === "NTCP POLO")?.parameters.find((p) => p.name === "risk_voi_name");
    expect(risk).toMatchObject({ type: "voi_ref", default: "Ventricles_4mm" });

    let res = await req("objectives.add", { structure: 1 });
    const added = res.structure.objectives[res.structure.objectives.length - 1];
    res = await req("objectives.change_type", { structure: 1, index: added.index, name: "NTCP POLO" });
    expect(res.structure.objectives[added.index].parameters.risk_voi_name).toBe("Ventricles_4mm");
    res = await req("objectives.update", { structure: 1, index: added.index, changes: { parameters: { risk_voi_name: "Core" } } });
    expect(res.structure.objectives[added.index].parameters.risk_voi_name).toBe("Core");
    const bad = await failure(req("objectives.update", { structure: 1, index: added.index, changes: { parameters: { risk_voi_name: 3 } } }));
    expect(bad.details).toEqual([{ field: "parameters.risk_voi_name", message: "Input should be a valid string" }]);
  });

  it("records the optimization for the status window and session history", async () => {
    const { req, events } = await loaded();
    await req("workflow.calc_dose_influence", {});
    await req("plan.set", { solver: { name: "scipy", values: { max_iter: 25 } } });
    await req("workflow.calc_dose_influence", {});
    const result = await req("workflow.optimize", {});
    expect(result.optimization).toMatchObject({ solver: "scipy", num_iter: 25 });
    const metrics = events.filter((e) => e.event === "progress" && e.data?.objective !== undefined);
    expect(metrics.length).toBeGreaterThan(3);
    const history = await req("session.history", {});
    expect(history.optimization).toMatchObject({ finished: true, outcome: "done" });
    expect(history.optimization?.points.length).toBe(metrics.length);
    expect((await req("objectives.catalog", {})).reference_images[0]).toMatchObject({ value: "physical_dose" });
    const pause = await failure(req("session.pause", {}));
    expect(pause.code).toBe("not_available");
  });
});

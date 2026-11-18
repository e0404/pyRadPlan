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
    expect(photons.engines.find((e) => e.value === "TOPAS")).toMatchObject({ enabled: false });
    expect(photons.solvers.find((s) => s.value === "ipopt")).toMatchObject({ enabled: false });
    expect(photons.fields.find((f) => f.name === "num_of_fractions")).toMatchObject({ min: 1, max: 1000, advanced: true });
    const carbon = await req("plan.options", { radiation_mode: "carbon", machine: "Generic", engine: "HongPB" });
    expect(carbon.bio_models.find((b) => b.value === "mMKM:NIRS")).toMatchObject({ enabled: false });
    expect(carbon.default_bio_model).toBe("LEM:HIT");
  });

  it("chooses the bio-model entry itself (bio_selection), the frontend only applies it", async () => {
    const { req } = await loaded();
    const restored = await req("plan.options", {});
    expect(restored.bio_selection).toMatchObject({ option: "none", replaced: false, unsupported: null, note: null });
    const followed = await req("plan.options", { radiation_mode: "carbon", current_bio: { radiation_mode: "photons", model: "none", option: "none" } });
    expect(followed.bio_selection).toMatchObject({ option: "LEM:HIT", model: "LEM", replaced: true, note: null });
    const kept = await req("plan.options", { radiation_mode: "carbon", machine: "HIT_APM", current_bio: { radiation_mode: "carbon", model: "mMKM", option: "mMKM:NIRS", label: "mMKM" } });
    expect(kept.bio_selection).toMatchObject({ option: "mMKM:NIRS", replaced: false });
    const replaced = await req("plan.options", { radiation_mode: "carbon", machine: "Generic", current_bio: { radiation_mode: "carbon", model: "mMKM", option: "mMKM:NIRS", label: "mMKM" } });
    expect(replaced.bio_selection?.option).toBe("LEM:HIT");
    expect(replaced.bio_selection?.note?.text).toBe("Biological model 'mMKM' is not available on machine 'Generic'; selected 'LEM · LEM IV (HIT)'.");
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
    expect(engine.fields.find((f) => f.name === "dij_sampling")?.fields?.[0].name).toBe("dij_sampling.rel_dose_threshold");
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
      solver: { name: "ipax", values: { max_iter: 20 } },
    });
    expect(state.stages.plan).toBe("done");
    const detail = (await req("plan.get", { detail: true })) as PlanDetail;
    expect(detail.engine_config.values).toEqual({ kernel_cutoff: 300 });
    expect(detail.scenario.model).toBe("wcScen");
    expect(detail.solver).toEqual({ name: "ipax", values: { max_iter: 20 } });
    const err = await failure(req("plan.set", { engine_config: { values: { kernel_cutoff: -1 } }, num_of_fractions: 0 }));
    expect(err.type).toBe("ValidationError");
    expect(err.code).toBe("invalid");
    expect(err.details).toEqual(
      expect.arrayContaining([
        { field: "num_of_fractions", message: "Input should be greater than or equal to 1" },
        { field: "engine.kernel_cutoff", message: "Input should be greater than 0" },
      ]),
    );
    const err2 = await failure(req("plan.set", { engine: "TOPAS" }));
    expect(err2.code).toBe("not_available");
  });

  it("checks tissues, evaluates, derives and applies them behind the error gate", async () => {
    const { req } = await loaded();
    await req("plan.set", { radiation_mode: "carbon" });
    const check = await req("plan.tissue_check", {});
    expect(check.marker).toBe("error");
    const dij = await failure(req("workflow.calc_dose_influence", {}));
    expect(dij.code).toBe("bio_model_unsupported");

    const schema = await req("tissue.schema", {});
    expect(schema.models.map((m) => m.label)).toEqual(["Generic", "LEM", "MKM", "mMKM"]);
    expect(schema.picks[0]).toMatchObject({ kind: "base_data", exact: true });

    const mkm = { model: "MKM" as const, values: { r_domain: 0.3, r_nucleus: 4, alpha_0: 0.15, beta_0: 0.05 } };
    const live = await req("tissue.evaluate", { rows: [{ structure: "Core", tissue: mkm }], edited: { structure: "Core", field: "alpha_0" } });
    expect(Number(live.rows[0].tissue.values.alpha_x)).toBeCloseTo(0.15 + 0.05 * (0.0058 / 0.09), 5);
    expect(live.rows[0].tissue.sources?.alpha_x).toBe("derived");
    expect(live.can_derive).toBe(true);
    expect(live.rows[0].issues[0].severity).toBe("error");

    const gate = await failure(req("tissue.apply", { rows: [{ structure: "Core", tissue: mkm }] }));
    expect(gate.code).toBe("tissue_errors");
    expect(Array.isArray(gate.details)).toBe(true);

    const structures = (await req("cst.get", {})).structures;
    const rows = structures.map((s) => ({ structure: s.name, tissue: s.tissue }));
    const picked = await req("tissue.evaluate", { rows, op: "apply_pick", pick: schema.picks[0].id });
    expect(picked.rows.every((r) => r.tissue.model === "LEM" && r.issues.length === 0)).toBe(true);
    expect(picked.notes).toHaveLength(1);
    await req("tissue.apply", { rows: picked.rows.map((r) => ({ structure: r.structure, tissue: r.tissue })) });
    expect((await req("plan.tissue_check", {})).marker).toBe("");
    const after = await req("workflow.calc_dose_influence", {});
    expect(after.stages.dij).toBe("done");
    expect(after.plan?.bio_model).toBe("LEM");
  });

  it("edits objectives (add / update / change type / remove / set all) and validates parameters", async () => {
    const { req } = await loaded();
    const catalog = await req("objectives.catalog", {});
    expect(catalog.objectives.find((o) => o.name === "Mean Dose")?.parameters.find((p) => p.name === "f_diff")?.type).toBe("choice");
    expect(catalog.objectives.find((o) => o.name === "Squared Mimicking")?.parameters[0].type).toBe("image_ref");
    expect(catalog.reference_images).toEqual([]);

    let res = await req("objectives.add", { structure: "OuterTarget" });
    const added = res.structure.objectives[res.structure.objectives.length - 1];
    expect(added).toMatchObject({ name: "Squared Deviation", quantity: "physical_dose" });
    expect(res.state.structures.find((s) => s.name === "OuterTarget")?.num_objectives).toBe(2);

    res = await req("objectives.update", { structure: "OuterTarget", index: added.index, changes: { priority: 250, parameters: { d_ref: 55 } } });
    expect(res.structure.objectives[added.index]).toMatchObject({ priority: 250, parameters: { d_ref: 55 } });
    const bad = await failure(req("objectives.update", { structure: "OuterTarget", index: added.index, changes: { parameters: { d_ref: -3 } } }));
    expect(bad.details).toEqual([{ field: "parameters.d_ref", message: "Input should be greater than or equal to 0" }]);

    await req("objectives.update", { structure: "OuterTarget", index: added.index, changes: { robustness: "composite_worst_case" } });
    res = await req("objectives.change_type", { structure: "OuterTarget", index: added.index, name: "Min DVH" });
    expect(res.structure.objectives[added.index]).toMatchObject({ name: "Min DVH", priority: 250, robustness: "composite_worst_case", parameters: { d: 30, v_min: 95 } });
    res = await req("objectives.change_type", { structure: "OuterTarget", index: added.index, name: "Variance" });
    expect(res.structure.objectives[added.index].robustness).toBe("scenario_stack");

    await req("objectives.set_quantity_all", { quantity: "rbe_x_dose" });
    const all = (await req("cst.get", {})).structures.flatMap((s) => s.objectives.map((o) => o.quantity));
    expect(new Set(all)).toEqual(new Set(["rbe_x_dose"]));

    res = await req("objectives.remove", { structure: "OuterTarget", index: added.index });
    expect(res.structure.objectives).toHaveLength(1);
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

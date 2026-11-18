import { describe, expect, it } from "vitest";
import { BackendError } from "../src/api/backend";
import { MockBackend } from "../src/api/mock/mockBackend";
import type { BackendEvent } from "../src/api/protocol";

async function newSession() {
  const backend = new MockBackend({ speed: 0 });
  await backend.ready();
  const { session } = await backend.request("session.create", {});
  const events: BackendEvent[] = [];
  backend.subscribe((e) => events.push(e));
  return { backend, session, events };
}

describe("MockBackend", () => {
  it("reports app info and an empty session", async () => {
    const { backend, session } = await newSession();
    const info = await backend.request("app.info", {});
    expect(info.phantoms).toContain("TG119");
    expect(info.machines.photons).toContain("Generic");
    const state = await backend.request("session.state", {}, session);
    expect(state.has.ct).toBe(false);
    expect(state.actions.generate_stf).toBe(false);
    expect(state.stages.data).toBe("missing");
  });

  it("runs the workflow with stage tracking, progress and a dose", async () => {
    const { backend, session, events } = await newSession();
    let state = await backend.request("data.load_phantom", { name: "TG119" }, session);
    expect(state.has).toMatchObject({ ct: true, cst: true, pln: true, dij: false });
    expect(state.ct?.shape).toEqual([129, 167, 167]);
    expect(state.structures.map((s) => s.name)).toEqual(["Core", "OuterTarget", "BODY"]);
    expect(state.actions.calc_dose_influence).toBe(true);
    expect(state.actions.optimize).toBe(false);

    state = await backend.request("workflow.calc_dose_influence", {}, session);
    expect(state.stages.stf).toBe("done");
    expect(state.stages.dij).toBe("done");
    expect(state.actions.optimize).toBe(true);

    state = await backend.request("workflow.optimize", { max_iter: 10 }, session);
    expect(state.has.result).toBe(true);
    expect(state.quantities).toEqual(["physical_dose"]);

    const progress = events.filter((e) => e.event === "progress");
    expect(progress.some((e) => e.event === "progress" && e.fraction === null)).toBe(true);
    expect(progress.some((e) => e.event === "progress" && typeof e.fraction === "number")).toBe(true);
    expect(events.some((e) => e.event === "state")).toBe(true);
    expect(events.some((e) => e.event === "log")).toBe(true);

    const dose = await backend.request("view.volume", { name: "physical_dose" }, session);
    expect(dose.data.dtype).toBe("float32");
    expect(dose.data.shape).toEqual([129, 167, 167]);
    expect(dose.range[1]).toBeGreaterThan(50);
    // Dose at the target (r = 25 mm, y > 0) is near the prescription, zero in air.
    const g = dose.grid;
    const at = (x: number, y: number, z: number) => {
      const i = Math.round((x - g.origin[0]) / g.spacing[0]);
      const j = Math.round((y - g.origin[1]) / g.spacing[1]);
      const k = Math.round((z - g.origin[2]) / g.spacing[2]);
      return dose.data.data[(k * g.shape[1] + j) * g.shape[2] + i];
    };
    expect(at(0, 25, 0)).toBeGreaterThan(50);
    expect(at(0, 0, 0)).toBeLessThan(at(0, 25, 0));
    expect(at(-240, -240, 0)).toBe(0);

    const ct = await backend.request("view.volume", { name: "ct", encoding: "auto" }, session);
    expect(ct.data.data).toBeInstanceOf(Int16Array);
    const ct32 = await backend.request("view.volume", { name: "ct" }, session);
    expect(ct32.data.data).toBeInstanceOf(Float32Array);
    expect(ct.range[0]).toBe(-1000);

    const masks = await backend.request("view.structure_masks", { names: ["Core"] }, session);
    expect(masks.masks).toHaveLength(1);
    expect((masks.masks[0] as { data: { data: Uint8Array } }).data.data).toBeInstanceOf(Uint8Array);
  });

  it("marks downstream stages stale after a plan change", async () => {
    const { backend, session } = await newSession();
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("workflow.run_all", {}, session);
    let state = await backend.request("plan.set", { gantry_angles: [0, 90, 180], couch_angles: [0, 0, 0] }, session);
    expect(state.stages).toMatchObject({ plan: "done", stf: "stale", dij: "stale", optimization: "stale" });
    expect(state.status).toMatch(/outdated/);
    state = await backend.request("workflow.generate_stf", {}, session);
    expect(state.stages).toMatchObject({ stf: "done", dij: "stale", optimization: "stale" });
    state = await backend.request("workflow.calc_dose_influence", {}, session);
    expect(state.stages).toMatchObject({ dij: "done", optimization: "stale" });
  });

  it("returns protocol errors as BackendError", async () => {
    const { backend, session } = await newSession();
    const pre = await backend.request("workflow.optimize", {}, session).catch((e: unknown) => e);
    expect(pre).toBeInstanceOf(BackendError);
    expect((pre as BackendError).type).toBe("PreconditionError");
    expect((pre as BackendError).code).toBe("no_data");
    await expect(backend.request("plan.set", { foo: 1 } as never, session)).rejects.toThrow(/LookupError|No CT/);
    await expect(backend.request("data.load_phantom", { name: "NOPE" }, session)).rejects.toThrow(/Unknown phantom/);
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    const err = await backend
      .request("plan.set", { gantry_angles: [0, 90], couch_angles: [0, 0, 0] }, session)
      .catch((e: unknown) => e);
    expect(err).toBeInstanceOf(BackendError);
    expect((err as BackendError).type).toBe("ValueError");
    await expect(backend.request("session.state", {}, "s999")).rejects.toThrow(/Unknown session/);
  });

  it("can cancel a running computation", async () => {
    const backend = new MockBackend({ speed: 0.2 });
    const { session } = await backend.request("session.create", {});
    await backend.request("data.load_phantom", { name: "BOXPHANTOM" }, session);
    const running = backend.request("workflow.calc_dose_influence", {}, session);
    await new Promise((r) => setTimeout(r, 20));
    expect(backend.cancel()).toBe(true);
    const err = await running.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(BackendError);
    expect((err as BackendError).type).toBe("KeyboardInterrupt");
    expect(backend.cancel()).toBe(false);
  });

  it("accepts uploads as uint8 arrays", async () => {
    const { backend, session } = await newSession();
    const data = new Uint8Array([1, 2, 3]);
    const state = await backend.request("data.load_file", { filename: "patient.mat", data: { dtype: "uint8", shape: [3], data } }, session);
    expect(state.has.ct).toBe(true);
    await expect(
      backend.request("data.load_file", { filename: "notes.txt", data: { dtype: "uint8", shape: [1], data: new Uint8Array(1) } }, session),
    ).rejects.toThrow(/Unsupported/);
  });

  it("computes DVHs", async () => {
    const { backend, session } = await newSession();
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("workflow.run_all", {}, session);
    const dvh = await backend.request("analysis.dvh", {}, session);
    const target = dvh.curves.find((c) => c.name === "OuterTarget")!;
    expect(target.volume.data[0]).toBeCloseTo(100);
    // Monotonically non-increasing.
    const v = Array.from(target.volume.data);
    expect(v.every((x, i) => i === 0 || x <= v[i - 1] + 1e-6)).toBe(true);
  });

  it("follows the protocol amendments (default plan, recalc, run_all, physical_only)", async () => {
    const { backend, session } = await newSession();
    let state = await backend.request("data.load_phantom", { name: "TG119" }, session);
    expect(state.plan).toMatchObject({ gantry_angles: [0], couch_angles: [0], dose_grid_resolution: [5, 5, 5] });
    state = await backend.request("plan.set", { gantry_angles: [0, 120, 240] }, session);
    expect(state.plan?.couch_angles).toEqual([0, 0, 0]);
    await expect(backend.request("plan.set", { foo: 1 } as never, session)).rejects.toThrow(/Unknown plan field/);

    const opt = await backend.request("workflow.calc_dose_influence", {}, session).then(() =>
      backend.request("workflow.optimize", { max_iter: 5 }, session),
    );
    expect(opt.optimization).toMatchObject({ solver: "scipy", num_iter: 5 });
    expect(opt.actions.recalc).toBe(true);
    state = await backend.request("workflow.recalc", {}, session);
    expect(state.stages.optimization).toBe("done");

    // run_all only runs the steps that are missing or stale: nothing to do here.
    const events: string[] = [];
    const off = backend.subscribe((e) => e.event === "progress" && events.push(e.label));
    await backend.request("workflow.run_all", {}, session);
    expect(events).toEqual([]);
    off();

    state = await backend.request("plan.set", { radiation_mode: "carbon" }, session);
    expect(state.plan).toMatchObject({ machine: "Generic", engine: "HongPB", bio_model: "LEM" });
    const err = await backend.request("workflow.calc_dose_influence", {}, session).catch((e: unknown) => e);
    expect((err as BackendError).code).toBe("bio_model_unsupported");
    state = await backend.request("workflow.calc_dose_influence", { physical_only: true }, session);
    expect(state.plan?.bio_model).toBe("none");
    expect(state.stages.dij).toBe("done");
    expect(state.busy).toBeNull();
  });
});

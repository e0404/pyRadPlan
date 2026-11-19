import { beforeEach, describe, expect, it } from "vitest";
import { MockBackend } from "../src/api/mock/mockBackend";
import { useApp, connect } from "../src/store/appStore";
import { selectQuantity, syncViews } from "../src/store/revisions";

const BEAM = "physical_dose_beam 0";

async function ready() {
  const backend = new MockBackend({ speed: 0 });
  const requests: string[] = [];
  const original = backend.request.bind(backend);
  (backend as { request: unknown }).request = (method: string, params: Record<string, unknown>, session?: string) => {
    requests.push(method === "view.volume" ? `${method}:${String(params.name)}` : method);
    return (original as (m: string, p: unknown, s?: string) => Promise<unknown>)(method, params, session);
  };
  await connect(backend);
  const session = useApp.getState().session!;
  await backend.request("data.load_phantom", { name: "TG119" }, session);
  await backend.request("plan.set", { gantry_angles: [0, 120, 240], couch_angles: [0, 0, 0] }, session);
  await backend.request("workflow.run_all", {}, session);
  useApp.setState({ state: await backend.request("session.state", {}, session) });
  await syncViews();
  return { backend, session, requests };
}

describe("revision-driven sync", () => {
  beforeEach(() => {
    useApp.setState({ ct: null, maskSet: null, quantities: {}, state: null });
  });

  it("refetches a cached component that a new result outdated when it is selected", async () => {
    const { backend, session, requests } = await ready();
    await selectQuantity(BEAM);
    const before = useApp.getState().quantities[BEAM];
    await backend.request("workflow.recalc", {}, session);
    useApp.setState({ state: await backend.request("session.state", {}, session) });
    await syncViews();
    requests.length = 0;
    await selectQuantity(BEAM);
    expect(requests).toContain(`view.volume:${BEAM}`);
    const after = useApp.getState();
    expect(after.quantities[BEAM]).not.toBe(before);
    expect(after.appliedRevisions.volumes[BEAM]).toBe(after.state!.revisions!.volumes[BEAM]);
  });

  it("updates the mask colors on a color-only structure change without refetching masks", async () => {
    const { backend, session, requests } = await ready();
    const target = useApp.getState().state!.structures[0];
    const masksRev = useApp.getState().state!.revisions!.masks;
    await backend.request("cst.update_structure", { name: target.index, changes: { color: [1, 2, 3] } }, session);
    useApp.setState({ state: await backend.request("session.state", {}, session) });
    requests.length = 0;
    await syncViews();
    expect(useApp.getState().state!.revisions!.masks).toBe(masksRev);
    expect(requests).not.toContain("view.structure_masks");
    const mask = useApp.getState().maskSet!.masks.find((m) => m.index === target.index)!;
    expect([...mask.color]).toEqual([1, 2, 3]);
  });
});

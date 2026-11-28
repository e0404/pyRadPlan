import { describe, expect, it } from "vitest";
import { BackendError } from "../src/api/backend";
import { MockBackend } from "../src/api/mock/mockBackend";
import { computeDvh, percentile, qiRow, sourceDirection } from "../src/api/mock/viewer";
import type { BitsMasks, CropMasks } from "../src/api/protocol/viewer";
import {
  BEAM_MIN_ORBIT_TILT,
  beamRadius,
  directionInSlice,
  indexToSliceMm,
  labelAnchor,
  orbitAngle,
  orbitPoint,
  pointToIndex,
  projectBeam,
  type BeamInput,
} from "../src/renderer/beams";
import { ContourCache, maskCenterOfMass } from "../src/renderer/contours";
import { sliceLayout, type VolumeGeometry } from "../src/renderer/geometry";
import { marchingSquares, segmentsLength } from "../src/renderer/marchingSquares";
import { directionLetter, projectMannequin, screenAxesLps } from "../src/renderer/orientation";
import { groupResultKeys, keyLabel, resultKey, sanitizeTag, splitResultKey } from "../src/results/resultGroups";
import { applyPoint, doseAtVolume, parseRefVols, qiMarkers, qiTableRows, toCsv } from "../src/components/analysis/dvhModel";
import { defaultCheckedVois } from "../src/store/viewerSlice";
import { decodeMasks, resultGroups } from "../src/store/revisions";

const geom: VolumeGeometry = { shape: [20, 30, 40], spacing: [2, 2, 3], origin: [-40, -30, -30] };

function beam(gantry: number, couch = 0, iso: [number, number, number] = [0, 0, 0]): BeamInput {
  const u = sourceDirection(0, couch);
  const w = sourceDirection(90, couch);
  return {
    index: 0,
    name: "#0",
    label: `Beam #0: gantry ${gantry}°, couch ${couch}°`,
    gantry,
    couch,
    iso_center: iso,
    source_direction: sourceDirection(gantry, couch),
    orbit_u: u,
    orbit_w: w,
    orbit_axis: [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]],
    pending: false,
  };
}

describe("beam geometry", () => {
  it("matches lps.get_beam_rotation_matrix @ (0, -1, 0)", () => {
    const close = (a: number[], b: number[]) => a.forEach((v, i) => expect(v).toBeCloseTo(b[i], 4));
    close(sourceDirection(0, 0), [0, -1, 0]);
    close(sourceDirection(90, 0), [1, 0, 0]);
    close(sourceDirection(45, 30), [0.6124, -0.7071, -0.3536]);
  });

  it("projects in-plane beams with the source on the largest circle inside the slice", () => {
    const layout = sliceLayout(geom, "axial");
    const p = projectBeam(geom, layout, beam(0));
    expect(p.perpendicular).toBe(false);
    expect(p.toward).toBeCloseTo(0);
    // Iso at the voxel centre of index (20, 15): slice mm (41, 31); radius limited by y: 14 voxels * 2 mm.
    expect(p.iso.h).toBeCloseTo(41);
    expect(p.iso.v).toBeCloseTo(31);
    expect(p.radius).toBeCloseTo(beamRadius(layout, 41, 31)!);
    expect(p.radius).toBeCloseTo(28);
    expect(p.source!.v).toBeCloseTo(3); // gantry 0: source anterior (-y), up on screen
    expect(p.draggable).toBe(true);
    expect(p.text).toBe("#0");
  });

  it("marks out-of-plane beams and the perpendicular view", () => {
    const sag = sliceLayout(geom, "sagittal");
    const p = projectBeam(geom, sag, beam(90));
    expect(p.perpendicular).toBe(true); // a lateral beam looks along the sagittal normal
    const tilted = projectBeam(geom, sag, beam(45));
    expect(tilted.perpendicular).toBe(false);
    expect(Math.abs(tilted.toward)).toBeCloseTo(Math.SQRT1_2, 4);
    expect(tilted.text).toMatch(/[⊙⊗]$/);
    // The orbit of a coplanar beam is edge-on in the sagittal view: not draggable there.
    expect(tilted.draggable).toBe(false);
    expect(tilted.tooltip).toMatch(/Drag it in the Axial view/);
  });

  it("inverts orbit points to gantry angles", () => {
    const layout = sliceLayout(geom, "axial");
    const p = projectBeam(geom, layout, beam(30));
    for (const g of [0, 45, 162, 270, 359]) {
      const at = orbitPoint(p, g)!;
      expect(orbitAngle(p.orbit!, { h: at.h - p.iso.h, v: at.v - p.iso.v })).toBe(g);
    }
    expect(BEAM_MIN_ORBIT_TILT).toBe(0.3);
    expect(labelAnchor(0.5)).toBe(0);
    expect(labelAnchor(-0.5)).toBe(1);
    expect(labelAnchor(0)).toBe(0.5);
  });

  it("maps LPS points through the direction matrix", () => {
    const flipped: VolumeGeometry = { ...geom, direction: [-1, 0, 0, 0, 1, 0, 0, 0, 1] };
    expect(pointToIndex(flipped, [-44, -30, -30])).toEqual([2, 0, 0]);
    const layout = sliceLayout(flipped, "axial");
    const d = directionInSlice(flipped, layout, [1, 0, 0]);
    expect(d.dh).toBeCloseTo(-1);
    expect(indexToSliceMm(layout, [0, 0, 5]).n).toBe(5);
  });
});

describe("orientation marker", () => {
  it("labels the screen edges like the Qt mannequin", () => {
    const axial = projectMannequin(...Object.values(screenAxesLps(geom, sliceLayout(geom, "axial"))) as [number[], number[]]);
    expect(axial.letters).toEqual({ right: "L", left: "R", bottom: "P", top: "A" });
    const sag = projectMannequin(...Object.values(screenAxesLps(geom, sliceLayout(geom, "sagittal"))) as [number[], number[]]);
    expect(sag.letters).toEqual({ right: "S", left: "I", bottom: "P", top: "A" });
    expect(Math.max(sag.width, sag.height)).toBeGreaterThan(100);
    expect(directionLetter([0, 0, -1])).toBe("I");
  });
});

describe("compact masks", () => {
  const g: VolumeGeometry = { shape: [3, 8, 8], spacing: [1, 1, 1], origin: [0, 0, 0] };
  const full = new Uint8Array(3 * 64);
  for (let j = 2; j < 5; j++) for (let i = 3; i < 6; i++) full[64 + j * 8 + i] = 1;

  it("traces crops and bit planes like the full mask", () => {
    const crop = new Uint8Array(9).fill(1);
    const plane = new Uint8Array(3 * 64);
    for (let n = 0; n < full.length; n++) if (full[n]) plane[n] = 1 << 5;
    const cache = new ContourCache();
    cache.setMasks(
      [
        { index: 0, name: "full", color: [1, 2, 3], data: full },
        { index: 1, name: "crop", color: [1, 2, 3], storage: { kind: "crop", data: crop, bbox: [1, 2, 3, 2, 5, 6] } },
        { index: 2, name: "bits", color: [1, 2, 3], storage: { kind: "bits", plane, bit: 5, bbox: null } },
        { index: 3, name: "bits2", color: [1, 2, 3], storage: { kind: "bits", plane, bit: 5 } },
      ],
      g,
    );
    const ref = segmentsLength(cache.contour(0, "axial", 1));
    expect(ref).toBeCloseTo(segmentsLength(marchingSquares(full.subarray(64, 128), 8, 8, 0.5)));
    expect(segmentsLength(cache.contour(1, "axial", 1))).toBeCloseTo(ref);
    expect(segmentsLength(cache.contour(1, "sagittal", 4))).toBeGreaterThan(0);
    expect(cache.contour(2, "axial", 1)).toHaveLength(0); // bbox null = empty
    expect(segmentsLength(cache.contour(3, "axial", 1))).toBeCloseTo(ref);
    expect(maskCenterOfMass({ index: 0, name: "c", color: [0, 0, 0], storage: { kind: "crop", data: crop, bbox: [1, 2, 3, 2, 5, 6] } }, g)).toEqual([4, 3, 1]);
  });

  it("decodes every view.structure_masks encoding", () => {
    const grid = { shape: [1, 2, 2] as [number, number, number], spacing: [1, 1, 1] as [number, number, number], origin: [0, 0, 0] as [number, number, number] };
    const bits: BitsMasks = {
      grid,
      encoding: "bits",
      planes: [{ dtype: "uint8", shape: [1, 2, 2], data: new Uint8Array([1, 3, 0, 2]) }],
      masks: [
        { index: 0, name: "A", color: [1, 1, 1], plane: 0, bit: 0, num_voxels: 2, bbox: [0, 0, 0, 1, 1, 2] },
        { index: 1, name: "B", color: [2, 2, 2], plane: 0, bit: 1, num_voxels: 2, bbox: [0, 0, 0, 1, 2, 2] },
      ],
    };
    expect(decodeMasks(bits).masks[1].storage).toMatchObject({ kind: "bits", bit: 1 });
    const crops: CropMasks = { grid, encoding: "crops", masks: [{ index: 2, name: "E", color: [0, 0, 0], num_voxels: 0, bbox: null, data: null }] };
    expect(decodeMasks(crops).masks[0].storage).toMatchObject({ kind: "bits", bbox: null });
  });
});

describe("result groups", () => {
  it("splits the result name off a component key at the first slash", () => {
    expect(splitResultKey("physical_dose_beam 0")).toEqual(["", "physical_dose_beam 0"]);
    expect(splitResultKey("protons_30fx/physical_dose_scenario 2")).toEqual(["protons_30fx", "physical_dose_scenario 2"]);
    expect(splitResultKey("A/b/c")).toEqual(["A", "b/c"]);
    expect(splitResultKey("/physical_dose")).toEqual(["", "/physical_dose"]);
    expect(splitResultKey("A/")).toEqual(["", "A/"]);
    expect(resultKey("", "physical_dose")).toBe("physical_dose");
    expect(resultKey("A", "physical_dose_beam 1")).toBe("A/physical_dose_beam 1");
  });

  it("names the quantities of a sequenced fluence after their quantity", () => {
    const groups = groupResultKeys(["physical_dose", "physical_dose_sequenced", "physical_dose_sequenced_beam 0", "A/rbe_x_dose_sequenced", "alpha_sequenced"]);
    expect(groups.map((g) => [g.base, g.label, g.meta_label, g.unit])).toEqual([
      ["physical_dose", "dose", "Dose", "Gy"],
      ["physical_dose_sequenced", "dose (sequenced)", "Dose", "Gy"],
      ["rbe_x_dose_sequenced", "RBExDose (sequenced) (A)", "RBE-weighted Dose", "Gy (RBE)"],
      ["alpha_sequenced", "alpha_sequenced", "Quantity", ""],
    ]);
    expect(groups[1].components.map((c) => [c.key, c.label, c.kind])).toEqual([
      ["physical_dose_sequenced", "full", "full"],
      ["physical_dose_sequenced_beam 0", "beam 0", "beam"],
    ]);
    expect(keyLabel(groups, "physical_dose_sequenced_beam 0")).toBe("dose (sequenced), beam 0");
  });

  it("groups components by result and quantity like group_result_keys", () => {
    const keys = [
      "physical_dose",
      "physical_dose_beam 0",
      "physical_dose_beam 1",
      "physical_dose_scenario 0",
      "A/physical_dose",
      "A/physical_dose_beam 0",
      "A/physical_dose_beam 1",
      "A/physical_dose_scenario 0",
      "beam/physical_dose",
      "physical_dose_var",
    ];
    const groups = groupResultKeys(keys, { "": ["nominal"], A: ["shifted"] });
    expect(groups.map((g) => g.label)).toEqual(["dose", "dose (A)", "dose (beam)", "physical_dose_var"]);
    expect(groups.map((g) => g.tag)).toEqual(["", "A", "beam", ""]);
    expect(groups[0].components.map((c) => c.label)).toEqual(["full", "beam 0", "beam 1", "scenario 0"]);
    expect(groups[1].components.map((c) => c.key)).toEqual(["A/physical_dose", "A/physical_dose_beam 0", "A/physical_dose_beam 1", "A/physical_dose_scenario 0"]);
    expect(groups[1].components.map((c) => c.kind)).toEqual(["full", "beam", "beam", "scenario"]);
    expect(groups[0].meta_label).toBe("Dose");
    expect(groups[1].unit).toBe("Gy");
    // Every group carries the scenario labels of its own result.
    expect(groups[0].scenario_labels).toEqual(["nominal"]);
    expect(groups[1].scenario_labels).toEqual(["shifted"]);
    expect(groups[2].scenario_labels).toEqual([]);
    // A key that merely ends like a result name belongs to no other result: nothing is guessed.
    expect(groupResultKeys(["physical_dose_A", "physical_dose_beam_A 0"]).map((g) => [g.base, g.tag])).toEqual([
      ["physical_dose_A", ""],
      ["physical_dose_beam_A", ""],
    ]);
    expect(keyLabel(groups, "A/physical_dose_beam 1")).toBe("dose (A), beam 1");
    expect(sanitizeTag("  plan A/2 ")).toBe("plan_A_2");
  });

  it("chooses the Qt default checked VOIs", () => {
    const s = (index: number, name: string, type: string, extra: Record<string, number> = {}) => ({ name, type, color: [0, 0, 0] as [number, number, number], visible: true, index, ...extra });
    expect([...defaultCheckedVois([s(0, "BODY", "OAR", { num_voxels: 10 }), s(1, "Core", "OAR", { num_voxels: 5 }), s(2, "PTV", "TARGET", { num_voxels: 3 })])].sort()).toEqual([0, 2]);
    expect([...defaultCheckedVois([s(0, "A", "OAR", { num_objectives: 1, num_voxels: 1 }), s(1, "B", "TARGET", { num_objectives: 0, num_voxels: 1 })])]).toEqual([0]);
    expect([...defaultCheckedVois([s(0, "Empty", "TARGET", { num_voxels: 0, num_objectives: 0 })])]).toEqual([]);
    expect([...defaultCheckedVois([s(0, "Twin", "OAR", { num_objectives: 1, num_voxels: 1 }), s(1, "Twin", "OAR", { num_objectives: 0, num_voxels: 1 })])]).toEqual([0]);
  });
});

describe("analysis helpers", () => {
  it("reads D_x from the histogram without interpolating plateaus", () => {
    const bins = [0, 1, 2, 3, 4];
    const cum = [100, 100, 60, 60, 10];
    expect(doseAtVolume(bins, cum, 95)).toBe(1);
    expect(doseAtVolume(bins, cum, 60)).toBe(3);
    expect(doseAtVolume(bins, cum, 0)).toBe(4);
  });

  it("builds DVHs and QIs with DVH.compute / QICollection semantics", () => {
    const values = Float32Array.from([0, 10, 20, 30, 40]);
    const mask = new Uint8Array([1, 1, 1, 1, 1]);
    const dvh = computeDvh(values, mask, 1000);
    expect(dvh.dose[0]).toBe(0);
    expect(dvh.volume[0]).toBeCloseTo(100);
    expect(dvh.dose[999]).toBeCloseTo(0.999 * 42, 4);
    expect(dvh.stats.mean).toBeCloseTo(20);
    expect(percentile(Float64Array.from([0, 10, 20, 30, 40]), 25)).toBe(10);
    const row = qiRow(values, mask, [50, 95], [20]);
    expect(row.D50).toBe(20);
    expect(row.D95).toBeCloseTo(2);
    expect(row.V20Gy).toBe(60);
    expect(row.std).toBeCloseTo(Math.sqrt(200));
  });

  it("formats markers, table rows and QI points", () => {
    const qi = {
      quantity: "physical_dose",
      label: "Dose",
      unit: "Gy",
      unit_symbol: "Gy",
      ref_vols: [95],
      ref_doses: [20],
      metrics: [
        { id: "mean", label: "mean [Gy]", kind: "mean" as const, unit: "Gy" },
        { id: "D95", label: "D95 [Gy]", kind: "dx" as const, parameter: 95, unit: "Gy" },
        { id: "V20Gy", label: "V20Gy [%]", kind: "vx" as const, parameter: 20, unit: "%" },
      ],
      rows: [
        { index: 4, name: "PTV", color: [255, 0, 0] as [number, number, number], values: { mean: 50.123, D95: 47.5, V20Gy: 100 } },
        { index: 5, name: "Empty", color: [0, 0, 0] as [number, number, number], values: { mean: null, D95: null, V20Gy: null } },
      ],
    };
    const markers = qiMarkers(qi, [4, 5], {});
    expect(markers.map((m) => [m.metric, m.kind, m.parameter, m.value])).toEqual([
      ["D95", "dx", 95, 47.5],
      ["V20Gy", "vx", 20, 100],
    ]);
    const rows = qiTableRows(qi, [4, 5], ["mean", "D95"]);
    expect(rows).toEqual([
      ["Structure", "mean [Gy]", "D95 [Gy]"],
      ["PTV", "50.12", "47.50"],
      ["Empty", "-", "-"],
    ]);
    expect(toCsv([["a,b", "c"]])).toBe('"a,b",c\r\n');
    expect(applyPoint([2, 50, 98], 90, 98)).toEqual([2, 50, 90]);
    expect(applyPoint([2, 50], 50)).toEqual([2, 50]);
    expect(parseRefVols("2 120")).toBeNull();
  });
});

describe("mock v1 viewer methods", () => {
  async function ready() {
    const backend = new MockBackend({ speed: 0 });
    const { session } = await backend.request("session.create", {});
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("plan.set", { gantry_angles: [0, 120, 240], couch_angles: [0, 0, 0] }, session);
    await backend.request("workflow.run_all", {}, session);
    return { backend, session };
  }

  it("reports revisions and result groups that change with writes", async () => {
    const { backend, session } = await ready();
    const s1 = await backend.request("session.state", {}, session);
    expect(s1.results?.groups[0].components.map((c) => c.key)).toEqual(["physical_dose", "physical_dose_beam 0", "physical_dose_beam 1", "physical_dose_beam 2"]);
    expect(s1.results?.default).toBe("physical_dose");
    const ct1 = s1.revisions!.volumes.ct;
    await backend.request("workflow.recalc", {}, session);
    const s2 = await backend.request("session.state", {}, session);
    expect(s2.revisions!.volumes.ct).toBe(ct1);
    expect(s2.revisions!.volumes.physical_dose).not.toBe(s1.revisions!.volumes.physical_dose);
    await backend.request("plan.set", { gantry_angles: [0, 90], couch_angles: [0, 0] }, session);
    const s3 = await backend.request("session.state", {}, session);
    expect(s3.revisions!.beams).not.toBe(s2.revisions!.beams);
  });

  it("snapshots the active result as a result of its own and keeps it across optimizations", async () => {
    const { backend, session } = await ready();
    const before = await backend.request("session.state", {}, session);
    expect(before.results).toMatchObject({ names: ["current"], active: "current", tags: [] });
    const { tag, state } = await backend.request("workflow.snapshot", { tag: "first try" }, session);
    expect(tag).toBe("first_try");
    expect(state.results).toMatchObject({ names: ["current", "first_try"], active: "current", tags: ["first_try"] });
    expect(state.results?.groups.map((g) => [g.label, g.tag])).toEqual([
      ["dose", ""],
      ["dose (first_try)", "first_try"],
    ]);
    expect(state.results?.groups[1].components.map((c) => c.key)).toEqual([
      "first_try/physical_dose",
      "first_try/physical_dose_beam 0",
      "first_try/physical_dose_beam 1",
      "first_try/physical_dose_beam 2",
    ]);
    // The copy shares the images of the active result, and so their revisions (amendment 14).
    expect(state.revisions!.volumes["first_try/physical_dose"]).toBe(state.revisions!.volumes.physical_dose);
    // "beam" and "scenario" are ordinary names now; only the name of the active result is refused.
    const beam = await backend.request("workflow.snapshot", { tag: "beam" }, session);
    expect(beam.state.results?.tags).toEqual(["first_try", "beam"]);
    await expect(backend.request("workflow.snapshot", { tag: "current" }, session)).rejects.toMatchObject({ type: "ValueError" });
    // Reusing a name overwrites that result.
    const again = await backend.request("workflow.snapshot", { tag: "first try" }, session);
    expect(again.state.results?.names).toEqual(["current", "first_try", "beam"]);
    // Optimize replaces only the active result.
    await backend.request("workflow.optimize", { max_iter: 3 }, session);
    const after = await backend.request("session.state", {}, session);
    expect(after.results?.names).toEqual(["current", "first_try", "beam"]);
    expect(after.quantities).toContain("first_try/physical_dose");
    expect(after.revisions!.volumes.physical_dose).not.toBe(state.revisions!.volumes.physical_dose);
    expect(after.revisions!.volumes["first_try/physical_dose"]).toBe(state.revisions!.volumes["first_try/physical_dose"]);
    const beamVol = await backend.request("view.volume", { name: "first_try/physical_dose_beam 1" }, session);
    expect(beamVol.data.shape).toEqual([129, 167, 167]);
    const dvh = await backend.request("analysis.dvh", { keys: ["first_try/physical_dose_beam 1"], num_points: 50, range: "per_structure" }, session);
    expect(dvh.sets?.[0]).toMatchObject({ name: "dose (first_try), beam 1", unit: "Gy" });
    await expect(backend.request("view.volume", { name: "physical_dose_first_try" }, session)).rejects.toMatchObject({ type: "KeyError" });
    // A recalculation keeps the further results too.
    await backend.request("workflow.recalc", {}, session);
    expect((await backend.request("session.state", {}, session)).results?.tags).toEqual(["first_try", "beam"]);
  });

  it("keeps the scenario labels of every result with its groups", async () => {
    const backend = new MockBackend({ speed: 0 });
    const { session } = await backend.request("session.create", {});
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("plan.set", { scenarios: { model: "wcScen", values: {} } }, session);
    await backend.request("workflow.calc_dose_influence", {}, session);
    await backend.request("workflow.optimize", { max_iter: 3 }, session);
    await backend.request("workflow.snapshot", { tag: "robust" }, session);
    await backend.request("plan.set", { scenarios: { model: "nomScen", values: {} } }, session);
    const state = await backend.request("workflow.run_all", { max_iter: 3 }, session);
    const group = (tag: string) => state.results!.groups.find((g) => g.base === "physical_dose" && g.tag === tag)!;
    // The nominal active result has no scenarios; the snapshot keeps its own.
    expect(state.results!.scenario_labels).toEqual([]);
    expect(group("").scenario_labels).toEqual([]);
    expect(group("").components.some((c) => c.kind === "scenario")).toBe(false);
    expect(group("robust").scenario_labels).toHaveLength(7);
    expect(group("robust").scenario_labels[1]).toBe("setup (-2.25, 0.0, 0.0) mm");
    expect(group("robust").components.filter((c) => c.kind === "scenario").map((c) => c.key)[2]).toBe("robust/physical_dose_scenario 2");
    expect(resultGroups(state)).toBe(state.results!.groups);
    // A backend without labels per group only describes those of the active result.
    const legacy = {
      ...state,
      results: { ...state.results!, scenario_labels: ["nominal"], groups: state.results!.groups.map(({ scenario_labels: _labels, ...g }) => g) },
    } as unknown as typeof state;
    expect(resultGroups(legacy).map((g) => g.scenario_labels)).toEqual([["nominal"], []]);
  });

  it("offers the scenario components of a robust plan, labelled, in the viewer and the analysis", async () => {
    const backend = new MockBackend({ speed: 0 });
    const { session } = await backend.request("session.create", {});
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    await backend.request("plan.set", { scenarios: { model: "wcScen", values: {} } }, session);
    await backend.request("workflow.calc_dose_influence", {}, session);
    const state = await backend.request("workflow.optimize", { max_iter: 3 }, session);
    const dose = state.results!.groups.find((g) => g.base === "physical_dose" && !g.tag)!;
    const scenarios = dose.components.filter((c) => c.kind === "scenario");
    expect(scenarios.map((c) => c.key)).toEqual(Array.from({ length: 7 }, (_, i) => `physical_dose_scenario ${i}`));
    expect(state.results!.scenario_labels[0]).toBe("nominal");
    expect(state.results!.scenario_labels[1]).toBe("setup (-2.25, 0.0, 0.0) mm");
    expect(dose.scenario_labels).toEqual(state.results!.scenario_labels);
    expect(state.quantities).toEqual(["physical_dose"]);
    const full = await backend.request("view.volume", { name: "physical_dose" }, session);
    const nominal = await backend.request("view.volume", { name: "physical_dose_scenario 0" }, session);
    const shifted = await backend.request("view.volume", { name: "physical_dose_scenario 2" }, session);
    expect(nominal.data.data).toEqual(full.data.data);
    expect(shifted.data.data).not.toEqual(full.data.data);
    const dvh = await backend.request("analysis.dvh", { keys: ["physical_dose_scenario 2"], num_points: 100, range: "per_structure" }, session);
    expect(dvh.sets?.[0].name).toBe("dose, scenario 2");
    expect(keyLabel(state.results!.groups, "physical_dose_scenario 2")).toBe("dose, scenario 2");
    await backend.request("plan.set", { scenarios: { model: "nomScen", values: {} } }, session);
    await backend.request("workflow.run_all", { max_iter: 3 }, session);
    const nominalPlan = await backend.request("session.state", {}, session);
    expect(nominalPlan.results!.scenario_labels).toEqual([]);
    expect(nominalPlan.results!.groups[0].components.some((c) => c.kind === "scenario")).toBe(false);
  });

  it("serves beams, compact masks, DVH sets and QIs", async () => {
    const { backend, session } = await ready();
    const beams = await backend.request("view.beams", {}, session);
    expect(beams.source).toBe("stf");
    expect(beams.beams.map((b) => b.gantry)).toEqual([0, 120, 240]);
    expect(beams.beams[1].source_direction[0]).toBeCloseTo(Math.sin((120 * Math.PI) / 180));
    const draft = await backend.request("view.beams", { draft: { gantry_angles: [0, 130, 240], couch_angles: [0, 0, 0] } }, session);
    expect(draft.source).toBe("plan");
    expect(draft.beams.map((b) => b.pending)).toEqual([false, true, false]);

    const crops = (await backend.request("view.structure_masks", { encoding: "crops" }, session)) as CropMasks;
    const core = crops.masks.find((m) => m.name === "Core")!;
    const [z0, y0, x0, z1, y1, x1] = core.bbox!;
    expect(core.data!.shape).toEqual([z1 - z0, y1 - y0, x1 - x0]);
    const bits = (await backend.request("view.structure_masks", { encoding: "bits" }, session)) as BitsMasks;
    expect(bits.masks.map((m) => [m.plane, m.bit])).toEqual([
      [0, 0],
      [0, 1],
      [0, 2],
    ]);

    const dvh = await backend.request("analysis.dvh", { keys: ["physical_dose", "physical_dose_beam 0"], num_points: 1000, range: "per_structure" }, session);
    expect(dvh.sets!.map((s) => s.key)).toEqual(["physical_dose", "physical_dose_beam 0"]);
    const target = dvh.sets![0].curves.find((c) => c.name === "OuterTarget")!;
    expect(target.dose.data.length).toBe(1000);
    expect(target.dose.data[999]).toBeCloseTo((target.stats!.max! * 1.05 * 999) / 1000, 2);
    const qi = await backend.request("analysis.qi", { quantity: "physical_dose", names: [1], ref_vols: [95] }, session);
    expect(qi.metrics.map((m) => m.id).slice(0, 5)).toEqual(["mean", "std", "max", "min", "D95"]);
    expect(qi.ref_doses.length).toBe(5);
    expect(qi.rows[0].values.D95).toBeGreaterThan(0);
    expect((await backend.request("analysis.compare_units", { a: "physical_dose", b: "physical_dose_beam 0" }, session)).shareable).toBe(true);
  });

  it("updates structures addressed by position or name", async () => {
    const { backend, session } = await ready();
    const r = await backend.request("cst.update_structure", { name: 0, changes: { color: [1, 2, 3], overlap_priority: 7 } }, session);
    expect(r.structure).toMatchObject({ index: 0, name: "Core", color: [1, 2, 3] });
    expect(r.state.structures.find((s) => s.name === "Core")?.overlap_priority).toBe(7);
    const ok = await backend.request("cst.update_structure", { name: "Core" as unknown as number, changes: { tissue: { model: "Generic", values: { alpha_x: 0.2, beta_x: 0.05 } } } }, session);
    expect(ok.structure.alpha_x).toBe(0.2);
    const detail = await backend.request("cst.get", { names: [0] }, session);
    expect(detail.structures).toHaveLength(1);
    expect(detail.structures[0].overlap_priority).toBe(7);
    const missing = await backend.request("cst.update_structure", { name: 9, changes: {} }, session).catch((e: unknown) => e);
    expect((missing as BackendError).type).toBe("KeyError");
  });

  it("answers per-structure views with positions and refuses ambiguous names", async () => {
    const { backend, session } = await ready();
    const masks = await backend.request("view.structure_masks", { names: [2, 1] }, session);
    expect(masks.masks.map((m) => [m.index, m.name])).toEqual([
      [1, "OuterTarget"],
      [2, "BODY"],
    ]);
    const crops = (await backend.request("view.structure_masks", { encoding: "crops", names: [1] }, session)) as CropMasks;
    expect(crops.masks.map((m) => m.index)).toEqual([1]);
    const dvh = await backend.request("analysis.dvh", { keys: ["physical_dose"], names: [1], range: "per_structure" }, session);
    expect(dvh.sets![0].curves.map((c) => [c.index, c.name])).toEqual([[1, "OuterTarget"]]);
    const qi = await backend.request("analysis.qi", { quantity: "physical_dose", names: [1] }, session);
    expect(qi.rows.map((r) => [r.index, r.name])).toEqual([[1, "OuterTarget"]]);
    const phantom = (backend as unknown as { sessions: Map<string, { phantom: { structures: { name: string }[] } }> }).sessions.get(session)!.phantom;
    phantom.structures[0].name = phantom.structures[1].name;
    const ambiguous = await backend.request("cst.update_structure", { name: "OuterTarget" as unknown as number, changes: {} }, session).catch((e: unknown) => e);
    expect((ambiguous as BackendError).message).toMatch(/ambiguous/);
    const byPosition = await backend.request("cst.update_structure", { name: 0, changes: { color: [9, 9, 9] } }, session);
    expect(byPosition.structure).toMatchObject({ index: 0, color: [9, 9, 9] });
    expect((await backend.request("cst.get", {}, session)).structures.map((d) => d.index)).toEqual([0, 1, 2]);
  });
});

import { describe, expect, it } from "vitest";
import { BackendError } from "../src/api/backend";
import { ndarray } from "../src/api/codec";
import { MockBackend } from "../src/api/mock/mockBackend";
import type { SequenceFluence, SequenceShape, SequenceTarget } from "../src/api/protocol/viewer";
import { mapColor } from "../src/renderer/colormaps";
import {
  apertureLimits,
  apertureRects,
  apertureTitle,
  axisScale,
  beamOfQuantity,
  bixelGrid,
  dwellMs,
  equalAspectBox,
  fluenceColor,
  fluenceGrid,
  fluenceLimits,
  fluenceTitle,
  formatG,
  gridPositions,
  leafNumber,
  leafRowBounds,
  mlcBorderRects,
  nextShape,
  sequencerLabel,
  targetSegments,
  weightColor,
} from "../src/components/sequence/sequenceModel";

// Three 5 mm leaf rows from z = -10; the middle one is closed at the left field edge.
const shape: SequenceShape = {
  leaf_width: 5,
  boundaries: [-10, -5, 0],
  leaf_positions: [
    [-20, 10],
    [-30, -30],
    [0, 30],
  ],
};
const window: [number, number, number, number] = [-30, 30, -10, 5];

const fluence = (optimized: number[] | null, delivered: number[] | null): SequenceFluence => ({
  window: [-10, 10, -10, 0],
  bixel_width: 10,
  optimized: optimized && ndarray(Float32Array.from(optimized), [1, 2]),
  delivered: delivered && ndarray(Float32Array.from(delivered), [1, 2]),
});

describe("sequence window geometry (_plot_aperture_info.py)", () => {
  it("places leaf rows by their lower edge, or numbers them from the lowest row", () => {
    expect(leafRowBounds(shape, "physical")).toEqual({ low: [-10, -5, 0], high: [-5, 0, 5] });
    expect(leafRowBounds(shape, "leaf_num")).toEqual({ low: [0.5, 1.5, 2.5], high: [1.5, 2.5, 3.5] });
    expect(apertureLimits(shape, window, "physical")).toEqual([-30, 30, -10, 5]);
    expect(apertureLimits(shape, window, "leaf_num")).toEqual([-30, 30, 0.5, 3.5]);
    expect(leafNumber(-10, shape)).toBe(0.5);
    expect(leafNumber(2.5, shape)).toBe(3);
    expect(leafNumber(7, { ...shape, boundaries: [] })).toBe(7);
  });

  it("blocks the field left and right of every opening", () => {
    const rects = apertureRects(shape, window, "physical");
    expect(rects).toHaveLength(6);
    expect(rects[0]).toEqual({ x: -30, y: -10, width: 10, height: 5 });
    expect(rects[1]).toEqual({ x: 10, y: -10, width: 20, height: 5 });
    expect(rects[2]).toEqual({ x: -30, y: -5, width: 0, height: 5 });
    expect(rects[3]).toEqual({ x: -30, y: -5, width: 60, height: 5 });
    expect(apertureRects(shape, window, "leaf_num")[5]).toEqual({ x: 30, y: 2.5, width: 0, height: 1 });
    // The outlines on the fluence map leave out leaves retracted to the field edge.
    expect(mlcBorderRects(shape, window).map((r) => [r.x, r.y, r.width])).toEqual([
      [-30, -10, 10],
      [10, -10, 20],
      [-30, -5, 60],
      [-30, 0, 30],
    ]);
  });

  it("shades the opening from light grey to blue by the weight relative to the beam's maximum", () => {
    expect(weightColor(0, 2)).toEqual([235, 235, 235]);
    expect(weightColor(2, 2)).toEqual([38, 89, 191]);
    expect(weightColor(1, 2)).toEqual([136, 162, 213]);
    expect(weightColor(5, 2)).toEqual(weightColor(2, 2));
    expect(weightColor(1, 0)).toEqual([235, 235, 235]);
  });

  it("puts the bixel grid on the bixel boundaries, on the leaf-pair axis if asked", () => {
    expect(gridPositions(-10, 10, 10)).toEqual([-10, 0, 10]);
    expect(gridPositions(-10, 10, 0)).toEqual([]);
    const f = fluence([1, 2], [1, 2]);
    expect(bixelGrid(f)).toEqual({ x: [-10, 0, 10], y: [-10, 0] });
    expect(bixelGrid(f, shape)).toEqual({ x: [-10, 0, 10], y: [0.5, 2.5] });
    expect(bixelGrid({ ...f, bixel_width: null })).toEqual({ x: [], y: [] });
  });

  it("maps the vertical axis upwards and keeps the fluence map's aspect ratio", () => {
    const y = axisScale(-10, 30, 100, 200, true);
    expect(y(-10)).toBe(300);
    expect(y(30)).toBe(100);
    const x = axisScale(-10, 30, 100, 200);
    expect([x(-10), x(30)]).toEqual([100, 300]);
    expect(equalAspectBox({ left: 0, top: 0, width: 300, height: 100 }, 20, 10)).toEqual({ left: 50, top: 0, width: 200, height: 100 });
    expect(equalAspectBox({ left: 10, top: 0, width: 100, height: 100 }, 20, 10)).toEqual({ left: 10, top: 25, width: 100, height: 50 });
  });

  it("outlines the projected target at the edges of its bins", () => {
    // 3 x 4 bins (stack, travel), bin centres at origin + index * spacing; one bin set.
    const mask = new Uint8Array(12);
    mask[1 * 4 + 2] = 1;
    const target: SequenceTarget = { origin: [100, -50], spacing: [2, 4], mask: ndarray(mask, [3, 4]) };
    const segments = targetSegments(target);
    const xs = [...segments].filter((_, i) => i % 2 === 0);
    const ys = [...segments].filter((_, i) => i % 2 === 1);
    // The contour runs half way between the centre (104, -46) and its neighbours.
    expect([Math.min(...xs), Math.max(...xs)]).toEqual([103, 105]);
    expect([Math.min(...ys), Math.max(...ys)]).toEqual([-48, -44]);
    expect(segments.length).toBe(16);
  });
});

describe("sequence window fluence panel", () => {
  it("shows the delivered, optimized or delivered − optimized map", () => {
    const f = fluence([1, NaN], [3, NaN]);
    expect(fluenceGrid(f, "delivered")).toMatchObject({ rows: 1, cols: 2 });
    expect([...fluenceGrid(f, "optimized")!.data]).toEqual([1, NaN]);
    expect([...fluenceGrid(f, "difference")!.data]).toEqual([2, NaN]);
    expect(fluenceGrid(fluence(null, [1, 2]), "difference")).toBeNull();
    expect(fluenceGrid(fluence(null, [1, 2]), "optimized")).toBeNull();
  });

  it("scales viridis from 0 to the maximum and RdBu_r symmetrically, ignoring bixels without a ray", () => {
    const grid = fluenceGrid(fluence([1, NaN], [4, NaN]), "delivered")!;
    expect(fluenceLimits(grid, "delivered")).toEqual([0, 4]);
    const diff = fluenceGrid(fluence([4, 1], [1, 2]), "difference")!;
    expect(fluenceLimits(diff, "difference")).toEqual([-3, 3]);
    expect(fluenceLimits({ data: Float32Array.from([NaN, 0]), rows: 1, cols: 2 }, "delivered")).toEqual([0, 1]);
    expect(fluenceLimits({ data: Float32Array.from([0, 0]), rows: 1, cols: 2 }, "difference")).toEqual([-1, 1]);
    expect(fluenceColor(NaN, [0, 4], "delivered")).toBeNull();
    expect(fluenceColor(0, [0, 4], "delivered")).toEqual(mapColor("viridis", 0));
    expect(fluenceColor(4, [0, 4], "delivered")).toEqual(mapColor("viridis", 1));
    expect(fluenceColor(0, [-3, 3], "difference")).toEqual(mapColor("RdBu_r", 0.5));
    // RdBu_r: negative (under-delivered) blue, positive red.
    const low = fluenceColor(-3, [-3, 3], "difference")!;
    const high = fluenceColor(3, [-3, 3], "difference")!;
    expect(low[2]).toBeGreaterThan(low[0]);
    expect(high[0]).toBeGreaterThan(high[2]);
  });

  it("titles the panels as Qt does", () => {
    expect(fluenceTitle("delivered")).toBe("Delivered fluence");
    expect(fluenceTitle("optimized")).toBe("Optimized fluence");
    expect(fluenceTitle("difference")).toBe("Difference fluence (delivered − optimized)");
    expect(apertureTitle("Beam 0 (45°)", 1, 7, 0.12345)).toBe("Beam 0 (45°) — shape 2/7, w=0.12");
    expect(sequencerLabel({ sequencer: "xia", num_of_levels: 5 })).toBe("Sequencer: xia (5 levels)");
    expect(sequencerLabel({ sequencer: "xia", num_of_levels: null })).toBe("Sequencer: xia");
    expect(sequencerLabel({ sequencer: null, num_of_levels: 5 })).toBeNull();
  });

  it("formats weights like Python's %.2g", () => {
    expect([1, 0.5, 12.3, 123, 0.00012345, 0.000012345, 0, 0.995, 99.5].map((v) => formatG(v))).toEqual([
      "1",
      "0.5",
      "12",
      "1.2e+02",
      "0.00012",
      "1.2e-05",
      "0",
      "0.99",
      "1e+02",
    ]);
  });
});

describe("sequence window playback", () => {
  it("dwells on a shape in proportion to its weight, at least 150 ms", () => {
    expect(dwellMs(2, 2, 1)).toBe(900);
    expect(dwellMs(1, 2, 1)).toBe(450);
    expect(dwellMs(1, 2, 2)).toBe(225);
    expect(dwellMs(0.1, 2, 1)).toBe(150);
    expect(dwellMs(2, 2, 10)).toBe(150);
    expect(dwellMs(2, 2, 0.1)).toBe(9000);
    expect(dwellMs(0, 0, 1)).toBe(900);
  });

  it("steps through the shapes and wraps only with Loop", () => {
    expect(nextShape(0, 3, false)).toBe(1);
    expect(nextShape(2, 3, true)).toBe(0);
    expect(nextShape(2, 3, false)).toBeNull();
  });

  it("finds the beam of a per-beam quantity", () => {
    expect(beamOfQuantity("physical_dose_beam 3")).toBe(3);
    expect(beamOfQuantity("physical_dose_sequenced_beam 12")).toBe(12);
    expect(beamOfQuantity("before/physical_dose_beam 1")).toBe(1);
    expect(beamOfQuantity("physical_dose")).toBeNull();
    expect(beamOfQuantity("physical_dose_scenario 2")).toBeNull();
    expect(beamOfQuantity(null)).toBeNull();
  });
});

describe("mock backend, view.sequence", () => {
  async function optimized() {
    const backend = new MockBackend({ speed: 0 });
    await backend.ready();
    const { session } = await backend.request("session.create", {});
    await backend.request("data.load_phantom", { name: "TG119" }, session);
    const req = <M extends Parameters<typeof backend.request>[0]>(method: M, params: Parameters<typeof backend.request<M>>[1]) =>
      backend.request(method, params, session);
    await req("plan.set", { gantry_angles: [0, 90, 200], couch_angles: [0, 0, 0], bixel_width: 5 });
    await req("workflow.calc_dose_influence", {});
    await req("workflow.optimize", { max_iter: 5 });
    return { backend, req };
  }

  it("serves the sequence of the active result once it is sequenced", async () => {
    const { backend, req } = await optimized();
    expect((await backend.request("app.info", {})).methods).toContain("view.sequence");
    let state = await req("plan.set", { sequencing: { run: true, sequencer: "xia", values: { num_of_levels: 4 } } });
    expect(state.sequencing).toEqual({ available: true, reason: null, sequenced: false });
    const refused = (await req("view.sequence", {}).catch((e: unknown) => e)) as BackendError;
    expect(refused).toBeInstanceOf(BackendError);
    expect([refused.type, refused.code]).toEqual(["PreconditionError", "no_sequence"]);

    state = await req("workflow.sequence", {});
    expect(state.sequencing?.sequenced).toBe(true);
    const view = await req("view.sequence", {});
    expect([view.sequencer, view.num_of_levels]).toEqual(["xia", 4]);
    expect(view.beams.map((b) => b.label)).toEqual(["Beam 0 (0°)", "Beam 1 (90°)", "Beam 2 (200°)"]);
    for (const beam of view.beams) {
      expect(beam.orientation).toBe("X");
      expect(beam.shapes).toHaveLength(4);
      expect(beam.weights).toHaveLength(4);
      const { window: w, bixel_width: bw, optimized: opt, delivered } = beam.fluence!;
      expect(bw).toBe(5);
      expect(w).toEqual(beam.window);
      expect(opt!.shape).toEqual(delivered!.shape);
      expect(opt!.shape[1]).toBe(Math.round((w[1] - w[0]) / bw!));
      expect(opt!.shape[0]).toBe(Math.round((w[3] - w[2]) / bw!));
      expect(opt!.data.some(Number.isNaN)).toBe(true);
      const shape = beam.shapes[0];
      expect(shape.boundaries).toHaveLength(shape.leaf_positions.length);
      expect(shape.boundaries[0]).toBe(w[2]);
      expect(shape.leaf_positions.every(([l, r]) => w[0] <= l && l <= r && r <= w[1])).toBe(true);
      // The delivered fluence is what the shapes add up to, within half a level of the optimized one.
      const [rows, cols] = opt!.shape;
      const sum = new Float32Array(rows * cols);
      beam.shapes.forEach((s, j) =>
        s.leaf_positions.forEach(([l, r], row) => {
          for (let c = Math.round((l - w[0]) / bw!); c < Math.round((r - w[0]) / bw!); c++) sum[row * cols + c] += beam.weights[j];
        }),
      );
      const step = Math.max(...beam.weights);
      delivered!.data.forEach((v, i) => {
        if (Number.isNaN(v)) return expect(Number.isNaN(opt!.data[i])).toBe(true);
        expect(v).toBeCloseTo(sum[i], 5);
        expect(Math.abs(v - opt!.data[i])).toBeLessThanOrEqual(step);
      });
      // The projected target lies inside the field the leaves can open, with an empty bin around.
      const target = beam.target!;
      const [tr, tc] = target.mask.shape;
      const xs: number[] = [];
      const ys: number[] = [];
      target.mask.data.forEach((v, i) => {
        if (!v) return;
        xs.push(target.origin[0] + (i % tc) * target.spacing[0]);
        ys.push(target.origin[1] + Math.floor(i / tc) * target.spacing[1]);
      });
      expect(w[0] < Math.min(...xs) && Math.max(...xs) < w[1] && w[2] < Math.min(...ys) && Math.max(...ys) < w[3]).toBe(true);
      expect(target.mask.data.subarray(0, tc).some(Boolean) || target.mask.data.subarray((tr - 1) * tc).some(Boolean)).toBe(false);
    }
    // The TG119 target is as long as it is wide seen from 0° and 90°, the sequence follows the levels.
    expect(view.beams[0].target!.mask.shape[0]).toBe(view.beams[1].target!.mask.shape[0]);
    await req("plan.set", { sequencing: { run: true, sequencer: "siochi", values: { num_of_levels: 6 } } });
    await req("workflow.sequence", {});
    const second = await req("view.sequence", {});
    expect([second.sequencer, second.num_of_levels, second.beams[0].shapes.length]).toEqual(["siochi", 6, 6]);

    // A new optimization is a result without a sequence.
    state = await req("workflow.optimize", { max_iter: 5 });
    expect(state.sequencing?.sequenced).toBe(false);
    expect(((await req("view.sequence", {}).catch((e: unknown) => e)) as BackendError).code).toBe("no_sequence");
  });
});

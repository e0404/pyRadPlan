import { describe, expect, it } from "vitest";
import { formatPick, sliceRange, toPrecision4, volumeRange } from "../src/renderer/analysis";
import { COLORMAP_NAMES, colormapRGB, lutIndex, mapColor, normalize } from "../src/renderer/colormaps";
import { ContourCache, LruCache, maskBounds, maskCenterOfMass } from "../src/renderer/contours";
import {
  canvasToSlice,
  extractSlice,
  fragToSliceMatrix,
  linearIndex,
  panBy,
  sliceLayout,
  sliceToCanvas,
  sliceVoxel,
  texelMapping,
  viewTransform,
  slicesThrough,
  voxelToWorld,
  worldToIndex,
  worldToVoxel,
  zoomAt,
  type VolumeGeometry,
} from "../src/renderer/geometry";
import { boundingBox, marchingSquares, segmentsLength } from "../src/renderer/marchingSquares";
import { niceTicks } from "../src/components/viewerModel";

// nx = 4, ny = 3, nz = 2; value = 100 k + 10 j + i
const geom: VolumeGeometry = { shape: [2, 3, 4], spacing: [1, 2, 3], origin: [-10, 20, 5] };
const volume = new Float32Array(24).map((_, n) => {
  const i = n % 4;
  const j = Math.floor(n / 4) % 3;
  const k = Math.floor(n / 12);
  return 100 * k + 10 * j + i;
});

describe("slice geometry", () => {
  it("picks the Qt viewer's axes per plane", () => {
    expect(sliceLayout(geom, "axial")).toMatchObject({ nh: 4, nv: 3, sh: 1, sv: 2, count: 2 });
    expect(sliceLayout(geom, "sagittal")).toMatchObject({ nh: 2, nv: 3, sh: 3, sv: 2, count: 4 });
    expect(sliceLayout(geom, "coronal")).toMatchObject({ nh: 2, nv: 4, sh: 3, sv: 1, count: 3 });
    expect(sliceLayout(geom, "coronal", "radiological")).toMatchObject({ nh: 4, nv: 2, count: 3 });
  });

  it("maps between world and voxel indices through a non-identity direction (flipped z)", () => {
    const flipped: VolumeGeometry = { ...geom, direction: [1, 0, 0, 0, 1, 0, 0, 0, -1] };
    expect(voxelToWorld(flipped, [1, 1, 1])).toEqual([-9, 22, 2]);
    expect(voxelToWorld(flipped, [0, 0, 1])).toEqual([-10, 20, 2]);
    for (const ijk of [
      [0, 0, 0],
      [3, 2, 1],
      [1, 2, 0],
    ] as [number, number, number][]) {
      expect(worldToVoxel(flipped, voxelToWorld(flipped, ijk)).map((v) => +v.toFixed(9))).toEqual(ijk);
      expect(worldToIndex(flipped, voxelToWorld(flipped, ijk))).toEqual(ijk);
    }
    // World z = 2 is the second axial slice when z runs against the index, the first one for an identity direction.
    expect(worldToIndex(flipped, [-9, 22, 2])[2]).toBe(1);
    expect(worldToIndex(geom, [-9, 22, 2])[2]).toBe(0);

    const rotated: VolumeGeometry = { ...geom, direction: [0, -1, 0, 1, 0, 0, 0, 0, 1] };
    const world = voxelToWorld(rotated, [2, 1, 1]);
    expect(worldToVoxel(rotated, world).map((v) => +v.toFixed(9))).toEqual([2, 1, 1]);
  });

  it("puts the slices through an isocentre on the voxel the world point falls in", () => {
    const flipped: VolumeGeometry = { ...geom, direction: [1, 0, 0, 0, 1, 0, 0, 0, -1] };
    // Voxel [2, 1, 1] centre.
    const iso = voxelToWorld(flipped, [2, 1, 1]);
    expect(slicesThrough(flipped, iso)).toEqual({ axial: 1, sagittal: 2, coronal: 1 });
    expect(slicesThrough(geom, iso)).toEqual({ axial: 0, sagittal: 2, coronal: 1 });
    const mask = new Uint8Array(24);
    mask[(1 * 3 + 1) * 4 + 2] = 1;
    expect(maskCenterOfMass({ index: 0, name: "m", color: [0, 0, 0], data: mask }, flipped)).toEqual(iso);
  });

  it("extracts slices in screen layout", () => {
    const ax = extractSlice(volume, geom, sliceLayout(geom, "axial"), 1, Float32Array);
    expect(ax.width).toBe(4);
    expect(Array.from(ax.values.slice(0, 4))).toEqual([100, 101, 102, 103]);
    expect(ax.values[2 * 4 + 3]).toBe(123);
    const sag = extractSlice(volume, geom, sliceLayout(geom, "sagittal"), 2, Float32Array);
    // h = z, v = y
    expect(Array.from(sag.values)).toEqual([2, 102, 12, 112, 22, 122]);
    const cor = extractSlice(volume, geom, sliceLayout(geom, "coronal"), 1, Float32Array);
    // h = z, v = x
    expect(Array.from(cor.values)).toEqual([10, 110, 11, 111, 12, 112, 13, 113]);
    // Radiological sagittal: z flipped so the head is up.
    const rsag = extractSlice(volume, geom, sliceLayout(geom, "sagittal", "radiological"), 0, Float32Array);
    expect(Array.from(rsag.values)).toEqual([100, 110, 120, 0, 10, 20]);
    const crop = extractSlice(volume, geom, sliceLayout(geom, "axial"), 0, Float32Array, { u0: 1, u1: 3, v0: 1, v1: 3 });
    expect(Array.from(crop.values)).toEqual([11, 12, 21, 22]);
  });

  it("maps slice voxels to volume indices and world positions", () => {
    const layout = sliceLayout(geom, "sagittal");
    const ijk = sliceVoxel(layout, 1, 2, 3);
    expect(ijk).toEqual([3, 2, 1]);
    expect(volume[linearIndex(geom, ijk)]).toBe(123);
    expect(voxelToWorld(geom, [1, 1, 1])).toEqual([-9, 22, 8]);
    expect(worldToIndex(geom, [-9, 22, 8.4])).toEqual([1, 1, 1]);
    expect(worldToIndex(geom, [1000, -1000, 1000])).toEqual([3, 0, 1]);
    const t = texelMapping(sliceLayout(geom, "sagittal", "radiological"), 2);
    expect(Array.from(t.origin)).toEqual([2, 0, 1]);
    expect(Array.from(t.dirH)).toEqual([0, 1, 0]);
    expect(Array.from(t.dirV)).toEqual([0, 0, -1]);
  });

  it("fits slices at physical aspect ratio", () => {
    // Axial: 4 x 1 mm by 3 x 2 mm = 4 mm x 6 mm in a 400 x 400 canvas -> 66.67 px/mm.
    const layout = sliceLayout(geom, "axial");
    const t = viewTransform(layout, 400, 400, { zoom: 1, panH: 0, panV: 0 });
    expect(t.scaleV / t.scaleU).toBeCloseTo(2);
    expect(t.scaleV * layout.nv).toBeCloseTo(400);
    const [x, y] = sliceToCanvas(t, layout.nh / 2, layout.nv / 2);
    expect(x).toBeCloseTo(200);
    expect(y).toBeCloseTo(200);
    const [u, v] = canvasToSlice(t, x, y);
    expect(u).toBeCloseTo(2);
    expect(v).toBeCloseTo(1.5);
  });

  it("zooms around the cursor and pans by pixels", () => {
    const layout = sliceLayout(geom, "axial");
    const v0 = { zoom: 1, panH: 0, panV: 0 };
    const before = canvasToSlice(viewTransform(layout, 400, 300, v0), 100, 80);
    const v1 = zoomAt(layout, 400, 300, v0, 2.5, 100, 80);
    const after = canvasToSlice(viewTransform(layout, 400, 300, v1), 100, 80);
    expect(v1.zoom).toBe(2.5);
    expect(after[0]).toBeCloseTo(before[0]);
    expect(after[1]).toBeCloseTo(before[1]);
    const t1 = viewTransform(layout, 400, 300, v1);
    const v2 = panBy(layout, 400, 300, v1, 30, -12);
    const t2 = viewTransform(layout, 400, 300, v2);
    expect(t2.offsetX - t1.offsetX).toBeCloseTo(30);
    expect(t2.offsetY - t1.offsetY).toBeCloseTo(-12);
  });

  it("builds the fragment-to-slice matrix (bottom-left origin)", () => {
    const layout = sliceLayout(geom, "axial");
    const t = viewTransform(layout, 400, 300, { zoom: 1.3, panH: 0.4, panV: -1 });
    const m = fragToSliceMatrix(t, 300);
    const apply = (fx: number, fy: number) => [m[0] * fx + m[3] * fy + m[6], m[1] * fx + m[4] * fy + m[7]];
    for (const [x, y] of [
      [10, 20],
      [250.5, 120.5],
    ]) {
      const [u, v] = apply(x, 300 - y);
      const [eu, ev] = canvasToSlice(t, x, y);
      expect(u).toBeCloseTo(eu, 5);
      expect(v).toBeCloseTo(ev, 5);
    }
  });
});

describe("colormaps", () => {
  it("ships all Qt colormaps as 256-entry tables", () => {
    for (const name of COLORMAP_NAMES) expect(colormapRGB(name)).toHaveLength(768);
  });

  it("matches matplotlib endpoints", () => {
    expect(mapColor("gray", 0)).toEqual([0, 0, 0]);
    expect(mapColor("gray", 1)).toEqual([255, 255, 255]);
    expect(mapColor("jet", 0)).toEqual([0, 0, 128]);
    expect(mapColor("jet", 1)).toEqual([128, 0, 0]);
    expect(mapColor("viridis", 0)).toEqual([68, 1, 84]);
    expect(mapColor("viridis", 1)).toEqual([253, 231, 37]);
  });

  it("indexes the LUT like pyqtgraph and clamps", () => {
    expect(lutIndex(-1)).toBe(0);
    expect(lutIndex(0.5)).toBe(128);
    expect(lutIndex(0.999)).toBe(255);
    expect(lutIndex(2)).toBe(255);
    expect(lutIndex(Number.NaN)).toBe(0);
    expect(normalize(40, -160, 240)).toBeCloseTo(0.5);
    expect(normalize(5, 5, 5)).toBe(0);
  });
});

describe("marching squares", () => {
  it("traces a closed square around a single voxel", () => {
    const img = new Uint8Array(9);
    img[4] = 1;
    const segs = marchingSquares(img, 3, 3, 0.5);
    // Diamond through the edge midpoints between voxel centres: 4 segments, each sqrt(0.5).
    expect(segs.length / 4).toBe(4);
    expect(segmentsLength(segs)).toBeCloseTo(4 * Math.SQRT1_2);
    for (let i = 0; i < segs.length; i += 2) {
      expect(segs[i]).toBeGreaterThanOrEqual(1);
      expect(segs[i]).toBeLessThanOrEqual(2);
    }
  });

  it("runs curves up to the slice border (edge padding, no closing)", () => {
    // Left column inside: a vertical line at u = 1 from the top to the bottom border.
    const segs = marchingSquares(new Uint8Array([1, 0, 1, 0]), 2, 2, 0.5);
    for (let i = 0; i < segs.length; i += 2) expect(segs[i]).toBeCloseTo(1);
    expect(segmentsLength(segs)).toBeCloseTo(2);
    // A slice filled completely has no contour, as with the Qt viewer's edge padding.
    expect(marchingSquares(new Uint8Array([1, 1, 1, 1]), 2, 2, 0.5)).toHaveLength(0);
  });

  it("interpolates level crossings linearly and applies offsets", () => {
    const img = new Float32Array([0, 10]);
    const segs = marchingSquares(img, 2, 1, 2.5, { offsetU: 5, offsetV: 7 });
    // Crossing at 25 % between the centres u = 0.5 and 1.5 -> u = 0.75.
    expect(segs[0]).toBeCloseTo(5.75);
    expect(segs[2]).toBeCloseTo(5.75);
    const vs = Array.from(segs).filter((_, i) => i % 2 === 1);
    expect(Math.min(...vs)).toBeCloseTo(7);
    expect(Math.max(...vs)).toBeCloseTo(8);
    for (let i = 0; i < segs.length; i += 2) expect(segs[i]).toBeCloseTo(5.75);
  });

  it("resolves saddles by the cell centre", () => {
    const img = new Float32Array([1, 0, 0, 1]);
    const above = marchingSquares(img, 2, 2, 0.4);
    const below = marchingSquares(img, 2, 2, 0.6);
    expect(above.length).toBeGreaterThan(0);
    expect(below.length).toBeGreaterThan(0);
    expect(segmentsLength(above)).not.toBeCloseTo(0);
  });

  it("finds bounding boxes", () => {
    const img = new Uint8Array([0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0]);
    expect(boundingBox(img, 4, 3)).toEqual({ u0: 1, u1: 3, v0: 1, v1: 3 });
    expect(boundingBox(new Uint8Array(4), 2, 2)).toBeNull();
  });
});

describe("contour cache", () => {
  const g: VolumeGeometry = { shape: [3, 8, 8], spacing: [1, 1, 1], origin: [0, 0, 0] };
  const mask = new Uint8Array(3 * 64);
  for (let j = 2; j < 5; j++) for (let i = 3; i < 6; i++) mask[64 + j * 8 + i] = 1;

  it("computes bounds and crops, matching an uncropped trace", () => {
    expect(maskBounds(mask, g)).toEqual({ lo: [3, 2, 1], hi: [6, 5, 2] });
    const cache = new ContourCache();
    cache.setMasks([{ index: 0, name: "A", color: [255, 0, 0], data: mask }], g);
    const segs = cache.contour(0, "axial", 1);
    const full = marchingSquares(mask.subarray(64, 128), 8, 8, 0.5);
    expect(segmentsLength(segs)).toBeCloseTo(segmentsLength(full));
    expect(cache.contour(0, "axial", 0)).toHaveLength(0);
    expect(cache.contour(0, "sagittal", 4).length).toBeGreaterThan(0);
    expect(cache.contour(9, "axial", 1)).toHaveLength(0);
  });

  it("evicts least recently used entries", () => {
    const lru = new LruCache<string, number>(2);
    lru.set("a", 1);
    lru.set("b", 2);
    lru.get("a");
    lru.set("c", 3);
    expect(lru.has("a")).toBe(true);
    expect(lru.has("b")).toBe(false);
  });
});

describe("analysis helpers", () => {
  it("computes ranges", () => {
    expect(volumeRange(volume)).toEqual([0, 123]);
    expect(sliceRange(volume, geom, "axial", 1)).toEqual([100, 123]);
    expect(sliceRange(new Float32Array(24), geom, "axial", 0, true)).toEqual([0, 1e-6]);
  });

  it("formats the readout like the Qt viewer", () => {
    const lines = formatPick([1, 2, 3], [-9, 24, 14], -1000, [{ label: "physical_dose", unit: "Gy", value: 1.23456 }]);
    expect(lines).toEqual([
      "Pos: (-9.0, 24.0, 14.0) mm",
      "Index: [1, 2, 3] (sitk) / [3, 2, 1] (numpy)",
      "CT: -1000 HU",
      "physical_dose: 1.235 Gy",
    ]);
    expect(toPrecision4(60)).toBe("60");
    expect(toPrecision4(0.000012345)).toBe("1.234e-05");
    expect(toPrecision4(123456)).toBe("1.235e+05");
  });

  it("chooses nice ticks", () => {
    expect(niceTicks(0, 60, 5)).toEqual([0, 20, 40, 60]);
    expect(niceTicks(-1000, 1000, 5)).toEqual([-1000, -500, 0, 500, 1000]);
  });
});

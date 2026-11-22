/**
 * Per-slice contours of structure masks and isolines of a quantity, computed on the CPU
 * with marching squares and cached per (structure, plane, slice), like the Qt viewer.
 */
import {
  extractSlice,
  linearIndex,
  sliceLayout,
  sliceVoxel,
  voxelToWorld,
  type Plane,
  type PlaneConvention,
  type SliceLayout,
  type VolumeGeometry,
} from "./geometry";
import { marchingSquares } from "./marchingSquares";

/** `[z0, y0, x0, z1, y1, x1]`, ends exclusive (protocol v1 mask bbox). */
export type MaskBBox = readonly [number, number, number, number, number, number];

/**
 * How a structure mask is held: the full uint8 volume (v0), a uint8 crop of its bounding box
 * (v1 `"crops"`), or one bit of a shared bit-plane volume (v1 `"bits"`).
 */
export type MaskStorage =
  | { kind: "full"; data: Uint8Array }
  | { kind: "crop"; data: Uint8Array; bbox: MaskBBox }
  | { kind: "bits"; plane: ArrayLike<number>; bit: number; bbox?: MaskBBox | null };

export interface MaskVolume {
  /** Position of the structure in the cst; names may repeat. */
  index: number;
  name: string;
  color: readonly [number, number, number];
  /** Full uint8 volume (shorthand for `storage: {kind: "full"}`). */
  data?: Uint8Array;
  storage?: MaskStorage;
}

export function maskStorage(mask: MaskVolume): MaskStorage {
  if (mask.storage) return mask.storage;
  if (mask.data) return { kind: "full", data: mask.data };
  throw new Error(`mask ${mask.name} has no data`);
}

/** Whether voxel `[i, j, k]` lies inside the mask. */
export function maskContains(storage: MaskStorage, geom: VolumeGeometry, ijk: readonly [number, number, number]): boolean {
  if (storage.kind === "full") return storage.data[linearIndex(geom, ijk)] !== 0;
  if (storage.kind === "bits") return ((storage.plane[linearIndex(geom, ijk)] >> storage.bit) & 1) === 1;
  const [z0, y0, x0, z1, y1, x1] = storage.bbox;
  const [i, j, k] = ijk;
  if (i < x0 || i >= x1 || j < y0 || j >= y1 || k < z0 || k >= z1) return false;
  return storage.data[((k - z0) * (y1 - y0) + (j - y0)) * (x1 - x0) + (i - x0)] !== 0;
}

/** Slice of a mask in screen layout, limited to `crop` (as `extractSlice`). */
export function extractMaskSlice(
  storage: MaskStorage,
  geom: VolumeGeometry,
  layout: SliceLayout,
  index: number,
  crop: { u0: number; u1: number; v0: number; v1: number },
): { values: Uint8Array; width: number; height: number } {
  if (storage.kind === "full") return extractSlice(storage.data, geom, layout, index, Uint8Array, crop);
  const width = crop.u1 - crop.u0;
  const height = crop.v1 - crop.v0;
  const values = new Uint8Array(width * height);
  if (storage.kind === "bits") {
    const slice = extractSlice(storage.plane, geom, layout, index, Uint32Array, crop);
    for (let n = 0; n < values.length; n++) values[n] = (slice.values[n] >> storage.bit) & 1;
    return { values, width, height };
  }
  for (let r = 0; r < height; r++) {
    for (let c = 0; c < width; c++) {
      values[r * width + c] = maskContains(storage, geom, sliceVoxel(layout, crop.u0 + c, crop.v0 + r, index)) ? 1 : 0;
    }
  }
  return { values, width, height };
}

function bboxToBox(b: MaskBBox): Box3 {
  return { lo: [b[2], b[1], b[0]], hi: [b[5], b[4], b[3]] };
}

interface Box3 {
  /** Inclusive-exclusive voxel ranges per world axis (x, y, z). */
  lo: [number, number, number];
  hi: [number, number, number];
}

/** Bounding box of the non-zero voxels of a `(nz, ny, nx)` mask, or null when empty. */
export function maskBounds(data: Uint8Array, geom: VolumeGeometry): Box3 | null {
  const [nz, ny, nx] = geom.shape;
  const lo: [number, number, number] = [nx, ny, nz];
  const hi: [number, number, number] = [-1, -1, -1];
  let i = 0;
  for (let k = 0; k < nz; k++) {
    for (let j = 0; j < ny; j++) {
      for (let c = 0; c < nx; c++, i++) {
        if (!data[i]) continue;
        if (c < lo[0]) lo[0] = c;
        if (c > hi[0]) hi[0] = c;
        if (j < lo[1]) lo[1] = j;
        if (j > hi[1]) hi[1] = j;
        if (k < lo[2]) lo[2] = k;
        if (k > hi[2]) hi[2] = k;
      }
    }
  }
  if (hi[0] < 0) return null;
  return { lo, hi: [hi[0] + 1, hi[1] + 1, hi[2] + 1] };
}

/** LRU map with a fixed capacity. */
export class LruCache<K, V> {
  private readonly map = new Map<K, V>();
  constructor(private readonly capacity: number) {}

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value !== undefined) {
      this.map.delete(key);
      this.map.set(key, value);
    }
    return value;
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  set(key: K, value: V): void {
    this.map.delete(key);
    this.map.set(key, value);
    if (this.map.size > this.capacity) this.map.delete(this.map.keys().next().value as K);
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}

const EMPTY: Float32Array = new Float32Array(0);

export class ContourCache {
  private masks: MaskVolume[] = [];
  private bounds = new Map<number, Box3 | null>();
  private readonly cache = new LruCache<string, Float32Array>(4096);
  private geom: VolumeGeometry | null = null;
  convention: PlaneConvention = "pyradplan";

  setMasks(masks: MaskVolume[], geom: VolumeGeometry): void {
    this.masks = masks;
    this.geom = geom;
    this.bounds.clear();
    this.cache.clear();
  }

  get indices(): number[] {
    return this.masks.map((m) => m.index);
  }

  mask(index: number): MaskVolume | undefined {
    return this.masks.find((m) => m.index === index);
  }

  private boundsOf(mask: MaskVolume): Box3 | null {
    if (!this.bounds.has(mask.index)) {
      const storage = maskStorage(mask);
      let box: Box3 | null;
      if (storage.kind === "full") box = maskBounds(storage.data, this.geom!);
      else if (storage.bbox) box = bboxToBox(storage.bbox);
      else if (storage.kind === "crop" || storage.bbox === null) box = null;
      else {
        const bits = new Uint8Array(storage.plane.length);
        for (let i = 0; i < bits.length; i++) bits[i] = (storage.plane[i] >> storage.bit) & 1;
        box = maskBounds(bits, this.geom!);
      }
      this.bounds.set(mask.index, box);
    }
    return this.bounds.get(mask.index) ?? null;
  }

  /** Segments `[u0, v0, u1, v1, ...]` of structure `structure` (index) on a slice (slice voxel units). */
  contour(structure: number, plane: Plane, index: number): Float32Array {
    const geom = this.geom;
    const mask = this.mask(structure);
    if (!geom || !mask) return EMPTY;
    const key = `${structure}|${plane}|${this.convention}|${index}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const layout = sliceLayout(geom, plane, this.convention);
    const box = this.boundsOf(mask);
    let result = EMPTY;
    const { axes } = layout;
    if (box && index >= box.lo[axes.slice] && index < box.hi[axes.slice]) {
      // Crop to the structure plus one background voxel, so the edge padding only ever
      // closes curves at the slice border (as in the Qt viewer).
      const toSlice = (a: 0 | 1 | 2, flip: boolean, n: number): [number, number] =>
        flip ? [n - box.hi[a], n - box.lo[a]] : [box.lo[a], box.hi[a]];
      const [hu0, hu1] = toSlice(axes.h, axes.flipH, layout.nh);
      const [hv0, hv1] = toSlice(axes.v, axes.flipV, layout.nv);
      const crop = {
        u0: Math.max(0, hu0 - 1),
        u1: Math.min(layout.nh, hu1 + 1),
        v0: Math.max(0, hv0 - 1),
        v1: Math.min(layout.nv, hv1 + 1),
      };
      const slice = extractMaskSlice(maskStorage(mask), geom, layout, index, crop);
      result = marchingSquares(slice.values, slice.width, slice.height, 0.5, {
        offsetU: crop.u0,
        offsetV: crop.v0,
      });
    }
    this.cache.set(key, result);
    return result;
  }

  clearCache(): void {
    this.cache.clear();
  }
}

/** Isoline cache of one quantity volume. */
export class IsolineCache {
  private readonly cache = new LruCache<string, Float32Array>(512);
  private data: ArrayLike<number> | null = null;
  private geom: VolumeGeometry | null = null;
  convention: PlaneConvention = "pyradplan";

  setVolume(data: ArrayLike<number> | null, geom: VolumeGeometry | null): void {
    this.data = data;
    this.geom = geom;
    this.cache.clear();
  }

  isolines(plane: Plane, index: number, level: number): Float32Array {
    if (!this.data || !this.geom) return EMPTY;
    const key = `${plane}|${this.convention}|${index}|${level}`;
    const hit = this.cache.get(key);
    if (hit) return hit;
    const layout = sliceLayout(this.geom, plane, this.convention);
    const slice = extractSlice(this.data, this.geom, layout, index, Float32Array);
    let min = Infinity;
    let max = -Infinity;
    for (const v of slice.values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    const result = level > max || level < min || max === min ? EMPTY : marchingSquares(slice.values, slice.width, slice.height, level);
    this.cache.set(key, result);
    return result;
  }
}

/** Centre of mass (mm) of a mask, or null when empty. */
export function maskCenterOfMass(mask: MaskVolume, geom: VolumeGeometry): [number, number, number] | null {
  const storage = maskStorage(mask);
  const [nz, ny, nx] = geom.shape;
  let [z0, y0, x0, z1, y1, x1] = [0, 0, 0, nz, ny, nx];
  if (storage.kind !== "full" && storage.bbox) [z0, y0, x0, z1, y1, x1] = storage.bbox;
  if (storage.kind !== "full" && storage.bbox === null) return null;
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let n = 0;
  for (let k = z0; k < z1; k++)
    for (let j = y0; j < y1; j++)
      for (let i = x0; i < x1; i++)
        if (maskContains(storage, geom, [i, j, k])) {
          sx += i;
          sy += j;
          sz += k;
          n++;
        }
  if (!n) return null;
  return voxelToWorld(geom, [sx / n, sy / n, sz / n]);
}

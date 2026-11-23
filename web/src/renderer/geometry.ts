/**
 * Slice geometry: which volume axes a plane shows, slice extraction, and the mapping
 * between canvas pixels and voxels (fit to the canvas at physical aspect, then zoom/pan).
 *
 * Conventions: volumes are C-order arrays of shape `(nz, ny, nx)`; a voxel index is written
 * `[i, j, k]` = (x, y, z) indices; spacing is `(sx, sy, sz)` in mm. A slice has continuous
 * coordinates `(u, v)` in voxel units, `u` to the right and `v` downwards on screen; voxel
 * `(iu, iv)` covers `[iu, iu + 1) x [iv, iv + 1)`.
 */

export type Plane = "axial" | "sagittal" | "coronal";
export const PLANES: readonly Plane[] = ["axial", "sagittal", "coronal"];

/** World axis index: 0 = x, 1 = y, 2 = z. */
export type Axis = 0 | 1 | 2;

export interface PlaneAxes {
  /** Axis the slice index runs along. */
  slice: Axis;
  /** Axis shown horizontally (left to right with increasing index unless `flipH`). */
  h: Axis;
  /** Axis shown vertically (top to bottom with increasing index unless `flipV`). */
  v: Axis;
  flipH: boolean;
  flipV: boolean;
}

/**
 * `pyradplan` mirrors the Qt viewer (and matRad): axial shows x right / y down, sagittal
 * z right / y down, coronal z right / x down. `radiological` shows sagittal and coronal
 * with the patient's head up (z up), y resp. x to the right.
 */
export type PlaneConvention = "pyradplan" | "radiological";

const PLANE_TABLES: Record<PlaneConvention, Record<Plane, PlaneAxes>> = {
  pyradplan: {
    axial: { slice: 2, h: 0, v: 1, flipH: false, flipV: false },
    sagittal: { slice: 0, h: 2, v: 1, flipH: false, flipV: false },
    coronal: { slice: 1, h: 2, v: 0, flipH: false, flipV: false },
  },
  radiological: {
    axial: { slice: 2, h: 0, v: 1, flipH: false, flipV: false },
    sagittal: { slice: 0, h: 1, v: 2, flipH: false, flipV: true },
    coronal: { slice: 1, h: 0, v: 2, flipH: false, flipV: true },
  },
};

export function planeAxes(plane: Plane, convention: PlaneConvention = "pyradplan"): PlaneAxes {
  return PLANE_TABLES[convention][plane];
}

export interface VolumeGeometry {
  /** `(nz, ny, nx)`. */
  shape: readonly [number, number, number];
  /** `(sx, sy, sz)` mm. */
  spacing: readonly [number, number, number];
  /** `(x, y, z)` mm of voxel `[0, 0, 0]`. */
  origin: readonly [number, number, number];
  /** Row-major 3x3 direction cosines (identity when absent). */
  direction?: readonly number[];
}

/** Number of voxels along world axis `a`. */
export function dim(geom: VolumeGeometry, a: Axis): number {
  return geom.shape[2 - a];
}

export interface SliceLayout {
  axes: PlaneAxes;
  /** Voxels horizontally / vertically. */
  nh: number;
  nv: number;
  /** Voxel size horizontally / vertically (mm). */
  sh: number;
  sv: number;
  /** Number of slices along the slice axis. */
  count: number;
}

export function sliceLayout(geom: VolumeGeometry, plane: Plane, convention?: PlaneConvention): SliceLayout {
  const axes = planeAxes(plane, convention);
  return {
    axes,
    nh: dim(geom, axes.h),
    nv: dim(geom, axes.v),
    sh: geom.spacing[axes.h],
    sv: geom.spacing[axes.v],
    count: dim(geom, axes.slice),
  };
}

/** Index `[i, j, k]` of the voxel at slice voxel `(iu, iv)` on slice `index`. */
export function sliceVoxel(layout: SliceLayout, iu: number, iv: number, index: number): [number, number, number] {
  const { axes, nh, nv } = layout;
  const out: [number, number, number] = [0, 0, 0];
  out[axes.h] = axes.flipH ? nh - 1 - iu : iu;
  out[axes.v] = axes.flipV ? nv - 1 - iv : iv;
  out[axes.slice] = index;
  return out;
}

/** Linear offset of voxel `[i, j, k]` in a `(nz, ny, nx)` C-order array. */
export function linearIndex(geom: VolumeGeometry, ijk: readonly [number, number, number]): number {
  const [, ny, nx] = geom.shape;
  return (ijk[2] * ny + ijk[1]) * nx + ijk[0];
}

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1] as const;

/** World position (mm) of the centre of voxel `[i, j, k]` (fractional indices give points between voxels). */
export function voxelToWorld(geom: VolumeGeometry, ijk: readonly [number, number, number]): [number, number, number] {
  const d = geom.direction ?? IDENTITY;
  const s = [ijk[0] * geom.spacing[0], ijk[1] * geom.spacing[1], ijk[2] * geom.spacing[2]];
  return [
    geom.origin[0] + d[0] * s[0] + d[1] * s[1] + d[2] * s[2],
    geom.origin[1] + d[3] * s[0] + d[4] * s[1] + d[5] * s[2],
    geom.origin[2] + d[6] * s[0] + d[7] * s[1] + d[8] * s[2],
  ];
}

/**
 * Fractional voxel index `[i, j, k]` of a world position (mm): the inverse of
 * {@link voxelToWorld}. The direction matrix is orthonormal, so its inverse is its transpose.
 */
export function worldToVoxel(geom: VolumeGeometry, world: readonly number[]): [number, number, number] {
  const d = geom.direction ?? IDENTITY;
  const r = [world[0] - geom.origin[0], world[1] - geom.origin[1], world[2] - geom.origin[2]];
  return [
    (d[0] * r[0] + d[3] * r[1] + d[6] * r[2]) / geom.spacing[0],
    (d[1] * r[0] + d[4] * r[1] + d[7] * r[2]) / geom.spacing[1],
    (d[2] * r[0] + d[5] * r[1] + d[8] * r[2]) / geom.spacing[2],
  ];
}

/** Voxel `[i, j, k]` nearest to a world position, clamped to the volume. */
export function worldToIndex(geom: VolumeGeometry, world: readonly number[]): [number, number, number] {
  const v = worldToVoxel(geom, world);
  const clamp = (x: number, n: number) => Math.min(n - 1, Math.max(0, Math.round(x)));
  return [clamp(v[0], dim(geom, 0)), clamp(v[1], dim(geom, 1)), clamp(v[2], dim(geom, 2))];
}

/** Slice index of every plane through a world position (the slices follow volume index axes, not world axes). */
export function slicesThrough(geom: VolumeGeometry, world: readonly number[]): Record<Plane, number> {
  const ijk = worldToIndex(geom, world);
  const out = {} as Record<Plane, number>;
  for (const plane of PLANES) out[plane] = ijk[planeAxes(plane).slice];
  return out;
}

export interface SliceData<T extends ArrayLike<number> = Float32Array> {
  values: T;
  /** Width = nh, height = nv; `values[iv * width + iu]`. */
  width: number;
  height: number;
}

type NumericArray = ArrayLike<number> & { length: number };
type ArrayCtor<T> = new (n: number) => T;

/**
 * Extract slice `index` of `data` (shape `(nz, ny, nx)`) in screen layout (row = v).
 * `crop` limits the extraction to `[u0, u1) x [v0, v1)` of the slice.
 */
export function extractSlice<T extends NumericArray>(
  data: NumericArray,
  geom: VolumeGeometry,
  layout: SliceLayout,
  index: number,
  ctor: ArrayCtor<T>,
  crop?: { u0: number; u1: number; v0: number; v1: number },
): SliceData<T> {
  const u0 = crop?.u0 ?? 0;
  const v0 = crop?.v0 ?? 0;
  const u1 = crop?.u1 ?? layout.nh;
  const v1 = crop?.v1 ?? layout.nv;
  const width = u1 - u0;
  const height = v1 - v0;
  const values = new ctor(width * height);
  const [, ny, nx] = geom.shape;
  const stride: [number, number, number] = [1, nx, nx * ny];
  const { axes, nh, nv } = layout;
  const base = index * stride[axes.slice];
  const strideU = axes.flipH ? -stride[axes.h] : stride[axes.h];
  const strideV = axes.flipV ? -stride[axes.v] : stride[axes.v];
  const startU = axes.flipH ? (nh - 1 - u0) * stride[axes.h] : u0 * stride[axes.h];
  const startV = axes.flipV ? (nv - 1 - v0) * stride[axes.v] : v0 * stride[axes.v];
  const out = values as unknown as { [i: number]: number };
  for (let r = 0; r < height; r++) {
    let src = base + startV + r * strideV + startU;
    const row = r * width;
    for (let c = 0; c < width; c++, src += strideU) out[row + c] = data[src];
  }
  return { values, width, height };
}

// ---------------------------------------------------------------------------
// View transform
// ---------------------------------------------------------------------------

export interface ViewState {
  /** 1 = slice fitted into the viewport. */
  zoom: number;
  /** Offset of the viewport centre from the slice centre, mm (h, v). */
  panH: number;
  panV: number;
}

export const DEFAULT_VIEW: ViewState = { zoom: 1, panH: 0, panV: 0 };

/** Affine map between canvas pixels (top-left origin, y down) and slice coordinates. */
export interface ViewTransform {
  /** Canvas pixels per slice voxel, horizontally / vertically. */
  scaleU: number;
  scaleV: number;
  /** Canvas position of slice coordinate (0, 0). */
  offsetX: number;
  offsetY: number;
}

/** Fit the slice (at physical aspect) into a `width x height` canvas, then zoom and pan. */
export function viewTransform(layout: SliceLayout, width: number, height: number, view: ViewState): ViewTransform {
  const extentH = layout.nh * layout.sh;
  const extentV = layout.nv * layout.sv;
  const ppm = view.zoom * Math.min(width / extentH, height / extentV);
  const centerH = extentH / 2 + view.panH;
  const centerV = extentV / 2 + view.panV;
  return {
    scaleU: ppm * layout.sh,
    scaleV: ppm * layout.sv,
    offsetX: width / 2 - centerH * ppm,
    offsetY: height / 2 - centerV * ppm,
  };
}

export function sliceToCanvas(t: ViewTransform, u: number, v: number): [number, number] {
  return [t.offsetX + u * t.scaleU, t.offsetY + v * t.scaleV];
}

export function canvasToSlice(t: ViewTransform, x: number, y: number): [number, number] {
  return [(x - t.offsetX) / t.scaleU, (y - t.offsetY) / t.scaleV];
}

/**
 * Column-major 3x3 matrix taking `gl_FragCoord.xy` (bottom-left origin) of a canvas of
 * height `height` to slice coordinates `(u, v)`.
 */
export function fragToSliceMatrix(t: ViewTransform, height: number): Float32Array {
  // u = (x - ox) / su ; v = ((height - fy) - oy) / sv
  const a = 1 / t.scaleU;
  const d = -1 / t.scaleV;
  return new Float32Array([a, 0, 0, 0, d, 0, -t.offsetX / t.scaleU, (height - t.offsetY) / t.scaleV, 1]);
}

/**
 * Zoom by `factor` keeping the slice point under canvas position `(x, y)` fixed.
 */
export function zoomAt(
  layout: SliceLayout,
  width: number,
  height: number,
  view: ViewState,
  factor: number,
  x: number,
  y: number,
): ViewState {
  const zoom = Math.min(64, Math.max(0.1, view.zoom * factor));
  const before = canvasToSlice(viewTransform(layout, width, height, view), x, y);
  const next = { ...view, zoom };
  const after = canvasToSlice(viewTransform(layout, width, height, next), x, y);
  return {
    zoom,
    panH: view.panH + (before[0] - after[0]) * layout.sh,
    panV: view.panV + (before[1] - after[1]) * layout.sv,
  };
}

/** Pan by a canvas-pixel delta. */
export function panBy(layout: SliceLayout, width: number, height: number, view: ViewState, dx: number, dy: number): ViewState {
  const t = viewTransform(layout, width, height, view);
  return { ...view, panH: view.panH - (dx / t.scaleU) * layout.sh, panV: view.panV - (dy / t.scaleV) * layout.sv };
}

/** Texel lookup for the shader: texel = origin + iu * dirH + iv * dirV (texel = (x, y, z)). */
export function texelMapping(layout: SliceLayout, index: number): { origin: Int32Array; dirH: Int32Array; dirV: Int32Array } {
  const origin = new Int32Array(sliceVoxel(layout, 0, 0, index));
  const dirH = new Int32Array(3);
  const dirV = new Int32Array(3);
  dirH[layout.axes.h] = layout.axes.flipH ? -1 : 1;
  dirV[layout.axes.v] = layout.axes.flipV ? -1 : 1;
  return { origin, dirH, dirV };
}

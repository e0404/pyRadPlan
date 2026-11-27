/**
 * Marching squares on a 2-D slice, matching the Qt viewer's `_isocurve_path`
 * (`contourpy` on the edge-padded slice): samples sit at pixel centres `(iu + 0.5, iv + 0.5)`,
 * the slice is padded by one edge-replicated pixel so curves run up to the slice border,
 * and coordinates are clipped to `[0, width] x [0, height]`.
 *
 * Output is a flat list of segments `[u0, v0, u1, v1, ...]` in slice voxel coordinates.
 */

export interface ContourOptions {
  /** Added to every output coordinate (position of `values[0]` in a larger slice). */
  offsetU?: number;
  offsetV?: number;
}

// For each case (bit 0 = top-left, 1 = top-right, 2 = bottom-right, 3 = bottom-left set
// "above"), the cell edges a segment connects. Edges: 0 = top, 1 = right, 2 = bottom, 3 = left.
const SEGMENTS: readonly (readonly number[])[] = [
  [],
  [3, 0],
  [0, 1],
  [3, 1],
  [1, 2],
  [3, 0, 1, 2], // saddle, resolved below
  [0, 2],
  [3, 2],
  [2, 3],
  [0, 2],
  [0, 1, 2, 3], // saddle, resolved below
  [1, 2],
  [3, 1],
  [0, 1],
  [3, 0],
  [],
];
const SADDLE_CUT_TR_BL = [0, 1, 3, 2];
const SADDLE_CUT_TL_BR = [3, 0, 1, 2];

export function marchingSquares(
  values: ArrayLike<number>,
  width: number,
  height: number,
  level: number,
  options: ContourOptions = {},
): Float32Array {
  const offU = options.offsetU ?? 0;
  const offV = options.offsetV ?? 0;
  if (width <= 0 || height <= 0) return new Float32Array(0);
  // Edge-padded sample at padded index (pu, pv) in [0, width + 1] x [0, height + 1].
  const sample = (pu: number, pv: number): number => {
    const u = Math.min(width - 1, Math.max(0, pu - 1));
    const v = Math.min(height - 1, Math.max(0, pv - 1));
    return values[v * width + u];
  };
  const out: number[] = [];
  const pw = width + 2;
  const ph = height + 2;
  for (let pv = 0; pv < ph - 1; pv++) {
    for (let pu = 0; pu < pw - 1; pu++) {
      const tl = sample(pu, pv);
      const tr = sample(pu + 1, pv);
      const br = sample(pu + 1, pv + 1);
      const bl = sample(pu, pv + 1);
      const c = (tl > level ? 1 : 0) | (tr > level ? 2 : 0) | (br > level ? 4 : 0) | (bl > level ? 8 : 0);
      if (c === 0 || c === 15) continue;
      let edges = SEGMENTS[c];
      if (c === 5 || c === 10) {
        // The cell centre decides which diagonal pair is connected; the other two
        // corners are cut off by one segment each.
        const centerAbove = (tl + tr + br + bl) / 4 > level;
        if (c === 5) edges = centerAbove ? SADDLE_CUT_TR_BL : SADDLE_CUT_TL_BR;
        else edges = centerAbove ? SADDLE_CUT_TL_BR : SADDLE_CUT_TR_BL;
      }
      for (let s = 0; s < edges.length; s++) {
        {
          const e = edges[s];
          let u: number;
          let v: number;
          switch (e) {
            case 0: // top: tl -> tr
              u = pu + frac(tl, tr, level);
              v = pv;
              break;
            case 1: // right: tr -> br
              u = pu + 1;
              v = pv + frac(tr, br, level);
              break;
            case 2: // bottom: bl -> br
              u = pu + frac(bl, br, level);
              v = pv + 1;
              break;
            default: // left: tl -> bl
              u = pu;
              v = pv + frac(tl, bl, level);
          }
          // Padded sample index p sits at slice coordinate p - 0.5.
          out.push(clamp(u - 0.5, 0, width) + offU, clamp(v - 0.5, 0, height) + offV);
        }
      }
    }
  }
  return new Float32Array(out);
}

function frac(a: number, b: number, level: number): number {
  const d = b - a;
  return d === 0 ? 0.5 : (level - a) / d;
}

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

/** Bounding box `[u0, u1) x [v0, v1)` of non-zero values, or null when empty. */
export function boundingBox(
  values: ArrayLike<number>,
  width: number,
  height: number,
): { u0: number; u1: number; v0: number; v1: number } | null {
  let u0 = width;
  let u1 = -1;
  let v0 = height;
  let v1 = -1;
  for (let v = 0; v < height; v++) {
    const row = v * width;
    for (let u = 0; u < width; u++) {
      if (values[row + u]) {
        if (u < u0) u0 = u;
        if (u > u1) u1 = u;
        if (v < v0) v0 = v;
        if (v > v1) v1 = v;
      }
    }
  }
  return u1 < 0 ? null : { u0, u1: u1 + 1, v0, v1: v1 + 1 };
}

/** Total polyline length of a segment list (for tests / diagnostics). */
export function segmentsLength(segments: Float32Array): number {
  let len = 0;
  for (let i = 0; i < segments.length; i += 4) {
    len += Math.hypot(segments[i + 2] - segments[i], segments[i + 3] - segments[i + 1]);
  }
  return len;
}

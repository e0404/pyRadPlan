/**
 * Mock of `view.sequence` (PROTOCOL.md amendment 28): a delivery sequence derived from the
 * phantom's target and the plan's beams. Per beam the target is projected (parallel) into the
 * beam's eye view, the rays cover it with one bixel margin, a smooth stand-in for the optimized
 * fluence is split into `levels` shapes, and the delivered fluence is what those shapes add up to.
 */
import { ndarray } from "../codec";
import type { PlanSummary, Vec3 } from "../protocol";
import type { BevWindow, SequenceBeam, SequenceShape, SequenceTarget, SequenceView } from "../protocol/viewer";
import type { Phantom } from "./phantom";

export interface MockSequence {
  sequencer: string | null;
  levels: number;
}

/** Relative weights of consecutive shapes (repeated), so that shapes differ in weight. */
const WEIGHT_PATTERN = [2, 1, 1.5];

/** `(travel, stack)` of the target voxels in the eye view of a beam at `gantry` degrees. */
function targetPoints(phantom: Phantom, gantry: number, iso: Vec3): Float32Array {
  const target = phantom.structures.find((st) => st.type === "TARGET");
  if (!target) return new Float32Array(0);
  const [nz, ny, nx] = phantom.grid.shape;
  const [sx, sy, sz] = phantom.grid.spacing;
  const [ox, oy, oz] = phantom.grid.origin;
  const cos = Math.cos((gantry * Math.PI) / 180);
  const sin = Math.sin((gantry * Math.PI) / 180);
  const points: number[] = [];
  let i = 0;
  for (let k = 0; k < nz; k++)
    for (let j = 0; j < ny; j++)
      for (let c = 0; c < nx; c++, i++)
        if (target.mask[i]) points.push((ox + c * sx - iso[0]) * cos + (oy + j * sy - iso[1]) * sin, oz + k * sz - iso[2]);
  return Float32Array.from(points);
}

/** The bins the projected target covers, with one empty bin around (`target_outline_mask`). */
function targetMask(points: Float32Array, resolution: number): SequenceTarget | null {
  if (!points.length) return null;
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (let i = 0; i < points.length; i += 2) {
    x0 = Math.min(x0, points[i]);
    x1 = Math.max(x1, points[i]);
    y0 = Math.min(y0, points[i + 1]);
    y1 = Math.max(y1, points[i + 1]);
  }
  const cols = Math.floor((x1 - x0) / resolution) + 3;
  const rows = Math.floor((y1 - y0) / resolution) + 3;
  const mask = new Uint8Array(rows * cols);
  for (let i = 0; i < points.length; i += 2) {
    mask[(Math.floor((points[i + 1] - y0) / resolution) + 1) * cols + Math.floor((points[i] - x0) / resolution) + 1] = 1;
  }
  return { origin: [x0 - resolution / 2, y0 - resolution / 2], spacing: [resolution, resolution], mask: ndarray(mask, [rows, cols]) };
}

function mockBeam(phantom: Phantom, index: number, gantry: number, iso: Vec3, bixelWidth: number, levels: number): SequenceBeam {
  const label = `Beam ${index} (${gantry.toFixed(0)}°)`;
  const points = targetPoints(phantom, gantry, iso);
  const empty: SequenceBeam = { index, label, orientation: null, window: null, weights: [], shapes: [], fluence: null, target: null };
  if (!points.length) return empty;

  // Rays: the bixels the target projects onto, and their neighbours.
  const cells = new Set<string>();
  let c0 = Infinity;
  let c1 = -Infinity;
  let r0 = Infinity;
  let r1 = -Infinity;
  for (let i = 0; i < points.length; i += 2) {
    const c = Math.round(points[i] / bixelWidth);
    const r = Math.round(points[i + 1] / bixelWidth);
    for (const [dc, dr] of [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]]) {
      cells.add(`${c + dc},${r + dr}`);
      c0 = Math.min(c0, c + dc);
      c1 = Math.max(c1, c + dc);
      r0 = Math.min(r0, r + dr);
      r1 = Math.max(r1, r + dr);
    }
  }
  const cols = c1 - c0 + 1;
  const rows = r1 - r0 + 1;
  const window: BevWindow = [(c0 - 0.5) * bixelWidth, (c1 + 0.5) * bixelWidth, (r0 - 0.5) * bixelWidth, (r1 + 0.5) * bixelWidth];

  // Stand-in for the optimized fluence: one bump per leaf row, off-centre by beam, rising with the stack position.
  const optimized = new Float32Array(rows * cols).fill(NaN);
  const centre = 15 * Math.sin(1.3 * index);
  let max = 0;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) {
      if (!cells.has(`${c + c0},${r + r0}`)) continue;
      const travel = (c + c0) * bixelWidth;
      const stack = (r + r0) * bixelWidth;
      const value = (0.35 + 0.65 * Math.exp(-((travel - centre) ** 2) / (2 * 30 ** 2))) * (0.75 + 0.25 * Math.sin(stack / 40));
      optimized[r * cols + c] = value;
      max = Math.max(max, value);
    }

  // Shape l opens every leaf row over the bixels at or above its fluence level.
  const pattern = Array.from({ length: levels }, (_, l) => WEIGHT_PATTERN[l % WEIGHT_PATTERN.length]);
  const total = pattern.reduce((a, b) => a + b, 0);
  const weights = pattern.map((p) => (max * p) / total);
  const delivered = optimized.map((v) => (Number.isNaN(v) ? NaN : 0));
  const boundaries = Array.from({ length: rows }, (_, r) => window[2] + r * bixelWidth);
  const closed = (window[0] + window[1]) / 2;
  const shapes: SequenceShape[] = [];
  let level = 0;
  for (const weight of weights) {
    const threshold = level + weight / 2;
    level += weight;
    const leafPositions: [number, number][] = [];
    for (let r = 0; r < rows; r++) {
      let first = -1;
      let last = -1;
      for (let c = 0; c < cols; c++) {
        if (optimized[r * cols + c] >= threshold) {
          if (first < 0) first = c;
          last = c;
        }
      }
      if (first < 0) {
        leafPositions.push([closed, closed]);
        continue;
      }
      leafPositions.push([window[0] + first * bixelWidth, window[0] + (last + 1) * bixelWidth]);
      for (let c = first; c <= last; c++) delivered[r * cols + c] += weight;
    }
    shapes.push({ leaf_width: bixelWidth, boundaries, leaf_positions: leafPositions });
  }

  return {
    index,
    label,
    orientation: "X",
    window,
    weights,
    shapes,
    fluence: { window, bixel_width: bixelWidth, optimized: ndarray(optimized, [rows, cols]), delivered: ndarray(delivered, [rows, cols]) },
    target: targetMask(points, Math.max(...phantom.grid.spacing)),
  };
}

export function mockSequenceView(phantom: Phantom, plan: PlanSummary, iso: Vec3, sequence: MockSequence): SequenceView {
  return {
    sequencer: sequence.sequencer,
    num_of_levels: sequence.levels,
    beams: plan.gantry_angles.map((gantry, index) => mockBeam(phantom, index, gantry, iso, plan.bixel_width, sequence.levels)),
  };
}

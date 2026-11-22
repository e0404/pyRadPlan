/**
 * Patient orientation marker (mannequin), ported from `result/_orientation_marker.py`: capsules
 * in LPS coordinates projected onto the screen axes with mild perspective, cut at the slice,
 * nearer parts lighter, edge letters for the patient directions.
 */
import type { SliceLayout, VolumeGeometry } from "./geometry";

type P3 = readonly [number, number, number];

const FIGURE: readonly [P3, P3, number][] = [
  [[0, 0, 75], [0, 0, 75], 11],
  [[0, -12, 74], [0, -12, 74], 4],
  [[0, 0, 56], [0, 0, 66], 5],
  [[-7, 0, 12], [-7, 0, 46], 10],
  [[7, 0, 12], [7, 0, 46], 10],
  [[-20, 0, 49], [-24, 0, 6], 4.5],
  [[20, 0, 49], [24, 0, 6], 4.5],
  [[-8, 0, 2], [-9, 0, -78], 6],
  [[8, 0, 2], [9, 0, -78], 6],
  [[-9, 0, -84], [-9, -14, -86], 4],
  [[9, 0, -84], [9, -14, -86], 4],
];

const LETTERS: [string, string][] = [
  ["R", "L"],
  ["A", "P"],
  ["I", "S"],
];
const CAMERA_DISTANCE = 280;
const CUT_DEPTH = 12;
const SAME_DEPTH = 1.5;
const FIGURE_PX = 104;
const LETTER_MARGIN = 13;
const NEAR_SHADE = 240;
const FAR_SHADE = 120;
export const ORIENTATION_MARGIN = 8;

const dot = (a: readonly number[], b: readonly number[]) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

export function directionLetter(d: readonly number[]): string {
  let axis = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(d[i]) > Math.abs(d[axis])) axis = i;
  return LETTERS[axis][d[axis] > 0 ? 1 : 0];
}

/** Patient (LPS) directions of the screen's right and down axes for a slice layout. */
export function screenAxesLps(geom: VolumeGeometry, layout: SliceLayout): { right: number[]; down: number[] } {
  const d = geom.direction ? [...geom.direction] : [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const column = (a: number) => [d[a], d[3 + a], d[6 + a]];
  const { axes } = layout;
  const right = column(axes.h).map((x) => (axes.flipH ? -x : x));
  const down = column(axes.v).map((x) => (axes.flipV ? -x : x));
  return { right, down };
}

export interface MarkerPart {
  depth: number;
  a: [number, number];
  b: [number, number];
  r: number;
}

export interface MarkerLayout {
  width: number;
  height: number;
  parts: MarkerPart[];
  letters: { left: string; right: string; top: string; bottom: string };
}

export function projectMannequin(right: readonly number[], down: readonly number[]): MarkerLayout {
  const into = [right[1] * down[2] - right[2] * down[1], right[2] * down[0] - right[0] * down[2], right[0] * down[1] - right[1] * down[0]];
  const parts: MarkerPart[] = [];
  for (const [s, e, radius] of FIGURE) {
    const depths = [dot(s, into), dot(e, into)];
    const mean = (depths[0] + depths[1]) / 2;
    if (mean < -CUT_DEPTH) continue;
    const f = depths.map((z) => CAMERA_DISTANCE / (CAMERA_DISTANCE + z));
    parts.push({
      depth: mean,
      a: [dot(s, right) * f[0], dot(s, down) * f[0]],
      b: [dot(e, right) * f[1], dot(e, down) * f[1]],
      r: radius * ((f[0] + f[1]) / 2),
    });
  }
  parts.sort((p, q) => q.depth - p.depth);
  let lo = [Infinity, Infinity];
  let hi = [-Infinity, -Infinity];
  for (const p of parts)
    for (const pt of [p.a, p.b])
      for (const sgn of [-1, 1]) {
        lo = [Math.min(lo[0], pt[0] + sgn * p.r), Math.min(lo[1], pt[1] + sgn * p.r)];
        hi = [Math.max(hi[0], pt[0] + sgn * p.r), Math.max(hi[1], pt[1] + sgn * p.r)];
      }
  const scale = FIGURE_PX / Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const width = Math.ceil(scale * (hi[0] - lo[0])) + 2 * LETTER_MARGIN;
  const height = Math.ceil(scale * (hi[1] - lo[1])) + 2 * LETTER_MARGIN;
  const off = [width / 2 - (scale * (lo[0] + hi[0])) / 2, height / 2 - (scale * (lo[1] + hi[1])) / 2];
  const tx = (p: [number, number]): [number, number] => [off[0] + scale * p[0], off[1] + scale * p[1]];
  return {
    width,
    height,
    parts: parts.map((p) => ({ depth: p.depth, a: tx(p.a), b: tx(p.b), r: scale * p.r })),
    letters: {
      right: directionLetter(right),
      left: directionLetter(right.map((x) => -x)),
      bottom: directionLetter(down),
      top: directionLetter(down.map((x) => -x)),
    },
  };
}

function capsule(ctx: CanvasRenderingContext2D, a: [number, number], b: [number, number], r: number, color: string) {
  if (a[0] === b[0] && a[1] === b[1]) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(a[0], a[1], r, 0, 2 * Math.PI);
    ctx.fill();
    return;
  }
  ctx.strokeStyle = color;
  ctx.lineWidth = 2 * r;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(b[0], b[1]);
  ctx.stroke();
}

/** Draw the marker with its top-left corner at `(x, y)` (CSS pixels). */
export function drawMannequin(ctx: CanvasRenderingContext2D, marker: MarkerLayout, x: number, y: number): void {
  ctx.save();
  ctx.translate(x, y);
  ctx.fillStyle = "rgba(0, 0, 0, 0.59)";
  ctx.beginPath();
  ctx.roundRect(0, 0, marker.width, marker.height, 4);
  ctx.fill();
  const depths = marker.parts.map((p) => p.depth);
  const maxD = Math.max(...depths);
  const span = Math.max(maxD - Math.min(...depths), 1e-9);
  const groups: MarkerPart[][] = [];
  for (const part of marker.parts) {
    const last = groups[groups.length - 1];
    if (last && last[0].depth - part.depth < SAME_DEPTH) last.push(part);
    else groups.push([part]);
  }
  for (const group of groups) {
    const weight = (maxD - group[0].depth) / span;
    const shade = Math.round(FAR_SHADE + weight * (NEAR_SHADE - FAR_SHADE));
    for (const p of group) capsule(ctx, p.a, p.b, p.r + 0.75, "rgb(30, 30, 30)");
    for (const p of group) capsule(ctx, p.a, p.b, p.r, `rgb(${shade}, ${shade}, ${shade})`);
  }
  ctx.fillStyle = "#fff";
  ctx.font = "bold 10px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const m = LETTER_MARGIN;
  ctx.fillText(marker.letters.left, m / 2, marker.height / 2);
  ctx.fillText(marker.letters.right, marker.width - m / 2, marker.height / 2);
  ctx.fillText(marker.letters.top, marker.width / 2, m / 2);
  ctx.fillText(marker.letters.bottom, marker.width / 2, marker.height - m / 2);
  ctx.restore();
}

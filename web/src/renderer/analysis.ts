/** Small numeric helpers around the viewer (ranges, isoline defaults, readout text). */
import { extractSlice, sliceLayout, type Plane, type PlaneConvention, type VolumeGeometry } from "./geometry";

export function volumeRange(data: ArrayLike<number>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return Number.isFinite(min) ? [min, max] : [0, 0];
}

/**
 * Min / max of one slice; `ignoreZero` skips zeros (the Qt viewer hides zero-valued
 * quantity voxels, so they do not count for the local range either).
 */
export function sliceRange(
  data: ArrayLike<number>,
  geom: VolumeGeometry,
  plane: Plane,
  index: number,
  ignoreZero = false,
  convention?: PlaneConvention,
): [number, number] {
  const layout = sliceLayout(geom, plane, convention);
  const slice = extractSlice(data, geom, layout, index, Float32Array);
  let min = Infinity;
  let max = -Infinity;
  for (const v of slice.values) {
    if (ignoreZero && v === 0) continue;
    if (Number.isNaN(v)) continue;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  return Number.isFinite(min) ? [min, max] : [0, 1e-6];
}

/** Qt default isolines: 50 / 80 / 95 % of the maximum. */
export function defaultIsolineLevels(max: number): number[] {
  return max > 0 ? [0.5 * max, 0.8 * max, 0.95 * max] : [];
}

export interface ReadoutQuantity {
  label: string;
  unit: string;
  value: number | null;
}

/** Cursor readout lines, as in the Qt viewer's info label. */
export function formatPick(
  index: readonly [number, number, number],
  position: readonly [number, number, number],
  ct: number | null,
  quantities: ReadoutQuantity[],
): string[] {
  const [i, j, k] = index;
  const lines = [
    `Pos: (${position[0].toFixed(1)}, ${position[1].toFixed(1)}, ${position[2].toFixed(1)}) mm`,
    `Index: [${i}, ${j}, ${k}] (sitk) / [${k}, ${j}, ${i}] (numpy)`,
  ];
  if (ct !== null) lines.push(`CT: ${ct.toFixed(0)} HU`);
  for (const q of quantities) {
    if (q.value === null) continue;
    const value = `${q.label}: ${toPrecision4(q.value)}`;
    lines.push(q.unit ? `${value} ${q.unit}` : value);
  }
  return lines;
}

/** Python's `{:.4g}`. */
export function toPrecision4(v: number): string {
  if (v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  if (exp < -4 || exp >= 4) {
    const [m, e] = v.toExponential(3).split("e");
    const mant = m.includes(".") ? m.replace(/0+$/, "").replace(/\.$/, "") : m;
    const n = Number(e);
    return `${mant}e${n < 0 ? "-" : "+"}${String(Math.abs(n)).padStart(2, "0")}`;
  }
  const s = v.toPrecision(4);
  return s.includes(".") ? s.replace(/0+$/, "").replace(/\.$/, "") : s;
}

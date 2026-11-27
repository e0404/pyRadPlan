/**
 * Pure helpers of the analysis window: DVH lookups (`pyRadPlan/analysis/_dvh.py`), plot
 * description (`_dvh_plot.py:build_dvh_plot`: curves, legend, D_x / V_x markers), QI table
 * text (`gui/widgets/analysis/_qi.py`).
 */
import type { QiResult } from "../../api/protocol/viewer";

export type LineStyle = "solid" | "dashed";

export interface PlotCurve {
  /** Position of the structure. */
  index: number;
  name: string;
  x: ArrayLike<number>;
  y: ArrayLike<number>;
  color: readonly number[];
  style: LineStyle;
  secondaryAxis: boolean;
}

export interface QiMarker {
  /** Position of the structure. */
  structure: number;
  name: string;
  metric: string;
  kind: "dx" | "vx";
  /** D_x: reference volume (%); V_x: reference dose. */
  parameter: number;
  /** D_x: dose; V_x: volume (%). */
  value: number;
  color: readonly number[];
  valueUnit: string;
  parameterUnit: string;
}

export const markerX = (m: QiMarker) => (m.kind === "dx" ? m.value : m.parameter);
export const markerY = (m: QiMarker) => (m.kind === "dx" ? m.parameter : m.value);

/** `format_marker_value`: "1.52 Gy" / "93.2 %". */
export function formatMarkerValue(value: number, unit: string): string {
  const text = unit === "%" ? value.toFixed(1) : value.toFixed(2);
  return unit ? `${text} ${unit}` : text;
}

/** `format_marker_parameter`: "D98" / "V20Gy". */
export function formatMarkerParameter(kind: "dx" | "vx", parameter: number, unit: string): string {
  if (kind === "dx") return `D${+parameter.toFixed(1)}`;
  return `V${+parameter.toFixed(2)}${unit}`;
}

/** `dose_at_volume`: largest bin still covering at least `y` % (no interpolation over plateaus). */
export function doseAtVolume(bins: ArrayLike<number>, cumVolume: ArrayLike<number>, y: number): number {
  const n = cumVolume.length;
  if (!n) return 0;
  // Search over the reversed (increasing) volumes: first index with value >= y.
  let lo = 0;
  let hi = n;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (cumVolume[n - 1 - mid] < y) lo = mid + 1;
    else hi = mid;
  }
  const index = Math.min(n - 1, Math.max(0, n - 1 - lo));
  return bins[index];
}

/** `np.interp` for increasing `xs`. */
export function interp(x: number, xs: ArrayLike<number>, ys: ArrayLike<number>): number {
  const n = xs.length;
  if (!n) return NaN;
  if (x <= xs[0]) return ys[0];
  if (x >= xs[n - 1]) return ys[n - 1];
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= x) lo = mid;
    else hi = mid;
  }
  const t = (x - xs[lo]) / (xs[hi] - xs[lo] || 1);
  return ys[lo] + t * (ys[hi] - ys[lo]);
}

/** `_qi_markers`: D_x / V_x points of the plotted structures (finite values only). */
export function qiMarkers(qi: QiResult | null, structures: readonly number[], colors: Record<number, readonly number[]>): QiMarker[] {
  if (!qi) return [];
  const markers: QiMarker[] = [];
  const shown = new Set(structures);
  for (const row of qi.rows) {
    if (!shown.has(row.index)) continue;
    for (const metric of qi.metrics) {
      if ((metric.kind !== "dx" && metric.kind !== "vx") || metric.parameter === undefined) continue;
      const value = row.values[metric.id];
      if (value === null || value === undefined || !Number.isFinite(value)) continue;
      markers.push({
        structure: row.index,
        name: row.name,
        metric: metric.id,
        kind: metric.kind,
        parameter: metric.parameter,
        value,
        color: colors[row.index] ?? row.color,
        valueUnit: metric.kind === "dx" ? qi.unit_symbol : "%",
        parameterUnit: metric.kind === "dx" ? "%" : qi.unit_symbol,
      });
    }
  }
  return markers;
}

/** `_axis_label`: "Dose [Gy]" (name defaults to "Dose"). */
export function axisLabel(label: string, unit: string): string {
  const name = label || "Dose";
  return unit ? `${name} [${unit}]` : name;
}

/** Table rows (header first) like `QITableWidget.to_rows`: "%.2f", "-" for missing. */
export function qiTableRows(qi: QiResult | null, structures: readonly number[], metrics: readonly string[]): string[][] {
  if (!qi) return [];
  const rows = qi.rows.filter((r) => structures.includes(r.index));
  const shown = qi.metrics.filter((m) => metrics.includes(m.id));
  if (!rows.length || !shown.length) return [];
  return [
    ["Structure", ...shown.map((m) => m.label)],
    ...rows.map((r) => [
      r.name,
      ...shown.map((m) => {
        const v = r.values[m.id];
        return v === null || v === undefined || Number.isNaN(v) ? "-" : v.toFixed(2);
      }),
    ]),
  ];
}

export function toTsv(rows: string[][]): string {
  return rows.map((r) => r.join("\t")).join("\n");
}

export function toCsv(rows: string[][]): string {
  const cell = (c: string) => (/[",\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c);
  return rows.map((r) => r.map(cell).join(",")).join("\r\n") + "\r\n";
}

/** CSV of the plotted curves (pyqtgraph-like export: one x/y column pair per curve). */
export function curvesCsv(curves: readonly PlotCurve[]): string {
  const header = curves.flatMap((c) => [`${c.name}${c.style === "dashed" ? " (compare)" : ""}_x`, `${c.name}${c.style === "dashed" ? " (compare)" : ""}_y`]);
  const n = Math.max(0, ...curves.map((c) => c.x.length));
  const rows: string[][] = [header];
  for (let i = 0; i < n; i++) rows.push(curves.flatMap((c) => (i < c.x.length ? [String(c.x[i]), String(c.y[i])] : ["", ""])));
  return toCsv(rows);
}

/** Parse the D_x input: numbers within [0, 100], else `null` (defaults, Qt `_qi_params`). */
export function parseRefVols(text: string): number[] | null {
  const values = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!values.length || values.some((v) => !Number.isFinite(v) || v < 0 || v > 100)) return null;
  return values;
}

/** Parse the V_x input: finite non-negative numbers, else `null`. */
export function parseRefDoses(text: string): number[] | null {
  const values = text.split(/[\s,;]+/).filter(Boolean).map(Number);
  if (!values.length || values.some((v) => !Number.isFinite(v) || v < 0)) return null;
  return values;
}

/** `_apply_qi_point`: put `parameter` into a list, replacing `replace` if given, no duplicates. */
export function applyPoint(values: readonly number[], parameter: number, replace?: number): number[] {
  const close = (a: number, b: number) => Math.abs(a - b) <= 1e-8 + 1e-5 * Math.abs(b);
  let out = [...values];
  if (replace !== undefined) out = out.map((v) => (close(v, replace) ? parameter : v));
  if (!out.some((v) => close(v, parameter))) out.push(parameter);
  return out;
}

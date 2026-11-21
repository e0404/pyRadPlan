/**
 * Pure helpers of the sequence window: the geometry Qt draws with
 * `pyRadPlan/visualization/_plot_aperture_info.py` and the playback rules of
 * `gui/widgets/result/aperture_widget.py`. All positions are in mm in the beam's eye view
 * (horizontal: leaf travel, vertical: leaf stacking), the vertical axis increasing upwards.
 */
import type { BevWindow, SequenceBeam, SequenceFluence, SequenceShape, SequenceTarget, SequenceView } from "../../api/protocol/viewer";
import { mapColor } from "../../renderer/colormaps";
import { marchingSquares } from "../../renderer/marchingSquares";

export type LeafCoordinate = "leaf_num" | "physical";
export type FluenceMode = "delivered" | "optimized" | "difference";
export type Rgb = [number, number, number];

export const LEAF_COORDINATES: readonly (readonly [string, LeafCoordinate])[] = [
  ["Leaf pair #", "leaf_num"],
  ["Physical [mm]", "physical"],
];

export const FLUENCE_MODES: readonly (readonly [string, FluenceMode])[] = [
  ["Delivered", "delivered"],
  ["Optimized", "optimized"],
  ["Difference", "difference"],
];

export const DEFAULT_TARGET_COLOR = "#ff0000";
export const BLOCKED_COLOR: Rgb = [105, 105, 105];
export const SPEED_RANGE = { min: 0.1, max: 10, step: 0.1 } as const;

const BASE_INTERVAL_MS = 900;
const MIN_INTERVAL_MS = 150;
const LIGHT: Rgb = [0.92, 0.92, 0.92];
const BLUE: Rgb = [0.15, 0.35, 0.75];

export interface Rect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** `_leaf_row_bounds`: lower and upper vertical edge of every leaf row. */
export function leafRowBounds(shape: SequenceShape, coordinate: LeafCoordinate): { low: number[]; high: number[] } {
  if (coordinate === "physical") {
    return { low: [...shape.boundaries], high: shape.boundaries.map((b) => b + shape.leaf_width) };
  }
  const rows = shape.leaf_positions.map((_, i) => i + 1);
  return { low: rows.map((r) => r - 0.5), high: rows.map((r) => r + 0.5) };
}

/** `_leaf_num_points`: a vertical position in mm on the leaf-pair axis. */
export function leafNumber(y: number, shape: SequenceShape): number {
  if (!shape.boundaries.length || !shape.leaf_width) return y;
  return 0.5 + (y - shape.boundaries[0]) / shape.leaf_width;
}

/** `_weight_color`: open-aperture colour, light grey (low weight) to blue (the beam's maximum). */
export function weightColor(weight: number, maxWeight: number): Rgb {
  const frac = maxWeight > 0 ? Math.min(Math.max(weight / maxWeight, 0), 1) : 0;
  return LIGHT.map((c, i) => Math.round(255 * (c + frac * (BLUE[i] - c)))) as Rgb;
}

/** `_draw_aperture`: the blocked region left and right of the opening of every leaf row. */
export function apertureRects(shape: SequenceShape, window: BevWindow, coordinate: LeafCoordinate): Rect[] {
  const { low, high } = leafRowBounds(shape, coordinate);
  return shape.leaf_positions.flatMap(([left, right], i) => [
    { x: window[0], y: low[i], width: left - window[0], height: high[i] - low[i] },
    { x: right, y: low[i], width: window[1] - right, height: high[i] - low[i] },
  ]);
}

/** `_draw_mlc_borders`: leaf outlines for the fluence map; fully retracted leaves are left out. */
export function mlcBorderRects(shape: SequenceShape, window: BevWindow, coordinate: LeafCoordinate = "physical"): Rect[] {
  return apertureRects(shape, window, coordinate).filter((r) => r.width > 0);
}

/** Axis limits `[x0, x1, y0, y1]` of the aperture panel. */
export function apertureLimits(shape: SequenceShape, window: BevWindow, coordinate: LeafCoordinate): BevWindow {
  const { low, high } = leafRowBounds(shape, coordinate);
  return [window[0], window[1], Math.min(...low), Math.max(...high)];
}

/** `np.arange(min, max + step / 2, step)`. */
export function gridPositions(min: number, max: number, step: number): number[] {
  if (!(step > 0)) return [];
  const count = Math.max(0, Math.ceil((max + 0.5 * step - min) / step));
  return Array.from({ length: count }, (_, i) => min + i * step);
}

/**
 * `_draw_bixel_grid`: positions of the bixel boundaries along both axes. With a `shape` the
 * vertical positions are mapped onto its leaf-pair axis (the aperture panel in leaf-pair mode).
 */
export function bixelGrid(fluence: SequenceFluence, leafPairShape: SequenceShape | null = null): { x: number[]; y: number[] } {
  const step = fluence.bixel_width ?? 0;
  const y = gridPositions(fluence.window[2], fluence.window[3], step);
  return {
    x: gridPositions(fluence.window[0], fluence.window[1], step),
    y: leafPairShape ? y.map((v) => leafNumber(v, leafPairShape)) : y,
  };
}

export interface FluenceGrid {
  /** Row-major, row 0 at the lowest stack position; NaN where the beam has no ray. */
  data: Float32Array;
  rows: number;
  cols: number;
}

/** Whether a fluence panel is drawn next to the aperture. */
export function hasFluence(beam: SequenceBeam): boolean {
  return Boolean(beam.shapes.length && beam.fluence && (beam.fluence.optimized || beam.fluence.delivered));
}

/** The map of a fluence mode (`difference` = delivered − optimized), `null` when it is not available. */
export function fluenceGrid(fluence: SequenceFluence, mode: FluenceMode): FluenceGrid | null {
  const of = (map: SequenceFluence["optimized"]): FluenceGrid | null =>
    map ? { data: map.data, rows: map.shape[0], cols: map.shape[1] } : null;
  if (mode !== "difference") return of(fluence[mode]);
  const delivered = of(fluence.delivered);
  const optimized = of(fluence.optimized);
  if (!delivered || !optimized) return null;
  return { ...delivered, data: delivered.data.map((v, i) => v - optimized.data[i]) };
}

/** Colour limits: `0..max` (viridis), symmetric about 0 for the difference (RdBu_r); 1 for an empty map. */
export function fluenceLimits(grid: FluenceGrid, mode: FluenceMode): [number, number] {
  let max = 0;
  let any = false;
  for (const v of grid.data) {
    if (!Number.isFinite(v)) continue;
    const m = mode === "difference" ? Math.abs(v) : v;
    if (!any || m > max) max = m;
    any = true;
  }
  const limit = max || 1;
  return mode === "difference" ? [-limit, limit] : [0, limit];
}

export const fluenceColormap = (mode: FluenceMode) => (mode === "difference" ? "RdBu_r" : "viridis");

/** Colour of a fluence value, `null` where the beam has no ray. */
export function fluenceColor(value: number, limits: readonly [number, number], mode: FluenceMode): Rgb | null {
  if (Number.isNaN(value)) return null;
  return mapColor(fluenceColormap(mode), (value - limits[0]) / (limits[1] - limits[0]));
}

/**
 * `_draw_target_outline`: the 0.5 contour of the projected target as segments
 * `[x0, y0, x1, y1, ...]` in mm (bin centres at `origin + index * spacing`).
 */
export function targetSegments(target: SequenceTarget): Float32Array {
  const [rows, cols] = target.mask.shape;
  const segments = marchingSquares(target.mask.data, cols, rows, 0.5);
  for (let i = 0; i < segments.length; i += 2) {
    segments[i] = target.origin[0] + (segments[i] - 0.5) * target.spacing[0];
    segments[i + 1] = target.origin[1] + (segments[i + 1] - 0.5) * target.spacing[1];
  }
  return segments;
}

/** Dwell time of a shape while playing: proportional to its weight, never below 150 ms. */
export function dwellMs(weight: number, maxWeight: number, speed: number): number {
  const frac = maxWeight > 0 ? weight / maxWeight : 1;
  return Math.max(Math.trunc((BASE_INTERVAL_MS * frac) / speed), MIN_INTERVAL_MS);
}

/** The shape after `index` while playing; `null` stops playback (last shape without Loop). */
export function nextShape(index: number, count: number, loop: boolean): number | null {
  if (index + 1 < count) return index + 1;
  return loop ? 0 : null;
}

/** Beam of a per-beam quantity key (`"..._beam <n>"`), else `null`. */
export function beamOfQuantity(key: string | null | undefined): number | null {
  const match = /_beam (\d+)$/.exec(key ?? "");
  return match ? Number(match[1]) : null;
}

/** "Sequencer: xia (5 levels)", `null` without a sequencer name. */
export function sequencerLabel(view: Pick<SequenceView, "sequencer" | "num_of_levels">): string | null {
  if (!view.sequencer) return null;
  return `Sequencer: ${view.sequencer}${view.num_of_levels ? ` (${view.num_of_levels} levels)` : ""}`;
}

/** Python `"%.<digits>g"`. */
export function formatG(value: number, digits = 2): string {
  if (!Number.isFinite(value)) return Number.isNaN(value) ? "nan" : value > 0 ? "inf" : "-inf";
  if (value === 0) return "0";
  const [mantissa, exponentText] = value.toExponential(digits - 1).split("e");
  const exponent = Number(exponentText);
  if (exponent < -4 || exponent >= digits) {
    return `${String(Number(mantissa))}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
  }
  return String(Number(value.toPrecision(digits)));
}

export function apertureTitle(label: string, index: number, count: number, weight: number): string {
  return `${label} — shape ${index + 1}/${count}, w=${formatG(weight)}`;
}

export function fluenceTitle(mode: FluenceMode): string {
  if (mode === "difference") return "Difference fluence (delivered − optimized)";
  return `${mode[0].toUpperCase()}${mode.slice(1)} fluence`;
}

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

/** The largest box of the data's aspect ratio centred in `box` (matplotlib `aspect="equal"`). */
export function equalAspectBox(box: Box, xSpan: number, ySpan: number): Box {
  if (!(xSpan > 0) || !(ySpan > 0) || !(box.width > 0) || !(box.height > 0)) return box;
  const scale = Math.min(box.width / xSpan, box.height / ySpan);
  const width = xSpan * scale;
  const height = ySpan * scale;
  return { left: box.left + (box.width - width) / 2, top: box.top + (box.height - height) / 2, width, height };
}

/** Pixel position of a data value on an axis; `flip` for the vertical axis (values increase upwards). */
export function axisScale(min: number, max: number, start: number, length: number, flip = false): (value: number) => number {
  const span = max - min || 1;
  return flip ? (v) => start + length - ((v - min) / span) * length : (v) => start + ((v - min) / span) * length;
}

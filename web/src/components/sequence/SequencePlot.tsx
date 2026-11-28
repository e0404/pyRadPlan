import { useEffect, useMemo, useRef, useState } from "react";
import type { BevWindow, SequenceBeam } from "../../api/protocol/viewer";
import { useTheme } from "../../theme";
import { plotColors, rgba, type PlotColors } from "../analysis/plotTheme";
import { niceTicks } from "../viewerModel";
import {
  BLOCKED_COLOR,
  apertureLimits,
  apertureRects,
  apertureTitle,
  axisScale,
  bixelGrid,
  equalAspectBox,
  fluenceColor,
  fluenceGrid,
  fluenceLimits,
  fluenceTitle,
  hasFluence,
  leafNumber,
  mlcBorderRects,
  targetSegments,
  weightColor,
  type Box,
  type FluenceMode,
  type LeafCoordinate,
  type Rect,
} from "./sequenceModel";

export interface SequenceOptions {
  coordinate: LeafCoordinate;
  mode: FluenceMode;
  mlcBorders: boolean;
  target: boolean;
  bixelGrid: boolean;
  targetColor: string;
}

const FONT = "11px system-ui, sans-serif";
const TITLE_FONT = "12px system-ui, sans-serif";
const MARGIN = { left: 52, right: 12, top: 26, bottom: 40 };
const COLORBAR = { gap: 10, width: 12, labels: 44 };
const APERTURE_GRID: readonly number[] = [153, 153, 153];
const FLUENCE_GRID: readonly number[] = [255, 255, 255];
const GRID_ALPHA = 0.6;

interface Axes {
  box: Box;
  X: (x: number) => number;
  Y: (y: number) => number;
}

function axes(box: Box, limits: BevWindow): Axes {
  return { box, X: axisScale(limits[0], limits[1], box.left, box.width), Y: axisScale(limits[2], limits[3], box.top, box.height, true) };
}

function clipped(ctx: CanvasRenderingContext2D, box: Box, draw: () => void): void {
  ctx.save();
  ctx.beginPath();
  ctx.rect(box.left, box.top, box.width, box.height);
  ctx.clip();
  draw();
  ctx.restore();
}

function pixelRect(a: Axes, r: Rect): [number, number, number, number] {
  return [a.X(r.x), a.Y(r.y + r.height), a.X(r.x + r.width) - a.X(r.x), a.Y(r.y) - a.Y(r.y + r.height)];
}

function frame(ctx: CanvasRenderingContext2D, a: Axes, limits: BevWindow, theme: PlotColors, labels: { x: string; y: string; title: string; titleFont: string }): void {
  const { box } = a;
  const fg = rgba(theme.foreground);
  ctx.strokeStyle = fg;
  ctx.fillStyle = fg;
  ctx.lineWidth = 1;
  ctx.font = FONT;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of niceTicks(limits[0], limits[1], 6)) {
    ctx.beginPath();
    ctx.moveTo(Math.round(a.X(t)) + 0.5, box.top + box.height);
    ctx.lineTo(Math.round(a.X(t)) + 0.5, box.top + box.height + 4);
    ctx.stroke();
    ctx.fillText(String(+t.toFixed(3)), a.X(t), box.top + box.height + 6);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of niceTicks(limits[2], limits[3], 6)) {
    ctx.beginPath();
    ctx.moveTo(box.left - 4, Math.round(a.Y(t)) + 0.5);
    ctx.lineTo(box.left, Math.round(a.Y(t)) + 0.5);
    ctx.stroke();
    ctx.fillText(String(+t.toFixed(3)), box.left - 6, a.Y(t));
  }
  ctx.strokeRect(Math.round(box.left) + 0.5, Math.round(box.top) + 0.5, Math.round(box.width), Math.round(box.height));
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText(labels.x, box.left + box.width / 2, box.top + box.height + 22);
  ctx.save();
  ctx.translate(box.left - 40, box.top + box.height / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText(labels.y, 0, 0);
  ctx.restore();
  ctx.font = labels.titleFont;
  ctx.textBaseline = "bottom";
  ctx.fillText(labels.title, box.left + box.width / 2, box.top - 6);
}

function gridLines(ctx: CanvasRenderingContext2D, a: Axes, grid: { x: number[]; y: number[] }, color: readonly number[]): void {
  ctx.strokeStyle = rgba(color, GRID_ALPHA);
  ctx.lineWidth = 0.6;
  ctx.beginPath();
  for (const x of grid.x) {
    ctx.moveTo(a.X(x), a.box.top);
    ctx.lineTo(a.X(x), a.box.top + a.box.height);
  }
  for (const y of grid.y) {
    ctx.moveTo(a.box.left, a.Y(y));
    ctx.lineTo(a.box.left + a.box.width, a.Y(y));
  }
  ctx.stroke();
}

function outline(ctx: CanvasRenderingContext2D, a: Axes, segments: Float32Array, color: string, mapY: (y: number) => number): void {
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (let i = 0; i < segments.length; i += 4) {
    ctx.moveTo(a.X(segments[i]), a.Y(mapY(segments[i + 1])));
    ctx.lineTo(a.X(segments[i + 2]), a.Y(mapY(segments[i + 3])));
  }
  ctx.stroke();
  ctx.lineCap = "butt";
}

/** Aperture and fluence panel of one beam (Qt `BeamApertureWidget` figure) on a canvas. */
export function SequencePlot(props: { beam: SequenceBeam; shape: number; options: SequenceOptions; testId?: string }) {
  const { beam, options } = props;
  const ref = useRef<HTMLCanvasElement>(null);
  const dark = useTheme((s) => s.dark);
  const [size, setSize] = useState<[number, number]>([0, 0]);
  const target = useMemo(() => (beam.target ? targetSegments(beam.target) : null), [beam.target]);
  const grid = useMemo(() => (beam.fluence ? fluenceGrid(beam.fluence, options.mode) : null), [beam.fluence, options.mode]);

  useEffect(() => {
    const canvas = ref.current!;
    const ro = new ResizeObserver(() => setSize([canvas.clientWidth, canvas.clientHeight]));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    const canvas = ref.current;
    const [w, h] = size;
    const shape = beam.shapes[props.shape];
    if (!canvas || w < 1 || h < 1) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const theme = plotColors();
    ctx.fillStyle = rgba(theme.figure);
    ctx.fillRect(0, 0, w, h);
    if (!shape || !beam.window) return;

    const withFluence = hasFluence(beam);
    const panelWidth = withFluence ? w / 2 : w;
    const panelBox = (left: number, right: number): Box => ({
      left: left + MARGIN.left,
      top: MARGIN.top,
      width: Math.max(1, panelWidth - MARGIN.left - right),
      height: Math.max(1, h - MARGIN.top - MARGIN.bottom),
    });

    // Aperture: blocked leaf regions on a background shaded by the shape's weight.
    const weight = beam.weights[props.shape];
    const limits = apertureLimits(shape, beam.window, options.coordinate);
    const aperture = axes(panelBox(0, MARGIN.right), limits);
    const leafPairs = options.coordinate === "leaf_num";
    clipped(ctx, aperture.box, () => {
      ctx.fillStyle = rgba(weightColor(weight, Math.max(...beam.weights)));
      ctx.fillRect(aperture.box.left, aperture.box.top, aperture.box.width, aperture.box.height);
      ctx.fillStyle = rgba(BLOCKED_COLOR);
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 0.5;
      for (const r of apertureRects(shape, beam.window!, options.coordinate)) {
        const px = pixelRect(aperture, r);
        ctx.fillRect(...px);
        ctx.strokeRect(...px);
      }
      if (options.bixelGrid && beam.fluence) gridLines(ctx, aperture, bixelGrid(beam.fluence, leafPairs ? shape : null), APERTURE_GRID);
      if (options.target && target) outline(ctx, aperture, target, options.targetColor, leafPairs ? (y) => leafNumber(y, shape) : (y) => y);
    });
    frame(ctx, aperture, limits, theme, {
      x: "horiz. pos. [mm]",
      y: leafPairs ? "leaf pair #" : "vert. pos. [mm]",
      title: apertureTitle(beam.label, props.shape, beam.shapes.length, weight),
      titleFont: TITLE_FONT,
    });
    if (!withFluence || !beam.fluence) return;

    // Fluence map with its colorbar; row 0 of the map is the lowest stack position.
    const outer = panelBox(panelWidth, COLORBAR.gap + COLORBAR.width + COLORBAR.labels);
    if (!grid) {
      ctx.fillStyle = rgba(theme.foreground);
      ctx.font = FONT;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`No ${options.mode} fluence`, outer.left + outer.width / 2, outer.top + outer.height / 2);
      return;
    }
    const fw = beam.fluence.window;
    const fluence = axes(equalAspectBox(outer, fw[1] - fw[0], fw[3] - fw[2]), fw);
    const range = fluenceLimits(grid, options.mode);
    const dx = (fw[1] - fw[0]) / grid.cols;
    const dy = (fw[3] - fw[2]) / grid.rows;
    clipped(ctx, fluence.box, () => {
      ctx.fillStyle = rgba(theme.axes);
      ctx.fillRect(fluence.box.left, fluence.box.top, fluence.box.width, fluence.box.height);
      for (let r = 0; r < grid.rows; r++) {
        for (let c = 0; c < grid.cols; c++) {
          const color = fluenceColor(grid.data[r * grid.cols + c], range, options.mode);
          if (!color) continue;
          // Cells overlap by a fraction of a pixel so no background shows between them.
          const x0 = Math.floor(fluence.X(fw[0] + c * dx));
          const x1 = Math.ceil(fluence.X(fw[0] + (c + 1) * dx));
          const y0 = Math.floor(fluence.Y(fw[2] + (r + 1) * dy));
          const y1 = Math.ceil(fluence.Y(fw[2] + r * dy));
          ctx.fillStyle = rgba(color);
          ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
        }
      }
      if (options.bixelGrid) gridLines(ctx, fluence, bixelGrid(beam.fluence!), FLUENCE_GRID);
      if (options.mlcBorders) {
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 0.8;
        for (const r of mlcBorderRects(shape, beam.window!)) ctx.strokeRect(...pixelRect(fluence, r));
      }
      if (options.target && target) outline(ctx, fluence, target, options.targetColor, (y) => y);
    });
    frame(ctx, fluence, fw, theme, { x: "x (BEV) [mm]", y: "z (BEV) [mm]", title: fluenceTitle(options.mode), titleFont: FONT });

    const bar: Box = {
      left: fluence.box.left + fluence.box.width + COLORBAR.gap,
      top: fluence.box.top + 0.1 * fluence.box.height,
      width: COLORBAR.width,
      height: 0.8 * fluence.box.height,
    };
    const barY = axisScale(range[0], range[1], bar.top, bar.height, true);
    const steps = Math.max(1, Math.round(bar.height));
    for (let i = 0; i < steps; i++) {
      const value = range[0] + ((i + 0.5) / steps) * (range[1] - range[0]);
      ctx.fillStyle = rgba(fluenceColor(value, range, options.mode)!);
      ctx.fillRect(bar.left, bar.top + bar.height - ((i + 1) / steps) * bar.height, bar.width, bar.height / steps + 0.5);
    }
    const fg = rgba(theme.foreground);
    ctx.strokeStyle = fg;
    ctx.fillStyle = fg;
    ctx.lineWidth = 1;
    ctx.font = FONT;
    ctx.strokeRect(Math.round(bar.left) + 0.5, Math.round(bar.top) + 0.5, bar.width, Math.round(bar.height));
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(range[0], range[1], 6)) {
      ctx.beginPath();
      ctx.moveTo(bar.left + bar.width, Math.round(barY(t)) + 0.5);
      ctx.lineTo(bar.left + bar.width + 3, Math.round(barY(t)) + 0.5);
      ctx.stroke();
      ctx.fillText(String(+t.toPrecision(3)), bar.left + bar.width + 5, barY(t));
    }
  }, [beam, props.shape, options, target, grid, size, dark]);

  return <canvas ref={ref} className="dvh-canvas" data-testid={props.testId} data-shape={props.shape} />;
}

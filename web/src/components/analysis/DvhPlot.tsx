import { useEffect, useRef, useState } from "react";
import { useTheme } from "../../theme";
import { niceTicks } from "../viewerModel";
import {
  doseAtVolume,
  formatMarkerParameter,
  formatMarkerValue,
  interp,
  markerX,
  markerY,
  type PlotCurve,
  type QiMarker,
} from "./dvhModel";
import { plotColors, rgba as rgbaOf } from "./plotTheme";

export interface LegendEntry {
  label: string;
  color: readonly number[] | null;
  style: "solid" | "dashed";
}

export interface DvhPlotProps {
  curves: PlotCurve[];
  bands: PlotCurve[];
  xLabel: string;
  secondaryXLabel: string | null;
  legend: LegendEntry[];
  /** Label of the dashed (compare) style in the cursor readout. */
  dashedName: string;
  markers: QiMarker[];
  visibleKinds: ReadonlySet<string>;
  highlight: { structure: number | null; metric: string | null };
  onStructureClick: (structure: number | null) => void;
  onMarkerClick: (structure: number, metric: string) => void;
  onPointRequested: (dose: number, volume: number) => void;
  onParameterMoved: (structure: number, metric: string, parameter: number) => void;
  canvasRef?: React.RefObject<HTMLCanvasElement | null>;
}

const LINE_WIDTH = 2;
const BAND_ALPHA = 70 / 255;
const HIGHLIGHT_WIDTH = 4;
const FADED_ALPHA = 60 / 255;
const MARKER_SIZE = 9;
const HIGHLIGHT_MARKER_SIZE = 14;
const CURVE_CLICK = 8;
const LEGEND_W = 150;
const Y_LABEL = "Volume [%]";

interface Frame {
  left: number;
  top: number;
  right: number;
  bottom: number;
  xMax: number;
  x2Max: number;
  yMin: number;
  yMax: number;
}

/** Interactive DVH plot (Qt `DVHPlotWidget`) on a canvas. */
export function DvhPlot(props: DvhPlotProps) {
  const ownRef = useRef<HTMLCanvasElement>(null);
  const ref = props.canvasRef ?? ownRef;
  const [hover, setHover] = useState<{ x: number; y: number } | null>(null);
  const [dragged, setDragged] = useState<{ parameter: number; value: number } | null>(null);
  const frameRef = useRef<Frame | null>(null);
  const dark = useTheme((s) => s.dark);
  const [size, setSize] = useState<[number, number]>([600, 300]);

  useEffect(() => {
    const canvas = ref.current!;
    const ro = new ResizeObserver(() => setSize([canvas.clientWidth, canvas.clientHeight]));
    ro.observe(canvas);
    return () => ro.disconnect();
  }, [ref]);

  const highlighted = props.markers.find((m) => m.structure === props.highlight.structure && m.metric === props.highlight.metric) ?? null;

  useEffect(() => setDragged(null), [highlighted?.structure, highlighted?.metric, highlighted?.parameter]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const [w, h] = size;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(w * dpr));
    canvas.height = Math.max(1, Math.round(h * dpr));
    const ctx = canvas.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const theme = plotColors();
    const fg = rgbaOf(theme.foreground);
    const rgba = (c: readonly number[] | null, a = 1) => rgbaOf(c ?? theme.foreground, a);
    ctx.fillStyle = rgbaOf(theme.figure);
    ctx.fillRect(0, 0, w, h);
    const primary = props.curves.filter((c) => !c.secondaryAxis);
    const secondary = props.curves.filter((c) => c.secondaryAxis);
    const maxOf = (cs: PlotCurve[]) => Math.max(0, ...cs.map((c) => (c.x.length ? c.x[c.x.length - 1] : 0)));
    const frame: Frame = {
      left: 58,
      top: props.secondaryXLabel ? 42 : 12,
      right: w - (props.legend.length ? LEGEND_W : 12),
      bottom: h - 40,
      xMax: Math.max(maxOf([...primary, ...props.bands]), 1e-6) * 1.02,
      x2Max: Math.max(maxOf(secondary), 1e-6) * 1.02,
      yMin: -2,
      yMax: 102,
    };
    frameRef.current = frame;
    ctx.fillStyle = rgbaOf(theme.axes);
    ctx.fillRect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    const X = (x: number) => frame.left + (x / frame.xMax) * (frame.right - frame.left);
    const X2 = (x: number) => frame.left + (x / frame.x2Max) * (frame.right - frame.left);
    const Y = (y: number) => frame.bottom - ((y - frame.yMin) / (frame.yMax - frame.yMin)) * (frame.bottom - frame.top);

    // Grid and axes.
    ctx.font = "11px system-ui, sans-serif";
    ctx.strokeStyle = rgbaOf(theme.grid);
    ctx.fillStyle = fg;
    ctx.lineWidth = 1;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (const t of niceTicks(0, frame.xMax, 7)) {
      ctx.beginPath();
      ctx.moveTo(X(t) + 0.5, frame.top);
      ctx.lineTo(X(t) + 0.5, frame.bottom);
      ctx.stroke();
      ctx.fillText(String(+t.toFixed(3)), X(t), frame.bottom + 4);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const t of niceTicks(0, 100, 6)) {
      ctx.beginPath();
      ctx.moveTo(frame.left, Y(t) + 0.5);
      ctx.lineTo(frame.right, Y(t) + 0.5);
      ctx.stroke();
      ctx.fillText(String(t), frame.left - 4, Y(t));
    }
    ctx.strokeStyle = fg;
    ctx.strokeRect(frame.left + 0.5, frame.top + 0.5, frame.right - frame.left, frame.bottom - frame.top);
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(props.xLabel, (frame.left + frame.right) / 2, h - 4);
    ctx.save();
    ctx.translate(14, (frame.top + frame.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textBaseline = "middle";
    ctx.fillText(Y_LABEL, 0, 0);
    ctx.restore();
    if (props.secondaryXLabel) {
      ctx.textBaseline = "bottom";
      for (const t of niceTicks(0, frame.x2Max, 7)) ctx.fillText(String(+t.toFixed(3)), X2(t), frame.top - 2);
      ctx.textBaseline = "top";
      ctx.fillText(props.secondaryXLabel, (frame.left + frame.right) / 2, 2);
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(frame.left, frame.top, frame.right - frame.left, frame.bottom - frame.top);
    ctx.clip();
    const hs = props.highlight.structure;
    const alpha = (structure: number) => (hs === null || hs === structure ? 1 : FADED_ALPHA);
    const stroke = (c: PlotCurve, width: number, a: number) => {
      const tx = c.secondaryAxis ? X2 : X;
      ctx.strokeStyle = rgba(c.color, a);
      ctx.lineWidth = width;
      ctx.setLineDash(c.style === "dashed" ? [7, 5] : []);
      ctx.beginPath();
      for (let i = 0; i < c.x.length; i++) {
        const px = tx(c.x[i]);
        const py = Y(c.y[i]);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.stroke();
    };
    for (const c of props.bands) stroke(c, 1, BAND_ALPHA);
    const ordered = [...props.curves].sort((a, b) => Number(a.index === hs) - Number(b.index === hs));
    for (const c of ordered) stroke(c, c.index === hs ? HIGHLIGHT_WIDTH : LINE_WIDTH, alpha(c.index));
    ctx.setLineDash([]);

    // Guide lines of the highlighted marker: solid at its value, dashed (draggable) at its parameter.
    if (highlighted) {
      const isDx = highlighted.kind === "dx";
      const param = dragged?.parameter ?? highlighted.parameter;
      const value = dragged?.value ?? highlighted.value;
      ctx.lineWidth = 2;
      ctx.strokeStyle = rgba(highlighted.color);
      ctx.beginPath();
      if (isDx) {
        ctx.moveTo(X(value), frame.top);
        ctx.lineTo(X(value), frame.bottom);
      } else {
        ctx.moveTo(frame.left, Y(value));
        ctx.lineTo(frame.right, Y(value));
      }
      ctx.stroke();
      ctx.setLineDash([7, 5]);
      ctx.beginPath();
      if (isDx) {
        ctx.moveTo(frame.left, Y(param));
        ctx.lineTo(frame.right, Y(param));
      } else {
        ctx.moveTo(X(param), frame.top);
        ctx.lineTo(X(param), frame.bottom);
      }
      ctx.stroke();
      ctx.setLineDash([]);
      const label = (text: string, x: number, y: number) => {
        ctx.font = "11px system-ui, sans-serif";
        const tw = ctx.measureText(text).width + 8;
        ctx.fillStyle = rgbaOf(theme.axes, 200 / 255);
        ctx.fillRect(x - tw / 2, y - 9, tw, 18);
        ctx.strokeStyle = rgbaOf(theme.grid);
        ctx.strokeRect(x - tw / 2 + 0.5, y - 8.5, tw - 1, 17);
        ctx.fillStyle = fg;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(text, x, y);
      };
      const at = 0.93;
      if (isDx) {
        label(formatMarkerValue(value, highlighted.valueUnit), X(value), frame.bottom - at * (frame.bottom - frame.top));
        label(formatMarkerParameter("dx", param, highlighted.parameterUnit), frame.left + at * (frame.right - frame.left), Y(param));
      } else {
        label(formatMarkerValue(value, highlighted.valueUnit), frame.left + at * (frame.right - frame.left), Y(value));
        label(formatMarkerParameter("vx", param, highlighted.parameterUnit), X(param), frame.bottom - at * (frame.bottom - frame.top));
      }
    }

    // Markers: D_x circles, V_x squares; the highlighted one always.
    for (const m of props.markers) {
      const isHighlighted = m === highlighted;
      if (!props.visibleKinds.has(m.kind) && !isHighlighted) continue;
      const pos = isHighlighted && dragged ? (m.kind === "dx" ? [dragged.value, dragged.parameter] : [dragged.parameter, dragged.value]) : [markerX(m), markerY(m)];
      const s = (isHighlighted ? HIGHLIGHT_MARKER_SIZE : MARKER_SIZE) / 2;
      const a = alpha(m.structure);
      ctx.fillStyle = rgba(m.color, a);
      ctx.strokeStyle = rgba(theme.foreground, a);
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (m.kind === "dx") ctx.arc(X(pos[0]), Y(pos[1]), s, 0, 2 * Math.PI);
      else ctx.rect(X(pos[0]) - s, Y(pos[1]) - s, 2 * s, 2 * s);
      ctx.fill();
      ctx.stroke();
    }

    // Data cursor.
    if (hover !== null && props.curves.length) {
      const { x, y } = hover;
      ctx.setLineDash([2, 3]);
      ctx.strokeStyle = fg;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(X(x) + 0.5, frame.top);
      ctx.lineTo(X(x) + 0.5, frame.bottom);
      ctx.moveTo(frame.left, Y(y) + 0.5);
      ctx.lineTo(frame.right, Y(y) + 0.5);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();

    if (hover !== null && props.curves.length) {
      const { x, y } = hover;
      const x2 = props.secondaryXLabel ? (x / frame.xMax) * frame.x2Max : null;
      const rows: { text: string; color: readonly number[] | null; symbol: string }[] = [{ text: `${props.xLabel}: ${x.toFixed(2)}`, color: null, symbol: "" }];
      if (x2 !== null) rows.push({ text: `${props.secondaryXLabel}: ${x2.toFixed(2)}`, color: null, symbol: "" });
      rows.push({ text: `${Y_LABEL}: ${y.toFixed(1)}`, color: null, symbol: "" });
      for (const c of props.curves) {
        const at = c.secondaryAxis && x2 !== null ? x2 : x;
        const v = interp(at, c.x, c.y);
        const name = c.style === "solid" ? c.name : `${c.name} (${props.dashedName})`;
        rows.push({ text: `${name}: ${v.toFixed(1)} %`, color: c.color, symbol: c.style === "solid" ? "━" : "┅" });
      }
      ctx.font = "11px system-ui, sans-serif";
      const width = Math.max(...rows.map((r) => ctx.measureText(`${r.symbol} ${r.text}`).width)) + 12;
      const height = rows.length * 15 + 8;
      let bx = X(x) + 6;
      if (X(x) > (frame.left + frame.right) / 2) bx = X(x) - width - 6;
      const by = frame.top + 4;
      ctx.fillStyle = rgbaOf(theme.axes, 220 / 255);
      ctx.fillRect(bx, by, width, height);
      ctx.strokeStyle = rgbaOf(theme.grid);
      ctx.strokeRect(bx + 0.5, by + 0.5, width - 1, height - 1);
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      rows.forEach((r, i) => {
        let tx = bx + 6;
        if (r.symbol) {
          ctx.fillStyle = rgba(r.color);
          ctx.fillText(r.symbol, tx, by + 4 + i * 15);
          tx += ctx.measureText(`${r.symbol} `).width;
        }
        ctx.fillStyle = fg;
        ctx.fillText(r.text, tx, by + 4 + i * 15);
      });
    }

    // Legend.
    if (props.legend.length) {
      const lx = frame.right + 10;
      let ly = frame.top;
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillStyle = rgbaOf(theme.axes);
      ctx.strokeStyle = rgbaOf(theme.grid);
      const lh = props.legend.length * 16 + 8;
      ctx.fillRect(lx, ly, LEGEND_W - 16, lh);
      ctx.strokeRect(lx + 0.5, ly + 0.5, LEGEND_W - 17, lh - 1);
      ly += 4;
      for (const e of props.legend) {
        ctx.strokeStyle = rgba(e.color);
        ctx.lineWidth = 2;
        ctx.setLineDash(e.style === "dashed" ? [5, 3] : []);
        ctx.beginPath();
        ctx.moveTo(lx + 6, ly + 8);
        ctx.lineTo(lx + 28, ly + 8);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = fg;
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(e.label, lx + 34, ly + 8, LEGEND_W - 54);
        ly += 16;
      }
    }
  }, [props, size, hover, dragged, highlighted, ref, dark]);

  // -------------------------------------------------------------------------
  // Interaction
  // -------------------------------------------------------------------------

  const toData = (e: { clientX: number; clientY: number }) => {
    const canvas = ref.current!;
    const frame = frameRef.current!;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const inside = px >= frame.left && px <= frame.right && py >= frame.top && py <= frame.bottom;
    const x = ((px - frame.left) / (frame.right - frame.left)) * frame.xMax;
    const y = frame.yMin + ((frame.bottom - py) / (frame.bottom - frame.top)) * (frame.yMax - frame.yMin);
    return { px, py, x, y, inside, frame };
  };

  const toPx = (frame: Frame, x: number, y: number, secondary = false) => [
    frame.left + (x / (secondary ? frame.x2Max : frame.xMax)) * (frame.right - frame.left),
    frame.bottom - ((y - frame.yMin) / (frame.yMax - frame.yMin)) * (frame.bottom - frame.top),
  ];

  const markerAt = (px: number, py: number, frame: Frame): QiMarker | null => {
    for (const m of props.markers) {
      if (!props.visibleKinds.has(m.kind) && m !== highlighted) continue;
      const [mx, my] = toPx(frame, markerX(m), markerY(m));
      if (Math.hypot(mx - px, my - py) <= 7) return m;
    }
    return null;
  };

  const curveAt = (px: number, py: number, frame: Frame): number | null => {
    let best: number | null = null;
    let bestD = CURVE_CLICK;
    for (const c of props.curves) {
      for (let i = 1; i < c.x.length; i++) {
        const [ax, ay] = toPx(frame, c.x[i - 1], c.y[i - 1], c.secondaryAxis);
        const [bx, by] = toPx(frame, c.x[i], c.y[i], c.secondaryAxis);
        const dx = bx - ax;
        const dy = by - ay;
        const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
        const d = Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
        if (d < bestD) {
          bestD = d;
          best = c.index;
        }
      }
    }
    return best;
  };

  const nearParameterLine = (px: number, py: number, frame: Frame): boolean => {
    if (!highlighted) return false;
    const [lx, ly] = toPx(frame, highlighted.kind === "dx" ? 0 : highlighted.parameter, highlighted.kind === "dx" ? highlighted.parameter : 0);
    return highlighted.kind === "dx" ? Math.abs(py - ly) <= 5 : Math.abs(px - lx) <= 5;
  };

  const draggedPoint = (m: QiMarker, parameter: number) => {
    const curve = props.curves.find((c) => c.index === m.structure && c.style === "solid");
    if (!curve) return null;
    if (m.kind === "dx") {
      const p = Math.min(100, Math.max(0, parameter));
      return { parameter: p, value: doseAtVolume(curve.x, curve.y, p) };
    }
    const p = Math.max(0, parameter);
    return { parameter: p, value: interp(p, curve.x, curve.y) };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (e.button !== 0) return;
    const { px, py, inside, frame } = toData(e);
    if (!inside) return;
    if (highlighted && nearParameterLine(px, py, frame)) {
      const target = e.currentTarget;
      target.setPointerCapture(e.pointerId);
      let last: { parameter: number; value: number } | null = null;
      const move = (ev: PointerEvent) => {
        const d = toData(ev);
        last = draggedPoint(highlighted, highlighted.kind === "dx" ? d.y : d.x);
        setDragged(last);
      };
      const up = (ev: PointerEvent) => {
        target.releasePointerCapture(ev.pointerId);
        target.removeEventListener("pointermove", move);
        target.removeEventListener("pointerup", up);
        if (last) {
          const decimals = highlighted.kind === "dx" ? 1 : 2;
          props.onParameterMoved(highlighted.structure, highlighted.metric, +last.parameter.toFixed(decimals));
        }
      };
      target.addEventListener("pointermove", move);
      target.addEventListener("pointerup", up);
      return;
    }
    const marker = markerAt(px, py, frame);
    if (marker) return props.onMarkerClick(marker.structure, marker.metric);
    props.onStructureClick(curveAt(px, py, frame));
  };

  return (
    <canvas
      ref={ref}
      className="dvh-canvas"
      data-testid="dvh-plot"
      onPointerMove={(e) => {
        const { x, y, inside, px, py, frame } = toData(e);
        setHover(inside ? { x: Math.max(0, x), y } : null);
        const m = inside ? markerAt(px, py, frame) : null;
        e.currentTarget.title = m ? `${m.name}: ${m.metric} = ${formatMarkerValue(m.value, m.valueUnit)}` : highlighted && nearParameterLine(px, py, frame) ? `Drag to change ${highlighted.metric}` : "";
        e.currentTarget.style.cursor = highlighted && nearParameterLine(px, py, frame) ? (highlighted.kind === "dx" ? "ns-resize" : "ew-resize") : "";
      }}
      onPointerLeave={() => setHover(null)}
      onPointerDown={onPointerDown}
      onDoubleClick={(e) => {
        const { x, y, inside } = toData(e);
        if (inside) props.onPointRequested(Math.max(0, x), Math.min(100, Math.max(0, y)));
      }}
    />
  );
}

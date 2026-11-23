/**
 * "Optimization Status" window (Qt `OptimizationStatusWidget`,
 * `widgets/optimization/_status_window.py`): a non-modal top-level window opened by Optimize
 * and reused, with the summary "iter N · f=… · Δf=…", the objective plot (log y, "+"
 * markers, x = iteration) and Pause / Resume + Stop. Finalized when the run ends: buttons off,
 * curves kept. Fed from `progress.data`; restored from `session.history` after a reload.
 */
import { useEffect, useMemo, useRef, useState, type PointerEvent as ReactPointerEvent } from "react";
import { createPortal } from "react-dom";
import { useApp } from "../../store/appStore";
import { feature, statusActions, usePlanStore, type OptPoint } from "../../store/planSlice";
import "../plan/plan.css";

/** Python's `f"{x:.3e}"` (two-digit exponent). */
export function pyExp(x: number, digits: number, sign = false): string {
  const s = x.toExponential(digits).replace(/e([+-])(\d)$/, "e$10$2");
  return sign && x >= 0 ? `+${s}` : s;
}

/** Summary line (`_format_summary`). */
export function formatSummary(last: Record<string, number> | null, points: readonly OptPoint[]): string {
  if (!last) return "Waiting for optimization…";
  const parts: string[] = [];
  if (typeof last.iteration === "number") parts.push(`iter ${Math.trunc(last.iteration)}`);
  if (typeof last.objective === "number") parts.push(`f=${pyExp(last.objective, 3)}`);
  if (points.length >= 2) {
    const a = points[points.length - 2].objective;
    const b = points[points.length - 1].objective;
    if (a !== 0) parts.push(`Δf=${pyExp((b - a) / Math.abs(a), 2, true)}`);
  }
  return parts.length ? parts.join(" · ") : "Optimizing…";
}

interface Axis {
  min: number;
  max: number;
  ticks: Array<{ v: number; label: string; major: boolean }>;
}

function niceStep(span: number, count: number): number {
  const raw = span / Math.max(1, count);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const f = raw / mag;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag;
}

export function linearAxis(lo: number, hi: number, count = 6): Axis {
  if (!(hi > lo)) {
    lo -= 1;
    hi += 1;
  }
  const step = niceStep(hi - lo, count);
  const min = Math.floor(lo / step) * step;
  const max = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = min; v <= max + step / 2; v += step) ticks.push({ v, label: String(Number(v.toPrecision(6))), major: true });
  return { min, max, ticks };
}

/** Log axis in log10 units: decades, with 2 / 5 sub-ticks labelled when it spans few decades. */
export function logAxis(lo: number, hi: number): Axis {
  let min = Math.floor(Math.log10(lo));
  let max = Math.ceil(Math.log10(hi));
  if (max === min) {
    min -= 0;
    max += 1;
  }
  const ticks: Axis["ticks"] = [];
  const few = max - min <= 2;
  for (let d = min; d <= max; d++) {
    ticks.push({ v: d, label: `1e${d}`, major: true });
    if (d < max)
      for (const m of [2, 3, 4, 5, 6, 7, 8, 9]) ticks.push({ v: d + Math.log10(m), label: few && (m === 2 || m === 5) ? `${m}e${d}` : "", major: false });
  }
  const fmt = (d: number) => {
    const v = 10 ** d;
    return v >= 0.01 && v < 1e5 ? String(Number(v.toPrecision(3))) : pyExp(v, 0);
  };
  for (const t of ticks) if (t.label) t.label = fmt(t.v);
  return { min, max, ticks };
}

export function ObjectivePlot(props: { points: readonly OptPoint[]; width?: number; height?: number }) {
  const W = props.width ?? 576;
  const H = props.height ?? 300;
  const m = { l: 62, r: 14, t: 26, b: 38 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const positive = props.points.filter((p) => p.objective > 0);
  const log = positive.length === props.points.length && positive.length > 0;
  const xs = props.points.map((p) => p.iteration);
  const ys = props.points.map((p) => (log ? Math.log10(p.objective) : p.objective));
  const xAxis = linearAxis(xs.length ? Math.min(...xs) : 0, xs.length ? Math.max(...xs) : 10, 7);
  const yAxis = log ? logAxis(Math.min(...positive.map((p) => p.objective)), Math.max(...positive.map((p) => p.objective))) : linearAxis(ys.length ? Math.min(...ys) : 0, ys.length ? Math.max(...ys) : 1, 5);
  const X = (v: number) => m.l + ((v - xAxis.min) / (xAxis.max - xAxis.min || 1)) * pw;
  const Y = (v: number) => m.t + ph - ((v - yAxis.min) / (yAxis.max - yAxis.min || 1)) * ph;
  return (
    <svg className="opt-plot" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Objective value per iteration" data-testid="opt-plot">
      <rect x={0} y={0} width={W} height={H} className="opt-plot-bg" />
      <text x={m.l + pw / 2} y={16} textAnchor="middle" className="opt-plot-title">
        Objective value
      </text>
      <g className="opt-grid">
        {xAxis.ticks.map((t) => (
          <line key={`gx${t.v}`} x1={X(t.v)} x2={X(t.v)} y1={m.t} y2={m.t + ph} />
        ))}
        {yAxis.ticks
          .filter((t) => t.major || t.label)
          .map((t) => (
            <line key={`gy${t.v}`} x1={m.l} x2={m.l + pw} y1={Y(t.v)} y2={Y(t.v)} />
          ))}
      </g>
      <g className="opt-axis">
        <line x1={m.l} x2={m.l + pw} y1={m.t + ph} y2={m.t + ph} />
        <line x1={m.l} x2={m.l} y1={m.t} y2={m.t + ph} />
        {xAxis.ticks.map((t) => (
          <g key={`x${t.v}`}>
            <line x1={X(t.v)} x2={X(t.v)} y1={m.t + ph} y2={m.t + ph + 5} />
            <text x={X(t.v)} y={m.t + ph + 17} textAnchor="middle">
              {t.label}
            </text>
          </g>
        ))}
        {yAxis.ticks.map((t) => (
          <g key={`y${t.v}`}>
            <line x1={m.l - (t.major ? 5 : 3)} x2={m.l} y1={Y(t.v)} y2={Y(t.v)} />
            {t.label && (
              <text x={m.l - 8} y={Y(t.v) + 4} textAnchor="end">
                {t.label}
              </text>
            )}
          </g>
        ))}
        <text x={m.l + pw / 2} y={H - 6} textAnchor="middle">
          Iteration
        </text>
      </g>
      <clipPath id="opt-clip">
        <rect x={m.l} y={m.t - 6} width={pw + 6} height={ph + 12} />
      </clipPath>
      <g className="opt-marks" clipPath="url(#opt-clip)">
        {props.points.map((_p, i) => {
          const x = X(xs[i]);
          const y = Y(ys[i]);
          return <path key={i} d={`M${x - 6} ${y}H${x + 6}M${x} ${y - 6}V${y + 6}`} />;
        })}
      </g>
    </svg>
  );
}

export function StatusWindow() {
  const status = usePlanStore((s) => s.status);
  const info = useApp((s) => s.info);
  const canCancel = useApp((s) => s.backend?.canCancel ?? false);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const drag = useRef<{ dx: number; dy: number } | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  const pause = feature(info, "pause");
  const summary = useMemo(() => formatSummary(status.last, status.points), [status.last, status.points]);

  useEffect(() => {
    if (!status.open || pos) return;
    const w = 600;
    const h = 440;
    setPos({ x: Math.max(8, (window.innerWidth - w) / 2), y: Math.max(8, Math.min((window.innerHeight - h) / 2, window.innerHeight - h - 8)) });
  }, [status.open, pos]);

  if (!status.open) return null;
  const onDown = (e: ReactPointerEvent) => {
    if (!pos) return;
    drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onMove = (e: ReactPointerEvent) => {
    if (!drag.current) return;
    const x = Math.max(0, Math.min(window.innerWidth - 120, e.clientX - drag.current.dx));
    const y = Math.max(0, Math.min(window.innerHeight - 40, e.clientY - drag.current.dy));
    setPos({ x, y });
  };
  const running = status.running;
  return createPortal(
    <div
      ref={ref}
      className="opt-window"
      role="dialog"
      aria-label="Optimization Status"
      data-testid="opt-status-window"
      data-running={running ? "true" : "false"}
      data-outcome={status.outcome ?? ""}
      style={pos ? { left: pos.x, top: pos.y } : { visibility: "hidden" }}
    >
      <div className="opt-titlebar" onPointerDown={onDown} onPointerMove={onMove} onPointerUp={() => (drag.current = null)}>
        <span>Optimization Status</span>
        <button type="button" className="title-close" onPointerDown={(e) => e.stopPropagation()} onClick={statusActions.close} aria-label="Close" data-testid="opt-close">
          ✕
        </button>
      </div>
      <div className="opt-body">
        <div className="opt-summary" data-testid="opt-summary">
          {summary}
        </div>
        <ObjectivePlot points={status.points} />
        <div className="row">
          <span className="tip-wrap" title={pause.available ? undefined : (pause.reason ?? "")}>
            <button
              type="button"
              className={`btn${status.paused ? " checked" : ""}`}
              disabled={!running || !pause.available}
              onClick={() => void statusActions.togglePause()}
              data-testid="opt-pause"
            >
              {status.paused ? "Resume" : "Pause"}
            </button>
          </span>
          <button type="button" className="btn" disabled={!running || status.stopRequested || !canCancel} onClick={statusActions.stop} data-testid="opt-stop">
            Stop
          </button>
          <span className="fill" />
        </div>
      </div>
    </div>,
    document.body,
  );
}

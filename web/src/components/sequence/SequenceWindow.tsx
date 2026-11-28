import { useEffect, useMemo, useRef, useState } from "react";
import { BackendError } from "../../api/backend";
import type { SequenceBeam, SequenceView } from "../../api/protocol/viewer";
import { useApp } from "../../store/appStore";
import { NumberField } from "../common";
import { SequencePlot, type SequenceOptions } from "./SequencePlot";
import { DEFAULT_TARGET_COLOR, FLUENCE_MODES, LEAF_COORDINATES, SPEED_RANGE, dwellMs, hasFluence, nextShape, sequencerLabel } from "./sequenceModel";
import "../viewer.css";

const DEFAULT_OPTIONS: SequenceOptions = {
  coordinate: "leaf_num",
  mode: "delivered",
  mlcBorders: true,
  target: true,
  bixelGrid: false,
  targetColor: DEFAULT_TARGET_COLOR,
};

/** Qt `BeamApertureWidget`: one shape of a beam at a time, with playback controls. */
function BeamPlayer(props: { beam: SequenceBeam; options: SequenceOptions; active: boolean }) {
  const { beam } = props;
  const count = beam.shapes.length;
  const [shape, setShape] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [loop, setLoop] = useState(true);
  const maxWeight = count ? Math.max(...beam.weights) : 0;
  const id = `sequence-beam-${beam.index}`;

  useEffect(() => {
    if (!playing) return;
    const timer = setTimeout(() => {
      const next = nextShape(shape, count, loop);
      if (next === null) setPlaying(false);
      else setShape(next);
    }, dwellMs(beam.weights[shape], maxWeight, speed));
    return () => clearTimeout(timer);
  }, [playing, shape, speed, loop, count, maxWeight, beam.weights]);

  return (
    <div className="sequence-beam" style={{ display: props.active ? undefined : "none" }} data-testid={id} data-shapes={count} data-fluence={hasFluence(beam)}>
      <div className="sequence-plot">
        <SequencePlot beam={beam} shape={shape} options={props.options} testId={`${id}-canvas`} />
      </div>
      <div className="row">
        <button className={`btn ${playing ? "checked" : ""}`} disabled={count < 2} onClick={() => setPlaying((p) => !p)} data-testid={`${id}-play`}>
          {playing ? "Pause" : "Play"}
        </button>
        <input
          type="range"
          className="fill"
          min={0}
          max={Math.max(count - 1, 0)}
          step={1}
          value={shape}
          disabled={count < 2}
          onChange={(e) => setShape(Number(e.target.value))}
          data-testid={`${id}-slider`}
        />
        <span data-testid={`${id}-label`}>{count ? `${shape + 1} / ${count}` : ""}</span>
      </div>
      <div className="row">
        <label className="check">
          Speed:&nbsp;
          <NumberField value={speed} onChange={setSpeed} min={SPEED_RANGE.min} max={SPEED_RANGE.max} step={SPEED_RANGE.step} decimals={1} className="sequence-speed" testId={`${id}-speed`} />
          x
        </label>
        <label className="check">
          <input type="checkbox" checked={loop} onChange={(e) => setLoop(e.target.checked)} data-testid={`${id}-loop`} />
          Loop
        </label>
      </div>
    </div>
  );
}

/** Qt "pyRadPlan Sequence Viewer" window (non-modal): MLC shapes and fluence of every beam. */
export function SequenceWindow(props: { onClose: () => void }) {
  const revision = useApp((s) => s.state?.revisions?.workspace.result);
  const request = useApp((s) => s.sequenceBeam);
  const [loaded, setLoaded] = useState<{ view: SequenceView; revision: number | undefined; serial: number } | null>(null);
  const [options, setOptions] = useState(DEFAULT_OPTIONS);
  const [tab, setTab] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [pos, setPos] = useState({ x: Math.max(16, window.innerWidth / 2 - 450), y: 50 });
  const shown = useRef<typeof loaded>(null);
  const onClose = useRef(props.onClose);
  onClose.current = props.onClose;

  // The sequence of the active result; a new one resets the view options (Qt `set_sequence`).
  useEffect(() => {
    const { backend, session } = useApp.getState();
    if (!backend || !session) return;
    let cancelled = false;
    backend
      .request("view.sequence", {}, session)
      .then((view) => {
        if (cancelled) return;
        setError(null);
        setOptions(DEFAULT_OPTIONS);
        setLoaded((previous) => ({ view, revision, serial: (previous?.serial ?? 0) + 1 }));
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof BackendError && err.code === "no_sequence") onClose.current();
        else setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, [revision]);

  // A new sequence starts on its first beam; Show Sequence selects the beam of the shown per-beam quantity.
  useEffect(() => {
    if (!loaded) return;
    const fresh = shown.current !== loaded;
    shown.current = loaded;
    const beam = request && request.revision === loaded.revision ? request.beam : null;
    if (beam !== null && beam < loaded.view.beams.length) setTab(beam);
    else if (fresh) setTab(0);
  }, [loaded, request]);

  const onTitleDown = (e: React.PointerEvent) => {
    const start = { x: e.clientX - pos.x, y: e.clientY - pos.y };
    const move = (ev: PointerEvent) => setPos({ x: Math.max(0, ev.clientX - start.x), y: Math.max(0, ev.clientY - start.y) });
    const up = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
  };

  const view = loaded?.view;
  const label = view ? sequencerLabel(view) : null;
  const overlays = useMemo(() => Boolean(view?.beams.some(hasFluence)), [view]);
  const set = (changes: Partial<SequenceOptions>) => setOptions((o) => ({ ...o, ...changes }));
  const check = (key: "mlcBorders" | "target" | "bixelGrid", text: string, title?: string) => (
    <label className="check" title={title}>
      <input type="checkbox" checked={options[key]} onChange={(e) => set({ [key]: e.target.checked })} data-testid={`sequence-${key}`} />
      {text}
    </label>
  );

  return (
    <div
      className="analysis-window sequence-window"
      style={{ left: pos.x, top: pos.y }}
      data-testid="sequence-window"
      data-beams={view?.beams.length ?? 0}
      role="dialog"
      aria-label="pyRadPlan Sequence Viewer"
    >
      <div className="analysis-title" onPointerDown={onTitleDown}>
        <span>pyRadPlan Sequence Viewer</span>
        <button className="btn small" onClick={props.onClose} onPointerDown={(e) => e.stopPropagation()} data-testid="sequence-close">
          ✕
        </button>
      </div>
      <div className="analysis-body">
        <div className="row">
          {label && <span data-testid="sequence-sequencer">{label}</span>}
          <span className="fill" />
          <span>Stacking axis:</span>
          <select value={options.coordinate} onChange={(e) => set({ coordinate: e.target.value as SequenceOptions["coordinate"] })} data-testid="sequence-coordinate">
            {LEAF_COORDINATES.map(([text, value]) => (
              <option key={value} value={value}>
                {text}
              </option>
            ))}
          </select>
        </div>
        {overlays && (
          <div className="row sequence-overlays" data-testid="sequence-overlays">
            <span>Fluence:</span>
            <select value={options.mode} onChange={(e) => set({ mode: e.target.value as SequenceOptions["mode"] })} data-testid="sequence-fluence">
              {FLUENCE_MODES.map(([text, value]) => (
                <option key={value} value={value}>
                  {text}
                </option>
              ))}
            </select>
            {check("mlcBorders", "MLC boundaries")}
            <label className="swatch swatch-button" style={{ background: options.targetColor }} title="Colour of the projected-target outline">
              <input
                type="color"
                style={{ opacity: 0, width: 0, height: 0, position: "absolute" }}
                value={options.targetColor}
                onChange={(e) => set({ targetColor: e.target.value })}
                data-testid="sequence-target-color"
              />
            </label>
            {check("target", "Target (BEV)", "Outline of the target projected into the beam's-eye view")}
            {check("bixelGrid", "Bixel grid", "Overlay the ray/bixel grid the fluence is sampled on")}
          </div>
        )}
        {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
        <div className="tabs sequence-tabs">
          {view?.beams.map((beam, i) => (
            <button key={beam.index} className={`tab ${tab === i ? "active" : ""}`} onClick={() => setTab(i)} data-testid={`sequence-tab-${beam.index}`}>
              {beam.label}
            </button>
          ))}
        </div>
        {/* Every beam keeps its own player, as the Qt tabs do; a new sequence starts them afresh. */}
        {loaded && view?.beams.map((beam, i) => <BeamPlayer key={`${loaded.serial}-${beam.index}`} beam={beam} options={options} active={tab === i} />)}
      </div>
    </div>
  );
}

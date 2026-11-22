import { useEffect, useMemo, useRef, useState } from "react";
import type { ResultGroup, StructureInfo, Vec3 } from "../../api/protocol";
import type { DvhSet, QiResult } from "../../api/protocol/viewer";
import { formatNumbers } from "../../results/numbers";
import { keyLabel, locateKey } from "../../results/resultGroups";
import { useApp } from "../../store/appStore";
import { hasMethod, resultGroups } from "../../store/revisions";
import { Dialog } from "../common";
import { QuantitySelector } from "../QuantitySelector";
import { groupByType, truncateName } from "../StructureVisibility";
import { DvhPlot, type LegendEntry } from "./DvhPlot";
import {
  applyPoint,
  axisLabel,
  curvesCsv,
  parseRefDoses,
  parseRefVols,
  qiMarkers,
  qiTableRows,
  toCsv,
  toTsv,
  type PlotCurve,
} from "./dvhModel";
import "../viewer.css";

const NONE_LABEL = "— None —";
const DEFAULT_REF_VOLS = "2 5 50 95 98";
const FIXED_METRICS = ["mean", "std", "max", "min"];

function download(name: string, mime: string, data: BlobPart): void {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function request<T>(method: string, params: Record<string, unknown>): Promise<T> {
  const { backend, session } = useApp.getState();
  if (!backend || !session) throw new Error("Backend not connected");
  return (await backend.request(method as never, params as never, session)) as T;
}

/** DVH sets per key (v1 `keys`; per key for a v0 backend). */
async function fetchDvhSets(keys: string[]): Promise<DvhSet[]> {
  try {
    const r = await request<{ sets?: DvhSet[] }>("analysis.dvh", { keys, num_points: 1000, range: "per_structure" });
    if (r.sets) return r.sets;
  } catch {
    /* v0 backend: fall through */
  }
  const sets: DvhSet[] = [];
  for (const key of keys) {
    const r = await request<{ quantity: string; unit: string; label?: string; curves: DvhSet["curves"] }>("analysis.dvh", {
      quantity: key,
      num_points: 1000,
    });
    sets.push({ key, label: r.label ?? "Dose", unit: r.unit, unit_info: { symbol: r.unit, dimensionless: !r.unit }, curves: r.curves });
  }
  return sets;
}

interface PointRequest {
  dose: number;
  volume: number;
  kind?: "dx" | "vx";
  replace?: string;
  replaceValue?: number;
  structure?: number;
}

/** Qt `QIPointDialog`: confirm a D_x / V_x point picked in the DVH plot. */
function QiPointDialog(props: { req: PointRequest; unit: string; onDone: (choice: { kind: "dx" | "vx"; parameter: number; replace: boolean } | null) => void }) {
  const { req } = props;
  const [kind, setKind] = useState<"dx" | "vx">(req.kind ?? "dx");
  const [volume, setVolume] = useState(+req.volume.toFixed(1));
  const [dose, setDose] = useState(+req.dose.toFixed(2));
  const choose = (replace: boolean) => props.onDone({ kind, parameter: kind === "dx" ? volume : dose, replace });
  return (
    <Dialog
      title={req.replace ? `Change ${req.replace}` : "Add quality indicator"}
      onClose={() => props.onDone(null)}
      testId="qi-point-dialog"
      buttons={
        <>
          <button className="btn" onClick={() => props.onDone(null)}>
            Cancel
          </button>
          {req.replace && (
            <button className="btn" onClick={() => choose(true)} data-testid="qi-point-replace">
              Replace {req.replace}
            </button>
          )}
          <button className="btn primary" onClick={() => choose(false)} data-testid="qi-point-add">
            {req.replace ? "Add as new" : "Add"}
          </button>
        </>
      }
    >
      <div className="form-grid" style={{ gridTemplateColumns: "auto 90px auto" }}>
        {req.kind !== "vx" && (
          <>
            <label className="check">
              <input type="radio" checked={kind === "dx"} onChange={() => setKind("dx")} />
              D_x: dose covering
            </label>
            <input type="number" min={0} max={100} step={0.1} value={volume} onChange={(e) => setVolume(Number(e.target.value))} data-testid="qi-point-volume" />
            <span>% of the volume</span>
          </>
        )}
        {req.kind !== "dx" && (
          <>
            <label className="check">
              <input type="radio" checked={kind === "vx"} onChange={() => setKind("vx")} />
              V_x: volume receiving at least
            </label>
            <input type="number" min={0} step={0.01} value={dose} onChange={(e) => setDose(Number(e.target.value))} data-testid="qi-point-dose" />
            <span>{props.unit}</span>
          </>
        )}
      </div>
    </Dialog>
  );
}

function StructureList(props: {
  structures: readonly StructureInfo[];
  checked: Record<number, boolean>;
  colors: Record<number, Vec3>;
  onToggle: (index: number, value: boolean) => void;
  onColor: (index: number, rgb: Vec3) => void;
  onAll: (value: boolean) => void;
}) {
  return (
    <>
      <div className="voi-scroll" style={{ maxHeight: 200, minHeight: 100 }}>
        {groupByType(props.structures).map(([title, members]) => (
          <div className="subgroup" key={title}>
            <div className="sg-title" style={{ background: "var(--base)" }}>
              {title}
            </div>
            <div className="voi-grid">
              {members.map((s) => {
                const rgb = props.colors[s.index] ?? s.color;
                return (
                  <div className="voi-row" key={s.index} title={s.name}>
                    <label className="swatch swatch-button" style={{ background: `rgb(${rgb.join(",")})` }} title={`Pick color for ${s.name}`}>
                      <input
                        type="color"
                        style={{ opacity: 0, width: 0, height: 0, position: "absolute" }}
                        value={`#${rgb.map((c) => c.toString(16).padStart(2, "0")).join("")}`}
                        onChange={(e) => {
                          const h = e.target.value;
                          props.onColor(s.index, [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16)) as Vec3);
                        }}
                      />
                    </label>
                    <label className="check">
                      <input
                        type="checkbox"
                        checked={props.checked[s.index] ?? false}
                        onChange={(e) => props.onToggle(s.index, e.target.checked)}
                        data-testid={`analysis-voi-${s.name}`}
                      />
                      {truncateName(s.name)}
                    </label>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>
      <div className="row">
        <button className="btn small" style={{ width: 48 }} onClick={() => props.onAll(true)}>
          All
        </button>
        <button className="btn small" style={{ width: 48 }} onClick={() => props.onAll(false)}>
          None
        </button>
      </div>
    </>
  );
}

/** Qt "DVH / QI Analysis" window (non-modal): DVH plot, QI table, QI points, gamma tab. */
export function AnalysisWindow(props: { onClose: () => void }) {
  const state = useApp((s) => s.state);
  const groups = useMemo<ResultGroup[]>(() => resultGroups(state), [state]);
  const structures = state?.structures ?? [];
  const initial = useRef({
    quantity: useApp.getState().viewer.activeQuantity,
    visible: useApp.getState().viewer.visible,
  });
  const [tab, setTab] = useState<"dvh" | "gamma">("dvh");
  const [primary, setPrimary] = useState<string | null>(initial.current.quantity ?? state?.results?.default ?? null);
  const [secondary, setSecondary] = useState<string | null>(null);
  const [allScenarios, setAllScenarios] = useState(false);
  const [checked, setChecked] = useState<Record<number, boolean>>(() => {
    const v = initial.current.visible;
    return Object.fromEntries(structures.map((s) => [s.index, v[s.index] ?? false]));
  });
  const [colors, setColors] = useState<Record<number, Vec3>>({});
  const [refVolsText, setRefVolsText] = useState(DEFAULT_REF_VOLS);
  const [refDosesByQuantity, setRefDosesByQuantity] = useState<Record<string, string>>({});
  const [refDosesText, setRefDosesText] = useState("");
  const [showKinds, setShowKinds] = useState<Set<string>>(new Set());
  const [metricChecks, setMetricChecks] = useState<Record<string, boolean>>({});
  const [sets, setSets] = useState<DvhSet[]>([]);
  const [qi, setQi] = useState<QiResult | null>(null);
  const [shared, setShared] = useState<{ shareable: boolean; scale: number | null }>({ shareable: true, scale: 1 });
  const [highlight, setHighlight] = useState<{ structure: number | null; metric: string | null }>({ structure: null, metric: null });
  const [pointRequest, setPointRequest] = useState<PointRequest | null>(null);
  const [pendingHighlight, setPendingHighlight] = useState<{ structure: number | null; kind: "dx" | "vx"; parameter: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [plotMenu, setPlotMenu] = useState<{ x: number; y: number } | null>(null);
  const [pos, setPos] = useState({ x: Math.max(16, window.innerWidth / 2 - 500), y: 40 });
  const plotCanvas = useRef<HTMLCanvasElement | null>(null);

  const selectedVois = structures.map((s) => s.index).filter((i) => checked[i]);
  const primaryGroup = locateKey(groups, primary)?.group;
  const scenarioKeys = primaryGroup?.components.filter((c) => c.kind === "scenario").map((c) => c.key) ?? [];
  const effectiveSecondary = secondary && secondary !== primary && locateKey(groups, secondary) ? secondary : null;
  const unitSymbol = qi?.unit_symbol ?? primaryGroup?.unit ?? "";

  // When the shown quantity changes, the V_x input switches to that quantity's own text.
  const shownRef = useRef<string | null>(null);
  useEffect(() => {
    if (shownRef.current === primary) return;
    shownRef.current = primary;
    setRefDosesText(primary ? (refDosesByQuantity[primary] ?? "") : "");
  }, [primary, refDosesByQuantity]);

  // DVH curves (primary, compare, scenario band).
  useEffect(() => {
    if (!primary) {
      setSets([]);
      return;
    }
    const keys = [primary, ...(effectiveSecondary ? [effectiveSecondary] : []), ...(allScenarios ? scenarioKeys : [])];
    let cancelled = false;
    fetchDvhSets([...new Set(keys)])
      .then((s) => !cancelled && setSets(s))
      .catch((err: unknown) => !cancelled && setError(err instanceof Error ? err.message : String(err)));
    if (effectiveSecondary && hasMethod("analysis.compare_units")) {
      request<{ shareable: boolean; scale: number | null }>("analysis.compare_units", { a: primary, b: effectiveSecondary })
        .then((r) => !cancelled && setShared(r))
        .catch(() => !cancelled && setShared({ shareable: true, scale: 1 }));
    }
    return () => {
      cancelled = true;
    };
  }, [primary, effectiveSecondary, allScenarios, scenarioKeys.join("|"), state?.revisions?.workspace.result]);

  // QI table of the primary quantity for the checked structures.
  const refVols = parseRefVols(refVolsText);
  const refDoses = parseRefDoses(refDosesText);
  useEffect(() => {
    if (!primary || !selectedVois.length || !hasMethod("analysis.qi")) {
      setQi(null);
      return;
    }
    let cancelled = false;
    request<QiResult>("analysis.qi", {
      quantity: primary,
      names: selectedVois,
      ...(refVols ? { ref_vols: refVols } : {}),
      ...(refDoses ? { ref_doses: refDoses } : {}),
    })
      .then((r) => {
        if (cancelled) return;
        setQi(r);
        // Write the parameters actually used back into the inputs (Qt `_sync_param_fields`).
        setRefVolsText(formatNumbers(r.ref_vols));
        setRefDosesText(formatNumbers(r.ref_doses));
      })
      .catch((err: unknown) => !cancelled && setError(err instanceof Error ? err.message : String(err)));
    return () => {
      cancelled = true;
    };
  }, [primary, selectedVois.join("|"), JSON.stringify(refVols), JSON.stringify(refDoses), state?.revisions?.workspace.result]);

  const colorOf = (index: number, fallback: readonly number[]) => colors[index] ?? fallback;
  const primarySet = sets.find((s) => s.key === primary);
  const secondarySet = effectiveSecondary ? sets.find((s) => s.key === effectiveSecondary) : undefined;
  const curves: PlotCurve[] = [];
  for (const c of primarySet?.curves ?? []) {
    if (!checked[c.index]) continue;
    curves.push({ index: c.index, name: c.name, x: c.dose.data, y: c.volume.data, color: colorOf(c.index, c.color), style: "solid", secondaryAxis: false });
  }
  for (const c of secondarySet?.curves ?? []) {
    if (!checked[c.index]) continue;
    const x = shared.shareable && shared.scale !== null && shared.scale !== 1 ? Float32Array.from(c.dose.data, (v) => v * shared.scale!) : c.dose.data;
    curves.push({ index: c.index, name: c.name, x, y: c.volume.data, color: colorOf(c.index, c.color), style: "dashed", secondaryAxis: !shared.shareable });
  }
  const bands: PlotCurve[] = allScenarios
    ? sets
        .filter((s) => scenarioKeys.includes(s.key))
        .flatMap((s) =>
          s.curves.filter((c) => checked[c.index]).map((c) => ({ index: c.index, name: c.name, x: c.dose.data, y: c.volume.data, color: colorOf(c.index, c.color), style: "solid" as const, secondaryAxis: false })),
        )
    : [];
  const shown = [...new Map(curves.map((c) => [c.index, c])).values()];
  const legend: LegendEntry[] = shown.map((c) => ({ label: c.name, color: c.color, style: "solid" }));
  const primaryName = primary ? keyLabel(groups, primary) : "";
  const secondaryName = effectiveSecondary ? keyLabel(groups, effectiveSecondary) : "";
  if (secondarySet && shown.length) {
    legend.push({ label: primaryName || "Primary", color: null, style: "solid" });
    legend.push({ label: secondaryName || "Secondary", color: null, style: "dashed" });
  }
  const colorMap = Object.fromEntries(structures.map((s) => [s.index, colorOf(s.index, s.color)]));
  const markers = qiMarkers(qi, (primarySet?.curves ?? []).filter((c) => checked[c.index]).map((c) => c.index), colorMap);
  const xLabel = axisLabel(primarySet?.label ?? primaryGroup?.meta_label ?? "", primarySet?.unit ?? primaryGroup?.unit ?? "");
  const secondaryXLabel = secondarySet && !shared.shareable ? axisLabel(secondarySet.label, secondarySet.unit) : null;

  const metricIds = qi?.metrics.map((m) => m.id) ?? [];
  const shownMetrics = metricIds.filter((id) => !FIXED_METRICS.includes(id) || (metricChecks[id] ?? true));
  const tableRows = qiTableRows(qi, selectedVois, shownMetrics);
  const tableStructures = tableRows.length ? (qi?.rows ?? []).filter((r) => selectedVois.includes(r.index)).map((r) => r.index) : [];

  const applyPointChoice = (choice: { kind: "dx" | "vx"; parameter: number; replace: boolean }, req: PointRequest) => {
    const replaceValue = choice.replace ? req.replaceValue : undefined;
    if (choice.kind === "dx") {
      setRefVolsText(formatNumbers(applyPoint(parseRefVols(refVolsText) ?? [], choice.parameter, replaceValue)));
    } else {
      const text = formatNumbers(applyPoint(parseRefDoses(refDosesText) ?? [], choice.parameter, replaceValue));
      setRefDosesText(text);
      if (primary) setRefDosesByQuantity((m) => ({ ...m, [primary]: text }));
    }
    setShowKinds((k) => new Set([...k, choice.kind]));
    setPendingHighlight({ structure: req.structure ?? highlight.structure, kind: choice.kind, parameter: choice.parameter });
  };

  // After the table is recomputed, highlight the new point (`_metric_for`).
  useEffect(() => {
    if (!pendingHighlight || !qi) return;
    const metric = qi.metrics.find(
      (m) => m.kind === pendingHighlight.kind && m.parameter !== undefined && Math.abs(m.parameter - pendingHighlight.parameter) < 1e-6,
    );
    if (!metric) return;
    setHighlight({ structure: pendingHighlight.structure, metric: metric.id });
    setPendingHighlight(null);
  }, [qi, pendingHighlight]);

  // Drag the window by its title bar.
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

  return (
    <div className="analysis-window" style={{ left: pos.x, top: pos.y }} data-testid="analysis-window" role="dialog" aria-label="DVH / QI Analysis">
      <div className="analysis-title" onPointerDown={onTitleDown}>
        <span>DVH / QI Analysis</span>
        <button className="btn small" onClick={props.onClose} onPointerDown={(e) => e.stopPropagation()} data-testid="analysis-close">
          ✕
        </button>
      </div>
      <div className="tabs">
        <button className={`tab ${tab === "dvh" ? "active" : ""}`} onClick={() => setTab("dvh")}>
          DVH / QI
        </button>
        <button className={`tab ${tab === "gamma" ? "active" : ""}`} onClick={() => setTab("gamma")} data-testid="analysis-tab-gamma">
          Gamma Analysis
        </button>
      </div>
      {tab === "gamma" ? (
        <div className="analysis-body" data-testid="gamma-tab">
          <div>Gamma analysis (not implemented).</div>
          <div>(Updated as soon as Viewer supports dose comparison)</div>
        </div>
      ) : (
        <div className="analysis-body">
          <div className="analysis-controls">
            <fieldset className="qt-group" style={{ flex: 1.5, minWidth: 300 }}>
              <legend>Quantities</legend>
              <div className="form-grid" style={{ gridTemplateColumns: "auto minmax(0, 1fr)" }}>
                <label>Primary:</label>
                <QuantitySelector
                  groups={groups}
                  value={primary}
                  onChange={setPrimary}
                  testId="analysis-primary"
                  title="Quantity shown in the DVH plot (solid) and the QI table"
                />
                <label>Compare:</label>
                <QuantitySelector
                  groups={groups}
                  value={secondary}
                  onChange={setSecondary}
                  noneLabel={NONE_LABEL}
                  testId="analysis-compare"
                  title="Optional second quantity plotted as dashed DVH curves; incompatible units get a second x axis"
                />
              </div>
              {scenarioKeys.length > 0 && (
                <label className="check" title="Draw the primary quantity's DVH of every uncertainty scenario as a thin curve behind the selected one">
                  <input type="checkbox" checked={allScenarios} onChange={(e) => setAllScenarios(e.target.checked)} data-testid="analysis-all-scenarios" />
                  All scenarios
                </label>
              )}
            </fieldset>
            <fieldset className="qt-group" style={{ flex: 2, minWidth: 320 }}>
              <legend>Structures</legend>
              <StructureList
                structures={structures}
                checked={checked}
                colors={colors}
                onToggle={(n, v) => setChecked((c) => ({ ...c, [n]: v }))}
                onColor={(n, rgb) => setColors((c) => ({ ...c, [n]: rgb }))}
                onAll={(v) => setChecked(Object.fromEntries(structures.map((s) => [s.index, v])))}
              />
            </fieldset>
            <fieldset className="qt-group" style={{ flex: 1 }}>
              <legend>QI parameters</legend>
              <div className="form-grid" style={{ gridTemplateColumns: "auto minmax(0, 1fr) auto" }}>
                <label>D_x [%]:</label>
                <input
                  type="text"
                  value={refVolsText}
                  title={"Reference volumes in % for D_x metrics, e.g. '2 50 98'.\nInvalid input falls back to the defaults."}
                  onChange={(e) => setRefVolsText(e.target.value)}
                  data-testid="analysis-ref-vols"
                />
                <ShowInDvh kind="dx" kinds={showKinds} setKinds={setShowKinds} />
                <label>{unitSymbol ? `V_x [${unitSymbol}]:` : "V_x:"}</label>
                <input
                  type="text"
                  value={refDosesText}
                  title={"Reference doses for V_x metrics, e.g. '10 20 30'.\nClear the field to re-derive five values from the maximum."}
                  onChange={(e) => {
                    setRefDosesText(e.target.value);
                    if (primary) setRefDosesByQuantity((m) => ({ ...m, [primary]: e.target.value }));
                  }}
                  data-testid="analysis-ref-doses"
                />
                <ShowInDvh kind="vx" kinds={showKinds} setKinds={setShowKinds} />
              </div>
              <div className="voi-scroll" style={{ minHeight: 40, marginTop: 4 }}>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "2px 8px" }}>
                  {metricIds
                    .filter((id) => FIXED_METRICS.includes(id))
                    .map((id) => (
                      <label className="check" key={id}>
                        <input
                          type="checkbox"
                          checked={metricChecks[id] ?? true}
                          onChange={(e) => setMetricChecks((m) => ({ ...m, [id]: e.target.checked }))}
                          data-testid={`analysis-metric-${id}`}
                        />
                        {id}
                      </label>
                    ))}
                </div>
              </div>
              <div className="row" style={{ marginTop: 4 }}>
                <button className="btn small" style={{ width: 48 }} onClick={() => setMetricChecks(Object.fromEntries(FIXED_METRICS.map((m) => [m, true])))}>
                  All
                </button>
                <button className="btn small" style={{ width: 48 }} onClick={() => setMetricChecks(Object.fromEntries(FIXED_METRICS.map((m) => [m, false])))}>
                  None
                </button>
              </div>
            </fieldset>
          </div>
          {error && <div style={{ color: "var(--danger)" }}>{error}</div>}
          <div
            className="analysis-plot"
            onContextMenu={(e) => {
              e.preventDefault();
              const r = e.currentTarget.getBoundingClientRect();
              setPlotMenu({ x: e.clientX - r.left, y: e.clientY - r.top });
            }}
            onPointerDown={() => setPlotMenu(null)}
          >
            <DvhPlot
              canvasRef={plotCanvas}
              curves={curves}
              bands={bands}
              xLabel={xLabel}
              secondaryXLabel={secondaryXLabel}
              legend={legend}
              dashedName={secondaryName}
              markers={markers}
              visibleKinds={showKinds}
              highlight={highlight}
              onStructureClick={(n) => setHighlight({ structure: n, metric: null })}
              onMarkerClick={(s, m) => setHighlight({ structure: s, metric: m })}
              onPointRequested={(dose, volume) => setPointRequest({ dose, volume })}
              onParameterMoved={(structure, metric, parameter) => {
                const m = markers.find((x) => x.structure === structure && x.metric === metric);
                if (!m) return;
                setPointRequest({
                  dose: m.kind === "dx" ? 0 : parameter,
                  volume: m.kind === "dx" ? parameter : 0,
                  kind: m.kind,
                  replace: metric,
                  replaceValue: m.parameter,
                  structure,
                });
              }}
            />
            {plotMenu && (
              <div className="menu-popup" style={{ left: plotMenu.x, top: plotMenu.y }} onPointerDown={(e) => e.stopPropagation()}>
                <button className="item" onClick={() => setPlotMenu(null)}>
                  View All
                </button>
                <button
                  className="item"
                  onClick={() => {
                    setPlotMenu(null);
                    plotCanvas.current?.toBlob((b) => b && download("dvh.png", "image/png", b));
                  }}
                >
                  Export PNG…
                </button>
                <button
                  className="item"
                  onClick={() => {
                    setPlotMenu(null);
                    download("dvh.csv", "text/csv", curvesCsv(curves));
                  }}
                >
                  Export CSV…
                </button>
              </div>
            )}
          </div>
          <div className="analysis-table">
            <div className="table-wrap">
              <table className="qt qi-table" data-testid="qi-table">
                <thead>
                  <tr>
                    {(tableRows[0] ?? []).map((h) => (
                      <th key={h}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {tableRows.slice(1).map((row, r) => (
                    <tr key={tableStructures[r]}>
                      {row.map((cell, col) => {
                        const metric = col === 0 ? null : shownMetrics[col - 1];
                        const selected = highlight.structure === tableStructures[r] && (col === 0 ? highlight.metric === null : highlight.metric === metric);
                        return (
                          <td
                            key={col}
                            className={`${col ? "num" : ""} ${selected ? "selected" : ""}`}
                            onClick={() => setHighlight({ structure: tableStructures[r], metric })}
                          >
                            {col === 0 && <span className="swatch" style={{ background: `rgb(${(colorMap[tableStructures[r]] ?? [128, 128, 128]).join(",")})`, marginRight: 4 }} />}
                            {cell}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="row" style={{ justifyContent: "flex-end", marginTop: 4 }}>
              <button
                className="btn"
                disabled={!tableRows.length}
                title="Copy the table to the clipboard (tab-separated)"
                onClick={() => void navigator.clipboard?.writeText(toTsv(tableRows))}
                data-testid="qi-copy"
              >
                Copy
              </button>
              <button
                className="btn"
                disabled={!tableRows.length}
                title="Save the table as a CSV file"
                onClick={() => download("quality_indicators.csv", "text/csv", toCsv(tableRows))}
                data-testid="qi-export"
              >
                Export CSV…
              </button>
            </div>
          </div>
        </div>
      )}
      {pointRequest && (
        <QiPointDialog
          req={pointRequest}
          unit={unitSymbol}
          onDone={(choice) => {
            const req = pointRequest;
            setPointRequest(null);
            if (choice) applyPointChoice(choice, req);
            else setHighlight((h) => ({ ...h }));
          }}
        />
      )}
    </div>
  );
}

function ShowInDvh(props: { kind: "dx" | "vx"; kinds: Set<string>; setKinds: (k: Set<string>) => void }) {
  const name = props.kind === "dx" ? "D_x" : "V_x";
  return (
    <label
      className="check"
      title={`Mark the ${name} points of the table on the DVH curves.\nSelect a D_x / V_x cell or click a marker to show its value and parameter lines;\ndrag the dashed parameter line or double-click the plot to add a point.`}
    >
      <input
        type="checkbox"
        checked={props.kinds.has(props.kind)}
        onChange={(e) => {
          const next = new Set(props.kinds);
          if (e.target.checked) next.add(props.kind);
          else next.delete(props.kind);
          props.setKinds(next);
        }}
        data-testid={`analysis-show-${props.kind}`}
      />
      Show in DVH
    </label>
  );
}

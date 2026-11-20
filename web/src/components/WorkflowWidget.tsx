import { useRef } from "react";
import { featureStatus, type StageStatus } from "../api";
import { actions, useApp, type Progress } from "../store/appStore";
import { chooseAndLoadDij, loadFile, snapshotResult } from "../store/ioActions";
import { resolvePendingChanges } from "../store/pendingChanges";
import { loadAccept } from "./io/capabilities";
import { statusWithFurtherResults } from "./io/models";
import { ProgressBar } from "./common";

const DOT_TOOLTIP: Record<StageStatus, string> = {
  missing: "Not yet computed",
  done: "Up to date",
  stale: "Outdated — inputs changed since this was computed",
};

function StageDot(props: { status: StageStatus; testId: string }) {
  return (
    <span className={`dot ${props.status}`} title={DOT_TOOLTIP[props.status]} data-testid={props.testId} data-status={props.status}>
      {props.status === "missing" ? "○" : "●"}
    </span>
  );
}

function worst(...statuses: StageStatus[]): StageStatus {
  if (statuses.includes("stale")) return "stale";
  if (statuses.every((s) => s === "done")) return "done";
  return statuses.includes("done") ? "stale" : "missing";
}

/** Status line while busy: the progress label, then its detail and fraction (Qt shows the progress levels there). */
export function busyStatusText(busy: string, progress: Progress | null): string {
  if (!progress?.label) return `${busy}…`;
  const parts = [progress.label];
  if (progress.detail) parts.push(progress.detail);
  if (progress.fraction !== null && progress.fraction !== undefined) parts.push(`${Math.round(progress.fraction * 100)} %`);
  return parts.length > 1 ? parts.join(" · ") : `${progress.label}…`;
}

const SNAPSHOT_TIP =
  "Keep a copy of the current result in the workspace. Use File > Save Result… to write it to disk.";

/**
 * Mirror of `pyRadPlan.gui.widgets.workflow.WorkflowWidget`: status row, busy bar and the stage
 * grid with the Qt buttons in the Qt places. Phantoms are opened from File > Open Example Phantom,
 * as in Qt.
 */
export function WorkflowWidget() {
  const state = useApp((s) => s.state);
  const info = useApp((s) => s.info);
  const busy = useApp((s) => s.busy);
  const progress = useApp((s) => s.progress);
  const backend = useApp((s) => s.backend);
  const fileInput = useRef<HTMLInputElement>(null);

  const connected = state !== null;
  const idle = connected && busy === null;
  const can = (key: string) => idle && Boolean(state?.actions[key]);
  const hasMethod = (m: string) => !info?.methods || info.methods.includes(m);

  const stages = state?.stages;
  const loadStage: StageStatus = stages ? worst(stages.data) : "missing";
  const dijStage: StageStatus = stages ? (stages.dij === "missing" ? "missing" : worst(stages.stf, stages.dij)) : "missing";
  const resultStage: StageStatus = stages ? stages.optimization : "missing";

  const statusText = busy ? busyStatusText(busy, progress) : statusWithFurtherResults(state?.status ?? "Connecting…", state?.results?.tags);

  const dicom = featureStatus(info, "dicom_import");
  const imageIo = featureStatus(info, "image_file_io");
  const folderReason = !dicom.available
    ? dicom.reason
    : !imageIo.available
      ? imageIo.reason
      : "Folder import is not available in the web interface.";
  const dijSupported = hasMethod("data.load_dij");
  const dijEnabled = idle && dijSupported && state?.actions.load_dij !== false;
  const snapshotEnabled = idle && hasMethod("workflow.snapshot") && Boolean(state?.actions.snapshot);

  /** Qt resolves unapplied plan edits before these steps (optimize and recalc outdate the dij). */
  const guarded = (step: string, outdatesInput: boolean, run: () => unknown) => async () => {
    if (await resolvePendingChanges(step, outdatesInput)) await run();
  };

  return (
    <div className="workflow" data-testid="workflow">
      <div className="status-row">
        <span>Status:</span>
        <span className="status-text" data-testid="workflow-status" title={statusText}>
          {statusText}
        </span>
        <button className="btn" disabled={!idle} onClick={() => void actions.refresh()}>
          Refresh
        </button>
        <button
          className="btn"
          disabled={busy === null || !backend?.canCancel}
          title={
            backend?.canCancel
              ? "Stop the running calculation.\nAn optimization stops after its current iteration and keeps the last iterate (Stop again discards it).\nOther calculations are discarded at the next Python bytecode check."
              : "Stopping needs SharedArrayBuffer (cross-origin isolation)"
          }
          onClick={() => actions.stop()}
          data-testid="btn-stop"
        >
          Stop
        </button>
      </div>
      {busy !== null && <ProgressBar fraction={progress?.fraction ?? null} testId="workflow-progress" />}

      <div className="wf-grid">
        <div className="wf-head">
          <StageDot status={loadStage} testId="dot-load" />
          Loading
        </div>
        <div className="wf-head">
          <StageDot status={dijStage} testId="dot-dij" />
          Dose Influence
        </div>
        <div className="wf-head">
          <StageDot status={resultStage} testId="dot-result" />
          Result Computation
        </div>
        <div className="wf-head">Export / Save</div>

        <button className="btn" disabled={!idle} onClick={() => fileInput.current?.click()} data-testid="btn-load-file">
          Load File
        </button>
        <button
          className="btn"
          disabled={!can("calc_dose_influence")}
          onClick={() => void guarded("Calculate Dose Influence", false, () => actions.calcDoseInfluence())()}
          data-testid="btn-dij"
        >
          Calc. Dose Influence
        </button>
        <button
          className="btn"
          disabled={!can("optimize")}
          onClick={() => void guarded("Optimize", true, () => actions.optimize())()}
          data-testid="btn-optimize"
        >
          Optimize
        </button>
        <button
          className="btn"
          disabled={!snapshotEnabled}
          title={SNAPSHOT_TIP}
          onClick={() => void snapshotResult()}
          data-testid="btn-snapshot"
        >
          Snapshot Result
        </button>

        <button
          className="btn"
          disabled
          title={`Import a folder of DICOM data, or a folder of image files (CT + binary structure masks)\n\nNot available: ${folderReason}`}
          data-testid="btn-load-folder"
        >
          Load Folder
        </button>
        <button
          className="btn"
          disabled={!dijEnabled}
          title={dijSupported ? "Load a precomputed dose-influence matrix (.mat)" : "This backend cannot load a dose-influence matrix."}
          onClick={() => void chooseAndLoadDij()}
          data-testid="btn-load-dij"
        >
          Load Dij
        </button>
        <button
          className="btn"
          disabled={!can("recalc") || !hasMethod("workflow.recalc")}
          title="Recalculate the dose of the current weights with the current plan"
          onClick={() => void guarded("Recalculate Dose", true, () => actions.recalc())()}
          data-testid="btn-recalc"
        >
          Recalculate Dose
        </button>
        <span />

        <button className="btn" disabled title="Not yet implemented" data-testid="btn-import-binary">
          Import from Binary
        </button>
        <span />
        <span />
        <span />
      </div>

      <input
        ref={fileInput}
        type="file"
        hidden
        accept={loadAccept(info)}
        data-testid="file-input"
        onChange={(e) => {
          const file = e.currentTarget.files?.[0];
          e.currentTarget.value = "";
          if (file) void loadFile(file);
        }}
      />
    </div>
  );
}

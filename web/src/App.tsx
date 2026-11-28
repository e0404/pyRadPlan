import { useState } from "react";
import { Dialog, GroupBox, useLayoutMode, type LayoutMode } from "./components/common";
import { ABOUT } from "./components/help/notice";
import { NoticeGate, openNoticeDialog } from "./components/help/NoticeDialog";
import { MenuBar } from "./components/MenuBar";
import { InfoWidget, LogConsole, LogoBanner, ObjectivesPanel } from "./components/panels";
import { PlanWidget } from "./components/PlanWidget";
import { SliceViewer } from "./components/SliceViewer";
import { Splash } from "./components/Splash";
import { StructureVisibility } from "./components/StructureVisibility";
import { ViewerOptionsWidget } from "./components/ViewerOptionsWidget";
import { VisualizationWidget } from "./components/VisualizationWidget";
import { WorkflowWidget } from "./components/WorkflowWidget";
import { dismissConfirm, dismissError, useApp } from "./store/appStore";

function ErrorDialog() {
  const error = useApp((s) => s.error);
  const [details, setDetails] = useState(false);
  if (!error) return null;
  return (
    <Dialog
      title={error.title}
      onClose={dismissError}
      testId="error-dialog"
      buttons={
        <>
          {error.details && (
            <button className="btn" onClick={() => setDetails(!details)}>
              {details ? "Hide Details" : "Show Details…"}
            </button>
          )}
          <button className="btn primary" onClick={dismissError} autoFocus>
            OK
          </button>
        </>
      }
    >
      <div className="icon-row">
        <div className="icon">!</div>
        <div style={{ whiteSpace: "pre-wrap" }}>{error.message}</div>
      </div>
      {details && error.details && <pre>{error.details}</pre>}
    </Dialog>
  );
}

function ConfirmDialog() {
  const confirm = useApp((s) => s.confirm);
  if (!confirm) return null;
  return (
    <Dialog
      title={confirm.title}
      onClose={dismissConfirm}
      testId="confirm-dialog"
      buttons={
        <>
          <button className="btn" onClick={dismissConfirm}>
            Cancel
          </button>
          <button className="btn primary" onClick={confirm.onConfirm} autoFocus>
            {confirm.confirmLabel}
          </button>
        </>
      }
    >
      <div style={{ whiteSpace: "pre-wrap" }}>{confirm.message}</div>
    </Dialog>
  );
}

type PanelId = "workflow" | "plan" | "objectives" | "log" | "visualization" | "viewerOptions" | "structures" | "info";

// Phones open with only the workflow (and the never-collapsible viewer) expanded.
const PHONE_COLLAPSED: PanelId[] = ["plan", "objectives", "log", "visualization", "viewerOptions", "structures", "info"];

interface FoldState {
  layout: LayoutMode;
  collapsed: ReadonlySet<PanelId>;
  configCollapsed: boolean;
}

function initialFold(layout: LayoutMode): FoldState {
  return { layout, collapsed: new Set(layout === "phone" ? PHONE_COLLAPSED : []), configCollapsed: false };
}

export function App() {
  const statusMessage = useApp((s) => s.statusMessage);
  const showLog = useApp((s) => s.showLogPanel);
  const kind = useApp((s) => s.backend?.kind);
  const layout = useLayoutMode();
  const [fold, setFold] = useState(() => initialFold(layout));
  if (fold.layout !== layout) setFold(initialFold(layout));
  const panel = (id: PanelId) =>
    layout === "desktop"
      ? {}
      : {
          collapsed: fold.collapsed.has(id),
          onToggle: () =>
            setFold((f) => {
              const collapsed = new Set(f.collapsed);
              if (!collapsed.delete(id)) collapsed.add(id);
              return { ...f, collapsed };
            }),
        };
  const configCollapsed = layout === "tablet" && fold.configCollapsed;
  return (
    <div className="main-window" data-layout={layout}>
      {kind === "mock" && (
        <div className="mock-banner" role="alert" data-testid="mock-banner">
          MOCK BACKEND: all plans, doses and results are invented for interface development and are not calculated by pyRadPlan. Do not use them for anything.
        </div>
      )}
      <MenuBar />
      <div className="columns" data-testid="main-columns">
        <div className={`column config ${configCollapsed ? "collapsed" : ""}`} data-testid="column-config">
          {layout === "tablet" && (
            <button
              type="button"
              className="config-toggle"
              aria-expanded={!configCollapsed}
              onClick={() => setFold((f) => ({ ...f, configCollapsed: !f.configCollapsed }))}
              data-testid="config-toggle"
            >
              <span className="gb-chevron" aria-hidden="true" />
              Configuration
            </button>
          )}
          <GroupBox title="Workflow" testId="panel-workflow" className="p-workflow" {...panel("workflow")}>
            <WorkflowWidget />
          </GroupBox>
          <GroupBox title="Plan" testId="panel-plan" className="p-plan" {...panel("plan")}>
            <PlanWidget />
          </GroupBox>
          <GroupBox title="Objectives & constraints" testId="panel-objectives" className="grow2 p-objectives" {...panel("objectives")}>
            <ObjectivesPanel />
          </GroupBox>
          {showLog && (
            <GroupBox title="Log" testId="panel-log" className="grow p-log" {...panel("log")}>
              <LogConsole />
            </GroupBox>
          )}
        </div>
        <div className="column view">
          <LogoBanner />
          <GroupBox title="Slice Viewer" testId="panel-viewer" className="grow p-viewer">
            <SliceViewer />
          </GroupBox>
          <GroupBox title="Visualization" testId="panel-visualization" className="p-visualization" {...panel("visualization")}>
            <VisualizationWidget />
          </GroupBox>
        </div>
        <div className="column side">
          <GroupBox title="Viewer Options" testId="panel-viewer-options" className="p-viewer-options" {...panel("viewerOptions")}>
            <ViewerOptionsWidget />
          </GroupBox>
          <GroupBox title="Structure Visibility" testId="panel-structures" className="grow p-structures" {...panel("structures")}>
            <StructureVisibility />
          </GroupBox>
          <GroupBox title="Info" testId="panel-info" className="p-info" {...panel("info")}>
            <InfoWidget />
          </GroupBox>
        </div>
      </div>
      <div className="statusbar" data-testid="statusbar">
        <span>{statusMessage}</span>
        <span className="spacer" />
        <button type="button" className="status-notice" onClick={() => void openNoticeDialog()} data-testid="status-notice" title="Disclaimer and privacy">
          {ABOUT.disclaimer_title} · data stays in this browser
        </button>
        <span className="status-env">{kind === "mock" ? "mock backend" : kind === "pyodide" ? "Pyodide backend" : ""}</span>
        <span className="status-env">{globalThis.crossOriginIsolated ? "cross-origin isolated" : "not isolated"}</span>
      </div>
      <Splash />
      <ConfirmDialog />
      <ErrorDialog />
      <NoticeGate />
    </div>
  );
}

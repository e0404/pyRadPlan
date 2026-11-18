import { useState } from "react";
import { GroupBox, Dialog } from "./components/common";
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

export function App() {
  const statusMessage = useApp((s) => s.statusMessage);
  const showLog = useApp((s) => s.showLogPanel);
  const kind = useApp((s) => s.backend?.kind);
  return (
    <div className="main-window">
      <MenuBar />
      <div className="columns">
        <div className="column">
          <GroupBox title="Workflow" testId="panel-workflow">
            <WorkflowWidget />
          </GroupBox>
          <GroupBox title="Plan" testId="panel-plan">
            <PlanWidget />
          </GroupBox>
          <GroupBox title="Objectives & constraints" className="grow2">
            <ObjectivesPanel />
          </GroupBox>
          {showLog && (
            <GroupBox title="Log" className="grow">
              <LogConsole />
            </GroupBox>
          )}
        </div>
        <div className="column">
          <LogoBanner />
          <GroupBox title="Slice Viewer" className="grow">
            <SliceViewer />
          </GroupBox>
          <GroupBox title="Visualization">
            <VisualizationWidget />
          </GroupBox>
        </div>
        <div className="column">
          <GroupBox title="Viewer Options">
            <ViewerOptionsWidget />
          </GroupBox>
          <GroupBox title="Structure Visibility" className="grow">
            <StructureVisibility />
          </GroupBox>
          <GroupBox title="Info">
            <InfoWidget />
          </GroupBox>
        </div>
      </div>
      <div className="statusbar" data-testid="statusbar">
        <span>{statusMessage}</span>
        <span className="spacer" />
        <span>{kind === "mock" ? "mock backend" : kind === "pyodide" ? "Pyodide backend" : ""}</span>
        <span>{globalThis.crossOriginIsolated ? "cross-origin isolated" : "not isolated"}</span>
      </div>
      <Splash />
      <ConfirmDialog />
      <ErrorDialog />
    </div>
  );
}

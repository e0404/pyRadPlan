import { useEffect, useRef, useState } from "react";
import pyradplanLogo from "../assets/pyradplan_logo_full_landscape.svg";
import { actions, useApp } from "../store/appStore";
import { ProgressBar } from "./common";
import { SHORT_NOTICE } from "./help/notice";
import { patientFromUrl, splashPhase } from "./io/models";
import "./io/io.css";

/** Startup splash (frameless card with logo, status line, `n/N` counter and progress bar). */
export function Splash() {
  const boot = useApp((s) => s.boot);
  const kind = useApp((s) => s.backend?.kind);
  const connected = useApp((s) => s.state !== null);
  const info = useApp((s) => s.info);
  const patient = useRef(patientFromUrl()).current;
  const [patientState, setPatientState] = useState<"waiting" | "loading" | "done">(patient ? "waiting" : "done");

  useEffect(() => {
    if (!patient || patientState !== "waiting" || !connected) return;
    setPatientState("loading");
    const name = info?.phantoms.find((p) => p.toLowerCase() === patient.toLowerCase()) ?? patient;
    void actions.loadPhantom(name).finally(() => setPatientState("done"));
  }, [patient, patientState, connected, info]);

  const failed = boot.phase === "failed";
  const open = !failed && connected && patientState === "done";
  if (open) return null;
  const phase = splashPhase(boot, connected, patient, patientState !== "done");
  return (
    <div className="splash" data-testid="boot-splash">
      <div className={`qt-splash ${failed ? "failed" : ""}`}>
        <img src={pyradplanLogo} alt="pyRadPlan" className="splash-logo" />
        {failed ? (
          <>
            <div className="splash-status failed" data-testid="splash-status">
              {boot.message}
            </div>
            <div className="dim">The Python backend could not be started. Use ?backend=mock to try the interface without it.</div>
          </>
        ) : (
          <>
            <div className="splash-line">
              <span data-testid="splash-status">{phase.message}</span>
              <span data-testid="splash-counter">
                {phase.step}/{phase.total}
              </span>
            </div>
            <ProgressBar fraction={phase.fraction} testId="splash-progress" />
            <div className="dim splash-detail msg" data-testid="splash-detail">
              {kind === "pyodide" && boot.phase === "starting" ? boot.message : ""}
            </div>
          </>
        )}
        <div className="splash-notice" data-testid="splash-notice">
          {SHORT_NOTICE}
        </div>
      </div>
    </div>
  );
}

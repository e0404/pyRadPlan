import dkfzLogo from "../assets/dkfz_logo_blue.svg";
import dkfzLogoWhite from "../assets/dkfz_logo_white.svg";
import pyradplanLogo from "../assets/pyradplan_logo_full_landscape.svg";
import { useApp } from "../store/appStore";
import { useTheme } from "../theme";
import { ABOUT } from "./help/notice";

export { LogConsole } from "./log/LogConsole";
export { ObjectivesPanel } from "./objectives/ObjectivesPanel";

export function LogoBanner() {
  const dark = useTheme((s) => s.dark);
  return (
    <div className="logo-banner">
      <div className="left">
        <img src={dark ? dkfzLogoWhite : dkfzLogo} alt="DKFZ" style={{ height: 40 }} />
        <span className="disclaimer">NOT FOR CLINICAL USE!</span>
      </div>
      <img src={pyradplanLogo} alt="pyRadPlan" style={{ height: 64 }} />
    </div>
  );
}

export function InfoWidget() {
  const info = useApp((s) => s.info);
  const backend = useApp((s) => s.backend);
  return (
    <div className="info-widget" data-testid="info">
      <div style={{ fontWeight: "bold" }}>
        pyRadPlan v{info?.version ?? "…"} (Python {info?.python ?? "…"})
      </div>
      <div className="dim">
        {backend?.kind === "pyodide" ? "Pyodide" : "Mock"} backend
        {info && backend?.kind === "pyodide" ? ` · arrays: ${info.backend}` : ""}
      </div>
      <a href="https://github.com/e0404/pyRadPlan" target="_blank" rel="noreferrer">
        github.com/e0404/pyRadPlan
      </a>
      <div className="legal-links" data-testid="info-legal">
        <a href={ABOUT.imprint_url} target="_blank" rel="noreferrer">
          Imprint
        </a>
        {" · "}
        <a href={ABOUT.privacy_url} target="_blank" rel="noreferrer">
          Privacy policy
        </a>
      </div>
    </div>
  );
}

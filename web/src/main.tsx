import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { backendKindFromUrl, createBackend } from "./api";
import { App } from "./App";
import { noticeAcknowledged } from "./components/help/notice";
import { connect, log } from "./store/appStore";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

const kind = backendKindFromUrl();
// Nothing is downloaded (Pyodide, the wheels, jsDelivr) before the first-visit notice is acknowledged.
noticeAcknowledged()
  .then(() => {
    log("info", `Starting ${kind} backend${globalThis.crossOriginIsolated ? "" : " (not cross-origin isolated: Stop is unavailable)"}`);
    return createBackend(kind);
  })
  .then(connect)
  .catch((err: unknown) => log("error", `Backend failed: ${err instanceof Error ? err.message : String(err)}`));

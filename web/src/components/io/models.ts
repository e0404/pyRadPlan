/** Pure helpers behind the package-C components (kept free of React and the store for unit tests). */
import type { BootStatus } from "../../api";
import type { LogEntry } from "../../store/appStore";
import { LEVEL_RANK, type LogThreshold } from "../../store/ioStore";

// ---- splash ---------------------------------------------------------------

export interface SplashPhase {
  /** Status line in the wording of the Qt splash (`apps.py`). */
  message: string;
  step: number;
  total: number;
  fraction: number | null;
}

/**
 * Map the boot status to the Qt splash's phases (`windows/_splash_win.py`: status line, `n/N`
 * counter, progress bar). The real stage stays visible as a detail line.
 */
export function splashPhase(boot: BootStatus, ready: boolean, patient: string | null, patientLoading: boolean): SplashPhase {
  const total = patient ? 6 : 5;
  if (ready) {
    if (patient && patientLoading) return { message: `Loading patient ${patient}…`, step: 4, total, fraction: null };
    return { message: "Opening main window…", step: total, total, fraction: 1 };
  }
  if (boot.phase === "ready") return { message: "Building main window…", step: total - 1, total, fraction: 0.97 };
  const text = boot.message;
  let message = "Loading core…";
  let step = 2;
  if (/Python runtime|^Python [\d.]+ ready/i.test(text) || (boot.fraction ?? 0) < 0.05) {
    message = "Loading Python runtime…";
    step = 1;
  } else if (/pyRadPlan wheel|Importing pyRadPlan|pyRadPlan installed/i.test(text)) {
    message = "Loading pyRadPlan…";
    step = 3;
  }
  return { message, step, total, fraction: boot.fraction };
}

/** The `?phantom=<name>` URL parameter: load that patient on startup (Qt `gui(patient)`). */
export function patientFromUrl(search: string = globalThis.location?.search ?? ""): string | null {
  return new URLSearchParams(search).get("phantom");
}

// ---- workflow status ------------------------------------------------------

/** Qt `_update_status`: the "· N saved snapshots" suffix of the result status. */
export function statusWithSnapshots(status: string, tags: readonly string[] | undefined): string {
  const n = tags?.length ?? 0;
  if (!n || !status.startsWith("Optimization result available") || /snapshot/.test(status)) return status;
  return `${status} · ${n} saved snapshot${n === 1 ? "" : "s"}`;
}

// ---- snapshots ------------------------------------------------------------

/** `re.sub(r"\W+", "_", tag.strip()).strip("_")`. */
export function sanitizeTag(tag: string): string {
  return tag.trim().replace(/[^\p{L}\p{N}_]+/gu, "_").replace(/^_+|_+$/g, "");
}

export const RESERVED_TAGS = ["beam", "scenario"] as const;

// ---- log console ----------------------------------------------------------

const pad = (n: number) => String(n).padStart(2, "0");
const fmtTime = (d: Date) => `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;

/** Qt format: `HH:MM:SS [LEVEL] logger: message` (records of the frontend have no logger). */
export function formatLogLine(entry: Pick<LogEntry, "time" | "level" | "message" | "logger">): string {
  const logger = entry.logger ? `${entry.logger}: ` : "";
  return `${fmtTime(entry.time)} [${entry.level.toUpperCase()}] ${logger}${entry.message}`;
}

/** Entries at or above the minimum level. */
export function visibleLogs(logs: readonly LogEntry[], level: LogThreshold): LogEntry[] {
  const floor = LEVEL_RANK[level];
  return logs.filter((l) => (LEVEL_RANK[l.level] ?? LEVEL_RANK.info) >= floor);
}

// ---- save dialogs ---------------------------------------------------------

/** Strip the format's extension if the user typed it. */
export function baseNameOf(name: string, extension: string): string {
  const trimmed = name.trim();
  return trimmed.toLowerCase().endsWith(extension.toLowerCase()) ? trimmed.slice(0, -extension.length) : trimmed;
}
